"""
Batch runner for the MVRC 2026 page.

Reads a CFD results CSV (one row per team, the same numbers a user would type into
the MVRC 2026 Streamlit page), runs the identical simulation for every row and
writes a new CSV in the MVRC season results format (see
MVRC/2025/results/MVRC_2025_*.csv): the CFD input columns carried through, followed
by lap time, sector times, Vmax and AYmax, sorted fastest first.

Each car also gets a telemetry plot next to the results CSV, in two forms:
  * <team>.html -- the same interactive Plotly panel the MVRC 2026 Streamlit page
    renders (velocity profile, track map, variable dropdown, linked hover), as a
    self-contained file
  * <team>.png  -- a static version of the same layout, with lateral acceleration
    in the lower panel
  * <team>.npz  -- the position/time channels, so make_mvrc_animation.py can render
    the ghost race without re-running the simulations

Usage:
    conda run -n lts313 python run_mvrc_batch.py <input.csv> [-o <output.csv>] [-t <track>]
                                                 [--plot-dir <dir>] [--no-plots]

The input CSV must contain the columns:
    Team, Cd, Cl(f), Cl(r), Cooling flow [m³/s]
"""

import argparse
import copy
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")  # batch script: render to file, never to a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from helpers.simulation import read_vehicle_params, run_simulation_advanced
from helpers.visualization import _BLUE_GRADIENT, build_simulation_plots_html

# ----------------------------------------------------------------------------------------------------------------------
# Constants mirrored from pages/3_MVRC_2026.py -- keep in sync with the page
# ----------------------------------------------------------------------------------------------------------------------

COOLING_FLOW_MIN = 1.15  # [m^3/s]
COOLING_FLOW_MAX = 2.30  # [m^3/s]
POW_MAX_AT_MIN_FLOW = 205.0  # [kW]
POW_MAX_AT_MAX_FLOW = 410.0  # [kW]

DEFAULT_TRACK = "BarcelonaGrandPrix_2026"


def cooling_flow_to_power(flow: float) -> float:
    """Return the maximum engine power [kW] for a given cooling flow [m^3/s]."""
    frac = (flow - COOLING_FLOW_MIN) / (COOLING_FLOW_MAX - COOLING_FLOW_MIN)
    return POW_MAX_AT_MIN_FLOW + frac * (POW_MAX_AT_MAX_FLOW - POW_MAX_AT_MIN_FLOW)


def build_vehicle_pars(base_pars: dict, c_w_a: float, c_z_a_f: float, c_z_a_r: float,
                       pow_max_kw: float) -> dict:
    """Apply the page's vehicle parameter overrides to a copy of the base MVRC_2026 params."""
    pars = copy.deepcopy(base_pars)
    pars["general"]["c_w_a"] = c_w_a
    pars["general"]["c_z_a_f"] = c_z_a_f
    pars["general"]["c_z_a_r"] = c_z_a_r
    pars["engine"]["pow_max"] = pow_max_kw * 1e3

    # Scale pow_diff along with pow_max so the ICE power curve keeps its shape (see page comment).
    pow_scale = pars["engine"]["pow_max"] / base_pars["engine"]["pow_max"]
    pars["engine"]["pow_diff"] = base_pars["engine"]["pow_diff"] * pow_scale

    return pars


def run_one(track: str, veh_pars: dict, base_pars: dict):
    """Run a single MVRC 2026 lap with the exact option dicts used by the page."""
    track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 1.0,
        "curv_filt_width": 10.0,
        "use_drs": True,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-5,
        "custom_vehicle_pars": veh_pars,
    }

    driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": base_pars["engine"]["max_e_energy_storage"],
        "em_strategy": "QUALY",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }

    return run_simulation_advanced(track_opts, solver_opts, driver_opts)


def format_lap_time(t: float) -> str:
    """Format a lap time in seconds as M:SS.mmm (console output only)."""
    return f"{int(t // 60)}:{t % 60:06.3f}"


# ----------------------------------------------------------------------------------------------------------------------
# Output format -- mirrors MVRC/2025/results/MVRC_2025_<track>.csv
# ----------------------------------------------------------------------------------------------------------------------

# Placeholder the season results use for a car that did not produce a lap.
DNF_VALUE = 10000

OUTPUT_COLUMNS = [
    "Position", "Team", "Cd", "Cl", "Cl/Cd", "Cl(f)", "Cl(r)", "CoP",
    "Exhaust Flow [N]", "Engine Flow [N]", "Cooling flow [m^3/s]",
    "rh_adj [mm]", "rake [deg]", "Penalties",
    "Lap Time", "Sector 1 Time", "Sector 2 Time", "Sector 3 Time",
    "Vmax (kph)", "AYmax (g)",
]


# ----------------------------------------------------------------------------------------------------------------------
# Telemetry plots
# ----------------------------------------------------------------------------------------------------------------------

G = 9.81  # [m/s^2]

# Streamlit's dark theme background. The plot panel's labels are light, so a standalone
# file has to paint the same ground the component iframe sits on.
_PAGE_BACKGROUND = "#0e1117"
_PANEL_BACKGROUND = "#161a23"
_FG = "#e6e6e6"
_MUTED = "#9aa0aa"
_LINE = "#1f77b4"
_SECTOR_COLORS = ["#f0e040", "#40e0f0"]  # S1|S2, S2|S3 -- same as the web panel

# The web panel's track-map colourscale, as a matplotlib colormap.
_TRACK_CMAP = LinearSegmentedColormap.from_list(
    "mvrc_blue", [(pos, color) for pos, color in _BLUE_GRADIENT]
)


def _safe_filename(name: str) -> str:
    """Turn a team name into a filesystem-safe file stem."""
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "car"


def write_telemetry_plot(res, team: str, track: str, pow_max: float, out_path: str) -> None:
    """Write the MVRC page's interactive telemetry panel for one car as a standalone HTML file."""
    header = f"""
<div style="color:#fff;font-family:'Source Sans Pro',sans-serif;padding:12px 4px 0 4px;">
  <div style="font-size:22px;font-weight:600;">{team}</div>
  <div style="font-size:14px;color:#bbb;margin-top:2px;">{_summary_line(res, track, pow_max)}</div>
</div>"""

    html = build_simulation_plots_html(
        res,
        background=_PAGE_BACKGROUND,
        title=f"{team} — {track}",
        header_html=header,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def write_telemetry_npz(res, team: str, track: str, pow_max: float, out_path: str) -> None:
    """Dump the arrays the ghost-race animation needs, so it can be re-rendered without re-simulating.

    Only the position/time channels are stored -- the full SimulationResult is not needed
    downstream and pickling it would tie the file to this module's class definition.
    """
    np.savez_compressed(
        out_path,
        team=team,
        track=track,
        pow_max=pow_max,
        lap_time=res.lap_time,
        sector_times=np.asarray(res.sector_times),
        time=res.time,
        distance=res.distance,
        track_x=res.track_x,
        track_y=res.track_y,
        velocity_kmh=res.velocity_kmh,
    )


def _summary_line(res, track: str, pow_max: float) -> str:
    """One-line run summary shared by the HTML header and the PNG subtitle."""
    fuel = f" · fuel {res.fuel_consumed:.2f} kg" if res.fuel_consumed is not None else ""
    return (
        f"{track} · {format_lap_time(res.lap_time)} "
        f"({res.sector_times[0]:.3f} / {res.sector_times[1]:.3f} / {res.sector_times[2]:.3f}) "
        f"· vmax {np.max(res.velocity_kmh):.0f} km/h "
        f"· ay,max {np.max(res.lat_acceleration) / G:.2f} g "
        f"· {pow_max:.0f} kW · {res.energy_consumed:.0f} kJ{fuel}"
    )


def _style_axis(ax) -> None:
    """Apply the dark panel styling used by the web UI to a matplotlib axis."""
    ax.set_facecolor(_PANEL_BACKGROUND)
    ax.grid(alpha=0.15, color=_MUTED)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_color("#2b303b")
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def _mark_sectors(ax, res) -> None:
    """Draw the sector splits as vertical guides on a distance-axis plot."""
    for i, d in enumerate(res.sector_distances or []):
        ax.axvline(d, color=_SECTOR_COLORS[i], lw=1.0, ls="--", alpha=0.8, zorder=1)


def write_telemetry_png(res, team: str, track: str, pow_max: float, out_path: str) -> None:
    """Write a static version of the telemetry panel: velocity + track map, lateral acceleration below."""
    d = res.distance

    fig = plt.figure(figsize=(16, 9), facecolor=_PAGE_BACKGROUND)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.75], width_ratios=[2.0, 1.0],
                          hspace=0.28, wspace=0.12)
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_lat = fig.add_subplot(gs[1, :])

    # --- Velocity profile ------------------------------------------------------------------
    _style_axis(ax_vel)
    ax_vel.plot(d, res.velocity_kmh, color=_LINE, lw=1.4, zorder=3)
    ax_vel.fill_between(d, 0, res.velocity_kmh, color=_LINE, alpha=0.2, lw=0, zorder=2)
    ax_vel.set_ylabel("Velocity [km/h]")
    ax_vel.set_xlabel("Distance [m]")
    ax_vel.set_xlim(d[0], d[-1])
    ax_vel.set_ylim(bottom=0)
    _mark_sectors(ax_vel, res)

    # --- Track map coloured by velocity ----------------------------------------------------
    ax_map.set_facecolor(_PAGE_BACKGROUND)
    ax_map.axis("off")
    pts = np.column_stack([res.track_x, res.track_y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=_TRACK_CMAP, lw=4.0)
    lc.set_array(res.velocity_kmh[:-1])
    ax_map.add_collection(lc)
    ax_map.autoscale()
    ax_map.set_aspect("equal")
    cbar = fig.colorbar(lc, ax=ax_map, fraction=0.04, pad=0.02)
    cbar.set_label("Velocity [km/h]", color=_FG)
    cbar.ax.tick_params(colors=_MUTED)
    cbar.outline.set_edgecolor("#2b303b")

    # Start/finish and the two sector splits, so the map reads against the profiles.
    ax_map.plot(res.track_x[0], res.track_y[0], "o", color="white",
                markeredgecolor="black", ms=8, zorder=4)
    for i, (x, y) in enumerate(res.sector_xy or []):
        ax_map.plot(x, y, "o", color=_SECTOR_COLORS[i], ms=7, zorder=4)

    # --- Lateral acceleration --------------------------------------------------------------
    _style_axis(ax_lat)
    lat_g = res.lat_acceleration / G
    ax_lat.plot(d, lat_g, color="#9467bd", lw=1.2, zorder=3)
    ax_lat.fill_between(d, 0, lat_g, color="#9467bd", alpha=0.2, lw=0, zorder=2)
    ax_lat.set_ylabel("Lateral acceleration [g]")
    ax_lat.set_xlabel("Distance [m]")
    ax_lat.set_xlim(d[0], d[-1])
    ax_lat.set_ylim(bottom=0)
    _mark_sectors(ax_lat, res)

    fig.suptitle(team, color=_FG, fontsize=18, fontweight="bold", x=0.09, ha="left", y=0.975)
    fig.text(0.09, 0.938, _summary_line(res, track, pow_max), color=_MUTED, fontsize=10,
             ha="left")

    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------

def _column(df: pd.DataFrame, *candidates: str) -> str:
    """Return the first candidate column present in df, else raise with a helpful message."""
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"none of the columns {candidates} found in the input CSV; available: {list(df.columns)}"
    )


def _passthrough(row: pd.Series, df: pd.DataFrame, *candidates: str, default=0):
    """Return an input column's value for this row, or 'default' if the column is absent."""
    for name in candidates:
        if name in df.columns:
            return row[name]
    return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="CFD results CSV with one row per team")
    parser.add_argument("-o", "--output", default=None,
                        help="output CSV path (default: <input>_laptimes.csv)")
    parser.add_argument("-t", "--track", default=DEFAULT_TRACK,
                        help=f"raceline/track name (default: {DEFAULT_TRACK})")
    parser.add_argument("--plot-dir", default=None,
                        help="directory for the per-car telemetry pages "
                             "(default: <output dir>/telemetry)")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip the telemetry pages, write only the results CSV")
    args = parser.parse_args()

    out_path = args.output or f"{os.path.splitext(args.input_csv)[0]}_laptimes.csv"
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plot_dir = None
    if not args.no_plots:
        plot_dir = args.plot_dir or os.path.join(os.path.dirname(out_path), "telemetry")
        os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    col_team = _column(df, "Team")
    col_cd = _column(df, "Cd", "CdA")
    col_clf = _column(df, "Cl(f)", "Cl_f")
    col_clr = _column(df, "Cl(r)", "Cl_r")
    col_cool = _column(df, "Cooling flow [m³/s]", "Cooling flow [m^3/s]", "Cooling flow")

    base_pars = read_vehicle_params("MVRC_2026")

    print(f"Track: {args.track}")
    print(f"Running {len(df)} simulations from {args.input_csv}\n")

    rows = []
    for i, row in df.iterrows():
        team = str(row[col_team])

        # The page's downforce inputs are magnitudes; CFD reports Cl as negative (downward).
        c_w_a = float(row[col_cd])
        c_z_a_f = abs(float(row[col_clf]))
        c_z_a_r = abs(float(row[col_clr]))
        cooling_raw = float(row[col_cool])

        # The page's slider is bounded, so a value outside the range is not reachable there.
        # Clamp to the same range and report it rather than extrapolating the power model.
        cooling_flow = min(max(cooling_raw, COOLING_FLOW_MIN), COOLING_FLOW_MAX)
        clamped = not np.isclose(cooling_flow, cooling_raw)

        pow_max = cooling_flow_to_power(cooling_flow)

        print(f"[{i + 1}/{len(df)}] {team}: cwA={c_w_a:.3f} czA_f={c_z_a_f:.3f} "
              f"czA_r={c_z_a_r:.3f} cooling={cooling_flow:.3f} -> {pow_max:.0f} kW")
        if clamped:
            print(f"      ! cooling flow {cooling_raw:.3f} clamped to "
                  f"{cooling_flow:.2f} m³/s (page slider range)")

        # The CFD input columns are carried through unchanged so the result file stays
        # a complete record of the entry, exactly like the 2025 season files.
        entry = {
            "Position": "",
            "Team": team,
            "Cd": row[col_cd],
            "Cl": _passthrough(row, df, "Cl", default=""),
            "Cl/Cd": _passthrough(row, df, "Cl/Cd", default=""),
            "Cl(f)": row[col_clf],
            "Cl(r)": row[col_clr],
            "CoP": _passthrough(row, df, "CoP", default=""),
            "Exhaust Flow [N]": _passthrough(row, df, "Exhaust Flow [N]", default=""),
            "Engine Flow [N]": _passthrough(row, df, "Engine Flow [N]", default=""),
            "Cooling flow [m^3/s]": cooling_raw,
            "rh_adj [mm]": _passthrough(row, df, "rh_adj [mm]", "rh_adj"),
            "rake [deg]": _passthrough(row, df, "rake [deg]", "rake"),
            "Penalties": _passthrough(row, df, "Penalties"),
        }

        veh_pars = build_vehicle_pars(base_pars, c_w_a, c_z_a_f, c_z_a_r, pow_max)

        t0 = time.time()
        try:
            res = run_one(args.track, veh_pars, base_pars)
        except Exception as exc:  # one failing car must not kill the whole batch
            print(f"      FAILED: {exc}")
            entry.update({
                "Lap Time": DNF_VALUE,
                "Sector 1 Time": DNF_VALUE,
                "Sector 2 Time": DNF_VALUE,
                "Sector 3 Time": DNF_VALUE,
                "Vmax (kph)": DNF_VALUE,
                "AYmax (g)": DNF_VALUE,
            })
            rows.append(entry)
            continue

        print(f"      {format_lap_time(res.lap_time)}  ({time.time() - t0:.1f} s)")

        g = base_pars["general"]["g"]
        entry.update({
            "Lap Time": round(res.lap_time, 3),
            "Sector 1 Time": round(res.sector_times[0], 3),
            "Sector 2 Time": round(res.sector_times[1], 3),
            "Sector 3 Time": round(res.sector_times[2], 3),
            "Vmax (kph)": int(round(float(np.max(res.velocity_kmh)))),
            "AYmax (g)": round(float(np.max(res.lat_acceleration)) / g, 2),
        })
        rows.append(entry)

        if plot_dir is not None:
            stem = os.path.join(plot_dir, _safe_filename(team))
            write_telemetry_png(res, team, args.track, pow_max, f"{stem}.png")
            write_telemetry_plot(res, team, args.track, pow_max, f"{stem}.html")
            write_telemetry_npz(res, team, args.track, pow_max, f"{stem}.npz")
            print(f"      plots: {stem}.png / .html / .npz")

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values("Lap Time").reset_index(drop=True)

    out.to_csv(out_path, index=False)

    print(f"\nResults written to {out_path}")
    if plot_dir is not None:
        print(f"Telemetry pages written to {plot_dir}")
    print()
    print(out[["Team", "Lap Time", "Sector 1 Time", "Sector 2 Time", "Sector 3 Time",
               "Vmax (kph)", "AYmax (g)"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
