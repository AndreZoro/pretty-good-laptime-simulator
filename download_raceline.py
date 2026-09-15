"""
Download a raceline from FastF1 qualifying telemetry.

Takes the 5 fastest qualifying laps, resamples each to 5 m resolution,
and averages them to produce a single x_m,y_m,z_m,kappa raceline CSV.

Curvature is computed with a Savitzky-Golay filter (window=SG_WINDOW points,
cubic polynomial) applied to the uniformly arc-length-sampled x/y positions.
This avoids the double-differentiation noise that plagues interpolating splines
on sparse GPS data, which tends to overestimate curvature at chicane apices.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import fastf1


# ── cache ──────────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".fastf1_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)

STEP_M = 5.0        # target point spacing [m]
N_LAPS = 5          # number of fastest laps to average
MIN_LAP_M = 2_000   # sanity bounds on lap length [m]
# Savitzky-Golay window for curvature (must be odd); 9 points → ±20 m centred on each sample.
# Increase to reduce GPS/spline noise further; decrease to preserve tighter corners.
SG_WINDOW = 17
# Gaussian post-smooth applied to kappa after SG, in GPS samples (sigma * STEP_M = physical width).
# 0.0 disables it. 2.0 → σ = 10 m — a light second pass independent of the SG window.
SG_SIGMA = 4.0
MAX_LAP_M = 15_000


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample_lap(pos_data: "fastf1.core.Telemetry", step: float = STEP_M):
    """Return (x, y, z) arrays resampled to *step* metre intervals.

    Raises ValueError if the raw arc-length is outside [MIN_LAP_M, MAX_LAP_M],
    which catches cases where FastF1 returned multi-lap position data.
    """
    # FastF1 position data is in decimeters — convert to metres
    x = pos_data["X"].to_numpy(dtype=float) / 10.0
    y = pos_data["Y"].to_numpy(dtype=float) / 10.0
    z = pos_data["Z"].to_numpy(dtype=float) / 10.0

    # cumulative arc-length
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])

    raw_len = cum[-1]
    if not (MIN_LAP_M <= raw_len <= MAX_LAP_M):
        raise ValueError(
            f"Position data length {raw_len:.0f} m is outside [{MIN_LAP_M}, {MAX_LAP_M}] m — "
            "FastF1 may have returned multi-lap data for this lap."
        )

    # drop duplicates that would break interp
    cum, idx = np.unique(cum, return_index=True)
    x, y, z = x[idx], y[idx], z[idx]

    n_pts = int(cum[-1] / step)
    s_new = np.arange(n_pts) * step  # 0, 5, 10, …  (open end)

    return (
        np.interp(s_new, cum, x),
        np.interp(s_new, cum, y),
        np.interp(s_new, cum, z),
    )


def _average_laps(laps_xyz):
    """Trim all laps to the shortest and return element-wise mean."""
    n = min(arr.shape[0] for arr, _, _ in laps_xyz)
    xs = np.mean([x[:n] for x, _, _ in laps_xyz], axis=0)
    ys = np.mean([y[:n] for _, y, _ in laps_xyz], axis=0)
    zs = np.mean([z[:n] for _, _, z in laps_xyz], axis=0)
    return xs, ys, zs


def _compute_kappa(x: np.ndarray, y: np.ndarray, step: float) -> np.ndarray:
    """Compute signed curvature (rad/m) from uniformly arc-length-sampled x, y.

    Uses a Savitzky-Golay filter to estimate first and second derivatives
    simultaneously, which is far more stable than double-differentiating an
    interpolating spline through sparse GPS data.  mode='wrap' treats the
    arrays as a closed loop (required for circular tracks).

    A Gaussian post-smooth (sigma=SG_SIGMA samples) is applied to the raw
    kappa as an independent second pass in the curvature domain.  Set
    SG_SIGMA=0 to disable.
    """
    sg_kw = dict(window_length=SG_WINDOW, polyorder=3, delta=step, mode="wrap")
    dx  = savgol_filter(x, deriv=1, **sg_kw)
    dy  = savgol_filter(y, deriv=1, **sg_kw)
    d2x = savgol_filter(x, deriv=2, **sg_kw)
    d2y = savgol_filter(y, deriv=2, **sg_kw)
    kappa = (dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5
    if SG_SIGMA > 0.0:
        kappa = gaussian_filter1d(kappa, sigma=SG_SIGMA, mode="wrap")
    return kappa


# ── core download ─────────────────────────────────────────────────────────────

def _check_wet(session, event_name: str) -> bool:
    """Print a warning if the session had rainfall; returns True if wet."""
    wd = getattr(session, "weather_data", None)
    if wd is None or wd.empty or "Rainfall" not in wd.columns:
        return False
    rain = wd["Rainfall"].astype(bool)
    wet_frac = rain.mean()
    if rain.any():
        print(f"  ⚠  WET SESSION: rainfall detected for {wet_frac:.0%} of {event_name} qualifying")
        return True
    return False


def _download_raceline(session, event_name: str, save_path: str) -> None:
    """Process a qualifying session and write a raceline CSV to *save_path*."""
    valid_laps = session.laps.pick_quicklaps().sort_values("LapTime")
    if len(valid_laps) < N_LAPS:
        print(f"  Warning: only {len(valid_laps)} valid lap(s) found, using all.")
    top_laps = valid_laps.head(N_LAPS)

    print(f"  Processing {len(top_laps)} fastest laps:")
    laps_xyz = []
    for rank, (_, lap) in enumerate(top_laps.iterrows(), start=1):
        driver = lap["Driver"]
        lap_time = lap["LapTime"]

        lap_end = lap["Time"]
        lap_start = lap["LapStartTime"]
        drv_num = lap["DriverNumber"]
        pos_all = None
        for key in (drv_num, str(int(drv_num)), lap["Driver"]):
            if key in session.pos_data:
                pos_all = session.pos_data[key]
                break
        if pos_all is None:
            print(f"    {rank}. {driver}  {lap_time}  →  SKIPPED (no position data found)")
            continue
        pos = pos_all[(pos_all["SessionTime"] >= lap_start) & (pos_all["SessionTime"] <= lap_end)]

        try:
            x, y, z = _resample_lap(pos)
        except ValueError as e:
            print(f"    {rank}. {driver}  {lap_time}  →  SKIPPED ({e})")
            continue
        laps_xyz.append((x, y, z))
        print(f"    {rank}. {driver}  {lap_time}  →  {len(x)} points @ {STEP_M} m")

    if not laps_xyz:
        raise RuntimeError(f"No valid laps for {event_name}. Check the FastF1 data for this session.")

    x_avg, y_avg, z_avg = _average_laps(laps_xyz)
    kappa = _compute_kappa(x_avg, y_avg, step=STEP_M)
    print(f"  Averaged raceline: {len(x_avg)} points  "
          f"|κ|_max={np.max(np.abs(kappa)):.4f}  |κ|_mean={np.mean(np.abs(kappa)):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    data = np.column_stack([x_avg, y_avg, z_avg, kappa])
    np.savetxt(save_path, data, delimiter=",", header="x_m,y_m,z_m,kappa", comments="#", fmt="%.6f")
    print(f"  Saved → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

_RACELINES_DIR = os.path.join(
    os.path.dirname(__file__), "laptimesim", "input", "tracks", "racelines"
)


def main():
    year = int(input("Year: "))

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = schedule[schedule["F1ApiSupport"] == True].reset_index(drop=True)  # noqa: E712

    print(f"\nAvailable races ({year}):")
    print(f"   0.  ALL (downloads every race with _{year} suffix)")
    for i, row in races.iterrows():
        print(f"  {i + 1:2d}.  {row['EventName']}")

    choice = int(input("\nSelect race number (0 = all): "))

    if choice == 0:
        # ── batch: download every race ──
        failed = []
        wet = []
        for i, row in races.iterrows():
            event_name = row["EventName"]
            safe_name = event_name.replace(" ", "")
            save_path = os.path.join(_RACELINES_DIR, f"{safe_name}_{year}.csv")
            print(f"\n[{i + 1}/{len(races)}] {event_name}")
            try:
                session = fastf1.get_session(year, event_name, "Q")
                session.load(telemetry=True, weather=True, messages=False)
                if _check_wet(session, event_name):
                    wet.append(event_name)
                _download_raceline(session, event_name, save_path)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append(event_name)

        print(f"\n{'=' * 60}")
        print(f"  {year} batch download complete")
        print(f"  Saved:  {len(races) - len(failed)}/{len(races)} tracks → {_RACELINES_DIR}")
        if wet:
            print(f"\n  ⚠  Wet qualifying sessions ({len(wet)}) — racelines may be unrepresentative:")
            for name in wet:
                print(f"       - {name}")
        if failed:
            print(f"\n  ✗  Failed ({len(failed)}):")
            for name in failed:
                print(f"       - {name}")
        print(f"{'=' * 60}")
    else:
        # ── single race ──
        event = races.iloc[choice - 1]
        event_name = event["EventName"]
        print(f"\nSelected: {event_name}")

        print("Loading qualifying session (this may take a moment)…")
        session = fastf1.get_session(year, event_name, "Q")
        session.load(telemetry=True, weather=True, messages=False)
        _check_wet(session, event_name)

        safe_name = event_name.replace(" ", "")
        default_path = os.path.join(_RACELINES_DIR, f"{safe_name}_{year}.csv")
        raw = input(f"\nSave path [{default_path}]: ").strip()
        save_path = raw if raw else default_path

        _download_raceline(session, event_name, save_path)


if __name__ == "__main__":
    main()
