"""
Download a raceline from FastF1 qualifying telemetry.

Takes the 5 fastest qualifying laps, resamples each to 5 m resolution,
and averages them to produce a single x_m,y_m,z_m raceline CSV.
"""

import os
import numpy as np
import fastf1


# ── cache ──────────────────────────────────────────────────────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".fastf1_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(_CACHE_DIR)

STEP_M = 5.0        # target point spacing [m]
N_LAPS = 5          # number of fastest laps to average
MIN_LAP_M = 2_000   # sanity bounds on lap length [m]
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── year ──
    year = int(input("Year: "))

    # ── race selection ──
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    races = schedule[schedule["F1ApiSupport"] == True].reset_index(drop=True)  # noqa: E712

    print(f"\nAvailable races ({year}):")
    for i, row in races.iterrows():
        print(f"  {i + 1:2d}.  {row['EventName']}")

    choice = int(input("\nSelect race number: ")) - 1
    event = races.iloc[choice]
    event_name = event["EventName"]
    print(f"\nSelected: {event_name}")

    # ── load qualifying ──
    print("Loading qualifying session (this may take a moment)…")
    session = fastf1.get_session(year, event_name, "Q")
    session.load(telemetry=True, weather=False, messages=False)

    # ── pick N fastest valid laps ──
    valid_laps = session.laps.pick_quicklaps().sort_values("LapTime")
    if len(valid_laps) < N_LAPS:
        print(f"Warning: only {len(valid_laps)} valid lap(s) found, using all.")
    top_laps = valid_laps.head(N_LAPS)

    print(f"\nProcessing {len(top_laps)} fastest laps:")
    laps_xyz = []
    for rank, (_, lap) in enumerate(top_laps.iterrows(), start=1):
        driver = lap["Driver"]
        lap_time = lap["LapTime"]

        # Manually slice session.pos_data to the lap's time window.
        # lap.get_pos_data() can return the whole session for some seasons.
        lap_end = lap["Time"]
        lap_start = lap["LapStartTime"]
        drv_num = lap["DriverNumber"]
        pos_all = None
        for key in (drv_num, str(int(drv_num)), lap["Driver"]):
            if key in session.pos_data:
                pos_all = session.pos_data[key]
                break
        if pos_all is None:
            print(f"  {rank}. {driver}  {lap_time}  →  SKIPPED (no position data found)")
            continue
        pos = pos_all[(pos_all["SessionTime"] >= lap_start) & (pos_all["SessionTime"] <= lap_end)]

        try:
            x, y, z = _resample_lap(pos)
        except ValueError as e:
            print(f"  {rank}. {driver}  {lap_time}  →  SKIPPED ({e})")
            continue
        laps_xyz.append((x, y, z))
        print(f"  {rank}. {driver}  {lap_time}  →  {len(x)} points @ {STEP_M} m")

    if not laps_xyz:
        raise RuntimeError("No valid laps could be processed. Check the FastF1 data for this session.")

    # ── average ──
    x_avg, y_avg, z_avg = _average_laps(laps_xyz)
    print(f"\nAveraged raceline: {len(x_avg)} points")

    # ── save ──
    safe_name = event_name.replace(" ", "")
    default_path = os.path.join(
        os.path.dirname(__file__),
        "laptimesim", "input", "tracks", "racelines",
        f"{safe_name}.csv",
    )
    raw = input(f"\nSave path [{default_path}]: ").strip()
    save_path = raw if raw else default_path

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    data = np.column_stack([x_avg, y_avg, z_avg])
    np.savetxt(
        save_path, data,
        delimiter=",",
        header="x_m,y_m,z_m",
        comments="#",
        fmt="%.6f",
    )
    print(f"Saved → {save_path}  ({len(x_avg)} points, {STEP_M} m spacing)")


if __name__ == "__main__":
    main()
