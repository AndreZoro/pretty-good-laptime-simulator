"""
FastF1 reference laps as comparable runs.

Turns a FastF1 telemetry download into a SimulationResult so a real lap can go into the
same saved-run store the simulations use and be rendered by the comparison view
unchanged.

Persistence is deliberately thin: FastF1 already caches the session data on disk
(laptimesim/input/fastf1_cache/), so the telemetry itself is never stored a second time.
What that cache cannot say is which laps the user picked -- it is keyed by session, not
by selection -- so only that list is kept, as a small JSON index. Re-loading an entry
replays load_speed_trace(), which comes back out of the FastF1 cache.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np

from helpers.fastf1_data import load_speed_trace
from helpers.simulation import SimulationResult, _lat_acceleration

INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "laptimesim",
    "output",
    "reference_laps.json",
)


def _curvature_from_position(x: np.ndarray, y: np.ndarray, s: np.ndarray,
                             smooth_m: float = 15.0) -> np.ndarray:
    """Curvature of the driven line, from the position trace.

    The raw position channel is far too noisy to differentiate twice, so x and y are
    smoothed over a window of roughly 'smooth_m' metres first. The result is good enough
    to show where the corners are and to drive the lateral-acceleration trace; it is not
    accurate enough to feed a simulation.
    """
    ds = np.median(np.diff(s))
    if not np.isfinite(ds) or ds <= 0:
        return np.zeros_like(s)

    # Odd, so the window stays centred, and at least 5 samples wide: car data comes in at
    # roughly 15 m spacing, and differentiating twice over fewer points than that leaves
    # the result dominated by position noise.
    window = max(5, int(round(smooth_m / ds)) | 1)
    kernel = np.ones(window) / window
    # Pad by edge repetition: a 'same' convolution alone would pull the ends toward zero.
    pad = window // 2
    xs = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")
    ys = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")

    dx, dy = np.gradient(xs, s), np.gradient(ys, s)
    ddx, ddy = np.gradient(dx, s), np.gradient(dy, s)

    denom = (dx ** 2 + dy ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-9, (dx * ddy - dy * ddx) / denom, 0.0)
    return np.nan_to_num(kappa)


_RACELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "laptimesim", "input", "tracks", "racelines",
)


def sim_track_length(track_name: str | None) -> float | None:
    """Closed length of a sim raceline in metres, or None when it cannot be determined."""
    if not track_name:
        return None

    path = os.path.join(_RACELINE_DIR, f"{track_name}.csv")
    if not os.path.isfile(path):
        return None

    try:
        xy = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))
    except (ValueError, OSError):
        return None

    # The raceline is a closed loop stored without a repeated first point.
    closed = np.vstack([xy, xy[:1]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def reference_label(entry: dict) -> str:
    """Legend label, e.g. 'VER · 2026 Barcelona Grand Prix (Barcelona) Qualifying'.

    The circuit is named alongside the event because the two do not track each other:
    the 2026 Spanish Grand Prix is at Madrid, not Barcelona.
    """
    driver = entry.get("driver") or "fastest"
    where = f" ({entry['location']})" if entry.get("location") else ""
    return (f"{driver} · {entry['year']} {entry['gp']}{where} "
            f"{entry.get('session', '')}").strip()


def fastf1_to_result(data: dict, entry: dict) -> SimulationResult:
    """Build a SimulationResult from a load_speed_trace() payload.

    Args:
        data: the dict returned by load_speed_trace(..., with_position=True)
        entry: the index entry describing the lap -- year, gp, session, driver, track.
            Mutated: 'raw_distance_m' and 'distance_scale' are written back so the
            caller can report how far the telemetry's distance axis had to be stretched.

    Returns:
        A SimulationResult carrying the channels F1 telemetry actually provides. The
        ones it cannot provide stay None, and energy_consumed is NaN rather than 0.0 --
        a real car obviously consumed energy, and a zero would read in the comparison
        table as a measured value.
    """
    distance = np.asarray(data["distance"], dtype=float)
    velocity = np.asarray(data["speed"], dtype=float)
    time = np.asarray(data["time"], dtype=float) if data.get("time") is not None else None

    # FastF1 integrates its distance channel from the speed samples, so it does not land
    # exactly on the raceline length; a linear stretch pins both ends so the profiles
    # line up for an overlay. It cannot fix drift within the lap.
    #
    # A large factor here is a warning sign, not something to accept: it usually means
    # the lap is from a different circuit than the tagged raceline, which is what a
    # stale event mapping produces. The page surfaces the factor for that reason.
    entry["raw_distance_m"] = float(distance[-1])
    entry["distance_scale"] = 1.0
    length_sim = sim_track_length(entry.get("track"))
    if length_sim and distance[-1] > 0:
        entry["distance_scale"] = length_sim / float(distance[-1])
        distance = distance * entry["distance_scale"]

    pos_x, pos_y = data.get("pos_x"), data.get("pos_y")
    has_pos = pos_x is not None and pos_y is not None
    track_x = np.asarray(pos_x, dtype=float) if has_pos else np.zeros_like(distance)
    track_y = np.asarray(pos_y, dtype=float) if has_pos else np.zeros_like(distance)

    curvature = (
        _curvature_from_position(track_x, track_y, distance)
        if has_pos else np.zeros_like(distance)
    )
    lat_acceleration = _lat_acceleration(velocity, curvature)

    # Longitudinal acceleration is dv/dt when the time channel is there: measured
    # directly, and unaffected by the distance drift corrected above. These are sampled
    # telemetry channels, not a solver's piecewise-constant output, so a centred derivative
    # is the right estimator here (unlike the per-step form used for a simulated lap).
    if time is not None and len(time) == len(velocity):
        acceleration = np.gradient(velocity, time)
    else:
        acceleration = velocity * np.gradient(velocity, distance)

    gear = data.get("gear")
    drs = data.get("drs_active")

    return SimulationResult(
        lap_time=float(data["lap_time"]),
        sector_times=[float(s) for s in data["sector_times"]],
        distance=distance,
        velocity=velocity,
        velocity_kmh=velocity * 3.6,
        acceleration=acceleration,
        lat_acceleration=lat_acceleration,
        curvature=curvature,
        gear=np.asarray(gear, dtype=float) if gear is not None else np.zeros_like(distance),
        track_x=track_x,
        track_y=track_y,
        energy_consumed=float("nan"),  # not measurable from public telemetry
        fuel_consumed=None,
        track_name=entry.get("track") or entry["gp"],
        vehicle=f"FastF1 {entry['year']}",
        weather="Real",
        rpm=np.asarray(data["rpm"], dtype=float) if data.get("rpm") is not None else None,
        drs=np.asarray(drs, dtype=bool) if drs is not None else None,
        time=time,
        label=reference_label(entry),
    )


def fetch_reference_lap(entry: dict) -> tuple[SimulationResult, dict]:
    """Load a lap described by an index entry, via FastF1 (served from its disk cache).

    Returns:
        (result, resolved entry). A lap asked for as "fastest overall" belongs to
        whichever driver set it; the returned entry names that driver, so the label is
        specific and storing the entry pins the lap instead of re-resolving it later.
    """
    data = load_speed_trace(
        entry["year"], entry["gp"], entry.get("session", "Q"),
        entry.get("driver"), with_position=True,
    )
    resolved = dict(entry)
    if not resolved.get("driver") and data.get("driver"):
        resolved["driver"] = data["driver"]
    return fastf1_to_result(data, resolved), resolved


# ----------------------------------------------------------------------------------------------------------------------
# Index of picked laps
# ----------------------------------------------------------------------------------------------------------------------

def entry_id(entry: dict) -> str:
    """Identity of a lap, so saving the same one twice updates rather than duplicates."""
    return (f"{entry['year']}|{entry['gp']}|{entry.get('session', 'Q')}"
            f"|{entry.get('driver') or ''}")


def load_index(path: str = INDEX_PATH) -> list[dict]:
    """Stored lap entries, newest first. A missing or corrupt index reads as empty."""
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    if not isinstance(entries, list):
        return []
    return sorted(entries, key=lambda e: e.get("saved_at", ""), reverse=True)


def _write_index(entries: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def save_to_index(entry: dict, path: str = INDEX_PATH) -> list[dict]:
    """Add (or refresh) a lap in the index and return the updated list."""
    entry = dict(entry)
    entry["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    entries = [e for e in load_index(path) if entry_id(e) != entry_id(entry)]
    entries.append(entry)
    _write_index(entries, path)
    return load_index(path)


def delete_from_index(entry: dict, path: str = INDEX_PATH) -> list[dict]:
    """Remove a lap from the index and return the updated list."""
    entries = [e for e in load_index(path) if entry_id(e) != entry_id(entry)]
    _write_index(entries, path)
    return load_index(path)
