"""
FastF1 telemetry data helper for parameter identification.

Downloads and prepares real F1 speed traces for use as reference data
in the parameter search optimizer.
"""

import os

import numpy as np

# Map sim track names to FastF1 GP identifiers
TRACK_NAME_MAP = {
    "Austin": "United States Grand Prix",
    "Budapest": "Hungarian Grand Prix",
    "Catalunya": "Spanish Grand Prix",
    "Hockenheim": "German Grand Prix",
    "Melbourne": "Australian Grand Prix",
    "MexicoCity": "Mexico City Grand Prix",
    "Montreal": "Canadian Grand Prix",
    "Monza": "Italian Grand Prix",
    "Sakhir": "Bahrain Grand Prix",
    "SaoPaulo": "São Paulo Grand Prix",
    "Shanghai": "Chinese Grand Prix",
    "Silverstone": "British Grand Prix",
    "Sochi": "Russian Grand Prix",
    "Spa": "Belgian Grand Prix",
    "Spielberg": "Austrian Grand Prix",
    "Suzuka": "Japanese Grand Prix",
    "YasMarina": "Abu Dhabi Grand Prix",
}


def setup_cache():
    """Configure FastF1 disk cache directory."""
    import fastf1

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "laptimesim",
        "input",
        "fastf1_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def get_available_years():
    """Return list of years with F1 data available via FastF1."""
    return list(range(2018, 2026))


def get_available_gps(sim_tracks: list[str]) -> dict[str, str]:
    """
    Return dict of sim track names that have a FastF1 mapping.

    Args:
        sim_tracks: List of available sim track names

    Returns:
        Dict mapping sim track name -> FastF1 GP name
    """
    return {
        track: gp
        for track, gp in TRACK_NAME_MAP.items()
        if track in sim_tracks
    }


def load_speed_trace(
    year: int,
    gp: str,
    session_type: str = "Q",
    driver: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """
    Download and extract a speed trace from FastF1.

    Args:
        year: Season year (e.g. 2023)
        gp: Grand Prix name (FastF1 format, e.g. "Chinese Grand Prix")
        session_type: "Q" for qualifying, "R" for race
        driver: Driver abbreviation (e.g. "VER"). None = fastest lap overall.

    Returns:
        Tuple of (distance_m, speed_mps, lap_time_s, sector_times)
        - distance_m: ndarray of cumulative distance in meters
        - speed_mps: ndarray of speed in m/s
        - lap_time_s: lap time in seconds
        - sector_times: [S1, S2, S3] in seconds
    """
    import fastf1

    setup_cache()

    session = fastf1.get_session(year, gp, session_type)
    session.load()

    if driver is not None:
        laps = session.laps.pick_drivers(driver)
        lap = laps.pick_fastest()
    else:
        lap = session.laps.pick_fastest()

    # Get car telemetry with distance
    car_data = lap.get_car_data().add_distance()

    distance_m = car_data["Distance"].to_numpy().astype(float)
    speed_kmh = car_data["Speed"].to_numpy().astype(float)
    speed_mps = speed_kmh / 3.6

    # Lap time in seconds
    lap_time_s = lap["LapTime"].total_seconds()

    # Sector times
    s1 = lap["Sector1Time"].total_seconds()
    s2 = lap["Sector2Time"].total_seconds()
    s3 = lap["Sector3Time"].total_seconds()
    sector_times = [s1, s2, s3]

    return distance_m, speed_mps, lap_time_s, sector_times


def get_drivers_in_session(year: int, gp: str, session_type: str = "Q") -> list[str]:
    """
    Get list of driver abbreviations available in a session.

    Returns:
        Sorted list of driver abbreviations (e.g. ["ALO", "HAM", "VER", ...])
    """
    import fastf1

    setup_cache()

    session = fastf1.get_session(year, gp, session_type)
    session.load()

    drivers = session.laps["Driver"].unique().tolist()
    return sorted(drivers)


def compute_trace_error(
    sim_distance: np.ndarray,
    sim_velocity: np.ndarray,
    ref_distance: np.ndarray,
    ref_velocity: np.ndarray,
) -> float:
    """
    Compute RMSE between simulated and reference speed traces.

    Both traces are normalized to 0-1 of track length before interpolation
    to handle slight differences in total track length.

    Args:
        sim_distance: Simulated distance array [m]
        sim_velocity: Simulated velocity array [m/s]
        ref_distance: Reference (FastF1) distance array [m]
        ref_velocity: Reference (FastF1) velocity array [m/s]

    Returns:
        RMSE of speed difference in m/s
    """
    # Normalize both distance arrays to 0-1
    sim_dist_norm = sim_distance / sim_distance[-1]
    ref_dist_norm = ref_distance / ref_distance[-1]

    # Interpolate sim velocity onto reference distance grid
    sim_vel_interp = np.interp(ref_dist_norm, sim_dist_norm, sim_velocity)

    # RMSE
    return float(np.sqrt(np.mean((sim_vel_interp - ref_velocity) ** 2)))
