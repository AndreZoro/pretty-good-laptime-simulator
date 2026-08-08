"""
FastF1 telemetry data helper for parameter identification.

Downloads and prepares real F1 speed traces for use as reference data
in the parameter search optimizer.
"""

import os

import numpy as np

# Map sim track names to FastF1 GP identifiers
TRACK_NAME_MAP = {
    # ── legacy / generic racelines ─────────────────────────────────────────
    "Austin": "United States Grand Prix",
    "Budapest": "Hungarian Grand Prix",
    "Catalunya": "Spanish Grand Prix",
    "Hockenheim": "German Grand Prix",
    "Melbourne": "Australian Grand Prix",
    "MexicoCity": "Mexico City Grand Prix",
    "Miami_2026_fastf1": "Miami Grand Prix",
    "Montreal": "Canadian Grand Prix",
    "Monza": "Italian Grand Prix",
    "Sakhir": "Bahrain Grand Prix",
    "SaoPaulo": "São Paulo Grand Prix",
    "Shanghai": "Chinese Grand Prix",
    "Shanghai_2026_fastf1": "Chinese Grand Prix",
    "Shanghai_2026_fastf1_smoothed": "Chinese Grand Prix",
    "Silverstone": "British Grand Prix",
    "Sochi": "Russian Grand Prix",
    "Spa": "Belgian Grand Prix",
    "Spielberg": "Austrian Grand Prix",
    "Suzuka": "Japanese Grand Prix",
    "Suzuka_2026_fastF1_extracted": "Japanese Grand Prix",
    "YasMarina": "Abu Dhabi Grand Prix",
    # ── 2025 season ───────────────────────────────────────────────────────
    "AbuDhabiGrandPrix_2025": "Abu Dhabi Grand Prix",
    "AustralianGrandPrix_2025": "Australian Grand Prix",
    "AustrianGrandPrix_2025": "Austrian Grand Prix",
    "AzerbaijanGrandPrix_2025": "Azerbaijan Grand Prix",
    "BahrainGrandPrix_2025": "Bahrain Grand Prix",
    "BelgianGrandPrix_2025": "Belgian Grand Prix",
    "BritishGrandPrix_2025": "British Grand Prix",
    "CanadianGrandPrix_2025": "Canadian Grand Prix",
    "ChineseGrandPrix_2025": "Chinese Grand Prix",
    "DutchGrandPrix_2025": "Dutch Grand Prix",
    "EmiliaRomagnaGrandPrix_2025": "Emilia Romagna Grand Prix",
    "HungarianGrandPrix_2025": "Hungarian Grand Prix",
    "ItalianGrandPrix_2025": "Italian Grand Prix",
    "JapaneseGrandPrix_2025": "Japanese Grand Prix",
    "LasVegasGrandPrix_2025": "Las Vegas Grand Prix",
    "MexicoCityGrandPrix_2025": "Mexico City Grand Prix",
    "MiamiGrandPrix_2025": "Miami Grand Prix",
    "MonacoGrandPrix_2025": "Monaco Grand Prix",
    "QatarGrandPrix_2025": "Qatar Grand Prix",
    "SãoPauloGrandPrix_2025": "São Paulo Grand Prix",
    "SaudiArabianGrandPrix_2025": "Saudi Arabian Grand Prix",
    "SingaporeGrandPrix_2025": "Singapore Grand Prix",
    "SpanishGrandPrix_2025": "Spanish Grand Prix",
    "UnitedStatesGrandPrix_2025": "United States Grand Prix",
    # ── 2026 season ───────────────────────────────────────────────────────
    "AustralianGrandPrix_2026": "Australian Grand Prix",
    "ChineseGrandPrix_2026": "Chinese Grand Prix",
    "JapaneseGrandPrix_2026": "Japanese Grand Prix",
    "MiamiGrandPrix_2026": "Miami Grand Prix",
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
    return list(range(2018, 2028))


def get_available_gps(sim_tracks: list[str]) -> dict[str, str]:
    """
    Return dict of sim track names that have a FastF1 mapping.

    Args:
        sim_tracks: List of available sim track names

    Returns:
        Dict mapping sim track name -> FastF1 GP name
    """
    return {track: gp for track, gp in TRACK_NAME_MAP.items() if track in sim_tracks}


def load_speed_trace(
    year: int,
    gp: str,
    session_type: str = "Q",
    driver: str | None = None,
) -> dict:
    """
    Download and extract telemetry channels from FastF1.

    Args:
        year: Season year (e.g. 2023)
        gp: Grand Prix name (FastF1 format, e.g. "Chinese Grand Prix")
        session_type: "Q" for qualifying, "R" for race
        driver: Driver abbreviation (e.g. "VER"). None = fastest lap overall.

    Returns:
        Dict with keys:
        - distance: ndarray of cumulative distance in meters
        - speed: ndarray of speed in m/s
        - lap_time: lap time in seconds
        - sector_times: [S1, S2, S3] in seconds
        - throttle: ndarray 0-100 or None
        - brake: ndarray 0-100 (or bool) or None
        - gear: ndarray of gear numbers or None
        - rpm: ndarray of engine RPM or None
        - drs: ndarray of raw DRS status codes or None
        - drs_active: ndarray of binary DRS active flag or None
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

    # Extract additional channels (gracefully handle missing columns)
    def _get_channel(name):
        if name in car_data.columns:
            return car_data[name].to_numpy().astype(float)
        return None

    throttle = _get_channel("Throttle")
    brake = _get_channel("Brake")
    gear = _get_channel("nGear")
    rpm = _get_channel("RPM")
    drs_raw = _get_channel("DRS")

    # Derive binary DRS active flag (codes 10, 12, 14 = open/active)
    drs_active = None
    if drs_raw is not None:
        drs_active = np.isin(drs_raw.astype(int), [10, 12, 14]).astype(float)

    return {
        "distance": distance_m,
        "speed": speed_mps,
        "lap_time": lap_time_s,
        "sector_times": sector_times,
        "throttle": throttle,
        "brake": brake,
        "gear": gear,
        "rpm": rpm,
        "drs": drs_raw,
        "drs_active": drs_active,
    }


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


# Metrics offered as a fit objective.
#
# "r2" is deliberately NOT here: for a fixed reference trace SS_tot and N are constants, so
# 1 - R^2 = (N / SS_tot) * RMSE^2 -- a strictly increasing function of RMSE. It has the same
# minimum, the same argmin and the same candidate ranking, so selecting it changes nothing.
# It is still computed for reporting, where being scale-free makes it comparable across
# circuits in a way RMSE is not.
TRACE_METRICS = ["rmse", "time_rel"]
DEFAULT_TRACE_METRIC = "rmse"
_SUPPORTED_TRACE_METRICS = TRACE_METRICS + ["r2"]


def compute_trace_error(
    sim_distance: np.ndarray,
    sim_velocity: np.ndarray,
    ref_distance: np.ndarray,
    ref_velocity: np.ndarray,
    metric: str = DEFAULT_TRACE_METRIC,
) -> float:
    """
    Compare a simulated speed trace against a reference one.

    Both traces are normalized to 0-1 of track length before interpolation
    to handle slight differences in total track length.

    Args:
        sim_distance: Simulated distance array [m]
        sim_velocity: Simulated velocity array [m/s]
        ref_distance: Reference (FastF1) distance array [m]
        ref_velocity: Reference (FastF1) velocity array [m/s]
        metric: "rmse"     -> root mean squared speed error in m/s
                "time_rel" -> time-weighted RMS relative speed error (see below)
                "r2"       -> 1 - R^2; report-only, equivalent to "rmse" as an objective

    Returns:
        A quantity to MINIMISE, so every metric can drive the same optimiser.

    "rmse" is an absolute error, so a 5 m/s miss counts the same everywhere. Because the
    traces are sampled along distance, fast sections also contribute more samples per second
    of lap time -- together that makes RMSE strongly straight-line weighted.

    "time_rel" changes both of those. Residuals are taken relative to the reference speed, so
    a given miss costs more where the car is slow, and each sample is weighted by the time the
    reference car spent in it (ds / v), so slow sections stop being under-represented. The
    result is dimensionless: 0.02 means a typical relative speed error of 2%.
    """
    metric = str(metric).lower()
    if metric not in _SUPPORTED_TRACE_METRICS:
        raise ValueError(
            f"unknown trace metric {metric!r}, expected one of {_SUPPORTED_TRACE_METRICS}")

    # Normalize both distance arrays to 0-1
    sim_dist_norm = sim_distance / sim_distance[-1]
    ref_dist_norm = ref_distance / ref_distance[-1]

    # Interpolate sim velocity onto reference distance grid
    sim_vel_interp = np.interp(ref_dist_norm, sim_dist_norm, sim_velocity)

    if metric == "time_rel":
        # guard the division: a reference trace can touch very low speeds
        v_ref = np.maximum(np.asarray(ref_velocity, dtype=float), 1.0)
        ds = np.abs(np.gradient(np.asarray(ref_distance, dtype=float)))  # [m] per sample
        w = ds / v_ref                                                   # [s] time per sample
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return float("inf")
        rel = (sim_vel_interp - ref_velocity) / v_ref
        return float(np.sqrt(float(np.sum(w * rel ** 2)) / w_sum))

    ss_res = float(np.sum((sim_vel_interp - ref_velocity) ** 2))
    if metric == "rmse":
        return float(np.sqrt(ss_res / ref_velocity.size))

    ss_tot = float(np.sum((ref_velocity - np.mean(ref_velocity)) ** 2))
    if ss_tot <= 0.0:  # constant reference trace, R^2 undefined
        return float(np.sqrt(ss_res / ref_velocity.size))
    return ss_res / ss_tot  # == 1 - R^2


def compute_trace_r2(
    sim_distance: np.ndarray,
    sim_velocity: np.ndarray,
    ref_distance: np.ndarray,
    ref_velocity: np.ndarray,
) -> float:
    """R^2 of the simulated speed trace against the reference (1.0 = perfect)."""
    return 1.0 - compute_trace_error(
        sim_distance, sim_velocity, ref_distance, ref_velocity, metric="r2"
    )
