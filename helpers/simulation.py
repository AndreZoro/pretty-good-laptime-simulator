"""
Simulation wrapper for Streamlit UI.

Provides a clean interface to run lap time simulations with sensible defaults.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

import main_laptimesim


def _smooth_array(arr: np.ndarray, window: int = 11) -> np.ndarray:
    """Apply moving average smoothing to an array."""
    if len(arr) < window:
        return arr
    if window % 2 == 0:
        window += 1
    return np.convolve(arr, np.ones(window) / window, mode='same')


def _compute_accelerations(velocity: np.ndarray, distance: np.ndarray, curvature: np.ndarray):
    """
    Compute longitudinal and lateral accelerations.

    Returns:
        Tuple of (longitudinal_acceleration, lateral_acceleration) in m/s²
    """
    # Longitudinal acceleration: a = v * dv/ds
    dv_ds = np.gradient(velocity, distance)
    acceleration = velocity * dv_ds

    # Lateral acceleration: a_lat = v^2 * kappa, with smoothing
    lat_acceleration_raw = velocity ** 2 * np.abs(curvature)
    lat_acceleration = _smooth_array(lat_acceleration_raw)

    return acceleration, lat_acceleration


# Vehicle-specific defaults for the simple simulation page
VEHICLE_DEFAULTS = {
    "F1_Shanghai": {"initial_energy": 4.0e6, "use_drs": True},
    "F1_2025": {"initial_energy": 4.0e6, "use_drs": True},
    "F1_2025_f1technical_shifts": {"initial_energy": 4.0e6, "use_drs": True},
    "F1_2025_optimized_shifts": {"initial_energy": 4.0e6, "use_drs": True},
    "F1_2026": {"initial_energy": 4.0e6, "use_drs": True},
    "MVRC_2026": {"initial_energy": 4.0e6, "use_drs": True},
    "FE_Berlin": {"initial_energy": 4.58e6, "use_drs": False},
}
DEFAULT_VEHICLE = {"initial_energy": 4.0e6, "use_drs": True}


@dataclass
class SimulationResult:
    """Container for simulation results."""

    # Lap time data
    lap_time: float  # Total lap time in seconds
    sector_times: list[float]  # [S1, S2, S3] in seconds

    # Profile data
    distance: np.ndarray  # Distance along track [m]
    velocity: np.ndarray  # Velocity profile [m/s]
    velocity_kmh: np.ndarray  # Velocity profile [km/h]
    acceleration: np.ndarray  # Longitudinal acceleration [m/s^2]
    lat_acceleration: np.ndarray  # Lateral acceleration [m/s^2]
    curvature: np.ndarray  # Track curvature [rad/m]
    gear: np.ndarray  # Gear number

    # Track data
    track_x: np.ndarray  # Track x coordinates
    track_y: np.ndarray  # Track y coordinates

    # Energy data
    energy_consumed: float  # Total energy consumed [kJ]
    fuel_consumed: Optional[float]  # Fuel consumed [kg] (F1 only)

    # Metadata
    track_name: str
    vehicle: str
    weather: str

    # Extended data (optional for backward compatibility)
    rpm: Optional[np.ndarray] = None  # Engine RPM [1/min]
    engine_torque: Optional[np.ndarray] = None  # Engine torque [Nm]
    e_motor_torque: Optional[np.ndarray] = None  # E-motor torque [Nm]
    tire_loads: Optional[np.ndarray] = None  # Tire loads (no_points, 4) [FL, FR, RL, RR] [N]
    energy_storage: Optional[np.ndarray] = None  # Battery/ERS state [kJ]
    fuel_consumed_profile: Optional[np.ndarray] = None  # Cumulative fuel [kg]
    energy_consumed_profile: Optional[np.ndarray] = None  # Cumulative energy [kJ]
    drs: Optional[np.ndarray] = None  # DRS active flag
    time: Optional[np.ndarray] = None  # Cumulative time [s]
    friction: Optional[np.ndarray] = None  # Track friction coefficient
    e_motor_power: Optional[np.ndarray] = None  # E-motor power [kW], negative = harvest
    harvest_power: Optional[np.ndarray] = None  # Instantaneous harvest power [kW]

    def format_lap_time(self) -> str:
        """Format lap time as M:SS.mmm"""
        minutes = int(self.lap_time // 60)
        seconds = self.lap_time % 60
        return f"{minutes}:{seconds:06.3f}"

    def format_sector_time(self, sector: int) -> str:
        """Format sector time as SS.mmm"""
        return f"{self.sector_times[sector]:.3f}"


def get_available_tracks() -> list[str]:
    """Get list of available track names that have both raceline and parameters defined."""
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracks_path = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines")
    pars_path = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")

    # Get tracks with raceline files
    raceline_tracks = set()
    for f in os.listdir(tracks_path):
        if f.endswith(".csv") and not f.endswith("_pit.csv"):
            raceline_tracks.add(f.replace(".csv", ""))

    # Get tracks defined in track_pars.ini
    import configparser
    import ast
    config = configparser.ConfigParser()
    config.read(pars_path)
    pars_str = config.get("TRACK_PARS", "track_pars")
    track_pars = ast.literal_eval(pars_str)
    defined_tracks = set(track_pars.keys())

    # Only return tracks that have both raceline and parameters
    valid_tracks = raceline_tracks & defined_tracks

    return sorted(valid_tracks)


def get_available_vehicles() -> list[str]:
    """Get list of available vehicle configuration names (without .ini extension)."""
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vehicles_path = os.path.join(repo_path, "laptimesim", "input", "vehicles")

    vehicles = []
    for f in os.listdir(vehicles_path):
        if f.endswith(".ini"):
            vehicles.append(f[:-4])  # Strip .ini extension

    return sorted(vehicles)


def run_simulation(
    track_name: str,
    vehicle: str,
    mu_weather: float = 1.0,
) -> SimulationResult:
    """
    Run a lap time simulation with simplified parameters.

    Args:
        track_name: Name of the track (e.g., "Shanghai", "Monza")
        vehicle: Vehicle configuration name (e.g., "F1_Shanghai", "FE_Berlin")
        mu_weather: Weather friction factor (0.6=wet, 1.0=dry)

    Returns:
        SimulationResult with lap time, velocity profile, and track data
    """
    config = VEHICLE_DEFAULTS.get(vehicle, DEFAULT_VEHICLE)

    # Build option dicts with sensible defaults
    track_opts = {
        "trackname": track_name,
        "flip_track": False,
        "mu_weather": mu_weather,
        "interp_stepsize_des": 1.0,
        "curv_filt_width": 10.0,
        "use_drs1": config["use_drs"],
        "use_drs2": config["use_drs"],
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": vehicle,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-5,
    }

    driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": config["initial_energy"],
        "em_strategy": "FCFB",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }

    sa_opts = {
        "use_sa": False,
        "sa_type": "mass",
        "range_1": [733.0, 833.0, 5],
        "range_2": None,
    }

    debug_opts = {
        "use_plot": False,
        "use_debug_plots": False,
        "use_plot_comparison_tph": False,
        "use_print": False,
        "use_print_result": False,
    }

    # Run simulation
    lap = main_laptimesim.main(
        track_opts=track_opts,
        solver_opts=solver_opts,
        driver_opts=driver_opts,
        sa_opts=sa_opts,
        debug_opts=debug_opts,
    )

    # Extract sector times
    zone_inds = lap.trackobj.zone_inds
    t_s1 = lap.t_cl[zone_inds["s12"]]
    t_s2 = lap.t_cl[zone_inds["s23"]] - t_s1
    t_s3 = lap.t_cl[-1] - lap.t_cl[zone_inds["s23"]]

    # Determine weather description
    if mu_weather >= 0.95:
        weather = "Dry"
    elif mu_weather >= 0.75:
        weather = "Damp"
    else:
        weather = "Wet"

    # Get unclosed arrays for consistent sizing with track coordinates
    # raceline is unclosed (no_points), vel_cl and dists_cl are closed (no_points + 1)
    no_points = lap.trackobj.no_points
    velocity_unclosed = lap.vel_cl[:no_points]
    distance_unclosed = lap.trackobj.dists_cl[:no_points]

    # Compute accelerations
    curvature = lap.trackobj.kappa
    acceleration, lat_acceleration = _compute_accelerations(
        velocity_unclosed, distance_unclosed, curvature
    )

    # Get gear data
    gear = lap.gear_cl[:no_points]

    # Build result
    return SimulationResult(
        lap_time=lap.t_cl[-1],
        sector_times=[t_s1, t_s2, t_s3],
        distance=distance_unclosed,
        velocity=velocity_unclosed,
        velocity_kmh=velocity_unclosed * 3.6,
        acceleration=acceleration,
        lat_acceleration=lat_acceleration,
        curvature=curvature,
        gear=gear,
        track_x=lap.trackobj.raceline[:, 0],
        track_y=lap.trackobj.raceline[:, 1],
        energy_consumed=lap.e_cons_cl[-1] / 1000.0,
        fuel_consumed=lap.fuel_cons_cl[-1] if hasattr(lap, 'fuel_cons_cl') and lap.fuel_cons_cl is not None else None,
        track_name=track_name,
        vehicle=vehicle,
        weather=weather,
        rpm=lap.n_cl[:no_points] * 60,
        engine_torque=lap.m_eng[:no_points],
        e_motor_torque=lap.m_e_motor[:no_points],
        tire_loads=lap.tire_loads[:no_points],
        energy_storage=lap.es_cl[:no_points] / 1000.0,
        fuel_consumed_profile=lap.fuel_cons_cl[:no_points],
        energy_consumed_profile=lap.e_cons_cl[:no_points] / 1000.0,
        drs=lap.trackobj.drs[:no_points],
        time=lap.t_cl[:no_points],
        friction=lap.trackobj.mu[:no_points],
        e_motor_power=2 * np.pi * lap.n_cl[:no_points] * lap.m_e_motor[:no_points] / 1000.0,
        harvest_power=lap.e_rec_e_motor[:no_points] / np.diff(lap.t_cl[:no_points + 1]) / 1000.0,
    )


def run_simulation_advanced(
    track_opts: dict,
    solver_opts: dict,
    driver_opts: dict,
) -> SimulationResult:
    """
    Run a lap time simulation with full parameter control.

    Args:
        track_opts: Track configuration dict
        solver_opts: Solver configuration dict
        driver_opts: Driver configuration dict

    Returns:
        SimulationResult with lap time, velocity profile, and track data
    """
    sa_opts = {
        "use_sa": False,
        "sa_type": "mass",
        "range_1": [733.0, 833.0, 5],
        "range_2": None,
    }

    debug_opts = {
        "use_plot": False,
        "use_debug_plots": False,
        "use_plot_comparison_tph": False,
        "use_print": False,
        "use_print_result": False,
    }

    # Run simulation
    lap = main_laptimesim.main(
        track_opts=track_opts,
        solver_opts=solver_opts,
        driver_opts=driver_opts,
        sa_opts=sa_opts,
        debug_opts=debug_opts,
    )

    # Extract sector times
    zone_inds = lap.trackobj.zone_inds
    t_s1 = lap.t_cl[zone_inds["s12"]]
    t_s2 = lap.t_cl[zone_inds["s23"]] - t_s1
    t_s3 = lap.t_cl[-1] - lap.t_cl[zone_inds["s23"]]

    # Determine weather description
    mu_weather = track_opts["mu_weather"]
    if mu_weather >= 0.95:
        weather = "Dry"
    elif mu_weather >= 0.75:
        weather = "Damp"
    else:
        weather = "Wet"

    # Get unclosed arrays for consistent sizing with track coordinates
    no_points = lap.trackobj.no_points
    velocity_unclosed = lap.vel_cl[:no_points]
    distance_unclosed = lap.trackobj.dists_cl[:no_points]

    # Compute accelerations
    curvature = lap.trackobj.kappa
    acceleration, lat_acceleration = _compute_accelerations(
        velocity_unclosed, distance_unclosed, curvature
    )

    # Get gear data
    gear = lap.gear_cl[:no_points]

    # Build result
    return SimulationResult(
        lap_time=lap.t_cl[-1],
        sector_times=[t_s1, t_s2, t_s3],
        distance=distance_unclosed,
        velocity=velocity_unclosed,
        velocity_kmh=velocity_unclosed * 3.6,
        acceleration=acceleration,
        lat_acceleration=lat_acceleration,
        curvature=curvature,
        gear=gear,
        track_x=lap.trackobj.raceline[:, 0],
        track_y=lap.trackobj.raceline[:, 1],
        energy_consumed=lap.e_cons_cl[-1] / 1000.0,
        fuel_consumed=lap.fuel_cons_cl[-1] if hasattr(lap, 'fuel_cons_cl') and lap.fuel_cons_cl is not None else None,
        track_name=track_opts["trackname"],
        vehicle=solver_opts.get("vehicle", "Custom"),
        weather=weather,
        rpm=lap.n_cl[:no_points] * 60,
        engine_torque=lap.m_eng[:no_points],
        e_motor_torque=lap.m_e_motor[:no_points],
        tire_loads=lap.tire_loads[:no_points],
        energy_storage=lap.es_cl[:no_points] / 1000.0,
        fuel_consumed_profile=lap.fuel_cons_cl[:no_points],
        energy_consumed_profile=lap.e_cons_cl[:no_points] / 1000.0,
        drs=lap.trackobj.drs[:no_points],
        time=lap.t_cl[:no_points],
        friction=lap.trackobj.mu[:no_points],
        e_motor_power=2 * np.pi * lap.n_cl[:no_points] * lap.m_e_motor[:no_points] / 1000.0,
        harvest_power=lap.e_rec_e_motor[:no_points] / np.diff(lap.t_cl[:no_points + 1]) / 1000.0,
    )
