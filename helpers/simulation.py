"""
Simulation wrapper for Streamlit UI.

Provides a clean interface to run lap time simulations with sensible defaults.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

import main_laptimesim


# Series-specific configuration
SERIES_CONFIG = {
    "F1": {
        "vehicle": "F1_Shanghai",
        "initial_energy": 4.0e6,  # 4 MJ
        "use_drs": True,
    },
    "FE": {
        "vehicle": "FE_Berlin",
        "initial_energy": 4.58e6,  # 4.58 MJ
        "use_drs": False,
    },
}


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
    series: str
    weather: str

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
    series: str,
    mu_weather: float = 1.0,
) -> SimulationResult:
    """
    Run a lap time simulation with simplified parameters.

    Args:
        track_name: Name of the track (e.g., "Shanghai", "Monza")
        series: Racing series ("F1" or "FE")
        mu_weather: Weather friction factor (0.6=wet, 1.0=dry)

    Returns:
        SimulationResult with lap time, velocity profile, and track data
    """
    config = SERIES_CONFIG[series]

    # Build option dicts with sensible defaults
    track_opts = {
        "trackname": track_name,
        "flip_track": False,
        "mu_weather": mu_weather,
        "interp_stepsize_des": 5.0,
        "curv_filt_width": 10.0,
        "use_drs1": config["use_drs"],
        "use_drs2": config["use_drs"],
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": config["vehicle"],
        "series": series,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
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

    # Compute longitudinal acceleration: a = v * dv/ds
    dv_ds = np.gradient(velocity_unclosed, distance_unclosed)
    acceleration = velocity_unclosed * dv_ds

    # Compute lateral acceleration: a_lat = v^2 * kappa
    curvature = lap.trackobj.kappa
    lat_acceleration = velocity_unclosed ** 2 * np.abs(curvature)

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
        series=series,
        weather=weather,
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

    # Compute longitudinal acceleration: a = v * dv/ds
    dv_ds = np.gradient(velocity_unclosed, distance_unclosed)
    acceleration = velocity_unclosed * dv_ds

    # Compute lateral acceleration: a_lat = v^2 * kappa
    curvature = lap.trackobj.kappa
    lat_acceleration = velocity_unclosed ** 2 * np.abs(curvature)

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
        series=solver_opts["series"],
        weather=weather,
    )
