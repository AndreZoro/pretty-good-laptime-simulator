"""
Parameter Identification Page

Determine vehicle aerodynamic parameters and weight from sector times and max velocity.
Uses 4 targets (S1, S2, S3, v_max) to identify 4 parameters (drag, downforce, mass, power).
"""

import configparser
import ast
import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import Bounds, minimize

from helpers.fastf1_data import (
    TRACK_NAME_MAP,
    compute_trace_error,
    get_available_gps,
    get_available_years,
    get_drivers_in_session,
    load_speed_trace,
)
from helpers.simulation import (
    get_available_tracks,
    get_available_vehicles,
    run_simulation_advanced,
)
from helpers.visualization import render_simulation_plots

st.set_page_config(
    page_title="Parameter Identification - Laptime Sim",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Parameter Identification")
st.caption(
    "Determine vehicle parameters from sector times, max velocity, or speed traces"
)

# Initialize session state
if "param_id_result" not in st.session_state:
    st.session_state.param_id_result = None
if "param_id_abort" not in st.session_state:
    st.session_state.param_id_abort = False
if "fastf1_trace" not in st.session_state:
    st.session_state.fastf1_trace = None


class AbortException(Exception):
    """Raised when user aborts the optimization."""

    pass


def check_abort():
    """Check if abort was requested and raise exception if so."""
    if st.session_state.param_id_abort:
        raise AbortException("Optimization aborted by user")


def load_vehicle_config(vehicle_name: str) -> dict:
    """Load vehicle configuration from an .ini file in the vehicles directory."""
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ini_path = os.path.join(
        repo_path, "laptimesim", "input", "vehicles", f"{vehicle_name}.ini"
    )
    parser = configparser.ConfigParser()
    if not parser.read(ini_path):
        raise RuntimeError(f"Vehicle config file not found: {ini_path}")
    config = ast.literal_eval(parser.get("VEH_PARS", "veh_pars"))
    return config


def build_vehicle_pars(
    base_config: dict, c_w_a: float, c_z_a_total: float, mass: float, pow_max: float
) -> dict:
    """
    Build vehicle parameters dict with the given aero, mass and power values.

    Total downforce is split front/rear based on CoG position (lf, lr) to maintain
    aerodynamic balance matching the static weight distribution.
    """
    import copy

    pars = copy.deepcopy(base_config)

    # Get CoG position
    lf = pars["general"]["lf"]
    lr = pars["general"]["lr"]
    wheelbase = lf + lr

    # Split downforce proportional to static weight distribution
    # Front axle carries: lr / wheelbase of the weight
    # Rear axle carries: lf / wheelbase of the weight
    c_z_a_f = c_z_a_total * (lr / wheelbase)
    c_z_a_r = c_z_a_total * (lf / wheelbase)

    pars["general"]["c_w_a"] = c_w_a
    pars["general"]["c_z_a_f"] = c_z_a_f
    pars["general"]["c_z_a_r"] = c_z_a_r
    pars["general"]["m"] = mass
    pars["engine"]["pow_max"] = pow_max
    return pars


def run_sim_with_params(
    track: str,
    base_config: dict,
    c_w_a: float,
    c_z_a_total: float,
    mass: float,
    pow_max: float,
):
    """Run simulation with given parameters and return sector times and max velocity."""
    vehicle_pars = build_vehicle_pars(base_config, c_w_a, c_z_a_total, mass, pow_max)

    # Use fastest settings for parameter search (less accurate but much faster)
    track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 30.0,  # Large step = fast
        "curv_filt_width": 60.0,  # Must result in odd window size (60/30 + 1 = 3)
        "use_drs": True,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "limit_braking_weak_side": None,  # Skip weak side calculation
        "v_start": 100.0 / 3.6,
        "find_v_start": True,  # Re-run with end-of-lap velocity as start
        "max_no_em_iters": 1,  # Single iteration
        "es_diff_max": 100.0,
        "vel_tol": 5e-2,  # Lower tolerance for faster parameter search
        "custom_vehicle_pars": vehicle_pars,
    }

    driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": 4.0e6,
        "em_strategy": "FCFB",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }

    try:
        result = run_simulation_advanced(track_opts, solver_opts, driver_opts)
        v_max = float(np.max(result.velocity))  # Max velocity in m/s
        return (
            result.sector_times,
            result.lap_time,
            v_max,
            result.distance,
            result.velocity,
        )
    except Exception:
        return None, None, None, None, None


def run_grid_search(
    track,
    base_config,
    target_sectors,
    target_v_max,
    bounds,
    log_container,
    aero_step=0.2,
    mass_step=5.0,
    power_step=10e3,
    ref_distance=None,
    ref_velocity=None,
):
    """
    Run a coarse grid search to find approximate best parameters.
    Uses fixed step sizes for each parameter.
    Returns best result found, even if aborted.
    """
    use_trace = ref_distance is not None and ref_velocity is not None

    c_w_a_vals = np.arange(bounds[0][0], bounds[0][1] + aero_step / 2, aero_step)
    c_z_a_vals = np.arange(bounds[1][0], bounds[1][1] + aero_step / 2, aero_step)
    mass_vals = np.arange(bounds[2][0], bounds[2][1] + mass_step / 2, mass_step)
    pow_max_vals = np.arange(bounds[3][0], bounds[3][1] + power_step / 2, power_step)

    best_error = float("inf")
    best_params = None
    best_sectors = None
    best_lap = None
    best_v_max = None
    best_sim_distance = None
    best_sim_velocity = None

    total = len(c_w_a_vals) * len(c_z_a_vals) * len(mass_vals) * len(pow_max_vals)
    count = 0

    # Weight for v_max error: 1 m/s error ~ 0.36 s sector error (squared: ~0.13)
    # This makes ~3 m/s v_max error equivalent to ~1 s sector error
    v_max_weight = 0.13

    try:
        for c_w_a in c_w_a_vals:
            for c_z_a in c_z_a_vals:
                for mass in mass_vals:
                    for pow_max in pow_max_vals:
                        # Check for abort request
                        check_abort()

                        count += 1
                        start = time.time()

                        sectors, lap_time, v_max, sim_dist, sim_vel = (
                            run_sim_with_params(
                                track, base_config, c_w_a, c_z_a, mass, pow_max
                            )
                        )
                        elapsed = time.time() - start

                        if sectors is None:
                            log_container.write(
                                f"[{count}/{total}] FAILED - drag={c_w_a:.2f}, df={c_z_a:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
                            )
                            continue

                        if use_trace:
                            error = compute_trace_error(
                                sim_dist, sim_vel, ref_distance, ref_velocity
                            )
                        else:
                            sector_error = sum(
                                (s - t) ** 2 for s, t in zip(sectors, target_sectors)
                            )
                            v_max_error = v_max_weight * (v_max - target_v_max) ** 2
                            error = sector_error + v_max_error

                        err_label = (
                            f"RMSE={error:.2f}m/s" if use_trace else f"err={error:.2f}"
                        )
                        status = "**BEST**" if error < best_error else ""
                        log_container.write(
                            f"[{count}/{total}] drag={c_w_a:.2f}, df={c_z_a:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
                            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, {err_label} ({elapsed:.2f}s) {status}"
                        )

                        if error < best_error:
                            best_error = error
                            best_params = [c_w_a, c_z_a, mass, pow_max]
                            best_sectors = sectors
                            best_lap = lap_time
                            best_v_max = v_max
                            best_sim_distance = sim_dist
                            best_sim_velocity = sim_vel
    except AbortException:
        # Log best result found before abort
        if best_params is not None:
            log_container.write(
                f"Best before abort: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW"
            )
        raise

    return (
        best_params,
        best_sectors,
        best_lap,
        best_v_max,
        best_error,
        best_sim_distance,
        best_sim_velocity,
    )


def run_nelder_mead(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
    ref_distance=None,
    ref_velocity=None,
):
    """
    Run Nelder-Mead optimization starting from an initial guess.
    Faster than grid search but may find local minimum.
    Parameters are clipped to bounds.
    """
    use_trace = ref_distance is not None and ref_velocity is not None

    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
        None,
        None,
    ]  # [params, sectors, lap_time, v_max, error, sim_dist, sim_vel]

    # Extract bounds for clipping
    c_w_a_bounds = bounds[0]
    c_z_a_bounds = bounds[1]
    mass_bounds = bounds[2]
    pow_max_bounds = bounds[3]

    # Weight for v_max error: ~3 m/s v_max error equivalent to ~1 s sector error
    v_max_weight = 0.13

    def objective(params):
        # Check for abort request
        check_abort()

        eval_count[0] += 1

        # Clip parameters to bounds
        c_w_a = np.clip(params[0], c_w_a_bounds[0], c_w_a_bounds[1])
        c_z_a_total = np.clip(params[1], c_z_a_bounds[0], c_z_a_bounds[1])
        mass = np.clip(params[2], mass_bounds[0], mass_bounds[1])
        pow_max = np.clip(params[3], pow_max_bounds[0], pow_max_bounds[1])

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity)
        else:
            sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            v_max_error = v_max_weight * (v_max - target_v_max) ** 2
            error = sector_error + v_max_error

        err_label = f"RMSE={error:.2f}m/s" if use_trace else f"err={error:.2f}"
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, {err_label} ({elapsed:.2f}s)"
        )

        # Track best result for abort case (store clipped values)
        if error < best_result[4]:
            best_result[0] = [c_w_a, c_z_a_total, mass, pow_max]
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel

        return error

    try:
        result = minimize(
            objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options={
                "maxiter": 500,
                "xatol": 0.01,
                "fatol": 0.15,
            },
        )

        # Get final result (clipped to bounds)
        c_w_a = np.clip(result.x[0], c_w_a_bounds[0], c_w_a_bounds[1])
        c_z_a_total = np.clip(result.x[1], c_z_a_bounds[0], c_z_a_bounds[1])
        mass = np.clip(result.x[2], mass_bounds[0], mass_bounds[1])
        pow_max = np.clip(result.x[3], pow_max_bounds[0], pow_max_bounds[1])

        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return (
            [c_w_a, c_z_a_total, mass, pow_max],
            sectors,
            lap_time,
            v_max,
            result.fun,
            sim_dist,
            sim_vel,
        )
    except AbortException:
        raise


def run_trust_constr(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
    ref_distance=None,
    ref_velocity=None,
):
    """
    Run Trust-Region Constrained optimization starting from an initial guess.
    Supports bounds natively for better convergence.
    Uses normalized parameters (0-1) internally for better scaling.
    """
    use_trace = ref_distance is not None and ref_velocity is not None

    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
        None,
        None,
    ]  # [params, sectors, lap_time, v_max, error, sim_dist, sim_vel]

    # Extract bounds for scaling
    lower = np.array([bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]])
    upper = np.array([bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]])
    scale = upper - lower

    # Weight for v_max error: ~3 m/s v_max error equivalent to ~1 s sector error
    v_max_weight = 0.13

    def to_normalized(params):
        """Convert real parameters to normalized (0-1) space."""
        return (np.array(params) - lower) / scale

    def to_real(normalized):
        """Convert normalized (0-1) parameters to real space."""
        return lower + np.array(normalized) * scale

    # Normalized bounds are always 0-1
    scipy_bounds = Bounds([0, 0, 0, 0], [1, 1, 1, 1])

    # Normalized initial guess
    x0_normalized = to_normalized(initial_guess)

    def objective(normalized_params):
        # Check for abort request
        check_abort()

        eval_count[0] += 1

        # Convert to real parameters
        real_params = to_real(normalized_params)
        c_w_a, c_z_a_total, mass, pow_max = real_params

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity)
        else:
            sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            v_max_error = v_max_weight * (v_max - target_v_max) ** 2
            error = sector_error + v_max_error

        err_label = f"RMSE={error:.2f}m/s" if use_trace else f"err={error:.2f}"
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, {err_label} ({elapsed:.2f}s)"
        )

        # Track best result for abort case (store real values)
        if error < best_result[4]:
            best_result[0] = list(real_params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel

        return error

    try:
        result = minimize(
            objective,
            x0=x0_normalized,
            method="trust-constr",
            bounds=scipy_bounds,
            options={
                "maxiter": 500,
                "gtol": 1e-5,
                "xtol": 1e-5,
            },
        )

        # Convert final result back to real parameters
        real_params = to_real(result.x)
        c_w_a, c_z_a_total, mass, pow_max = real_params
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return (
            list(real_params),
            sectors,
            lap_time,
            v_max,
            result.fun,
            sim_dist,
            sim_vel,
        )
    except AbortException:
        raise


def run_lbfgsb(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
    ref_distance=None,
    ref_velocity=None,
):
    """
    Run L-BFGS-B optimization starting from an initial guess.
    Fast gradient-based optimizer with bounds support.
    Uses normalized parameters (0-1) internally for better scaling.
    """
    use_trace = ref_distance is not None and ref_velocity is not None

    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
        None,
        None,
    ]  # [params, sectors, lap_time, v_max, error, sim_dist, sim_vel]

    # Extract bounds for scaling
    lower = np.array([bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]])
    upper = np.array([bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]])
    scale = upper - lower

    # Weight for v_max error: ~3 m/s v_max error equivalent to ~1 s sector error
    v_max_weight = 0.13

    def to_normalized(params):
        """Convert real parameters to normalized (0-1) space."""
        return (np.array(params) - lower) / scale

    def to_real(normalized):
        """Convert normalized (0-1) parameters to real space."""
        return lower + np.array(normalized) * scale

    # Normalized bounds are always 0-1
    scipy_bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]

    # Normalized initial guess
    x0_normalized = to_normalized(initial_guess)

    def objective(normalized_params):
        # Check for abort request
        check_abort()

        eval_count[0] += 1

        # Convert to real parameters
        real_params = to_real(normalized_params)
        c_w_a, c_z_a_total, mass, pow_max = real_params

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity)
        else:
            sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            v_max_error = v_max_weight * (v_max - target_v_max) ** 2
            error = sector_error + v_max_error

        err_label = f"RMSE={error:.2f}m/s" if use_trace else f"err={error:.2f}"
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, {err_label} ({elapsed:.2f}s)"
        )

        # Track best result for abort case (store real values)
        if error < best_result[4]:
            best_result[0] = list(real_params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel

        return error

    try:
        result = minimize(
            objective,
            x0=x0_normalized,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={
                "maxiter": 1000,
                "ftol": 1e-5,
                "gtol": 1e-5,
            },
        )

        # Convert final result back to real parameters
        real_params = to_real(result.x)
        c_w_a, c_z_a_total, mass, pow_max = real_params
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return (
            list(real_params),
            sectors,
            lap_time,
            v_max,
            result.fun,
            sim_dist,
            sim_vel,
        )
    except AbortException:
        raise


# Sidebar controls
st.sidebar.header("Configuration")

# Track selection
available_tracks = get_available_tracks()
track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Spa") if "Spa" in available_tracks else 0,
)

# Vehicle base configuration
available_vehicles = get_available_vehicles()
vehicle_base = st.sidebar.selectbox(
    "Base Vehicle",
    options=available_vehicles,
)

# Target source toggle
st.sidebar.header("Target Source")
target_source = st.sidebar.radio(
    "Source",
    options=["Manual", "FastF1 Telemetry"],
    horizontal=True,
    help="Manual: enter sector times + v_max. FastF1: download real telemetry speed traces.",
)

use_trace_mode = False
ref_distance = None
ref_velocity = None

if target_source == "FastF1 Telemetry":
    available_gps = get_available_gps(available_tracks)
    # Filter to tracks that have FastF1 mapping
    fastf1_tracks = list(available_gps.keys())

    if not fastf1_tracks:
        st.sidebar.warning("No tracks with FastF1 mapping available.")
    else:
        ff1_track = st.sidebar.selectbox(
            "GP Track",
            options=fastf1_tracks,
            index=fastf1_tracks.index(track) if track in fastf1_tracks else 0,
            help="Select track (must match sim track for meaningful comparison)",
        )
        # Sync track selection with FastF1 track
        track = ff1_track

        ff1_year = st.sidebar.selectbox("Year", options=get_available_years(), index=7)
        ff1_session = st.sidebar.radio(
            "Session",
            options=["Q", "R"],
            horizontal=True,
            help="Q = Qualifying, R = Race",
        )
        ff1_driver = st.sidebar.text_input(
            "Driver (optional)",
            value="",
            help="3-letter abbreviation (e.g. VER, HAM). Leave empty for fastest lap.",
        )
        ff1_driver = ff1_driver.strip().upper() or None

        download_button = st.sidebar.button(
            "Download Telemetry", type="secondary", width="stretch"
        )

        if download_button:
            gp_name = available_gps[ff1_track]
            with st.spinner(
                f"Downloading {ff1_year} {gp_name} {ff1_session} telemetry..."
            ):
                try:
                    ff1_data = load_speed_trace(
                        ff1_year, gp_name, ff1_session, ff1_driver
                    )
                    dist = ff1_data["distance"]
                    vel = ff1_data["speed"]
                    lap_time_ff1 = ff1_data["lap_time"]
                    sectors_ff1 = ff1_data["sector_times"]
                    st.session_state.fastf1_trace = {
                        "distance": dist,
                        "velocity": vel,
                        "lap_time": lap_time_ff1,
                        "sector_times": sectors_ff1,
                        "v_max": float(np.max(vel)),
                        "year": ff1_year,
                        "gp": gp_name,
                        "session": ff1_session,
                        "driver": ff1_driver,
                        "track": ff1_track,
                        "throttle": ff1_data["throttle"],
                        "brake": ff1_data["brake"],
                        "gear": ff1_data["gear"],
                        "rpm": ff1_data["rpm"],
                        "drs": ff1_data["drs"],
                        "drs_active": ff1_data["drs_active"],
                    }
                    st.sidebar.success(
                        f"Lap: {int(lap_time_ff1 // 60)}:{lap_time_ff1 % 60:06.3f} | "
                        f"V_max: {float(np.max(vel)) * 3.6:.1f} km/h"
                    )
                except Exception as e:
                    st.sidebar.error(f"Download failed: {e}")
                    st.session_state.fastf1_trace = None

        # Use stored trace data
        if st.session_state.fastf1_trace is not None:
            trace = st.session_state.fastf1_trace
            ref_distance = trace["distance"]
            ref_velocity = trace["velocity"]
            use_trace_mode = True

            # Auto-fill sector times and v_max from telemetry
            target_s1 = trace["sector_times"][0]
            target_s2 = trace["sector_times"][1]
            target_s3 = trace["sector_times"][2]
            target_v_max_ms = trace["v_max"]

            st.sidebar.caption(
                f"Sectors: {target_s1:.3f} | {target_s2:.3f} | {target_s3:.3f} | "
                f"V_max: {target_v_max_ms * 3.6:.1f} km/h"
            )
        else:
            st.sidebar.info("Click 'Download Telemetry' to fetch data.")

if target_source == "Manual":
    st.sidebar.header("Target Sector Times")

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        target_s1 = st.number_input(
            "S1 [s]",
            min_value=10.0,
            max_value=120.0,
            value=24.0,
            step=0.1,
            format="%.3f",
        )
    with col2:
        target_s2 = st.number_input(
            "S2 [s]",
            min_value=10.0,
            max_value=120.0,
            value=26.5,
            step=0.1,
            format="%.3f",
        )
    with col3:
        target_s3 = st.number_input(
            "S3 [s]",
            min_value=10.0,
            max_value=120.0,
            value=40.4,
            step=0.1,
            format="%.3f",
        )

    target_total = target_s1 + target_s2 + target_s3
    st.sidebar.caption(
        f"Total: **{int(target_total // 60)}:{target_total % 60:06.3f}**"
    )

    st.sidebar.header("Target Max Velocity")
    target_v_max = st.sidebar.number_input(
        "V_max [km/h]",
        min_value=200.0,
        max_value=400.0,
        value=320.0,
        step=1.0,
        help="Speed trap / maximum velocity target. Adding this 4th constraint makes the system well-determined (4 params, 4 targets).",
    )
    target_v_max_ms = target_v_max / 3.6  # Convert to m/s

st.sidebar.header("Search Method")
search_method = st.sidebar.radio(
    "Method",
    options=["Trust-Constr", "L-BFGS-B", "Grid Search", "Nelder-Mead"],
    horizontal=True,
    help="Trust-Constr: bounded optimizer. L-BFGS-B: fast gradient-based. Grid Search: tests all combinations. Nelder-Mead: derivative-free local optimizer.",
)

st.sidebar.header("Parameter Bounds")

with st.sidebar.expander("Aerodynamics Bounds"):
    # Default bounds: ±10% of MVRC_2026 values (c_w_a=1.56, c_z_a_total=4.88)
    c_w_a_min = st.number_input("Drag min [m²]", value=1.00, step=0.2)
    c_w_a_max = st.number_input("Drag max [m²]", value=1.80, step=0.2)
    c_z_a_total_min = st.number_input("Total Downforce min [m²]", value=4.4, step=0.2)
    c_z_a_total_max = st.number_input("Total Downforce max [m²]", value=5.4, step=0.2)

with st.sidebar.expander("Mass Bounds"):
    # Default bounds: ±10% of MVRC_2026 value (m=872)
    mass_min = st.number_input("Mass min [kg]", value=820.0, step=5.0)
    mass_max = st.number_input("Mass max [kg]", value=880.0, step=5.0)

with st.sidebar.expander("Power Bounds"):
    # Default bounds: ±10% of MVRC_2026 value (pow_max=575e3)
    pow_max_min = (
        st.number_input("Power min [kW]", value=520.0, step=10.0) * 1e3
    )  # Convert to W
    pow_max_max = (
        st.number_input("Power max [kW]", value=600.0, step=10.0) * 1e3
    )  # Convert to W

st.sidebar.divider()

col_run, col_abort = st.sidebar.columns(2)
with col_run:
    run_button = st.button("🔍 Find Parameters", type="primary", width="stretch")
with col_abort:
    abort_button = st.button("⏹ Abort", type="secondary", width="stretch")

if abort_button:
    st.session_state.param_id_abort = True
    st.warning("Optimization aborted")
    st.rerun()

# Main area
if run_button:
    # Reset abort flag when starting new search
    st.session_state.param_id_abort = False

    target_sectors = [target_s1, target_s2, target_s3]
    bounds = [
        (c_w_a_min, c_w_a_max),
        (c_z_a_total_min, c_z_a_total_max),
        (mass_min, mass_max),
        (pow_max_min, pow_max_max),
    ]
    base_config = load_vehicle_config(vehicle_base)

    # Initial guess (center of bounds)
    initial_guess = [
        (c_w_a_min + c_w_a_max) / 2,
        (c_z_a_total_min + c_z_a_total_max) / 2,
        (mass_min + mass_max) / 2,
        (pow_max_min + pow_max_max) / 2,
    ]

    # Common optimizer kwargs for trace mode
    trace_kwargs = {}
    if use_trace_mode:
        trace_kwargs = {"ref_distance": ref_distance, "ref_velocity": ref_velocity}

    aborted = False
    best_params = None
    best_sectors = None
    best_lap = None
    best_v_max = None
    best_error = None
    best_sim_distance = None
    best_sim_velocity = None

    if search_method == "Grid Search":
        # Step sizes
        aero_step = 0.2
        mass_step = 5.0
        power_step = 10e3  # 10 kW

        # Calculate number of simulations
        n_drag = len(np.arange(c_w_a_min, c_w_a_max + aero_step / 2, aero_step))
        n_df = len(
            np.arange(c_z_a_total_min, c_z_a_total_max + aero_step / 2, aero_step)
        )
        n_mass = len(np.arange(mass_min, mass_max + mass_step / 2, mass_step))
        n_power = len(np.arange(pow_max_min, pow_max_max + power_step / 2, power_step))
        total_sims = n_drag * n_df * n_mass * n_power

        with st.status(
            f"Running grid search ({total_sims} simulations)...", expanded=True
        ) as status:
            st.write(
                f"Grid: {n_drag} drag × {n_df} downforce × {n_mass} mass × {n_power} power = {total_sims} combinations"
            )
            st.write(
                f"Step sizes: aero={aero_step} m², mass={mass_step} kg, power={power_step / 1e3:.0f} kW"
            )
            if use_trace_mode:
                st.write("Objective: speed trace RMSE (m/s)")
            st.write("---")
            log_container = st.container()

            try:
                (
                    best_params,
                    best_sectors,
                    best_lap,
                    best_v_max,
                    best_error,
                    best_sim_distance,
                    best_sim_velocity,
                ) = run_grid_search(
                    track,
                    base_config,
                    target_sectors,
                    target_v_max_ms,
                    bounds,
                    log_container,
                    aero_step,
                    mass_step,
                    power_step,
                    **trace_kwargs,
                )

                if best_params is None:
                    st.error("All simulations failed!")
                    status.update(label="Failed", state="error")
                    st.stop()

                st.write("---")
                st.write(
                    f"Best found: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW, v_max={best_v_max * 3.6:.1f}km/h"
                )
                status.update(
                    label="Grid search complete!", state="complete", expanded=False
                )

            except AbortException:
                aborted = True
                st.write("---")
                st.warning("Search aborted by user")
                status.update(label="Aborted", state="error", expanded=False)

            except Exception as e:
                st.error(f"Search failed: {e}")
                status.update(label="Failed", state="error")
                st.exception(e)
                st.stop()

    elif search_method == "Nelder-Mead":
        with st.status("Running Nelder-Mead optimization...", expanded=True) as status:
            st.write(
                f"Starting from: drag={initial_guess[0]:.2f}, df={initial_guess[1]:.2f}, m={initial_guess[2]:.0f}, P={initial_guess[3] / 1e3:.0f}kW"
            )
            if use_trace_mode:
                st.write("Objective: speed trace RMSE (m/s)")
            st.write("---")
            log_container = st.container()

            try:
                (
                    best_params,
                    best_sectors,
                    best_lap,
                    best_v_max,
                    best_error,
                    best_sim_distance,
                    best_sim_velocity,
                ) = run_nelder_mead(
                    track,
                    base_config,
                    target_sectors,
                    target_v_max_ms,
                    initial_guess,
                    bounds,
                    log_container,
                    **trace_kwargs,
                )

                if best_params is None:
                    st.error("Optimization failed!")
                    status.update(label="Failed", state="error")
                    st.stop()

                st.write("---")
                st.write(
                    f"Optimum: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW, v_max={best_v_max * 3.6:.1f}km/h"
                )
                status.update(
                    label="Optimization complete!", state="complete", expanded=False
                )

            except AbortException:
                aborted = True
                st.write("---")
                st.warning("Optimization aborted by user")
                status.update(label="Aborted", state="error", expanded=False)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                status.update(label="Failed", state="error")
                st.exception(e)
                st.stop()

    elif search_method == "Trust-Constr":
        with st.status(
            "Running Trust-Region Constrained optimization...", expanded=True
        ) as status:
            st.write(
                f"Starting from: drag={initial_guess[0]:.2f}, df={initial_guess[1]:.2f}, m={initial_guess[2]:.0f}, P={initial_guess[3] / 1e3:.0f}kW"
            )
            if use_trace_mode:
                st.write("Objective: speed trace RMSE (m/s)")
            st.write("---")
            log_container = st.container()

            try:
                (
                    best_params,
                    best_sectors,
                    best_lap,
                    best_v_max,
                    best_error,
                    best_sim_distance,
                    best_sim_velocity,
                ) = run_trust_constr(
                    track,
                    base_config,
                    target_sectors,
                    target_v_max_ms,
                    initial_guess,
                    bounds,
                    log_container,
                    **trace_kwargs,
                )

                if best_params is None:
                    st.error("Optimization failed!")
                    status.update(label="Failed", state="error")
                    st.stop()

                st.write("---")
                st.write(
                    f"Optimum: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW, v_max={best_v_max * 3.6:.1f}km/h"
                )
                status.update(
                    label="Optimization complete!", state="complete", expanded=False
                )

            except AbortException:
                aborted = True
                st.write("---")
                st.warning("Optimization aborted by user")
                status.update(label="Aborted", state="error", expanded=False)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                status.update(label="Failed", state="error")
                st.exception(e)
                st.stop()

    else:  # L-BFGS-B
        with st.status("Running L-BFGS-B optimization...", expanded=True) as status:
            st.write(
                f"Starting from: drag={initial_guess[0]:.2f}, df={initial_guess[1]:.2f}, m={initial_guess[2]:.0f}, P={initial_guess[3] / 1e3:.0f}kW"
            )
            if use_trace_mode:
                st.write("Objective: speed trace RMSE (m/s)")
            st.write("---")
            log_container = st.container()

            try:
                (
                    best_params,
                    best_sectors,
                    best_lap,
                    best_v_max,
                    best_error,
                    best_sim_distance,
                    best_sim_velocity,
                ) = run_lbfgsb(
                    track,
                    base_config,
                    target_sectors,
                    target_v_max_ms,
                    initial_guess,
                    bounds,
                    log_container,
                    **trace_kwargs,
                )

                if best_params is None:
                    st.error("Optimization failed!")
                    status.update(label="Failed", state="error")
                    st.stop()

                st.write("---")
                st.write(
                    f"Optimum: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW, v_max={best_v_max * 3.6:.1f}km/h"
                )
                status.update(
                    label="Optimization complete!", state="complete", expanded=False
                )

            except AbortException:
                aborted = True
                st.write("---")
                st.warning("Optimization aborted by user")
                status.update(label="Aborted", state="error", expanded=False)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                status.update(label="Failed", state="error")
                st.exception(e)
                st.stop()

    # Clear abort flag
    st.session_state.param_id_abort = False

    # Skip storing results if aborted with no results
    if best_params is None:
        st.stop()

    # Store results
    c_w_a_opt, c_z_a_total_opt, mass_opt, pow_max_opt = best_params
    final_sectors, final_lap = best_sectors, best_lap

    # Calculate front/rear split for display
    lf = base_config["general"]["lf"]
    lr = base_config["general"]["lr"]
    wheelbase = lf + lr
    c_z_a_f_opt = c_z_a_total_opt * (lr / wheelbase)
    c_z_a_r_opt = c_z_a_total_opt * (lf / wheelbase)

    # Run a final full-resolution simulation with best parameters for detailed visualization
    vehicle_pars = build_vehicle_pars(
        base_config, c_w_a_opt, c_z_a_total_opt, mass_opt, pow_max_opt
    )
    final_track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 5.0,
        "curv_filt_width": 10.0,
        "use_drs": True,
        "use_pit": False,
    }
    final_solver_opts = {
        "vehicle": None,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-5,
        "custom_vehicle_pars": vehicle_pars,
    }
    final_driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": 4.0e6,
        "em_strategy": "FCFB",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }
    try:
        final_sim_result = run_simulation_advanced(
            final_track_opts, final_solver_opts, final_driver_opts
        )
    except Exception:
        final_sim_result = None

    st.session_state.param_id_result = {
        "success": True,
        "c_w_a": c_w_a_opt,
        "c_z_a_total": c_z_a_total_opt,
        "c_z_a_f": c_z_a_f_opt,
        "c_z_a_r": c_z_a_r_opt,
        "mass": mass_opt,
        "pow_max": pow_max_opt,
        "target_sectors": target_sectors,
        "simulated_sectors": final_sectors,
        "simulated_lap": final_lap,
        "target_v_max": target_v_max_ms,
        "simulated_v_max": best_v_max,
        "track": track,
        "vehicle": vehicle_base,
        "use_trace": use_trace_mode,
        "sim_distance": best_sim_distance,
        "sim_velocity": best_sim_velocity,
        "ref_distance": ref_distance,
        "ref_velocity": ref_velocity,
        "sim_result": final_sim_result,
        "fastf1_trace": st.session_state.fastf1_trace if use_trace_mode else None,
    }

# Display results
if st.session_state.param_id_result is not None:
    res = st.session_state.param_id_result

    st.divider()
    st.header("Identified Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drag (c_w × A)", f"{res['c_w_a']:.2f} m²")
    with col2:
        st.metric("Total Downforce (c_z × A)", f"{res['c_z_a_total']:.2f} m²")
    with col3:
        st.metric("Mass", f"{res['mass']:.0f} kg")
    with col4:
        st.metric("Power", f"{res['pow_max'] / 1e3:.0f} kW")

    st.caption(
        f"Downforce split: Front {res['c_z_a_f']:.2f} m² / Rear {res['c_z_a_r']:.2f} m² (based on CoG position)"
    )

    st.divider()
    st.header("Sector Time Comparison")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Sector**")
        st.write("S1")
        st.write("S2")
        st.write("S3")
        st.write("**Total**")

    with col2:
        st.markdown("**Target**")
        st.write(f"{res['target_sectors'][0]:.3f}s")
        st.write(f"{res['target_sectors'][1]:.3f}s")
        st.write(f"{res['target_sectors'][2]:.3f}s")
        target_total = sum(res["target_sectors"])
        st.write(f"**{target_total:.3f}s**")

    with col3:
        st.markdown("**Simulated**")
        st.write(f"{res['simulated_sectors'][0]:.3f}s")
        st.write(f"{res['simulated_sectors'][1]:.3f}s")
        st.write(f"{res['simulated_sectors'][2]:.3f}s")
        st.write(f"**{res['simulated_lap']:.3f}s**")

    with col4:
        st.markdown("**Difference**")
        for i in range(3):
            diff = res["simulated_sectors"][i] - res["target_sectors"][i]
            color = (
                "green" if abs(diff) < 0.5 else "orange" if abs(diff) < 1.0 else "red"
            )
            st.markdown(f":{color}[{diff:+.3f}s]")
        total_diff = res["simulated_lap"] - target_total
        color = (
            "green"
            if abs(total_diff) < 0.5
            else "orange"
            if abs(total_diff) < 1.0
            else "red"
        )
        st.markdown(f"**:{color}[{total_diff:+.3f}s]**")

    # Display v_max comparison
    st.divider()
    st.header("Max Velocity Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target V_max", f"{res['target_v_max'] * 3.6:.1f} km/h")
    with col2:
        st.metric("Simulated V_max", f"{res['simulated_v_max'] * 3.6:.1f} km/h")
    with col3:
        v_diff = (res["simulated_v_max"] - res["target_v_max"]) * 3.6
        color = "green" if abs(v_diff) < 5 else "orange" if abs(v_diff) < 10 else "red"
        st.markdown(f"**Difference:** :{color}[{v_diff:+.1f} km/h]")

    # Speed trace overlay plot (when trace data is available)
    if (
        res.get("use_trace")
        and res.get("sim_distance") is not None
        and res.get("ref_distance") is not None
    ):
        st.divider()
        st.header("Speed Trace Comparison")

        sim_dist = res["sim_distance"]
        sim_vel = res["sim_velocity"]
        r_dist = res["ref_distance"]
        r_vel = res["ref_velocity"]

        # Normalize distances to 0-1 for overlay
        sim_dist_norm = sim_dist / sim_dist[-1]
        ref_dist_norm = r_dist / r_dist[-1]

        # Interpolate sim onto ref grid for delta calculation
        sim_vel_interp = np.interp(ref_dist_norm, sim_dist_norm, sim_vel)
        delta_vel = sim_vel_interp - r_vel

        rmse = compute_trace_error(sim_dist, sim_vel, r_dist, r_vel)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.75, 0.25],
            vertical_spacing=0.06,
        )
        fig.add_trace(
            go.Scatter(
                x=ref_dist_norm * r_dist[-1] / 1000,
                y=r_vel * 3.6,
                name="FastF1 Reference",
                line=dict(color="#1f77b4"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sim_dist / 1000,
                y=sim_vel * 3.6,
                name="Simulation (Best Fit)",
                line=dict(color="#d62728"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ref_dist_norm * r_dist[-1] / 1000,
                y=delta_vel * 3.6,
                name="Delta",
                line=dict(color="#2ca02c"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1
        )
        fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1)
        fig.update_yaxes(title_text="Delta [km/h]", row=2, col=1)
        fig.update_xaxes(title_text="Distance [km]", row=2, col=1)
        fig.update_layout(
            title=f"Speed Trace Overlay (RMSE: {rmse:.2f} m/s)",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch")

        # Additional telemetry comparison plots
        ff1_trace = res.get("fastf1_trace")
        sim_result = res.get("sim_result")

        if ff1_trace is not None:
            # Shared normalized distance grids
            ref_dist_km = ref_dist_norm * r_dist[-1] / 1000
            sim_dist_km = sim_dist / 1000 if sim_result is not None else None
            sim_dist_full_km = (
                sim_result.distance / 1000 if sim_result is not None else None
            )

            # --- Gear comparison ---
            if ff1_trace.get("gear") is not None:
                st.subheader("Gear Comparison")
                gear_data = (
                    ff1_trace["gear"][: len(ref_dist_km)]
                    if len(ff1_trace["gear"]) >= len(ref_dist_km)
                    else ff1_trace["gear"]
                )
                fig_gear = go.Figure()
                fig_gear.add_trace(
                    go.Scatter(
                        x=ref_dist_km,
                        y=gear_data,
                        name="FastF1",
                        line=dict(color="#1f77b4", shape="hv"),
                    )
                )
                if sim_result is not None:
                    fig_gear.add_trace(
                        go.Scatter(
                            x=sim_dist_full_km,
                            y=sim_result.gear,
                            name="Simulation",
                            line=dict(color="#d62728", shape="hv"),
                        )
                    )
                fig_gear.update_layout(
                    xaxis_title="Distance [km]",
                    yaxis_title="Gear",
                    height=250,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                )
                st.plotly_chart(fig_gear, width="stretch")

            # --- RPM comparison ---
            if ff1_trace.get("rpm") is not None:
                st.subheader("RPM Comparison")
                rpm_data = (
                    ff1_trace["rpm"][: len(ref_dist_km)]
                    if len(ff1_trace["rpm"]) >= len(ref_dist_km)
                    else ff1_trace["rpm"]
                )
                fig_rpm = go.Figure()
                fig_rpm.add_trace(
                    go.Scatter(
                        x=ref_dist_km,
                        y=rpm_data,
                        name="FastF1",
                        line=dict(color="#1f77b4"),
                    )
                )
                if sim_result is not None and sim_result.rpm is not None:
                    fig_rpm.add_trace(
                        go.Scatter(
                            x=sim_dist_full_km,
                            y=sim_result.rpm,
                            name="Simulation",
                            line=dict(color="#d62728"),
                        )
                    )
                fig_rpm.update_layout(
                    xaxis_title="Distance [km]",
                    yaxis_title="RPM",
                    height=250,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                )
                st.plotly_chart(fig_rpm, width="stretch")

            # --- DRS comparison ---
            if ff1_trace.get("drs_active") is not None:
                st.subheader("DRS Comparison")
                drs_data = (
                    ff1_trace["drs_active"][: len(ref_dist_km)]
                    if len(ff1_trace["drs_active"]) >= len(ref_dist_km)
                    else ff1_trace["drs_active"]
                )
                fig_drs = go.Figure()
                fig_drs.add_trace(
                    go.Scatter(
                        x=ref_dist_km,
                        y=drs_data,
                        name="FastF1",
                        line=dict(color="#1f77b4", shape="hv"),
                        fill="tozeroy",
                        fillcolor="rgba(31,119,180,0.3)",
                    )
                )
                if sim_result is not None and sim_result.drs is not None:
                    fig_drs.add_trace(
                        go.Scatter(
                            x=sim_dist_full_km,
                            y=sim_result.drs.astype(float),
                            name="Simulation",
                            line=dict(color="#d62728", shape="hv"),
                            fill="tozeroy",
                            fillcolor="rgba(214,39,40,0.3)",
                        )
                    )
                fig_drs.update_layout(
                    xaxis_title="Distance [km]",
                    yaxis_title="DRS",
                    height=200,
                    hovermode="x unified",
                    yaxis=dict(tickvals=[0, 1], ticktext=["Closed", "Open"]),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                )
                st.plotly_chart(fig_drs, width="stretch")

            # --- Throttle (FastF1) with sim longitudinal acceleration as proxy ---
            if ff1_trace.get("throttle") is not None:
                st.subheader("Throttle & Longitudinal Acceleration")
                throttle_data = (
                    ff1_trace["throttle"][: len(ref_dist_km)]
                    if len(ff1_trace["throttle"]) >= len(ref_dist_km)
                    else ff1_trace["throttle"]
                )
                fig_thr = make_subplots(specs=[[{"secondary_y": True}]])
                fig_thr.add_trace(
                    go.Scatter(
                        x=ref_dist_km,
                        y=throttle_data,
                        name="FastF1 Throttle [%]",
                        line=dict(color="#1f77b4"),
                    ),
                    secondary_y=False,
                )
                if sim_result is not None:
                    fig_thr.add_trace(
                        go.Scatter(
                            x=sim_dist_full_km,
                            y=sim_result.acceleration,
                            name="Sim Longitudinal Accel [m/s²]",
                            line=dict(color="#d62728"),
                            opacity=0.6,
                        ),
                        secondary_y=True,
                    )
                fig_thr.update_xaxes(title_text="Distance [km]")
                fig_thr.update_yaxes(title_text="Throttle [%]", secondary_y=False)
                fig_thr.update_yaxes(title_text="Acceleration [m/s²]", secondary_y=True)
                fig_thr.update_layout(
                    height=250,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                )
                st.plotly_chart(fig_thr, width="stretch")

            # --- Brake (FastF1) with sim negative acceleration as proxy ---
            if ff1_trace.get("brake") is not None:
                st.subheader("Brake & Deceleration")
                brake_data = ff1_trace["brake"]
                # Brake can be boolean (0/1) or percentage (0-100)
                brake_vals = (
                    brake_data[: len(ref_dist_km)]
                    if len(brake_data) >= len(ref_dist_km)
                    else brake_data
                )
                fig_brk = make_subplots(specs=[[{"secondary_y": True}]])
                fig_brk.add_trace(
                    go.Scatter(
                        x=ref_dist_km,
                        y=brake_vals,
                        name="FastF1 Brake",
                        line=dict(color="#1f77b4", shape="hv"),
                        fill="tozeroy",
                        fillcolor="rgba(31,119,180,0.4)",
                    ),
                    secondary_y=False,
                )
                if sim_result is not None:
                    # Show negative acceleration (braking) as positive values
                    decel = np.clip(-sim_result.acceleration, 0, None)
                    fig_brk.add_trace(
                        go.Scatter(
                            x=sim_dist_full_km,
                            y=decel,
                            name="Sim Deceleration [m/s²]",
                            line=dict(color="#d62728"),
                            opacity=0.6,
                        ),
                        secondary_y=True,
                    )
                fig_brk.update_xaxes(title_text="Distance [km]")
                fig_brk.update_yaxes(title_text="Brake", secondary_y=False)
                fig_brk.update_yaxes(title_text="Deceleration [m/s²]", secondary_y=True)
                fig_brk.update_layout(
                    height=250,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                    ),
                )
                st.plotly_chart(fig_brk, width="stretch")

    # Full simulation data visualization
    if res.get("sim_result") is not None:
        st.divider()
        st.header("Simulation Data")
        render_simulation_plots(res["sim_result"], key_prefix="paramid_")

    st.divider()
    st.caption(f"Track: {res['track']} | Base vehicle: {res['vehicle']}")

else:
    # Show preview of downloaded telemetry if available
    if st.session_state.fastf1_trace is not None:
        trace = st.session_state.fastf1_trace
        driver_str = trace["driver"] if trace["driver"] else "Fastest"
        title_str = f"{trace['year']} {trace['gp']} ({trace['session']}) - {driver_str}"
        st.subheader(f"Downloaded Telemetry Preview")

        dist_km = trace["distance"] / 1000

        # Count available channels for subplot layout
        channels = []
        channels.append(("Speed [km/h]", trace["velocity"] * 3.6, "tab:blue", "line"))
        if trace.get("gear") is not None:
            channels.append(("Gear", trace["gear"], "tab:orange", "step"))
        if trace.get("rpm") is not None:
            channels.append(("RPM", trace["rpm"], "tab:green", "line"))
        if trace.get("throttle") is not None:
            channels.append(("Throttle [%]", trace["throttle"], "tab:blue", "line"))
        if trace.get("brake") is not None:
            channels.append(("Brake", trace["brake"], "tab:red", "fill"))
        if trace.get("drs_active") is not None:
            channels.append(("DRS", trace["drs_active"], "tab:purple", "fill"))

        n_panels = len(channels)
        fig = make_subplots(
            rows=n_panels,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
        )

        # Color map for plotly hex colors
        color_map = {
            "tab:blue": "#1f77b4",
            "tab:orange": "#ff7f0e",
            "tab:green": "#2ca02c",
            "tab:red": "#d62728",
            "tab:purple": "#9467bd",
        }

        for i, (label, data, color, style) in enumerate(channels, 1):
            d = dist_km[: len(data)] if len(data) <= len(dist_km) else dist_km
            v = data[: len(d)] if len(data) >= len(d) else data
            hex_color = color_map.get(color, color)
            if style == "step":
                fig.add_trace(
                    go.Scatter(
                        x=d,
                        y=v,
                        name=label,
                        line=dict(color=hex_color, shape="hv"),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )
            elif style == "fill":
                fig.add_trace(
                    go.Scatter(
                        x=d,
                        y=v,
                        name=label,
                        line=dict(color=hex_color, shape="hv"),
                        fill="tozeroy",
                        fillcolor=hex_color.replace(")", ",0.5)").replace("rgb", "rgba")
                        if "rgb" in hex_color
                        else f"rgba{tuple(list(int(hex_color.lstrip('#')[j : j + 2], 16) for j in (0, 2, 4)) + [0.5])}",
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=d,
                        y=v,
                        name=label,
                        line=dict(color=hex_color),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )
            fig.update_yaxes(title_text=label, row=i, col=1)

        fig.update_xaxes(title_text="Distance [km]", row=n_panels, col=1)
        fig.update_layout(
            title=title_str,
            height=200 * n_panels,
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Lap time: {int(trace['lap_time'] // 60)}:{trace['lap_time'] % 60:06.3f} | "
            f"V_max: {trace['v_max'] * 3.6:.1f} km/h | "
            f"Sectors: {trace['sector_times'][0]:.3f} / {trace['sector_times'][1]:.3f} / {trace['sector_times'][2]:.3f}"
        )

    st.info(
        "Select a track, vehicle, enter target sector times (or download FastF1 telemetry), then click **Find Parameters**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### How it works

        This tool uses optimization to find vehicle parameters that match
        your input targets.

        **Parameters identified (4):**
        - Drag coefficient x area (c_w x A)
        - Total downforce coefficient x area (c_z x A)
        - Vehicle mass
        - Engine power (pow_max)

        **Target modes:**
        - **Manual:** 3 sector times + max velocity (4 constraints)
        - **FastF1 Telemetry:** Full speed trace from real F1 data (hundreds of constraints)
        """)

    with col2:
        st.markdown("""
        ### Tips

        - Use realistic sector times for the selected track
        - Adjust parameter bounds if the optimization struggles
        - The optimization runs multiple simulations, so it may take a few minutes
        - **FastF1 mode** uses the full speed trace as objective (RMSE in m/s),
          capturing braking zones, apex speeds, and straights
        """)
