"""
Parameter Identification Page

Determine vehicle aerodynamic parameters and weight from sector times and max velocity.
Uses 4 targets (S1, S2, S3, v_max) to identify 4 parameters (drag, downforce, mass, power).
"""

import time

import numpy as np
import streamlit as st
from scipy.optimize import Bounds, minimize

from helpers.simulation import (
    get_available_tracks,
    get_available_vehicles,
    run_simulation_advanced,
)

st.set_page_config(
    page_title="Parameter Identification - Laptime Sim",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Parameter Identification")
st.caption("Determine vehicle parameters from sector times and max velocity")

# Initialize session state
if "param_id_result" not in st.session_state:
    st.session_state.param_id_result = None
if "param_id_abort" not in st.session_state:
    st.session_state.param_id_abort = False


class AbortException(Exception):
    """Raised when user aborts the optimization."""

    pass


def check_abort():
    """Check if abort was requested and raise exception if so."""
    if st.session_state.param_id_abort:
        raise AbortException("Optimization aborted by user")


# Base vehicle configurations
VEHICLE_CONFIGS = {
    "F1_Shanghai": {
        "powertrain_type": "hybrid",
        "general": {
            "lf": 1.968,
            "lr": 1.632,
            "h_cog": 0.335,
            "sf": 1.6,
            "sr": 1.6,
            "f_roll": 0.03,
            "g": 9.81,
            "rho_air": 1.18,
            "drs_factor": 0.17,
        },
        "engine": {
            "topology": "RWD",
            "pow_max": 575e3,
            "pow_diff": 41e3,
            "n_begin": 10500.0,
            "n_max": 11400.0,
            "n_end": 12200.0,
            "be_max": 100.0,
            "pow_e_motor": 120e3,
            "eta_e_motor": 0.9,
            "eta_e_motor_re": 0.15,
            "eta_etc_re": 0.10,
            "vel_min_e_motor": 27.777,
            "torque_e_motor_max": 200.0,
        },
        "gearbox": {
            "i_trans": [0.04, 0.070, 0.095, 0.117, 0.143, 0.172, 0.190, 0.206],
            "n_shift": [
                10000.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                13000.0,
            ],
            "e_i": [1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07],
            "eta_g": 0.96,
        },
        "tires": {
            "f": {
                "circ_ref": 2.073,
                "fz_0": 3000.0,
                "mux": 1.65,
                "muy": 1.85,
                "dmux_dfz": -5.0e-5,
                "dmuy_dfz": -5.0e-5,
            },
            "r": {
                "circ_ref": 2.073,
                "fz_0": 3000.0,
                "mux": 1.95,
                "muy": 2.15,
                "dmux_dfz": -5.0e-5,
                "dmuy_dfz": -5.0e-5,
            },
            "tire_model_exp": 2.0,
        },
        "series": "F1",
    },
    "MVRC_2026": {
        "powertrain_type": "hybrid",
        "general": {
            "lf": 1.557,
            "lr": 1.842,
            "h_cog": 0.300,
            "sf": 1.618,
            "sr": 1.536,
            "f_roll": 0.03,
            "g": 9.81,
            "rho_air": 1.18,
            "drs_factor": 0.17,
        },
        "engine": {
            "topology": "RWD",
            "pow_max": 575e3,
            "pow_diff": 41e3,
            "n_begin": 10500.0,
            "n_max": 11400.0,
            "n_end": 12200.0,
            "be_max": 100.0,
            "pow_e_motor": 120e3,
            "eta_e_motor": 0.9,
            "eta_e_motor_re": 0.15,
            "eta_etc_re": 0.10,
            "vel_min_e_motor": 27.777,
            "torque_e_motor_max": 200.0,
        },
        "gearbox": {
            "i_trans": [0.04, 0.070, 0.095, 0.117, 0.143, 0.172, 0.190, 0.206],
            "n_shift": [
                10000.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                11800.0,
                13000.0,
            ],
            "e_i": [1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07],
            "eta_g": 0.96,
        },
        "tires": {
            "f": {
                "circ_ref": 2.073,
                "fz_0": 3000.0,
                "mux": 1.65,
                "muy": 1.85,
                "dmux_dfz": -5.0e-5,
                "dmuy_dfz": -5.0e-5,
            },
            "r": {
                "circ_ref": 2.073,
                "fz_0": 3000.0,
                "mux": 1.95,
                "muy": 2.15,
                "dmux_dfz": -5.0e-5,
                "dmuy_dfz": -5.0e-5,
            },
            "tire_model_exp": 2.0,
        },
        "series": "F1",
    },
}


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
        "use_drs1": False,  # Disable DRS for simplicity
        "use_drs2": False,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "series": base_config.get("series", "F1"),
        "limit_braking_weak_side": None,  # Skip weak side calculation
        "v_start": 100.0 / 3.6,
        "find_v_start": False,  # Skip velocity search
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
        return result.sector_times, result.lap_time, v_max
    except Exception:
        return None, None, None


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
):
    """
    Run a coarse grid search to find approximate best parameters.
    Uses fixed step sizes for each parameter.
    Returns best result found, even if aborted.
    """
    c_w_a_vals = np.arange(bounds[0][0], bounds[0][1] + aero_step / 2, aero_step)
    c_z_a_vals = np.arange(bounds[1][0], bounds[1][1] + aero_step / 2, aero_step)
    mass_vals = np.arange(bounds[2][0], bounds[2][1] + mass_step / 2, mass_step)
    pow_max_vals = np.arange(bounds[3][0], bounds[3][1] + power_step / 2, power_step)

    best_error = float("inf")
    best_params = None
    best_sectors = None
    best_lap = None
    best_v_max = None

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

                        sectors, lap_time, v_max = run_sim_with_params(
                            track, base_config, c_w_a, c_z_a, mass, pow_max
                        )
                        elapsed = time.time() - start

                        if sectors is None:
                            log_container.write(
                                f"[{count}/{total}] FAILED - drag={c_w_a:.2f}, df={c_z_a:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
                            )
                            continue

                        sector_error = sum(
                            (s - t) ** 2 for s, t in zip(sectors, target_sectors)
                        )
                        v_max_error = v_max_weight * (v_max - target_v_max) ** 2
                        error = sector_error + v_max_error

                        status = "**BEST**" if error < best_error else ""
                        log_container.write(
                            f"[{count}/{total}] drag={c_w_a:.2f}, df={c_z_a:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
                            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, err={error:.2f} ({elapsed:.2f}s) {status}"
                        )

                        if error < best_error:
                            best_error = error
                            best_params = [c_w_a, c_z_a, mass, pow_max]
                            best_sectors = sectors
                            best_lap = lap_time
                            best_v_max = v_max
    except AbortException:
        # Log best result found before abort
        if best_params is not None:
            log_container.write(
                f"Best before abort: drag={best_params[0]:.2f}, df={best_params[1]:.2f}, m={best_params[2]:.0f}, P={best_params[3] / 1e3:.0f}kW"
            )
        raise

    return best_params, best_sectors, best_lap, best_v_max, best_error


def run_nelder_mead(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
):
    """
    Run Nelder-Mead optimization starting from an initial guess.
    Faster than grid search but may find local minimum.
    Parameters are clipped to bounds.
    """
    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
    ]  # [params, sectors, lap_time, v_max, error]

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
        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
        v_max_error = v_max_weight * (v_max - target_v_max) ** 2
        error = sector_error + v_max_error
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, err={error:.2f} ({elapsed:.1f}s)"
        )

        # Track best result for abort case (store clipped values)
        if error < best_result[4]:
            best_result[0] = [c_w_a, c_z_a_total, mass, pow_max]
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error

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

        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return [c_w_a, c_z_a_total, mass, pow_max], sectors, lap_time, v_max, result.fun
    except AbortException:
        # Return best result found so far
        raise


def run_trust_constr(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
):
    """
    Run Trust-Region Constrained optimization starting from an initial guess.
    Supports bounds natively for better convergence.
    Uses normalized parameters (0-1) internally for better scaling.
    """
    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
    ]  # [params, sectors, lap_time, v_max, error]

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
        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
        v_max_error = v_max_weight * (v_max - target_v_max) ** 2
        error = sector_error + v_max_error
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, err={error:.2f} ({elapsed:.1f}s)"
        )

        # Track best result for abort case (store real values)
        if error < best_result[4]:
            best_result[0] = list(real_params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error

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
        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return list(real_params), sectors, lap_time, v_max, result.fun
    except AbortException:
        # Return best result found so far
        raise


def run_lbfgsb(
    track,
    base_config,
    target_sectors,
    target_v_max,
    initial_guess,
    bounds,
    log_container,
):
    """
    Run L-BFGS-B optimization starting from an initial guess.
    Fast gradient-based optimizer with bounds support.
    Uses normalized parameters (0-1) internally for better scaling.
    """
    eval_count = [0]
    best_result = [
        None,
        None,
        None,
        None,
        float("inf"),
    ]  # [params, sectors, lap_time, v_max, error]

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
        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(
                f"[{eval_count[0]}] FAILED - drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW"
            )
            return 1e6

        sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
        v_max_error = v_max_weight * (v_max - target_v_max) ** 2
        error = sector_error + v_max_error
        log_container.write(
            f"[{eval_count[0]}] drag={c_w_a:.2f}, df={c_z_a_total:.2f}, m={mass:.0f}, P={pow_max / 1e3:.0f}kW → "
            f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, err={error:.2f} ({elapsed:.1f}s)"
        )

        # Track best result for abort case (store real values)
        if error < best_result[4]:
            best_result[0] = list(real_params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error

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
        sectors, lap_time, v_max = run_sim_with_params(
            track, base_config, c_w_a, c_z_a_total, mass, pow_max
        )

        return list(real_params), sectors, lap_time, v_max, result.fun
    except AbortException:
        # Return best result found so far
        raise


# Sidebar controls
st.sidebar.header("Configuration")

# Track selection
available_tracks = get_available_tracks()
track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Shanghai") if "Shanghai" in available_tracks else 0,
)

# Vehicle base configuration
vehicle_base = st.sidebar.selectbox(
    "Base Vehicle",
    options=list(VEHICLE_CONFIGS.keys()),
)

st.sidebar.header("Target Sector Times")

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    target_s1 = st.number_input(
        "S1 [s]", min_value=10.0, max_value=120.0, value=24.0, step=0.1, format="%.3f"
    )
with col2:
    target_s2 = st.number_input(
        "S2 [s]", min_value=10.0, max_value=120.0, value=26.5, step=0.1, format="%.3f"
    )
with col3:
    target_s3 = st.number_input(
        "S3 [s]", min_value=10.0, max_value=120.0, value=40.4, step=0.1, format="%.3f"
    )

target_total = target_s1 + target_s2 + target_s3
st.sidebar.caption(f"Total: **{int(target_total // 60)}:{target_total % 60:06.3f}**")

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
    c_w_a_min = st.number_input("Drag min [m²]", value=1.40, step=0.2)
    c_w_a_max = st.number_input("Drag max [m²]", value=1.72, step=0.2)
    c_z_a_total_min = st.number_input("Total Downforce min [m²]", value=4.4, step=0.2)
    c_z_a_total_max = st.number_input("Total Downforce max [m²]", value=5.4, step=0.2)

with st.sidebar.expander("Mass Bounds"):
    # Default bounds: ±10% of MVRC_2026 value (m=872)
    mass_min = st.number_input("Mass min [kg]", value=720.0, step=5.0)
    mass_max = st.number_input("Mass max [kg]", value=780.0, step=5.0)

with st.sidebar.expander("Power Bounds"):
    # Default bounds: ±10% of MVRC_2026 value (pow_max=575e3)
    pow_max_min = (
        st.number_input("Power min [kW]", value=517.0, step=10.0) * 1e3
    )  # Convert to W
    pow_max_max = (
        st.number_input("Power max [kW]", value=633.0, step=10.0) * 1e3
    )  # Convert to W

st.sidebar.divider()

col_run, col_abort = st.sidebar.columns(2)
with col_run:
    run_button = st.button(
        "🔍 Find Parameters", type="primary", use_container_width=True
    )
with col_abort:
    abort_button = st.button("⏹ Abort", type="secondary", use_container_width=True)

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
    base_config = VEHICLE_CONFIGS[vehicle_base]

    # Initial guess (center of bounds)
    initial_guess = [
        (c_w_a_min + c_w_a_max) / 2,
        (c_z_a_total_min + c_z_a_total_max) / 2,
        (mass_min + mass_max) / 2,
        (pow_max_min + pow_max_max) / 2,
    ]

    aborted = False
    best_params = None
    best_sectors = None
    best_lap = None
    best_v_max = None
    best_error = None

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
            st.write("---")
            log_container = st.container()

            try:
                best_params, best_sectors, best_lap, best_v_max, best_error = (
                    run_grid_search(
                        track,
                        base_config,
                        target_sectors,
                        target_v_max_ms,
                        bounds,
                        log_container,
                        aero_step,
                        mass_step,
                        power_step,
                    )
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
            st.write("---")
            log_container = st.container()

            try:
                best_params, best_sectors, best_lap, best_v_max, best_error = (
                    run_nelder_mead(
                        track,
                        base_config,
                        target_sectors,
                        target_v_max_ms,
                        initial_guess,
                        bounds,
                        log_container,
                    )
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
            st.write("---")
            log_container = st.container()

            try:
                best_params, best_sectors, best_lap, best_v_max, best_error = (
                    run_trust_constr(
                        track,
                        base_config,
                        target_sectors,
                        target_v_max_ms,
                        initial_guess,
                        bounds,
                        log_container,
                    )
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
            st.write("---")
            log_container = st.container()

            try:
                best_params, best_sectors, best_lap, best_v_max, best_error = (
                    run_lbfgsb(
                        track,
                        base_config,
                        target_sectors,
                        target_v_max_ms,
                        initial_guess,
                        bounds,
                        log_container,
                    )
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

    st.divider()
    st.caption(f"Track: {res['track']} | Base vehicle: {res['vehicle']}")

else:
    st.info(
        "👈 Select a track, vehicle, enter target sector times and max velocity, then click **Find Parameters**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### How it works

        This tool uses optimization to find vehicle parameters that match
        your input targets (3 sector times + max velocity).

        **Parameters identified (4):**
        - Drag coefficient × area (c_w × A)
        - Total downforce coefficient × area (c_z × A)
        - Vehicle mass
        - Engine power (pow_max)

        **Target constraints (4):**
        - Sector 1, 2, 3 times
        - Maximum velocity (speed trap)

        Having 4 targets for 4 parameters makes the system well-determined,
        leading to better convergence than using sector times alone.
        """)

    with col2:
        st.markdown("""
        ### Tips

        - Use realistic sector times for the selected track
        - Adjust parameter bounds if the optimization struggles
        - The optimization runs multiple simulations, so it may take a few minutes
        - Results work best when sector times are achievable with the base vehicle's powertrain
        """)
