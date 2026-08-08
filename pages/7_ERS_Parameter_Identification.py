"""
ERS Parameter Identification Page

Extends parameter identification with ERS-specific parameters:
p_harvest_straight and ers_harvest_speed_kmh, alongside the standard
aero/mass/power parameters. Optimized for QUALY strategy matching.
"""

import ast
import configparser
import copy
import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import Bounds, minimize

from helpers.fastf1_data import (
    compute_trace_r2,
    DEFAULT_TRACE_METRIC,
    TRACE_METRICS,
    TRACK_NAME_MAP,
    compute_trace_error,
    get_available_gps,
    get_available_years,
    get_drivers_in_session,
    load_speed_trace,
)
from helpers.simulation import (
    DEFAULT_EM_STRATEGY,
    EM_STRATEGIES,
    get_available_tracks,
    get_available_vehicles,
    run_simulation_advanced,
)
from helpers.visualization import render_simulation_plots

# The parameter search runs hundreds of simulations, so it uses a coarser track resolution
# than the final result. The final simulation behind the plots is re-run at
# FINAL_PLOT_STEPSIZE: at 5 m the velocity profile visibly saw-tooths through long constant
# radius corners (the car accelerates over a step, overshoots the cornering limit and is
# clipped back), and the ripple amplitude scales with the step size.
# Metric used to compare the simulated speed trace with the FastF1 reference.
# Set from the sidebar before a search starts; the objective functions read it at
# call time, so it does not need threading through every optimiser signature.
TRACE_METRIC = DEFAULT_TRACE_METRIC

SEARCH_STEPSIZE_DEFAULT = 2.5  # [m] resolution used during the parameter search
FINAL_PLOT_STEPSIZE = 1.0      # [m] resolution of the final simulation used for the plots


st.set_page_config(
    page_title="ERS Parameter Identification - Laptime Sim",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ ERS Parameter Identification")
st.caption(
    "Identify vehicle parameters including ERS harvest power and harvest speed threshold"
)

# Initialize session state
if "ers_param_id_result" not in st.session_state:
    st.session_state.ers_param_id_result = None
if "ers_param_id_abort" not in st.session_state:
    st.session_state.ers_param_id_abort = False
if "ers_fastf1_trace" not in st.session_state:
    st.session_state.ers_fastf1_trace = None


class AbortException(Exception):
    pass


class EarlyStopException(Exception):
    pass


def check_abort():
    if st.session_state.ers_param_id_abort:
        raise AbortException("Optimization aborted by user")


def load_vehicle_config(vehicle_name: str) -> dict:
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ini_path = os.path.join(
        repo_path, "laptimesim", "input", "vehicles", f"{vehicle_name}.ini"
    )
    parser = configparser.ConfigParser()
    if not parser.read(ini_path):
        raise RuntimeError(f"Vehicle config file not found: {ini_path}")
    return ast.literal_eval(parser.get("VEH_PARS", "veh_pars"))


def build_vehicle_pars(
    base_config: dict,
    c_w_a: float,
    c_z_a_total: float,
    mass: float,
    pow_max: float,
    p_harvest_straight: float,
    ers_harvest_speed_kmh: float,
) -> dict:
    pars = copy.deepcopy(base_config)

    lf = pars["general"]["lf"]
    lr = pars["general"]["lr"]
    wheelbase = lf + lr
    c_z_a_f = c_z_a_total * (lr / wheelbase)
    c_z_a_r = c_z_a_total * (lf / wheelbase)

    pars["general"]["c_w_a"] = c_w_a
    pars["general"]["c_z_a_f"] = c_z_a_f
    pars["general"]["c_z_a_r"] = c_z_a_r
    pars["general"]["m"] = mass
    pars["engine"]["pow_max"] = pow_max
    pars["engine"]["p_harvest_straight"] = p_harvest_straight
    pars["engine"]["ers_harvest_speed_kmh"] = ers_harvest_speed_kmh
    return pars


def run_sim_with_params(
    track: str,
    base_config: dict,
    c_w_a: float,
    c_z_a_total: float,
    mass: float,
    pow_max: float,
    p_harvest_straight: float,
    ers_harvest_speed_kmh: float,
    mu_weather: float = 1.0,
    em_strategy: str = "QUALY",
    driver_kwargs: dict = None,
    interp_stepsize: float = 1.0,
    curv_filt: float = 10.0,
):
    vehicle_pars = build_vehicle_pars(
        base_config, c_w_a, c_z_a_total, mass, pow_max,
        p_harvest_straight, ers_harvest_speed_kmh,
    )

    track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": mu_weather,
        "interp_stepsize_des": interp_stepsize,
        "curv_filt_width": curv_filt,
        "use_drs": True,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 1,
        "es_diff_max": 100.0,
        "vel_tol": 5e-2,
        "custom_vehicle_pars": vehicle_pars,
    }

    _dk = driver_kwargs or {}
    driver_opts = {
        "vel_subtr_corner": _dk.get("vel_subtr_corner", 0.5),
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": _dk.get("initial_energy", 4.0e6),
        "em_strategy": em_strategy,
        "use_recuperation": _dk.get("use_recuperation", True),
        "use_lift_coast": _dk.get("use_lift_coast", False),
        "lift_coast_dist": _dk.get("lift_coast_dist", 10.0),
    }

    try:
        result = run_simulation_advanced(track_opts, solver_opts, driver_opts)
        v_max = float(np.max(result.velocity))
        return result.sector_times, result.lap_time, v_max, result.distance, result.velocity
    except Exception:
        return None, None, None, None, None


def _log_line(params, lap_time, v_max, error, count, elapsed, use_trace):
    c_w_a, c_z_a, mass, pow_max, p_harv, spd = params
    err_label = (f"relerr={error * 100:.3f}%" if TRACE_METRIC == "time_rel" else f"RMSE={error:.2f}m/s") if use_trace else f"err={error:.2f}"
    return (
        f"[{count}] drag={c_w_a:.2f}, df={c_z_a:.2f}, m={mass:.0f}, "
        f"P={pow_max / 1e3:.0f}kW, Pharv={p_harv / 1e3:.0f}kW, "
        f"v_harv={spd:.0f}km/h → "
        f"lap={lap_time:.2f}s, v_max={v_max * 3.6:.1f}km/h, "
        f"{err_label} ({elapsed:.2f}s)"
    )


def _make_objective(
    track, base_config, target_sectors, target_v_max,
    bounds_list, log_container, count_ref, no_improve_ref, best_result,
    ref_distance, ref_velocity, mu_weather, em_strategy, driver_kwargs,
    interp_stepsize, curv_filt, early_stop=True,
):
    use_trace = ref_distance is not None and ref_velocity is not None
    v_max_weight = 0.13
    lower = np.array([b[0] for b in bounds_list])
    upper = np.array([b[1] for b in bounds_list])

    def objective(params_raw):
        check_abort()
        count_ref[0] += 1

        params = np.clip(params_raw, lower, upper)
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        elapsed = time.time() - start

        if sectors is None:
            log_container.write(f"[{count_ref[0]}] FAILED")
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity,
                                        metric=TRACE_METRIC)
        else:
            sector_error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            v_max_error = v_max_weight * (v_max - target_v_max) ** 2
            error = sector_error + v_max_error

        log_container.write(
            _log_line(list(params), lap_time, v_max, error, count_ref[0], elapsed, use_trace)
        )

        if error < best_result[4] - (1e-5 if (use_trace and TRACE_METRIC != "rmse") else 0.001):
            best_result[0] = list(params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel
            no_improve_ref[0] = 0
        else:
            no_improve_ref[0] += 1
            if early_stop and no_improve_ref[0] >= 10:
                raise EarlyStopException()

        return error

    return objective


def run_nelder_mead(
    track, base_config, target_sectors, target_v_max, initial_guess, bounds,
    log_container, ref_distance=None, ref_velocity=None, mu_weather=1.0,
    em_strategy="QUALY", driver_kwargs=None, interp_stepsize=1.0, curv_filt=10.0,
):
    count_ref = [0]
    no_improve_ref = [0]
    best_result = [None, None, None, None, float("inf"), None, None]

    obj = _make_objective(
        track, base_config, target_sectors, target_v_max,
        bounds, log_container, count_ref, no_improve_ref, best_result,
        ref_distance, ref_velocity, mu_weather, em_strategy, driver_kwargs,
        interp_stepsize, curv_filt, early_stop=True,
    )

    try:
        result = minimize(
            obj, x0=initial_guess, method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 0.01, "fatol": 0.15},
        )
        params = np.clip(result.x, [b[0] for b in bounds], [b[1] for b in bounds])
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        return list(params), sectors, lap_time, v_max, result.fun, sim_dist, sim_vel
    except EarlyStopException:
        log_container.write("Early stopping: no improvement for 10 consecutive evaluations.")
        return tuple(best_result)
    except AbortException:
        raise


def run_trust_constr(
    track, base_config, target_sectors, target_v_max, initial_guess, bounds,
    log_container, ref_distance=None, ref_velocity=None, mu_weather=1.0,
    em_strategy="QUALY", driver_kwargs=None, interp_stepsize=1.0, curv_filt=10.0,
):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    scale = upper - lower

    count_ref = [0]
    no_improve_ref = [0]
    best_result = [None, None, None, None, float("inf"), None, None]

    def to_real(x): return lower + np.array(x) * scale
    def to_norm(p): return (np.array(p) - lower) / scale

    scipy_bounds = Bounds([0] * 6, [1] * 6)
    x0 = to_norm(initial_guess)

    def objective(x_norm):
        check_abort()
        count_ref[0] += 1
        params = to_real(x_norm)
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        elapsed = time.time() - start

        use_trace = ref_distance is not None
        if sectors is None:
            log_container.write(f"[{count_ref[0]}] FAILED")
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity,
                                        metric=TRACE_METRIC)
        else:
            error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            error += 0.13 * (v_max - target_v_max) ** 2

        log_container.write(
            _log_line(list(params), lap_time, v_max, error, count_ref[0], elapsed, use_trace)
        )

        if error < best_result[4] - (1e-5 if (use_trace and TRACE_METRIC != "rmse") else 0.001):
            best_result[0] = list(params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel

        return error

    cb_last_best = [float("inf")]

    def callback_tc(xk, state):
        if best_result[4] < cb_last_best[0] - 0.001:
            cb_last_best[0] = best_result[4]
            no_improve_ref[0] = 0
        else:
            no_improve_ref[0] += 1
        if no_improve_ref[0] >= 10:
            log_container.write("Early stopping: no improvement for 10 consecutive iterations.")
            return True
        return False

    try:
        result = minimize(
            objective, x0=x0, method="trust-constr",
            bounds=scipy_bounds, callback=callback_tc,
            options={"maxiter": 500, "gtol": 1e-5, "xtol": 1e-5},
        )
        params = to_real(result.x)
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        return list(params), sectors, lap_time, v_max, result.fun, sim_dist, sim_vel
    except EarlyStopException:
        log_container.write("Early stopping: no improvement for 10 consecutive evaluations.")
        return tuple(best_result)
    except AbortException:
        raise


def run_lbfgsb(
    track, base_config, target_sectors, target_v_max, initial_guess, bounds,
    log_container, ref_distance=None, ref_velocity=None, mu_weather=1.0,
    em_strategy="QUALY", driver_kwargs=None, interp_stepsize=1.0, curv_filt=10.0,
):
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    scale = upper - lower

    count_ref = [0]
    no_improve_ref = [0]
    best_result = [None, None, None, None, float("inf"), None, None]

    def to_real(x): return lower + np.array(x) * scale
    def to_norm(p): return (np.array(p) - lower) / scale

    scipy_bounds_lbfgsb = [(0, 1)] * 6
    x0 = to_norm(initial_guess)

    def objective(x_norm):
        check_abort()
        count_ref[0] += 1
        params = to_real(x_norm)
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params

        start = time.time()
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        elapsed = time.time() - start

        use_trace = ref_distance is not None
        if sectors is None:
            log_container.write(f"[{count_ref[0]}] FAILED")
            return 1e6

        if use_trace:
            error = compute_trace_error(sim_dist, sim_vel, ref_distance, ref_velocity,
                                        metric=TRACE_METRIC)
        else:
            error = sum((s - t) ** 2 for s, t in zip(sectors, target_sectors))
            error += 0.13 * (v_max - target_v_max) ** 2

        log_container.write(
            _log_line(list(params), lap_time, v_max, error, count_ref[0], elapsed, use_trace)
        )

        if error < best_result[4] - (1e-5 if (use_trace and TRACE_METRIC != "rmse") else 0.001):
            best_result[0] = list(params)
            best_result[1] = sectors
            best_result[2] = lap_time
            best_result[3] = v_max
            best_result[4] = error
            best_result[5] = sim_dist
            best_result[6] = sim_vel

        return error

    cb_last_best = [float("inf")]

    def callback_lbfgsb(xk):
        if best_result[4] < cb_last_best[0] - 0.001:
            cb_last_best[0] = best_result[4]
            no_improve_ref[0] = 0
        else:
            no_improve_ref[0] += 1
        if no_improve_ref[0] >= 10:
            log_container.write("Early stopping: no improvement for 10 consecutive iterations.")
            raise EarlyStopException()

    try:
        result = minimize(
            objective, x0=x0, method="L-BFGS-B",
            bounds=scipy_bounds_lbfgsb, callback=callback_lbfgsb,
            options={"maxiter": 1000, "ftol": 1e-5, "gtol": 1e-5},
        )
        params = to_real(result.x)
        c_w_a, c_z_a, mass, pow_max, p_harv, spd = params
        sectors, lap_time, v_max, sim_dist, sim_vel = run_sim_with_params(
            track, base_config, c_w_a, c_z_a, mass, pow_max, p_harv, spd,
            mu_weather=mu_weather, em_strategy=em_strategy,
            driver_kwargs=driver_kwargs,
            interp_stepsize=interp_stepsize, curv_filt=curv_filt,
        )
        return list(params), sectors, lap_time, v_max, result.fun, sim_dist, sim_vel
    except EarlyStopException:
        log_container.write("Early stopping: no improvement for 10 consecutive evaluations.")
        return tuple(best_result)
    except AbortException:
        raise


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

available_tracks = get_available_tracks()
track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Spa") if "Spa" in available_tracks else 0,
)

available_vehicles = get_available_vehicles()
vehicle_base = st.sidebar.selectbox(
    "Base Vehicle",
    options=available_vehicles,
    index=available_vehicles.index("F1_2026") if "F1_2026" in available_vehicles else 0,
)

st.sidebar.header("Simulation Settings")

with st.sidebar.expander("Track Processing"):
    interp_stepsize = st.slider(
        "Interpolation Step Size [m]", min_value=1.0, max_value=20.0, value=SEARCH_STEPSIZE_DEFAULT, step=0.5
    )
    curv_filt_width = st.slider(
        "Curvature Filter Width [m]", min_value=0.0, max_value=30.0, value=10.0, step=1.0,
        help="Set to 0 to disable filtering",
    )
    curv_filt = curv_filt_width if curv_filt_width > 0 else None

st.sidebar.header("Target Source")
target_source = st.sidebar.radio(
    "Source", options=["FastF1 Telemetry", "Manual"], horizontal=True,
    help="Manual: enter sector times + v_max. FastF1: download real telemetry speed traces.",
)

use_trace_mode = False
ref_distance = None
ref_velocity = None

if target_source == "FastF1 Telemetry":
    available_gps = get_available_gps(available_tracks)
    fastf1_tracks = list(available_gps.keys())

    if not fastf1_tracks:
        st.sidebar.warning("No tracks with FastF1 mapping available.")
    else:
        ff1_track = st.sidebar.selectbox(
            "GP Track", options=fastf1_tracks,
            index=fastf1_tracks.index(track) if track in fastf1_tracks else 0,
        )
        track = ff1_track
        ff1_year = st.sidebar.selectbox("Year", options=get_available_years(), index=8)
        ff1_session = st.sidebar.radio("Session", options=["Q", "R"], horizontal=True,
                                        help="Q = Qualifying, R = Race")
        ff1_driver = st.sidebar.text_input("Driver (optional)", value="",
                                            help="3-letter abbreviation. Leave empty for fastest lap.")
        ff1_driver = ff1_driver.strip().upper() or None

        download_button = st.sidebar.button("Download Telemetry", type="secondary", width="stretch")

        if download_button:
            gp_name = available_gps[ff1_track]
            with st.spinner(f"Downloading {ff1_year} {gp_name} {ff1_session} telemetry..."):
                try:
                    ff1_data = load_speed_trace(ff1_year, gp_name, ff1_session, ff1_driver)
                    dist = ff1_data["distance"]
                    vel = ff1_data["speed"]
                    lap_time_ff1 = ff1_data["lap_time"]
                    sectors_ff1 = ff1_data["sector_times"]
                    st.session_state.ers_fastf1_trace = {
                        "distance": dist, "velocity": vel,
                        "lap_time": lap_time_ff1, "sector_times": sectors_ff1,
                        "v_max": float(np.max(vel)),
                        "year": ff1_year, "gp": gp_name,
                        "session": ff1_session, "driver": ff1_driver, "track": ff1_track,
                        "throttle": ff1_data["throttle"], "brake": ff1_data["brake"],
                        "gear": ff1_data["gear"], "rpm": ff1_data["rpm"],
                        "drs": ff1_data["drs"], "drs_active": ff1_data["drs_active"],
                    }
                    st.sidebar.success(
                        f"Lap: {int(lap_time_ff1 // 60)}:{lap_time_ff1 % 60:06.3f} | "
                        f"V_max: {float(np.max(vel)) * 3.6:.1f} km/h"
                    )
                except Exception as e:
                    st.sidebar.error(f"Download failed: {e}")
                    st.session_state.ers_fastf1_trace = None

        if st.session_state.ers_fastf1_trace is not None:
            trace = st.session_state.ers_fastf1_trace
            ref_distance = trace["distance"]
            ref_velocity = trace["velocity"]
            use_trace_mode = True
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
        target_s1 = st.number_input("S1 [s]", min_value=10.0, max_value=120.0, value=24.0,
                                     step=0.1, format="%.3f")
    with col2:
        target_s2 = st.number_input("S2 [s]", min_value=10.0, max_value=120.0, value=26.5,
                                     step=0.1, format="%.3f")
    with col3:
        target_s3 = st.number_input("S3 [s]", min_value=10.0, max_value=120.0, value=40.4,
                                     step=0.1, format="%.3f")
    target_total = target_s1 + target_s2 + target_s3
    st.sidebar.caption(f"Total: **{int(target_total // 60)}:{target_total % 60:06.3f}**")

    st.sidebar.header("Target Max Velocity")
    target_v_max = st.sidebar.number_input(
        "V_max [km/h]", min_value=200.0, max_value=400.0, value=320.0, step=1.0,
    )
    target_v_max_ms = target_v_max / 3.6


# Objective metric for FastF1 trace mode (ignored when no trace is loaded, in which case
# the fallback sector-time + v_max cost is used instead).
_metric_label = st.sidebar.radio(
    "Trace Objective",
    options=["RMSE", "Time-weighted relative"],
    index=0,
    horizontal=True,
    help=(
        "RMSE: absolute speed error in m/s. Because the trace is sampled along distance, "
        "fast sections contribute both larger errors and more samples, so RMSE is strongly "
        "straight-line weighted.\n\n"
        "Time-weighted relative: residuals taken relative to the reference speed and weighted "
        "by the time spent at each sample, so slow corners carry their fair share. "
        "Dimensionless -- 0.02 means a typical 2% speed error."
    ),
)
TRACE_METRIC = "rmse" if _metric_label == "RMSE" else "time_rel"

st.sidebar.header("Search Method")
search_method = st.sidebar.selectbox(
    "Method",
    options=["Trust-Constr", "L-BFGS-B", "Nelder-Mead"],
    help="Gradient-based methods (Trust-Constr, L-BFGS-B) work best with 6 parameters. "
         "Grid search is omitted as it is impractical for 6-dimensional spaces.",
)

st.sidebar.header("Parameter Bounds")

mu_weather = st.sidebar.slider(
    "Tire Grip (mu)", min_value=0.6, max_value=1.4, value=1.0, step=0.05,
    help="Track grip multiplier: 1.0 = dry, 0.8 = damp, 0.6 = wet",
)

em_strategy = st.sidebar.selectbox(
    "Energy Strategy",
    options=EM_STRATEGIES,
    index=EM_STRATEGIES.index(DEFAULT_EM_STRATEGY),
    help="QUALY is recommended: ERS harvest parameters are only active with QUALY.",
)

default_energy_j = 4.0e6
initial_energy_mj = st.sidebar.slider(
    "Initial Energy [MJ]", min_value=0.0, max_value=6.0,
    value=default_energy_j / 1e6, step=0.1,
)

with st.sidebar.expander("Driver Behavior"):
    vel_subtr_corner = st.slider(
        "Corner Safety Margin [m/s]", min_value=0.0, max_value=3.0, value=0.5, step=0.1
    )
    use_recuperation = st.checkbox("Use Energy Recuperation", value=True)

with st.sidebar.expander("Aerodynamics Bounds"):
    c_w_a_min = st.number_input("Drag min [m²]", value=0.60, step=0.2)
    c_w_a_max = st.number_input("Drag max [m²]", value=1.80, step=0.2)
    c_z_a_total_min = st.number_input("Total Downforce min [m²]", value=3.6, step=0.2)
    c_z_a_total_max = st.number_input("Total Downforce max [m²]", value=5.4, step=0.2)

with st.sidebar.expander("Mass Bounds"):
    mass_min = st.number_input("Mass min [kg]", value=820.0, step=5.0)
    mass_max = st.number_input("Mass max [kg]", value=880.0, step=5.0)

with st.sidebar.expander("Power Bounds"):
    pow_max_min = st.number_input("Power min [kW]", value=360.0, step=10.0) * 1e3
    pow_max_max = st.number_input("Power max [kW]", value=600.0, step=10.0) * 1e3

with st.sidebar.expander("ERS Harvest Bounds"):
    p_harv_min = st.number_input(
        "Harvest power min [kW]", value=50.0, step=10.0,
        help="p_harvest_straight: straight-line MGU-K harvest power",
    ) * 1e3
    p_harv_max = st.number_input(
        "Harvest power max [kW]", value=350.0, step=10.0,
    ) * 1e3
    ers_speed_min = st.number_input(
        "Harvest speed min [km/h]", value=150.0, step=10.0,
        help="ers_harvest_speed_kmh: minimum speed to start harvesting on straights",
    )
    ers_speed_max = st.number_input(
        "Harvest speed max [km/h]", value=320.0, step=10.0,
    )

st.sidebar.divider()

col_run, col_abort = st.sidebar.columns(2)
with col_run:
    run_button = st.button("⚡ Find Parameters", type="primary", width="stretch")
with col_abort:
    abort_button = st.button("⏹ Abort", type="secondary", width="stretch")

if abort_button:
    st.session_state.ers_param_id_abort = True
    st.warning("Optimization aborted")
    st.rerun()

# ---------------------------------------------------------------------------
# Optimization run
# ---------------------------------------------------------------------------
if run_button:
    st.session_state.ers_param_id_abort = False

    target_sectors = [target_s1, target_s2, target_s3]
    bounds = [
        (c_w_a_min, c_w_a_max),
        (c_z_a_total_min, c_z_a_total_max),
        (mass_min, mass_max),
        (pow_max_min, pow_max_max),
        (p_harv_min, p_harv_max),
        (ers_speed_min, ers_speed_max),
    ]
    base_config = load_vehicle_config(vehicle_base)

    # Initial guess from base vehicle config, clipped to bounds
    base_p_harv = base_config.get("engine", {}).get("p_harvest_straight", 120e3)
    base_ers_spd = base_config.get("engine", {}).get("ers_harvest_speed_kmh", 250.0)
    initial_guess = [
        np.clip(base_config["general"]["c_w_a"], c_w_a_min, c_w_a_max),
        np.clip(
            base_config["general"]["c_z_a_f"] + base_config["general"]["c_z_a_r"],
            c_z_a_total_min, c_z_a_total_max,
        ),
        np.clip(base_config["general"]["m"], mass_min, mass_max),
        np.clip(base_config["engine"]["pow_max"], pow_max_min, pow_max_max),
        np.clip(base_p_harv, p_harv_min, p_harv_max),
        np.clip(base_ers_spd, ers_speed_min, ers_speed_max),
    ]

    driver_kwargs = {
        "vel_subtr_corner": vel_subtr_corner,
        "initial_energy": initial_energy_mj * 1e6,
        "use_recuperation": use_recuperation,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }
    opt_kwargs = dict(
        ref_distance=ref_distance if use_trace_mode else None,
        ref_velocity=ref_velocity if use_trace_mode else None,
        mu_weather=mu_weather,
        em_strategy=em_strategy,
        driver_kwargs=driver_kwargs,
        interp_stepsize=interp_stepsize,
        curv_filt=curv_filt,
    )

    aborted = False
    best_params = None
    best_sectors = None
    best_lap = None
    best_v_max = None
    best_error = None
    best_sim_distance = None
    best_sim_velocity = None

    runner_map = {
        "Nelder-Mead": run_nelder_mead,
        "Trust-Constr": run_trust_constr,
        "L-BFGS-B": run_lbfgsb,
    }
    method_label = {
        "Nelder-Mead": "Nelder-Mead",
        "Trust-Constr": "Trust-Region Constrained",
        "L-BFGS-B": "L-BFGS-B",
    }

    runner = runner_map[search_method]
    with st.status(f"Running {method_label[search_method]} optimization...", expanded=True) as status:
        ig = initial_guess
        st.write(
            f"Starting from: drag={ig[0]:.2f}, df={ig[1]:.2f}, m={ig[2]:.0f}, "
            f"P={ig[3] / 1e3:.0f}kW, Pharv={ig[4] / 1e3:.0f}kW, v_harv={ig[5]:.0f}km/h"
        )
        if use_trace_mode:
            st.write("Objective: speed trace RMSE (m/s)")
        st.write("---")
        log_container = st.container()

        try:
            (
                best_params, best_sectors, best_lap, best_v_max,
                best_error, best_sim_distance, best_sim_velocity,
            ) = runner(
                track, base_config, target_sectors, target_v_max_ms,
                initial_guess, bounds, log_container, **opt_kwargs,
            )

            if best_params is None:
                st.error("Optimization failed!")
                status.update(label="Failed", state="error")
                st.stop()

            st.write("---")
            bp = best_params
            st.write(
                f"Optimum: drag={bp[0]:.2f}, df={bp[1]:.2f}, m={bp[2]:.0f}, "
                f"P={bp[3] / 1e3:.0f}kW, Pharv={bp[4] / 1e3:.0f}kW, "
                f"v_harv={bp[5]:.0f}km/h, v_max={best_v_max * 3.6:.1f}km/h"
            )
            status.update(label="Optimization complete!", state="complete", expanded=False)

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

    st.session_state.ers_param_id_abort = False

    if best_params is None:
        st.stop()

    c_w_a_opt, c_z_a_total_opt, mass_opt, pow_max_opt, p_harv_opt, spd_opt = best_params

    lf = base_config["general"]["lf"]
    lr = base_config["general"]["lr"]
    wheelbase = lf + lr
    c_z_a_f_opt = c_z_a_total_opt * (lr / wheelbase)
    c_z_a_r_opt = c_z_a_total_opt * (lf / wheelbase)

    # Final full-resolution simulation
    vehicle_pars = build_vehicle_pars(
        base_config, c_w_a_opt, c_z_a_total_opt, mass_opt, pow_max_opt,
        p_harv_opt, spd_opt,
    )
    final_track_opts = {
        "trackname": track, "flip_track": False, "mu_weather": mu_weather,
        "interp_stepsize_des": min(interp_stepsize, FINAL_PLOT_STEPSIZE), "curv_filt_width": curv_filt,
        "use_drs": True, "use_pit": False,
    }
    final_solver_opts = {
        "vehicle": None, "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6, "find_v_start": True,
        "max_no_em_iters": 5, "es_diff_max": 1.0, "vel_tol": 1e-5,
        "custom_vehicle_pars": vehicle_pars,
    }
    final_driver_opts = {
        "vel_subtr_corner": driver_kwargs.get("vel_subtr_corner", 0.5),
        "vel_lim_glob": None,
        "yellow_s1": False, "yellow_s2": False, "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": driver_kwargs.get("initial_energy", 4.0e6),
        "em_strategy": em_strategy,
        "use_recuperation": driver_kwargs.get("use_recuperation", True),
        "use_lift_coast": False, "lift_coast_dist": 10.0,
    }
    try:
        final_sim_result = run_simulation_advanced(final_track_opts, final_solver_opts, final_driver_opts)
    except Exception:
        final_sim_result = None

    if final_sim_result is not None:
        _display_sectors = list(final_sim_result.sector_times)
        _display_lap = final_sim_result.lap_time
        _display_v_max = float(np.max(final_sim_result.velocity))
    else:
        _display_sectors = best_sectors
        _display_lap = best_lap
        _display_v_max = best_v_max

    st.session_state.ers_param_id_result = {
        "success": True,
        "c_w_a": c_w_a_opt, "c_z_a_total": c_z_a_total_opt,
        "c_z_a_f": c_z_a_f_opt, "c_z_a_r": c_z_a_r_opt,
        "mass": mass_opt, "pow_max": pow_max_opt,
        "p_harvest_straight": p_harv_opt,
        "ers_harvest_speed_kmh": spd_opt,
        "target_sectors": target_sectors,
        "simulated_sectors": _display_sectors,
        "simulated_lap": _display_lap,
        "target_v_max": target_v_max_ms,
        "simulated_v_max": _display_v_max,
        "track": track, "vehicle": vehicle_base,
        "use_trace": use_trace_mode,
        "sim_distance": best_sim_distance, "sim_velocity": best_sim_velocity,
        "ref_distance": ref_distance, "ref_velocity": ref_velocity,
        "sim_result": final_sim_result,
        "fastf1_trace": st.session_state.ers_fastf1_trace if use_trace_mode else None,
    }

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.ers_param_id_result is not None:
    res = st.session_state.ers_param_id_result

    st.divider()
    st.header("Identified Parameters")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Drag (c_w × A)", f"{res['c_w_a']:.2f} m²")
    with col2:
        st.metric("Total Downforce", f"{res['c_z_a_total']:.2f} m²")
    with col3:
        st.metric("Cl/Cd", f"{res['c_z_a_total'] / res['c_w_a']:.2f}")
    with col4:
        st.metric("Mass", f"{res['mass']:.0f} kg")
    with col5:
        st.metric("Power", f"{res['pow_max'] / 1e3:.0f} kW")
    with col6:
        st.metric("Harvest Power", f"{res['p_harvest_straight'] / 1e3:.0f} kW")
    with col7:
        st.metric("Harvest Speed", f"{res['ers_harvest_speed_kmh']:.0f} km/h")

    st.caption(
        f"Downforce split: Front {res['c_z_a_f']:.3f} m² / Rear {res['c_z_a_r']:.3f} m² | "
        f"Add to vehicle INI: `\"p_harvest_straight\": {res['p_harvest_straight']:.0f}`, "
        f"`\"ers_harvest_speed_kmh\": {res['ers_harvest_speed_kmh']:.1f}`"
    )

    st.divider()
    st.header("Sector Time Comparison")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Sector**")
        for s in ["S1", "S2", "S3", "**Total**"]:
            st.write(s)
    with col2:
        st.markdown("**Target**")
        for t in res["target_sectors"]:
            st.write(f"{t:.3f}s")
        st.write(f"**{sum(res['target_sectors']):.3f}s**")
    with col3:
        st.markdown("**Simulated**")
        for s in res["simulated_sectors"]:
            st.write(f"{s:.3f}s")
        st.write(f"**{res['simulated_lap']:.3f}s**")
    with col4:
        st.markdown("**Difference**")
        for i in range(3):
            diff = res["simulated_sectors"][i] - res["target_sectors"][i]
            color = "green" if abs(diff) < 0.5 else "orange" if abs(diff) < 1.0 else "red"
            st.markdown(f":{color}[{diff:+.3f}s]")
        total_diff = res["simulated_lap"] - sum(res["target_sectors"])
        color = "green" if abs(total_diff) < 0.5 else "orange" if abs(total_diff) < 1.0 else "red"
        st.markdown(f"**:{color}[{total_diff:+.3f}s]**")

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

    # Speed trace overlay
    _has_sim = res.get("sim_result") is not None or res.get("sim_distance") is not None
    if res.get("use_trace") and _has_sim and res.get("ref_distance") is not None:
        st.divider()
        st.header("Speed Trace Comparison")

        if res.get("sim_result") is not None:
            sim_dist = res["sim_result"].distance
            sim_vel = res["sim_result"].velocity
        else:
            sim_dist = res["sim_distance"]
            sim_vel = res["sim_velocity"]
        r_dist = res["ref_distance"]
        r_vel = res["ref_velocity"]

        sim_dist_norm = sim_dist / sim_dist[-1]
        ref_dist_norm = r_dist / r_dist[-1]
        x_km = ref_dist_norm * r_dist[-1] / 1000

        sim_vel_interp = np.interp(ref_dist_norm, sim_dist_norm, sim_vel)
        delta_vel = sim_vel_interp - r_vel
        rmse = compute_trace_error(sim_dist, sim_vel, r_dist, r_vel, metric="rmse")
        trace_r2 = compute_trace_r2(sim_dist, sim_vel, r_dist, r_vel)

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(x=x_km, y=r_vel * 3.6, name="FastF1 Reference",
                                  line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_km, y=sim_vel_interp * 3.6, name="Simulation",
                                  line=dict(color="#d62728")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_km, y=delta_vel * 3.6, name="Delta",
                                  line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1)
        fig.update_yaxes(title_text="Delta [km/h]", row=2, col=1)
        fig.update_xaxes(title_text="Distance [km]", row=2, col=1)
        fig.update_layout(
            title=f"Speed Trace Overlay (R²: {trace_r2:.4f}, RMSE: {rmse:.2f} m/s)",
            height=500, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch")

    if res.get("sim_result") is not None:
        st.divider()
        st.header("Simulation Data")
        render_simulation_plots(res["sim_result"], key_prefix="ers_paramid_")

    st.divider()
    st.caption(f"Track: {res['track']} | Base vehicle: {res['vehicle']}")

else:
    st.info(
        "Select a track, vehicle, download FastF1 telemetry (or enter target sector times), "
        "then click **Find Parameters**."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### How it works

        Extends the standard Parameter Identification with two ERS-specific parameters:

        **Parameters identified (6):**
        - Drag coefficient × area (c_w × A)
        - Total downforce coefficient × area (c_z × A)
        - Vehicle mass
        - Engine power (pow_max)
        - **Straight-line harvest power (p_harvest_straight)**
        - **Harvest activation speed (ers_harvest_speed_kmh)**

        **Target modes:**
        - **FastF1 Telemetry:** Full speed trace RMSE — captures the velocity plateau
          and pre-braking deceleration caused by ERS harvest
        - **Manual:** 3 sector times + max velocity
        """)
    with col2:
        st.markdown("""
        ### Tips

        - Use **QUALY** strategy: ERS harvest parameters only activate with QUALY
        - **FastF1 trace mode** is ideal — the harvest plateau shape is captured in
          the velocity profile and drives the optimizer toward realistic ERS behaviour
        - Set initial energy to a full battery (4 MJ) for qualifying simulations
        - Grid search is omitted for this page (6 parameters → combinatorial explosion);
          Trust-Constr or L-BFGS-B are recommended
        - After identification, copy `p_harvest_straight` and `ers_harvest_speed_kmh`
          values into your vehicle `.ini` file under the `engine` section
        """)
