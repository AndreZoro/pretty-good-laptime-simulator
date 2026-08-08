"""
MVRC 2026 Simulation Page

Simulation page for the MVRC 2026 vehicle with limited parameter adjustments.
"""

import copy

import numpy as np
import streamlit as st

from helpers.simulation import (
    get_available_tracks,
    read_vehicle_params,
    run_simulation_advanced,
)
from helpers.visualization import render_simulation_plots

st.set_page_config(
    page_title="MVRC 2026 - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.image(
    "https://mantiumchallenge.com/wp-content/uploads/2016/08/mantiumchallenge.png",
    # width=300,
    width="stretch",
    # caption="## MVRC 2026",
)
st.title("MVRC 2026")
st.caption("Laptime Simulator for the MVRC 2026 Season")

# Initialize session state
if "mvrc_result" not in st.session_state:
    st.session_state.mvrc_result = None
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

MAX_RUNS = 3

# Base vehicle configuration read from MVRC_2026.ini
BASE_VEH_PARS = read_vehicle_params("MVRC_2026")

# Cooling flow -> engine power model (linear interpolation)
COOLING_FLOW_MIN = 1.15  # [m^3/s]
COOLING_FLOW_MAX = 2.30  # [m^3/s]
POW_MAX_AT_MIN_FLOW = 205.0  # [kW]
POW_MAX_AT_MAX_FLOW = 410.0  # [kW]


def cooling_flow_to_power(flow: float) -> float:
    """Return the maximum engine power [kW] for a given cooling flow [m^3/s]."""
    frac = (flow - COOLING_FLOW_MIN) / (COOLING_FLOW_MAX - COOLING_FLOW_MIN)
    return POW_MAX_AT_MIN_FLOW + frac * (POW_MAX_AT_MAX_FLOW - POW_MAX_AT_MIN_FLOW)


# Sidebar controls
st.sidebar.header("Track Selection")

# Restrict the selection to the 2026 calendar racelines. Other "_2026" files in the
# raceline folder are intermediate FastF1 extractions that duplicate these.
TRACK_SUFFIX_2026 = "GrandPrix_2026"
DEFAULT_TRACK = "BarcelonaGrandPrix_2026"

available_tracks = sorted(
    t for t in get_available_tracks() if t.endswith(TRACK_SUFFIX_2026)
)
if not available_tracks:  # fall back rather than render an empty dropdown
    available_tracks = get_available_tracks()

track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index(DEFAULT_TRACK)
    if DEFAULT_TRACK in available_tracks
    else 0,
)

st.sidebar.header("Vehicle Parameters")

# Aerodynamics
c_w_a = st.sidebar.number_input(
    "Drag (c_w × A) [m²]",
    min_value=0.5,
    max_value=3.0,
    value=BASE_VEH_PARS["general"]["c_w_a"],
    step=0.05,
    format="%.2f",
    help="Drag coefficient times frontal area",
)

c_z_a_f = st.sidebar.number_input(
    "Front Downforce (c_z_f × A) [m²]",
    min_value=0.5,
    max_value=5.0,
    value=BASE_VEH_PARS["general"]["c_z_a_f"],
    step=0.05,
    format="%.2f",
    help="Front downforce coefficient times reference area",
)

c_z_a_r = st.sidebar.number_input(
    "Rear Downforce (c_z_r × A) [m²]",
    min_value=0.5,
    max_value=5.0,
    value=BASE_VEH_PARS["general"]["c_z_a_r"],
    step=0.05,
    format="%.2f",
    help="Rear downforce coefficient times reference area",
)

# Power via cooling flow
st.sidebar.header("Cooling")

cooling_flow = st.sidebar.slider(
    "Cooling Flow [m³/s]",
    min_value=COOLING_FLOW_MIN,
    max_value=COOLING_FLOW_MAX,
    value=COOLING_FLOW_MAX,
    step=0.05,
    format="%.2f",
    help=(
        f"Cooling air flow rate. Maximum engine power scales linearly from "
        f"{POW_MAX_AT_MIN_FLOW:.0f} kW at {COOLING_FLOW_MIN:.2f} m³/s to "
        f"{POW_MAX_AT_MAX_FLOW:.0f} kW at {COOLING_FLOW_MAX:.2f} m³/s."
    ),
)

pow_max = cooling_flow_to_power(cooling_flow)
st.sidebar.metric("Resulting Max Power", f"{pow_max:.0f} kW")

st.sidebar.divider()

# Run button
run_button = st.sidebar.button(
    "🚀 Run Simulation", type="primary", use_container_width=True
)

# Build vehicle parameters from MVRC_2026.ini with user modifications
custom_vehicle_pars = copy.deepcopy(BASE_VEH_PARS)
custom_vehicle_pars["general"]["c_w_a"] = c_w_a
custom_vehicle_pars["general"]["c_z_a_f"] = c_z_a_f
custom_vehicle_pars["general"]["c_z_a_r"] = c_z_a_r
custom_vehicle_pars["engine"]["pow_max"] = pow_max * 1e3  # Convert kW to W

# Scale pow_diff along with pow_max so the ICE power curve keeps its shape.
# pow_diff sets the curvature of the cubic power curve independently of pow_max,
# so lowering pow_max on its own drops the whole curve until the low-rev end
# clips to zero power (below ~215 kW the engine makes no power under ~8600 rpm
# and the car cannot pull out of 2nd gear).
_pow_scale = (
    custom_vehicle_pars["engine"]["pow_max"] / BASE_VEH_PARS["engine"]["pow_max"]
)
custom_vehicle_pars["engine"]["pow_diff"] = (
    BASE_VEH_PARS["engine"]["pow_diff"] * _pow_scale
)

# Main area
if run_button:
    track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 1.0,
        "curv_filt_width": 10.0,
        "use_drs": True,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-5,
        "custom_vehicle_pars": custom_vehicle_pars,
    }

    driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": BASE_VEH_PARS["engine"]["max_e_energy_storage"],
        # QUALY = qualifying strategy: spends the initial energy plus the energy
        # recovered during the lap where it gains the most time. Lift & coast is
        # an FCFB-only feature and must stay off here.
        "em_strategy": "QUALY",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }

    with st.spinner(f"Simulating MVRC 2026 at {track}..."):
        try:
            result = run_simulation_advanced(track_opts, solver_opts, driver_opts)
            st.session_state.mvrc_result = result
            st.session_state.mvrc_cooling_flow = cooling_flow
            st.session_state.mvrc_pow_max = pow_max
            st.success(f"Simulation completed for {track}")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)

# Display results if we have them
if st.session_state.mvrc_result is not None:
    result = st.session_state.mvrc_result

    # Save to Compare button
    col_save, col_spacer = st.columns([1, 3])
    with col_save:
        if len(st.session_state.saved_runs) >= MAX_RUNS:
            st.warning(f"Max {MAX_RUNS} runs saved. Clear some in Comparison page.")
        else:
            if st.button("💾 Save to Compare", use_container_width=True):
                st.session_state.saved_runs.append(result)
                st.success(f"Saved! ({len(st.session_state.saved_runs)}/{MAX_RUNS})")

    st.divider()

    # Lap time display
    st.header("Lap Time")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", result.format_lap_time())
    with col2:
        st.metric("Sector 1", f"{result.sector_times[0]:.3f}s")
    with col3:
        st.metric("Sector 2", f"{result.sector_times[1]:.3f}s")
    with col4:
        st.metric("Sector 3", f"{result.sector_times[2]:.3f}s")

    # Additional metrics
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Max Speed", f"{np.max(result.velocity_kmh):.1f} km/h")
    with col2:
        st.metric("Avg Speed", f"{np.mean(result.velocity_kmh):.1f} km/h")
    with col3:
        st.metric("Energy Used", f"{result.energy_consumed:.1f} kJ")
    with col4:
        st.metric(
            "Cooling Flow",
            f"{st.session_state.get('mvrc_cooling_flow', COOLING_FLOW_MAX):.2f} m³/s",
        )
    with col5:
        st.metric(
            "Max Power",
            f"{st.session_state.get('mvrc_pow_max', POW_MAX_AT_MAX_FLOW):.0f} kW",
        )

    # Render profile chart and track map
    render_simulation_plots(result, key_prefix="mvrc_")

else:
    # Initial state - show instructions
    st.info(
        "👈 Select a track and adjust vehicle parameters, then click **Run Simulation**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        ### Vehicle Parameters

        - **Drag (c_w × A)** - Aerodynamic drag coefficient times frontal area
        - **Front Downforce** - Front wing downforce coefficient times area
        - **Rear Downforce** - Rear wing downforce coefficient times area
        - **Cooling Flow** - Cooling air flow rate; sets the maximum engine power
          linearly from {POW_MAX_AT_MIN_FLOW:.0f} kW at {COOLING_FLOW_MIN:.2f} m³/s
          to {POW_MAX_AT_MAX_FLOW:.0f} kW at {COOLING_FLOW_MAX:.2f} m³/s
        """)

    with col2:
        st.markdown(f"""
        ### MVRC 2026 Specs

        - **Mass**: {BASE_VEH_PARS["general"]["m"]:.0f} kg
        - **Powertrain**: Hybrid (ICE + E-Motor)
        - **E-Motor Power**: {BASE_VEH_PARS["engine"]["pow_e_motor"] / 1e3:.0f} kW
        - **Gears**: {len(BASE_VEH_PARS["gearbox"]["i_trans"])}-speed
        """)
