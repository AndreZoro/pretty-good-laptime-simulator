"""
MVRC 2026 Simulation Page

Simulation page for the MVRC 2026 vehicle with limited parameter adjustments.
"""

import numpy as np
import streamlit as st

from helpers.simulation import get_available_tracks, run_simulation_advanced
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

# Default values from MVRC_2026.ini
DEFAULTS = {
    "c_w_a": 1.56,
    "c_z_a_f": 2.20,
    "c_z_a_r": 2.68,
    "pow_max": 575.0,  # kW
}

# Sidebar controls
st.sidebar.header("Track Selection")

available_tracks = get_available_tracks()
track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Shanghai") if "Shanghai" in available_tracks else 0,
)

st.sidebar.header("Vehicle Parameters")

# Aerodynamics
c_w_a = st.sidebar.number_input(
    "Drag (c_w × A) [m²]",
    min_value=0.5,
    max_value=3.0,
    value=DEFAULTS["c_w_a"],
    step=0.05,
    format="%.2f",
    help="Drag coefficient times frontal area",
)

c_z_a_f = st.sidebar.number_input(
    "Front Downforce (c_z_f × A) [m²]",
    min_value=0.5,
    max_value=5.0,
    value=DEFAULTS["c_z_a_f"],
    step=0.05,
    format="%.2f",
    help="Front downforce coefficient times reference area",
)

c_z_a_r = st.sidebar.number_input(
    "Rear Downforce (c_z_r × A) [m²]",
    min_value=0.5,
    max_value=5.0,
    value=DEFAULTS["c_z_a_r"],
    step=0.05,
    format="%.2f",
    help="Rear downforce coefficient times reference area",
)

# Power
pow_max = st.sidebar.number_input(
    "Max Power [kW]",
    min_value=300.0,
    max_value=800.0,
    value=DEFAULTS["pow_max"],
    step=5.0,
    format="%.0f",
    help="Maximum engine power",
)

st.sidebar.divider()

# Run button
run_button = st.sidebar.button(
    "🚀 Run Simulation", type="primary", use_container_width=True
)

# Build custom vehicle parameters based on MVRC_2026 with user modifications
custom_vehicle_pars = {
    "powertrain_type": "hybrid",
    "general": {
        "lf": 1.557,
        "lr": 1.842,
        "h_cog": 0.300,
        "sf": 1.618,
        "sr": 1.536,
        "m": 872.0,
        "f_roll": 0.03,
        "c_w_a": c_w_a,
        "c_z_a_f": c_z_a_f,
        "c_z_a_r": c_z_a_r,
        "g": 9.81,
        "rho_air": 1.18,
        "drs_factor": 0.17,
    },
    "engine": {
        "topology": "RWD",
        "pow_max": pow_max * 1e3,  # Convert kW to W
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
}

# Main area
if run_button:
    track_opts = {
        "trackname": track,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 5.0,
        "curv_filt_width": 10.0,
        "use_drs1": True,
        "use_drs2": True,
        "use_pit": False,
    }

    solver_opts = {
        "vehicle": None,
        "series": "F1",
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
        "initial_energy": 4.0e6,
        "em_strategy": "FCFB",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }

    with st.spinner(f"Simulating MVRC 2026 at {track}..."):
        try:
            result = run_simulation_advanced(track_opts, solver_opts, driver_opts)
            st.session_state.mvrc_result = result
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
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Max Speed", f"{np.max(result.velocity_kmh):.1f} km/h")
    with col2:
        st.metric("Avg Speed", f"{np.mean(result.velocity_kmh):.1f} km/h")
    with col3:
        st.metric("Energy Used", f"{result.energy_consumed:.1f} kJ")

    # Render profile chart and track map
    render_simulation_plots(result, key_prefix="mvrc_")

else:
    # Initial state - show instructions
    st.info(
        "👈 Select a track and adjust vehicle parameters, then click **Run Simulation**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Vehicle Parameters

        - **Drag (c_w × A)** - Aerodynamic drag coefficient times frontal area
        - **Front Downforce** - Front wing downforce coefficient times area
        - **Rear Downforce** - Rear wing downforce coefficient times area
        - **Max Power** - Maximum engine power output
        """)

    with col2:
        st.markdown("""
        ### MVRC 2026 Specs

        - **Mass**: 872 kg
        - **Powertrain**: Hybrid (ICE + E-Motor)
        - **E-Motor Power**: 120 kW
        - **Gears**: 8-speed
        """)
