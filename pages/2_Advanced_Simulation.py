"""
Advanced Simulation Page

Full control over all simulation parameters.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from helpers.simulation import (
    run_simulation_advanced,
    get_available_tracks,
    get_available_vehicles,
    SERIES_CONFIG,
)

st.set_page_config(
    page_title="Advanced Simulation - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.title("🔧 Advanced Simulation")
st.caption("Full control over all simulation parameters")

# Initialize session state for results
if "adv_result" not in st.session_state:
    st.session_state.adv_result = None
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

MAX_RUNS = 3

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# =============================================================================
# TRACK OPTIONS
# =============================================================================
st.sidebar.subheader("Track Options")

available_tracks = get_available_tracks()
track_name = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Shanghai") if "Shanghai" in available_tracks else 0,
    key="adv_track",
)

flip_track = st.sidebar.checkbox("Flip Track Direction", value=False)

mu_weather = st.sidebar.slider(
    "Grip Level (μ)",
    min_value=0.5,
    max_value=1.3,
    value=1.0,
    step=0.05,
    help="1.0 = Dry, 0.6 = Wet, >1.0 = High grip surface",
)

with st.sidebar.expander("Track Processing"):
    interp_stepsize = st.slider(
        "Interpolation Step Size [m]",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=1.0,
    )
    curv_filt_width = st.slider(
        "Curvature Filter Width [m]",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="Set to 0 to disable filtering",
    )
    curv_filt = curv_filt_width if curv_filt_width > 0 else None

# =============================================================================
# SOLVER OPTIONS
# =============================================================================
st.sidebar.subheader("Vehicle & Solver")

series = st.sidebar.radio(
    "Series",
    options=["F1", "FE"],
    horizontal=True,
)

available_vehicles = get_available_vehicles() + ["Custom"]
default_vehicle = SERIES_CONFIG[series]["vehicle"]
vehicle = st.sidebar.selectbox(
    "Vehicle Configuration",
    options=available_vehicles,
    index=available_vehicles.index(default_vehicle) if default_vehicle in available_vehicles else 0,
)

# Custom vehicle parameters
custom_vehicle_pars = None
if vehicle == "Custom":
    st.sidebar.info("Configure custom vehicle parameters below")

    # Default values based on series
    if series == "F1":
        defaults = {
            "powertrain_type": "hybrid",
            "m": 733.0, "lf": 1.968, "lr": 1.632, "h_cog": 0.335,
            "sf": 1.6, "sr": 1.6, "f_roll": 0.03,
            "c_w_a": 1.56, "c_z_a_f": 2.20, "c_z_a_r": 2.68,
            "drs_factor": 0.17,
            "pow_max": 575.0, "pow_diff": 41.0,
            "n_begin": 10500.0, "n_max": 11400.0, "n_end": 12200.0,
            "be_max": 100.0, "pow_e_motor": 120.0,
            "eta_e_motor": 0.9, "eta_e_motor_re": 0.15, "eta_etc_re": 0.10,
            "vel_min_e_motor": 100.0, "torque_e_motor_max": 200.0,
            "tire_mux_f": 1.65, "tire_muy_f": 1.85,
            "tire_mux_r": 1.95, "tire_muy_r": 2.15,
        }
    else:  # FE
        defaults = {
            "powertrain_type": "electric",
            "m": 880.0, "lf": 1.906, "lr": 1.194, "h_cog": 0.345,
            "sf": 1.3, "sr": 1.3, "f_roll": 0.02,
            "c_w_a": 1.15, "c_z_a_f": 1.24, "c_z_a_r": 1.52,
            "drs_factor": 0.0,
            "pow_e_motor": 200.0,
            "eta_e_motor": 0.9, "eta_e_motor_re": 0.9,
            "torque_e_motor_max": 150.0,
            "tire_mux_f": 1.22, "tire_muy_f": 1.22,
            "tire_mux_r": 1.42, "tire_muy_r": 1.42,
        }

    with st.sidebar.expander("General Parameters", expanded=True):
        veh_mass = st.number_input("Mass [kg]", value=defaults["m"], step=10.0)
        col1, col2 = st.columns(2)
        with col1:
            veh_lf = st.number_input("Front axle to CoG [m]", value=defaults["lf"], step=0.1, format="%.3f")
            veh_sf = st.number_input("Track width front [m]", value=defaults["sf"], step=0.1)
        with col2:
            veh_lr = st.number_input("Rear axle to CoG [m]", value=defaults["lr"], step=0.1, format="%.3f")
            veh_sr = st.number_input("Track width rear [m]", value=defaults["sr"], step=0.1)
        veh_h_cog = st.number_input("CoG height [m]", value=defaults["h_cog"], step=0.01, format="%.3f")
        veh_f_roll = st.number_input("Rolling resistance [-]", value=defaults["f_roll"], step=0.005, format="%.3f")

    with st.sidebar.expander("Aerodynamics"):
        veh_c_w_a = st.number_input("Drag coeff × area [m²]", value=defaults["c_w_a"], step=0.1)
        veh_c_z_a_f = st.number_input("Front downforce coeff × area [m²]", value=defaults["c_z_a_f"], step=0.1)
        veh_c_z_a_r = st.number_input("Rear downforce coeff × area [m²]", value=defaults["c_z_a_r"], step=0.1)
        veh_drs_factor = st.number_input("DRS factor [-]", value=defaults["drs_factor"], step=0.01, format="%.2f")

    with st.sidebar.expander("Powertrain"):
        if series == "F1":
            veh_pow_max = st.number_input("Max engine power [kW]", value=defaults["pow_max"], step=10.0)
            veh_pow_diff = st.number_input("Power drop from max [kW]", value=defaults["pow_diff"], step=5.0)
            col1, col2, col3 = st.columns(3)
            with col1:
                veh_n_begin = st.number_input("RPM begin", value=defaults["n_begin"], step=100.0)
            with col2:
                veh_n_max = st.number_input("RPM max", value=defaults["n_max"], step=100.0)
            with col3:
                veh_n_end = st.number_input("RPM end", value=defaults["n_end"], step=100.0)
            veh_be_max = st.number_input("Fuel consumption [kg/h]", value=defaults["be_max"], step=5.0)

        veh_pow_e_motor = st.number_input("E-motor power [kW]", value=defaults["pow_e_motor"], step=10.0)
        veh_eta_e_motor = st.number_input("E-motor efficiency (drive) [-]", value=defaults["eta_e_motor"], step=0.05, format="%.2f")
        veh_eta_e_motor_re = st.number_input("E-motor efficiency (regen) [-]", value=defaults["eta_e_motor_re"], step=0.05, format="%.2f")
        veh_torque_e_motor_max = st.number_input("E-motor max torque [Nm]", value=defaults["torque_e_motor_max"], step=10.0)

        if series == "F1":
            veh_eta_etc_re = st.number_input("Turbo regen efficiency [-]", value=defaults["eta_etc_re"], step=0.05, format="%.2f")
            veh_vel_min_e_motor = st.number_input("Min velocity for e-motor [km/h]", value=defaults["vel_min_e_motor"], step=10.0)

    with st.sidebar.expander("Tires"):
        st.write("**Front Tires**")
        col1, col2 = st.columns(2)
        with col1:
            tire_mux_f = st.number_input("μx front [-]", value=defaults["tire_mux_f"], step=0.05, format="%.2f")
        with col2:
            tire_muy_f = st.number_input("μy front [-]", value=defaults["tire_muy_f"], step=0.05, format="%.2f")

        st.write("**Rear Tires**")
        col1, col2 = st.columns(2)
        with col1:
            tire_mux_r = st.number_input("μx rear [-]", value=defaults["tire_mux_r"], step=0.05, format="%.2f")
        with col2:
            tire_muy_r = st.number_input("μy rear [-]", value=defaults["tire_muy_r"], step=0.05, format="%.2f")

    # Build custom vehicle parameters dict
    custom_vehicle_pars = {
        "powertrain_type": defaults["powertrain_type"],
        "general": {
            "lf": veh_lf, "lr": veh_lr, "h_cog": veh_h_cog,
            "sf": veh_sf, "sr": veh_sr, "m": veh_mass,
            "f_roll": veh_f_roll, "c_w_a": veh_c_w_a,
            "c_z_a_f": veh_c_z_a_f, "c_z_a_r": veh_c_z_a_r,
            "g": 9.81, "rho_air": 1.18, "drs_factor": veh_drs_factor,
        },
        "engine": {
            "topology": "RWD",
            "pow_e_motor": veh_pow_e_motor * 1e3,
            "eta_e_motor": veh_eta_e_motor,
            "eta_e_motor_re": veh_eta_e_motor_re,
            "torque_e_motor_max": veh_torque_e_motor_max,
        },
        "gearbox": {
            "i_trans": [0.056, 0.091] if series == "FE" else [0.04, 0.070, 0.095, 0.117, 0.143, 0.172, 0.190, 0.206],
            "n_shift": [19000.0, 19000.0] if series == "FE" else [10000.0, 11800.0, 11800.0, 11800.0, 11800.0, 11800.0, 11800.0, 13000.0],
            "e_i": [1.04, 1.04] if series == "FE" else [1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07],
            "eta_g": 0.96,
        },
        "tires": {
            "f": {"circ_ref": 2.073, "fz_0": 3000.0, "mux": tire_mux_f, "muy": tire_muy_f, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
            "r": {"circ_ref": 2.073, "fz_0": 3000.0, "mux": tire_mux_r, "muy": tire_muy_r, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
            "tire_model_exp": 2.0,
        },
    }

    # Add F1-specific engine parameters
    if series == "F1":
        custom_vehicle_pars["engine"].update({
            "pow_max": veh_pow_max * 1e3,
            "pow_diff": veh_pow_diff * 1e3,
            "n_begin": veh_n_begin,
            "n_max": veh_n_max,
            "n_end": veh_n_end,
            "be_max": veh_be_max,
            "eta_etc_re": veh_eta_etc_re,
            "vel_min_e_motor": veh_vel_min_e_motor / 3.6,
        })

with st.sidebar.expander("DRS Options"):
    use_drs1 = st.checkbox("Enable DRS Zone 1", value=(series == "F1"))
    use_drs2 = st.checkbox("Enable DRS Zone 2", value=(series == "F1"))

use_pit = st.sidebar.checkbox("Use Pit Lane", value=False)

with st.sidebar.expander("Solver Settings"):
    limit_braking = st.selectbox(
        "Limit Braking Weak Side",
        options=["FA", "RA", "all", "None"],
        index=0,
        help="FA=Front Axle, RA=Rear Axle",
    )
    limit_braking_val = None if limit_braking == "None" else limit_braking

    v_start_kmh = st.number_input(
        "Start Velocity [km/h]",
        min_value=50.0,
        max_value=350.0,
        value=100.0,
        step=10.0,
    )

    find_v_start = st.checkbox("Auto-find Start Velocity", value=True)

    max_em_iters = st.number_input(
        "Max Energy Management Iterations",
        min_value=1,
        max_value=20,
        value=5,
    )

    es_diff_max = st.number_input(
        "Energy Convergence Threshold [J]",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

# =============================================================================
# DRIVER OPTIONS
# =============================================================================
st.sidebar.subheader("Driver & Strategy")

em_strategy = st.sidebar.selectbox(
    "Energy Management Strategy",
    options=["FCFB", "LBP", "LS", "NONE"],
    index=0,
    help="FCFB=First Come First Boost, LBP=Longest to Breakpoint, LS=Lowest Speed",
)

default_energy = SERIES_CONFIG[series]["initial_energy"]
initial_energy_mj = st.sidebar.slider(
    "Initial Energy [MJ]",
    min_value=0.0,
    max_value=6.0,
    value=default_energy / 1e6,
    step=0.1,
)

with st.sidebar.expander("Driver Behavior"):
    vel_subtr_corner = st.slider(
        "Corner Safety Margin [m/s]",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.1,
    )

    vel_lim_glob_kmh = st.number_input(
        "Global Speed Limit [km/h]",
        min_value=0,
        max_value=400,
        value=0,
        help="Set to 0 for no limit",
    )
    vel_lim_glob = vel_lim_glob_kmh / 3.6 if vel_lim_glob_kmh > 0 else None

    use_recuperation = st.checkbox("Use Energy Recuperation", value=True)

    use_lift_coast = st.checkbox("Use Lift & Coast", value=False)
    lift_coast_dist = st.slider(
        "Lift & Coast Distance [m]",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=5.0,
        disabled=not use_lift_coast,
    )

with st.sidebar.expander("Yellow Flags"):
    yellow_s1 = st.checkbox("Yellow Flag Sector 1", value=False)
    yellow_s2 = st.checkbox("Yellow Flag Sector 2", value=False)
    yellow_s3 = st.checkbox("Yellow Flag Sector 3", value=False)
    yellow_throttle = st.slider(
        "Yellow Flag Throttle",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        disabled=not (yellow_s1 or yellow_s2 or yellow_s3),
    )

# =============================================================================
# RUN BUTTON
# =============================================================================
st.sidebar.divider()
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# =============================================================================
# BUILD OPTIONS DICTS AND RUN
# =============================================================================
if run_button:
    track_opts = {
        "trackname": track_name,
        "flip_track": flip_track,
        "mu_weather": mu_weather,
        "interp_stepsize_des": interp_stepsize,
        "curv_filt_width": curv_filt,
        "use_drs1": use_drs1,
        "use_drs2": use_drs2,
        "use_pit": use_pit,
    }

    solver_opts = {
        "vehicle": vehicle if vehicle != "Custom" else None,
        "series": series,
        "limit_braking_weak_side": limit_braking_val,
        "v_start": v_start_kmh / 3.6,
        "find_v_start": find_v_start,
        "max_no_em_iters": max_em_iters,
        "es_diff_max": es_diff_max,
        "custom_vehicle_pars": custom_vehicle_pars,
    }

    driver_opts = {
        "vel_subtr_corner": vel_subtr_corner,
        "vel_lim_glob": vel_lim_glob,
        "yellow_s1": yellow_s1,
        "yellow_s2": yellow_s2,
        "yellow_s3": yellow_s3,
        "yellow_throttle": yellow_throttle,
        "initial_energy": initial_energy_mj * 1e6,
        "em_strategy": em_strategy,
        "use_recuperation": use_recuperation,
        "use_lift_coast": use_lift_coast,
        "lift_coast_dist": lift_coast_dist,
    }

    with st.spinner(f"Simulating {series} lap at {track_name}..."):
        try:
            result = run_simulation_advanced(track_opts, solver_opts, driver_opts)
            st.session_state.adv_result = result
            st.success(f"Simulation completed for {track_name} ({series})")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
if st.session_state.adv_result is not None:
    result = st.session_state.adv_result

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
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Speed", f"{np.max(result.velocity_kmh):.1f} km/h")
    with col2:
        st.metric("Min Speed", f"{np.min(result.velocity_kmh):.1f} km/h")
    with col3:
        st.metric("Avg Speed", f"{np.mean(result.velocity_kmh):.1f} km/h")
    with col4:
        st.metric("Energy Used", f"{result.energy_consumed:.1f} kJ")

    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["Velocity Profile", "Track Map"])

    with tab1:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=result.distance,
            y=result.velocity_kmh,
            mode='lines',
            name='Velocity',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
        ))
        fig_vel.update_layout(
            xaxis_title="Distance [m]",
            yaxis_title="Velocity [km/h]",
            hovermode='x unified',
            height=450,
        )
        st.plotly_chart(fig_vel, width="stretch")

    with tab2:
        # Normalize velocity for color mapping
        vel_normalized = (result.velocity_kmh - np.min(result.velocity_kmh)) / (
            np.max(result.velocity_kmh) - np.min(result.velocity_kmh)
        )

        fig_track = go.Figure()

        # Plot track segments colored by velocity
        for i in range(len(result.track_x) - 1):
            color = px.colors.sample_colorscale('RdYlGn', vel_normalized[i])[0]
            fig_track.add_trace(go.Scatter(
                x=result.track_x[i:i+2],
                y=result.track_y[i:i+2],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Add start/finish marker
        fig_track.add_trace(go.Scatter(
            x=[result.track_x[0]],
            y=[result.track_y[0]],
            mode='markers',
            marker=dict(size=12, color='white', line=dict(color='black', width=2)),
            name='Start/Finish',
        ))

        fig_track.update_layout(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            height=550,
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig_track, width="stretch")
        st.caption(f"🔴 Low speed ({np.min(result.velocity_kmh):.0f} km/h) → 🟢 High speed ({np.max(result.velocity_kmh):.0f} km/h)")

else:
    # Initial state
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to start.")

    st.markdown("""
    ### Advanced Options

    This page gives you full control over the simulation:

    **Track Options**
    - Flip track direction
    - Adjust grip level for different weather
    - Control interpolation and curvature filtering

    **Vehicle & Solver**
    - Select vehicle configuration
    - Enable/disable DRS zones
    - Configure solver convergence settings

    **Driver & Strategy**
    - Energy management strategy (FCFB, LBP, LS)
    - Initial energy level
    - Lift & coast behavior
    - Yellow flag scenarios
    """)
