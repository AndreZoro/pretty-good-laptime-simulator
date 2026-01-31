"""
Simple Simulation Page

Basic lap time simulation with minimal configuration options.
"""

import streamlit as st
import numpy as np

from helpers.simulation import run_simulation, get_available_tracks, SERIES_CONFIG
from helpers.visualization import render_simulation_plots

st.set_page_config(
    page_title="Simple Simulation - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏁 Simple Simulation")

# Initialize session state
if "simple_result" not in st.session_state:
    st.session_state.simple_result = None
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

MAX_RUNS = 3

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Track selection
available_tracks = get_available_tracks()
track = st.sidebar.selectbox(
    "Track",
    options=available_tracks,
    index=available_tracks.index("Shanghai") if "Shanghai" in available_tracks else 0,
)

# Series selection
series = st.sidebar.radio(
    "Series",
    options=["F1", "FE"],
    horizontal=True,
    help="F1 = Formula 1 (hybrid), FE = Formula E (electric)",
)

# Weather condition
mu_weather = st.sidebar.slider(
    "Weather Conditions",
    min_value=0.6,
    max_value=1.0,
    value=1.0,
    step=0.05,
    format="%.2f",
    help="Grip level: 1.0 = Dry, 0.6 = Wet",
)

# Show weather label
weather_desc = "Dry" if mu_weather >= 0.95 else "Damp" if mu_weather >= 0.75 else "Wet"
st.sidebar.caption(f"Conditions: **{weather_desc}**")

# Run button
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Show current config
st.sidebar.divider()
st.sidebar.caption(f"Vehicle: `{SERIES_CONFIG[series]['vehicle']}`")

# Main area
if run_button:
    with st.spinner(f"Simulating {series} lap at {track}..."):
        try:
            result = run_simulation(track, series, mu_weather)
            st.session_state.simple_result = result
            st.success(f"Simulation completed for {track} ({series})")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)

# Display results if we have them
if st.session_state.simple_result is not None:
    result = st.session_state.simple_result

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
    render_simulation_plots(result, key_prefix="simple_")

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to start.")

    e_col1, e_col2, = st.columns(2)

    with e_col1:
        st.markdown("""
        ### Quick Start

        1. **Select a track** from the dropdown
        2. **Choose the series** (F1 for Formula 1, FE for Formula E)
        3. **Adjust weather** if desired (lower = more slippery)
        4. Click **Run Simulation**
        """)
    with e_col2:
        st.markdown("""
        ### What you'll see

        - **Lap Time** - Total time and sector breakdown
        - **Velocity Profile** - Speed throughout the lap
        - **Track Map** - Circuit colored by velocity (red=slow, green=fast)
        """)
