"""
Simple Simulation Page

Basic lap time simulation with minimal configuration options.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from helpers.simulation import run_simulation, get_available_tracks, SERIES_CONFIG

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

    # Velocity profile chart
    st.header("Velocity Profile")
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
        height=400,
    )
    st.plotly_chart(fig_vel, width="stretch")

    # Track map with velocity coloring
    st.header("Track Map")

    # Normalize velocity for color mapping
    vel_normalized = (result.velocity_kmh - np.min(result.velocity_kmh)) / (
        np.max(result.velocity_kmh) - np.min(result.velocity_kmh)
    )

    fig_track = go.Figure()

    # Plot track segments colored by velocity
    for i in range(len(result.track_x) - 1):
        color = px.colors.sample_colorscale(
            'RdYlGn',
            vel_normalized[i]
        )[0]
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
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig_track, width="stretch")

    # Color scale legend
    st.caption(f"🔴 Low speed ({np.min(result.velocity_kmh):.0f} km/h) → 🟢 High speed ({np.max(result.velocity_kmh):.0f} km/h)")

else:
    # Initial state - show instructions
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to start.")

    st.markdown("""
    ### Quick Start

    1. **Select a track** from the dropdown
    2. **Choose the series** (F1 for Formula 1, FE for Formula E)
    3. **Adjust weather** if desired (lower = more slippery)
    4. Click **Run Simulation**

    ### What you'll see

    - **Lap Time** - Total time and sector breakdown
    - **Velocity Profile** - Speed throughout the lap
    - **Track Map** - Circuit colored by velocity (red=slow, green=fast)
    """)
