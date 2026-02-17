"""
Comparison Page

Compare up to 3 saved simulation runs.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Comparison - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.title("📊 Comparison")
st.caption("Compare up to 3 simulation runs")

# Initialize session state for saved runs
if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

MAX_RUNS = 3
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

# Sidebar - Manage saved runs
st.sidebar.header("Saved Runs")

if len(st.session_state.saved_runs) == 0:
    st.sidebar.info("No runs saved yet. Run a simulation and click 'Save to Compare'.")
else:
    for i, run in enumerate(st.session_state.saved_runs):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(
                f"<span style='color:{COLORS[i]}'>●</span> **{run.track_name}** ({run.vehicle})",
                unsafe_allow_html=True,
            )
            st.caption(f"{run.format_lap_time()} | {run.weather}")
        with col2:
            if st.button("🗑️", key=f"del_{i}", help="Remove this run"):
                st.session_state.saved_runs.pop(i)
                st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Clear All", use_container_width=True):
        st.session_state.saved_runs = []
        st.rerun()

# Main content
if len(st.session_state.saved_runs) == 0:
    st.info("👈 No runs to compare. Go to **Simple Simulation** or **Advanced Simulation**, run a simulation, and click **Save to Compare**.")

    st.markdown("""
    ### How to use

    1. Go to **Simple Simulation** or **Advanced Simulation**
    2. Configure and run a simulation
    3. Click **Save to Compare** to store the result
    4. Repeat for up to 3 different configurations
    5. Come back here to see the comparison

    ### What you can compare

    - Different tracks
    - Different series (F1 vs FE)
    - Different weather conditions
    - Different driver strategies
    """)

elif len(st.session_state.saved_runs) == 1:
    st.warning("Add at least one more run to compare. You have 1 run saved.")

    run = st.session_state.saved_runs[0]
    st.subheader(f"{run.track_name} ({run.vehicle}) - {run.format_lap_time()}")

else:
    # We have 2 or 3 runs to compare
    runs = st.session_state.saved_runs

    # ==========================================================================
    # LAP TIME COMPARISON TABLE
    # ==========================================================================
    st.header("Lap Times")

    # Find fastest lap
    fastest_idx = np.argmin([r.lap_time for r in runs])

    # Create columns for each run
    cols = st.columns(len(runs))

    for i, (col, run) in enumerate(zip(cols, runs)):
        with col:
            delta = run.lap_time - runs[fastest_idx].lap_time
            delta_str = f"+{delta:.3f}s" if delta > 0 else "Fastest"

            st.markdown(
                f"### <span style='color:{COLORS[i]}'>●</span> {run.track_name}",
                unsafe_allow_html=True,
            )
            st.caption(f"{run.vehicle} | {run.weather}")
            st.metric("Lap Time", run.format_lap_time(), delta_str if delta > 0 else None)

            # Sector times
            st.write("**Sectors**")
            for s in range(3):
                fastest_sector = min(r.sector_times[s] for r in runs)
                sector_delta = run.sector_times[s] - fastest_sector
                if sector_delta > 0:
                    st.write(f"S{s+1}: {run.sector_times[s]:.3f}s (+{sector_delta:.3f})")
                else:
                    st.write(f"S{s+1}: {run.sector_times[s]:.3f}s ✓")

    st.divider()

    # ==========================================================================
    # SPEED COMPARISON
    # ==========================================================================
    st.header("Speed Statistics")

    speed_data = {
        "Run": [],
        "Max Speed": [],
        "Min Speed": [],
        "Avg Speed": [],
        "Energy Used": [],
    }

    for i, run in enumerate(runs):
        speed_data["Run"].append(f"{run.track_name} ({run.vehicle})")
        speed_data["Max Speed"].append(f"{np.max(run.velocity_kmh):.1f} km/h")
        speed_data["Min Speed"].append(f"{np.min(run.velocity_kmh):.1f} km/h")
        speed_data["Avg Speed"].append(f"{np.mean(run.velocity_kmh):.1f} km/h")
        speed_data["Energy Used"].append(f"{run.energy_consumed:.1f} kJ")

    # Display as columns
    cols = st.columns(len(runs))
    for i, (col, run) in enumerate(zip(cols, runs)):
        with col:
            st.metric("Max Speed", f"{np.max(run.velocity_kmh):.1f} km/h")
            st.metric("Avg Speed", f"{np.mean(run.velocity_kmh):.1f} km/h")
            st.metric("Energy", f"{run.energy_consumed:.1f} kJ")

    st.divider()

    # ==========================================================================
    # VELOCITY PROFILE OVERLAY
    # ==========================================================================
    st.header("Velocity Profiles")

    # Check if all runs are on the same track (for meaningful overlay)
    same_track = len(set(r.track_name for r in runs)) == 1

    if not same_track:
        st.warning("Runs are on different tracks. Velocity profiles shown separately.")

        tabs = st.tabs([f"{r.track_name} ({r.series})" for r in runs])
        for i, (tab, run) in enumerate(zip(tabs, runs)):
            with tab:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=run.distance,
                    y=run.velocity_kmh,
                    mode='lines',
                    name=f"{run.track_name}",
                    line=dict(color=COLORS[i], width=2),
                ))
                fig.update_layout(
                    xaxis_title="Distance [m]",
                    yaxis_title="Velocity [km/h]",
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")
    else:
        # Same track - overlay the profiles
        fig = go.Figure()

        for i, run in enumerate(runs):
            label = f"{run.vehicle} - {run.weather}"
            fig.add_trace(go.Scatter(
                x=run.distance,
                y=run.velocity_kmh,
                mode='lines',
                name=label,
                line=dict(color=COLORS[i], width=2),
            ))

        fig.update_layout(
            xaxis_title="Distance [m]",
            yaxis_title="Velocity [km/h]",
            hovermode='x unified',
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )
        st.plotly_chart(fig, width="stretch")

        # ==========================================================================
        # DELTA TIME CHART
        # ==========================================================================
        st.header("Delta Time")
        st.caption("Time difference compared to fastest run (positive = slower)")

        # Calculate cumulative time for each run
        fig_delta = go.Figure()

        # Use the fastest run as reference
        ref_run = runs[fastest_idx]
        ref_time_cumulative = np.zeros(len(ref_run.distance))
        ref_velocity = ref_run.velocity

        # Calculate time at each point for reference (approximate)
        for j in range(1, len(ref_run.distance)):
            ds = ref_run.distance[j] - ref_run.distance[j-1]
            avg_vel = (ref_velocity[j] + ref_velocity[j-1]) / 2
            ref_time_cumulative[j] = ref_time_cumulative[j-1] + ds / avg_vel

        for i, run in enumerate(runs):
            if i == fastest_idx:
                continue  # Skip reference run

            # Calculate cumulative time for this run
            time_cumulative = np.zeros(len(run.distance))
            for j in range(1, len(run.distance)):
                ds = run.distance[j] - run.distance[j-1]
                avg_vel = (run.velocity[j] + run.velocity[j-1]) / 2
                time_cumulative[j] = time_cumulative[j-1] + ds / avg_vel

            # Interpolate to match reference distance points
            delta = np.interp(ref_run.distance, run.distance, time_cumulative) - ref_time_cumulative

            label = f"{run.vehicle} - {run.weather} vs Fastest"
            fig_delta.add_trace(go.Scatter(
                x=ref_run.distance,
                y=delta,
                mode='lines',
                name=label,
                line=dict(color=COLORS[i], width=2),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(COLORS[i].lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + [0.2])}",
            ))

        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_delta.update_layout(
            xaxis_title="Distance [m]",
            yaxis_title="Delta Time [s]",
            hovermode='x unified',
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )
        st.plotly_chart(fig_delta, width="stretch")

# Show capacity info
st.sidebar.divider()
st.sidebar.caption(f"Saved: {len(st.session_state.saved_runs)}/{MAX_RUNS}")
