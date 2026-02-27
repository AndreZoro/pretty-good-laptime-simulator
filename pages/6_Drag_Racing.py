"""
Drag Racing Page

Standing-start acceleration simulation for 1/8 mile, 1/4 mile and 1 km.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helpers.simulation import get_available_vehicles, run_drag_simulation
from laptimesim.src.drag_test import DRAG_DISTANCES

st.set_page_config(
    page_title="Drag Racing - Pretty Decent Straight Line Sim",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 Drag Racing")
st.caption("Standing-start acceleration simulation — 1/8 mile · 1/4 mile · 1 km")

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

if "drag_result" not in st.session_state:
    st.session_state.drag_result = None
if "drag_saved" not in st.session_state:
    st.session_state.drag_saved = []     # list of run dicts for comparison

MAX_SAVED = 3

# Chart colours for overlaid comparison runs
RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Test Parameters")

available_vehicles = get_available_vehicles()
default_idx = (
    available_vehicles.index("EH_Zapovic_Breeze")
    if "EH_Zapovic_Breeze" in available_vehicles
    else 0
)
vehicle = st.sidebar.selectbox(
    "Vehicle",
    options=available_vehicles,
    index=default_idx,
)

mu = st.sidebar.slider(
    "Track Grip (μ)",
    min_value=0.6,
    max_value=1.3,
    value=1.0,
    step=0.05,
    format="%.2f",
    help="1.0 = dry tarmac, 0.6 = wet, 1.2+ = slick/sticky",
)

grip_label = (
    "🌧️ Wet" if mu < 0.75
    else "🌦️ Damp" if mu < 0.95
    else "☀️ Dry" if mu <= 1.05
    else "🏁 High grip"
)
st.sidebar.caption(f"Conditions: **{grip_label}**")

run_btn = st.sidebar.button("🚀 Run Drag Pull", type="primary", use_container_width=True)

st.sidebar.divider()

if st.session_state.drag_saved:
    st.sidebar.subheader("Saved runs")
    for i, run in enumerate(st.session_state.drag_saved):
        st.sidebar.caption(
            f"{i + 1}. {run['vehicle']}  μ={run['mu']:.2f}"
        )
    if st.sidebar.button("🗑️ Clear saved runs", use_container_width=True):
        st.session_state.drag_saved = []
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _interp_at_dist(dist_arr, t_arr, vel_arr, d_target):
    idx = np.searchsorted(dist_arr, d_target)
    if idx == 0 or idx >= len(dist_arr):
        return None, None
    alpha = (d_target - dist_arr[idx - 1]) / (dist_arr[idx] - dist_arr[idx - 1])
    return (
        t_arr[idx - 1] + alpha * (t_arr[idx] - t_arr[idx - 1]),
        vel_arr[idx - 1] + alpha * (vel_arr[idx] - vel_arr[idx - 1]),
    )


def _interp_at_vel(vel_arr, t_arr, v_target):
    idx = np.searchsorted(vel_arr, v_target)
    if idx == 0 or idx >= len(vel_arr):
        return None
    alpha = (v_target - vel_arr[idx - 1]) / (vel_arr[idx] - vel_arr[idx - 1])
    return t_arr[idx - 1] + alpha * (t_arr[idx] - t_arr[idx - 1])


def _build_velocity_chart(runs: list) -> go.Figure:
    """Velocity vs distance chart with milestone annotations, supporting multiple runs."""
    fig = go.Figure()

    for i, run in enumerate(runs):
        r   = run["result"]
        col = RUN_COLORS[i % len(RUN_COLORS)]
        label = f"{run['vehicle']}  μ={run['mu']:.2f}"

        fig.add_trace(go.Scatter(
            x=r["dist"],
            y=r["vel"] * 3.6,
            mode="lines",
            name=label,
            line=dict(color=col, width=2.5),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(31, 119, 180, 0.12)" if i == 0 else None,
            hovertemplate="Distance: %{x:.1f} m<br>Speed: %{y:.1f} km/h<extra>" + label + "</extra>",
        ))

    # Milestone markers (based on first run)
    r0 = runs[0]["result"]
    milestone_styles = {
        "1/8 mile": ("#6c757d", DRAG_DISTANCES["1/8 mile"]),
        "1/4 mile": ("#6c757d", DRAG_DISTANCES["1/4 mile"]),
        "1 km":     ("#6c757d", DRAG_DISTANCES["1 km"]),
    }
    for name, (color, d_m) in milestone_styles.items():
        t_val, v_val = _interp_at_dist(r0["dist"], r0["t"], r0["vel"], d_m)
        annotation_text = (
            f"<b>{name}</b><br>{t_val:.3f} s<br>{v_val * 3.6:.0f} km/h"
            if t_val else name
        )
        fig.add_vline(
            x=d_m,
            line=dict(color=color, dash="dash", width=1.2),
        )
        fig.add_annotation(
            x=d_m,
            y=1.0,
            yref="paper",
            text=annotation_text,
            showarrow=False,
            xanchor="left",
            xshift=6,
            yanchor="top",
            font=dict(size=11, color=color),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_layout(
        xaxis_title="Distance [m]",
        yaxis_title="Speed [km/h]",
        hovermode="x unified",
        height=420,
        margin=dict(t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _build_detail_charts(run: dict) -> go.Figure:
    """Two-row subplot: acceleration and gear vs distance."""
    r = run["result"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Acceleration", "Gear"),
        row_heights=[0.6, 0.4],
    )

    # Acceleration
    fig.add_trace(go.Scatter(
        x=r["dist"],
        y=r["a_x"],
        mode="lines",
        name="Acceleration",
        line=dict(color="#ff7f0e", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 127, 14, 0.15)",
        hovertemplate="%{y:.2f} m/s²<extra></extra>",
    ), row=1, col=1)

    # Gear (step)
    fig.add_trace(go.Scatter(
        x=r["dist"],
        y=r["gear"] + 1,
        mode="lines",
        name="Gear",
        line=dict(color="#2ca02c", width=2, shape="hv"),
        hovertemplate="Gear %{y}<extra></extra>",
    ), row=2, col=1)

    # Milestone lines on both rows
    for d_m in DRAG_DISTANCES.values():
        for row in (1, 2):
            fig.add_vline(
                x=d_m,
                line=dict(color="#6c757d", dash="dash", width=1.0),
                row=row, col=1,
            )

    fig.update_yaxes(title_text="m/s²", row=1, col=1)
    fig.update_yaxes(
        title_text="Gear",
        tickvals=list(range(1, len(run["result"]["gear"]) + 2)),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Distance [m]", row=2, col=1)
    fig.update_layout(
        height=380,
        margin=dict(t=30, b=40),
        showlegend=False,
    )
    return fig


def _render_metrics(run: dict):
    """Render speed benchmarks and distance milestones."""
    r = run["result"]

    # Speed benchmarks
    targets = [
        ("0–60 mph",   60.0 * 1.60934 / 3.6),
        ("0–100 km/h", 100.0 / 3.6),
        ("0–200 km/h", 200.0 / 3.6),
    ]
    cols = st.columns(3)
    for col, (label, v_t) in zip(cols, targets):
        t_val = _interp_at_vel(r["vel"], r["t"], v_t)
        col.metric(label, f"{t_val:.2f} s" if t_val else "—")

    st.divider()

    # Distance milestones
    dist_targets = [
        ("⅛ Mile",  DRAG_DISTANCES["1/8 mile"]),
        ("¼ Mile",  DRAG_DISTANCES["1/4 mile"]),
        ("1 km",    DRAG_DISTANCES["1 km"]),
    ]
    cols = st.columns(3)
    for col, (label, d_m) in zip(cols, dist_targets):
        t_val, v_val = _interp_at_dist(r["dist"], r["t"], r["vel"], d_m)
        if t_val is not None:
            col.metric(
                label,
                f"{t_val:.3f} s",
                f"{v_val * 3.6:.1f} km/h trap",
                delta_color="off",
            )
        else:
            col.metric(label, "—")


# ──────────────────────────────────────────────────────────────────────────────
# Run simulation
# ──────────────────────────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Running drag test for {vehicle}…"):
        try:
            run_data = run_drag_simulation(vehicle=vehicle, mu=mu)
            st.session_state.drag_result = run_data
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)

# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state.drag_result is not None:
    run = st.session_state.drag_result

    # Header
    st.subheader(f"{run['vehicle']}  ·  {run['series']}")
    st.caption(
        f"Mass: {run['mass_kg']:.0f} kg  ·  "
        f"Power: {run['power_kw']:.0f} kW  ·  "
        f"Topology: {run['topology']}  ·  "
        f"μ = {run['mu']:.2f}"
    )

    # Save to compare
    col_save, _ = st.columns([1, 4])
    with col_save:
        if len(st.session_state.drag_saved) >= MAX_SAVED:
            st.warning(f"Max {MAX_SAVED} saved. Clear in the sidebar.")
        else:
            if st.button("💾 Save to compare", use_container_width=True):
                st.session_state.drag_saved.append(run)
                st.success(
                    f"Saved ({len(st.session_state.drag_saved)}/{MAX_SAVED})"
                )

    st.divider()

    # Metrics
    _render_metrics(run)

    st.divider()

    # Build chart data: saved runs + current run (deduplicated)
    chart_runs = list(st.session_state.drag_saved)
    if not any(r is run for r in chart_runs):
        chart_runs = [run] + chart_runs

    st.subheader("Speed vs Distance")
    st.plotly_chart(_build_velocity_chart(chart_runs), use_container_width=True)

    st.subheader("Acceleration & Gear")
    st.plotly_chart(_build_detail_charts(run), use_container_width=True)

else:
    # Empty state
    st.info("👈 Select a vehicle in the sidebar and click **Run Drag Pull** to start.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What this simulates
        - Standing start from **0 km/h**
        - Full throttle on a **flat, straight road**
        - Traction limit enforced per the car's AWD / RWD / FWD setup
        - Aerodynamic drag and rolling resistance included
        """)
    with col2:
        st.markdown("""
        ### Results shown
        - **0–60 mph / 0–100 / 0–200** sprint times
        - **⅛ mile** · **¼ mile** · **1 km** elapsed time and trap speed
        - Speed, acceleration and gear profiles vs distance
        - Save up to 3 runs for **side-by-side comparison**
        """)
