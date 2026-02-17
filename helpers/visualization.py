"""
Visualization helpers for simulation results.

Provides reusable plotting functions for the Streamlit UI.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from helpers.simulation import SimulationResult


def get_viz_options(result: SimulationResult) -> dict:
    """Get available visualization options for a simulation result."""
    G = 9.81  # m/s²
    options = {
        "Velocity": {"data": result.velocity_kmh, "unit": "km/h", "colorscale": "RdYlGn"},
        "Acceleration": {"data": result.acceleration / G, "unit": "g", "colorscale": "RdBu_r"},
        "Lateral Acceleration": {"data": result.lat_acceleration / G, "unit": "g", "colorscale": "Viridis"},
        "Curvature": {"data": np.abs(result.curvature), "unit": "rad/m", "colorscale": "Plasma"},
        "Gear": {"data": result.gear.astype(float), "unit": "", "colorscale": "Turbo"},
    }

    if result.rpm is not None:
        options["RPM"] = {"data": result.rpm, "unit": "rpm", "colorscale": "Inferno"}
    if result.engine_torque is not None:
        options["Engine Torque"] = {"data": result.engine_torque, "unit": "Nm", "colorscale": "Hot"}
    if result.e_motor_torque is not None:
        options["E-Motor Torque"] = {"data": result.e_motor_torque, "unit": "Nm", "colorscale": "Cividis"}
    if result.tire_loads is not None:
        for i, corner in enumerate(["FL", "FR", "RL", "RR"]):
            options[f"Tire Load {corner}"] = {"data": result.tire_loads[:, i], "unit": "N", "colorscale": "YlOrRd"}
    if result.energy_storage is not None:
        options["Energy Storage"] = {"data": result.energy_storage, "unit": "kJ", "colorscale": "Blues"}
    if result.fuel_consumed_profile is not None:
        options["Fuel Consumed"] = {"data": result.fuel_consumed_profile, "unit": "kg", "colorscale": "Oranges"}
    if result.energy_consumed_profile is not None:
        options["Energy Consumed"] = {"data": result.energy_consumed_profile, "unit": "kJ", "colorscale": "Reds"}
    if result.drs is not None:
        options["DRS"] = {"data": result.drs.astype(float), "unit": "", "colorscale": "Picnic"}
    if result.friction is not None:
        options["Friction"] = {"data": result.friction, "unit": "μ", "colorscale": "Greens"}
    if result.e_motor_power is not None:
        options["E-Motor Power"] = {"data": result.e_motor_power, "unit": "kW", "colorscale": "RdBu_r"}
    if result.harvest_power is not None:
        options["Harvest Power"] = {"data": result.harvest_power, "unit": "kW", "colorscale": "Reds"}

    return options


def render_simulation_plots(result: SimulationResult, key_prefix: str = "") -> None:
    """
    Render the profile chart and track map for a simulation result.

    Args:
        result: SimulationResult object containing the data to visualize
        key_prefix: Optional prefix for Streamlit widget keys (for use on multiple pages)
    """
    viz_options = get_viz_options(result)
    selected_viz = st.selectbox(
        "Display",
        options=list(viz_options.keys()),
        key=f"{key_prefix}viz_select" if key_prefix else None,
    )

    viz_data = viz_options[selected_viz]["data"]
    viz_unit = viz_options[selected_viz]["unit"]
    viz_colorscale = viz_options[selected_viz]["colorscale"]

    # Normalize data for color mapping
    data_min = np.min(viz_data)
    data_max = np.max(viz_data)
    if data_max - data_min > 0:
        data_normalized = (viz_data - data_min) / (data_max - data_min)
    else:
        data_normalized = np.zeros_like(viz_data)

    col_profile, col_track = st.columns([4, 2])

    with col_profile:
        st.header(f"{selected_viz} Profile")
        fig_profile = create_profile_chart(
            result.distance, viz_data, selected_viz, viz_unit
        )
        st.plotly_chart(fig_profile, width="stretch")

    with col_track:
        st.header("Track Map")
        fig_track = create_track_map(
            result.track_x, result.track_y, data_normalized, viz_colorscale
        )
        st.plotly_chart(fig_track, width="stretch")

        # Color scale legend
        if viz_unit:
            st.caption(f"Low ({data_min:.1f} {viz_unit}) → High ({data_max:.1f} {viz_unit})")
        else:
            st.caption(f"Low ({data_min:.0f}) → High ({data_max:.0f})")


def create_profile_chart(
    distance: np.ndarray,
    data: np.ndarray,
    name: str,
    unit: str,
    height: int = 400,
) -> go.Figure:
    """
    Create a profile chart showing data vs distance.

    Args:
        distance: Distance array [m]
        data: Data array to plot
        name: Name of the data (for labels)
        unit: Unit string for y-axis
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distance,
        y=data,
        mode='lines',
        name=name,
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))

    y_label = f"{name} [{unit}]" if unit else name
    fig.update_layout(
        xaxis_title="Distance [m]",
        yaxis_title=y_label,
        hovermode='x unified',
        height=height,
    )
    return fig


def create_track_map(
    track_x: np.ndarray,
    track_y: np.ndarray,
    color_data: np.ndarray,
    colorscale: str,
    height: int = 500,
) -> go.Figure:
    """
    Create a track map colored by the given data.

    Args:
        track_x: Track x coordinates
        track_y: Track y coordinates
        color_data: Normalized data (0-1) for coloring
        colorscale: Plotly colorscale name
        height: Chart height in pixels

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Plot track segments colored by data
    for i in range(len(track_x) - 1):
        color = px.colors.sample_colorscale(colorscale, color_data[i])[0]
        fig.add_trace(go.Scatter(
            x=track_x[i:i+2],
            y=track_y[i:i+2],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Add start/finish marker
    fig.add_trace(go.Scatter(
        x=[track_x[0]],
        y=[track_y[0]],
        mode='markers',
        marker=dict(size=12, color='white', line=dict(color='black', width=2)),
    ))

    fig.update_layout(
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        height=height,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig
