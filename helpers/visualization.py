"""
Visualization helpers for simulation results.

Provides reusable plotting functions for the Streamlit UI.
"""

import json
import os

import numpy as np
import plotly
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

_PLOTLY_JS: str | None = None

# Matches the profile chart's blue fill gradient (dark → #1f77b4)
_BLUE_GRADIENT = [[0, "#061626"], [0.5, "#0f3d6b"], [1, "#1f77b4"]]


def _get_plotly_js() -> str:
    global _PLOTLY_JS
    if _PLOTLY_JS is None:
        path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
        with open(path) as f:
            _PLOTLY_JS = f.read()
    return _PLOTLY_JS

from helpers.simulation import SimulationResult


def get_viz_options(result: SimulationResult) -> dict:
    """Get available visualization options for a simulation result."""
    G = 9.81  # m/s²
    cs = _BLUE_GRADIENT
    options = {
        "Velocity": {"data": result.velocity_kmh, "unit": "km/h", "colorscale": cs},
        "Acceleration": {"data": result.acceleration / G, "unit": "g", "colorscale": cs},
        "Lateral Acceleration": {"data": result.lat_acceleration / G, "unit": "g", "colorscale": cs},
        "Curvature": {"data": np.abs(result.curvature), "unit": "rad/m", "colorscale": cs},
        "Gear": {"data": result.gear.astype(float), "unit": "", "colorscale": cs},
    }

    if result.rpm is not None:
        options["RPM"] = {"data": result.rpm, "unit": "rpm", "colorscale": cs}
    if result.engine_torque is not None:
        options["Engine Torque"] = {"data": result.engine_torque, "unit": "Nm", "colorscale": cs}
    if result.e_motor_torque is not None:
        options["E-Motor Torque"] = {"data": result.e_motor_torque, "unit": "Nm", "colorscale": cs}
    if result.tire_loads is not None:
        for i, corner in enumerate(["FL", "FR", "RL", "RR"]):
            options[f"Tire Load {corner}"] = {"data": result.tire_loads[:, i], "unit": "N", "colorscale": cs}
    if result.energy_storage is not None:
        options["Energy Storage"] = {"data": result.energy_storage, "unit": "kJ", "colorscale": cs}
    if result.fuel_consumed_profile is not None:
        options["Fuel Consumed"] = {"data": result.fuel_consumed_profile, "unit": "kg", "colorscale": cs}
    if result.energy_consumed_profile is not None:
        options["Energy Consumed"] = {"data": result.energy_consumed_profile, "unit": "kJ", "colorscale": cs}
    if result.drs is not None:
        drs_label = "Active Aero" if result.active_aero else "DRS"
        options[drs_label] = {"data": result.drs.astype(float), "unit": "", "colorscale": cs}
    if result.friction is not None:
        options["Friction"] = {"data": result.friction, "unit": "μ", "colorscale": cs}
    if result.e_motor_power is not None:
        options["E-Motor Power"] = {"data": result.e_motor_power, "unit": "kW", "colorscale": cs}
    if result.harvest_power is not None:
        options["Harvest Power"] = {"data": result.harvest_power, "unit": "kW", "colorscale": cs}

    return options


def render_simulation_plots(result: SimulationResult, key_prefix: str = "") -> None:
    """
    Render the profile chart and track map for a simulation result.
    Hover on the profile moves a cursor on the track map; hover on the
    track map draws a vertical line on the profile.

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

    data_min = float(np.min(viz_data))
    data_max = float(np.max(viz_data))
    if data_max - data_min > 0:
        data_normalized = (viz_data - data_min) / (data_max - data_min)
    else:
        data_normalized = np.zeros_like(viz_data)

    # Downsample to ~10 m resolution for track rendering
    n_pts = len(result.track_x)
    step_m = result.distance[-1] / n_pts if n_pts > 1 else 1.0
    ds = max(1, round(10.0 / step_m))
    tx = result.track_x[::ds].tolist()
    ty = result.track_y[::ds].tolist()
    dn = data_normalized[::ds].tolist()
    dist = result.distance[::ds].tolist()
    vd = viz_data[::ds].tolist()

    y_label = f"{selected_viz} [{viz_unit}]" if viz_unit else selected_viz
    unit_str = f" {viz_unit}" if viz_unit else ""

    payload = json.dumps({
        "prof_dist": result.distance.tolist(),
        "prof_data": viz_data.tolist(),
        "track_x": tx,
        "track_y": ty,
        "color_vals": dn,
        "colorscale": viz_colorscale,
        "dist": dist,
        "vd": vd,
        "viz_name": selected_viz,
        "y_label": y_label,
        "unit_str": unit_str,
        "viz_unit": viz_unit,
    })

    html = f"""<!DOCTYPE html>
<html><head>
<script>{_get_plotly_js()}</script>
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  #wrap {{ display:flex; width:100%; gap:6px; }}
  #profile {{ flex:4; min-width:0; }}
  #trackmap {{ flex:2; min-width:0; position:relative; }}
  @keyframes pulse-ring {{
    0%   {{ transform:scale(0.8); opacity:0.9; }}
    100% {{ transform:scale(2.6); opacity:0;   }}
  }}
  #pulse-ring {{
    position:absolute; pointer-events:none; display:none;
    width:14px; height:14px; margin:-7px 0 0 -7px;
    border-radius:50%; border:2px solid #FF6B00;
    animation:pulse-ring 1.0s ease-out infinite;
  }}
</style>
</head><body>
<div id="wrap">
  <div id="profile"></div>
  <div id="trackmap"></div>
</div>
<script>
const D = {payload};

// ── Profile chart ──────────────────────────────────────────────
Plotly.newPlot('profile', [{{
  x: D.prof_dist, y: D.prof_data,
  type: 'scatter', mode: 'lines',
  fill: 'tozeroy',
  fillgradient: {{type: 'vertical', colorscale: [[0, 'rgba(31,119,180,0)'], [1, 'rgba(31,119,180,0.45)']]}},
  line: {{color: '#1f77b4', width: 2}},
  hovertemplate: '<b>%{{x:.0f}} m</b><br>' + D.viz_name + ': %{{y:.2f}}' + D.unit_str + '<extra></extra>',
}}], {{
  xaxis: {{title: 'Distance [m]', showgrid: false, zeroline: false,
           showspikes: true, spikemode: 'across', spikesnap: 'cursor',
           spikecolor: '#FF6B00', spikethickness: 1, spikedash: 'solid'}},
  yaxis: {{title: D.y_label, showgrid: true, gridcolor: 'rgba(128,128,128,0.15)', zeroline: false}},
  hovermode: 'x unified',
  hoverlabel: {{bgcolor: '#FF6B00', font: {{color: 'white'}}, bordercolor: '#FF6B00'}},
  height: 400,
  margin: {{l:60, r:20, t:20, b:50}},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  shapes: [{{
    type: 'line', xref: 'x', yref: 'paper',
    x0: 0, x1: 0, y0: 0, y1: 1,
    line: {{color: 'red', width: 1, dash: 'dot'}},
    visible: false,
  }}],
}}, {{responsive: true, displayModeBar: false}});

// ── Track map ──────────────────────────────────────────────────
const hoverFmt = D.viz_unit
  ? '<b>%{{customdata[0]:.0f}} m</b><br>' + D.viz_name + ': %{{customdata[1]:.2f}} ' + D.viz_unit + '<extra></extra>'
  : '<b>%{{customdata[0]:.0f}} m</b><br>' + D.viz_name + ': %{{customdata[1]:.2f}}<extra></extra>';

// Single colored-marker trace — one trace instead of N segment traces
const segs = [{{
  x: D.track_x, y: D.track_y,
  type: 'scatter', mode: 'markers',
  marker: {{
    color: D.color_vals, colorscale: D.colorscale,
    cmin: 0, cmax: 1, size: 10, showscale: false,
  }},
  customdata: D.track_x.map((_, i) => [D.dist[i], D.vd[i]]),
  hovertemplate: hoverFmt,
  showlegend: false,
}}];

// Start/finish marker
segs.push({{
  x: [D.track_x[0]], y: [D.track_y[0]],
  type: 'scatter', mode: 'markers',
  marker: {{size: 12, color: 'white', line: {{color: 'black', width: 2}}}},
  showlegend: false, hoverinfo: 'skip',
}});

// Cursor dot (moved on profile hover)
const cursorIdx = segs.length;
segs.push({{
  x: [null], y: [null],
  type: 'scatter', mode: 'markers',
  marker: {{size: 14, color: 'red', line: {{color: 'white', width: 2}}, symbol: 'circle'}},
  showlegend: false, hoverinfo: 'skip',
}});

Plotly.newPlot('trackmap', segs, {{
  xaxis: {{visible: false}},
  yaxis: {{visible: false, scaleanchor: 'x', scaleratio: 1}},
  hoverlabel: {{bgcolor: '#FF6B00', font: {{color: 'white'}}, bordercolor: '#FF6B00'}},
  height: 400,
  margin: {{l:10, r:10, t:20, b:10}},
  showlegend: false,
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
}}, {{responsive: true, displayModeBar: false}});

// ── Pulse ring overlay ─────────────────────────────────────────
const ring = document.createElement('div');
ring.id = 'pulse-ring';
document.getElementById('trackmap').appendChild(ring);

function showPulse(dataX, dataY) {{
  const fl = document.getElementById('trackmap')._fullLayout;
  const xpx = fl.xaxis.d2p(dataX) + fl.margin.l;
  const ypx = fl.yaxis.d2p(dataY) + fl.margin.t;
  ring.style.left = xpx + 'px';
  ring.style.top  = ypx + 'px';
  ring.style.display = 'block';
}}

// ── Linking ────────────────────────────────────────────────────
function bisect(arr, val) {{
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1; else hi = mid;
  }}
  return lo;
}}

// Profile hover → move cursor dot + pulse ring on track map
document.getElementById('profile').on('plotly_hover', function(ev) {{
  const idx = bisect(D.dist, ev.points[0].x);
  Plotly.restyle('trackmap', {{x: [[D.track_x[idx]]], y: [[D.track_y[idx]]]}}, [cursorIdx]);
  showPulse(D.track_x[idx], D.track_y[idx]);
}});
document.getElementById('profile').on('plotly_unhover', function() {{
  Plotly.restyle('trackmap', {{x: [[null]], y: [[null]]}}, [cursorIdx]);
  ring.style.display = 'none';
}});

// Track map hover → show vertical line on profile
document.getElementById('trackmap').on('plotly_hover', function(ev) {{
  if (ev.points[0].customdata) {{
    Plotly.relayout('profile', {{
      'shapes[0].x0': ev.points[0].customdata[0],
      'shapes[0].x1': ev.points[0].customdata[0],
      'shapes[0].visible': true,
    }});
  }}
}});
document.getElementById('trackmap').on('plotly_unhover', function() {{
  Plotly.relayout('profile', {{'shapes[0].visible': false}});
}});
</script>
</body></html>"""

    components.html(html, height=430)

    if viz_unit:
        st.caption(f"Color: Low ({data_min:.1f} {viz_unit}) → High ({data_max:.1f} {viz_unit})")
    else:
        st.caption(f"Color: Low ({data_min:.0f}) → High ({data_max:.0f})")


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
    distance: np.ndarray | None = None,
    viz_data: np.ndarray | None = None,
    viz_unit: str = "",
    viz_name: str = "",
) -> go.Figure:
    """
    Create a track map colored by the given data.

    Args:
        track_x: Track x coordinates
        track_y: Track y coordinates
        color_data: Normalized data (0-1) for coloring
        colorscale: Plotly colorscale name
        height: Chart height in pixels
        distance: Distance array [m] for hover labels
        viz_data: Raw (un-normalized) data for hover values
        viz_unit: Unit string for hover label
        viz_name: Name of the visualized quantity

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

    # Invisible hover overlay with distance info
    if distance is not None:
        if viz_data is not None and viz_name:
            value_fmt = ".1f" if viz_unit else ".0f"
            unit_str = f" {viz_unit}" if viz_unit else ""
            hover_template = (
                f"<b>%{{customdata[0]:.0f}} m</b><br>"
                f"{viz_name}: %{{customdata[1]:{value_fmt}}}{unit_str}"
                "<extra></extra>"
            )
            custom = np.column_stack([distance, viz_data])
        else:
            hover_template = "<b>%{customdata:.0f} m</b><extra></extra>"
            custom = distance

        fig.add_trace(go.Scatter(
            x=track_x,
            y=track_y,
            mode='markers',
            marker=dict(size=10, opacity=0),
            customdata=custom,
            hovertemplate=hover_template,
            showlegend=False,
        ))

    # Add start/finish marker
    fig.add_trace(go.Scatter(
        x=[track_x[0]],
        y=[track_y[0]],
        mode='markers',
        marker=dict(size=12, color='white', line=dict(color='black', width=2)),
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.update_layout(
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        height=height,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig
