"""
Visualization helpers for simulation results.

Provides reusable plotting functions for the Streamlit UI.
"""

import json
import os

import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

_PLOTLY_JS: str | None = None

# Matches the profile chart's blue fill gradient (dark → #1f77b4)
_BLUE_GRADIENT = [[0, "#061626"], [0.5, "#0f3d6b"], [1, "#1f77b4"]]


def _get_plotly_js() -> str:
    global _PLOTLY_JS
    if _PLOTLY_JS is None:
        path = os.path.join(
            os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"
        )
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
        "Acceleration": {
            "data": result.acceleration / G,
            "unit": "g",
            "colorscale": cs,
        },
        "Lateral Acceleration": {
            "data": result.lat_acceleration / G,
            "unit": "g",
            "colorscale": cs,
        },
        "Curvature": {
            "data": np.abs(result.curvature),
            "unit": "rad/m",
            "colorscale": cs,
        },
        "Gear": {"data": result.gear.astype(float), "unit": "", "colorscale": cs},
    }

    if result.rpm is not None:
        options["RPM"] = {"data": result.rpm, "unit": "rpm", "colorscale": cs}
    if result.engine_torque is not None:
        options["Engine Torque"] = {
            "data": result.engine_torque,
            "unit": "Nm",
            "colorscale": cs,
        }
    if result.e_motor_torque is not None:
        options["E-Motor Torque"] = {
            "data": result.e_motor_torque,
            "unit": "Nm",
            "colorscale": cs,
        }
    if result.tire_loads is not None:
        for i, corner in enumerate(["FL", "FR", "RL", "RR"]):
            options[f"Tire Load {corner}"] = {
                "data": result.tire_loads[:, i],
                "unit": "N",
                "colorscale": cs,
            }
    if result.energy_storage is not None:
        options["Energy Storage"] = {
            "data": result.energy_storage,
            "unit": "kJ",
            "colorscale": cs,
        }
    if result.fuel_consumed_profile is not None:
        options["Fuel Consumed"] = {
            "data": result.fuel_consumed_profile,
            "unit": "kg",
            "colorscale": cs,
        }
    if result.energy_consumed_profile is not None:
        options["Energy Consumed"] = {
            "data": result.energy_consumed_profile,
            "unit": "kJ",
            "colorscale": cs,
        }
    if result.drs is not None:
        drs_label = "Active Aero" if result.active_aero else "DRS"
        options[drs_label] = {
            "data": result.drs.astype(float),
            "unit": "",
            "colorscale": cs,
        }
    if result.friction is not None:
        options["Friction"] = {"data": result.friction, "unit": "μ", "colorscale": cs}
    if result.e_motor_power is not None:
        options["E-Motor Power"] = {
            "data": result.e_motor_power,
            "unit": "kW",
            "colorscale": cs,
        }
    if result.harvest_power is not None:
        options["Harvest Power"] = {
            "data": result.harvest_power,
            "unit": "kW",
            "colorscale": cs,
        }
    if result.harvest_energy_profile is not None:
        options["Harvest Energy"] = {
            "data": result.harvest_energy_profile,
            "unit": "kJ",
            "colorscale": cs,
        }

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
    # Streamlit recreates the component iframe on every rerun (e.g. switching the
    # selectbox below); the empty iframe box defaults to a white background until
    # its document paints, causing a white flash on a dark theme. Force it transparent.
    st.markdown(
        "<style>iframe{background-color:transparent!important;}</style>",
        unsafe_allow_html=True,
    )

    # All visualization options are embedded in a single payload and switching
    # happens via client-side Plotly calls (see applyViz() below) rather than a
    # Streamlit rerun. A Streamlit-triggered rerun would hand the iframe a new
    # srcdoc, forcing it to reload/renavigate — Firefox paints that transitional
    # navigation opaque white before the new document's CSS/background applies,
    # producing a white flash (Chromium happens to mask it). Never reloading the
    # iframe at all avoids the flash in every browser, not just Chromium.
    viz_options = get_viz_options(result)
    default_viz = next(iter(viz_options))

    # Downsample to ~10 m resolution for track rendering (geometry is shared by
    # every visualization option)
    n_pts = len(result.track_x)
    step_m = result.distance[-1] / n_pts if n_pts > 1 else 1.0
    ds = max(1, round(10.0 / step_m))
    tx = result.track_x[::ds].tolist()
    ty = result.track_y[::ds].tolist()
    dist = result.distance[::ds].tolist()

    options_payload = {}
    for name, opt in viz_options.items():
        viz_data = opt["data"]
        viz_unit = opt["unit"]

        data_min = float(np.min(viz_data))
        data_max = float(np.max(viz_data))
        if data_max - data_min > 0:
            data_normalized = (viz_data - data_min) / (data_max - data_min)
        else:
            data_normalized = np.zeros_like(viz_data)

        options_payload[name] = {
            "prof_data": viz_data.tolist(),
            "vd": viz_data[::ds].tolist(),
            "color_vals": data_normalized[::ds].tolist(),
            "colorscale": opt["colorscale"],
            "y_label": f"{name} [{viz_unit}]" if viz_unit else name,
            "unit_str": f" {viz_unit}" if viz_unit else "",
            "viz_unit": viz_unit,
            "data_min": data_min,
            "data_max": data_max,
        }

    sector_dists = result.sector_distances if result.sector_distances else []
    sector_x = [xy[0] for xy in result.sector_xy] if result.sector_xy else []
    sector_y = [xy[1] for xy in result.sector_xy] if result.sector_xy else []

    option_tags = "".join(
        f'<option value="{name}">{name}</option>' for name in viz_options.keys()
    )

    payload = json.dumps(
        {
            "prof_dist": result.distance.tolist(),
            "track_x": tx,
            "track_y": ty,
            "dist": dist,
            "default_viz": default_viz,
            "options": options_payload,
            "sector_dists": sector_dists,
            "sector_x": sector_x,
            "sector_y": sector_y,
        }
    )

    html = f"""<!DOCTYPE html>
<html><head>
<script>{_get_plotly_js()}</script>
<style>
  html, body {{ margin:0; padding:0; background:transparent; font-family:"Source Sans Pro",sans-serif; }}
  #controls {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
  #controls label {{ color:#fff; font-size:14px; }}
  #legend {{ color:#bbb; font-size:13px; margin-top:4px; }}
  #viz-select {{
    background:#262730; color:#fff; border:1px solid #4a4a52; border-radius:6px;
    padding:4px 8px; font-size:14px; font-family:inherit;
  }}
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
<div id="controls">
  <label for="viz-select">Display</label>
  <select id="viz-select">{option_tags}</select>
</div>
<div id="wrap">
  <div id="profile"></div>
  <div id="trackmap"></div>
</div>
<div id="legend"></div>
<script>
const D = {payload};
let currentViz = D.default_viz;
document.getElementById('viz-select').value = currentViz;
const O = D.options[currentViz];

// ── Profile chart ──────────────────────────────────────────────
const sectorLabels = ['S1|S2', 'S2|S3'];
const sectorColors = ['#f0e040', '#40e0f0'];
const profileShapes = [{{
  type: 'line', xref: 'x', yref: 'paper',
  x0: 0, x1: 0, y0: 0, y1: 1,
  line: {{color: 'red', width: 1, dash: 'dot'}},
  visible: false,
}}];
const profileAnnotations = [];
(D.sector_dists || []).forEach(function(d, i) {{
  profileShapes.push({{
    type: 'line', xref: 'x', yref: 'paper',
    x0: d, x1: d, y0: 0, y1: 1,
    line: {{color: sectorColors[i], width: 1, dash: 'dash'}},
  }});
  profileAnnotations.push({{
    xref: 'x', yref: 'paper',
    x: d, y: 1,
    text: sectorLabels[i],
    showarrow: false,
    xanchor: 'center', yanchor: 'bottom',
    font: {{color: sectorColors[i], size: 11}},
  }});
}});

Plotly.newPlot('profile', [{{
  x: D.prof_dist, y: O.prof_data,
  type: 'scatter', mode: 'lines',
  fill: 'tozeroy',
  fillgradient: {{type: 'vertical', colorscale: [[0, 'rgba(31,119,180,0)'], [1, 'rgba(31,119,180,0.45)']]}},
  line: {{color: '#1f77b4', width: 2}},
  hovertemplate: '<b>%{{x:.0f}} m</b><br>' + currentViz + ': %{{y:.2f}}' + O.unit_str + '<extra></extra>',
}}], {{
  xaxis: {{title: {{text: 'Distance [m]', font: {{color: '#fff'}}}},
           tickfont: {{color: '#fff'}},
           showgrid: false, zeroline: false,
           showspikes: true, spikemode: 'across', spikesnap: 'cursor',
           spikecolor: '#FF6B00', spikethickness: 1, spikedash: 'solid'}},
  yaxis: {{title: {{text: O.y_label, font: {{color: '#fff'}}}},
           tickfont: {{color: '#fff'}},
           showgrid: true, gridcolor: 'rgba(128,128,128,0.15)', zeroline: false}},
  hovermode: 'x unified',
  hoverlabel: {{bgcolor: '#FF6B00', font: {{color: 'white'}}, bordercolor: '#FF6B00'}},
  height: 400,
  margin: {{l:60, r:20, t:20, b:50}},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  shapes: profileShapes,
  annotations: profileAnnotations,
}}, {{responsive: true, displayModeBar: false}});

// ── Track map ──────────────────────────────────────────────────
function mapHoverFmt(name, unit) {{
  return unit
    ? '<b>%{{customdata[0]:.0f}} m</b><br>' + name + ': %{{customdata[1]:.2f}} ' + unit + '<extra></extra>'
    : '<b>%{{customdata[0]:.0f}} m</b><br>' + name + ': %{{customdata[1]:.2f}}<extra></extra>';
}}

// Single colored-marker trace — one trace instead of N segment traces
const segs = [{{
  x: D.track_x, y: D.track_y,
  type: 'scatter', mode: 'markers',
  marker: {{
    color: O.color_vals, colorscale: O.colorscale,
    cmin: 0, cmax: 1, size: 10, showscale: false,
  }},
  customdata: D.track_x.map((_, i) => [D.dist[i], O.vd[i]]),
  hovertemplate: mapHoverFmt(currentViz, O.viz_unit),
  showlegend: false,
}}];

// Start/finish marker
segs.push({{
  x: [D.track_x[0]], y: [D.track_y[0]],
  type: 'scatter', mode: 'markers',
  marker: {{size: 12, color: 'white', line: {{color: 'black', width: 2}}}},
  showlegend: false, hoverinfo: 'skip',
}});

// Sector boundary markers
if (D.sector_x && D.sector_x.length) {{
  segs.push({{
    x: D.sector_x, y: D.sector_y,
    type: 'scatter', mode: 'markers',
    marker: {{size: 10, color: sectorColors, line: {{color: 'black', width: 1}}}},
    showlegend: false, hoverinfo: 'skip',
  }});
}}

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

function setLegend(o) {{
  const el = document.getElementById('legend');
  el.textContent = o.viz_unit
    ? 'Color: Low (' + o.data_min.toFixed(1) + ' ' + o.viz_unit + ') → High (' + o.data_max.toFixed(1) + ' ' + o.viz_unit + ')'
    : 'Color: Low (' + o.data_min.toFixed(0) + ') → High (' + o.data_max.toFixed(0) + ')';
}}
setLegend(O);

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

// ── Visualization switching (in-place, no iframe reload) ────────
function applyViz(name) {{
  currentViz = name;
  const o = D.options[name];

  Plotly.update('profile',
    {{y: [o.prof_data], hovertemplate: ['<b>%{{x:.0f}} m</b><br>' + name + ': %{{y:.2f}}' + o.unit_str + '<extra></extra>']}},
    {{'yaxis.title.text': o.y_label}},
    [0]
  );

  Plotly.restyle('trackmap', {{
    'marker.color': [o.color_vals],
    'marker.colorscale': [o.colorscale],
    customdata: [D.track_x.map((_, i) => [D.dist[i], o.vd[i]])],
    hovertemplate: [mapHoverFmt(name, o.viz_unit)],
  }}, [0]);

  setLegend(o);
}}

document.getElementById('viz-select').addEventListener('change', function(ev) {{
  applyViz(ev.target.value);
}});
</script>
</body></html>"""

    # 400px charts + ~40px controls row (moved in-iframe from st.selectbox) + ~25px legend
    components.html(html, height=470)


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
    fig.add_trace(
        go.Scatter(
            x=distance,
            y=data,
            mode="lines",
            name=name,
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    _axis_font = dict(color="#fff")
    y_label = f"{name} [{unit}]" if unit else name
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Distance [m]", font=_axis_font), tickfont=_axis_font
        ),
        yaxis=dict(title=dict(text=y_label, font=_axis_font), tickfont=_axis_font),
        hovermode="x unified",
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
        fig.add_trace(
            go.Scatter(
                x=track_x[i : i + 2],
                y=track_y[i : i + 2],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

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

        fig.add_trace(
            go.Scatter(
                x=track_x,
                y=track_y,
                mode="markers",
                marker=dict(size=10, opacity=0),
                customdata=custom,
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    # Add start/finish marker
    fig.add_trace(
        go.Scatter(
            x=[track_x[0]],
            y=[track_y[0]],
            mode="markers",
            marker=dict(size=12, color="white", line=dict(color="black", width=2)),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    _axis_font = dict(color="#fff")
    fig.update_layout(
        xaxis=dict(title=dict(text="X [m]", font=_axis_font), tickfont=_axis_font),
        yaxis=dict(
            title=dict(text="Y [m]", font=_axis_font),
            tickfont=_axis_font,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=height,
        showlegend=False,
    )
    return fig
