"""
Drag Racing Page

Standing-start acceleration with fully editable vehicle parameters.
Select a vehicle to load its defaults, tweak any value, then run.
"""

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from helpers.simulation import (
    get_available_vehicles,
    read_vehicle_params,
    run_drag_simulation,
)
from laptimesim.src.drag_test import DRAG_DISTANCES

st.set_page_config(
    page_title="Drag Racing - Laptime Sim",
    page_icon="🏁",
    layout="wide",
)

st.title("🏁 Drag Run")
st.caption(
    "Load a vehicle, adjust any parameter, then run a standing-start acceleration test."
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

for _k, _v in [
    ("drag_result", None),
    ("drag_saved", []),
    ("drag_vehicle", None),
    ("drag_param_ver", 0),
    ("drag_raw", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

MAX_SAVED   = 3
RUN_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Run Controls")

available_vehicles = get_available_vehicles()
default_idx = (
    available_vehicles.index("Zapovic_Breeze")
    if "Zapovic_Breeze" in available_vehicles
    else 0
)
vehicle = st.sidebar.selectbox("Vehicle", available_vehicles, index=default_idx)

if st.sidebar.button("↺ Reset parameters to defaults", use_container_width=True):
    # Force a param version bump without changing the vehicle
    st.session_state.drag_param_ver += 1
    st.session_state.drag_raw = read_vehicle_params(vehicle)

st.sidebar.divider()

if st.session_state.drag_saved:
    st.sidebar.subheader("Saved runs")
    for i, run in enumerate(st.session_state.drag_saved):
        st.sidebar.caption(f"{i + 1}. {run['vehicle']}  μ={run['mu']:.2f}")
    if st.sidebar.button("🗑️ Clear saved runs", use_container_width=True):
        st.session_state.drag_saved = []
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Detect vehicle change → reload defaults
# ──────────────────────────────────────────────────────────────────────────────

if vehicle != st.session_state.drag_vehicle:
    st.session_state.drag_vehicle  = vehicle
    st.session_state.drag_param_ver += 1
    st.session_state.drag_raw = read_vehicle_params(vehicle)

ver  = st.session_state.drag_param_ver
raw  = st.session_state.drag_raw

ptype = raw.get("powertrain_type", "electric")
gen   = raw["general"]
eng   = raw["engine"]
gbx   = raw["gearbox"]
tires = raw["tires"]

# ──────────────────────────────────────────────────────────────────────────────
# Parameter editor
# ──────────────────────────────────────────────────────────────────────────────

st.header("Vehicle Parameters")

p        = {}        # collects every widget's return value
g_edited = None      # gear data-editor return value

tab_gen, tab_eng, tab_gbx, tab_tire = st.tabs(
    ["🚗  Chassis & Aero", "⚡  Powertrain", "⚙️  Gearbox", "🔵  Tires"]
)

# ── Chassis & Aero ────────────────────────────────────────────────────────────
with tab_gen:
    st.subheader("Geometry")
    c1, c2, c3 = st.columns(3)
    p["lf"]    = c1.number_input("Front axle → CoG  (lf) [m]",   value=float(gen["lf"]),    step=0.001, format="%.3f", key=f"p_lf_{ver}")
    p["lr"]    = c2.number_input("Rear axle → CoG   (lr) [m]",   value=float(gen["lr"]),    step=0.001, format="%.3f", key=f"p_lr_{ver}")
    p["h_cog"] = c3.number_input("CoG height        (h_cog) [m]", value=float(gen["h_cog"]), step=0.001, format="%.3f", key=f"p_h_cog_{ver}")

    c1, c2, c3 = st.columns(3)
    p["sf"]    = c1.number_input("Front track width  (sf) [m]", value=float(gen["sf"]),  step=0.001, format="%.3f", key=f"p_sf_{ver}")
    p["sr"]    = c2.number_input("Rear track width   (sr) [m]", value=float(gen["sr"]),  step=0.001, format="%.3f", key=f"p_sr_{ver}")
    p["m"]     = c3.number_input("Vehicle mass       (m) [kg]", value=float(gen["m"]),   step=1.0,   format="%.1f", key=f"p_m_{ver}")

    st.subheader("Aerodynamics")
    c1, c2, c3 = st.columns(3)
    p["c_w_a"]   = c1.number_input("Drag area      c_w·A  [m²]",       value=float(gen["c_w_a"]),   step=0.01, format="%.3f", key=f"p_cwa_{ver}")
    p["c_z_a_f"] = c2.number_input("Front DnF area c_z_f·A [m²]",      value=float(gen["c_z_a_f"]), step=0.01, format="%.3f", key=f"p_czaf_{ver}")
    p["c_z_a_r"] = c3.number_input("Rear DnF area  c_z_r·A [m²]",      value=float(gen["c_z_a_r"]), step=0.01, format="%.3f", key=f"p_czar_{ver}")

    c1, c2, c3 = st.columns(3)
    p["f_roll"]   = c1.number_input("Rolling resistance  (f_roll) [-]", value=float(gen["f_roll"]),   step=0.001, format="%.4f", key=f"p_froll_{ver}")
    p["g"]        = c2.number_input("Gravity             (g) [m/s²]",   value=float(gen["g"]),        step=0.01,  format="%.2f", key=f"p_g_{ver}")
    p["rho_air"]  = c3.number_input("Air density         (ρ) [kg/m³]",  value=float(gen["rho_air"]),  step=0.01,  format="%.3f", key=f"p_rho_{ver}")

    if "drs_factor" in gen:
        p["drs_factor"] = st.number_input(
            "DRS drag reduction factor [-]",
            value=float(gen["drs_factor"]), step=0.01, format="%.3f", key=f"p_drs_{ver}"
        )

    has_aa = any(k in gen for k in ("active_aero_drag_reduction", "active_aero_dz_f", "active_aero_dz_r"))
    if has_aa:
        st.subheader("Active Aerodynamics")
        c1, c2, c3 = st.columns(3)
        if "active_aero_drag_reduction" in gen:
            p["active_aero_drag_reduction"] = c1.number_input(
                "Drag reduction when deployed [-]",
                value=float(gen["active_aero_drag_reduction"]), step=0.01, format="%.2f", key=f"p_aa_drag_{ver}"
            )
        if "active_aero_dz_f" in gen:
            p["active_aero_dz_f"] = c2.number_input(
                "Front downforce reduction [-]",
                value=float(gen["active_aero_dz_f"]), step=0.01, format="%.2f", key=f"p_aa_dzf_{ver}"
            )
        if "active_aero_dz_r" in gen:
            p["active_aero_dz_r"] = c3.number_input(
                "Rear downforce reduction [-]",
                value=float(gen["active_aero_dz_r"]), step=0.01, format="%.2f", key=f"p_aa_dzr_{ver}"
            )

# ── Powertrain ────────────────────────────────────────────────────────────────
with tab_eng:
    c1, c2 = st.columns(2)
    topo_opts = ["AWD", "RWD", "FWD"]
    topo_val  = eng.get("topology", "RWD")
    p["topology"] = c1.selectbox(
        "Drive topology", topo_opts,
        index=topo_opts.index(topo_val) if topo_val in topo_opts else 1,
        key=f"p_topo_{ver}"
    )
    p["series"] = c2.text_input("Series", value=eng.get("series", ""), key=f"p_series_{ver}")

    # ICE (hybrid only)
    if "pow_max" in eng:
        st.subheader("Internal Combustion Engine")
        c1, c2 = st.columns(2)
        p["pow_max"]  = c1.number_input("Max ICE power [kW]",              value=float(eng["pow_max"]) / 1e3,  step=1.0,   format="%.1f", key=f"p_pow_max_{ver}")
        p["pow_diff"] = c2.number_input("Power drop at band edges [kW]",   value=float(eng["pow_diff"]) / 1e3, step=1.0,   format="%.1f", key=f"p_pow_diff_{ver}")
        c1, c2, c3 = st.columns(3)
        p["n_begin"]  = c1.number_input("n_begin [rpm]",                   value=float(eng["n_begin"]),         step=100.0, format="%.0f", key=f"p_n_begin_{ver}")
        p["n_max"]    = c2.number_input("n_max [rpm]",                     value=float(eng["n_max"]),           step=100.0, format="%.0f", key=f"p_n_max_{ver}")
        p["n_end"]    = c3.number_input("n_end [rpm]",                     value=float(eng["n_end"]),           step=100.0, format="%.0f", key=f"p_n_end_{ver}")
        p["be_max"]   = st.number_input("Max fuel flow [kg/h]",            value=float(eng["be_max"]),          step=1.0,   format="%.1f", key=f"p_be_max_{ver}")

    # Electric motors
    is_dual = "pow_e_motor_f" in eng
    if is_dual:
        st.subheader("Electric Motors — Front & Rear")
        c1, c2 = st.columns(2)
        p["pow_e_motor_f"]      = c1.number_input("Front motor power [kW]",      value=float(eng["pow_e_motor_f"]) / 1e3,      step=1.0,  format="%.1f", key=f"p_pow_f_{ver}")
        p["pow_e_motor_r"]      = c2.number_input("Rear motor power [kW]",       value=float(eng["pow_e_motor_r"]) / 1e3,      step=1.0,  format="%.1f", key=f"p_pow_r_{ver}")
        c1, c2 = st.columns(2)
        p["torque_e_motor_max_f"] = c1.number_input("Front motor max torque [Nm]", value=float(eng["torque_e_motor_max_f"]), step=10.0, format="%.0f", key=f"p_tq_f_{ver}")
        p["torque_e_motor_max_r"] = c2.number_input("Rear motor max torque [Nm]",  value=float(eng["torque_e_motor_max_r"]), step=10.0, format="%.0f", key=f"p_tq_r_{ver}")
    elif "pow_e_motor" in eng:
        st.subheader("Electric Motor")
        c1, c2 = st.columns(2)
        p["pow_e_motor"]      = c1.number_input("Motor power [kW]",     value=float(eng["pow_e_motor"]) / 1e3,      step=1.0,  format="%.1f", key=f"p_pow_em_{ver}")
        p["torque_e_motor_max"] = c2.number_input("Max torque [Nm]",    value=float(eng["torque_e_motor_max"]), step=10.0, format="%.0f", key=f"p_tq_em_{ver}")

    if "pow_e_motor" in eng or is_dual:
        st.subheader("Motor Efficiency")
        c1, c2 = st.columns(2)
        p["eta_e_motor"]    = c1.number_input("Drive efficiency  (η_drive) [-]", value=float(eng["eta_e_motor"]),    step=0.01, format="%.3f", key=f"p_eta_em_{ver}")
        p["eta_e_motor_re"] = c2.number_input("Regen efficiency  (η_regen) [-]", value=float(eng["eta_e_motor_re"]), step=0.01, format="%.3f", key=f"p_eta_re_{ver}")

    if "eta_etc_re" in eng:
        p["eta_etc_re"] = st.number_input("ETC recuperation efficiency [-]", value=float(eng["eta_etc_re"]), step=0.01, format="%.3f", key=f"p_eta_etc_{ver}")
    if "vel_min_e_motor" in eng:
        p["vel_min_e_motor"] = st.number_input("Min speed for e-motor [m/s]", value=float(eng["vel_min_e_motor"]), step=1.0, format="%.2f", key=f"p_vel_min_{ver}")
    if "vel_lim_glob" in eng:
        p["vel_lim_glob"] = st.number_input("Global velocity limit [m/s]", value=float(eng["vel_lim_glob"]), step=1.0, format="%.2f", key=f"p_vel_lim_{ver}")

    # ERS regulations (F1 2026 style)
    if "ers_speed_limit" in eng:
        st.subheader("ERS Regulations")
        c1, c2, c3 = st.columns(3)
        p["ers_speed_limit"]     = c1.checkbox("ERS speed limit (FIA C5.2.8)", value=bool(eng["ers_speed_limit"]), key=f"p_ers_{ver}")
        p["max_e_energy_storage"] = c2.number_input("Max battery [MJ]",            value=float(eng["max_e_energy_storage"]) / 1e6, step=0.1, format="%.2f", key=f"p_es_{ver}")
        p["e_rec_e_motor_max"]   = c3.number_input("Max recovery per lap [MJ]",    value=float(eng["e_rec_e_motor_max"]) / 1e6,    step=0.1, format="%.2f", key=f"p_rec_{ver}")

# ── Gearbox ───────────────────────────────────────────────────────────────────
with tab_gbx:
    c1, c2 = st.columns(2)
    p["eta_g"] = c1.number_input(
        "Gearbox efficiency (η_g) [-]",
        value=float(gbx["eta_g"]), step=0.001, format="%.4f", key=f"p_eta_g_{ver}"
    )
    if "diff_lock_ratio" in gbx:
        p["diff_lock_ratio"] = c2.number_input(
            "Diff lock ratio [-]  (0 = open, 1 = locked)",
            value=float(gbx["diff_lock_ratio"]), step=0.05, format="%.2f", key=f"p_diff_{ver}"
        )
    if "t_shift" in gbx:
        p["t_shift"] = st.number_input(
            "Gear shift time [s]",
            value=float(gbx["t_shift"]), step=0.001, format="%.4f", key=f"p_tshift_{ver}"
        )

    st.subheader("Gear Ratios")
    st.caption(
        "**i_trans** is from the tire to the motor shaft — a smaller value means a taller gear. "
        "**n_shift** is the motor speed at which an upshift occurs [rpm]."
    )
    n_gears  = len(gbx["i_trans"])
    gear_df  = pd.DataFrame(
        {
            "i_trans":        [float(x) for x in gbx["i_trans"]],
            "n_shift [rpm]":  [float(x) for x in gbx["n_shift"]],
            "e_i":            [float(x) for x in gbx["e_i"]],
        },
        index=pd.Index([f"Gear {i + 1}" for i in range(n_gears)], name="Gear"),
    )
    g_edited = st.data_editor(
        gear_df,
        key=f"p_gear_{ver}",
        use_container_width=True,
        column_config={
            "i_trans":       st.column_config.NumberColumn("Ratio (i_trans) [-]",       format="%.5f", step=0.001),
            "n_shift [rpm]": st.column_config.NumberColumn("Shift speed [rpm]",         format="%.0f", step=100.0),
            "e_i":           st.column_config.NumberColumn("Inertia factor (e_i) [-]",  format="%.3f", step=0.01),
        },
    )

# ── Tires ─────────────────────────────────────────────────────────────────────
with tab_tire:
    tire_fields = [
        ("circ_ref", "Reference circumference [m]",    0.001,  "%.4f"),
        ("fz_0",     "Nominal tire load [N]",           100.0,  "%.0f"),
        ("mux",      "Longitudinal friction (μₓ) [-]",  0.01,   "%.3f"),
        ("muy",      "Lateral friction      (μᵧ) [-]",  0.01,   "%.3f"),
        ("dmux_dfz", "μₓ load sensitivity [1/N]",        1e-6,   "%.2e"),
        ("dmuy_dfz", "μᵧ load sensitivity [1/N]",        1e-6,   "%.2e"),
    ]

    col_f, col_r = st.columns(2)
    with col_f:
        st.subheader("Front")
        for key, label, step, fmt in tire_fields:
            p[f"tf_{key}"] = st.number_input(
                label, value=float(tires["f"][key]),
                step=step, format=fmt, key=f"p_tf_{key}_{ver}"
            )
    with col_r:
        st.subheader("Rear")
        for key, label, step, fmt in tire_fields:
            p[f"tr_{key}"] = st.number_input(
                label, value=float(tires["r"][key]),
                step=step, format=fmt, key=f"p_tr_{key}_{ver}"
            )

    p["tire_model_exp"] = st.number_input(
        "Friction circle exponent [-]  (1.0 = ellipse, 2.0 = circle)",
        value=float(tires["tire_model_exp"]), step=0.1, format="%.1f", key=f"p_texp_{ver}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Build modified params dict from widget values
# ──────────────────────────────────────────────────────────────────────────────

def _build_custom_pars(p: dict, g_edited, raw: dict) -> dict:
    """Reconstruct a full params dict from widget values, starting from a deep copy of raw."""
    custom = copy.deepcopy(raw)

    # General
    for key in ("lf", "lr", "h_cog", "sf", "sr", "m", "f_roll",
                "c_w_a", "c_z_a_f", "c_z_a_r", "g", "rho_air",
                "drs_factor",
                "active_aero_drag_reduction", "active_aero_dz_f", "active_aero_dz_r"):
        if key in p:
            custom["general"][key] = p[key]

    # Engine — topology & series
    custom["engine"]["topology"] = p["topology"]
    custom["engine"]["series"]   = p["series"]

    # ICE (kW → W in UI; rpm and kg/h stored as-is, constructor converts)
    for kw_key, w_key in [("pow_max", "pow_max"), ("pow_diff", "pow_diff")]:
        if w_key in p:
            custom["engine"][kw_key] = p[w_key] * 1e3
    for key in ("n_begin", "n_max", "n_end", "be_max"):
        if key in p:
            custom["engine"][key] = p[key]

    # Electric dual motor (kW → W)
    for kw_key, w_key in [("pow_e_motor_f", "pow_e_motor_f"), ("pow_e_motor_r", "pow_e_motor_r")]:
        if w_key in p:
            custom["engine"][kw_key] = p[w_key] * 1e3
    for key in ("torque_e_motor_max_f", "torque_e_motor_max_r"):
        if key in p:
            custom["engine"][key] = p[key]

    # Electric single motor (kW → W)
    if "pow_e_motor" in p:
        custom["engine"]["pow_e_motor"] = p["pow_e_motor"] * 1e3
    if "torque_e_motor_max" in p:
        custom["engine"]["torque_e_motor_max"] = p["torque_e_motor_max"]

    # Shared motor params
    for key in ("eta_e_motor", "eta_e_motor_re", "eta_etc_re",
                "vel_min_e_motor", "vel_lim_glob"):
        if key in p:
            custom["engine"][key] = p[key]

    # ERS (displayed in MJ → stored in J)
    if "ers_speed_limit" in p:
        custom["engine"]["ers_speed_limit"] = p["ers_speed_limit"]
    if "max_e_energy_storage" in p:
        custom["engine"]["max_e_energy_storage"] = p["max_e_energy_storage"] * 1e6
    if "e_rec_e_motor_max" in p:
        custom["engine"]["e_rec_e_motor_max"] = p["e_rec_e_motor_max"] * 1e6

    # Gearbox scalars
    custom["gearbox"]["eta_g"] = p["eta_g"]
    for key in ("diff_lock_ratio", "t_shift"):
        if key in p:
            custom["gearbox"][key] = p[key]

    # Gear table (n_shift in rpm — constructor converts to 1/s)
    if g_edited is not None:
        custom["gearbox"]["i_trans"] = g_edited["i_trans"].tolist()
        custom["gearbox"]["n_shift"] = g_edited["n_shift [rpm]"].tolist()
        custom["gearbox"]["e_i"]     = g_edited["e_i"].tolist()

    # Tires
    for ax, prefix in [("f", "tf_"), ("r", "tr_")]:
        for key in ("circ_ref", "fz_0", "mux", "muy", "dmux_dfz", "dmuy_dfz"):
            wk = f"{prefix}{key}"
            if wk in p:
                custom["tires"][ax][key] = p[wk]
    custom["tires"]["tire_model_exp"] = p["tire_model_exp"]

    return custom


# ──────────────────────────────────────────────────────────────────────────────
# Test conditions & run button (in main window)
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.header("Test Conditions")

mu_col, btn_col = st.columns([3, 1], vertical_alignment="bottom")
mu = mu_col.slider(
    "Track Grip (μ)",
    min_value=0.6, max_value=1.3, value=1.0, step=0.05, format="%.2f",
    help="1.0 = dry, 0.6 = wet, 1.2+ = slick / very sticky",
)
grip_label = (
    "🌧️ Wet"      if mu < 0.75  else
    "🌦️ Damp"     if mu < 0.95  else
    "☀️ Dry"       if mu <= 1.05 else
    "🏁 High grip"
)
mu_col.caption(f"Conditions: **{grip_label}**")
run_btn = btn_col.button(
    "🚀 Run Drag Pull", type="primary", use_container_width=True
)

# ──────────────────────────────────────────────────────────────────────────────
# Run simulation
# ──────────────────────────────────────────────────────────────────────────────

if run_btn:
    custom_pars = _build_custom_pars(p, g_edited, raw)
    with st.spinner(f"Running drag test for {vehicle}…"):
        try:
            run_data = run_drag_simulation(vehicle, mu, custom_pars=custom_pars)
            st.session_state.drag_result = run_data
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.exception(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

def _interp_at_dist(dist_arr, t_arr, vel_arr, d_target):
    idx = np.searchsorted(dist_arr, d_target)
    if idx == 0 or idx >= len(dist_arr):
        return None, None
    a = (d_target - dist_arr[idx - 1]) / (dist_arr[idx] - dist_arr[idx - 1])
    return (t_arr[idx - 1] + a * (t_arr[idx] - t_arr[idx - 1]),
            vel_arr[idx - 1] + a * (vel_arr[idx] - vel_arr[idx - 1]))


def _interp_at_vel(vel_arr, t_arr, v_target):
    idx = np.searchsorted(vel_arr, v_target)
    if idx == 0 or idx >= len(vel_arr):
        return None
    a = (v_target - vel_arr[idx - 1]) / (vel_arr[idx] - vel_arr[idx - 1])
    return t_arr[idx - 1] + a * (t_arr[idx] - t_arr[idx - 1])


def _velocity_chart(runs: list) -> go.Figure:
    fig = go.Figure()
    for i, run in enumerate(runs):
        r   = run["result"]
        col = RUN_COLORS[i % len(RUN_COLORS)]
        lbl = f"{run['vehicle']}  μ={run['mu']:.2f}"
        fig.add_trace(go.Scatter(
            x=r["dist"], y=r["vel"] * 3.6,
            mode="lines", name=lbl,
            line=dict(color=col, width=2.5),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(31,119,180,0.12)" if i == 0 else None,
            hovertemplate="Distance: %{x:.1f} m<br>Speed: %{y:.1f} km/h<extra>" + lbl + "</extra>",
        ))

    r0 = runs[0]["result"]
    for name, d_m in DRAG_DISTANCES.items():
        t_v, v_v = _interp_at_dist(r0["dist"], r0["t"], r0["vel"], d_m)
        ann = (f"<b>{name}</b><br>{t_v:.3f} s<br>{v_v * 3.6:.0f} km/h" if t_v else name)
        fig.add_vline(x=d_m, line=dict(color="#6c757d", dash="dash", width=1.2))
        fig.add_annotation(x=d_m, y=1.0, yref="paper", text=ann,
                           showarrow=False, xanchor="left", xshift=6, yanchor="top",
                           font=dict(size=11, color="#6c757d"),
                           bgcolor="rgba(255,255,255,0.75)")

    fig.update_layout(
        xaxis_title="Distance [m]", yaxis_title="Speed [km/h]",
        hovermode="x unified", height=420, margin=dict(t=20, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _detail_charts(run: dict) -> go.Figure:
    r   = run["result"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Acceleration [m/s²]", "Gear"),
                        row_heights=[0.6, 0.4])
    fig.add_trace(go.Scatter(
        x=r["dist"], y=r["a_x"], mode="lines",
        line=dict(color="#ff7f0e", width=2),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.15)",
        hovertemplate="%{y:.2f} m/s²<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=r["dist"], y=r["gear"] + 1, mode="lines",
        line=dict(color="#2ca02c", width=2, shape="hv"),
        hovertemplate="Gear %{y}<extra></extra>",
    ), row=2, col=1)
    for d_m in DRAG_DISTANCES.values():
        for row in (1, 2):
            fig.add_vline(x=d_m, line=dict(color="#6c757d", dash="dash", width=1.0),
                          row=row, col=1)
    fig.update_yaxes(title_text="m/s²", row=1, col=1)
    fig.update_yaxes(title_text="Gear", tickvals=list(range(1, len(r["gear"]) + 2)), row=2, col=1)
    fig.update_xaxes(title_text="Distance [m]", row=2, col=1)
    fig.update_layout(height=380, margin=dict(t=30, b=40), showlegend=False)
    return fig


if st.session_state.drag_result is not None:
    run = st.session_state.drag_result
    r   = run["result"]

    st.divider()
    st.header("Results")
    st.subheader(f"{run['vehicle']}  ·  {run['series']}")
    st.caption(
        f"Mass: **{run['mass_kg']:.0f} kg**  ·  "
        f"Power: **{run['power_kw']:.0f} kW**  ·  "
        f"Topology: **{run['topology']}**  ·  "
        f"μ = **{run['mu']:.2f}**"
    )

    # Save to compare
    col_save, _ = st.columns([1, 4])
    with col_save:
        if len(st.session_state.drag_saved) >= MAX_SAVED:
            st.warning(f"Max {MAX_SAVED} saved. Clear in the sidebar.")
        else:
            if st.button("💾 Save to compare", use_container_width=True):
                st.session_state.drag_saved.append(run)
                st.success(f"Saved ({len(st.session_state.drag_saved)}/{MAX_SAVED})")

    st.divider()

    # Speed benchmarks
    speed_cols = st.columns(3)
    for col, (lbl, v_t) in zip(speed_cols, [
        ("0–60 mph",   60.0 * 1.60934 / 3.6),
        ("0–100 km/h", 100.0 / 3.6),
        ("0–200 km/h", 200.0 / 3.6),
    ]):
        t_v = _interp_at_vel(r["vel"], r["t"], v_t)
        col.metric(lbl, f"{t_v:.2f} s" if t_v is not None else "—")

    st.divider()

    # Distance milestones
    dist_cols = st.columns(3)
    for col, (lbl, d_m) in zip(dist_cols, [
        ("⅛ Mile",  DRAG_DISTANCES["1/8 mile"]),
        ("¼ Mile",  DRAG_DISTANCES["1/4 mile"]),
        ("1 km",    DRAG_DISTANCES["1 km"]),
    ]):
        t_v, v_v = _interp_at_dist(r["dist"], r["t"], r["vel"], d_m)
        if t_v is not None:
            col.metric(lbl, f"{t_v:.3f} s", f"{v_v * 3.6:.1f} km/h trap", delta_color="off")
        else:
            col.metric(lbl, "—")

    st.divider()

    # Charts — overlay all saved runs for comparison
    chart_runs = list(st.session_state.drag_saved)
    if not any(r_ is run for r_ in chart_runs):
        chart_runs = [run] + chart_runs

    st.subheader("Speed vs Distance")
    st.plotly_chart(_velocity_chart(chart_runs), use_container_width=True)

    st.subheader("Acceleration & Gear")
    st.plotly_chart(_detail_charts(run), use_container_width=True)
