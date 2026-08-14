"""
Comparison helpers for saved simulation runs.

Runs are kept in st.session_state.saved_runs (a list of SimulationResult) so every
page can save into the same store and render the same comparison view.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from helpers.simulation import SimulationResult

MAX_RUNS = 3
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green


def saved_runs() -> list[SimulationResult]:
    """Return the saved-run store, creating it on first use."""
    if "saved_runs" not in st.session_state:
        st.session_state.saved_runs = []
    return st.session_state.saved_runs


def run_label(run: SimulationResult) -> str:
    """Short description of a run for legends and headings.

    Pages that vary vehicle parameters instead of picking a vehicle file (e.g. the MVRC
    page, which passes custom parameters and therefore has no vehicle name) set
    'label' on the result; everything else falls back to the vehicle/weather metadata.
    """
    label = getattr(run, "label", None)
    if label:
        return label

    return f"{run.vehicle} - {run.weather}"


_SAVED_MSG = "_comparison_saved_msg"


def save_run_button(
    result: SimulationResult, label: str | None = None, key: str | None = None
) -> None:
    """Render the 'Save to Compare' button (or the capacity warning when full)."""
    runs = saved_runs()

    # confirmation of the save that triggered the rerun below
    msg = st.session_state.pop(_SAVED_MSG, None)
    if msg:
        st.success(msg)

    if len(runs) >= MAX_RUNS:
        st.warning(f"Max {MAX_RUNS} runs saved. Remove one to save another.")
        return

    if st.button("💾 Save to Compare", use_container_width=True, key=key):
        if label is not None:
            result.label = label
        runs.append(result)
        # the saved-run list and any run counters are rendered above this button, so
        # rerun to show them including the run just saved instead of one save behind
        st.session_state[_SAVED_MSG] = f"Saved! ({len(runs)}/{MAX_RUNS})"
        st.rerun()


def render_saved_runs_manager(container=None, key_prefix: str = "") -> None:
    """List the saved runs with delete/clear controls. Defaults to the sidebar."""
    box = container if container is not None else st.sidebar
    runs = saved_runs()

    box.header("Saved Runs")

    if not runs:
        box.info("No runs saved yet. Run a simulation and click 'Save to Compare'.")
    else:
        for i, run in enumerate(runs):
            col1, col2 = box.columns([3, 1])
            with col1:
                st.markdown(
                    f"<span style='color:{COLORS[i]}'>●</span> **{run.track_name}**",
                    unsafe_allow_html=True,
                )
                st.caption(f"{run.format_lap_time()} | {run_label(run)}")
            with col2:
                if st.button(
                    "🗑️", key=f"{key_prefix}del_{i}", help="Remove this run"
                ):
                    runs.pop(i)
                    st.rerun()

        box.divider()
        if box.button(
            "Clear All", use_container_width=True, key=f"{key_prefix}clear_all"
        ):
            st.session_state.saved_runs = []
            st.rerun()

    box.caption(f"Saved: {len(runs)}/{MAX_RUNS}")


def _cumulative_time(run: SimulationResult) -> np.ndarray:
    """Cumulative lap time along the distance axis.

    Prefers the solver's own time trace (it includes the shift penalties) and falls back
    to integrating 1/v for results saved before that field existed.
    """
    if run.time is not None and len(run.time) == len(run.distance):
        return np.asarray(run.time) - run.time[0]

    ds = np.diff(run.distance)
    v_mean = 0.5 * (run.velocity[1:] + run.velocity[:-1])
    return np.concatenate(([0.0], np.cumsum(ds / v_mean)))


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def render_comparison(runs: list[SimulationResult] | None = None) -> None:
    """Render the full comparison view: lap/sector times, speed stats, overlays, delta."""
    runs = saved_runs() if runs is None else runs

    if not runs:
        st.info(
            "No runs to compare yet. Run a simulation and click **Save to Compare**."
        )
        return

    if len(runs) == 1:
        run = runs[0]
        st.warning("Add at least one more run to compare. You have 1 run saved.")
        st.subheader(f"{run.track_name} ({run_label(run)}) - {run.format_lap_time()}")
        return

    fastest_idx = int(np.argmin([r.lap_time for r in runs]))

    # ------------------------------------------------------------------ lap times
    st.header("Lap Times")

    for i, (col, run) in enumerate(zip(st.columns(len(runs)), runs)):
        with col:
            delta = run.lap_time - runs[fastest_idx].lap_time

            st.markdown(
                f"### <span style='color:{COLORS[i]}'>●</span> {run.track_name}",
                unsafe_allow_html=True,
            )
            st.caption(run_label(run))
            st.metric(
                "Lap Time",
                run.format_lap_time(),
                f"+{delta:.3f}s" if delta > 0 else None,
                delta_color="inverse",
            )

            st.write("**Sectors**")
            for s in range(3):
                fastest_sector = min(r.sector_times[s] for r in runs)
                sector_delta = run.sector_times[s] - fastest_sector
                if sector_delta > 0:
                    st.write(
                        f"S{s + 1}: {run.sector_times[s]:.3f}s (+{sector_delta:.3f})"
                    )
                else:
                    st.write(f"S{s + 1}: {run.sector_times[s]:.3f}s ✓")

    st.divider()

    # -------------------------------------------------------------- speed / energy
    st.header("Speed Statistics")

    for col, run in zip(st.columns(len(runs)), runs):
        with col:
            st.metric("Max Speed", f"{np.max(run.velocity_kmh):.1f} km/h")
            st.metric("Avg Speed", f"{np.mean(run.velocity_kmh):.1f} km/h")
            st.metric("Energy", f"{run.energy_consumed:.1f} kJ")

    st.divider()

    # ---------------------------------------------------------- velocity profiles
    st.header("Velocity Profiles")

    same_track = len({r.track_name for r in runs}) == 1

    if not same_track:
        st.warning("Runs are on different tracks. Velocity profiles shown separately.")

        tabs = st.tabs([f"{r.track_name} ({run_label(r)})" for r in runs])
        for i, (tab, run) in enumerate(zip(tabs, runs)):
            with tab:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=run.distance,
                        y=run.velocity_kmh,
                        mode="lines",
                        name=run.track_name,
                        line=dict(color=COLORS[i], width=2),
                    )
                )
                fig.update_layout(
                    xaxis_title="Distance [m]",
                    yaxis_title="Velocity [km/h]",
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")
        return

    fig = go.Figure()
    for i, run in enumerate(runs):
        fig.add_trace(
            go.Scatter(
                x=run.distance,
                y=run.velocity_kmh,
                mode="lines",
                name=run_label(run),
                line=dict(color=COLORS[i], width=2),
            )
        )
    fig.update_layout(
        xaxis_title="Distance [m]",
        yaxis_title="Velocity [km/h]",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")

    # ----------------------------------------------------------------- delta time
    st.header("Delta Time")
    st.caption("Time difference compared to fastest run (positive = slower)")

    ref_run = runs[fastest_idx]
    ref_time = _cumulative_time(ref_run)

    fig_delta = go.Figure()
    for i, run in enumerate(runs):
        if i == fastest_idx:
            continue

        delta = (
            np.interp(ref_run.distance, run.distance, _cumulative_time(run)) - ref_time
        )

        fig_delta.add_trace(
            go.Scatter(
                x=ref_run.distance,
                y=delta,
                mode="lines",
                name=f"{run_label(run)} vs fastest",
                line=dict(color=COLORS[i], width=2),
                fill="tozeroy",
                fillcolor=_rgba(COLORS[i], 0.2),
            )
        )

    fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_delta.update_layout(
        xaxis_title="Distance [m]",
        yaxis_title="Delta Time [s]",
        hovermode="x unified",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_delta, width="stretch")
