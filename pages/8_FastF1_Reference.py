"""
FastF1 Reference Laps

Download a real F1 lap and put it into the saved-run store, so the Comparison page can
overlay it against simulated runs.

The telemetry itself is not stored here -- FastF1 caches the session data on disk. Only
the list of picked laps is kept (helpers/reference_laps.py), and re-loading one replays
the download out of that cache.
"""

import numpy as np
import streamlit as st

from helpers.comparison import (
    MAX_RUNS,
    render_comparison,
    render_saved_runs_manager,
    save_run_button,
    saved_runs,
)
from helpers.fastf1_ui import render_event_picker
from helpers.reference_laps import (
    delete_from_index,
    entry_id,
    fetch_reference_lap,
    load_index,
    reference_label,
    save_to_index,
    sim_track_length,
)
from helpers.simulation import TRACK_SUFFIX_2026, get_available_tracks
from helpers.visualization import render_simulation_plots

st.set_page_config(
    page_title="FastF1 Reference - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.title("📡 [FastF1](https://github.com/theOehrly/Fast-F1) Reference Laps")
st.caption("Download a real lap and compare it against your simulations")

if "ff1_ref_result" not in st.session_state:
    st.session_state.ff1_ref_result = None
if "ff1_ref_index" not in st.session_state:
    st.session_state.ff1_ref_index = load_index()
saved_runs()  # creates st.session_state.saved_runs on first use


def _load_entry(entry: dict) -> None:
    """Fetch a lap and make it the page's current result."""
    result, resolved = fetch_reference_lap(entry)
    st.session_state.ff1_ref_result = result
    st.session_state.ff1_ref_entry = resolved  # carries the driver a 'fastest' pick resolved to


# ---------------------------------------------------------------------------- sidebar
st.sidebar.header("Session")

# Tagging the lap with a sim track sets the track name the Comparison page matches on,
# and gives the raceline length used to correct the telemetry's distance axis. It only
# preselects an event -- the event itself comes from FastF1's calendar (see
# helpers/fastf1_ui.py for why a static track-to-event map cannot be trusted).
# The MVRC entry point sets this, so the standalone MVRC app offers the same 2026
# calendar racelines its simulation page does instead of the full legacy list.
tracks = get_available_tracks(
    TRACK_SUFFIX_2026 if st.session_state.get("mvrc_app") else None
)
track = st.sidebar.selectbox(
    "Sim track (optional)",
    options=["(none)"] + tracks,
    help="Tags the lap so the Comparison page can tell whether it is on the same "
         "circuit as a simulated run, and rescales its distance axis to that raceline.",
)
sim_track = None if track.startswith("(") else track

current_entry = render_event_picker(sim_track=sim_track, key_prefix="ff1ref_")

download = st.sidebar.button(
    "📡 Download Lap", type="primary", width="stretch", disabled=current_entry is None
)

st.sidebar.divider()
render_saved_runs_manager(key_prefix="ff1ref_")

# --------------------------------------------------------------------------- download
if download and current_entry is not None:
    with st.spinner(
        f"Downloading {current_entry['year']} {current_entry['gp']} "
        f"{current_entry['session']}..."
    ):
        try:
            _load_entry(current_entry)
            st.success(f"Loaded {reference_label(st.session_state.ff1_ref_entry)}")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.exception(e)

# ------------------------------------------------------------------------ stored laps
result = st.session_state.ff1_ref_result

tab_lap, tab_stored, tab_compare = st.tabs(
    ["🏁 Lap", f"💾 Stored ({len(st.session_state.ff1_ref_index)})",
     f"📊 Comparison ({len(saved_runs())}/{MAX_RUNS})"]
)

with tab_lap:
    if result is None:
        st.info("👈 Pick a session and click **Download Lap**.")
    else:
        entry = st.session_state.ff1_ref_entry

        col_save, col_store, _ = st.columns([1, 1, 2])
        with col_save:
            save_run_button(result, key="ff1ref_save")
        with col_store:
            if st.button("⭐ Keep in Stored", width="stretch",
                         help="Remember this lap so it can be re-loaded after a restart"):
                st.session_state.ff1_ref_index = save_to_index(entry)
                st.rerun()

        st.divider()

        st.header(reference_label(entry))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lap Time", result.format_lap_time())
        with col2:
            st.metric("Sector 1", f"{result.sector_times[0]:.3f}s")
        with col3:
            st.metric("Sector 2", f"{result.sector_times[1]:.3f}s")
        with col4:
            st.metric("Sector 3", f"{result.sector_times[2]:.3f}s")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Speed", f"{np.max(result.velocity_kmh):.1f} km/h")
        with col2:
            st.metric("Avg Speed", f"{np.mean(result.velocity_kmh):.1f} km/h")
        with col3:
            st.metric("Samples", f"{len(result.distance)}")

        scale = entry.get("distance_scale", 1.0)
        if abs(scale - 1.0) > 0.05:
            # More than a few percent is not integration drift -- it means the lap and
            # the tagged raceline are different circuits.
            st.warning(
                f"Distance axis had to be rescaled by ×{scale:.3f} "
                f"({entry['raw_distance_m']:.0f} m → {result.distance[-1]:.0f} m) to fit the "
                f"**{entry['track']}** raceline. That is far more than measurement drift — "
                f"check that **{entry['gp']}** at **{entry.get('location')}** is really the "
                f"same circuit as this raceline."
            )
        elif abs(scale - 1.0) > 1e-3:
            st.caption(
                f"Distance axis rescaled by ×{scale:.3f} "
                f"({entry['raw_distance_m']:.0f} m → {result.distance[-1]:.0f} m) to match the "
                f"{entry['track']} raceline."
            )
        elif sim_track_length(entry.get("track")) is None:
            st.warning(
                "No sim raceline for this lap, so the distance axis is FastF1's own and "
                "may be several percent long. Tag the lap with a sim track to correct it."
            )

        st.caption(
            "Curvature and lateral acceleration are derived from the position channel "
            "and smoothed; treat them as indicative. Energy and fuel are not available "
            "from public telemetry."
        )

        render_simulation_plots(result, key_prefix="ff1ref_")

with tab_stored:
    entries = st.session_state.ff1_ref_index

    if not entries:
        st.info(
            "No laps kept yet. Download a lap and click **Keep in Stored** to remember "
            "it. Only the selection is stored — the telemetry stays in the FastF1 cache."
        )
    else:
        st.caption(
            "Loading a stored lap replays the download from the FastF1 cache, so it is "
            "fast and works offline once the session has been fetched at least once."
        )
        for i, entry in enumerate(entries):
            col_desc, col_load, col_del = st.columns([4, 1, 1])
            with col_desc:
                st.markdown(f"**{reference_label(entry)}**")
                st.caption(
                    f"{entry.get('track') or 'no sim track'} · saved "
                    f"{entry.get('saved_at', 'unknown')}"
                )
            # Keys are prefixed 'stored_' to stay clear of the saved-runs manager in the
            # sidebar, which numbers its own delete buttons from the same key_prefix.
            with col_load:
                if st.button("Load", key=f"ff1ref_stored_load_{i}", width="stretch"):
                    with st.spinner("Loading from FastF1 cache..."):
                        try:
                            _load_entry(entry)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not load: {e}")
            with col_del:
                if st.button("🗑️", key=f"ff1ref_stored_del_{i}", help="Forget this lap"):
                    st.session_state.ff1_ref_index = delete_from_index(entry)
                    st.rerun()

with tab_compare:
    render_comparison()
