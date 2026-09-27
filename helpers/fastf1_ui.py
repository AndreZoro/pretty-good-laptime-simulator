"""
Shared Streamlit widgets for picking a FastF1 session.

Kept in one place because three pages offer the same picker (parameter identification,
ERS parameter identification and the reference-lap page) and they must not drift apart:
each one that guessed the event from a static track-name map was liable to download the
wrong circuit.

The selection is driven by FastF1's own calendar rather than a hand-maintained mapping.
An event name does not identify a circuit -- the 2026 calendar has a "Barcelona Grand
Prix" at Barcelona *and* a "Spanish Grand Prix" at Madrid -- so the season's real
schedule is the only sound source.
"""

import streamlit as st

from helpers.fastf1_data import (
    event_display,
    get_drivers_in_session,
    get_events,
    get_seasons,
    suggest_event,
)


@st.cache_data(show_spinner=False)
def cached_events(year: int) -> list[dict]:
    """Race calendar for a season. Cached -- it is a network call on a cold cache."""
    return get_events(year, past_only=True)


@st.cache_data(show_spinner=False)
def cached_drivers(year: int, gp: str, session: str) -> list[str]:
    """Drivers in a session. Cached because it costs a full session load."""
    return get_drivers_in_session(year, gp, session)


def render_event_picker(
    sim_track: str | None = None,
    key_prefix: str = "",
    container=None,
    default_session: str = "Qualifying",
) -> dict | None:
    """Render season / event / session / driver pickers and return the selection.

    Args:
        sim_track: sim track name used only to preselect the matching event; it never
            decides the event on its own, so a track the calendar does not recognise
            still leaves a usable picker
        key_prefix: prefix for the widget keys, so several pages can coexist
        container: where to render (defaults to the sidebar)
        default_session: session preselected when the event ran one by that name

    Returns:
        Dict with year, gp, session, driver, location, round and track -- or None when
        the season has no completed events to choose from.
    """
    box = container if container is not None else st.sidebar

    seasons = get_seasons()
    year = box.selectbox(
        "Season", options=seasons, index=len(seasons) - 1,  # latest season by default
        key=f"{key_prefix}ff1_year",
    )

    try:
        events = cached_events(year)
    except Exception as e:
        box.error(f"Could not read the {year} calendar: {e}")
        return None

    if not events:
        box.warning(f"No completed events in {year}. Pick an earlier season.")
        return None

    hint = suggest_event(sim_track, events)

    # A keyed widget takes 'index' only on its first render; afterwards session state
    # wins. So the selection is steered through session state instead, or picking a
    # different sim track would leave the previously chosen event in place -- exactly
    # the silent circuit mismatch this picker exists to prevent. The stored value is
    # also re-seeded whenever it is not in the current options, which is what happens
    # when the season changes underneath it.
    event_key = f"{key_prefix}ff1_event"
    track_key = f"{key_prefix}ff1_last_track"

    track_changed = st.session_state.get(track_key, object()) != sim_track
    st.session_state[track_key] = sim_track

    if st.session_state.get(event_key) not in events or (track_changed and hint):
        st.session_state[event_key] = hint or events[0]

    event = box.selectbox(
        "Event",
        options=events,
        format_func=event_display,
        key=event_key,
        help="Straight from the FastF1 calendar for this season.",
    )

    if sim_track and hint is None:
        box.caption(f"No {year} event obviously matches '{sim_track}' — check the pick.")

    # Offer the sessions the event actually ran, so sprint weekends work. Same story as
    # the event above: re-seed when the current pick is not on this event's programme.
    sessions = event["sessions"] or ["Qualifying", "Race"]
    session_key = f"{key_prefix}ff1_session"
    if st.session_state.get(session_key) not in sessions:
        st.session_state[session_key] = (
            default_session if default_session in sessions else sessions[0]
        )

    session = box.selectbox("Session", options=sessions, key=session_key)

    driver = None
    if box.checkbox("List drivers in this session", value=False,
                    key=f"{key_prefix}ff1_list_drivers",
                    help="Needs the session loaded, which is slow on a cold cache."):
        try:
            with st.spinner("Loading session..."):
                options = ["(fastest lap)"] + cached_drivers(year, event["name"], session)
            picked = box.selectbox("Driver", options=options, key=f"{key_prefix}ff1_driver_pick")
            driver = None if picked.startswith("(") else picked
        except Exception as e:
            box.error(f"Could not list drivers: {e}")
    else:
        typed = box.text_input(
            "Driver (optional)", value="", key=f"{key_prefix}ff1_driver",
            help="3-letter abbreviation (e.g. VER, HAM). Empty = fastest lap.",
        )
        driver = typed.strip().upper() or None

    return {
        "year": year,
        "gp": event["name"],
        "session": session,
        "driver": driver,
        # The circuit, not the event name, is what identifies the track.
        "location": event["location"],
        "round": event["round"],
        "track": sim_track,
    }
