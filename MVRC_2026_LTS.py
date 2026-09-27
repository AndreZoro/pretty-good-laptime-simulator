"""
MVRC 2026 - Streamlit Web UI

Standalone entry point for the MVRC 2026 season:

    streamlit run MVRC_2026_LTS.py

Carries the MVRC simulation page plus the FastF1 reference-lap page, so an entrant can
overlay a real lap against their own setup without the rest of the multi-page app.

Pretty_Decent_LTS.py stays the entry point for the full multi-page app.
"""

import streamlit as st

# Both pages are shared with the multi-page app, so the two entry points stay in sync.
MVRC_PAGE = "pages/3_MVRC_2026.py"
REFERENCE_PAGE = "pages/8_FastF1_Reference.py"

# Tells the shared pages they are running in the MVRC app: the reference page uses it to
# offer the 2026 calendar racelines rather than the full legacy track list. Set before
# page.run() so the page script sees it on its first render.
st.session_state["mvrc_app"] = True

# st.navigation replaces Streamlit's automatic pages/ discovery, which would otherwise
# list every page of the full app here. The saved-run store lives in session state, so
# a lap saved on either page shows up in the other's comparison tab.
page = st.navigation(
    [
        st.Page(MVRC_PAGE, title="MVRC 2026", icon="🏎️", default=True),
        st.Page(REFERENCE_PAGE, title="FastF1 Reference", icon="📡"),
    ],
)
page.run()
