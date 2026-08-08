"""
MVRC 2026 - Streamlit Web UI

Standalone entry point that opens straight into the MVRC 2026 simulation page:

    streamlit run MVRC_2026_LTS.py

Pretty_Decent_LTS.py stays the entry point for the full multi-page app.
"""

import streamlit as st

# The MVRC page is shared with the multi-page app, so both entry points stay in sync.
MVRC_PAGE = "pages/3_MVRC_2026.py"

# st.navigation replaces Streamlit's automatic pages/ discovery: declaring the MVRC page
# as the only page keeps the other pages of the full app out of this one, and
# position="hidden" drops the navigation widget that would list just this single entry.
# The page script sets its own page config (title, icon, wide layout).
page = st.navigation(
    [st.Page(MVRC_PAGE, title="MVRC 2026", icon="🏎️", default=True)],
    position="hidden",
)
page.run()
