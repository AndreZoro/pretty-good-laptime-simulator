"""
Comparison Page

Compare up to 3 saved simulation runs. The comparison itself lives in
helpers/comparison.py so the MVRC 2026 page can show the same view in a tab.
"""

import streamlit as st

from helpers.comparison import render_comparison, render_saved_runs_manager, saved_runs

st.set_page_config(
    page_title="Comparison - Laptime Sim",
    page_icon="🏎️",
    layout="wide",
)

st.title("📊 Comparison")
st.caption("Compare up to 3 simulation runs")

runs = saved_runs()

render_saved_runs_manager(key_prefix="cmp_")

if not runs:
    st.info(
        "👈 No runs to compare. Go to **Simple Simulation**, **Advanced Simulation** or "
        "**MVRC 2026**, run a simulation, and click **Save to Compare**."
    )

    st.markdown("""
    ### How to use

    1. Go to **Simple Simulation**, **Advanced Simulation** or **MVRC 2026**
    2. Configure and run a simulation
    3. Click **Save to Compare** to store the result
    4. Repeat for up to 3 different configurations
    5. Come back here to see the comparison

    ### What you can compare

    - Different tracks
    - Different series (F1 vs FE)
    - Different weather conditions
    - Different driver strategies
    - Different MVRC 2026 setups (power, drag, downforce)
    """)
else:
    render_comparison(runs)
