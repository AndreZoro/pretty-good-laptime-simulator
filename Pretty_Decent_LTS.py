"""
Great Laptime Simulation - Streamlit Web UI

A quasi-steady-state lap time simulation for race cars.
"""

import streamlit as st

st.set_page_config(
    page_title="Pretty Decent LTS - Laptime Simulation with Python in Streamlit",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ Decent LTS")


st.markdown("""
Welcome to the **Pretty Decent Laptime Simulation** web interface!

This tool allows you to simulate lap times for Formula 1 and Formula E race cars
on various circuits around the world.

## Features

- **25+ Race Tracks** - From Monaco to Monza, Shanghai to Silverstone
- **F1 & Formula E** - Simulate hybrid and electric powertrains
- **Weather Conditions** - Adjust grip levels for dry, damp, or wet conditions
- **Detailed Results** - Velocity profiles, sector times, and energy consumption

## Getting Started

👈 Select **Simple Simulation** from the sidebar to run your first simulation.

## About

This simulation is based on a quasi-steady-state approach that calculates the
maximum possible velocity at each point along the track, considering:

- Vehicle dynamics and tire grip limits
- Powertrain characteristics (engine, motor, gearbox)
- Aerodynamic forces (downforce and drag)
- Energy management strategies

Original work forked from [Technical University of Munich (TUM)](https://github.com/TUMFTM).
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Available Tracks", "25+")

with col2:
    st.metric("Racing Series", "F1 & FE")

with col3:
    st.metric("Simulation Time", "~5 sec")
