# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `lts313` conda environment (Python 3.13) for all commands:

```bash
conda run -n lts313 python -m pytest tests/ -v          # Run all tests
conda run -n lts313 python -m pytest tests/test_car.py -v  # Run a single test file
conda run -n lts313 python -m pytest tests/ -k "test_name"  # Run a single test by name
conda run -n lts313 python -m pytest tests/ -m "not slow"   # Skip slow tests
conda run -n lts313 python main_laptimesim.py            # Run lap simulation CLI
conda run -n lts313 python main_drag_test.py             # Run drag racing simulation
```

Install dependencies: `pip install -e .` or `pip install -r requirements.txt`

## Architecture

This is a **quasi-steady-state lap time simulation** for race cars. The core workflow:

1. Load **Track** (`track.py`) — reads raceline CSV + `track_pars.ini`, computes curvature and DRS zones
2. Load **Car** (`car.py`, `car_hybrid.py`, `car_electric.py`) — reads `.ini` vehicle config, builds tire/aero model
3. Create **Driver** (`driver.py`) — energy management strategy (FCFB, LBP, LS, ERSO, NONE)
4. Run **Lap** solver (`lap.py`) — forward/backward quasi-steady-state iterations until convergence
5. Output velocity profile, lap time, energy consumption, sector times

```
Track + Car + Driver → Lap (solver) → results/plots
```

### Key Source Files

- **`laptimesim/src/lap.py`** (1400+ lines) — main solver; `__fbplus()` is the hot loop (forward/backward pass)
- **`laptimesim/src/car.py`** — base `Car` class with tire forces, aero, resistance; uses property getters/setters with `__slots__`
- **`laptimesim/src/car_hybrid.py`** — hybrid ICE + e-motor (F1-style), torque curves via LES coefficients
- **`laptimesim/src/car_electric.py`** — pure electric, supports dual-motor AWD
- **`laptimesim/src/driver.py`** — energy management strategies, lift-and-coast, boost/harvest decisions
- **`laptimesim/src/track.py`** — track geometry, friction, DRS zones, sector boundaries
- **`laptimesim/src/_jit_kernels.py`** — Numba JIT-compiled tight loops: `_tire_force_pots()`, `_v_max_cornering()`, `_calc_max_ax()`
- **`laptimesim/src/drag_test.py`** — standing-start drag racing: 1/8 mile, 1/4 mile, 1 km
- **`helpers/simulation.py`** — Streamlit simulation wrapper and drag test runner
- **`helpers/visualization.py`** — Matplotlib/Plotly plot generation for the web UI

### Configuration

- **Vehicle params**: `laptimesim/input/vehicles/*.ini` — INI format with Python dict literals. Current configs: `F1_2025.ini`, `F1_2026.ini`, `MVRC_2026.ini`, `Zapovic_Breeze.ini`, etc.
- **Track params**: `laptimesim/input/tracks/track_pars.ini` — maps track name to sector boundaries, friction coefficients, DRS zones, pit lane speeds
- **Racelines**: `laptimesim/input/tracks/racelines/*.csv` — 25+ F1 tracks, columns: `[x, y, curvature, width_right, width_left]`

### Tests

- **`test_car.py`** — unit tests for car physics (tire forces, resistance)
- **`test_integration.py`** — full lap simulations compared against reference data (numerical tolerance `rtol/atol=1e-2`)
- **`test_laptimesim.py`** — integration tests against pickled reference objects; **numerical changes will break this**
- **`test_f1_2026.py`** — 2026-specific features (active aero gating)

## Critical Implementation Notes

### Active Aero (F1 2026)
Must gate on **both** speed AND lateral acceleration (`active_aero_ay_limit`). Speed-only gating causes active aero to open in high-speed corners, reducing rear downforce, failing the lateral grip check, and triggering erroneous backward braking sweeps → non-convergence.

`v_max_cornering` is precomputed with full downforce (active aero off) — this is correct because active aero is off during cornering. Do not precompute with reduced downforce.

### Solver Sensitivity
The `v_max_cornering` step count (546 steps) is sensitive. Reducing it changes velocity profiles enough to break integration tests (`rtol/atol=1e-2`). Don't change without regenerating reference data in `test_laptimesim.py`.

### Code Patterns
- `Car` and subclasses use `__slots__` with property getter/setter pattern throughout
- JIT kernels in `_jit_kernels.py` use Numba — first call compiles (slow), subsequent calls are fast
- The linter auto-reformats `car.py` on save — verify intentional formatting changes survive
