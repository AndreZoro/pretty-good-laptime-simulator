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
- **`test_integration.py`** — full lap simulations checked against *properties* (lap time ranges, monotonic time, ERS budget/mask invariants, gate behaviour), not frozen numbers. Prefer adding here: it survives deliberate numerical changes.
- **`test_laptimesim.py`** — F1_Shanghai + FCFB on Shanghai_pre_2026, velocity profile vs. a pickled `Lap` at `rtol/atol=1e-2`; the oldest frozen baseline. **Numerical changes will break this** — see *Regenerating Reference Data*.
- **`test_f1_2026.py`** — F1 2026 car vs. pickled `Lap` objects on Spa, Catalunya and Monza (same tolerance, also FCFB). Frozen baselines, not feature tests.

## Critical Implementation Notes

### Active Aero (F1 2026)
Gated on curvature via `active_aero_kappa_threshold` (a vehicle `general` parameter; see `active_aero_kappa_threshold()` in `car.py`, default 0.01 1/m ≈ R 100 m), applied in `lap.py` as `drs & (abs(kappa) <= threshold)`. Opening active aero in high-speed corners reduces rear downforce, fails the lateral grip check and triggers erroneous backward braking sweeps → non-convergence.

`v_max_cornering` uses full downforce (active aero off) — correct, because active aero is off during cornering. Do not compute it with reduced downforce.

### ERS Lateral-Acceleration Gates
Deployment and active harvest are both bounded by the lateral acceleration the car reaches, not by curvature alone — see `ay_max_deploy()` / `ay_max_harvest()` in `driver.py` (both default 3 g, overridable per run via `driver_opts`, `np.inf` disables).

Curvature alone does not say how much grip is left: a 110 m bend passes a `kappa < 0.01` gate yet pulls over 4 g at 250 km/h, where the driven tires have no spare longitudinal capacity. Asking for torque there (either sign) makes the solver answer with a grip-limited braking sweep, which showed up as a deploy/harvest limit cycle chattering every few metres mid-corner. The gates are enforced twice: in the strategy (QUALY/ERSO, so the plan is priced honestly) and in the `lap.py` forward pass against the `a_y` actually reached, because the masks are planned from the *previous* EM iteration's velocities.

**FCFB is exempt from the deploy gate.** The gate's premise is that deployment the car cannot hold is better spent elsewhere, which presupposes a strategy that allocates a budget. QUALY/ERSO redeploy the savings and get faster; FCFB deploys unconditionally with no allocation step, so the same gate is a pure loss for it. FCFB therefore still deploys mid-corner.

Because the masks are planned on stale velocities, a point can be planned just inside the ceiling and land just outside it once converged. Assert on what the solver **applied**, not on the mask — see `TestLateralAccelerationGates` in `test_integration.py`.

### EM Pricing: What May and May Not Depend on Velocity
Both strategies price every point by `tau / v²`, where `tau` (`__time_to_next_event`) is the time until a speed gain is absorbed. The **event mask feeding `tau` must be identical from one EM iteration to the next**, because it prices the whole lap: if it moves, the ranking churns and the plan never settles, so the lap time ends up depending on `max_no_em_iters`. Guarded by `TestERSOStrategy::test_em_iterations_converge`.

This is why the two uses of the corner threshold deliberately differ, and must stay different:

| use | basis | why |
|---|---|---|
| deployment/harvest **eligibility** | `a_y` (`ay_max_deploy` / `ay_max_harvest`) | decides only where torque is allowed; a point flipping affects that point alone |
| `tau` **event mask** | curvature (`kappa_max_deploy`) | prices every point, so it must be static — `a_y = v²·κ` moves with the profile and makes points near the threshold oscillate |

Do not "tidy up" the `tau` mask to use `a_y` for consistency with the gate. That was tried: it is ~0.5 s faster on Barcelona but non-converging, i.e. the number is wherever the oscillation stopped rather than a solution.

ERSO's `tau` mask originally had **no** corner term at all (QUALY always had one), so it priced a deployment feeding into a fast bend as if the gain survived the apex. That mispricing is why the `a_y` gates initially made ERSO slower while making QUALY faster — the gate removed the points ERSO was overvaluing and its charge-sustaining bisection re-equilibrated worse rather than redistributing.

### Strategy Choice and Energy Bookkeeping
The strategies target different race formats, so their end-of-lap battery state differs **by design** — do not read one as a defect of the other:

| | QUALY | ERSO |
|---|---|---|
| format | single flying lap | multi-lap, charge-sustaining |
| ES at finish | 0 MJ — spends the battery | retains charge (Barcelona/MVRC_2026: 2.55 MJ) |
| recovered (`e_rec_e_motor`) | 3.50 MJ | 7.63 MJ |
| net (`energy_consumed`) | 7.15 MJ | 1.28 MJ |
| gross deployed | — | 7.57 MJ (measured) |

ERSO being slower than QUALY over one lap (~1.9 s on Barcelona/MVRC_2026) is the cost of sustaining charge, not a bug. MVRC pages and `run_mvrc_batch.py` both use QUALY, which is correct for a one-lap format.

**`SimulationResult.energy_consumed` is a NET figure, not gross deployment.** `Car.e_cons()` sums `power_demand_e_motor_drive * dt` over every point, and `m_e_motor` is negative while harvesting, so recovery is subtracted. ERSO therefore reports ~1.3 MJ "consumed" while actually deploying 7.57 MJ. To get gross deployment, sum that power only where `m_e_motor > 0`. (Note the same call applies the *drive* efficiency to negative torque, so the recovery half of that sum is not a physically meaningful recovered-energy figure either — use `e_rec_e_motor` for recovery.) Misreading this field as gross deployment makes ERSO look badly unbalanced when it is not.

**Genuine known limitation, in QUALY:** 31.7 % of QUALY's planned deploy points get no deployment (615 of 1938 on Barcelona/MVRC_2026) because the plan respects the *total* budget while calling for deployment before the charge has been recovered; the solver then silently skips those points, leaving the plan mis-priced. The opt-in guard is `use_es_feasibility` (off by default — it makes the ES trace far more plausible but costs a little lap time on balance). ERSO does not have this problem (19 of 1416, 1.3 %).

### Cornering Velocity Ceiling
`v_max_cornering` is a **continuous bisection** (27 halvings, `_cornering_feasible` in `_jit_kernels.py`), and the solver holds it as a per-point array (`Car.v_max_cornering_arr` → folded into `vel_lim_cl`) that the forward pass tracks by lifting and coasting when it falls.

Both properties matter and both were once wrong:
- Evaluating the ceiling only reactively (in CASE 2, after the grip check already failed) made the solver accelerate until it broke grip, get clamped by a maximum-braking backward sweep, then accelerate again.
- The old implementation bisected a **fixed grid of 546 velocities**, quantising the ceiling to 0.2 m/s; shedding 0.2 m/s inside a 1 m step needs ≈ −13 m/s², so the profile followed the stair-steps in jolts.

Together these produced a mid-corner sawtooth of roughly +0.5 / −24 m/s². The old "546 step count is sensitive" warning is obsolete — there is no step count now. Regression guards: `TestCorneringLimitTracking` in `test_integration.py`.

### Regenerating Reference Data
`test_laptimesim.py` and `test_f1_2026.py` compare velocity profiles against pickled `Lap` objects at `rtol/atol=1e-2`, so any numerical change to the solver breaks them. Both use `em_strategy="FCFB"`, so changes confined to QUALY/ERSO leave them untouched — but anything upstream of energy management (tires, aero, the cornering ceiling) moves all four. Regenerate with:

```bash
conda run -n lts313 python tests/regenerate_f1_2026_refs.py      # Spa, Catalunya, Monza
conda run -n lts313 python tests/regenerate_laptimesim_ref.py    # Shanghai
```

Both print `old → new` lap time and max `|dv|` per track. Check those deltas before committing, and record in the commit message which baseline moved and why.

### Result Channels
`SimulationResult.acceleration` is the per-step `dv/dt` (`_long_acceleration_steps` in `helpers/simulation.py`), built from the closed `vel_cl`/`t_cl` arrays. Do **not** switch it to a centred derivative (`np.gradient`): that averages the steps either side of each point, smearing short events (an ERS switch, a brake application) across three points, and it can report deceleration at a point where the car is accelerating. Measured telemetry in `helpers/reference_laps.py` is sampled data and does use a centred derivative — that is deliberate, and keeps sim and real laps on comparable estimators.

`lat_acceleration` is smoothed with an 11-point boxcar, so it clips genuine `a_y` peaks; the solver's own `a_y = v²·κ` is unsmoothed. Use the unsmoothed form when asserting on physics.

### Code Patterns
- `Car` and subclasses use `__slots__` with property getter/setter pattern throughout
- JIT kernels in `_jit_kernels.py` use Numba — first call compiles (slow), subsequent calls are fast
- The linter auto-reformats `car.py` on save — verify intentional formatting changes survive
- Streamlit caches imported modules: after editing `laptimesim/` or `helpers/`, restart the app rather than just re-running the page (same for `.ini` edits)
