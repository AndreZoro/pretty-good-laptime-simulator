"""
Optimize gear ratios (i_trans) and shift RPMs (n_shift) for the F1_2025 vehicle
on the Catalunya circuit using scipy.optimize.differential_evolution.

Parameterizes the 8-speed gearbox with 4 parameters:
  - i_first: first gear ratio
  - i_last: last gear ratio
  - progression: power-law spacing exponent (1.0 = geometric)
  - n_shift_mid: shift RPM for gears 2-7

Usage:
    conda run -n lts313 python optimize_gearbox.py
"""

import ast
import configparser
import copy
import os
import time

import numpy as np
from scipy.optimize import differential_evolution

import main_laptimesim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
VEHICLE_INI = os.path.join(REPO_PATH, "laptimesim", "input", "vehicles", "F1_2025_optimized_shifts.ini")
TRACK_NAME = "Catalunya"

# Parameter bounds: [i_first, i_last, progression, n_shift_mid]
# n_shift_mid must be >= n_max (14500) to shift near peak power, as real F1 cars do
BOUNDS = [
    (0.025, 0.065),  # i_first
    (0.14, 0.28),  # i_last
    (0.5, 2.0),  # progression
    (11000, 12200.0),  # n_shift_mid (near n_max=14500 to n_end=15000)
]

# ---------------------------------------------------------------------------
# Load base vehicle parameters from INI
# ---------------------------------------------------------------------------


def load_vehicle_pars(ini_path: str) -> dict:
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    return ast.literal_eval(parser.get("VEH_PARS", "veh_pars"))


BASE_VEH_PARS = load_vehicle_pars(VEHICLE_INI)

# ---------------------------------------------------------------------------
# Gearbox parameterization
# ---------------------------------------------------------------------------


def build_gearbox(
    i_first: float, i_last: float, progression: float, n_shift_mid: float
) -> tuple:
    """Generate 8 gear ratios and 8 shift RPMs from 4 parameters."""
    n_gears = 8
    # Power-law interpolation: t_k = (k / (n-1))^progression, then i = i_first + t_k * (i_last - i_first)
    k = np.arange(n_gears)
    t = (k / (n_gears - 1)) ** progression
    i_trans = (i_first + t * (i_last - i_first)).tolist()

    # Shift RPMs: gear 1 = n_begin, gears 2-7 = n_shift_mid, gear 8 = n_end
    n_begin = BASE_VEH_PARS["engine"]["n_begin"]
    n_end = BASE_VEH_PARS["engine"]["n_end"]
    n_shift = [n_begin] + [n_shift_mid] * 6 + [n_end]

    return i_trans, n_shift


# ---------------------------------------------------------------------------
# Simulation options (configured once)
# ---------------------------------------------------------------------------

TRACK_OPTS = {
    "trackname": TRACK_NAME,
    "flip_track": False,
    "mu_weather": 1.0,
    "interp_stepsize_des": 5.0,
    "curv_filt_width": 10.0,
    "use_drs1": True,
    "use_drs2": True,
    "use_pit": False,
}

DRIVER_OPTS = {
    "vel_subtr_corner": 0.5,
    "vel_lim_glob": None,
    "yellow_s1": False,
    "yellow_s2": False,
    "yellow_s3": False,
    "yellow_throttle": 0.3,
    "initial_energy": 4.0e6,
    "em_strategy": "FCFB",
    "use_recuperation": True,
    "use_lift_coast": False,
    "lift_coast_dist": 10.0,
}

SA_OPTS = {
    "use_sa": False,
    "sa_type": "mass",
    "range_1": [733.0, 833.0, 5],
    "range_2": None,
}

DEBUG_OPTS = {
    "use_plot": False,
    "use_debug_plots": False,
    "use_plot_comparison_tph": False,
    "use_print": False,
    "use_print_result": False,
}

# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

eval_count = 0
best_lap_time = float("inf")
best_params = None


def objective(x):
    global eval_count, best_lap_time, best_params
    eval_count += 1

    i_first, i_last, progression, n_shift_mid = x
    i_trans, n_shift = build_gearbox(i_first, i_last, progression, n_shift_mid)

    # Build modified vehicle parameters
    veh_pars = copy.deepcopy(BASE_VEH_PARS)
    veh_pars["gearbox"]["i_trans"] = i_trans
    veh_pars["gearbox"]["n_shift"] = n_shift

    solver_opts = {
        "vehicle": "F1_2025",
        "series": "F1",
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-4,
        "custom_vehicle_pars": veh_pars,
    }

    try:
        lap = main_laptimesim.main(
            track_opts=TRACK_OPTS,
            solver_opts=solver_opts,
            driver_opts=DRIVER_OPTS,
            sa_opts=SA_OPTS,
            debug_opts=DEBUG_OPTS,
        )
        lap_time = lap.t_cl[-1]
    except Exception as e:
        print(f"  [eval {eval_count}] FAILED: {e}")
        return 200.0  # penalty for failed simulations

    if lap_time < best_lap_time:
        best_lap_time = lap_time
        best_params = x.copy()
        print(
            f"  [eval {eval_count}] NEW BEST: {lap_time:.3f}s | "
            f"i_first={i_first:.4f} i_last={i_last:.4f} prog={progression:.3f} n_shift={n_shift_mid:.0f}"
        )
    elif eval_count % 50 == 0:
        print(f"  [eval {eval_count}] best so far: {best_lap_time:.3f}s")

    return lap_time


# ---------------------------------------------------------------------------
# Write results to INI
# ---------------------------------------------------------------------------


def update_ini(ini_path: str, i_trans: list, n_shift: list):
    """Update the i_trans and n_shift values in the vehicle INI file."""
    with open(ini_path, "r") as f:
        content = f.read()

    # Parse current values to find and replace them
    pars = load_vehicle_pars(ini_path)
    old_i_trans = repr(pars["gearbox"]["i_trans"])
    old_n_shift = repr(pars["gearbox"]["n_shift"])

    # Format new values
    new_i_trans = "[" + ", ".join(f"{v:.4f}" for v in i_trans) + "]"
    new_n_shift = "[" + ", ".join(f"{v:.1f}" for v in n_shift) + "]"

    content = content.replace(old_i_trans, new_i_trans)
    content = content.replace(old_n_shift, new_n_shift)

    with open(ini_path, "w") as f:
        f.write(content)

    print(f"\nUpdated {ini_path}")
    print(f"  i_trans = {new_i_trans}")
    print(f"  n_shift = {new_n_shift}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Gearbox Optimization for F1_2025 on Catalunya")
    print("=" * 60)

    # Run baseline first
    print("\nRunning baseline simulation...")
    t0 = time.perf_counter()
    baseline_pars = copy.deepcopy(BASE_VEH_PARS)
    baseline_solver = {
        "vehicle": "F1_2025",
        "series": "F1",
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
        "vel_tol": 1e-5,
        "custom_vehicle_pars": baseline_pars,
    }
    baseline_lap = main_laptimesim.main(
        track_opts=TRACK_OPTS,
        solver_opts=baseline_solver,
        driver_opts=DRIVER_OPTS,
        sa_opts=SA_OPTS,
        debug_opts=DEBUG_OPTS,
    )
    baseline_time = baseline_lap.t_cl[-1]
    sim_duration = time.perf_counter() - t0
    print(f"Baseline lap time: {baseline_time:.3f}s (sim took {sim_duration:.1f}s)")

    # Run optimizer
    print(f"\nStarting differential evolution (popsize=10, maxiter=15)...")
    print(f"Bounds: {BOUNDS}")
    t0 = time.perf_counter()

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=42,
        popsize=10,
        maxiter=15,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
    )

    opt_duration = time.perf_counter() - t0

    # Extract results
    i_first, i_last, progression, n_shift_mid = result.x
    opt_i_trans, opt_n_shift = build_gearbox(i_first, i_last, progression, n_shift_mid)

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total evaluations: {eval_count}")
    print(f"Total time: {opt_duration:.0f}s ({opt_duration / 60:.1f} min)")
    print(f"Baseline lap time: {baseline_time:.3f}s")
    print(f"Optimized lap time: {result.fun:.3f}s")
    print(f"Improvement: {baseline_time - result.fun:.3f}s")
    print(f"\nOptimal parameters:")
    print(f"  i_first    = {i_first:.4f}")
    print(f"  i_last     = {i_last:.4f}")
    print(f"  progression = {progression:.3f}")
    print(f"  n_shift_mid = {n_shift_mid:.0f}")
    print(f"\nOptimal gear ratios (i_trans): {[round(v, 4) for v in opt_i_trans]}")
    print(f"Optimal shift RPMs (n_shift):  {[round(v, 1) for v in opt_n_shift]}")

    # Update INI file
    update_ini(VEHICLE_INI, opt_i_trans, opt_n_shift)

    print("\nDone!")
