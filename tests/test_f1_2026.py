import os
import pickle

import numpy as np
import pytest

import main_laptimesim

# Hardcoded F1_2026 vehicle parameters — frozen so INI changes don't break the test.
VEHICLE_PARS = {
    "powertrain_type": "hybrid",
    "general": {
        "lf": 1.557, "lr": 1.842, "h_cog": 0.200,
        "sf": 1.620, "sr": 1.525, "m": 820.0, "f_roll": 0.03,
        "c_w_a": 1.56, "c_z_a_f": 2.20, "c_z_a_r": 2.68,
        "g": 9.81, "rho_air": 1.18, "drs_factor": 0.5,
    },
    "engine": {
        "topology": "RWD", "pow_max": 400e3, "pow_diff": 25e3,
        "n_begin": 10500.0, "n_max": 11500.0, "n_end": 12500.0,
        "be_max": 100.0, "pow_e_motor": 350e3, "eta_e_motor": 0.9,
        "eta_e_motor_re": 0.15, "eta_etc_re": 0.0,
        "vel_min_e_motor": 0.0, "torque_e_motor_max": 500.0,
        "ers_speed_limit": True, "max_e_energy_storage": 4.0e6,
        "e_rec_e_motor_max": 8.5e6, "series": "F1_2026",
    },
    "gearbox": {
        "i_trans": [0.04, 0.070, 0.095, 0.117, 0.143, 0.172, 0.190, 0.206],
        "n_shift": [10500.0, 11006.2, 11006.2, 11006.2, 11006.2, 11006.2, 11006.2, 12000.0],
        "e_i": [1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07],
        "eta_g": 0.96, "diff_lock_ratio": 0.7, "t_shift": 0.025,
    },
    "tires": {
        "f": {"circ_ref": 2.104, "fz_0": 3000.0, "mux": 1.65, "muy": 1.85, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
        "r": {"circ_ref": 2.120, "fz_0": 3000.0, "mux": 1.95, "muy": 2.15, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
        "tire_model_exp": 2.0,
    },
}

SOLVER_OPTS = {
    "vehicle": None,
    "limit_braking_weak_side": "FA",
    "v_start": 100.0 / 3.6,
    "find_v_start": True,
    "max_no_em_iters": 5,
    "es_diff_max": 1.0,
    "vel_tol": 1e-5,
    "custom_vehicle_pars": VEHICLE_PARS,
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


def _run_and_compare(track_name: str):
    track_opts = {
        "trackname": track_name,
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 1.0,
        "curv_filt_width": 10.0,
        "use_drs": True,
        "use_pit": False,
    }

    lap = main_laptimesim.main(
        track_opts=track_opts,
        solver_opts=SOLVER_OPTS,
        driver_opts=DRIVER_OPTS,
        sa_opts=SA_OPTS,
        debug_opts=DEBUG_OPTS,
    )

    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "testdata", "testobjects",
        f"testobj_f1_2026_{track_name}.pkl",
    )

    with open(ref_path, "rb") as fh:
        target_lap = pickle.load(fh)

    assert np.allclose(target_lap.vel_cl, lap.vel_cl, rtol=1e-2, atol=1e-2), \
        f"{track_name}: velocity profile mismatch"


def test_f1_2026_spa():
    _run_and_compare("Spa")


def test_f1_2026_catalunya():
    _run_and_compare("Catalunya")


def test_f1_2026_monza():
    _run_and_compare("Monza")


if __name__ == "__main__":
    for track in ["Spa", "Catalunya", "Monza"]:
        _run_and_compare(track)
        print(f"{track}: passed")
