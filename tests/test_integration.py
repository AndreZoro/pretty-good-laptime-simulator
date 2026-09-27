"""
Integration tests for the laptime simulation.

These tests run full lap simulations to verify the complete system works correctly.
"""

import pytest
import numpy as np
import os


@pytest.fixture
def repo_path():
    """Get the repository root path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_default_opts():
    """Get default options for lap simulation."""
    track_opts = {
        "trackname": "Shanghai",
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 5.0,
        "curv_filt_width": 10.0,
        "use_drs": True,
        "use_pit": False,
    }
    solver_opts = {
        "vehicle": "F1_Shanghai.ini",
        "limit_braking_weak_side": "FA",
        "v_start": 100.0 / 3.6,
        "find_v_start": True,
        "max_no_em_iters": 5,
        "es_diff_max": 1.0,
    }
    driver_opts = {
        "vel_subtr_corner": 0.5,
        "vel_lim_glob": None,
        "yellow_s1": False,
        "yellow_s2": False,
        "yellow_s3": False,
        "yellow_throttle": 0.3,
        "initial_energy": 4.58e6,
        "em_strategy": "FCFB",
        "use_recuperation": True,
        "use_lift_coast": False,
        "lift_coast_dist": 10.0,
    }
    sa_opts = {
        "use_sa": False,
        "sa_type": "mass",
        "range_1": [733.0, 833.0, 5],
        "range_2": None,
    }
    debug_opts = {
        "use_plot": False,
        "use_debug_plots": False,
        "use_plot_comparison_tph": False,
        "use_print": False,
        "use_print_result": False,
    }
    return track_opts, solver_opts, driver_opts, sa_opts, debug_opts


class TestLapSimulationBasic:
    """Basic integration tests for lap simulation."""

    def test_simulation_completes(self):
        """Test that simulation completes without errors."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        assert lap is not None

    def test_velocity_profile_valid(self):
        """Test that velocity profile has valid values."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Velocity should be positive everywhere
        assert np.all(lap.vel_cl > 0)

        # Velocity should be below some reasonable maximum (400 km/h)
        assert np.all(lap.vel_cl < 400 / 3.6)

    def test_lap_time_reasonable(self):
        """Test that lap time is in reasonable range."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Shanghai F1 lap time should be around 90-100 seconds
        lap_time = lap.t_cl[-1]
        assert 80 < lap_time < 120

    def test_time_array_monotonic(self):
        """Test that time array is monotonically increasing."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        assert np.all(np.diff(lap.t_cl) > 0)


class TestEnergyManagement:
    """Tests for different energy management strategies."""

    def test_fcfb_strategy(self):
        """Test Full Charge Full Boost strategy."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()
        driver_opts["em_strategy"] = "FCFB"

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        assert lap is not None

    def test_none_strategy(self):
        """Test no hybrid strategy (combustion only)."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()
        driver_opts["em_strategy"] = "NONE"

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        assert lap is not None


# Hardcoded F1_2026 vehicle parameters for the ERS strategy tests — frozen snapshot so INI
# tuning doesn't break them. ERSO/QUALY are designed around the 2026 powertrain (350 kW MGU-K,
# active straight-line harvesting); on low-power hybrids the harvest drag outweighs the
# deployment gain and the strategies are not expected to beat NONE.
ERS_VEHICLE_PARS = {
    "powertrain_type": "hybrid",
    "general": {
        "lf": 1.870, "lr": 1.530, "h_cog": 0.200,
        "sf": 1.620, "sr": 1.525, "m": 820.0, "f_roll": 0.03,
        "c_w_a": 1.16, "c_z_a_f": 2.10, "c_z_a_r": 2.48,
        "g": 9.81, "rho_air": 1.18,
        "active_aero_drag_reduction": 0.20,
        "active_aero_dz_f": 0.20,
        "active_aero_dz_r": 0.30,
        "active_aero_kappa_threshold": 0.01,
    },
    "engine": {
        "topology": "RWD", "pow_max": 400e3, "pow_diff": 25e3,
        "n_begin": 10500.0, "n_max": 11500.0, "n_end": 12500.0,
        "be_max": 100.0, "pow_e_motor": 350e3,
        "p_harvest_straight": 200e3, "ers_harvest_speed_kmh": 280.0,
        "eta_e_motor": 0.9, "eta_e_motor_re": 0.9, "eta_etc_re": 0.0,
        "vel_min_e_motor": 0.0, "torque_e_motor_max": 500.0,
        "ers_speed_limit": True, "max_e_energy_storage": 4.0e6,
        "e_rec_e_motor_max": 8.0e6, "series": "F1_2026",
    },
    "gearbox": {
        "i_trans": [0.05, 0.0809, 0.0983, 0.1188, 0.1417, 0.1622, 0.1873, 0.2161],
        "n_shift": [10500.0, 11879.3, 11879.3, 11879.3, 11879.3, 11879.3, 11879.3, 12500.0],
        "e_i": [1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07],
        "eta_g": 0.96, "diff_lock_ratio": 0.7, "t_shift": 0.025,
    },
    "tires": {
        "f": {"circ_ref": 2.104, "fz_0": 3000.0, "mux": 1.65, "muy": 1.85, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
        "r": {"circ_ref": 2.120, "fz_0": 3000.0, "mux": 1.95, "muy": 2.15, "dmux_dfz": -5.0e-5, "dmuy_dfz": -5.0e-5},
        "tire_model_exp": 2.0,
    },
}


def _run_with_strategy(em_strategy: str, max_no_em_iters: int = None):
    """Run a full lap on Shanghai with the frozen F1_2026 params and the given EM strategy."""
    import main_laptimesim

    track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()
    solver_opts["vehicle"] = None
    solver_opts["custom_vehicle_pars"] = ERS_VEHICLE_PARS
    if max_no_em_iters is not None:
        solver_opts["max_no_em_iters"] = max_no_em_iters
    driver_opts["em_strategy"] = em_strategy
    driver_opts["initial_energy"] = 4.0e6

    return main_laptimesim.main(
        track_opts=track_opts,
        solver_opts=solver_opts,
        driver_opts=driver_opts,
        sa_opts=sa_opts,
        debug_opts=debug_opts,
    )


@pytest.fixture(scope="module")
def erso_lap():
    """Lap simulated with the ERSO strategy (shared across tests)."""
    return _run_with_strategy("ERSO")


@pytest.fixture(scope="module")
def qualy_lap():
    """Lap simulated with the QUALY strategy (shared across tests)."""
    return _run_with_strategy("QUALY")


@pytest.fixture(scope="module")
def none_lap():
    """Lap simulated without hybrid deployment (baseline for comparisons)."""
    return _run_with_strategy("NONE")


@pytest.fixture(scope="module")
def fcfb_lap():
    """Lap simulated with greedy full-charge-full-boost deployment (baseline)."""
    return _run_with_strategy("FCFB")


class TestERSOStrategy:
    """Tests for the ERS-optimized (ERSO) energy management strategy."""

    def test_simulation_completes_and_valid(self, erso_lap):
        assert erso_lap is not None
        assert np.all(erso_lap.vel_cl > 0)
        assert np.all(erso_lap.vel_cl < 400 / 3.6)
        assert np.all(np.diff(erso_lap.t_cl) > 0)
        assert 80 < erso_lap.t_cl[-1] < 120

    def test_recovery_within_limit(self, erso_lap):
        """Total recuperated energy must respect the per-lap harvest cap
        (min of vehicle limit and track FIA limit). Tolerance of one solver
        step because the cap is checked before adding each step's energy."""
        e_rec_total = float(np.sum(erso_lap.e_rec_e_motor))
        assert e_rec_total <= erso_lap.e_rec_e_motor_max + 1e5

    def test_storage_capacity_respected(self, erso_lap):
        """Battery charge must never exceed max_e_energy_storage (4 MJ for F1 2026)."""
        es_max = ERS_VEHICLE_PARS["engine"]["max_e_energy_storage"]
        assert np.max(erso_lap.es_cl) <= es_max + 1e3

    def test_deploy_and_harvest_masks_disjoint(self, erso_lap):
        """The e-motor cannot deploy and harvest at the same point."""
        driverobj = erso_lap.driverobj
        assert not np.any(driverobj.em_boost_use & driverobj.em_harvest_use)

    def test_faster_than_none(self, erso_lap, none_lap):
        """Deploying recovered energy must beat running combustion-only."""
        assert erso_lap.t_cl[-1] < none_lap.t_cl[-1]

    def test_em_iterations_converge(self):
        """The self-consistent harvest/deploy plan must converge: once the EM loop settles,
        extra iterations may not change the result (guards against the budget-feedback
        oscillation that made the lap time depend on max_no_em_iters)."""
        lap_a = _run_with_strategy("ERSO", max_no_em_iters=12)
        lap_b = _run_with_strategy("ERSO", max_no_em_iters=25)
        assert abs(lap_a.t_cl[-1] - lap_b.t_cl[-1]) < 1e-6
        assert np.array_equal(lap_a.driverobj.em_boost_use, lap_b.driverobj.em_boost_use)
        assert np.array_equal(lap_a.driverobj.em_harvest_use, lap_b.driverobj.em_harvest_use)


class TestQualyStrategy:
    """Tests for the qualifying-lap (QUALY) energy management strategy."""

    def test_simulation_completes_and_valid(self, qualy_lap):
        assert qualy_lap is not None
        assert np.all(qualy_lap.vel_cl > 0)
        assert np.all(qualy_lap.vel_cl < 400 / 3.6)
        assert np.all(np.diff(qualy_lap.t_cl) > 0)
        assert 80 < qualy_lap.t_cl[-1] < 120

    def test_recovery_within_limit(self, qualy_lap):
        e_rec_total = float(np.sum(qualy_lap.e_rec_e_motor))
        assert e_rec_total <= qualy_lap.e_rec_e_motor_max + 1e5

    def test_battery_depleted_at_lap_end(self, qualy_lap):
        """A qualifying strategy deploys its starting charge instead of hoarding it."""
        es_initial = qualy_lap.driverobj.pars_driver["initial_energy"]
        assert qualy_lap.es_cl[-1] < es_initial

    def test_storage_capacity_respected(self, qualy_lap):
        """Battery charge must never exceed max_e_energy_storage (4 MJ for F1 2026)."""
        es_max = ERS_VEHICLE_PARS["engine"]["max_e_energy_storage"]
        assert np.max(qualy_lap.es_cl) <= es_max + 1e3

    def test_deploy_and_harvest_masks_disjoint(self, qualy_lap):
        driverobj = qualy_lap.driverobj
        assert not np.any(driverobj.em_boost_use & driverobj.em_harvest_use)

    def test_deploy_respects_curvature_gate(self, qualy_lap):
        """QUALY only deploys on low-curvature sections (default kappa_max_deploy=0.01)."""
        boost_mask = qualy_lap.driverobj.em_boost_use
        kappa_max = qualy_lap.driverobj.pars_driver.get("kappa_max_deploy", 0.01)
        assert np.all(np.abs(qualy_lap.trackobj.kappa[boost_mask]) < kappa_max)

    def test_harvest_respects_curvature_gate(self, qualy_lap):
        """The harvest latch releases in sharp corners, so no harvesting there."""
        harvest_mask = qualy_lap.driverobj.em_harvest_use
        kappa_max = qualy_lap.driverobj.pars_driver.get("kappa_max_deploy", 0.01)
        assert np.all(np.abs(qualy_lap.trackobj.kappa[harvest_mask]) < kappa_max)

    def test_faster_than_none(self, qualy_lap, none_lap):
        assert qualy_lap.t_cl[-1] < none_lap.t_cl[-1]

    def test_faster_than_fcfb(self, qualy_lap, fcfb_lap):
        """The persistence-weighted deploy ranking plus profitable straight-line harvest
        must beat greedy deploy-everywhere on the 2026 powertrain."""
        assert qualy_lap.t_cl[-1] < fcfb_lap.t_cl[-1]


class TestLateralAccelerationGates:
    """ERS deployment and active harvest are both bounded by the lateral acceleration the car
    reaches, not by curvature alone.

    Curvature alone does not say how much grip is left: a 110 m bend passes a kappa < 0.01 gate
    yet pulls over 4 g at 250 km/h, where the driven tires have no spare longitudinal capacity.
    Asking for torque there (either sign) makes the solver answer with a grip-limited braking
    sweep, which showed up as a deploy/harvest limit cycle chattering every few metres
    mid-corner. See ay_max_deploy / ay_max_harvest in laptimesim.src.driver.
    """

    @staticmethod
    def _ay(lap):
        """Lateral acceleration per point, as the solver computes it (unsmoothed)."""
        return np.abs(np.asarray(lap.vel_cl[:-1]) ** 2 * np.asarray(lap.trackobj.kappa))

    def test_qualy_deploy_respects_ay_gate(self, qualy_lap):
        from laptimesim.src.driver import ay_max_deploy

        limit = ay_max_deploy(qualy_lap.driverobj.pars_driver)
        assert np.all(self._ay(qualy_lap)[qualy_lap.driverobj.em_boost_use] <= limit)

    def test_qualy_harvest_respects_ay_gate(self, qualy_lap):
        from laptimesim.src.driver import ay_max_harvest

        limit = ay_max_harvest(qualy_lap.driverobj.pars_driver)
        assert np.all(self._ay(qualy_lap)[qualy_lap.driverobj.em_harvest_use] <= limit)

    def test_erso_deploy_respects_ay_gate(self, erso_lap):
        from laptimesim.src.driver import ay_max_deploy

        limit = ay_max_deploy(erso_lap.driverobj.pars_driver)
        assert np.all(self._ay(erso_lap)[erso_lap.driverobj.em_boost_use] <= limit)

    def test_erso_harvest_respects_ay_gate(self, erso_lap):
        from laptimesim.src.driver import ay_max_harvest

        limit = ay_max_harvest(erso_lap.driverobj.pars_driver)
        assert np.all(self._ay(erso_lap)[erso_lap.driverobj.em_harvest_use] <= limit)

    def test_fcfb_is_exempt_from_deploy_gate(self, fcfb_lap):
        """FCFB is the deliberately naive deploy-everywhere baseline: it has no allocation step
        to spend the saved energy elsewhere, so gating it is a pure loss. Its frozen references
        (tests/test_f1_2026.py, tests/test_laptimesim.py) depend on this staying unchanged."""
        from laptimesim.src.driver import ay_max_deploy

        assert ay_max_deploy(fcfb_lap.driverobj.pars_driver) == np.inf

    def test_gates_are_overridable(self):
        """An explicit driver parameter wins over both the default and the FCFB exemption."""
        from laptimesim.src.driver import (DEFAULT_AY_MAX_DEPLOY, DEFAULT_AY_MAX_HARVEST,
                                           ay_max_deploy, ay_max_harvest)

        assert ay_max_deploy({"em_strategy": "QUALY"}) == DEFAULT_AY_MAX_DEPLOY
        assert ay_max_harvest({"em_strategy": "QUALY"}) == DEFAULT_AY_MAX_HARVEST
        assert ay_max_deploy({"em_strategy": "FCFB", "ay_max_deploy": 25.0}) == 25.0
        assert ay_max_harvest({"ay_max_harvest": 12.5}) == 12.5

    def test_gate_suppresses_deploy_harvest_reversals(self, qualy_lap):
        """The ERS must not flip between deploy and harvest on consecutive steps: that pattern
        is the deploy/harvest limit cycle the gates exist to stop.

        NOTE this covers only ERS-driven chatter. It does NOT detect the separate braking
        sawtooth in fast corners (see TestCorneringLimitTracking) -- that one produces no sign
        reversal at all, because it alternates between coasting and braking regen.
        """
        # e-motor torque: positive = deploying, negative = harvesting (same sign as its power)
        m_e = np.asarray(qualy_lap.m_e_motor)
        sign = np.sign(np.where(np.abs(m_e) < 1e-9, 0.0, m_e))
        reversals = int(np.sum((sign[:-1] * sign[1:]) < 0))
        # a handful of legitimate transitions remain; the ungated solver produced dozens
        assert reversals <= 10, f"{reversals} deploy/harvest reversals -- limit cycle is back"


class TestCorneringLimitTracking:
    """The forward pass has no per-point cornering ceiling, so it cannot track a falling one.

    v_max_cornering() is evaluated reactively, only once the lateral grip check has already
    failed (lap.py CASE 2), rather than being precomputed per point and enforced as a ceiling
    the way trackobj.vel_lim is. Through a tightening bend the solver therefore accelerates
    until the grip check fails, clamps down to v_max_cornering, accelerates again, and so on --
    a 4-5 m sawtooth of +0.5 / -24 m/s^2 instead of a smooth deceleration.

    This is independent of the ERS gates: it reproduces with active harvest disabled entirely
    (ay_max_harvest=0.0), and the -350 kW seen during the dips is braking regen riding along on
    the braking, not harvest. Fixing it means precomputing the ceiling and clamping the forward
    pass to it, which moves every velocity profile and so needs the frozen references
    regenerated -- tracked rather than silently tolerated.
    """

    @pytest.fixture(scope="class")
    def fast_corner_lap(self):
        """MVRC 2026 at Barcelona: a 110 m radius bend taken at ~250 km/h and ~4.4 g."""
        import main_laptimesim

        from helpers.simulation import read_vehicle_params

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()
        track_opts["trackname"] = "BarcelonaGrandPrix_2026"
        track_opts["interp_stepsize_des"] = 1.0
        solver_opts["vehicle"] = None
        solver_opts["custom_vehicle_pars"] = read_vehicle_params("MVRC_2026")
        solver_opts["vel_tol"] = 1e-5
        driver_opts["em_strategy"] = "QUALY"
        driver_opts["initial_energy"] = 4.0e6
        return main_laptimesim.main(
            track_opts=track_opts, solver_opts=solver_opts, driver_opts=driver_opts,
            sa_opts=sa_opts, debug_opts=debug_opts,
        )

    @staticmethod
    def _braking_bursts(lap, lo=1100.0, hi=1160.0, thresh=-5.0):
        """Count separate hard-deceleration events inside a constant-radius bend."""
        d = np.asarray(lap.trackobj.dists_cl)
        v = np.asarray(lap.vel_cl)
        t = np.asarray(lap.t_cl)
        a = np.diff(v) / np.diff(t)
        w = (d[:-1] >= lo) & (d[:-1] <= hi)
        hard = a[w] < thresh
        return int(np.sum(hard[1:] & ~hard[:-1])) + int(hard[0])

    @pytest.mark.xfail(strict=True, reason="forward pass has no precomputed cornering ceiling; "
                                          "see class docstring")
    def test_no_braking_sawtooth_in_fast_corner(self, fast_corner_lap):
        """A constant-radius bend should be one smooth deceleration, not repeated braking."""
        assert self._braking_bursts(fast_corner_lap) <= 1

    def test_sawtooth_does_not_get_worse(self, fast_corner_lap):
        """Regression bound on the known sawtooth: 6 bursts at the time of writing."""
        assert self._braking_bursts(fast_corner_lap) <= 6


class TestHarvestCostBenefit:
    """The harvest cost-benefit gate must keep ERSO/QUALY from losing to NONE even on
    low-power hybrids (F1_Shanghai: 120 kW MGU-K), where unconstrained straight-line
    harvest drag used to outweigh the deployment gain."""

    @pytest.fixture(scope="class")
    def low_power_laps(self):
        import main_laptimesim

        laps = {}
        for strategy in ["NONE", "ERSO", "QUALY"]:
            track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()
            driver_opts["em_strategy"] = strategy
            laps[strategy] = main_laptimesim.main(
                track_opts=track_opts,
                solver_opts=solver_opts,
                driver_opts=driver_opts,
                sa_opts=sa_opts,
                debug_opts=debug_opts,
            )
        return laps

    def test_erso_not_slower_than_none(self, low_power_laps):
        assert low_power_laps["ERSO"].t_cl[-1] < low_power_laps["NONE"].t_cl[-1]

    def test_qualy_not_slower_than_none(self, low_power_laps):
        assert low_power_laps["QUALY"].t_cl[-1] < low_power_laps["NONE"].t_cl[-1]


class TestDRSEffect:
    """Tests for DRS effect on lap time."""

    def test_drs_improves_lap_time(self):
        """Test that DRS reduces lap time."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        # With DRS
        track_opts["use_drs"] = True
        lap_with_drs = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Without DRS
        track_opts["use_drs"] = False
        lap_without_drs = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # DRS should reduce lap time
        assert lap_with_drs.t_cl[-1] < lap_without_drs.t_cl[-1]


class TestWeatherEffect:
    """Tests for weather/friction effect on lap time."""

    def test_wet_track_slower(self):
        """Test that wet conditions increase lap time."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        # Dry conditions
        track_opts["mu_weather"] = 1.0
        lap_dry = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Wet conditions
        track_opts["mu_weather"] = 0.8
        lap_wet = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Wet track should be slower
        assert lap_wet.t_cl[-1] > lap_dry.t_cl[-1]


class TestVelocityLimits:
    """Tests for velocity limit handling."""

    def test_global_velocity_limit(self):
        """Test that global velocity limit is respected."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        # Set a low velocity limit
        vel_limit = 200 / 3.6  # 200 km/h in m/s
        driver_opts["vel_lim_glob"] = vel_limit

        lap = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Velocity should not exceed limit (with small tolerance for numerical reasons)
        assert np.all(lap.vel_cl <= vel_limit + 0.1)


class TestYellowFlags:
    """Tests for yellow flag handling."""

    def test_yellow_flag_slows_sector(self):
        """Test that yellow flag in a sector increases lap time."""
        import main_laptimesim

        track_opts, solver_opts, driver_opts, sa_opts, debug_opts = get_default_opts()

        # No yellow flags
        driver_opts["yellow_s1"] = False
        driver_opts["yellow_s2"] = False
        driver_opts["yellow_s3"] = False
        lap_clear = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Yellow flag in sector 1
        driver_opts["yellow_s1"] = True
        lap_yellow = main_laptimesim.main(
            track_opts=track_opts,
            solver_opts=solver_opts,
            driver_opts=driver_opts,
            sa_opts=sa_opts,
            debug_opts=debug_opts,
        )

        # Yellow flag should increase lap time
        assert lap_yellow.t_cl[-1] > lap_clear.t_cl[-1]
