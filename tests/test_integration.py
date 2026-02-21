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
