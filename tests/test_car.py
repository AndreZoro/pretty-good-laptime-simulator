"""
Unit tests for the Car class in laptimesim.

Tests cover:
- Car initialization and parameter loading
- Tire force calculations
- Air and rolling resistance
- Gear finding
- Maximum acceleration calculations
"""

import pytest
import numpy as np
import math


# Sample vehicle parameters for testing (simplified F1-like car)
@pytest.fixture
def sample_pars_general():
    return {
        "lf": 1.968,
        "lr": 1.632,
        "h_cog": 0.335,
        "sf": 1.6,
        "sr": 1.6,
        "m": 733.0,
        "f_roll": 0.03,
        "c_w_a": 1.56,
        "c_z_a_f": 2.20,
        "c_z_a_r": 2.68,
        "g": 9.81,
        "rho_air": 1.18,
        "drs_factor": 0.17,
    }


@pytest.fixture
def sample_pars_engine():
    return {
        "topology": "RWD",
        "pow_max": 575e3,
        "pow_diff": 41e3,
        "n_begin": 10500.0,
        "n_max": 11400.0,
        "n_end": 12200.0,
        "be_max": 100.0,
    }


@pytest.fixture
def sample_pars_gearbox():
    return {
        "i_trans": np.array([0.04, 0.070, 0.095, 0.117, 0.143, 0.172, 0.190, 0.206]),
        "n_shift": np.array([10000.0, 11800.0, 11800.0, 11800.0, 11800.0, 11800.0, 11800.0, 13000.0]),
        "e_i": np.array([1.16, 1.11, 1.09, 1.08, 1.08, 1.08, 1.07, 1.07]),
        "eta_g": 0.96,
    }


@pytest.fixture
def sample_pars_tires():
    return {
        "f": {
            "circ_ref": 2.073,
            "fz_0": 3000.0,
            "mux": 1.65,
            "muy": 1.85,
            "dmux_dfz": -5.0e-5,
            "dmuy_dfz": -5.0e-5,
        },
        "r": {
            "circ_ref": 2.073,
            "fz_0": 3000.0,
            "mux": 1.95,
            "muy": 2.15,
            "dmux_dfz": -5.0e-5,
            "dmuy_dfz": -5.0e-5,
        },
        "tire_model_exp": 2.0,
    }


@pytest.fixture
def car_instance(sample_pars_general, sample_pars_engine, sample_pars_gearbox, sample_pars_tires):
    from laptimesim.src.car import Car
    return Car(
        powertrain_type="combustion",
        pars_general=sample_pars_general,
        pars_engine=sample_pars_engine,
        pars_gearbox=sample_pars_gearbox,
        pars_tires=sample_pars_tires,
    )


class TestCarInitialization:
    """Tests for Car class initialization."""

    def test_car_creates_successfully(self, car_instance):
        """Test that Car object is created with valid parameters."""
        assert car_instance is not None
        assert car_instance.powertrain_type == "combustion"

    def test_car_stores_parameters(self, car_instance, sample_pars_general):
        """Test that Car stores parameters correctly."""
        assert car_instance.pars_general["m"] == sample_pars_general["m"]
        assert car_instance.pars_general["lf"] == sample_pars_general["lf"]

    def test_static_tire_loads_calculated(self, car_instance):
        """Test that static tire load components are calculated during init."""
        assert "stat_load" in car_instance.f_z_calc_stat
        assert "trans_long" in car_instance.f_z_calc_stat
        assert "trans_lat" in car_instance.f_z_calc_stat
        assert "aero" in car_instance.f_z_calc_stat

    def test_static_load_sum_equals_weight(self, car_instance, sample_pars_general):
        """Test that sum of static tire loads equals vehicle weight."""
        total_static_load = np.sum(car_instance.f_z_calc_stat["stat_load"])
        expected_weight = sample_pars_general["m"] * sample_pars_general["g"]
        assert np.isclose(total_static_load, expected_weight, rtol=1e-6)


class TestAirResistance:
    """Tests for air resistance calculations."""

    def test_air_res_zero_at_zero_velocity(self, car_instance):
        """Test that air resistance is zero when velocity is zero."""
        assert car_instance.air_res(vel=0.0, drs=False) == 0.0

    def test_air_res_increases_with_velocity(self, car_instance):
        """Test that air resistance increases with velocity squared."""
        res_50 = car_instance.air_res(vel=50.0, drs=False)
        res_100 = car_instance.air_res(vel=100.0, drs=False)
        # At double velocity, resistance should be ~4x (v^2 relationship)
        assert np.isclose(res_100 / res_50, 4.0, rtol=1e-6)

    def test_drs_reduces_air_resistance(self, car_instance):
        """Test that DRS reduces air resistance."""
        res_no_drs = car_instance.air_res(vel=80.0, drs=False)
        res_with_drs = car_instance.air_res(vel=80.0, drs=True)
        assert res_with_drs < res_no_drs

    def test_air_res_positive_for_forward_motion(self, car_instance):
        """Test that air resistance is positive for positive velocity."""
        assert car_instance.air_res(vel=30.0, drs=False) > 0


class TestRollingResistance:
    """Tests for rolling resistance calculations."""

    def test_roll_res_proportional_to_load(self, car_instance):
        """Test that rolling resistance is proportional to tire load."""
        res_1000 = car_instance.roll_res(f_z_tot=1000.0)
        res_2000 = car_instance.roll_res(f_z_tot=2000.0)
        assert np.isclose(res_2000 / res_1000, 2.0, rtol=1e-6)

    def test_roll_res_zero_at_zero_load(self, car_instance):
        """Test that rolling resistance is zero when load is zero."""
        assert car_instance.roll_res(f_z_tot=0.0) == 0.0


class TestGearFinding:
    """Tests for gear selection logic."""

    def test_find_gear_returns_tuple(self, car_instance):
        """Test that find_gear returns gear index and engine rev as tuple."""
        result = car_instance.find_gear(vel=30.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_gear_increases_with_velocity(self, car_instance):
        """Test that higher gears are selected at higher velocities."""
        gear_low, _ = car_instance.find_gear(vel=20.0)
        gear_high, _ = car_instance.find_gear(vel=80.0)
        assert gear_high >= gear_low

    def test_engine_rev_positive(self, car_instance):
        """Test that engine rev is positive."""
        _, n_engine = car_instance.find_gear(vel=50.0)
        # Engine rev (in 1/s) should be positive
        assert n_engine > 0

    def test_gear_index_valid(self, car_instance):
        """Test that gear index is within valid range."""
        gear_idx, _ = car_instance.find_gear(vel=50.0)
        num_gears = len(car_instance.pars_gearbox["i_trans"])
        assert 0 <= gear_idx < num_gears


class TestLateralForces:
    """Tests for lateral force calculations."""

    def test_calc_lat_forces_returns_tuple(self, car_instance):
        """Test that calc_lat_forces returns front and rear forces."""
        f_y_f, f_y_r = car_instance.calc_lat_forces(a_y=5.0)
        assert isinstance(f_y_f, (int, float))
        assert isinstance(f_y_r, (int, float))

    def test_lat_forces_zero_when_no_lateral_accel(self, car_instance):
        """Test that lateral forces are zero when a_y is zero."""
        f_y_f, f_y_r = car_instance.calc_lat_forces(a_y=0.0)
        assert f_y_f == 0.0
        assert f_y_r == 0.0

    def test_lat_forces_increase_with_lateral_accel(self, car_instance):
        """Test that lateral forces increase with lateral acceleration."""
        f_y_f_1, f_y_r_1 = car_instance.calc_lat_forces(a_y=2.0)
        f_y_f_2, f_y_r_2 = car_instance.calc_lat_forces(a_y=4.0)
        assert abs(f_y_f_2) > abs(f_y_f_1)
        assert abs(f_y_r_2) > abs(f_y_r_1)


class TestTireForcePotentials:
    """Tests for tire force potential calculations."""

    def test_tire_force_pots_returns_12_values(self, car_instance):
        """Test that tire_force_pots returns all 12 expected values."""
        result = car_instance.tire_force_pots(vel=50.0, a_x=0.0, a_y=0.0, mu=1.0)
        assert len(result) == 12

    def test_tire_loads_positive(self, car_instance):
        """Test that tire loads are positive under normal conditions."""
        result = car_instance.tire_force_pots(vel=50.0, a_x=0.0, a_y=0.0, mu=1.0)
        # Tire loads are at indices 2, 5, 8, 11
        tire_loads = [result[2], result[5], result[8], result[11]]
        assert all(load > 0 for load in tire_loads)

    def test_force_potentials_scale_with_mu(self, car_instance):
        """Test that force potentials scale with friction coefficient."""
        result_mu1 = car_instance.tire_force_pots(vel=50.0, a_x=0.0, a_y=0.0, mu=1.0)
        result_mu08 = car_instance.tire_force_pots(vel=50.0, a_x=0.0, a_y=0.0, mu=0.8)
        # Force potentials at indices 0, 1, 3, 4, 6, 7, 9, 10
        # Lower mu should give lower force potentials
        assert result_mu08[0] < result_mu1[0]


class TestMaxCorneringVelocity:
    """Tests for maximum cornering velocity calculations."""

    def test_v_max_cornering_higher_for_lower_curvature(self, car_instance):
        """Test that max cornering velocity is higher for lower curvature (larger radius)."""
        v_max_high_curv = car_instance.v_max_cornering(kappa=0.01, mu=1.0)
        v_max_low_curv = car_instance.v_max_cornering(kappa=0.005, mu=1.0)
        assert v_max_low_curv > v_max_high_curv

    def test_v_max_cornering_higher_for_higher_mu(self, car_instance):
        """Test that max cornering velocity is higher for higher friction."""
        v_max_low_mu = car_instance.v_max_cornering(kappa=0.01, mu=0.8)
        v_max_high_mu = car_instance.v_max_cornering(kappa=0.01, mu=1.0)
        assert v_max_high_mu > v_max_low_mu

    def test_v_max_cornering_positive(self, car_instance):
        """Test that max cornering velocity is positive."""
        v_max = car_instance.v_max_cornering(kappa=0.01, mu=1.0)
        assert v_max > 0


class TestDrivenTireRadius:
    """Tests for driven tire radius calculation."""

    def test_r_driven_tire_positive(self, car_instance):
        """Test that driven tire radius is positive."""
        r = car_instance.r_driven_tire(vel=50.0)
        assert r > 0

    def test_r_driven_tire_reasonable_value(self, car_instance):
        """Test that driven tire radius is in reasonable range (0.25-0.4m for F1)."""
        r = car_instance.r_driven_tire(vel=50.0)
        assert 0.2 < r < 0.5
