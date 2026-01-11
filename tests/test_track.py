"""
Unit tests for the Track class in laptimesim.

Tests cover:
- Track loading and initialization
- Raceline interpolation
- Curvature calculations
- DRS zone handling
- Friction coefficient handling
"""

import pytest
import numpy as np
import os


@pytest.fixture
def repo_path():
    """Get the repository root path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_pars_track():
    """Sample track parameters for testing."""
    return {
        "trackname": "Shanghai",
        "flip_track": False,
        "mu_weather": 1.0,
        "interp_stepsize_des": 5.0,
        "curv_filt_width": 10.0,
        "use_drs1": True,
        "use_drs2": True,
        "use_pit": False,
    }


@pytest.fixture
def track_instance(repo_path, sample_pars_track):
    """Create a Track instance for testing."""
    from laptimesim.src.track import Track

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")
    trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines", "Shanghai.csv")

    return Track(
        pars_track=sample_pars_track,
        parfilepath=parfilepath,
        trackfilepath=trackfilepath,
        vel_lim_glob=np.inf,
    )


class TestTrackInitialization:
    """Tests for Track class initialization."""

    def test_track_creates_successfully(self, track_instance):
        """Test that Track object is created with valid parameters."""
        assert track_instance is not None

    def test_track_has_raceline(self, track_instance):
        """Test that track has a raceline after initialization."""
        assert track_instance.raceline is not None
        assert len(track_instance.raceline) > 0

    def test_track_has_curvature(self, track_instance):
        """Test that track has curvature values."""
        assert track_instance.kappa is not None
        assert len(track_instance.kappa) > 0

    def test_track_has_friction(self, track_instance):
        """Test that track has friction values."""
        assert track_instance.mu is not None
        assert len(track_instance.mu) > 0

    def test_no_points_matches_arrays(self, track_instance):
        """Test that no_points matches the length of track arrays."""
        assert track_instance.no_points == len(track_instance.kappa)
        assert track_instance.no_points == len(track_instance.mu)


class TestTrackCurvature:
    """Tests for track curvature calculations."""

    def test_curvature_reasonable_range(self, track_instance):
        """Test that curvature values are in reasonable range."""
        # Curvature should be within reasonable limits for a race track
        # Typical values: -0.1 to 0.1 rad/m (radius > 10m)
        assert np.all(np.abs(track_instance.kappa) < 0.2)

    def test_curvature_not_all_zero(self, track_instance):
        """Test that curvature has non-zero values (track has turns)."""
        assert np.any(track_instance.kappa != 0)

    def test_curvature_length_matches_points(self, track_instance):
        """Test that curvature array matches number of track points."""
        assert len(track_instance.kappa) == track_instance.no_points


class TestTrackFriction:
    """Tests for track friction coefficient handling."""

    def test_friction_in_valid_range(self, track_instance):
        """Test that friction values are in valid range (0.5-1.3)."""
        assert np.all(track_instance.mu >= 0.5)
        assert np.all(track_instance.mu <= 1.3)

    def test_friction_affected_by_weather(self, repo_path):
        """Test that mu_weather parameter affects friction values."""
        from laptimesim.src.track import Track

        parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")
        trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines", "Shanghai.csv")

        # Create track with dry conditions
        pars_dry = {
            "trackname": "Shanghai",
            "flip_track": False,
            "mu_weather": 1.0,
            "interp_stepsize_des": 5.0,
            "curv_filt_width": 10.0,
            "use_drs1": False,
            "use_drs2": False,
            "use_pit": False,
        }
        track_dry = Track(pars_track=pars_dry, parfilepath=parfilepath, trackfilepath=trackfilepath)

        # Create track with wet conditions
        pars_wet = {
            "trackname": "Shanghai",
            "flip_track": False,
            "mu_weather": 0.7,
            "interp_stepsize_des": 5.0,
            "curv_filt_width": 10.0,
            "use_drs1": False,
            "use_drs2": False,
            "use_pit": False,
        }
        track_wet = Track(pars_track=pars_wet, parfilepath=parfilepath, trackfilepath=trackfilepath)

        # Wet track should have lower friction
        assert np.mean(track_wet.mu) < np.mean(track_dry.mu)


class TestTrackDRS:
    """Tests for DRS zone handling."""

    def test_drs_array_correct_length(self, track_instance):
        """Test that DRS array has correct length."""
        assert len(track_instance.drs) == track_instance.no_points

    def test_drs_is_boolean_array(self, track_instance):
        """Test that DRS array contains boolean values."""
        assert track_instance.drs.dtype == bool

    def test_drs_zones_exist_when_enabled(self, track_instance):
        """Test that DRS zones exist when DRS is enabled."""
        # Shanghai should have DRS zones
        assert np.any(track_instance.drs)


class TestTrackDistances:
    """Tests for track distance calculations."""

    def test_distances_monotonically_increasing(self, track_instance):
        """Test that closed distances are monotonically increasing."""
        dists = track_instance.dists_cl
        assert np.all(np.diff(dists) > 0)

    def test_distances_start_at_zero(self, track_instance):
        """Test that distances start at zero."""
        assert track_instance.dists_cl[0] == 0.0

    def test_track_length_reasonable(self, track_instance):
        """Test that track length is reasonable (Shanghai ~5.4km)."""
        track_length = track_instance.dists_cl[-1]
        # Shanghai is approximately 5451m
        assert 5000 < track_length < 6000


class TestTrackStepsize:
    """Tests for track stepsize after interpolation."""

    def test_stepsize_positive(self, track_instance):
        """Test that stepsize is positive."""
        assert track_instance.stepsize > 0

    def test_stepsize_close_to_desired(self, track_instance, sample_pars_track):
        """Test that actual stepsize is close to desired stepsize."""
        desired = sample_pars_track["interp_stepsize_des"]
        actual = track_instance.stepsize
        # Should be within 10% of desired
        assert abs(actual - desired) / desired < 0.1


class TestTrackZones:
    """Tests for track zone/sector handling."""

    def test_zone_indices_exist(self, track_instance):
        """Test that zone indices dictionary exists."""
        assert hasattr(track_instance, "zone_inds")
        assert isinstance(track_instance.zone_inds, dict)


class TestTrackFlipping:
    """Tests for track direction flipping."""

    def test_flipped_track_same_length(self, repo_path):
        """Test that flipped track has same length."""
        from laptimesim.src.track import Track

        parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")
        trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines", "Shanghai.csv")

        pars_normal = {
            "trackname": "Shanghai",
            "flip_track": False,
            "mu_weather": 1.0,
            "interp_stepsize_des": 5.0,
            "curv_filt_width": 10.0,
            "use_drs1": False,
            "use_drs2": False,
            "use_pit": False,
        }

        pars_flipped = {
            "trackname": "Shanghai",
            "flip_track": True,
            "mu_weather": 1.0,
            "interp_stepsize_des": 5.0,
            "curv_filt_width": 10.0,
            "use_drs1": False,
            "use_drs2": False,
            "use_pit": False,
        }

        track_normal = Track(pars_track=pars_normal, parfilepath=parfilepath, trackfilepath=trackfilepath)
        track_flipped = Track(pars_track=pars_flipped, parfilepath=parfilepath, trackfilepath=trackfilepath)

        # Track lengths should be approximately the same
        assert np.isclose(track_normal.dists_cl[-1], track_flipped.dists_cl[-1], rtol=0.01)


class TestVelocityLimits:
    """Tests for velocity limit handling."""

    def test_vel_lim_array_correct_length(self, track_instance):
        """Test that velocity limit array has correct length."""
        assert len(track_instance.vel_lim) == track_instance.no_points

    def test_vel_lim_positive(self, track_instance):
        """Test that velocity limits are positive."""
        assert np.all(track_instance.vel_lim > 0)
