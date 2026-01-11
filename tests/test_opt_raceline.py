"""
Tests for opt_raceline module.
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock

import opt_raceline.src.import_csv_track as import_csv_track


class TestImportCsvTrack:
    """Tests for import_csv_track function."""

    @pytest.fixture
    def sample_track_4col(self, tmp_path):
        """Create a sample 4-column track CSV file."""
        track_data = np.array([
            [0.0, 0.0, 5.0, 5.0],
            [10.0, 0.0, 5.0, 5.0],
            [20.0, 5.0, 4.5, 4.5],
            [30.0, 10.0, 5.0, 5.0],
            [40.0, 10.0, 5.5, 5.5],
        ])
        filepath = tmp_path / "test_track_4col.csv"
        np.savetxt(filepath, track_data, delimiter=',', header="x_m,y_m,w_tr_right_m,w_tr_left_m")
        return str(filepath), track_data

    @pytest.fixture
    def sample_track_3col(self, tmp_path):
        """Create a sample 3-column track CSV file (total width only)."""
        track_data = np.array([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
            [20.0, 5.0, 9.0],
            [30.0, 10.0, 10.0],
            [40.0, 10.0, 11.0],
        ])
        filepath = tmp_path / "test_track_3col.csv"
        np.savetxt(filepath, track_data, delimiter=',', header="x_m,y_m,w_tr_total_m")
        return str(filepath), track_data

    def test_import_4_column_track(self, sample_track_4col):
        """Test importing a standard 4-column track file."""
        filepath, expected = sample_track_4col

        result = import_csv_track.import_csv_track(filepath)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 4)
        np.testing.assert_array_almost_equal(result, expected)

    def test_import_3_column_track_converts_to_4(self, sample_track_3col):
        """Test that 3-column track is converted to 4 columns with split width."""
        filepath, original = sample_track_3col

        result = import_csv_track.import_csv_track(filepath)

        assert result.shape == (5, 4)
        # x, y should be unchanged
        np.testing.assert_array_almost_equal(result[:, :2], original[:, :2])
        # width should be split in half
        np.testing.assert_array_almost_equal(result[:, 2], original[:, 2] / 2)
        np.testing.assert_array_almost_equal(result[:, 3], original[:, 2] / 2)

    def test_import_real_track_file(self, repo_path):
        """Test importing an actual track file from the repository."""
        trackpath = os.path.join(repo_path, "opt_raceline", "input", "tracks", "Austin.csv")

        if not os.path.exists(trackpath):
            pytest.skip("Austin.csv track file not found")

        result = import_csv_track.import_csv_track(trackpath)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4
        assert result.shape[0] > 10  # Should have many track points
        # Track widths should be positive
        assert np.all(result[:, 2] > 0)
        assert np.all(result[:, 3] > 0)

    def test_import_nonexistent_file_raises_error(self):
        """Test that importing a non-existent file raises an error."""
        with pytest.raises(Exception):
            import_csv_track.import_csv_track("/nonexistent/path/track.csv")

    def test_track_coordinates_are_numeric(self, sample_track_4col):
        """Test that all loaded values are numeric."""
        filepath, _ = sample_track_4col

        result = import_csv_track.import_csv_track(filepath)

        assert result.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestImportGeojsonGpsCenterline:
    """Tests for import_geojson_gps_centerline function."""

    @pytest.fixture
    def sample_geojson(self, tmp_path):
        """Create a sample GeoJSON file with GPS coordinates."""
        import json

        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [11.3, 50.9],  # lon, lat format
                            [11.31, 50.91],
                            [11.32, 50.92],
                        ]
                    }
                }
            ]
        }
        filepath = tmp_path / "test_centerline.geojson"
        with open(filepath, 'w') as f:
            json.dump(geojson_data, f)
        return str(filepath)

    @pytest.mark.skip(reason="Requires interactive matplotlib GUI")
    def test_import_geojson_returns_array(self, sample_geojson):
        """Test that importing GeoJSON returns a numpy array."""
        from opt_raceline.src.import_geojson_gps_centerline import import_geojson_gps_centerline

        result = import_geojson_gps_centerline(sample_geojson)
        assert isinstance(result, np.ndarray)

    def test_geojson_file_exists_in_repo(self, repo_path):
        """Test that expected GeoJSON files exist in the repository."""
        centerlines_path = os.path.join(repo_path, "opt_raceline", "input", "centerlines")

        if not os.path.exists(centerlines_path):
            pytest.skip("Centerlines directory not found")

        geojson_files = [f for f in os.listdir(centerlines_path) if f.endswith('.geojson')]
        assert len(geojson_files) > 0, "No GeoJSON centerline files found"


class TestPlotTrack:
    """Tests for plot_track function."""

    @pytest.fixture
    def sample_tracks(self):
        """Create sample track data for plotting tests."""
        # Create a simple circular track
        n_points = 50
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r = 100  # radius in meters

        x = r * np.cos(t)
        y = r * np.sin(t)
        w_right = np.ones(n_points) * 5.0
        w_left = np.ones(n_points) * 5.0

        track = np.column_stack([x, y, w_right, w_left])
        return track, track.copy()  # track_imp, track_interp

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_track_runs_without_error(self, mock_savefig, mock_show, sample_tracks):
        """Test that plot_track can be called without errors."""
        from opt_raceline.src.plot_track import plot_track

        track_imp, track_interp = sample_tracks

        # Should not raise any exceptions
        plot_track(track_imp=track_imp, track_interp=track_interp)

        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_track_saves_to_file(self, mock_savefig, mock_show, sample_tracks, tmp_path):
        """Test that plot_track saves plot when filepath is provided."""
        from opt_raceline.src.plot_track import plot_track

        track_imp, track_interp = sample_tracks
        output_path = str(tmp_path / "test_plot.png")

        plot_track(
            track_imp=track_imp,
            track_interp=track_interp,
            filepath_tr_plot=output_path
        )

        mock_savefig.assert_called_once_with(output_path, dpi=250)

    def test_plot_track_input_validation(self, sample_tracks):
        """Test that plot_track handles invalid input appropriately."""
        from opt_raceline.src.plot_track import plot_track

        # Empty arrays should cause issues
        with pytest.raises(Exception):
            plot_track(track_imp=np.array([]), track_interp=np.array([]))


class TestIntegration:
    """Integration tests for opt_raceline module."""

    def test_csv_tracks_available(self, repo_path):
        """Test that CSV track files are available and can be loaded."""
        tracks_path = os.path.join(repo_path, "opt_raceline", "input", "tracks")

        if not os.path.exists(tracks_path):
            pytest.skip("Tracks directory not found")

        csv_files = [f for f in os.listdir(tracks_path) if f.endswith('.csv')]
        assert len(csv_files) > 0, "No CSV track files found"

        # Try loading each track
        for csv_file in csv_files[:3]:  # Test first 3 to keep it fast
            filepath = os.path.join(tracks_path, csv_file)
            result = import_csv_track.import_csv_track(filepath)
            assert result.shape[1] == 4, f"Track {csv_file} should have 4 columns"
            assert result.shape[0] > 10, f"Track {csv_file} should have multiple points"

    def test_module_imports(self):
        """Test that opt_raceline module can be imported."""
        import opt_raceline
        import opt_raceline.src
        import opt_raceline.src.import_csv_track
        import opt_raceline.src.plot_track

        assert hasattr(opt_raceline, 'src')

    def test_track_data_consistency(self, repo_path):
        """Test that track data is consistent (positive widths, closed loop approximation)."""
        tracks_path = os.path.join(repo_path, "opt_raceline", "input", "tracks")

        if not os.path.exists(tracks_path):
            pytest.skip("Tracks directory not found")

        trackpath = os.path.join(tracks_path, "Austin.csv")
        if not os.path.exists(trackpath):
            pytest.skip("Austin.csv not found")

        track = import_csv_track.import_csv_track(trackpath)

        # All widths should be positive
        assert np.all(track[:, 2] > 0), "Right track width should be positive"
        assert np.all(track[:, 3] > 0), "Left track width should be positive"

        # Track should form approximately a closed loop
        start_point = track[0, :2]
        end_point = track[-1, :2]
        distance = np.linalg.norm(end_point - start_point)
        # For a closed track, start and end shouldn't be too far apart
        # relative to track length
        track_length = np.sum(np.linalg.norm(np.diff(track[:, :2], axis=0), axis=1))
        assert distance < track_length * 0.1, "Track should be approximately closed"
