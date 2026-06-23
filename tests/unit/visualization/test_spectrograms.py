"""
Tests for spectrogram visualization module.

Tests the SpectrogramVisualizationPipeline for detail scan visualization
including heatmaps, band ratio maps, and PCA component maps.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from sherloc_pipeline.visualization.spectrograms import (
    SpectrogramVisualizationConfig,
    SpectrogramVisualizationPipeline,
    SpatialSpectralData,
)


class TestSpectrogramVisualizationConfig:
    """Tests for SpectrogramVisualizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpectrogramVisualizationConfig()

        assert config.grid_width == 10
        assert config.grid_height == 10
        assert config.n_components == 5
        assert config.export_png is True
        assert config.export_html is True
        assert config.dpi == 300

    def test_default_wavenumbers(self):
        """Test default target wavenumbers are set."""
        config = SpectrogramVisualizationConfig()

        assert config.target_wavenumbers is not None
        assert len(config.target_wavenumbers) > 0
        # Should include common Mars mineral bands
        assert 1086 in config.target_wavenumbers  # Carbonate
        assert 816 in config.target_wavenumbers   # Olivine

    def test_default_band_ratios(self):
        """Test default band ratios are set."""
        config = SpectrogramVisualizationConfig()

        assert config.band_ratios is not None
        assert len(config.band_ratios) >= 3

        # Check structure of band ratios
        for ratio in config.band_ratios:
            assert "name" in ratio
            assert "numerator" in ratio
            assert "denominator" in ratio

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpectrogramVisualizationConfig(
            grid_width=5,
            grid_height=5,
            n_components=3,
            dpi=150
        )

        assert config.grid_width == 5
        assert config.grid_height == 5
        assert config.n_components == 3
        assert config.dpi == 150


class TestSpatialSpectralData:
    """Tests for SpatialSpectralData container."""

    def test_creation(self):
        """Test SpatialSpectralData creation."""
        spectra = np.random.rand(100, 1000)
        wavenumbers = np.linspace(200, 4000, 1000)
        x_coords = np.repeat(np.arange(10), 10)
        y_coords = np.tile(np.arange(10), 10)

        data = SpatialSpectralData(
            spectra=spectra,
            wavenumbers=wavenumbers,
            x_coords=x_coords,
            y_coords=y_coords,
            scan_name="test_scan",
            target="Mars",
            metadata={"sol_number": 100}
        )

        assert data.spectra.shape == (100, 1000)
        assert len(data.wavenumbers) == 1000
        assert len(data.x_coords) == 100
        assert len(data.y_coords) == 100
        assert data.scan_name == "test_scan"
        assert data.target == "Mars"


class TestSpectrogramVisualizationPipeline:
    """Tests for SpectrogramVisualizationPipeline."""

    @pytest.fixture
    def mock_scan_data(self):
        """Create mock scan data for testing."""
        np.random.seed(42)
        spectra = np.random.rand(100, 1000)
        wavenumbers = np.linspace(200, 4000, 1000)
        x_coords = np.repeat(np.arange(10), 10).astype(float)
        y_coords = np.tile(np.arange(10), 10).astype(float)

        return SpatialSpectralData(
            spectra=spectra,
            wavenumbers=wavenumbers,
            x_coords=x_coords,
            y_coords=y_coords,
            scan_name="test_scan",
            target="TestTarget",
            metadata={"sol_number": 100, "n_points": 100}
        )

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default config."""
        pipeline = SpectrogramVisualizationPipeline()
        assert pipeline.config is not None
        assert pipeline.config.grid_width == 10

    def test_pipeline_custom_config(self):
        """Test pipeline initialization with custom config."""
        config = SpectrogramVisualizationConfig(grid_width=5)
        pipeline = SpectrogramVisualizationPipeline(config)
        assert pipeline.config.grid_width == 5

    def test_create_wavenumber_heatmaps(self, mock_scan_data):
        """Test wavenumber heatmap generation."""
        config = SpectrogramVisualizationConfig(
            target_wavenumbers=[500, 1000, 1500],
            export_png=True
        )
        pipeline = SpectrogramVisualizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = pipeline.create_wavenumber_heatmaps(mock_scan_data, output_dir)

            # Should create one heatmap per wavenumber
            assert len(paths) == 3
            for path in paths:
                assert path.exists()
                assert path.suffix == ".png"

    def test_create_band_ratio_maps(self, mock_scan_data):
        """Test band ratio map generation."""
        config = SpectrogramVisualizationConfig(
            band_ratios=[
                {"name": "Test_Ratio", "numerator": [800, 850], "denominator": [900, 950]}
            ],
            export_png=True
        )
        pipeline = SpectrogramVisualizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = pipeline.create_band_ratio_maps(mock_scan_data, output_dir)

            # Should create one ratio map
            assert len(paths) == 1
            assert paths[0].exists()
            assert "ratio_Test_Ratio" in str(paths[0])

    def test_create_pca_component_maps(self, mock_scan_data):
        """Test PCA component map generation."""
        config = SpectrogramVisualizationConfig(
            n_components=3,
            export_png=True
        )
        pipeline = SpectrogramVisualizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths, pca_results = pipeline.create_pca_component_maps(mock_scan_data, output_dir)

            # Should create one map per component
            assert len(paths) == 3
            for path in paths:
                assert path.exists()
                assert "PCA" in str(path)

            # Check PCA results structure
            assert "explained_variance_ratio" in pca_results
            assert len(pca_results["explained_variance_ratio"]) == 3
            assert "total_variance_explained" in pca_results
            assert pca_results["total_variance_explained"] > 0

    def test_create_interactive_html(self, mock_scan_data):
        """Test interactive HTML generation."""
        config = SpectrogramVisualizationConfig(
            target_wavenumbers=[500, 1000],
            n_components=2,
            export_html=True
        )
        pipeline = SpectrogramVisualizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"
            result = pipeline.create_interactive_html(
                mock_scan_data,
                output_path,
                pca_results={"dummy": True}
            )

            assert result.exists()
            assert result.suffix == ".html"
            # HTML should be non-empty
            assert result.stat().st_size > 1000


class TestSpectrogramIntegration:
    """Integration tests for spectrogram visualization (requires database)."""

    @pytest.mark.skipif(
        not Path("./phase.db").exists(),
        reason="Database not available"
    )
    def test_load_detail_scan_data(self):
        """Test loading real detail scan data from database."""
        pipeline = SpectrogramVisualizationPipeline()

        # Load a small number of scans
        scan_data_list = pipeline.load_detail_scan_data()

        # Should find some detail scans
        assert len(scan_data_list) > 0

        # Check structure of first scan
        scan = scan_data_list[0]
        assert hasattr(scan, "spectra")
        assert hasattr(scan, "wavenumbers")
        assert hasattr(scan, "x_coords")
        assert hasattr(scan, "y_coords")
        assert hasattr(scan, "scan_name")
        assert hasattr(scan, "target")

    @pytest.mark.skipif(
        not Path("./phase.db").exists(),
        reason="Database not available"
    )
    def test_generate_visualization_suite(self):
        """Test full visualization suite generation with real data."""
        config = SpectrogramVisualizationConfig(
            target_wavenumbers=[500, 1000],  # Reduced for speed
            n_components=2,
            export_html=False  # Skip HTML for speed
        )
        pipeline = SpectrogramVisualizationPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run on a specific scan
            results = pipeline.generate_visualization_suite(
                scan_name="maze_01",  # Small calibration scan
                output_dir=Path(tmpdir)
            )

            assert "metadata" in results
            assert results["metadata"]["schema_version"] == "1.0.0"
            assert "scans_processed" in results
            assert len(results["scans_processed"]) > 0
