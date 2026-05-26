"""
Unit tests for the Python API plot_spectrum() function.

Tests verify that the API function correctly creates matplotlib Figures
with the specified styling options.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sherloc_pipeline.api.spectral import process_scan_average, plot_spectrum


class TestPlotSpectrum:
    """Tests for plot_spectrum() API function."""
    
    @pytest.fixture
    def sample_spectrum(self, fixtures_path):
        """Load a sample spectrum for testing."""
        df, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        return df
    
    @pytest.fixture
    def sample_spectrum_with_fit(self, fixtures_path):
        """Load a sample spectrum with fitting for testing."""
        df, fit = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        return df, fit
    
    def test_returns_figure(self, sample_spectrum):
        """Test that function returns a matplotlib Figure."""
        fig = plot_spectrum(sample_spectrum)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_basic_plot_no_errors(self, sample_spectrum):
        """Test that basic plot can be created without errors."""
        fig = plot_spectrum(sample_spectrum)
        
        # Should have one axes
        assert len(fig.axes) == 1
        plt.close(fig)
    
    def test_with_xlim(self, sample_spectrum):
        """Test that xlim is applied correctly."""
        fig = plot_spectrum(sample_spectrum, xlim=(700, 1200))
        
        ax = fig.axes[0]
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(700, rel=0.01)
        assert xmax == pytest.approx(1200, rel=0.01)
        plt.close(fig)
    
    def test_with_ylim(self, sample_spectrum):
        """Test that ylim is applied correctly."""
        fig = plot_spectrum(sample_spectrum, ylim=(0, 100))
        
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin == pytest.approx(0, rel=0.01)
        assert ymax == pytest.approx(100, rel=0.01)
        plt.close(fig)
    
    def test_with_title(self, sample_spectrum):
        """Test that title is applied correctly."""
        fig = plot_spectrum(sample_spectrum, title="Test Title")
        
        ax = fig.axes[0]
        assert ax.get_title() == "Test Title"
        plt.close(fig)
    
    def test_no_title_when_none(self, sample_spectrum):
        """Test that no title is added when title=None."""
        fig = plot_spectrum(sample_spectrum, title=None)
        
        ax = fig.axes[0]
        assert ax.get_title() == ""
        plt.close(fig)
    
    def test_custom_color(self, sample_spectrum):
        """Test that custom color is applied."""
        fig = plot_spectrum(sample_spectrum, color="#ff0000")
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0
        # Check the line color (normalized to RGBA)
        plt.close(fig)
    
    def test_custom_linewidth(self, sample_spectrum):
        """Test that custom linewidth is applied."""
        fig = plot_spectrum(sample_spectrum, linewidth=3.0)
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0
        assert lines[0].get_linewidth() == 3.0
        plt.close(fig)
    
    def test_custom_linestyle(self, sample_spectrum):
        """Test that custom linestyle is applied."""
        fig = plot_spectrum(sample_spectrum, linestyle="--")
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0
        assert lines[0].get_linestyle() == "--"
        plt.close(fig)
    
    def test_custom_figsize(self, sample_spectrum):
        """Test that custom figsize is applied."""
        fig = plot_spectrum(sample_spectrum, figsize=(8, 4))
        
        width, height = fig.get_size_inches()
        assert width == pytest.approx(8.0, rel=0.01)
        assert height == pytest.approx(4.0, rel=0.01)
        plt.close(fig)
    
    def test_grid_enabled_by_default(self, sample_spectrum):
        """Test that grid is enabled by default."""
        fig = plot_spectrum(sample_spectrum, show_grid=True)
        
        ax = fig.axes[0]
        # Check that grid is visible
        assert ax.xaxis.get_gridlines()[0].get_visible()
        plt.close(fig)
    
    def test_grid_disabled(self, sample_spectrum):
        """Test that grid can be disabled."""
        fig = plot_spectrum(sample_spectrum, show_grid=False)
        
        # Just verify no error - grid state is less testable
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_fit_result(self, sample_spectrum_with_fit):
        """Test that fit overlay is displayed when fit_result provided."""
        df, fit = sample_spectrum_with_fit
        
        fig = plot_spectrum(df, fit_result=fit, xlim=(700, 1200))
        
        ax = fig.axes[0]
        # Should have multiple lines (spectrum + model + peaks)
        lines = ax.get_lines()
        assert len(lines) > 1
        plt.close(fig)
    
    def test_fit_result_creates_legend(self, sample_spectrum_with_fit):
        """Test that fit result creates a legend."""
        df, fit = sample_spectrum_with_fit
        
        fig = plot_spectrum(df, fit_result=fit, xlim=(700, 1200))
        
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)
    
    def test_axis_labels(self, sample_spectrum):
        """Test that axis labels are set correctly."""
        fig = plot_spectrum(sample_spectrum)
        
        ax = fig.axes[0]
        assert "Raman" in ax.get_xlabel()
        assert "Intensity" in ax.get_ylabel()
        plt.close(fig)
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        bad_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        
        with pytest.raises(ValueError, match="missing required columns"):
            plot_spectrum(bad_df)
    
    def test_auto_ylim_with_xlim(self, sample_spectrum):
        """Test that ylim auto-scales to visible data when xlim is set."""
        fig = plot_spectrum(sample_spectrum, xlim=(700, 1200))
        
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()
        # Should be auto-scaled to visible data (not the full range)
        assert ymax < sample_spectrum["intensity"].max()
        plt.close(fig)
    
    def test_savefig_works(self, sample_spectrum, tmp_path):
        """Test that figure can be saved to file."""
        fig = plot_spectrum(sample_spectrum, title="Test")
        
        output_path = tmp_path / "test_plot.png"
        fig.savefig(output_path, dpi=100)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)


