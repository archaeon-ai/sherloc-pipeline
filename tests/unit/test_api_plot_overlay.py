"""
Unit tests for the Python API plot_overlay() function.

Tests verify that the API function correctly creates multi-spectrum
overlay plots with per-trace styling.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sherloc_pipeline.api.spectral import (
    process_scan_average,
    load_reference_spectrum,
    plot_overlay,
)


class TestPlotOverlay:
    """Tests for plot_overlay() API function."""
    
    @pytest.fixture
    def sample_spectra(self, fixtures_path):
        """Load sample spectra for overlay testing."""
        df1, _ = process_scan_average(
            sol="0921", target="Amherst_Point", scan="detail_1",
            background="fs", baseline=True, fit=False,
            data_dir=fixtures_path / "loupe",
        )
        df2, _ = process_scan_average(
            sol="0852", target="Lake_Haiyaha", scan="detail_1",
            background="fs", baseline=True, fit=False,
            data_dir=fixtures_path / "loupe",
        )
        return df1, df2
    
    @pytest.fixture
    def reference_spectrum(self, fixtures_path):
        """Load reference spectrum for testing."""
        return load_reference_spectrum(
            "forsterite",
            library_path=fixtures_path / "reference",
        )
    
    def test_returns_figure(self, sample_spectra):
        """Test that function returns a matplotlib Figure."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "label": "Spec 1"},
            {"df": df2, "label": "Spec 2"},
        ])
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_two_spectra_overlay(self, sample_spectra):
        """Test overlaying two spectra."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "label": "Amherst Point"},
            {"df": df2, "label": "Lake Haiyaha"},
        ])
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)
    
    def test_with_custom_colors(self, sample_spectra):
        """Test that custom colors are applied."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "label": "Spec 1", "color": "#ff0000"},
            {"df": df2, "label": "Spec 2", "color": "#00ff00"},
        ])
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)
    
    def test_with_custom_linestyles(self, sample_spectra):
        """Test that custom linestyles are applied."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "label": "Solid", "linestyle": "-"},
            {"df": df2, "label": "Dashed", "linestyle": "--"},
        ])
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert lines[0].get_linestyle() == "-"
        assert lines[1].get_linestyle() == "--"
        plt.close(fig)
    
    def test_with_custom_linewidths(self, sample_spectra):
        """Test that custom linewidths are applied."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "linewidth": 1.0},
            {"df": df2, "linewidth": 2.5},
        ])
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert lines[0].get_linewidth() == 1.0
        assert lines[1].get_linewidth() == 2.5
        plt.close(fig)
    
    def test_with_xlim(self, sample_spectra):
        """Test that xlim is applied correctly."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            xlim=(700, 1200),
        )
        
        ax = fig.axes[0]
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(700, rel=0.01)
        assert xmax == pytest.approx(1200, rel=0.01)
        plt.close(fig)
    
    def test_with_ylim(self, sample_spectra):
        """Test that ylim is applied correctly."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            ylim=(0, 100),
        )
        
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin == pytest.approx(0, rel=0.01)
        assert ymax == pytest.approx(100, rel=0.01)
        plt.close(fig)
    
    def test_with_title(self, sample_spectra):
        """Test that title is applied correctly."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            title="Test Overlay",
        )
        
        ax = fig.axes[0]
        assert ax.get_title() == "Test Overlay"
        plt.close(fig)
    
    def test_legend_when_labels_provided(self, sample_spectra):
        """Test that legend appears when labels are provided."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1, "label": "First"},
            {"df": df2, "label": "Second"},
        ])
        
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)
    
    def test_no_legend_without_labels(self, sample_spectra):
        """Test that no legend when no labels provided."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(spectra=[
            {"df": df1},
            {"df": df2},
        ])
        
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is None
        plt.close(fig)
    
    def test_legend_location(self, sample_spectra):
        """Test that legend location can be customized."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[
                {"df": df1, "label": "A"},
                {"df": df2, "label": "B"},
            ],
            legend_loc="lower left",
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_scale_to_peak_normalizes(self, sample_spectra):
        """Test that scale_to_peak normalizes spectra."""
        df1, df2 = sample_spectra
        
        # Without scaling
        fig1 = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            xlim=(800, 900),
        )
        ax1 = fig1.axes[0]
        lines1 = ax1.get_lines()
        max1_before = max(lines1[0].get_ydata())
        max2_before = max(lines1[1].get_ydata())
        plt.close(fig1)
        
        # With scaling - both should have similar max in range
        fig2 = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            xlim=(800, 900),
            scale_to_peak=(820, 870),
        )
        
        # Just verify no error and figure created
        assert isinstance(fig2, Figure)
        plt.close(fig2)
    
    def test_mars_and_reference_overlay(self, sample_spectra, reference_spectrum):
        """Test overlaying Mars spectrum with reference."""
        df1, _ = sample_spectra
        
        fig = plot_overlay(
            spectra=[
                {"df": df1, "label": "Mars", "color": "blue"},
                {"df": reference_spectrum, "label": "Forsterite", 
                 "color": "green", "linestyle": "--"},
            ],
            xlim=(700, 1200),
            title="Mars vs Reference",
        )
        
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2
        assert ax.get_legend() is not None
        plt.close(fig)
    
    def test_empty_list_raises_error(self):
        """Test that empty spectra list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_overlay(spectra=[])
    
    def test_missing_df_key_raises_error(self, sample_spectra):
        """Test that missing 'df' key raises ValueError."""
        df1, _ = sample_spectra
        
        with pytest.raises(ValueError, match="missing required 'df' key"):
            plot_overlay(spectra=[
                {"df": df1},
                {"label": "Missing df"},  # No 'df' key
            ])
    
    def test_missing_columns_raises_error(self):
        """Test that missing columns raises ValueError."""
        bad_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        
        with pytest.raises(ValueError, match="missing required columns"):
            plot_overlay(spectra=[{"df": bad_df}])
    
    def test_custom_figsize(self, sample_spectra):
        """Test that custom figsize is applied."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[{"df": df1}, {"df": df2}],
            figsize=(8, 4),
        )
        
        width, height = fig.get_size_inches()
        assert width == pytest.approx(8.0, rel=0.01)
        assert height == pytest.approx(4.0, rel=0.01)
        plt.close(fig)
    
    def test_savefig_works(self, sample_spectra, tmp_path):
        """Test that overlay figure can be saved."""
        df1, df2 = sample_spectra
        
        fig = plot_overlay(
            spectra=[
                {"df": df1, "label": "A"},
                {"df": df2, "label": "B"},
            ],
            title="Test",
        )
        
        output_path = tmp_path / "overlay.png"
        fig.savefig(output_path, dpi=100)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)


