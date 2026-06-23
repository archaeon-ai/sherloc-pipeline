"""Integration tests for CLI --single-peak and --n-peaks options.

Tests the `sherloc plot` command with granular fitting controls.

T2.6: Add --single-peak and --n-peaks CLI options
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


runner = CliRunner()


class TestCLIFittingOptionsHelp:
    """Tests for CLI help documentation of fitting options."""

    def test_help_shows_single_peak_option(self):
        """Help should document --single-peak option."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "--single-peak" in result.output
        assert "single peak" in result.output.lower()

    def test_help_shows_n_peaks_option(self):
        """Help should document --n-peaks option."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "--n-peaks" in result.output
        assert "maximum" in result.output.lower() or "peaks" in result.output.lower()


class TestCLIFittingOptionsValidation:
    """Tests for CLI validation of fitting options."""

    def test_single_peak_requires_fit(self, test_context, tmp_results):
        """--single-peak should require --fit to be enabled."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--single-peak", "1090",
            # Missing --fit
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        assert result.exit_code != 0
        assert "--single-peak requires --fit" in result.output

    def test_n_peaks_requires_fit(self, test_context, tmp_results):
        """--n-peaks should require --fit to be enabled."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--n-peaks", "2",
            # Missing --fit
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        assert result.exit_code != 0
        assert "--n-peaks requires --fit" in result.output

    def test_single_peak_and_n_peaks_mutually_exclusive(self, test_context, tmp_results):
        """--single-peak and --n-peaks should be mutually exclusive."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--fit",
            "--single-peak", "1090",
            "--n-peaks", "2",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_n_peaks_must_be_positive(self, test_context, tmp_results):
        """--n-peaks must be a positive integer."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--fit",
            "--n-peaks", "0",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        assert result.exit_code != 0
        assert "between 1 and 10" in result.output

    def test_n_peaks_max_limit(self, test_context, tmp_results):
        """--n-peaks should not exceed 10."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--fit",
            "--n-peaks", "11",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        assert result.exit_code != 0
        assert "between 1 and 10" in result.output


class TestCLISinglePeakExecution:
    """Tests for CLI execution with --single-peak option."""

    def test_single_peak_executes_successfully(self, test_context, tmp_results):
        """--single-peak should execute successfully with valid parameters."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--single-peak", "1090",
            "--fit-range", "1000,1200",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        # Should succeed
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "✓" in result.output or "Generated" in result.output

    def test_single_peak_creates_output_files(self, test_context, tmp_results):
        """--single-peak should create output files."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--single-peak", "850",
            "--fit-range", "700,1000",
            "--export", "both",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Check that output files exist
        plots_dir = tmp_results / "Amherst_Point" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        
        assert len(csv_files) > 0, "No CSV files created"
        assert len(png_files) > 0, "No PNG files created"


class TestCLINPeaksExecution:
    """Tests for CLI execution with --n-peaks option."""

    def test_n_peaks_executes_successfully(self, test_context, tmp_results):
        """--n-peaks should execute successfully with valid parameters."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--n-peaks", "2",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        # Should succeed
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "✓" in result.output or "Generated" in result.output

    def test_n_peaks_one(self, test_context, tmp_results):
        """--n-peaks 1 should execute successfully."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--baseline",
            "--fit",
            "--n-peaks", "1",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_n_peaks_creates_output_files(self, test_context, tmp_results):
        """--n-peaks should create output files."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--n-peaks", "3",
            "--export", "both",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Check that output files exist
        plots_dir = tmp_results / "Amherst_Point" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        
        assert len(csv_files) > 0, "No CSV files created"
        assert len(png_files) > 0, "No PNG files created"


class TestCLIFittingOptionsOnMultipleDatasets:
    """Tests for CLI fitting options on different fixture datasets."""

    def test_single_peak_on_lake_haiyaha(self, test_context, tmp_results):
        """--single-peak on Lake Haiyaha (olivine-rich) dataset."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0852",
            "--target", "Lake_Haiyaha",
            "--scan", "detail_1",
            "--baseline",
            "--fit",
            "--single-peak", "850",
            "--fit-range", "700,1000",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_n_peaks_on_stigbreen(self, test_context, tmp_results):
        """--n-peaks on Stigbreen line scan dataset."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "1634",
            "--target", "Stigbreen",
            "--scan", "line_1",
            "--background", "as",
            "--baseline",
            "--fit",
            "--n-peaks", "2",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(tmp_results),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

