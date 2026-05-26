"""Integration tests for CLI --points option (subset mode).

Tests the `sherloc plot` command with subset point averaging.

T2.9: Add --points CLI option for subset mode
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


runner = CliRunner()


class TestCLISubsetModeHelp:
    """Tests for CLI help documentation of subset mode."""

    def test_help_shows_points_option(self):
        """Help should document --points option."""
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0
        assert "--points" in result.output
        assert "subset" in result.output.lower() or "comma" in result.output.lower()


class TestCLISubsetModeValidation:
    """Tests for CLI validation of subset mode options."""

    def test_points_and_point_mutually_exclusive(self, test_context):
        """--points and --point should be mutually exclusive."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--level", "normalized",
            "--points", "0,1,2,3,4",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_points_requires_at_least_two(self, test_context):
        """--points should require at least 2 points."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "5",  # Only 1 point
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        assert result.exit_code != 0
        assert "at least 2 points" in result.output

    def test_points_invalid_format(self, test_context):
        """--points with invalid format should show error."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "a,b,c",  # Not integers
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        assert result.exit_code != 0
        assert "Invalid --points format" in result.output


class TestCLISubsetModeExecution:
    """Tests for CLI execution with --points option."""

    def test_subset_mode_executes_successfully(self, test_context):
        """--points should execute successfully with valid parameters."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3,4",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "✓" in result.output or "subset" in result.output.lower()

    def test_subset_mode_with_background(self, test_context):
        """--points with background subtraction should work."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,5,10,15,20",
            "--background", "fs",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_mode_with_full_processing(self, test_context):
        """--points with full processing chain should work."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,5,10,15,20,25,30,35,40",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_mode_creates_output_files(self, test_context):
        """--points should create output files with subset naming.
        
        For ≤10 points, filename includes actual point numbers: subset-pts0-1-2-3-4-5-<method>
        For >10 points, filename uses count: subset-<n>pts-<method>
        """
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3,4,5",
            "--export", "both",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Check that output files exist with subset naming
        # For ≤10 points, filename includes pts0-1-2-3-4-5
        plots_dir = test_context.results_root / "Amherst_Point" / "plots"
        csv_files = list(plots_dir.glob("*subset-pts0-1-2-3-4-5*.csv"))
        png_files = list(plots_dir.glob("*subset-pts0-1-2-3-4-5*.png"))
        
        assert len(csv_files) > 0, "No CSV files with subset naming created"
        assert len(png_files) > 0, "No PNG files with subset naming created"


class TestCLISubsetModeWithAveragingOptions:
    """Tests for subset mode with different averaging options."""

    def test_subset_with_mean(self, test_context):
        """--points with mean averaging."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3",
            "--avg", "mean",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_with_median(self, test_context):
        """--points with median averaging."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3,4",
            "--avg", "median",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_with_trim_mean(self, test_context):
        """--points with trim-mean averaging."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3,4,5,6,7,8,9",
            "--avg", "trim-mean",
            "--trim-pct", "5",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"


class TestCLISubsetModeOnDifferentDatasets:
    """Tests for subset mode on different fixture datasets."""

    def test_subset_on_lake_haiyaha(self, test_context):
        """--points on Lake Haiyaha dataset."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0852",
            "--target", "Lake_Haiyaha",
            "--scan", "detail_1",
            "--points", "0,5,10,15",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_on_stigbreen(self, test_context):
        """--points on Stigbreen line scan."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "1634",
            "--target", "Stigbreen",
            "--scan", "line_1",
            "--points", "0,1,2,3",
            "--background", "as",
            "--baseline",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"


class TestCLISubsetModeWithFittingOptions:
    """Tests for subset mode with fitting options."""

    def test_subset_with_single_peak(self, test_context):
        """--points with --single-peak fitting option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,1,2,3,4,5,6,7,8,9",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--single-peak", "1090",
            "--fit-range", "1000,1200",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_subset_with_n_peaks(self, test_context):
        """--points with --n-peaks fitting option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--points", "0,5,10,15,20,25,30",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--n-peaks", "2",
            "--export", "csv",
            "--data-dir", str(test_context.data_root),
            "--results-dir", str(test_context.results_root),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

