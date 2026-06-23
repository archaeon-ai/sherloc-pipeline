"""Integration tests for CLI plot command.

These tests exercise the `sherloc plot` command using the typer CliRunner
to validate CLI argument parsing and end-to-end workflow.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


runner = CliRunner()


class TestCLIPlotHelp:
    """Tests for CLI plot command help and argument validation."""

    def test_plot_help_displays_all_options(self):
        """Verify --help displays all expected options."""
        result = runner.invoke(app, ["plot", "--help"])
        
        assert result.exit_code == 0
        output = result.output
        
        # Required options
        assert "--sol" in output
        assert "--target" in output
        assert "--scan" in output
        
        # Optional averaging options
        assert "--avg" in output
        assert "--trim-pct" in output
        
        # Processing options
        assert "--background" in output
        assert "--bgscale" in output
        assert "--baseline" in output
        assert "--fit" in output
        assert "--fit-range" in output
        
        # Axis controls
        assert "--xlim" in output
        assert "--ylim" in output
        
        # Export
        assert "--export" in output
        
        # Path overrides
        assert "--data-dir" in output
        assert "--results-dir" in output

    def test_plot_requires_sol_target_scan(self):
        """Verify required options are enforced."""
        # Missing all required
        result = runner.invoke(app, ["plot"])
        assert result.exit_code != 0
        
        # Missing target and scan
        result = runner.invoke(app, ["plot", "--sol", "921"])
        assert result.exit_code != 0
        
        # Missing scan
        result = runner.invoke(app, ["plot", "--sol", "921", "--target", "Test"])
        assert result.exit_code != 0


class TestCLIPlotValidation:
    """Tests for CLI argument validation."""

    def test_invalid_avg_method_rejected(self, fixtures_path, tmp_path):
        """Verify invalid averaging method is rejected."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--avg", "invalid_method",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code != 0
        assert "Invalid averaging method" in result.output or result.exit_code == 1

    def test_invalid_background_rejected(self, fixtures_path, tmp_path):
        """Verify invalid background type is rejected."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "invalid_bg",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code != 0
        assert "Invalid background" in result.output or result.exit_code == 1

    def test_invalid_export_rejected(self, fixtures_path, tmp_path):
        """Verify invalid export format is rejected."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--export", "pdf",  # Invalid format
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code != 0
        assert "Invalid export format" in result.output or result.exit_code == 1

    def test_invalid_bgscale_rejected(self, fixtures_path, tmp_path):
        """Verify invalid bgscale value is rejected."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--bgscale", "not_a_number",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code != 0
        assert "Invalid bgscale" in result.output or result.exit_code == 1

    def test_point_mode_without_level_processes_from_loupe(self, fixtures_path, tmp_path):
        """Verify point mode without --level processes from Loupe data (T4.3+).
        
        After T4.2/T4.3, point mode without --level processes from raw Loupe data.
        This enables full processing chain (bg-sub, baseline, fit) for individual points.
        """
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",  # Use a valid point index
            # No --level = process from Loupe data
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        # Should succeed - processes from Loupe data
        assert result.exit_code == 0
        assert "Generated files:" in result.output


class TestCLIPlotAveragedMode:
    """Integration tests for averaged mode via CLI."""

    def test_minimal_averaged_workflow(self, fixtures_path, tmp_path):
        """Test minimal averaged workflow via CLI."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Generated files:" in result.output
        
        # Verify output file created
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        assert plots_dir.exists()
        csv_files = list(plots_dir.glob("*.csv"))
        assert len(csv_files) == 1

    def test_full_processing_workflow(self, fixtures_path, tmp_path):
        """Test full workflow with background, baseline, and fit via CLI."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--avg", "trim-mean",
            "--trim-pct", "2",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--export", "both",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Verify both CSV and PNG created
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        assert plots_dir.exists()
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        assert len(csv_files) == 1
        assert len(png_files) == 1

    def test_different_averaging_methods(self, fixtures_path, tmp_path):
        """Test different averaging methods via CLI."""
        for method in ["mean", "median", "trim-mean"]:
            result = runner.invoke(app, [
                "plot",
                "--sol", "0921",
                "--target", "Amherst_Point",
                "--scan", "detail_1",
                "--avg", method,
                "--export", "csv",
                "--data-dir", str(fixtures_path / "loupe"),
                "--results-dir", str(tmp_path),
            ])
            
            assert result.exit_code == 0, f"Failed for method {method}: {result.output}"

    def test_explicit_bgscale_override(self, fixtures_path, tmp_path):
        """Test explicit bgscale override via CLI."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--bgscale", "0.5",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_axis_limits(self, fixtures_path, tmp_path):
        """Test axis limits via CLI."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--xlim", "700,1300",
            "--ylim", "-100,5000",
            "--export", "png",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Verify PNG created
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) == 1

    def test_fit_range_option(self, fixtures_path, tmp_path):
        """Test fit range option via CLI."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--fit-range", "800,1100",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"


class TestCLIPlotOutputNaming:
    """Tests for output file naming conventions via CLI."""

    def test_output_filename_contains_expected_components(self, fixtures_path, tmp_path):
        """Verify output filename follows expected convention."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--avg", "trim-mean",
            "--trim-pct", "2",
            "--background", "fs",
            "--baseline",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Find output file
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        assert len(csv_files) == 1
        
        filename = csv_files[0].stem
        assert "0921" in filename
        assert "Amherst_Point" in filename
        assert "detail_1" in filename
        assert "R1" in filename
        assert "avg-2p_trim_mean" in filename
        assert "fs" in filename
        assert "baselined" in filename


class TestCLIPlotMultipleSols:
    """Tests running CLI on different fixture datasets."""

    def test_lake_haiyaha_fixture(self, fixtures_path, tmp_path):
        """Test CLI on Lake Haiyaha fixture."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0852",
            "--target", "Lake_Haiyaha",
            "--scan", "detail_1",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        plots_dir = tmp_path / "Lake_Haiyaha" / "plots"
        assert plots_dir.exists()
        csv_files = list(plots_dir.glob("*.csv"))
        assert len(csv_files) == 1

    def test_stigbreen_fixture(self, fixtures_path, tmp_path):
        """Test CLI on Stigbreen line scan fixture (900 PPP)."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "1634",
            "--target", "Stigbreen",
            "--scan", "line_1",
            "--background", "fs",
            "--baseline",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        plots_dir = tmp_path / "Stigbreen" / "plots"
        assert plots_dir.exists()
        csv_files = list(plots_dir.glob("*.csv"))
        assert len(csv_files) == 1

