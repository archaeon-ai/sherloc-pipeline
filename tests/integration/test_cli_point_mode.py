"""
Integration tests for CLI point mode in sherloc plot command.

T2.3: Tests for --point and --level options that enable visualization
of single points from existing pipeline outputs.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


runner = CliRunner()


class TestCLIPointModeHelp:
    """Test that --point and --level options appear in help."""
    
    def test_plot_help_includes_point_option(self):
        """Verify --point option is documented in help."""
        result = runner.invoke(app, ["plot", "--help"])
        
        assert result.exit_code == 0
        assert "--point" in result.stdout
        assert "Point index" in result.stdout
    
    def test_plot_help_includes_level_option(self):
        """Verify --level option is documented in help."""
        result = runner.invoke(app, ["plot", "--help"])
        
        assert result.exit_code == 0
        assert "--level" in result.stdout
        assert "normalized" in result.stdout


class TestCLIPointModeValidation:
    """Test validation of --point and --level options."""
    
    def test_point_without_level_processes_from_loupe(self, fixtures_path, tmp_path):
        """Test that --point without --level processes from Loupe data (T4.3+)."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ], catch_exceptions=False)
        
        # Should succeed - processes from Loupe data
        assert result.exit_code == 0
        assert "Generated files:" in result.output
    
    def test_invalid_level_fails(self):
        """Test that invalid --level value shows error."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--level", "invalid_level",
        ], catch_exceptions=False)
        
        assert result.exit_code == 1
    
    @pytest.mark.parametrize("level", [
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined",
    ])
    def test_valid_levels_accepted(self, level, fixtures_path, tmp_path):
        """Test that all valid level values are accepted."""
        # Use fixture data
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--level", level,
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "csv",
        ])
        
        # Should complete successfully (not fail on validation)
        assert result.exit_code == 0, f"Failed with output: {result.stdout}"


class TestCLIPointModeExecution:
    """Test actual execution of point mode via CLI."""
    
    def test_point_mode_basic(self, fixtures_path, tmp_path):
        """Test basic point mode execution."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--level", "normalized",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "csv",
        ])
        
        assert result.exit_code == 0
        assert "point 0" in result.stdout
        assert "normalized" in result.stdout
    
    def test_point_mode_with_xlim(self, fixtures_path, tmp_path):
        """Test point mode with x-axis limits."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--level", "normalized_baselined",
            "--xlim", "500,1500",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "csv",
        ])
        
        assert result.exit_code == 0
        assert "point 5" in result.stdout
    
    def test_point_mode_with_ylim(self, fixtures_path, tmp_path):
        """Test point mode with y-axis limits."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "3",
            "--level", "normalized_despiked_baselined",
            "--ylim", "-100,500",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "csv",
        ])
        
        assert result.exit_code == 0
        assert "point 3" in result.stdout
    
    def test_point_mode_creates_files(self, fixtures_path, tmp_path):
        """Test that point mode creates output files."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "2",
            "--level", "normalized",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "both",
        ])
        
        assert result.exit_code == 0
        assert "Generated files" in result.stdout


class TestCLIPointModeErrors:
    """Test error handling in CLI point mode."""
    
    def test_point_not_found_error(self, fixtures_path, tmp_path):
        """Test error when point is out of range."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "999",  # Out of range
            "--level", "normalized",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
        ], catch_exceptions=False)
        
        assert result.exit_code == 1
    
    def test_missing_pipeline_output_error(self, fixtures_path, tmp_path):
        """Test error when pipeline output doesn't exist."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "9999",
            "--target", "Nonexistent",
            "--scan", "detail_1",
            "--point", "0",
            "--level", "normalized",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
        ], catch_exceptions=False)
        
        assert result.exit_code == 1


class TestCLIModeAutoDetection:
    """Test that mode is auto-detected based on --point presence."""
    
    def test_averaged_mode_by_default(self, fixtures_path, tmp_path):
        """Test that averaged mode is used when --point is not provided."""
        # Hermetic results dir: averaged-mode plot writes under results_root,
        # which without --results-dir resolves to the operator's
        # cwd-relative ./results path. That works on a developer box but
        # fails in clean test environments like the §18.3 / G18.13
        # container test stage (unprivileged user, read-only cwd parent),
        # so the test passes --results-dir explicitly.
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--avg", "mean",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
            "--export", "csv",
        ])

        # Should use averaged mode
        assert result.exit_code == 0
        assert "avg" in result.stdout or "Processed averaged spectrum" in result.stdout
    
    def test_point_mode_when_point_specified(self, fixtures_path):
        """Test that point mode is used when --point is provided."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--level", "normalized",
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
            "--export", "csv",
        ])
        
        # Should use point mode
        assert result.exit_code == 0
        assert "point 0" in result.stdout

