"""
Integration tests for CLI point mode with Loupe processing (T4.4).

Tests that the CLI correctly passes processing flags to point mode
when --level is not specified, enabling full processing from Loupe data.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


runner = CliRunner()


class TestCLIPointLoupeBasic:
    """Test basic CLI point mode with Loupe processing."""

    def test_point_mode_with_background(self, fixtures_path, tmp_path):
        """Test point mode with --background flag."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "fs",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "fs bg-subtracted" in result.output
        assert "Generated files:" in result.output

    def test_point_mode_with_baseline(self, fixtures_path, tmp_path):
        """Test point mode with --baseline flag."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--baseline",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "baselined" in result.output

    def test_point_mode_with_fit(self, fixtures_path, tmp_path):
        """Test point mode with --fit flag."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--fit-range", "700,1200",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "fit" in result.output
        assert "peaks" in result.output


class TestCLIPointLoupeFitting:
    """Test CLI point mode with fitting options."""

    def test_point_mode_with_single_peak(self, fixtures_path, tmp_path):
        """Test point mode with --single-peak option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--single-peak", "1090",
            "--fit-range", "1000,1200",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "fit (1 peaks)" in result.output

    def test_point_mode_with_n_peaks(self, fixtures_path, tmp_path):
        """Test point mode with --n-peaks option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--n-peaks", "2",
            "--fit-range", "700,1200",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "fit" in result.output


class TestCLIPointLoupeProcessingChain:
    """Test full processing chain via CLI for point mode."""

    def test_full_processing_chain(self, fixtures_path, tmp_path):
        """Test full processing: bg-sub + baseline + fit."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "fs",
            "--baseline",
            "--fit",
            "--fit-range", "700,1200",
            "--export", "both",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        
        # Verify outputs created
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        assert plots_dir.exists()
        
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        json_files = list(plots_dir.glob("*.json"))
        
        assert len(csv_files) == 1
        assert len(png_files) == 1
        assert len(json_files) == 1

    def test_as_background_option(self, fixtures_path, tmp_path):
        """Test ARM Stowed background option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--background", "as",
            "--baseline",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "as bg-subtracted" in result.output


class TestCLIPointModeAxisControls:
    """Test axis controls work with point Loupe mode."""

    def test_point_mode_with_xlim(self, fixtures_path, tmp_path):
        """Test point mode with --xlim option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--xlim", "700,1300",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"

    def test_point_mode_with_ylim(self, fixtures_path, tmp_path):
        """Test point mode with --ylim option."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "5",
            "--ylim", "0,1000",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed: {result.output}"


class TestCLIPointModeWarnings:
    """Test warning messages for point mode."""

    def test_warning_when_processing_flags_with_level(self, fixtures_path, tmp_path):
        """Test that warning is shown when using processing flags with --level."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0921",
            "--target", "Amherst_Point",
            "--scan", "detail_1",
            "--point", "0",
            "--level", "normalized",
            "--background", "fs",  # Should trigger warning
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(fixtures_path / "pipeline_outputs"),
        ])
        
        # Command should succeed but show warning
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "ignored" in result.output.lower() or "⚠" in result.output


class TestCLIPointModeAllDatasets:
    """Test point mode on all fixture datasets."""

    @pytest.mark.parametrize("sol,target,scan,point", [
        ("0921", "Amherst_Point", "detail_1", 5),
        ("0852", "Lake_Haiyaha", "detail_1", 10),
        ("1634", "Stigbreen", "line_1", 15),
    ])
    def test_point_mode_on_all_fixtures(self, sol, target, scan, point, fixtures_path, tmp_path):
        """Test point mode works on all fixture datasets."""
        result = runner.invoke(app, [
            "plot",
            "--sol", sol,
            "--target", target,
            "--scan", scan,
            "--point", str(point),
            "--background", "fs",
            "--baseline",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed for {sol}/{target}/{scan}: {result.output}"
        assert "Generated files:" in result.output

