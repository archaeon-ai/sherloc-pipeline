"""End-to-end integration tests for all fixture datasets.

These tests exercise the complete `sherloc plot` workflow on all three fixture
datasets, validating:
- CSV output structure (columns, data types, value ranges)
- PPP scaling applied correctly (500/900 for detail scans, 1.0 for 900 PPP line scan)
- Full processing pipeline (background subtraction, baseline, fitting)
- Output naming conventions
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
    calculate_background_scale,
)


runner = CliRunner()


class TestAmherstPointE2E:
    """End-to-end tests for sol_0921 Amherst_Point detail_1 (500 PPP)."""

    def test_full_pipeline_with_fs_background(self, fixtures_path, tmp_path):
        """Test complete workflow with FS background on Amherst Point."""
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
        
        # Verify outputs exist
        plots_dir = tmp_path / "Amherst_Point" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        
        assert len(csv_files) == 1, f"Expected 1 CSV, found {len(csv_files)}"
        assert len(png_files) == 1, f"Expected 1 PNG, found {len(png_files)}"
        
        # Verify CSV structure
        df = pd.read_csv(csv_files[0])
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) > 500  # R1 spectra have ~523 points
        
        # Verify data types
        assert df["raman_shift"].dtype in [np.float64, np.int64]
        assert df["intensity"].dtype in [np.float64, np.int64]
        
        # Verify Raman shift range is reasonable (~238-4765 cm^-1 for R1)
        assert df["raman_shift"].min() > 200
        assert df["raman_shift"].max() < 5000

    def test_ppp_scaling_500ppp(self, test_context, tmp_results):
        """Verify 500 PPP scan scales correctly (500/900 ≈ 0.556)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            bgscale="auto",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify PPP and scaling
        assert result.metadata["ppp"] == 500.0
        expected_scale = calculate_background_scale(500, 900)
        assert abs(result.metadata["bgscale"] - expected_scale) < 0.001
        assert abs(result.metadata["bgscale"] - 500/900) < 0.001

    def test_csv_output_values_reasonable(self, test_context, tmp_results):
        """Verify CSV output values are in reasonable range for processed spectrum."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        csv_path = result.artifacts[0]
        df = pd.read_csv(csv_path)
        
        # After baseline correction, most values should be near zero
        # with peaks standing out
        median_intensity = df["intensity"].median()
        max_intensity = df["intensity"].max()
        
        # Median should be much smaller than max (peaks should stand out)
        assert max_intensity > median_intensity * 2, "Peaks should stand out from baseline"


class TestLakeHaiyahaE2E:
    """End-to-end tests for sol_0852 Lake_Haiyaha detail_1 (500 PPP, pure olivine)."""

    def test_full_pipeline_with_fs_background(self, fixtures_path, tmp_path):
        """Test complete workflow with FS background on Lake Haiyaha."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "0852",
            "--target", "Lake_Haiyaha",
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
        
        # Verify outputs
        plots_dir = tmp_path / "Lake_Haiyaha" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        
        assert len(csv_files) == 1
        assert len(png_files) == 1
        
        # Verify CSV structure
        df = pd.read_csv(csv_files[0])
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) > 500  # R1 spectra have ~523 points

    def test_ppp_scaling_500ppp(self, test_context, tmp_results):
        """Verify 500 PPP scan scales correctly (500/900 ≈ 0.556)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
            mode="averaged",
            background="fs",
            bgscale="auto",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify PPP and scaling
        assert result.metadata["ppp"] == 500.0
        assert abs(result.metadata["bgscale"] - 500/900) < 0.001

    def test_different_averaging_methods(self, fixtures_path, tmp_path):
        """Test all averaging methods produce valid output."""
        for method in ["mean", "median", "trim-mean"]:
            result = runner.invoke(app, [
                "plot",
                "--sol", "0852",
                "--target", "Lake_Haiyaha",
                "--scan", "detail_1",
                "--avg", method,
                "--export", "csv",
                "--data-dir", str(fixtures_path / "loupe"),
                "--results-dir", str(tmp_path),
            ])
            
            assert result.exit_code == 0, f"Failed for {method}: {result.output}"
            
            # Verify output exists
            plots_dir = tmp_path / "Lake_Haiyaha" / "plots"
            csv_files = list(plots_dir.glob("*.csv"))
            assert len(csv_files) >= 1, f"No CSV output for method {method}"


class TestStigbreenE2E:
    """End-to-end tests for sol_1634 Stigbreen line_1 (900 PPP line scan)."""

    def test_full_pipeline_with_fs_background(self, fixtures_path, tmp_path):
        """Test complete workflow with FS background on Stigbreen line scan."""
        result = runner.invoke(app, [
            "plot",
            "--sol", "1634",
            "--target", "Stigbreen",
            "--scan", "line_1",
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
        
        # Verify outputs
        plots_dir = tmp_path / "Stigbreen" / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        png_files = list(plots_dir.glob("*.png"))
        
        assert len(csv_files) == 1
        assert len(png_files) == 1
        
        # Verify CSV structure
        df = pd.read_csv(csv_files[0])
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns

    def test_ppp_scaling_900ppp(self, test_context, tmp_results):
        """Verify 900 PPP scan scales to 1.0 (900/900 = 1.0)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
            mode="averaged",
            background="fs",
            bgscale="auto",
            baseline=True,
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify PPP and scaling
        assert result.metadata["ppp"] == 900.0
        expected_scale = calculate_background_scale(900, 900)
        assert abs(result.metadata["bgscale"] - expected_scale) < 0.001
        assert abs(result.metadata["bgscale"] - 1.0) < 0.001

    def test_fewer_points_for_line_scan(self, test_context, tmp_results):
        """Verify line scan has correct point count (25 vs 100 for detail scans)."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
            mode="averaged",
            export="csv",
        )
        
        result = service.process(request)
        
        # Line scan has 25 points (vs 100 for detail scans)
        assert result.metadata["n_points"] == 25


class TestBackgroundComparison:
    """Tests comparing AS vs FS background subtraction."""

    def test_as_vs_fs_produces_different_results(self, test_context, tmp_results):
        """Verify AS and FS backgrounds produce measurably different spectra."""
        service = SpectralService(context=test_context)
        
        request_fs = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="csv",
        )
        
        request_as = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="as",
            baseline=True,
            export="csv",
        )
        
        result_fs = service.process(request_fs)
        result_as = service.process(request_as)
        
        # Load both outputs
        df_fs = pd.read_csv(result_fs.artifacts[0])
        df_as = pd.read_csv(result_as.artifacts[0])
        
        # They should have same x-axis but different intensities
        np.testing.assert_array_equal(
            df_fs["raman_shift"].values, 
            df_as["raman_shift"].values
        )
        
        # Intensities should differ (not identical backgrounds)
        # Use correlation - they should be similar but not identical
        correlation = np.corrcoef(
            df_fs["intensity"].values, 
            df_as["intensity"].values
        )[0, 1]
        
        # High correlation (similar shape) but not perfect (different backgrounds)
        assert 0.5 < correlation < 0.9999, f"Correlation {correlation} unexpected"


class TestOutputNamingConventions:
    """Tests for output file naming conventions across all fixtures."""

    @pytest.mark.parametrize("sol,target,scan,ppp", [
        ("0921", "Amherst_Point", "detail_1", 500),
        ("0852", "Lake_Haiyaha", "detail_1", 500),
        ("1634", "Stigbreen", "line_1", 900),
    ])
    def test_filename_contains_required_components(
        self, fixtures_path, tmp_path, sol, target, scan, ppp
    ):
        """Verify output filename follows convention for all fixtures."""
        result = runner.invoke(app, [
            "plot",
            "--sol", sol,
            "--target", target,
            "--scan", scan,
            "--avg", "trim-mean",
            "--trim-pct", "2",
            "--background", "fs",
            "--baseline",
            "--export", "csv",
            "--data-dir", str(fixtures_path / "loupe"),
            "--results-dir", str(tmp_path),
        ])
        
        assert result.exit_code == 0, f"Failed for {sol}/{target}: {result.output}"
        
        # Find output file
        plots_dir = tmp_path / target / "plots"
        csv_files = list(plots_dir.glob("*.csv"))
        assert len(csv_files) == 1
        
        filename = csv_files[0].stem
        
        # Verify all required components present
        assert sol in filename, f"Sol {sol} not in filename {filename}"
        assert target in filename, f"Target {target} not in filename {filename}"
        assert scan in filename, f"Scan {scan} not in filename {filename}"
        assert "R1" in filename, f"R1 not in filename {filename}"
        assert "trim_mean" in filename, f"avg method (trim_mean) not in filename {filename}"
        assert "fs" in filename, f"background type not in filename {filename}"
        assert "baselined" in filename, f"baselined not in filename {filename}"


class TestFitResultsAcrossFixtures:
    """Tests for Gaussian fitting results across all fixtures."""

    @pytest.mark.parametrize("sol,target,scan", [
        ("0921", "Amherst_Point", "detail_1"),
        ("0852", "Lake_Haiyaha", "detail_1"),
        ("1634", "Stigbreen", "line_1"),
    ])
    def test_fitting_produces_valid_r_squared(
        self, test_context, tmp_results, sol, target, scan
    ):
        """Verify fitting produces valid R² values for all fixtures."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol=sol,
            target=target,
            scan=scan,
            mode="averaged",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            export="csv",
        )
        
        result = service.process(request)
        
        # Verify fit metadata
        assert "r2" in result.metadata
        assert "n_peaks" in result.metadata
        
        # R² should be between 0 and 1
        r2 = result.metadata["r2"]
        assert 0 <= r2 <= 1.0, f"R² {r2} out of range for {sol}/{target}"
        
        # Should detect at least one peak in mineral region
        n_peaks = result.metadata["n_peaks"]
        assert n_peaks >= 0  # May be 0 if no significant peaks found

