"""
Unit tests for spectral export functionality.

Tests the _export() method and _build_filename() helper in SpectralService,
verifying correct output paths, filenames, and file creation.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
)
from sherloc_pipeline.models.fitting import FitResult, PeakFit


class TestBuildFilename:
    """Tests for the _build_filename helper method."""

    def test_basic_averaged_filename(self, test_context):
        """Test basic averaged mode filename without processing."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean"

    def test_filename_with_mean_method(self, test_context):
        """Test filename with mean averaging method."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="mean",
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-mean"

    def test_filename_with_median_method(self, test_context):
        """Test filename with median averaging method."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="median",
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-median"

    def test_filename_with_5_percent_trim(self, test_context):
        """Test filename with 5% trim-mean."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=5.0,
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-5p_trim_mean"

    def test_filename_with_background(self, test_context):
        """Test filename with background subtraction."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs"

    def test_filename_with_baseline(self, test_context):
        """Test filename with baseline correction."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_baselined"

    def test_filename_with_fit(self, test_context):
        """Test filename with fitting (requires fit_result)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
            fit=True,
        )
        
        # Create a mock fit result
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        
        filename = service._build_filename(request, fit_result=fit_result)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_baselined_fit"

    def test_filename_fit_without_fit_result(self, test_context):
        """Test that fit suffix is not added without fit_result."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
            fit=True,  # Requested but no fit_result
        )
        
        # No fit_result provided - should not include _fit suffix
        filename = service._build_filename(request, fit_result=None)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_baselined"

    def test_full_processing_filename(self, test_context):
        """Test filename with all processing options enabled."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        
        filename = service._build_filename(request, fit_result=fit_result)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined_fit"

    def test_point_mode_filename(self, test_context):
        """Test filename for point visualization mode."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            level="normalized_despiked_baselined",
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p91_normalized_despiked_baselined"

    def test_filename_with_as_background(self, test_context):
        """Test filename with arm-stowed background."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="as",
            baseline=True,
            fit=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_as_baselined"


class TestExport:
    """Tests for the _export method."""

    @pytest.fixture
    def simple_spectrum(self):
        """Create a simple test spectrum DataFrame."""
        return pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 100),
            "intensity": np.random.randn(100) + 100,
        })

    @pytest.fixture
    def simple_figure(self, simple_spectrum):
        """Create a simple test figure."""
        fig, ax = plt.subplots()
        ax.plot(simple_spectrum["raman_shift"], simple_spectrum["intensity"])
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        yield fig
        plt.close(fig)

    def test_export_csv_only(self, test_context, simple_spectrum, simple_figure):
        """Test exporting CSV only."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        # Should have CSV + JSON (metadata always exported by default)
        assert len(artifacts) == 2
        csv_path = [p for p in artifacts if p.suffix == ".csv"][0]
        json_path = [p for p in artifacts if p.suffix == ".json"][0]
        
        # Verify files exist
        assert csv_path.exists()
        assert json_path.exists()
        
        # Verify CSV content
        loaded = pd.read_csv(csv_path)
        assert list(loaded.columns) == ["raman_shift", "intensity"]
        assert len(loaded) == 100

    def test_export_png_only(self, test_context, simple_spectrum, simple_figure):
        """Test exporting PNG only."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="png",
            baseline=False,
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        # Should have PNG + JSON
        assert len(artifacts) == 2
        png_path = [p for p in artifacts if p.suffix == ".png"][0]
        json_path = [p for p in artifacts if p.suffix == ".json"][0]
        
        # Verify files exist
        assert png_path.exists()
        assert json_path.exists()
        
        # Verify PNG is a valid image file (has some content)
        assert png_path.stat().st_size > 0

    def test_export_both(self, test_context, simple_spectrum, simple_figure):
        """Test exporting both CSV and PNG."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="both",
            baseline=False,
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        # Should have CSV, PNG, and JSON
        assert len(artifacts) == 3
        
        # Find CSV, PNG, and JSON
        extensions = {p.suffix for p in artifacts}
        assert extensions == {".csv", ".png", ".json"}
        
        # All should exist
        for path in artifacts:
            assert path.exists()

    def test_export_creates_plots_directory(self, test_context, simple_spectrum, simple_figure):
        """Test that export creates the plots subdirectory."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        # Directory shouldn't exist yet
        expected_dir = test_context.results_root / "Amherst_Point" / "plots"
        assert not expected_dir.exists()
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        # Directory should now exist
        assert expected_dir.exists()
        assert expected_dir.is_dir()
        
        # File should be in the plots directory
        assert artifacts[0].parent == expected_dir

    def test_export_output_path_format(self, test_context, simple_spectrum, simple_figure):
        """Test that output path follows convention: results/<target>/plots/."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        csv_path = artifacts[0]
        # Check path structure
        assert csv_path.parent.name == "plots"
        assert csv_path.parent.parent.name == "Amherst_Point"

    def test_export_filename_matches_request(self, test_context, simple_spectrum, simple_figure):
        """Test that exported filename matches the naming convention."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="both",
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        expected_base = "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined"
        for path in artifacts:
            assert path.stem == expected_base

    def test_export_with_different_target(self, test_context, simple_spectrum, simple_figure):
        """Test export with a different target creates correct directory."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        artifacts = service._export(simple_spectrum, simple_figure, request)
        
        # Should be in Lake_Haiyaha/plots/
        expected_dir = test_context.results_root / "Lake_Haiyaha" / "plots"
        assert expected_dir.exists()
        assert artifacts[0].parent == expected_dir

    def test_export_missing_columns_raises_error(self, test_context, simple_figure):
        """Test that export raises error for missing columns."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        # DataFrame missing 'intensity' column
        bad_df = pd.DataFrame({"raman_shift": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="missing required columns"):
            service._export(bad_df, simple_figure, request)

    def test_export_with_fit_result(self, test_context, simple_spectrum, simple_figure):
        """Test export includes _fit suffix when fit_result provided."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
            fit=True,
            export="csv",
        )
        
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        
        artifacts = service._export(simple_spectrum, simple_figure, request, fit_result=fit_result)
        
        # Filename should include _fit suffix
        assert "_fit" in artifacts[0].stem

    def test_export_overwrites_existing_files(self, test_context, simple_spectrum, simple_figure):
        """Test that export overwrites existing files."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            export="csv",
            baseline=False,
            fit=False,
        )
        
        # First export
        artifacts1 = service._export(simple_spectrum, simple_figure, request)
        original_mtime = artifacts1[0].stat().st_mtime
        
        # Modify data and export again
        modified_spectrum = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 50),  # Different length
            "intensity": np.random.randn(50) + 200,
        })
        
        # Small delay to ensure different mtime
        import time
        time.sleep(0.01)
        
        artifacts2 = service._export(modified_spectrum, simple_figure, request)
        
        # Should be same path
        assert artifacts1[0] == artifacts2[0]
        
        # File should be modified
        new_mtime = artifacts2[0].stat().st_mtime
        assert new_mtime >= original_mtime
        
        # Content should be updated
        loaded = pd.read_csv(artifacts2[0])
        assert len(loaded) == 50  # New length


class TestExportOnRealData:
    """Integration tests for export with real fixture data."""

    def test_export_real_averaged_spectrum(self, test_context):
        """Test export of real averaged spectrum from fixtures."""
        service = SpectralService(context=test_context)
        
        # Load and process real data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Build request
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=False,
            export="both",
            fit=False,
        )
        
        # Generate plot
        fig = service._generate_plot(avg_df, request)
        
        try:
            # Export
            artifacts = service._export(avg_df, fig, request)
            
            # Verify files exist (CSV + PNG + JSON)
            assert len(artifacts) == 3
            for path in artifacts:
                assert path.exists()
            
            # Verify CSV content matches original
            csv_path = [p for p in artifacts if p.suffix == ".csv"][0]
            loaded = pd.read_csv(csv_path)
            np.testing.assert_array_almost_equal(
                loaded["raman_shift"].values,
                avg_df["raman_shift"].values,
                decimal=6
            )
            np.testing.assert_array_almost_equal(
                loaded["intensity"].values,
                avg_df["intensity"].values,
                decimal=6
            )
        finally:
            plt.close(fig)

    def test_export_with_full_processing(self, test_context):
        """Test export of spectrum with background subtraction and baseline."""
        service = SpectralService(context=test_context)
        from sherloc_pipeline.services.spectral import calculate_background_scale
        
        # Load and process real data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Apply processing
        scale = calculate_background_scale(data.ppp, 900)
        bg_sub = service._apply_background_subtraction(avg_df, "fs", scale)
        baselined = service._apply_baseline(bg_sub)
        
        # Build request
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            export="both",
            fit=False,
        )
        
        # Generate plot
        fig = service._generate_plot(baselined, request)
        
        try:
            # Export
            artifacts = service._export(baselined, fig, request)
            
            # Verify filename convention
            csv_path = [p for p in artifacts if p.suffix == ".csv"][0]
            assert "fs" in csv_path.stem
            assert "baselined" in csv_path.stem
            assert csv_path.parent.name == "plots"
        finally:
            plt.close(fig)

    def test_export_with_fitting(self, test_context):
        """Test export of spectrum with fitting applied."""
        service = SpectralService(context=test_context)
        from sherloc_pipeline.services.spectral import calculate_background_scale
        
        # Load and process real data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Apply processing
        scale = calculate_background_scale(data.ppp, 900)
        bg_sub = service._apply_background_subtraction(avg_df, "fs", scale)
        baselined = service._apply_baseline(bg_sub)
        
        # Apply fitting
        fit_result, model_array = service._apply_fitting(baselined, fit_range=(700, 1200))
        
        # Build request
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            fit=True,
            export="both",
        )
        
        # Generate plot
        fig = service._generate_plot(baselined, request, fit_result, model_array)
        
        try:
            # Export
            artifacts = service._export(baselined, fig, request, fit_result=fit_result)
            
            # Verify filename convention
            for path in artifacts:
                assert "fs" in path.stem
                assert "baselined" in path.stem
                assert "fit" in path.stem
        finally:
            plt.close(fig)


class TestSubsetModeFilename:
    """Tests for subset mode filename and title generation.
    
    Filename format depends on point count:
    - ≤10 points: subset-pts<p1>-<p2>-...-<method>
    - >10 points: subset-<n>pts-<method>
    """

    def test_subset_filename_basic_with_points(self, test_context):
        """Test basic subset mode filename with point numbers (≤10 points)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4],
            avg_method="trim-mean",
            trim_pct=2.0,
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        # ≤10 points uses actual point numbers
        assert "subset-pts0-1-2-3-4" in filename
        assert "2p_trim_mean" in filename
        assert filename == "0921_Amherst_Point_detail_1_R1_subset-pts0-1-2-3-4-2p_trim_mean"

    def test_subset_filename_with_background(self, test_context):
        """Test subset filename with background subtraction (>10 points uses count)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[21, 41, 49, 71, 86, 87, 88, 90, 91, 92, 98],
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        # >10 points uses count only
        assert "subset-11pts" in filename
        assert "_fs" in filename
        assert filename == "0921_Amherst_Point_detail_1_R1_subset-11pts-2p_trim_mean_fs"

    def test_subset_filename_with_full_processing(self, test_context):
        """Test subset filename with full processing chain (>10 points)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=True,
            fit=True,
        )
        
        # Provide fit_result to get _fit suffix
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        filename = service._build_filename(request, fit_result=fit_result)
        
        # >10 points uses count only
        assert "subset-11pts" in filename
        assert "_fs" in filename
        assert "_baselined" in filename
        assert "_fit" in filename
        assert filename == "0921_Amherst_Point_detail_1_R1_subset-11pts-2p_trim_mean_fs_baselined_fit"

    def test_subset_filename_with_mean_method(self, test_context):
        """Test subset filename with mean averaging (≤10 points)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 5, 10],
            avg_method="mean",
            baseline=False,
            fit=False,
        )
        
        filename = service._build_filename(request)
        # ≤10 points uses actual point numbers
        assert "subset-pts0-5-10-mean" in filename

    def test_subset_title_basic(self, test_context):
        """Test subset mode plot title includes full points list."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4],
            avg_method="trim-mean",
            trim_pct=2.0,
            baseline=False,
            fit=False,
        )
        
        title = service._build_plot_title(request)
        assert "subset (5 pts)" in title
        assert "2p_trim_mean" in title
        assert "sol 0921" in title
        # Title should include the points list
        assert "points:" in title
        assert "0, 1, 2, 3, 4" in title

    def test_subset_title_with_processing(self, test_context):
        """Test subset title with processing indicators."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[21, 41, 49, 71, 86, 87, 88, 90, 91, 92, 98],
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=True,
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        title = service._build_plot_title(request, fit_result=fit_result)
        
        assert "subset (11 pts)" in title
        assert "fs" in title
        assert "baselined" in title
        assert "fit" in title


class TestPointLoupeNaming:
    """Tests for point Loupe mode filename and title generation (T4.5)."""

    def test_point_loupe_filename_basic(self, test_context):
        """Test point Loupe mode filename without processing."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            # No level = Loupe mode
            baseline=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p5"

    def test_point_loupe_filename_with_background(self, test_context):
        """Test point Loupe mode filename with background subtraction."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p5_fs"

    def test_point_loupe_filename_with_baseline(self, test_context):
        """Test point Loupe mode filename with baseline correction."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=True,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p5_baselined"

    def test_point_loupe_filename_full_processing(self, test_context):
        """Test point Loupe mode filename with all processing options."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        filename = service._build_filename(request, fit_result=fit_result)
        assert filename == "0921_Amherst_Point_detail_1_R1_p5_fs_baselined_fit"

    def test_point_loupe_filename_with_xlim(self, test_context):
        """Test point Loupe mode filename includes xlim range."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            xlim=(700, 1200),
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p5_fs_baselined_700-1200"

    def test_point_results_filename_unchanged(self, test_context):
        """Test point results mode filename is unchanged (backward compat)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            level="normalized_despiked_baselined",
            baseline=False,
        )
        
        filename = service._build_filename(request)
        assert filename == "0921_Amherst_Point_detail_1_R1_p91_normalized_despiked_baselined"

    def test_point_loupe_title_basic(self, test_context):
        """Test point Loupe mode title without processing."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
        )
        
        title = service._build_plot_title(request)
        assert "sol 0921" in title
        assert "Amherst_Point" in title
        assert "detail_1" in title
        assert "R1" in title
        assert "point 5" in title

    def test_point_loupe_title_with_processing(self, test_context):
        """Test point Loupe mode title with processing indicators."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=10, warnings=[])
        title = service._build_plot_title(request, fit_result=fit_result)
        
        assert "sol 0921" in title
        assert "point 5" in title
        assert "fs" in title
        assert "baselined" in title
        assert "fit" in title

    def test_point_results_title_unchanged(self, test_context):
        """Test point results mode title is unchanged (backward compat)."""
        service = SpectralService(context=test_context)
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            level="normalized",
            baseline=False,
        )
        
        title = service._build_plot_title(request)
        assert title == "sol 0921 Amherst_Point detail_1 R1 point 91 normalized"

