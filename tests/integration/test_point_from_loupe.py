"""
Integration tests for point mode processing from Loupe data (T4.8).

Comprehensive tests for the unified point mode workflow that processes
single points directly from raw Loupe data with full processing chain.

These tests complement existing tests by focusing on:
- Service layer _process_point_from_loupe() method
- Output file naming for Loupe-sourced point spectra
- Edge cases: invalid points, missing data, boundary conditions
- End-to-end workflow verification across all fixture datasets
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
    SpectralPlotError,
)
from sherloc_pipeline.services.runtime import RuntimeContext
from sherloc_pipeline.services.base import ServiceResult


class TestProcessPointFromLoupeBasic:
    """Test basic _process_point_from_loupe() functionality."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_basic_point_processing(self, service):
        """Test basic point processing from Loupe returns valid ServiceResult."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata["point"] == 5
        assert "ppp" in result.metadata
        assert "total_points" in result.metadata
        assert result.metadata["n_datapoints"] > 0

    def test_returns_correct_metadata(self, service):
        """Test that metadata includes all expected fields."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=True,  # Request processing to get processing flags in metadata
        )
        
        result = service._process_point_from_loupe(request)
        
        # Required metadata fields
        assert "point" in result.metadata
        assert "ppp" in result.metadata
        assert "total_points" in result.metadata
        assert "n_datapoints" in result.metadata
        
        # Processing flags (only present when processing is applied)
        assert result.metadata.get("baseline") is True

    def test_summary_includes_point_info(self, service):
        """Test that summary message includes point information."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=42,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert "point 42" in result.summary.lower()
        assert "0921" in result.summary
        assert "Amherst_Point" in result.summary


class TestProcessPointFromLoupeProcessing:
    """Test processing options in _process_point_from_loupe()."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_with_fs_background(self, service):
        """Test point processing with fused silica background."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata["background"] == "fs"
        assert "bgscale" in result.metadata
        # PPP-based scaling: 500 PPP scan / 900 PPP background
        assert abs(result.metadata["bgscale"] - (500.0 / 900.0)) < 0.01

    def test_with_as_background(self, service):
        """Test point processing with Arm Stowed background."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="as",
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata["background"] == "as"
        assert "bgscale" in result.metadata

    def test_with_baseline(self, service):
        """Test point processing with baseline correction."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=True,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata.get("baseline") is True
        assert "baselined" in result.summary.lower()

    def test_with_fitting(self, service):
        """Test point processing with Gaussian fitting."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata.get("fit") is True
        assert "n_peaks" in result.metadata
        assert "r2" in result.metadata
        assert "fit" in result.summary.lower()

    def test_with_single_peak_fitting(self, service):
        """Test point processing with single-peak fitting."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(1000, 1200),
            single_peak_center=1090.0,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata.get("fit") is True
        assert result.metadata["n_peaks"] == 1

    def test_full_processing_chain(self, service):
        """Test complete processing chain: bg-sub + baseline + fit."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            export="both",
        )
        
        result = service._process_point_from_loupe(request)
        
        # All processing flags should be recorded
        assert result.metadata["background"] == "fs"
        assert result.metadata.get("baseline") is True
        assert result.metadata.get("fit") is True
        
        # Should have artifacts
        assert len(result.artifacts) > 0


class TestProcessPointFromLoupeOutputNaming:
    """Test output file naming for Loupe point mode."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_filename_without_processing(self, service, tmp_path):
        """Test filename pattern without processing flags."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=7,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        assert len(csv_files) == 1
        
        filename = csv_files[0].name
        assert "0921" in filename
        assert "Amherst_Point" in filename
        assert "detail_1" in filename
        assert "p7" in filename
        # Should NOT include processing indicators
        assert "_fs" not in filename
        assert "_baselined" not in filename

    def test_filename_with_background(self, service, tmp_path):
        """Test filename includes background indicator."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        filename = csv_files[0].name
        
        assert "_fs" in filename

    def test_filename_with_baseline(self, service, tmp_path):
        """Test filename includes baselined indicator."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=True,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        filename = csv_files[0].name
        
        assert "_baselined" in filename

    def test_filename_with_fit(self, service, tmp_path):
        """Test filename includes fit indicator."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        filename = csv_files[0].name
        
        assert "_fit" in filename

    def test_filename_full_pattern(self, service, tmp_path):
        """Test complete filename pattern with all flags."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        filename = csv_files[0].name
        
        # Pattern: <sol>_<target>_<scan>_R1_p<point>[_<bg>][_baselined][_fit]
        assert "0921" in filename
        assert "Amherst_Point" in filename
        assert "detail_1" in filename
        assert "R1" in filename
        assert "p91" in filename
        assert "_fs" in filename
        assert "_baselined" in filename
        assert "_fit" in filename

    def test_filename_with_xlim(self, service, tmp_path):
        """Test filename includes xlim range when specified."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            xlim=(700, 1300),
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        filename = csv_files[0].name
        
        # Should include xlim range
        assert "700-1300" in filename


class TestProcessPointFromLoupeEdgeCases:
    """Test edge cases for Loupe point processing."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_first_point(self, service):
        """Test processing first point (index 0)."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata["point"] == 0
        assert isinstance(result, ServiceResult)

    def test_last_valid_point(self, service):
        """Test processing last valid point."""
        # First, get total points
        loupe_data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        last_point = loupe_data.n_points - 1
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=last_point,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata["point"] == last_point
        assert isinstance(result, ServiceResult)

    def test_invalid_point_raises_error(self, service):
        """Test that out-of-range point raises appropriate error."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=99999,
            baseline=False,
        )
        
        with pytest.raises(SpectralPlotError) as excinfo:
            service._process_point_from_loupe(request)
        
        assert "out of range" in str(excinfo.value).lower() or "not found" in str(excinfo.value).lower()

    def test_negative_point_raises_error(self, service):
        """Test that negative point index raises error."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=-1,
            baseline=False,
        )
        
        with pytest.raises(SpectralPlotError):
            service._process_point_from_loupe(request)

    def test_missing_loupe_data_raises_error(self, service):
        """Test that missing Loupe data raises appropriate error."""
        request = SpectralPlotRequest(
            sol="9999",
            target="Nonexistent_Target",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=False,
        )
        
        with pytest.raises(SpectralPlotError):
            service._process_point_from_loupe(request)


class TestProcessPointFromLoupeAllDatasets:
    """Test Loupe point processing on all fixture datasets."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    @pytest.mark.parametrize("sol,target,scan,ppp", [
        ("0921", "Amherst_Point", "detail_1", 500),  # Detail scan, 500 PPP
        ("0852", "Lake_Haiyaha", "detail_1", 500),   # Detail scan, 500 PPP
        ("1634", "Stigbreen", "line_1", 900),        # Line scan, 900 PPP
    ])
    def test_basic_processing_on_all_datasets(self, service, sol, target, scan, ppp):
        """Test basic point processing works on all fixture datasets."""
        request = SpectralPlotRequest(
            sol=sol,
            target=target,
            scan=scan,
            mode="point",
            point=5,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata["ppp"] == ppp
        assert result.metadata["point"] == 5

    @pytest.mark.parametrize("sol,target,scan", [
        ("0921", "Amherst_Point", "detail_1"),
        ("0852", "Lake_Haiyaha", "detail_1"),
        ("1634", "Stigbreen", "line_1"),
    ])
    def test_full_processing_on_all_datasets(self, service, sol, target, scan):
        """Test full processing chain on all fixture datasets."""
        request = SpectralPlotRequest(
            sol=sol,
            target=target,
            scan=scan,
            mode="point",
            point=10,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
        )
        
        result = service._process_point_from_loupe(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata.get("fit") is True
        assert "n_peaks" in result.metadata

    @pytest.mark.parametrize("sol,target,scan,expected_min_points", [
        ("0921", "Amherst_Point", "detail_1", 100),  # Should have 100+ points
        ("0852", "Lake_Haiyaha", "detail_1", 100),
        ("1634", "Stigbreen", "line_1", 20),  # Line scans have fewer points
    ])
    def test_total_points_on_all_datasets(self, service, sol, target, scan, expected_min_points):
        """Test that total_points metadata is correct for each dataset."""
        request = SpectralPlotRequest(
            sol=sol,
            target=target,
            scan=scan,
            mode="point",
            point=0,
            baseline=False,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata["total_points"] >= expected_min_points


class TestProcessPointFromLoupeCSVOutput:
    """Test CSV output structure from Loupe point processing."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_csv_has_correct_columns(self, service, tmp_path):
        """Test that output CSV has correct column structure."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        df = pd.read_csv(csv_files[0])
        
        assert 'raman_shift' in df.columns
        assert 'intensity' in df.columns
        assert len(df.columns) == 2

    def test_csv_has_expected_rows(self, service, tmp_path):
        """Test that output CSV has expected number of data points."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        df = pd.read_csv(csv_files[0])
        
        # R1 spectra typically have ~523 data points
        assert len(df) > 500
        assert len(df) == result.metadata["n_datapoints"]

    def test_csv_values_are_numeric(self, service, tmp_path):
        """Test that CSV values are valid numeric data."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        df = pd.read_csv(csv_files[0])
        
        # Check values are numeric and not NaN
        assert df['raman_shift'].dtype in [np.float64, np.int64, float, int]
        assert df['intensity'].dtype in [np.float64, np.int64, float, int]
        assert not df['raman_shift'].isna().any()
        assert not df['intensity'].isna().any()

    def test_csv_raman_shift_range(self, service, tmp_path):
        """Test that Raman shift values are in expected range."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point_from_loupe(request)
        
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        df = pd.read_csv(csv_files[0])
        
        # R1 detector spans approximately 238-4765 cm^-1
        assert df['raman_shift'].min() < 250
        assert df['raman_shift'].max() > 4700


class TestProcessPointFromLoupeComparison:
    """Test that different points/processing produce expected differences."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data processing."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_different_points_produce_different_spectra(self, service, tmp_path):
        """Test that different points produce different intensity values."""
        request0 = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=False,
            export="csv",
        )
        
        request5 = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result0 = service._process_point_from_loupe(request0)
        result5 = service._process_point_from_loupe(request5)
        
        csv0 = [a for a in result0.artifacts if str(a).endswith('.csv')][0]
        csv5 = [a for a in result5.artifacts if str(a).endswith('.csv')][0]
        
        df0 = pd.read_csv(csv0)
        df5 = pd.read_csv(csv5)
        
        # Different points should have different intensities
        assert not np.allclose(df0['intensity'].values, df5['intensity'].values)

    def test_processing_modifies_spectrum(self, service, tmp_path):
        """Test that processing steps modify the spectrum."""
        # Raw spectrum
        request_raw = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        # Processed spectrum
        request_proc = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=True,
            export="csv",
        )
        
        result_raw = service._process_point_from_loupe(request_raw)
        result_proc = service._process_point_from_loupe(request_proc)
        
        csv_raw = [a for a in result_raw.artifacts if str(a).endswith('.csv')][0]
        csv_proc = [a for a in result_proc.artifacts if str(a).endswith('.csv')][0]
        
        df_raw = pd.read_csv(csv_raw)
        df_proc = pd.read_csv(csv_proc)
        
        # Processing should change the spectrum
        assert not np.allclose(df_raw['intensity'].values, df_proc['intensity'].values)


class TestBackwardCompatibility:
    """Verify backward compatibility - existing point mode with --level still works."""

    @pytest.fixture
    def service_for_results(self, fixtures_path, tmp_path):
        """Service configured to load from pipeline output fixtures."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=fixtures_path / "pipeline_outputs",
        ))

    def test_level_mode_still_works(self, service_for_results):
        """Test that point mode with --level still loads from results."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            level="normalized",
        )
        
        result = service_for_results._process_point(request)
        
        # Should use results path (has "level" in metadata, no "ppp")
        assert result.metadata["level"] == "normalized"
        assert "ppp" not in result.metadata

    @pytest.mark.parametrize("level", [
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined",
    ])
    def test_all_levels_still_work(self, service_for_results, level):
        """Test that all processing levels still work."""
        # Use appropriate point for each level's fixture
        point = {"normalized": 0, "normalized_baselined": 5, "normalized_despiked_baselined": 3}[level]
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=point,
            level=level,
        )
        
        result = service_for_results._process_point(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata["level"] == level

