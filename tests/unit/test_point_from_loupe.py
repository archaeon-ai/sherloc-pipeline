"""
Unit tests for _process_point_from_loupe() method (T4.2).

Tests the processing of single points directly from raw Loupe data,
including background subtraction, baseline correction, and fitting.
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
    def service(self, test_context):
        """Service configured with test fixtures."""
        return SpectralService(context=test_context)

    def test_method_exists(self, service):
        """Verify the method exists."""
        assert hasattr(service, '_process_point_from_loupe')
        assert callable(service._process_point_from_loupe)

    def test_basic_point_processing(self, service, tmp_path, test_context):
        """Test basic single point processing returns ServiceResult."""
        # Override results_dir to use tmp_path for output
        service.context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        )
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=False,  # No processing for simplest case
        )
        
        result = service._process_point_from_loupe(request)
        
        assert isinstance(result, ServiceResult)
        assert "point 0" in result.summary
        assert result.metadata["point"] == 0
        assert "n_datapoints" in result.metadata
        assert "ppp" in result.metadata

    def test_point_processing_all_fixtures(self, test_context, tmp_path):
        """Test point processing works on all fixture datasets."""
        service = SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))
        
        test_cases = [
            ("0921", "Amherst_Point", "detail_1", 0),
            ("0852", "Lake_Haiyaha", "detail_1", 5),
            ("1634", "Stigbreen", "line_1", 10),
        ]
        
        for sol, target, scan, point in test_cases:
            request = SpectralPlotRequest(
                sol=sol,
                target=target,
                scan=scan,
                mode="point",
                point=point,
                baseline=False,
            )
            
            result = service._process_point_from_loupe(request)
            
            assert isinstance(result, ServiceResult)
            assert result.metadata["point"] == point
            assert f"{sol}/{target}/{scan}" in result.summary


class TestProcessPointFromLoupeValidation:
    """Test validation in _process_point_from_loupe()."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured with test fixtures."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_point_out_of_range_raises_error(self, service):
        """Test that out-of-range point raises SpectralPlotError."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=9999,  # Way out of range
            baseline=False,
        )
        
        with pytest.raises(SpectralPlotError, match="out of range"):
            service._process_point_from_loupe(request)

    def test_negative_point_raises_error(self, service):
        """Test that negative point raises SpectralPlotError."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=-1,
            baseline=False,
        )
        
        with pytest.raises(SpectralPlotError, match="out of range"):
            service._process_point_from_loupe(request)

    def test_first_point_valid(self, service):
        """Test that point 0 (first point) is valid."""
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

    def test_last_point_valid(self, service):
        """Test that the last valid point is accepted."""
        # First, get the number of points
        request_first = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            baseline=False,
        )
        result_first = service._process_point_from_loupe(request_first)
        total_points = result_first.metadata["total_points"]
        
        # Now try the last point
        request_last = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=total_points - 1,
            baseline=False,
        )
        result_last = service._process_point_from_loupe(request_last)
        assert result_last.metadata["point"] == total_points - 1


class TestProcessPointFromLoupeProcessing:
    """Test processing options in _process_point_from_loupe()."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured with test fixtures."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_background_subtraction_fs(self, service):
        """Test point processing with FS background subtraction."""
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
        assert "fs bg-subtracted" in result.summary

    def test_background_subtraction_as(self, service):
        """Test point processing with AS background subtraction."""
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
        assert "as bg-subtracted" in result.summary

    def test_baseline_correction(self, service):
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
        assert "baselined" in result.summary

    def test_fitting(self, service):
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
        assert "fit" in result.summary

    def test_single_peak_fitting(self, service):
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
        # Single peak mode should produce exactly 1 peak
        assert result.metadata["n_peaks"] == 1

    def test_n_peaks_limit(self, service):
        """Test point processing with n-peaks limit."""
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
            n_peaks=2,
        )
        
        result = service._process_point_from_loupe(request)
        
        assert result.metadata.get("fit") is True
        assert result.metadata["n_peaks"] <= 2

    def test_full_processing_chain(self, service):
        """Test full processing chain: bg-sub + baseline + fit."""
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
        
        assert result.metadata["background"] == "fs"
        assert result.metadata.get("baseline") is True
        assert result.metadata.get("fit") is True


class TestProcessPointFromLoupeOutput:
    """Test output generation from _process_point_from_loupe()."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured with test fixtures."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_creates_output_files(self, service, tmp_path):
        """Test that output files are created."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="both",
        )
        
        result = service._process_point_from_loupe(request)
        
        # Should have artifacts
        assert len(result.artifacts) > 0
        
        # Check that files exist
        for artifact in result.artifacts:
            assert Path(artifact).exists()

    def test_csv_export_has_correct_columns(self, service, tmp_path):
        """Test that CSV export has correct structure."""
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
        
        # Find CSV file (artifacts are Path objects)
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        assert len(csv_files) == 1
        
        # Load and verify structure
        df = pd.read_csv(csv_files[0])
        assert 'raman_shift' in df.columns
        assert 'intensity' in df.columns

    def test_metadata_includes_expected_fields(self, service, tmp_path):
        """Test that metadata includes all expected fields."""
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
        
        # Check required metadata fields
        assert "point" in result.metadata
        assert "ppp" in result.metadata
        assert "total_points" in result.metadata
        assert "n_datapoints" in result.metadata
        assert "background" in result.metadata
        assert "bgscale" in result.metadata
        assert "baseline" in result.metadata
        assert "fit" in result.metadata
        assert "n_peaks" in result.metadata
        assert "r2" in result.metadata


class TestProcessPointFromLoupeComparison:
    """Test that point processing produces expected results."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured with test fixtures."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_different_points_produce_different_results(self, service):
        """Test that different points produce different spectra."""
        results = []
        
        for point in [0, 5, 10]:
            request = SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=point,
                baseline=False,
                export="csv",
            )
            result = service._process_point_from_loupe(request)
            
            # Load CSV to compare (artifacts are Path objects)
            csv_file = [a for a in result.artifacts if str(a).endswith('.csv')][0]
            df = pd.read_csv(csv_file)
            results.append(df['intensity'].values)
        
        # Different points should have different intensities
        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[1], results[2])

    def test_processing_modifies_spectrum(self, service):
        """Test that processing steps modify the spectrum."""
        # Get raw spectrum
        request_raw = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        result_raw = service._process_point_from_loupe(request_raw)
        csv_raw = [a for a in result_raw.artifacts if str(a).endswith('.csv')][0]
        df_raw = pd.read_csv(csv_raw)
        
        # Get processed spectrum
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
        result_proc = service._process_point_from_loupe(request_proc)
        csv_proc = [a for a in result_proc.artifacts if str(a).endswith('.csv')][0]
        df_proc = pd.read_csv(csv_proc)
        
        # Processed should be different from raw
        assert not np.allclose(df_raw['intensity'].values, df_proc['intensity'].values)

