"""
Integration tests for point mode dispatch logic (T4.3).

Tests that _process_point() correctly dispatches to:
- _process_point_from_loupe(): When level is None
- _process_point_from_results(): When level is specified
"""

import pytest
import pandas as pd
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
    SpectralPlotError,
)
from sherloc_pipeline.services.runtime import RuntimeContext
from sherloc_pipeline.services.base import ServiceResult


class TestPointDispatchLogic:
    """Test the dispatch logic in _process_point()."""

    @pytest.fixture
    def service_loupe(self, test_context, tmp_path):
        """Service configured for Loupe data (test fixtures)."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    @pytest.fixture
    def service_results(self, fixtures_path, tmp_path):
        """Service configured for pipeline results (pipeline output fixtures)."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=fixtures_path / "pipeline_outputs",
        ))

    def test_dispatch_to_loupe_when_no_level(self, service_loupe):
        """When level is None, dispatch to _process_point_from_loupe()."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            # No level - should dispatch to Loupe processing
            baseline=False,
        )
        
        result = service_loupe._process_point(request)
        
        # Should return ServiceResult from Loupe processing
        assert isinstance(result, ServiceResult)
        assert result.metadata["point"] == 5
        # Loupe processing includes ppp and total_points
        assert "ppp" in result.metadata
        assert "total_points" in result.metadata
        # Should NOT have level in metadata
        assert "level" not in result.metadata

    def test_dispatch_to_results_when_level_specified(self, service_results, tmp_path):
        """When level is specified, dispatch to _process_point_from_results()."""
        # Update service to use tmp_path for output
        service_results.context = RuntimeContext.bootstrap(
            data_dir=service_results.context.data_root,
            results_dir=tmp_path,
        )
        # But we need results_dir to point to fixtures for loading
        service_results.context = RuntimeContext.bootstrap(
            results_dir=service_results.context.data_root.parent / "pipeline_outputs",
        )
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            level="normalized",  # Specified - should dispatch to results loading
            baseline=False,
        )
        
        result = service_results._process_point(request)
        
        # Should return ServiceResult from results loading
        assert isinstance(result, ServiceResult)
        assert result.metadata["point"] == 0
        # Results processing includes level
        assert result.metadata["level"] == "normalized"
        # Should NOT have ppp/total_points (those are from Loupe processing)
        assert "ppp" not in result.metadata

    def test_dispatch_validates_point_required(self, service_loupe):
        """Point mode requires --point (validated in SpectralPlotRequest)."""
        # Note: Validation happens in SpectralPlotRequest.__post_init__
        with pytest.raises(ValueError, match="requires --point"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=None,  # Missing - fails at request creation
                baseline=False,
            )


class TestDispatchWithProcessing:
    """Test dispatch with various processing options."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_loupe_dispatch_with_background(self, service):
        """Loupe dispatch supports background subtraction."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            background="fs",
            baseline=False,
        )
        
        result = service._process_point(request)
        
        assert result.metadata["background"] == "fs"
        assert "bgscale" in result.metadata

    def test_loupe_dispatch_with_baseline(self, service):
        """Loupe dispatch supports baseline correction."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=True,
        )
        
        result = service._process_point(request)
        
        assert result.metadata.get("baseline") is True

    def test_loupe_dispatch_with_fitting(self, service):
        """Loupe dispatch supports fitting."""
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
        
        result = service._process_point(request)
        
        assert result.metadata.get("fit") is True
        assert "n_peaks" in result.metadata

    def test_loupe_dispatch_with_single_peak(self, service):
        """Loupe dispatch supports single-peak fitting."""
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
        
        result = service._process_point(request)
        
        assert result.metadata.get("fit") is True
        assert result.metadata["n_peaks"] == 1


class TestDispatchBackwardCompatibility:
    """Test backward compatibility with existing point mode tests."""

    @pytest.fixture
    def service(self, fixtures_path, tmp_path):
        """Service configured with pipeline output fixtures as results."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=fixtures_path / "pipeline_outputs",
        ))

    def test_existing_point_mode_with_level_still_works(self, service, tmp_path):
        """Existing point mode with --level should continue to work."""
        # Override output directory
        service.context = RuntimeContext.bootstrap(
            results_dir=service.context.results_root,
        )
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            level="normalized",
        )
        
        result = service._process_point(request)
        
        assert isinstance(result, ServiceResult)
        assert "point 0" in result.summary
        assert "normalized" in result.summary

    def test_all_processing_levels_work(self, service):
        """All processing levels should work via dispatch."""
        levels = ["normalized", "normalized_baselined", "normalized_despiked_baselined"]
        points = [0, 5, 9]  # Different points for different levels in fixtures
        
        for level, point in zip(levels, points):
            request = SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=point,
                level=level,
            )
            
            result = service._process_point(request)
            
            assert isinstance(result, ServiceResult)
            assert result.metadata["level"] == level


class TestDispatchOutputs:
    """Test output generation via dispatch."""

    @pytest.fixture
    def service(self, test_context, tmp_path):
        """Service configured for Loupe data."""
        return SpectralService(context=RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=tmp_path,
        ))

    def test_dispatch_creates_output_files(self, service, tmp_path):
        """Dispatch should create output files via _process_point_from_loupe()."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="both",
        )
        
        result = service._process_point(request)
        
        # Should have created files
        assert len(result.artifacts) > 0
        for artifact in result.artifacts:
            assert Path(artifact).exists()

    def test_dispatch_csv_has_correct_structure(self, service, tmp_path):
        """CSV from dispatch should have correct structure."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            baseline=False,
            export="csv",
        )
        
        result = service._process_point(request)
        
        # Find and verify CSV
        csv_files = [a for a in result.artifacts if str(a).endswith('.csv')]
        assert len(csv_files) == 1
        
        df = pd.read_csv(csv_files[0])
        assert 'raman_shift' in df.columns
        assert 'intensity' in df.columns

