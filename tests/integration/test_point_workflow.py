"""
Integration tests for point visualization workflow in SpectralService.

T2.2: Tests for _process_point() method that loads and visualizes
single points from existing pipeline outputs.
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


class TestProcessPointWorkflow:
    """Test the full _process_point() workflow."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_basic_point_workflow(self, service_with_pipeline_fixtures, tmp_path):
        """Test basic point visualization workflow."""
        service = service_with_pipeline_fixtures
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            level="normalized",
            export="both",
        )
        
        result = service.process(request)
        
        # Check ServiceResult structure
        assert isinstance(result, ServiceResult)
        assert "point 0" in result.summary
        assert "0921/Amherst_Point/detail_1" in result.summary
        assert "normalized" in result.summary
        
        # Check metadata
        assert result.metadata["point"] == 0
        assert result.metadata["level"] == "normalized"
        assert result.metadata["n_datapoints"] == 32
    
    def test_point_workflow_with_xlim(self, service_with_pipeline_fixtures, tmp_path):
        """Test point visualization with x-axis limits."""
        service = service_with_pipeline_fixtures
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            level="normalized_baselined",
            xlim=(500, 1500),
            export="png",
        )
        
        result = service.process(request)
        
        # Should complete successfully
        assert isinstance(result, ServiceResult)
        assert result.metadata["point"] == 5
        assert result.metadata["level"] == "normalized_baselined"
    
    def test_point_workflow_with_ylim(self, service_with_pipeline_fixtures, tmp_path):
        """Test point visualization with y-axis limits."""
        service = service_with_pipeline_fixtures
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=3,
            level="normalized_despiked_baselined",
            ylim=(-100, 500),
            export="csv",
        )
        
        result = service.process(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata["point"] == 3
        assert result.metadata["level"] == "normalized_despiked_baselined"
    
    def test_point_workflow_creates_artifacts(self, service_with_pipeline_fixtures, tmp_path):
        """Test that point workflow creates expected output files."""
        service = service_with_pipeline_fixtures
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=2,
            level="normalized",
            export="both",
        )
        
        result = service.process(request)
        
        # Should have created files (CSV + PNG + JSON metadata)
        assert len(result.artifacts) == 3
        
        # Check artifact paths
        csv_artifact = [a for a in result.artifacts if str(a).endswith('.csv')][0]
        png_artifact = [a for a in result.artifacts if str(a).endswith('.png')][0]
        json_artifact = [a for a in result.artifacts if str(a).endswith('.json')][0]
        
        assert csv_artifact.exists()
        assert png_artifact.exists()
        assert json_artifact.exists()
        
        # Check filename pattern
        assert "p2" in csv_artifact.name
        assert "normalized" in csv_artifact.name


class TestProcessPointErrors:
    """Test error handling in _process_point()."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_invalid_point_raises_error(self, service_with_pipeline_fixtures):
        """Test that out-of-range point index raises error."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=100,  # Out of range
            level="normalized",
        )
        
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures.process(request)
        
        assert "Point 100 not found" in str(excinfo.value)
    
    def test_missing_pipeline_output_raises_error(self, service_with_pipeline_fixtures):
        """Test that missing pipeline output file raises error."""
        request = SpectralPlotRequest(
            sol="9999",
            target="Nonexistent",
            scan="detail_1",
            mode="point",
            point=0,
            level="normalized",
        )
        
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures.process(request)
        
        assert "Pipeline output not found" in str(excinfo.value)


class TestProcessPointOutputNaming:
    """Test output filename generation for point mode."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    @pytest.mark.parametrize("point,level,expected_parts", [
        (0, "normalized", ["p0", "normalized"]),
        (5, "normalized_baselined", ["p5", "normalized_baselined"]),
        (9, "normalized_despiked_baselined", ["p9", "normalized_despiked_baselined"]),
    ])
    def test_filename_contains_point_and_level(
        self,
        service_with_pipeline_fixtures,
        point,
        level,
        expected_parts
    ):
        """Test that output filenames contain point number and level."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=point,
            level=level,
            export="csv",
        )
        
        result = service_with_pipeline_fixtures.process(request)
        
        # CSV + JSON metadata
        assert len(result.artifacts) == 2
        csv_artifact = [a for a in result.artifacts if str(a).endswith('.csv')][0]
        filename = csv_artifact.name
        
        for part in expected_parts:
            assert part in filename, f"Expected '{part}' in filename '{filename}'"


class TestProcessPointTitleGeneration:
    """Test plot title generation for point mode."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_point_title_format(self, service_with_pipeline_fixtures):
        """Test that point mode title has correct format."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            level="normalized_baselined",
        )
        
        title = service_with_pipeline_fixtures._build_plot_title(request)
        
        assert "sol 0921" in title
        assert "Amherst_Point" in title
        assert "detail_1" in title
        assert "point 91" in title
        assert "normalized_baselined" in title
        # Should NOT have "avg" in point mode
        assert "avg" not in title


class TestProcessPointWithAllLevels:
    """Parametrized tests for all processing levels."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    @pytest.mark.parametrize("level", [
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined",
    ])
    def test_process_each_level(self, service_with_pipeline_fixtures, level):
        """Test that each processing level can be visualized."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=0,
            level=level,
            export="csv",
        )
        
        result = service_with_pipeline_fixtures.process(request)
        
        assert isinstance(result, ServiceResult)
        assert result.metadata["level"] == level
        assert len(result.artifacts) == 2  # CSV + JSON

