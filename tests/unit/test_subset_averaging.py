"""Unit tests for subset point averaging.

Tests the _process_subset() method that averages a user-specified subset
of points from Loupe data.

T2.7: Implement subset point loading and averaging
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
    SpectralPlotError,
    LoupeData,
    calculate_background_scale,
)
from sherloc_pipeline.services.base import ServiceResult


class TestSpectralPlotRequestSubsetValidation:
    """Tests for SpectralPlotRequest subset mode validation."""

    def test_subset_mode_requires_points(self):
        """Subset mode should require points to be specified."""
        with pytest.raises(ValueError, match="requires --points"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="subset",
                points=None,
            )

    def test_subset_mode_requires_at_least_two_points(self):
        """Subset mode should require at least 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="subset",
                points=[0],
            )

    def test_subset_mode_accepts_valid_points(self):
        """Subset mode should accept valid points list."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 5, 10],
        )
        assert request.mode == "subset"
        assert request.points == [0, 5, 10]


class TestProcessSubset:
    """Tests for SpectralService._process_subset() method."""

    def test_process_subset_returns_service_result(self, test_context):
        """_process_subset should return a ServiceResult."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4],
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert isinstance(result, ServiceResult)
        assert "subset" in result.summary.lower()

    def test_process_subset_validates_point_indices(self, test_context):
        """_process_subset should validate point indices are in range."""
        service = SpectralService(context=test_context)
        
        # Request points that are out of range
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 9999],  # 9999 is out of range
            export="csv",
        )
        
        with pytest.raises(SpectralPlotError, match="Invalid point indices"):
            service._process_subset(request)

    def test_process_subset_with_background(self, test_context):
        """_process_subset should support background subtraction."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4, 5],
            background="fs",
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert "background" in result.metadata
        assert result.metadata["background"] == "fs"

    def test_process_subset_with_baseline(self, test_context):
        """_process_subset should support baseline correction."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3],
            baseline=True,
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata.get("baseline") == True

    def test_process_subset_with_fitting(self, test_context):
        """_process_subset should support Gaussian fitting."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            background="fs",
            baseline=True,
            fit=True,
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata.get("fit") == True
        assert "n_peaks" in result.metadata

    def test_process_subset_metadata_includes_subset_info(self, test_context):
        """Metadata should include subset-specific information."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[5, 10, 15, 20, 25],
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata["n_points"] == 5
        assert result.metadata["subset_points"] == [5, 10, 15, 20, 25]
        assert "total_points" in result.metadata

    def test_process_subset_summary_mentions_subset(self, test_context):
        """Summary should mention subset mode."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2],
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert "subset" in result.summary.lower()
        assert "3" in result.summary  # Number of points


class TestProcessSubsetOnAllDatasets:
    """Tests for subset averaging on all fixture datasets."""

    def test_subset_on_amherst_point(self, test_context):
        """Subset averaging on Amherst Point detail scan."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 10, 20, 30, 40, 50],
            avg_method="trim-mean",
            trim_pct=2.0,
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) > 0
        assert result.metadata["n_points"] == 6

    def test_subset_on_lake_haiyaha(self, test_context):
        """Subset averaging on Lake Haiyaha detail scan."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
            mode="subset",
            points=[0, 5, 10, 15],
            avg_method="mean",
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert len(result.artifacts) > 0
        assert result.metadata["n_points"] == 4

    def test_subset_on_stigbreen(self, test_context):
        """Subset averaging on Stigbreen line scan."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
            mode="subset",
            points=[0, 1, 2, 3],
            avg_method="median",
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert len(result.artifacts) > 0
        assert result.metadata["n_points"] == 4


class TestSubsetAveragingMethods:
    """Tests for different averaging methods in subset mode."""

    def test_subset_mean_averaging(self, test_context):
        """Subset with mean averaging."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4],
            avg_method="mean",
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata["avg_method"] == "mean"

    def test_subset_median_averaging(self, test_context):
        """Subset with median averaging."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4],
            avg_method="median",
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata["avg_method"] == "median"

    def test_subset_trim_mean_averaging(self, test_context):
        """Subset with trim-mean averaging."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            avg_method="trim-mean",
            trim_pct=5.0,
            export="csv",
        )
        
        result = service._process_subset(request)
        
        assert result.metadata["avg_method"] == "trim-mean"
        assert result.metadata["trim_pct"] == 5.0


class TestSubsetProcessDispatch:
    """Tests for process() method dispatching to subset mode."""

    def test_process_dispatches_to_subset(self, test_context):
        """process() should dispatch to _process_subset for subset mode."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2, 3],
            export="csv",
        )
        
        result = service.process(request)
        
        assert isinstance(result, ServiceResult)
        assert "subset" in result.summary.lower()
