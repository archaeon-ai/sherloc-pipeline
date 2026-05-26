"""
Unit tests for SpectralPlotRequest validation (T4.1).

Tests the updated validation logic that makes --level optional for point mode,
enabling point mode to process from raw Loupe data (without --level) or
load from pre-processed pipeline outputs (with --level).
"""

import pytest
import logging

from sherloc_pipeline.services.spectral import (
    SpectralPlotRequest,
)


class TestPointModeValidation:
    """Test point mode validation with optional --level."""

    def test_point_mode_valid_without_level(self):
        """SpectralPlotRequest with point but no level is now valid."""
        # This used to raise ValueError, now it should succeed
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            # No level - will process from Loupe data
        )
        
        assert request.mode == "point"
        assert request.point == 5
        assert request.level is None

    def test_point_mode_valid_with_level(self):
        """SpectralPlotRequest with point and level still works (legacy)."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=91,
            level="normalized",
        )
        
        assert request.mode == "point"
        assert request.point == 91
        assert request.level == "normalized"

    def test_point_mode_valid_with_all_levels(self):
        """All valid level values should work."""
        valid_levels = [
            "normalized",
            "normalized_baselined",
            "normalized_despiked_baselined",
        ]
        
        for level in valid_levels:
            request = SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=0,
                level=level,
            )
            assert request.level == level

    def test_point_mode_still_requires_point(self):
        """Point mode without --point should still raise error."""
        with pytest.raises(ValueError, match="Point mode requires --point"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=None,  # Missing point
            )

    def test_point_mode_invalid_level_rejected(self):
        """Invalid level values should be rejected."""
        with pytest.raises(ValueError, match="Invalid level"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=5,
                level="invalid_level",
            )


class TestPointModeProcessingFlags:
    """Test processing flags validation for point mode."""

    def test_processing_flags_allowed_without_level(self):
        """Processing flags (--background, --baseline, --fit) allowed when level=None."""
        # This enables point mode to use full processing chain from Loupe data
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            # No level
            background="fs",
            baseline=True,
            fit=True,
        )
        
        assert request.background == "fs"
        assert request.baseline is True
        assert request.fit is True
        assert request.level is None

    def test_processing_flags_with_level_logs_warning(self, caplog):
        """Processing flags with level set should log a warning."""
        with caplog.at_level(logging.WARNING):
            request = SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=5,
                level="normalized",
                background="fs",  # Will be ignored
                baseline=True,    # Will be ignored
            )
        
        # Should have logged a warning
        assert any("Processing flags" in record.message for record in caplog.records)
        assert any("ignored" in record.message.lower() for record in caplog.records)

    def test_fit_with_level_logs_warning(self, caplog):
        """Fit flag with level set should log a warning."""
        with caplog.at_level(logging.WARNING):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=5,
                level="normalized_baselined",
                fit=True,  # Will be ignored
            )
        
        assert any("ignored" in record.message.lower() for record in caplog.records)

    def test_no_warning_without_processing_flags(self, caplog):
        """No warning when level is set but no processing flags."""
        with caplog.at_level(logging.WARNING):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="point",
                point=5,
                level="normalized",
                # Explicitly disable all processing flags (baseline defaults to True)
                background=None,
                baseline=False,
                fit=False,
            )
        
        # Should not have logged any warning about processing flags
        assert not any("Processing flags" in record.message for record in caplog.records)


class TestPointModeFittingOptions:
    """Test fitting options for point mode without level."""

    def test_single_peak_allowed_without_level(self):
        """--single-peak allowed for point mode without level."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            fit=True,
            single_peak_center=1090.0,
        )
        
        assert request.single_peak_center == 1090.0

    def test_n_peaks_allowed_without_level(self):
        """--n-peaks allowed for point mode without level."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            fit=True,
            n_peaks=3,
        )
        
        assert request.n_peaks == 3

    def test_fit_range_allowed_without_level(self):
        """--fit-range allowed for point mode without level."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            fit=True,
            fit_range=(700, 1200),
        )
        
        assert request.fit_range == (700, 1200)


class TestPointModeAxisControls:
    """Test axis controls for point mode."""

    def test_xlim_ylim_work_without_level(self):
        """Axis controls work for point mode without level."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            xlim=(700, 1300),
            ylim=(0, 1000),
        )
        
        assert request.xlim == (700, 1300)
        assert request.ylim == (0, 1000)

    def test_xlim_ylim_work_with_level(self):
        """Axis controls work for point mode with level (unchanged)."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="point",
            point=5,
            level="normalized",
            xlim=(700, 1300),
            ylim=(0, 1000),
        )
        
        assert request.xlim == (700, 1300)
        assert request.ylim == (0, 1000)


class TestOtherModesUnchanged:
    """Verify other modes (averaged, subset) are unchanged by T4.1."""

    def test_averaged_mode_unchanged(self):
        """Averaged mode validation should be unchanged."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
            baseline=True,
            fit=True,
        )
        
        assert request.mode == "averaged"

    def test_subset_mode_unchanged(self):
        """Subset mode validation should be unchanged."""
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="subset",
            points=[0, 1, 2],
            background="fs",
            baseline=True,
        )
        
        assert request.mode == "subset"
        assert request.points == [0, 1, 2]

    def test_subset_mode_still_requires_points(self):
        """Subset mode should still require --points."""
        with pytest.raises(ValueError, match="Subset mode requires"):
            SpectralPlotRequest(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                mode="subset",
                points=None,
            )

