"""
Unit tests for Loupe data loading in SpectralService.

These tests verify that the _load_loupe_data() method correctly:
- Locates working directories by sol/target/scan
- Loads and parses loupe.csv metadata
- Loads laser-normalized spectra (darkSubSpectraN.csv)
- Restructures data to R1 (Raman) format
- Returns correct n_points and PPP values
"""

import pytest
import pandas as pd
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotError,
    LoupeData,
)
from sherloc_pipeline.services.runtime import RuntimeContext


class TestLoupeDataLoading:
    """Tests for _load_loupe_data() method."""

    def test_load_amherst_point_data(self, test_context: RuntimeContext):
        """Load sol 0921 Amherst_Point detail_1 (500 PPP, 100 points)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        # Verify return type
        assert isinstance(data, LoupeData)
        
        # Verify metadata
        assert data.n_points == 100
        assert data.ppp == 500.0
        assert data.working_dir.exists()
        
        # Verify DataFrame structure
        assert isinstance(data.spectra_df, pd.DataFrame)
        assert "raman_shift" in data.spectra_df.columns
        
        # Verify point columns exist (0, 1, 2, ..., 99)
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        assert len(point_cols) == 100
        assert 0 in point_cols
        assert 99 in point_cols

    def test_load_lake_haiyaha_data(self, test_context: RuntimeContext):
        """Load sol 0852 Lake_Haiyaha detail_1 (500 PPP, 100 points)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
        )
        
        assert data.n_points == 100
        assert data.ppp == 500.0
        assert "raman_shift" in data.spectra_df.columns

    def test_load_stigbreen_data(self, test_context: RuntimeContext):
        """Load sol 1634 Stigbreen line_1 (900 PPP, 25 points)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
        )
        
        assert data.n_points == 25
        assert data.ppp == 900.0
        
        # Verify point columns
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        assert len(point_cols) == 25

    def test_raman_shift_range(self, test_context: RuntimeContext):
        """Verify raman_shift values are in expected R1 range."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        raman_shift = data.spectra_df["raman_shift"]
        
        # R1 region spans roughly 238-4765 cm^-1 based on SHERLOC calibration
        # (wavelength range 250-282 nm corresponds to these Raman shifts)
        assert raman_shift.min() > 200
        assert raman_shift.max() < 5000
        
        # Should have many spectral points (>500 channels in R1)
        assert len(raman_shift) > 400

    def test_spectra_have_valid_intensities(self, test_context: RuntimeContext):
        """Verify intensity values are numeric and reasonable."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        # Check first point column
        point_0 = data.spectra_df[0]
        
        # Should be numeric
        assert point_0.dtype in [float, "float64", "float32"]
        
        # Should have some non-zero values
        assert point_0.abs().max() > 0
        
        # Should not be all NaN
        assert not point_0.isna().all()

    def test_metadata_contains_required_keys(self, test_context: RuntimeContext):
        """Verify metadata dict contains expected keys from loupe.csv."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        # Check for standard loupe.csv keys
        assert "n_spectra" in data.metadata
        assert "shots_per_spec" in data.metadata


class TestLoupeDataLoadingErrors:
    """Tests for error handling in _load_loupe_data()."""

    def test_invalid_sol_raises_error(self, test_context: RuntimeContext):
        """Non-existent sol should raise SpectralPlotError."""
        service = SpectralService(context=test_context)
        
        with pytest.raises(SpectralPlotError) as exc_info:
            service._load_loupe_data(
                sol="9999",
                target="Nonexistent",
                scan="detail_1",
            )
        
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_scan_raises_error(self, test_context: RuntimeContext):
        """Non-existent scan should raise SpectralPlotError."""
        service = SpectralService(context=test_context)
        
        with pytest.raises(SpectralPlotError) as exc_info:
            service._load_loupe_data(
                sol="0921",
                target="Amherst_Point",
                scan="nonexistent_scan",
            )
        
        assert "not found" in str(exc_info.value).lower()


class TestLoupeDataPPPValues:
    """Tests verifying PPP values match manifest expectations."""

    def test_500ppp_scans(self, test_context: RuntimeContext, manifest: dict):
        """Verify 500 PPP scans load with correct PPP value."""
        service = SpectralService(context=test_context)
        
        # Find 500 PPP datasets from manifest
        datasets_500ppp = [d for d in manifest["datasets"] if d["ppp"] == 500]
        
        for dataset in datasets_500ppp:
            data = service._load_loupe_data(
                sol=dataset["sol"],
                target=dataset["target"],
                scan=dataset["scan"],
            )
            
            assert data.ppp == 500.0, f"Expected 500 PPP for {dataset['sol']}/{dataset['scan']}"
            assert data.n_points == dataset["n_points"]

    def test_900ppp_scans(self, test_context: RuntimeContext, manifest: dict):
        """Verify 900 PPP scans load with correct PPP value."""
        service = SpectralService(context=test_context)
        
        # Find 900 PPP datasets from manifest
        datasets_900ppp = [d for d in manifest["datasets"] if d["ppp"] == 900]
        
        for dataset in datasets_900ppp:
            data = service._load_loupe_data(
                sol=dataset["sol"],
                target=dataset["target"],
                scan=dataset["scan"],
            )
            
            assert data.ppp == 900.0, f"Expected 900 PPP for {dataset['sol']}/{dataset['scan']}"
            assert data.n_points == dataset["n_points"]

