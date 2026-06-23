"""
Unit tests for the Python API load_reference_spectrum() function.

Tests verify that the API function correctly loads reference mineral spectra
from the library.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


from sherloc_pipeline.api.spectral import load_reference_spectrum


class TestLoadReferenceSpectrum:
    """Tests for load_reference_spectrum() API function."""
    
    @pytest.fixture
    def reference_dir(self, fixtures_path):
        """Return path to reference spectrum fixtures."""
        return fixtures_path / "reference"
    
    def test_basic_return_type(self, reference_dir):
        """Test that function returns a DataFrame."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        assert isinstance(df, pd.DataFrame)
    
    def test_dataframe_structure(self, reference_dir):
        """Test that DataFrame has raman_shift and intensity columns."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df.columns) == 2
    
    def test_dataframe_has_data(self, reference_dir):
        """Test that DataFrame contains spectral data."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        assert len(df) > 0
        # Should have meaningful data
        assert not df["raman_shift"].isna().all()
        assert not df["intensity"].isna().all()
    
    def test_case_insensitive_matching(self, reference_dir):
        """Test that mineral matching is case-insensitive."""
        # All of these should work
        for mineral in ["forsterite", "Forsterite", "FORSTERITE", "ForsterITE"]:
            df = load_reference_spectrum(
                mineral=mineral,
                library_path=reference_dir,
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_partial_name_matching(self, reference_dir):
        """Test that partial mineral names work."""
        # "forst" should match "forsterite"
        df = load_reference_spectrum(
            mineral="forst",
            library_path=reference_dir,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_forsterite_raman_shift_range(self, reference_dir):
        """Test that forsterite reference has expected Raman shift range."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        # Forsterite reference should span negative to positive Raman shifts
        # (includes the full wavelength range)
        assert df["raman_shift"].min() < 0
        assert df["raman_shift"].max() > 4000  # Extends beyond typical R1 range
    
    def test_forsterite_data_values(self, reference_dir):
        """Test that forsterite intensity values are reasonable."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        # Baselined intensity can have negative values (over-subtraction)
        # but should have positive peaks in the mineral region
        olivine_mask = (df["raman_shift"] >= 800) & (df["raman_shift"] <= 900)
        olivine_region = df[olivine_mask]
        assert olivine_region["intensity"].max() > 0  # Olivine peak should be positive
        
        # Should have some variation
        assert df["intensity"].std() > 0
    
    def test_nonexistent_mineral_raises_error(self, reference_dir):
        """Test that non-existent mineral raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No reference spectrum found"):
            load_reference_spectrum(
                mineral="unobtainium",
                library_path=reference_dir,
            )
    
    def test_nonexistent_library_raises_error(self, tmp_path):
        """Test that non-existent library path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Reference library not found"):
            load_reference_spectrum(
                mineral="forsterite",
                library_path=tmp_path / "nonexistent",
            )
    
    def test_data_types(self, reference_dir):
        """Test that columns have correct data types."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        # Both columns should be numeric
        assert pd.api.types.is_numeric_dtype(df["raman_shift"])
        assert pd.api.types.is_numeric_dtype(df["intensity"])
    
    def test_no_nan_values(self, reference_dir):
        """Test that DataFrame has no NaN values."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        assert not df["raman_shift"].isna().any()
        assert not df["intensity"].isna().any()
    
    def test_sorted_raman_shift(self, reference_dir):
        """Test that Raman shift values are in order (ascending)."""
        df = load_reference_spectrum(
            mineral="forsterite",
            library_path=reference_dir,
        )
        
        # Check monotonically increasing
        diff = df["raman_shift"].diff().dropna()
        assert (diff > 0).all(), "Raman shift values should be monotonically increasing"


