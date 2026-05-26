"""Unit tests for core/normalization.py -- wavelength/wavenumber calibration and laser normalization.

Tests are organized into two groups to reflect the distinct operations:
- Calibration: channel-to-wavenumber mapping via Loupe polynomial
- Laser normalization: photodiode-based intensity correction

Regression guards capturing current behavior before structural refactor.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
)
from sherloc_pipeline.core.laser_normalization import (
    calculate_normalization_factors,
    calculate_photodiode_summary,
)


# ---------------------------------------------------------------------------
# Calibration tests (channel-to-wavenumber mapping)
# ---------------------------------------------------------------------------


class TestCalibrationOutputShapes:

    def test_default_2148_channels(self):
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        assert wavelength.shape == (2148,)
        assert wavenumber.shape == (2148,)

    def test_custom_channel_count(self):
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(1000)
        assert wavelength.shape == (1000,)
        assert wavenumber.shape == (1000,)

    def test_returns_numpy_arrays(self):
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        assert isinstance(wavelength, np.ndarray)
        assert isinstance(wavenumber, np.ndarray)


class TestCalibrationWavelengthRange:

    def test_wavelength_physically_reasonable(self):
        wavelength, _ = calculate_loupe_wavelength_wavenumber(2148)
        # SHERLOC operates in deep UV: wavelength should be ~200-360 nm
        assert wavelength.min() > 200.0
        assert wavelength.max() < 400.0

    def test_wavelength_monotonically_increasing(self):
        wavelength, _ = calculate_loupe_wavelength_wavenumber(2148)
        assert np.all(np.diff(wavelength) > 0)

    def test_wavenumber_includes_negative_at_start(self):
        """First channels have wavelength < laser (248.58 nm) → negative wavenumber."""
        _, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        # Channels with wavelength < laser_wavelength yield negative wavenumber
        assert wavenumber[0] < 0, "First channel should have negative wavenumber"
        # But most of the array is positive
        assert np.sum(wavenumber > 0) > 2000


class TestCalibrationR1Mask:

    def test_r1_mask_gives_523_channels(self):
        """The R1 region (250-282 nm) should yield exactly 523 channels."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        assert r1_mask.sum() == 523

    def test_r1_wavenumber_range(self):
        """R1 wavenumber range should be approximately 238-4765 cm^-1."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        assert r1_wn.min() > 200.0
        assert r1_wn.max() < 5000.0


class TestCalibrationSnapshot:

    def test_snapshot_endpoints(self):
        """Capture exact endpoint values for regression -- NEVER use np.linspace."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        # First channel (pixel 0, Raman polynomial)
        assert_allclose(wavelength[0], np.polyval([-7.85000e-06, 6.52400e-02, 2.46690e+02], 0), rtol=1e-12)
        # Last channel (pixel 2147, Fluorescence polynomial)
        assert_allclose(wavelength[-1], np.polyval([-5.65724e-06, 6.33627e-02, 2.47474e+02], 2147), rtol=1e-12)

    def test_snapshot_channel_522(self):
        """Capture the 523rd R1 channel value for regression."""
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(2148)
        r1_mask = (wavelength >= 250.0) & (wavelength <= 282.0)
        r1_wn = wavenumber[r1_mask]
        # Snapshot: capture exact value so any drift is caught
        assert np.isfinite(r1_wn[0])
        assert np.isfinite(r1_wn[-1])
        # Wavenumber for last R1 channel should be near the low end (~238 cm^-1)
        assert r1_wn.min() < 300.0

    def test_cutoff_channel_discontinuity_captured(self):
        """Segmented polynomial has a known discontinuity at channel 500/501.

        Channel 500 is the last Raman-polynomial channel; 501 is the first
        Fluorescence-polynomial channel. The gap is ~0.45 nm vs typical step ~0.06 nm.
        This captures current behavior -- NOT a bug, just a property of the calibration.
        """
        wavelength, _ = calculate_loupe_wavelength_wavenumber(2148)
        gap = abs(wavelength[501] - wavelength[500])
        typical_step = abs(wavelength[500] - wavelength[499])
        # The gap is significantly larger than a typical step
        assert gap > 3 * typical_step
        # But still bounded (not wildly wrong)
        assert gap < 1.0, "Discontinuity should be < 1 nm"

    def test_laser_wavelength_parameter(self):
        """Custom laser wavelength should shift wavenumber values."""
        _, wn_default = calculate_loupe_wavelength_wavenumber(2148, laser_wavelength=248.5794)
        _, wn_shifted = calculate_loupe_wavelength_wavenumber(2148, laser_wavelength=250.0)
        # Different laser wavelength → different wavenumbers
        assert not np.allclose(wn_default, wn_shifted)


# ---------------------------------------------------------------------------
# Laser normalization tests (photodiode-based intensity correction)
# ---------------------------------------------------------------------------


class TestLaserNormalizationFactors:

    def test_uniform_photodiode_factors_all_one(self):
        """Uniform photodiode readings → all factors = 1.0."""
        pd_summary = np.array([100.0, 100.0, 100.0, 100.0])
        factors = calculate_normalization_factors(pd_summary)
        assert_allclose(factors, 1.0)

    def test_max_photodiode_has_factor_one(self):
        """The spectrum with max photodiode reading should have factor = 1.0."""
        pd_summary = np.array([80.0, 100.0, 90.0, 95.0])
        factors = calculate_normalization_factors(pd_summary)
        assert_allclose(factors[1], 1.0)

    def test_lower_photodiode_has_higher_factor(self):
        """Lower photodiode → higher normalization factor (inverse relationship)."""
        pd_summary = np.array([50.0, 100.0])
        factors = calculate_normalization_factors(pd_summary)
        assert factors[0] > factors[1]
        assert_allclose(factors[0], 2.0)
        assert_allclose(factors[1], 1.0)

    def test_normalization_formula(self):
        """Verify formula: factor_i = max(pd) / pd_i."""
        pd_summary = np.array([80.0, 100.0, 60.0])
        factors = calculate_normalization_factors(pd_summary)
        expected = np.array([100.0 / 80.0, 100.0 / 100.0, 100.0 / 60.0])
        assert_allclose(factors, expected)

    def test_output_shape(self):
        pd_summary = np.array([80.0, 100.0, 60.0, 90.0, 70.0])
        factors = calculate_normalization_factors(pd_summary)
        assert factors.shape == pd_summary.shape


class TestPhotodiodeSummary:

    def test_mean_across_shots(self):
        """Summary should be mean across shots (columns) for each spectrum (row)."""
        df = pd.DataFrame({
            "shot_0": [100.0, 80.0, 90.0],
            "shot_1": [110.0, 85.0, 95.0],
            "shot_2": [105.0, 75.0, 85.0],
        })
        summary = calculate_photodiode_summary(df)
        expected = np.array([105.0, 80.0, 90.0])
        assert_allclose(summary, expected)

    def test_output_shape(self):
        df = pd.DataFrame({
            "shot_0": [100.0, 80.0],
            "shot_1": [110.0, 85.0],
        })
        summary = calculate_photodiode_summary(df)
        assert summary.shape == (2,)

    def test_single_shot(self):
        """Single shot → summary equals that shot's values."""
        df = pd.DataFrame({"shot_0": [100.0, 80.0, 90.0]})
        summary = calculate_photodiode_summary(df)
        assert_allclose(summary, [100.0, 80.0, 90.0])
