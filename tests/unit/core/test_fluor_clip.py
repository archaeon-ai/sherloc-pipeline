"""The in-fitter cosmic-ray clip must preserve legal fluorescence bands.

Synthetic spectra on the real Loupe wavelength axis over the fluorescence fit
range. The narrowest band the fitter accepts is 7 nm (``fwhm_range``), about
120-160 channels here, against an 11-channel clip window.

The clip may replace a few noise channels in flat stretches: on the steep flanks
of a strong band the window is monotonic, its median equals the centre value,
and those zero residuals pull the MAD below the true noise. Those replacements
stay at noise level, so the tests bound the change instead of requiring none.
"""

import numpy as np
import pytest

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.fluor_fitting import clip_cosmic_spikes

FIT_RANGE = (276.0, 357.0)
NOISE = 2.0


@pytest.fixture(scope="module")
def axis():
    wavelength, _ = calculate_loupe_wavelength_wavenumber(2148)
    wavelength = np.asarray(wavelength, dtype=float)
    return wavelength[(wavelength >= FIT_RANGE[0]) & (wavelength <= FIT_RANGE[1])]


def _band(x, centre, fwhm, amplitude):
    sigma = fwhm / 2.35482
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def _noisy(x, signal, seed):
    return 100.0 + signal + np.random.default_rng(seed).normal(0.0, NOISE, size=x.shape)


def _assert_band_preserved(x, before, after, centre, fwhm):
    inside = np.abs(x - centre) <= fwhm
    assert np.max(np.abs(after - before)) <= 5.0 * NOISE
    assert np.max(after[inside]) == pytest.approx(np.max(before[inside]), abs=3.0 * NOISE)
    area_before = np.trapezoid(before[inside] - 100.0, x[inside])
    area_after = np.trapezoid(after[inside] - 100.0, x[inside])
    assert area_after == pytest.approx(area_before, rel=0.01)


@pytest.mark.parametrize("fwhm", [7.0, 8.0])
@pytest.mark.parametrize("seed", [7, 11, 42])
def test_a_narrowest_legal_band_is_preserved(axis, fwhm, seed):
    y = _noisy(axis, _band(axis, 285.0, fwhm, 400.0), seed=seed)
    _assert_band_preserved(axis, y, clip_cosmic_spikes(y.copy()), 285.0, fwhm)


@pytest.mark.parametrize("seed", [7, 11, 42])
def test_the_304_325_doublet_is_preserved(axis, seed):
    y = _noisy(axis, _band(axis, 304.0, 7.0, 500.0) + _band(axis, 325.0, 7.0, 450.0), seed=seed)
    clipped = clip_cosmic_spikes(y.copy())
    _assert_band_preserved(axis, y, clipped, 304.0, 7.0)
    _assert_band_preserved(axis, y, clipped, 325.0, 7.0)


def test_a_spike_on_the_doublet_is_removed(axis):
    clean = _noisy(axis, _band(axis, 304.0, 7.0, 500.0) + _band(axis, 325.0, 7.0, 450.0), seed=13)
    spiked = clean.copy()
    k = int(np.argmin(np.abs(axis - 315.0)))
    spiked[k] += 3000.0

    clipped = clip_cosmic_spikes(spiked.copy())

    assert abs(clipped[k] - clean[k]) < 5.0 * NOISE
    others = np.arange(axis.size) != k
    assert np.max(np.abs(clipped[others] - spiked[others])) <= 5.0 * NOISE
