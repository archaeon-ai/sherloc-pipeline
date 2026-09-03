"""Unit tests for the hydration cosmic-ray veto (issue #38).

Each mechanism — spike-mask hit, raw->despiked amplitude drop, and FWHM
bound-pinning — is exercised independently, plus the default-OFF contract.
"""

import numpy as np
import pytest

from sherloc_pipeline.core.hydration_veto import (
    FLAG_AMPLITUDE_DROP,
    FLAG_FWHM_FLOOR_PINNED,
    FLAG_MASK_HIT,
    HydrationVetoConfig,
    despike_for_veto,
    evaluate_hydration_peak,
)


def _broad_oh_spectrum(center=3400.0, fwhm=250.0, amplitude=500.0, noise=5.0):
    """A wide, authentic-looking OH stretch band over the hydration window.

    Detector noise is included deliberately: the despiker's threshold is a
    MAD-derived sigma, so a perfectly noiseless synthetic would make every
    channel a 6-sigma outlier and tell us nothing about real behaviour.
    """
    x = np.linspace(2800.0, 3900.0, 400)
    sigma = fwhm / 2.3548
    y = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    y = y + np.random.default_rng(38).normal(0.0, noise, size=x.shape)
    return x, y


def _with_cosmic_ray(x, y, center=3200.0, amplitude=4000.0):
    """Add a two-channel spike — the intrinsic width of a cosmic ray."""
    y = y.copy()
    idx = int(np.argmin(np.abs(x - center)))
    y[idx] += amplitude
    y[idx + 1] += amplitude * 0.6
    return y


class TestDefaultOff:
    def test_flag_defaults_to_off(self):
        cfg = HydrationVetoConfig.from_fitting_config({"hydration_fwhm_min_cm1": 50.0})
        assert cfg.enabled is False

    def test_missing_config_section_is_off(self):
        assert HydrationVetoConfig.from_fitting_config(None).enabled is False

    def test_floor_tracks_the_fit_bound(self):
        cfg = HydrationVetoConfig.from_fitting_config({"hydration_fwhm_min_cm1": 61.5})
        assert cfg.fwhm_floor_cm1 == pytest.approx(61.5)

    def test_action_flag_never_rejects(self):
        cfg = HydrationVetoConfig(enabled=True, action="flag")
        assert cfg.rejects is False


class TestMaskVeto:
    def test_candidate_on_a_masked_spike_bin_is_vetoed(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)
        assert mask.any(), "despiker must flag the injected spike"

        cfg = HydrationVetoConfig(enabled=True)
        result = evaluate_hydration_peak(3200.0, 50.0, x, y, despiked, mask, cfg)

        assert result.mask_hit is True
        assert FLAG_MASK_HIT in result.flags
        assert result.vetoed is True

    def test_broad_band_away_from_the_spike_survives(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)

        cfg = HydrationVetoConfig(enabled=True)
        result = evaluate_hydration_peak(3400.0, 250.0, x, y, despiked, mask, cfg)

        assert result.mask_hit is False
        assert result.vetoed is False
        assert result.flags == ()

    def test_flag_action_annotates_without_rejecting(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)

        cfg = HydrationVetoConfig(enabled=True, action="flag")
        result = evaluate_hydration_peak(3200.0, 50.0, x, y, despiked, mask, cfg)

        assert result.mask_hit is True
        assert result.vetoed is False

    def test_trivial_masked_nick_on_a_broad_band_does_not_fire(self):
        # False-positive guard: the rolling-median despiker routinely masks a
        # single noise channel near the apex of an authentic broad band. That
        # bin removes ~1% of the band height and must not veto it.
        x, clean = _broad_oh_spectrum()
        y = clean.copy()
        idx = int(np.argmin(np.abs(x - 3400.0)))
        y[idx] += 6.0
        despiked = y.copy()
        despiked[idx] = clean[idx]
        mask = np.zeros(x.shape, dtype=bool)
        mask[idx] = True

        cfg = HydrationVetoConfig(enabled=True)
        result = evaluate_hydration_peak(3400.0, 250.0, x, y, despiked, mask, cfg)

        assert result.mask_hit is False
        assert result.vetoed is False

    def test_lowering_the_mask_floor_admits_the_trivial_nick(self):
        x, clean = _broad_oh_spectrum()
        y = clean.copy()
        idx = int(np.argmin(np.abs(x - 3400.0)))
        y[idx] += 6.0
        despiked = y.copy()
        despiked[idx] = clean[idx]
        mask = np.zeros(x.shape, dtype=bool)
        mask[idx] = True

        cfg = HydrationVetoConfig(enabled=True, mask_min_drop_ratio=0.0)
        result = evaluate_hydration_peak(3400.0, 250.0, x, y, despiked, mask, cfg)

        assert result.mask_hit is True

    def test_despiker_failure_yields_an_empty_mask(self):
        # A one-sample spectrum cannot be rolling-median filtered meaningfully;
        # the veto must degrade to "no opinion", never to a rejection.
        x = np.array([3400.0])
        y = np.array([10.0])
        despiked, mask = despike_for_veto(x, y)
        assert mask.shape == y.shape
        assert not mask.any()


class TestAmplitudeRatio:
    def test_spike_dominated_candidate_is_vetoed_on_ratio_alone(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)
        # Isolate the amplitude mechanism by suppressing the mask signal.
        blank = np.zeros_like(mask)

        cfg = HydrationVetoConfig(enabled=True)
        result = evaluate_hydration_peak(3200.0, 50.0, x, y, despiked, blank, cfg)

        assert result.mask_hit is False
        assert result.amplitude_drop_ratio is not None
        assert result.amplitude_drop_ratio > cfg.amplitude_drop_ratio_max
        assert FLAG_AMPLITUDE_DROP in result.flags
        assert result.vetoed is True

    def test_authentic_band_has_a_small_drop_ratio(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)

        cfg = HydrationVetoConfig(enabled=True)
        result = evaluate_hydration_peak(3400.0, 250.0, x, clean, despiked, mask, cfg)

        assert result.amplitude_drop_ratio is not None
        assert result.amplitude_drop_ratio <= cfg.amplitude_drop_ratio_max
        assert FLAG_AMPLITUDE_DROP not in result.flags

    def test_permissive_threshold_spares_the_spike(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)
        blank = np.zeros_like(mask)

        cfg = HydrationVetoConfig(enabled=True, amplitude_drop_ratio_max=1.0)
        result = evaluate_hydration_peak(3200.0, 50.0, x, y, despiked, blank, cfg)

        assert result.vetoed is False


class TestBoundPinning:
    def test_fit_at_the_floor_is_flagged(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)

        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=50.0)
        result = evaluate_hydration_peak(3400.0, 50.2, x, clean, despiked, mask, cfg)

        assert result.bound_pinned is True
        assert FLAG_FWHM_FLOOR_PINNED in result.flags

    def test_bound_pinning_alone_never_rejects(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)

        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=50.0)
        result = evaluate_hydration_peak(3400.0, 50.0, x, clean, despiked, mask, cfg)

        assert result.bound_pinned is True
        assert result.vetoed is False

    def test_interior_width_is_not_pinned(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)

        cfg = HydrationVetoConfig(enabled=True, fwhm_floor_cm1=50.0)
        result = evaluate_hydration_peak(3400.0, 250.0, x, clean, despiked, mask, cfg)

        assert result.bound_pinned is False

    def test_epsilon_widens_the_bound_band(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)

        cfg = HydrationVetoConfig(
            enabled=True, fwhm_floor_cm1=50.0, fwhm_floor_epsilon_cm1=10.0
        )
        result = evaluate_hydration_peak(3400.0, 58.0, x, clean, despiked, mask, cfg)

        assert result.bound_pinned is True


class TestSerialisation:
    def test_row_carries_every_signal(self):
        x, clean = _broad_oh_spectrum()
        y = _with_cosmic_ray(x, clean)
        despiked, mask = despike_for_veto(x, y)

        cfg = HydrationVetoConfig(enabled=True)
        row = evaluate_hydration_peak(3200.0, 50.0, x, y, despiked, mask, cfg).as_row()

        assert row["cr_vetoed"] is True
        assert row["cr_mask_hit"] is True
        assert row["fwhm_floor_pinned"] is True
        assert FLAG_MASK_HIT in row["cr_veto_flags"]
        assert 0.0 <= row["cr_amplitude_drop_ratio"] <= 1.0

    def test_missing_center_is_a_no_opinion(self):
        x, clean = _broad_oh_spectrum()
        despiked, mask = despike_for_veto(x, clean)
        result = evaluate_hydration_peak(
            None, 50.0, x, clean, despiked, mask, HydrationVetoConfig(enabled=True)
        )
        assert result.vetoed is False
        assert result.flags == ()
