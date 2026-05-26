"""Tests for parallel fitting worker functions.

Verifies that the extracted module-level worker functions (_fit_point_minerals,
_fit_point_hydration, _fit_point_organics) produce correct output structures
and can be called from ProcessPoolExecutor (pickling requirement).
"""

import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pytest

from sherloc_pipeline.services.fitting import (
    _fit_point_minerals,
    _fit_point_hydration,
    _fit_point_organics,
)


def _make_synthetic_spectrum(n=523, peak_center=1015.0, peak_amp=100.0,
                              fwhm=35.0, noise_std=5.0, x_lo=700.0, x_hi=1200.0):
    """Create a synthetic Raman spectrum with one Gaussian peak + noise."""
    rng = np.random.RandomState(42)
    x = np.linspace(x_lo, x_hi, n)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y = peak_amp * np.exp(-0.5 * ((x - peak_center) / sigma) ** 2)
    y += rng.normal(0, noise_std, n)
    return x, y


def _minimal_fit_cfg():
    """Return a minimal fitting config dict for testing."""
    return {
        'max_peaks': 5,
        'min_snr': 3.0,
        'min_seed_snr': 2.0,
        'min_display_snr': 2.0,
        'fit_fwhm_min_initial_cm1': 22.0,
        'filter_fwhm_min_cm1': 25.0,
        'reviewable_fwhm_min_cm1': 25.0,
        'fwhm_max_cm1': 90.0,
        'slit_width_cm1_default': 34.1,
        'slit_pref_weight': 0.2,
        'low_fwhm_edge_penalty': 0.1,
        'r_squared_min': 0.25,
        'min_amp_sigma_multiplier': 0.3,
        'peak_separation_cm1': 25.0,
        'noise_estimation': {'window': [2000.0, 2100.0]},
        'parsimony': {
            'model_selection': 'ftest',
            'ftest_alpha': 0.01,
            'use_aicc': False,
            'aicc_min_peaks': 1,
            'aicc_max_peaks': 5,
            'aicc_improve_threshold': 0.0,
        },
        'dynamic_slit': {'enabled': False},
        'posthoc_filters': {
            'r2_min': 0.0,
            'sharpness_max': 3.0,
        },
    }


class TestFitPointMinerals:
    """Tests for _fit_point_minerals worker function."""

    def test_returns_expected_keys(self, tmp_path):
        """Worker returns dict with all required keys."""
        x, y = _make_synthetic_spectrum()
        point_data = {'point_idx': 0, 'y': y}

        result = _fit_point_minerals(
            point_data,
            x=x,
            fit_cfg=_minimal_fit_cfg(),
            fit_roi=(700.0, 1200.0),
            plot_roi=(700.0, 1300.0),
            out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1", region="R1",
            snr_min_cfg=3.0, fwhm_min_cfg=25.0,
        )

        assert 'point_idx' in result
        assert 'summary_row' in result
        assert 'accepted_peaks' in result
        assert 'artifacts' in result
        assert 'warnings' in result
        assert 'count_accepted' in result
        assert result['point_idx'] == 0

    def test_summary_row_has_aicc(self, tmp_path):
        """Summary row includes AICc and peak count."""
        x, y = _make_synthetic_spectrum()
        point_data = {'point_idx': 5, 'y': y}

        result = _fit_point_minerals(
            point_data,
            x=x,
            fit_cfg=_minimal_fit_cfg(),
            fit_roi=(700.0, 1200.0),
            plot_roi=(700.0, 1300.0),
            out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1", region="R1",
            snr_min_cfg=3.0, fwhm_min_cfg=25.0,
        )

        sr = result['summary_row']
        assert sr is not None
        assert 'aicc' in sr
        assert 'num_peaks' in sr
        assert 'r2' in sr
        assert sr['point'] == 5

    def test_accepted_peaks_for_strong_signal(self, tmp_path):
        """Strong peak should produce at least one accepted peak."""
        x, y = _make_synthetic_spectrum(peak_amp=500.0, noise_std=3.0)
        point_data = {'point_idx': 0, 'y': y}

        result = _fit_point_minerals(
            point_data,
            x=x,
            fit_cfg=_minimal_fit_cfg(),
            fit_roi=(700.0, 1200.0),
            plot_roi=(700.0, 1300.0),
            out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1", region="R1",
            snr_min_cfg=3.0, fwhm_min_cfg=25.0,
        )

        assert result['count_accepted'] >= 1
        assert len(result['accepted_peaks']) >= 1
        # Check accepted peak structure
        peak = result['accepted_peaks'][0]
        assert 'center_cm1' in peak
        assert 'amplitude_a' in peak
        assert 'fwhm_cm1' in peak

    def test_handles_flat_spectrum(self, tmp_path):
        """Flat/noise-only spectrum should produce 0 accepted peaks."""
        rng = np.random.RandomState(42)
        x = np.linspace(700, 1200, 523)
        y = rng.normal(0, 5, 523)
        point_data = {'point_idx': 0, 'y': y}

        result = _fit_point_minerals(
            point_data,
            x=x,
            fit_cfg=_minimal_fit_cfg(),
            fit_roi=(700.0, 1200.0),
            plot_roi=(700.0, 1300.0),
            out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1", region="R1",
            snr_min_cfg=3.0, fwhm_min_cfg=25.0,
        )

        assert result['count_accepted'] == 0

    def test_picklable_for_multiprocessing(self, tmp_path):
        """Worker function is picklable and works via ProcessPoolExecutor."""
        x, y = _make_synthetic_spectrum()
        point_data = {'point_idx': 0, 'y': y}

        worker = partial(
            _fit_point_minerals,
            x=x,
            fit_cfg=_minimal_fit_cfg(),
            fit_roi=(700.0, 1200.0),
            plot_roi=(700.0, 1300.0),
            out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1", region="R1",
            snr_min_cfg=3.0, fwhm_min_cfg=25.0,
        )

        with ProcessPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(worker, [point_data]))

        assert len(results) == 1
        assert results[0]['point_idx'] == 0


class TestFitPointHydration:
    """Tests for _fit_point_hydration worker function."""

    def test_returns_expected_keys(self, tmp_path):
        """Worker returns dict with all required keys."""
        x = np.linspace(238.0, 4765.0, 523)
        y = np.random.RandomState(42).normal(0, 5, 523)
        oh_mask = (x >= 2800.0) & (x <= 3900.0)
        plot_mask = (x >= 2600.0) & (x <= 4000.0)
        point_data = {'point_idx': 10, 'y': y}

        fit_cfg_oh = _minimal_fit_cfg()
        fit_cfg_oh['max_peaks'] = 2
        fit_cfg_oh['fit_fwhm_min_initial_cm1'] = 50.0
        fit_cfg_oh['filter_fwhm_min_cm1'] = 50.0
        fit_cfg_oh['fwhm_max_cm1'] = 300.0
        fit_cfg_oh['parsimony'] = {'model_selection': 'ftest', 'ftest_alpha': 0.01}

        result = _fit_point_hydration(
            point_data,
            x=x, fit_cfg_oh=fit_cfg_oh, oh_roi=(2800.0, 3900.0),
            oh_plot=(2600.0, 4000.0), oh_mask=oh_mask, plot_mask_oh=plot_mask,
            n_edge=5, min_snr=3.0, r2_min=0.25, center_lo=3000.0,
            center_hi=3900.0, out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1",
        )

        assert 'point_idx' in result
        assert 'summary_row' in result
        assert 'accepted_peaks' in result
        assert 'artifacts' in result
        assert result['point_idx'] == 10

    def test_no_detection_for_noise_only(self, tmp_path):
        """Noise-only spectrum → no hydration detection."""
        x = np.linspace(238.0, 4765.0, 523)
        rng = np.random.RandomState(42)
        y = rng.normal(0, 5, 523)
        oh_mask = (x >= 2800.0) & (x <= 3900.0)
        plot_mask = (x >= 2600.0) & (x <= 4000.0)
        point_data = {'point_idx': 0, 'y': y}

        fit_cfg_oh = _minimal_fit_cfg()
        fit_cfg_oh['max_peaks'] = 2
        fit_cfg_oh['fit_fwhm_min_initial_cm1'] = 50.0
        fit_cfg_oh['filter_fwhm_min_cm1'] = 50.0
        fit_cfg_oh['fwhm_max_cm1'] = 300.0
        fit_cfg_oh['parsimony'] = {'model_selection': 'ftest', 'ftest_alpha': 0.01}

        result = _fit_point_hydration(
            point_data,
            x=x, fit_cfg_oh=fit_cfg_oh, oh_roi=(2800.0, 3900.0),
            oh_plot=(2600.0, 4000.0), oh_mask=oh_mask, plot_mask_oh=plot_mask,
            n_edge=5, min_snr=3.0, r2_min=0.25, center_lo=3000.0,
            center_hi=3900.0, out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1",
        )

        assert result['summary_row']['oh_detected'] is False
        assert len(result['accepted_peaks']) == 0

    def test_picklable_for_multiprocessing(self, tmp_path):
        """Worker function is picklable."""
        x = np.linspace(238.0, 4765.0, 523)
        y = np.random.RandomState(42).normal(0, 5, 523)
        oh_mask = (x >= 2800.0) & (x <= 3900.0)
        plot_mask = (x >= 2600.0) & (x <= 4000.0)
        point_data = {'point_idx': 0, 'y': y}

        fit_cfg_oh = _minimal_fit_cfg()
        fit_cfg_oh['max_peaks'] = 2
        fit_cfg_oh['fit_fwhm_min_initial_cm1'] = 50.0
        fit_cfg_oh['filter_fwhm_min_cm1'] = 50.0
        fit_cfg_oh['fwhm_max_cm1'] = 300.0
        fit_cfg_oh['parsimony'] = {'model_selection': 'ftest', 'ftest_alpha': 0.01}

        worker = partial(
            _fit_point_hydration,
            x=x, fit_cfg_oh=fit_cfg_oh, oh_roi=(2800.0, 3900.0),
            oh_plot=(2600.0, 4000.0), oh_mask=oh_mask, plot_mask_oh=plot_mask,
            n_edge=5, min_snr=3.0, r2_min=0.25, center_lo=3000.0,
            center_hi=3900.0, out_dir=str(tmp_path),
            sol="0921", target="Test", scan="detail_1",
        )

        with ProcessPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(worker, [point_data]))

        assert len(results) == 1


class TestFitPointOrganics:
    """Tests for _fit_point_organics worker function."""

    def test_returns_expected_keys(self, tmp_path):
        """Worker returns dict with all required keys."""
        x = np.linspace(238.0, 4765.0, 523)
        rng = np.random.RandomState(42)
        y = rng.normal(0, 5, 523)
        org_mask = (x >= 1250.0) & (x <= 1850.0)
        point_data = {'point_idx': 0, 'y': y}

        result = _fit_point_organics(
            point_data,
            x=x,
            fit_cfg_org={**_minimal_fit_cfg(), 'max_peaks': 2, 'fwhm_max_cm1': 200.0,
                         'fit_fwhm_min_initial_cm1': 40.0},
            g_roi=(1500.0, 1700.0), d_roi=(1250.0, 1500.0),
            org_roi=(1250.0, 1850.0), org_plot=(1250.0, 1850.0),
            org_mask=org_mask,
            g_acc_lo=40.0, g_acc_hi=100.0, d_acc_lo=100.0, d_acc_hi=200.0,
            persist_min_snr=3.0, organics_fwhm_mins={},
            use_norm_input=False, rebaseline_cfg={},
            out_dir=str(tmp_path),
            sol="0712", target="SAU008", scan="detail_1",
        )

        assert 'point_idx' in result
        assert 'summary_row' in result
        assert 'accepted_peaks' in result
        assert 'artifacts' in result
        assert 'warnings' in result
        assert result['point_idx'] == 0

    def test_summary_row_structure(self, tmp_path):
        """Summary row has expected fields."""
        x = np.linspace(238.0, 4765.0, 523)
        y = np.random.RandomState(42).normal(0, 5, 523)
        org_mask = (x >= 1250.0) & (x <= 1850.0)
        point_data = {'point_idx': 3, 'y': y}

        result = _fit_point_organics(
            point_data,
            x=x,
            fit_cfg_org={**_minimal_fit_cfg(), 'max_peaks': 2, 'fwhm_max_cm1': 200.0,
                         'fit_fwhm_min_initial_cm1': 40.0},
            g_roi=(1500.0, 1700.0), d_roi=(1250.0, 1500.0),
            org_roi=(1250.0, 1850.0), org_plot=(1250.0, 1850.0),
            org_mask=org_mask,
            g_acc_lo=40.0, g_acc_hi=100.0, d_acc_lo=100.0, d_acc_hi=200.0,
            persist_min_snr=3.0, organics_fwhm_mins={},
            use_norm_input=False, rebaseline_cfg={},
            out_dir=str(tmp_path),
            sol="0712", target="SAU008", scan="detail_1",
        )

        sr = result['summary_row']
        assert sr['point'] == 3
        assert 'g_detected' in sr
        assert 'd_detected' in sr
        assert 'g_r2' in sr

    def test_picklable_for_multiprocessing(self, tmp_path):
        """Worker function is picklable."""
        x = np.linspace(238.0, 4765.0, 523)
        y = np.random.RandomState(42).normal(0, 5, 523)
        org_mask = (x >= 1250.0) & (x <= 1850.0)
        point_data = {'point_idx': 0, 'y': y}

        worker = partial(
            _fit_point_organics,
            x=x,
            fit_cfg_org={**_minimal_fit_cfg(), 'max_peaks': 2, 'fwhm_max_cm1': 200.0,
                         'fit_fwhm_min_initial_cm1': 40.0},
            g_roi=(1500.0, 1700.0), d_roi=(1250.0, 1500.0),
            org_roi=(1250.0, 1850.0), org_plot=(1250.0, 1850.0),
            org_mask=org_mask,
            g_acc_lo=40.0, g_acc_hi=100.0, d_acc_lo=100.0, d_acc_hi=200.0,
            persist_min_snr=3.0, organics_fwhm_mins={},
            use_norm_input=False, rebaseline_cfg={},
            out_dir=str(tmp_path),
            sol="0712", target="SAU008", scan="detail_1",
        )

        with ProcessPoolExecutor(max_workers=1) as pool:
            results = list(pool.map(worker, [point_data]))

        assert len(results) == 1
