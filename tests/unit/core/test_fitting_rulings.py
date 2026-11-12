"""Mineral fitter rules of 2026-11-12: AICc gain threshold, merge-admissible candidates,
negative-spike channel masking, cosmic-ray review flag, configured caps and flags."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sherloc_pipeline.core.fitting import _merge_would_drop, fit_spectrum, save_peak_table

CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text()
)["fitting"]


def _gauss(x, c, a, f):
    return a * np.exp(-4 * np.log(2) * (x - c) ** 2 / f ** 2)


def _spectrum(peaks, spike=None, seed=7):
    x = np.linspace(500.0, 2200.0, 1800)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 10.0, x.size)
    for c, a, f in peaks:
        y += _gauss(x, c, a, f)
    if spike is not None:
        y[int(np.argmin(np.abs(x - spike[0])))] = spike[1]
    return x, y


def _cfg(**parsimony):
    cfg = {**CONFIG, "parsimony": {**CONFIG["parsimony"], **parsimony}}
    return cfg


def test_config_carries_the_rulings():
    assert CONFIG["parsimony"]["model_selection"] == "aicc"
    assert CONFIG["parsimony"]["aicc_improve_threshold"] == 0.0
    assert CONFIG["parsimony"]["reject_merged_candidates"] is False
    assert CONFIG["filter_fwhm_min_cm1"] == 25
    assert CONFIG["r_squared_min"] == 0.0
    assert CONFIG["posthoc_filters"]["r2_min"] == 0.0
    assert CONFIG["minerals_negative_spike_sigma"] == 10.0
    assert CONFIG["max_peaks"] == 5 and CONFIG["parsimony"]["aicc_max_peaks"] == 5


def test_aicc_improve_threshold_is_read_by_the_selection_loop():
    x, y = _spectrum([(970.0, 400.0, 30.0), (1080.0, 300.0, 30.0)])
    two, _ = fit_spectrum(x, y, _cfg(aicc_improve_threshold=0.0), roi=(700, 1200))
    one, _ = fit_spectrum(x, y, _cfg(aicc_improve_threshold=1e9), roi=(700, 1200))
    assert len(two.peaks) == 2
    assert len(one.peaks) == 1


def test_merge_would_drop_matches_the_merge_step():
    noise = 10.0
    close = np.array([1005.0, 400.0, 85.0, 1016.0, 2500.0, 23.0])
    apart = np.array([976.0, 400.0, 30.0, 1016.0, 2500.0, 23.0])
    weak = np.array([1005.0, 1.0, 85.0, 1016.0, 2500.0, 23.0])  # pruned before the merge
    assert _merge_would_drop(close, 3.0, 2.0, noise, 12.5)
    assert not _merge_would_drop(apart, 3.0, 2.0, noise, 12.5)
    assert not _merge_would_drop(weak, 3.0, 2.0, noise, 12.5)


def test_negative_spike_channels_get_zero_weight():
    x, y = _spectrum([(1015.0, 600.0, 26.0), (1130.0, 500.0, 44.0)], spike=(1190.0, -20000.0))
    plain, _ = fit_spectrum(x, y, _cfg(), roi=(700, 1200))
    masked, model = fit_spectrum(x, y, _cfg(), roi=(700, 1200), negative_spike_sigma=10.0)
    assert masked.warnings == ["negative_spike_channels_masked:1"]
    assert model.shape == y.shape
    assert masked.r2 > 0.9 > plain.r2
    assert any(abs(p.m_cm1 - 1130.0) < 10 for p in masked.peaks)
    # no channel below the bound: identical to the unmasked fit
    x2, y2 = _spectrum([(1015.0, 600.0, 26.0)])
    a, _ = fit_spectrum(x2, y2, _cfg(), roi=(700, 1200))
    b, _ = fit_spectrum(x2, y2, _cfg(), roi=(700, 1200), negative_spike_sigma=10.0)
    assert [p.m_cm1 for p in a.peaks] == [p.m_cm1 for p in b.peaks] and b.warnings == []


def test_cr_mask_is_a_review_flag_never_a_filter(tmp_path):
    x, y = _spectrum([(1015.0, 600.0, 30.0)])
    none, _ = fit_spectrum(x, y, _cfg(), roi=(700, 1200))
    hit, _ = fit_spectrum(x, y, _cfg(), roi=(700, 1200), cr_mask_cm1=[1012.0])
    miss, _ = fit_spectrum(x, y, _cfg(), roi=(700, 1200), cr_mask_cm1=[1150.0])
    assert [p.cr_mask_review for p in none.peaks] == [None]
    assert [p.cr_mask_review for p in hit.peaks] == [True]
    assert [p.cr_mask_review for p in miss.peaks] == [False]
    assert [p.m_cm1 for p in hit.peaks] == [p.m_cm1 for p in none.peaks]
    out = tmp_path / "peaks.csv"
    save_peak_table(hit.peaks, str(out))
    assert pd.read_csv(out).cr_mask_review.tolist() == [True]
