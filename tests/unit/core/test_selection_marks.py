"""Selection marks: the AICc size search keeps a log of every model size, the
F-test pick is derived from that log with no extra fit, and each output component carries its
step's AICc gain and F-test p-value. Records only: what the fit keeps does not change."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from sherloc_pipeline.core.fitting import SIZE_LOG_VERSION, _compute_aicc, fit_spectrum

ROOT = Path(__file__).resolve().parents[3]
CONFIG = yaml.safe_load((ROOT / "src/sherloc_pipeline/config.yaml").read_text())["fitting"]
GOLDEN = [ROOT / "tests/golden" / g / "preprocessed_spectra.npz" for g in ("sol_921_detail_1", "sol_921_detail_1_ml")]
ROI = tuple(CONFIG["r1_fit_range"])


def _cfg(**parsimony):
    return {**CONFIG, "parsimony": {**CONFIG["parsimony"], **parsimony}}


def _fit(x, y, cfg):
    return fit_spectrum(x, y, cfg, roi=ROI, negative_spike_sigma=cfg.get("minerals_negative_spike_sigma"))[0]


@pytest.fixture(scope="module")
def golden_fits():
    """Every golden point fitted once under each rule: (set, point, AICc-rule result, F-test-rule result)."""
    out = []
    for path in GOLDEN:
        d = np.load(path)
        x = d["raman_shift"].astype(float)
        for i in range(d["spectra"].shape[0]):
            y = d["spectra"][i].astype(float)
            out.append((path.parent.name, i, _fit(x, y, _cfg()), _fit(x, y, _cfg(model_selection="ftest"))))
    return out


def _triples(peaks):
    return [(p["center_cm1"], p["amplitude_a"], p["fwhm_cm1"]) if isinstance(p, dict) else (p.m_cm1, p.a, p.fwhm)
            for p in peaks]


def test_ftest_pick_from_the_log_equals_the_ftest_rule_on_every_golden_point(golden_fits):
    """Both rules fit size p from the same start, bounds and data, so the derived pick must equal
    the F-test rule's output exactly: the same parameters, not only the same curve. A degenerate
    fit (two slots trading places) is still the same deterministic call, so exact equality is
    asserted rather than a curve tolerance; a failure here would mean the two paths diverged."""
    n = no_seed = 0
    for name, i, res, ft in golden_fits:
        sel = res.selection
        n += 1
        if sel is None:  # no seed passed the seed SNR: no size search under either rule
            assert "no_peaks_detected" in res.warnings and ft.peaks == [], (name, i)
            no_seed += 1
            continue
        assert sel["version"] == SIZE_LOG_VERSION
        assert sel["ftest_size"] is not None, (name, i)
        assert _triples(sel["ftest_peaks"]) == _triples(ft.peaks), (name, i)
    assert n == 200 and no_seed < n


def test_the_aicc_pick_is_the_lowest_aicc_size_the_merge_rule_admits(golden_fits):
    for name, i, res, _ in golden_fits:
        sel = res.selection
        if sel is None:
            continue
        admissible = [r for r in sel["sizes"] if r["aicc_eligible"] and not r["failed"] and not r["merge_skip"]]
        if not admissible:
            assert sel["aicc_size"] is None and res.peaks == []
            continue
        best = min(admissible, key=lambda r: (r["aicc"], r["p"]))
        assert sel["aicc_size"] == best["p"], (name, i)


def test_marks_are_the_steps_at_each_components_seed_rank(golden_fits):
    seen = 0
    for name, i, res, _ in golden_fits:
        sel = res.selection
        if sel is None:
            continue
        by_p = {r["p"]: r for r in sel["sizes"]}
        for pk in res.peaks:
            k = pk.seed_rank
            assert k is not None and 1 <= k <= sel["aicc_size"]
            assert pk.aicc_gain == by_p[k]["step_aicc_gain"] and pk.step_p == by_p[k]["step_p"]
            prev = sel["aicc0"] if k == 1 else by_p[k - 1]["aicc"]
            assert pk.aicc_gain == pytest.approx(prev - by_p[k]["aicc"], abs=0)
            seen += 1
    assert seen > 0


def test_size_zero_is_the_null_model():
    x = np.linspace(700.0, 1200.0, 300)
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 5.0, x.size) + 400.0 * np.exp(-4 * np.log(2) * (x - 960.0) ** 2 / 30.0 ** 2)
    sel = fit_spectrum(x, y, _cfg(), roi=ROI)[0].selection
    assert sel["rss0"] == pytest.approx(float(np.sum(y ** 2)))
    assert sel["aicc0"] == pytest.approx(_compute_aicc(n_samples=sel["n"], rss=sel["rss0"], num_params=0))
    assert sel["sizes"][0]["p"] == 1 and sel["sizes"][0]["step_p"] < 1e-6


def test_no_size_search_means_no_log_and_no_marks():
    x = np.linspace(700.0, 1200.0, 300)
    y = 300.0 * np.exp(-4 * np.log(2) * (x - 960.0) ** 2 / 30.0 ** 2) + 1.0
    for res in (fit_spectrum(x, y, _cfg(model_selection="ftest"), roi=ROI)[0],
                fit_spectrum(x, y, _cfg(), roi=ROI, seed_centers=[960.0])[0]):
        assert res.selection is None
        assert res.peaks and all(p.aicc_gain is None and p.step_p is None and p.seed_rank is None for p in res.peaks)


def test_a_failed_size_is_logged_and_stops_the_ftest_pick(monkeypatch):
    """A size whose fit raises is skipped by AICc and ends the F-test run (its branch breaks)."""
    import sherloc_pipeline.core.fitting as F

    x = np.linspace(700.0, 1200.0, 300)
    g = lambda c, a: a * np.exp(-4 * np.log(2) * (x - c) ** 2 / 30.0 ** 2)
    y = g(800.0, 500.0) + g(950.0, 400.0) + g(1100.0, 300.0) + np.random.default_rng(1).normal(0, 2.0, x.size)
    real = F.least_squares

    def flaky(fun, p0, *a, **kw):
        if p0.size == 6:
            raise RuntimeError("planted failure at size 2")
        return real(fun, p0, *a, **kw)

    monkeypatch.setattr(F, "least_squares", flaky)
    res = fit_spectrum(x, y, _cfg(), roi=ROI)[0]
    rows = {r["p"]: r for r in res.selection["sizes"]}
    assert rows[2]["failed"] and rows[2]["aicc"] is None
    assert rows[3]["step_p"] is None and rows[3]["step_aicc_gain"] is None
    assert res.selection["ftest_size"] == 1
    ft = fit_spectrum(x, y, _cfg(model_selection="ftest"), roi=ROI)[0]
    assert _triples(res.selection["ftest_peaks"]) == _triples(ft.peaks)


@pytest.mark.parametrize("aicc_min", [2, 3])
def test_sizes_below_the_aicc_minimum_are_logged_but_never_picked(aicc_min):
    """aicc_min_peaks limits which sizes AICc may pick, not which the log records: the derived
    F-test pick still equals the F-test rule, every component still has its step, and the AICc
    pick is the one the minimum alone gives."""
    x = np.linspace(700.0, 1200.0, 300)
    g = lambda c, a: a * np.exp(-4 * np.log(2) * (x - c) ** 2 / 30.0 ** 2)
    y = g(820.0, 500.0) + g(1010.0, 350.0) + np.random.default_rng(5).normal(0, 3.0, x.size)
    cfg = _cfg(aicc_min_peaks=aicc_min)
    res = _fit(x, y, cfg)
    sel = res.selection
    rows = {r["p"]: r for r in sel["sizes"]}
    min_p = min(aicc_min, max(rows))  # as on main: a minimum above the largest size means the largest
    assert rows[1]["aicc_eligible"] is False and all(rows[p]["aicc_eligible"] == (p >= min_p) for p in rows)
    assert sel["aicc_size"] >= min_p
    assert _triples(sel["ftest_peaks"]) == _triples(_fit(x, y, _cfg(model_selection="ftest")).peaks)
    assert res.peaks and all(p.aicc_gain is not None and p.step_p is not None for p in res.peaks)
    eligible = [r for r in sel["sizes"] if r["aicc_eligible"] and not r["failed"] and not r["merge_skip"]]
    assert sel["aicc_size"] == min(eligible, key=lambda r: (r["aicc"], r["p"]))["p"]
