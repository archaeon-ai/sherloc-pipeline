"""Hydration fits of synthetic Gaussian bands on tilted backgrounds."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f

from sherloc_pipeline.core.fitting import gaussian
from sherloc_pipeline.services.fitting import size_sidecar_path
from tests.unit.services.test_hydration_rerun_artifacts import SOL, TARGET, SCAN, _run_fit, _service


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    root = tmp_path_factory.mktemp("hydration")
    base = root / "results" / SCAN
    base.mkdir(parents=True)
    x = np.linspace(2800, 3900, 137)
    line = 20 + 0.02 * (x - 2800)
    frame = pd.DataFrame({
        "raman_shift": x,
        "0": line + gaussian(x, 3800, 25, 299.5),
        "1": line + gaussian(x, 3350, 12, 120) + gaussian(x, 3700, 1, 55)
             + np.random.default_rng(27).normal(size=x.size),
        "2": line + gaussian(x, 3250, 25, 298.5),
    })
    stem = f"{SOL}_{TARGET}_{SCAN}_R1"
    frame.to_csv(base / f"{stem}_normalized.csv", index=False)
    _run_fit(_service(root), base)
    return [base / "hydration_fit" / f"{stem}_point{point}_hydration_peaks.csv"
            for point in range(3)]


def test_joint_line_recovers_band_overlapping_the_window_end(scan):
    row, = pd.read_csv(scan[0]).itertuples()
    assert row.center_cm1 == pytest.approx(3800, abs=1e-5)
    assert row.amplitude_a == pytest.approx(25, abs=1e-5)
    assert row.fwhm_cm1 == pytest.approx(299.5, abs=1e-5)
    assert row.sharpness_ratio == pytest.approx(1, abs=.001)


def test_aicc_selects_second_component_when_ftest_stops_at_one(scan):
    assert len(pd.read_csv(scan[1])) == 2
    log = json.loads(size_sidecar_path(scan[1]).read_text())
    assert (log["aicc_size"], log["ftest_size"]) == (2, 1)
    n = log["n"]
    previous_rss, previous_aicc = log["rss0"], log["aicc0"]
    assert previous_aicc == pytest.approx(n * np.log(previous_rss / n) + 4 + 12 / (n - 3))
    for size in log["sizes"]:
        k = 3 * size["p"] + 2
        rss = size["rss"]
        score = n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        assert size["aicc"] == pytest.approx(score)
        assert size["step_aicc_gain"] == pytest.approx(previous_aicc - score)
        assert size["step_p"] == pytest.approx(f.sf((previous_rss - rss) / 3 / (rss / (n - k)), 3, n - k))
        previous_rss, previous_aicc = rss, score


def test_size_log_and_step_marks_written_for_hydration(scan):
    table = pd.read_csv(scan[0])
    assert table.aicc_gain.notna().all() and table.step_p.notna().all()
    log = json.loads(size_sidecar_path(scan[0]).read_text())
    step = log["sizes"][0]
    assert log["ftest_size"] == log["aicc_size"] == 1
    assert step["step_aicc_gain"] == pytest.approx(table.aicc_gain[0])
    assert step["step_p"] == pytest.approx(table.step_p[0])


def test_width_mark_within_one_cm1_of_cap(scan):
    assert pd.read_csv(scan[0]).fwhm_at_cap.tolist() == [True]
    assert pd.read_csv(scan[2]).fwhm_at_cap.tolist() == [False]
    for point, at_cap in [(0, True), (2, False)]:
        log = json.loads(size_sidecar_path(scan[point]).read_text())
        assert [row["fwhm_at_cap"] for row in log["ftest_peaks"]] == [at_cap]


def test_map_recovers_hydration_band_on_a_tilted_line():
    from sherloc_pipeline.services.map_fitting import _fit_raman_domain

    x = np.linspace(1950, 3900, 300)
    y = 20 + .02 * (x - 2800) + gaussian(x, 3800, 25, 299.5)
    result = _fit_raman_domain(x, y, {}, "hydration", preprocessed=True)
    peak, = result.peaks
    assert peak["center_cm1"] == 3800
    assert peak["fwhm_cm1"] == 299.5
