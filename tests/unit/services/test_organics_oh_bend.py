"""O-H bend vs G band in the organics fit (core/oh_bend.py), on synthetic spectra.

Each case runs the real organics point worker (real ``fit_spectrum``, the
package config) on a noisy synthetic R1 spectrum: a water bend with no carbon,
carbon with no stretch, and both. The stretch reaches the worker as the
hydration fit's accepted components for the same point.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

import sherloc_pipeline
from sherloc_pipeline.core.oh_bend import OhBendRule, stretch_height
from sherloc_pipeline.services.fitting import (
    FittingService,
    _fit_point_organics,
    _fit_run_marker_path,
    _begin_fit_run,
    _write_fit_run_marker,
)

SOL, TARGET, SCAN = "0851", "Lake_Haiyaha", "detail_1"
NOISE = 20.0
# One broad water-like stretch component (center, amplitude, fwhm).
WATER_STRETCH = [(3400.0, 25000.0, 250.0)]


def _fitting_cfg():
    path = Path(sherloc_pipeline.__file__).with_name("config.yaml")
    return yaml.safe_load(path.read_text())["fitting"]


def _gauss(x, c, a, w):
    return a * np.exp(-4.0 * np.log(2.0) * (x - c) ** 2 / w ** 2)


def _spectrum(*bands, seed=0):
    x = np.arange(400.0, 4000.0, 2.0)
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, NOISE, x.size)
    for c, a, w in bands:
        y = y + _gauss(x, c, a, w)
    return x, y


def _run(tmp_path, x, y, stretch):
    cfg = _fitting_cfg()
    f_lo, f_hi = cfg["organics_fwhm_bounds"]
    fit_cfg_org = dict(cfg)
    fit_cfg_org["max_peaks"] = int(cfg["organics_max_peaks"])
    fit_cfg_org["fit_fwhm_min_initial_cm1"] = float(max(cfg.get("fit_fwhm_min_initial_cm1", 22.0), f_lo))
    fit_cfg_org["fwhm_max_cm1"] = float(f_hi)
    fit_cfg_org["min_snr"] = float(cfg["organics_min_snr"])
    org_roi = tuple(cfg["organics_fit_range"])
    out_dir = tmp_path / "organics_fit"
    out_dir.mkdir(exist_ok=True)
    with patch("sherloc_pipeline.visualization.fitting_plots.plot_fit_overlay"):
        result = _fit_point_organics(
            {"point_idx": 0, "y": y, "oh_stretch": stretch},
            x=x,
            fit_cfg_org=fit_cfg_org,
            g_roi=tuple(cfg["organics_g_roi"]),
            d_roi=tuple(cfg["organics_d_roi"]),
            org_roi=org_roi,
            org_plot=org_roi,
            org_mask=(x >= org_roi[0]) & (x <= org_roi[1]),
            g_acc_lo=40.0, g_acc_hi=100.0,
            d_acc_lo=100.0, d_acc_hi=200.0,
            persist_min_snr=float(cfg["min_snr"]),
            organics_fwhm_mins=dict(cfg["posthoc_filters"]["organics_fwhm"]),
            use_norm_input=False,
            rebaseline_cfg={},
            out_dir=str(out_dir),
            sol=SOL, target=TARGET, scan=SCAN,
            bend_rule=OhBendRule.from_fitting_config(cfg),
        )
    tables = sorted(out_dir.glob("*_peaks.csv"))
    assert len(tables) == 1
    return result["accepted_peaks"], pd.read_csv(tables[0])


def _g_window(rows):
    return [r for r in rows if 1500.0 <= r["center_cm1"] <= 1700.0]


def test_water_bend_without_carbon_is_labeled_h2o_bend(tmp_path):
    """A stretch with its own bend and no carbon: the window peak is the bend."""
    height = stretch_height(WATER_STRETCH)
    x, y = _spectrum((3400.0, 25000.0, 250.0), (1640.0, 0.036 * height, 74.0))
    rows, table = _run(tmp_path, x, y, WATER_STRETCH)

    window = _g_window(rows)
    assert window, "the bend should pass the G gate (it is a 1500-1700 peak)"
    assert {r["band"] for r in window} == {"H2O_bend"}
    assert all(r["bend_rule"] == "H2O_bend" for r in window)
    assert all(r["bend_predicted"] > 0 for r in window)
    # Persistence reads the per-point peak table: the label travels there too.
    in_window = table[(table.center_cm1 >= 1500) & (table.center_cm1 <= 1700)]
    assert set(in_window.assignment) == {"H2O_bend"}


def test_carbon_without_stretch_stays_g_band(tmp_path):
    """No accepted stretch: the rule does not run and the peak keeps its G label."""
    x, y = _spectrum((1595.0, 600.0, 60.0))
    rows, table = _run(tmp_path, x, y, [])

    window = _g_window(rows)
    assert window
    assert {r["band"] for r in window} == {"G"}
    assert all(r["bend_rule"] == "" for r in window)
    assert table.assignment.fillna("").eq("").all()


def test_carbon_with_water_keeps_g_and_reports_the_excess(tmp_path):
    """Carbon on a hydrated point: the G peak exceeds the predicted bend and stays G."""
    height = stretch_height(WATER_STRETCH)
    x, y = _spectrum(
        (3400.0, 25000.0, 250.0),
        (1640.0, 0.036 * height, 74.0),
        (1590.0, 1500.0, 60.0),
    )
    rows, table = _run(tmp_path, x, y, WATER_STRETCH)

    g_rows = [r for r in _g_window(rows) if r["band"] == "G"]
    assert g_rows, "the carbon band must still be labeled G"
    for r in g_rows:
        assert r["bend_rule"] == "G_band"
        assert r["amplitude_a"] > r["bend_predicted"]
        assert r["bend_excess_snr"] >= 3.0
    assert "G_band" in set(table.assignment.fillna(""))


def test_rule_needs_a_completed_hydration_run(tmp_path):
    """Without a completed hydration run for the scan the rule reports why it did not run."""
    cfg = _fitting_cfg()
    service = FittingService.__new__(FittingService)
    base = tmp_path
    h_dir = base / "hydration_fit"
    h_dir.mkdir()

    rule, stretches, status = service._load_oh_bend_inputs(cfg, base, SOL, TARGET, SCAN)
    assert rule is None and stretches == {} and "absent" in status

    marker = _fit_run_marker_path(h_dir, SOL, TARGET, SCAN, "R1", "hydration")
    _begin_fit_run(marker, domain="hydration")
    rule, _, status = service._load_oh_bend_inputs(cfg, base, SOL, TARGET, SCAN)
    assert rule is None and "running" in status

    _write_fit_run_marker(marker, domain="hydration", points_fitted=2, accepted_peaks=1)
    pd.DataFrame([{"point": 1, "center_cm1": 3400.0, "amplitude_a": 900.0, "fwhm_cm1": 240.0}]).to_csv(
        h_dir / f"{SOL}_{TARGET}_{SCAN}_R1_hydration_accepted_peaks.csv", index=False
    )
    rule, stretches, status = service._load_oh_bend_inputs(cfg, base, SOL, TARGET, SCAN)
    assert status == "applied" and rule is not None
    assert stretches == {1: [(3400.0, 900.0, 240.0)]}

    no_block = {k: v for k, v in cfg.items() if k != "organics_oh_bend"}
    rule, _, status = service._load_oh_bend_inputs(no_block, base, SOL, TARGET, SCAN)
    assert rule is None and "config" in status
