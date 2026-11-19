"""The stretch-to-bend rule (core/oh_bend.py)."""

import math
from pathlib import Path

import pytest
import yaml

import sherloc_pipeline
from sherloc_pipeline.core.oh_bend import (
    BEND_LABEL, G_LABEL, UNCERTAIN_LABEL, OhBendRule, load_hydration_stretches, stretch_height,
)


def _rule(**over):
    cfg = {
        "organics_oh_bend": {
            "bend_sigma": 2.0,
            "excess_sigma": 3.0,
            "stretch_classes": [
                {"name": "water_broad", "center_max": 3500.0, "fwhm_min": 150.0, "ratio": 0.04},
                {"name": "narrow_low", "center_max": 3500.0, "fwhm_max": 150.0, "ratio": 0.0},
                {"name": "high", "center_min": 3500.0, "ratio": 0.01},
            ],
            **over,
        }
    }
    return OhBendRule.from_fitting_config(cfg)


def test_package_config_builds_a_rule_that_covers_every_stretch():
    cfg = yaml.safe_load(Path(sherloc_pipeline.__file__).with_name("config.yaml").read_text())["fitting"]
    rule = OhBendRule.from_fitting_config(cfg)
    assert rule is not None
    for center in (3000.0, 3200.0, 3499.0, 3500.0, 3649.0, 3650.0, 3900.0):
        for fwhm in (50.0, 149.0, 150.0, 300.0):
            assert rule.classify_stretch([(center, 100.0, fwhm)]) is not None, (center, fwhm)


def test_stretch_class_follows_the_largest_component():
    rule = _rule()
    comps = [(3600.0, 50.0, 60.0), (3300.0, 400.0, 250.0)]
    assert rule.classify_stretch(comps).name == "water_broad"


def test_stretch_height_is_the_model_maximum():
    assert stretch_height([(3400.0, 1000.0, 200.0)]) == pytest.approx(1000.0, rel=1e-4)
    # Two overlapping components add.
    assert stretch_height([(3400.0, 1000.0, 200.0), (3420.0, 1000.0, 200.0)]) > 1900.0


@pytest.mark.parametrize(
    "amplitude, expected",
    [
        (40.0 + 2.0 * 10.0, BEND_LABEL),         # predicted 40, noise 10: within 2 sigma
        (40.0 + 2.5 * 10.0, UNCERTAIN_LABEL),    # 2.5 sigma above the prediction
        (40.0 + 3.0 * 10.0 + 1.0, G_LABEL),      # more than 3 sigma of excess
    ],
)
def test_three_outcomes(amplitude, expected):
    rule = _rule()
    stretch = [(3400.0, 1000.0, 250.0)]          # water_broad -> predicted 0.04 x 1000 = 40
    snr = amplitude / 10.0                        # noise 10
    v = rule.evaluate(amplitude, snr, stretch)
    assert v.label == expected
    assert v.bend_predicted == pytest.approx(40.0, rel=1e-4)
    assert v.stretch_class == "water_broad"


def test_no_stretch_no_verdict():
    assert _rule().evaluate(100.0, 10.0, []) is None


def test_structural_oh_predicts_no_bend():
    v = _rule().evaluate(35.0, 3.5, [(3300.0, 5000.0, 60.0)])   # narrow_low, ratio 0
    assert v.bend_predicted == 0.0
    assert v.label == G_LABEL


def test_bad_config_is_refused():
    with pytest.raises(ValueError):
        _rule(excess_sigma=1.0)
    with pytest.raises(ValueError):
        OhBendRule.from_fitting_config({"organics_oh_bend": {"bend_sigma": 2, "excess_sigma": 3, "stretch_classes": []}})
    assert OhBendRule.from_fitting_config({}) is None


def test_load_hydration_stretches_groups_by_point_and_skips_bad_rows():
    rows = [
        {"point": 3, "center_cm1": 3400.0, "amplitude_a": 10.0, "fwhm_cm1": 200.0},
        {"point": 3, "center_cm1": 3600.0, "amplitude_a": 5.0, "fwhm_cm1": 80.0},
        {"point": 4, "center_cm1": math.nan, "amplitude_a": 5.0, "fwhm_cm1": 80.0},
        {"point": 5},
    ]
    assert load_hydration_stretches(rows) == {3: [(3400.0, 10.0, 200.0), (3600.0, 5.0, 80.0)]}
