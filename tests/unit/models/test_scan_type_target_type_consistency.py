"""Cross-consistency between the orthogonal scan_type (geometry) and target_type
(purpose) axes — WS-1 §4.2 calibration-vocabulary widening.

The widened calibration vocabulary in :func:`classify_scan_type` overlaps the
cal/engineering scan-name rules in :func:`classify_target_type`. These tests
lock the relationship proven value-blind on the full mission corpus and guard
the two vocabularies against drifting apart:

* a scan the *geometry* axis calls ``calibration`` is never, at its real
  cal/engineering target, called ``mars_target`` by the *purpose* axis — the
  two axes never contradict each other on "is this science";
* a real science *geometry* (survey/detail/line/HDR) at a cal/engineering
  target keeps its geometry (the axes are orthogonal, not redundant).

Value-blind: every vector is a scan name / target name, never a science value.
"""

import pytest

from sherloc_pipeline.models.spectra import (
    ScanType,
    classify_scan_type,
    classify_target_type,
)


def _st(name):
    r = classify_scan_type(name, None, 100)
    return r.value if isinstance(r, ScanType) else r


# (scan_name, its real cal/engineering target) — one per widened family.
CAL_ENG_AT_THEIR_TARGETS = [
    ("AlGaN_1", "AlGaN340 calibration"),
    ("SRLC1_AlGaN_2", "external calibration"),
    ("15ppp_1", "conjunction"),
    ("500ppp_1_laser_disabled", "b conjunction"),
    ("power_on", "Amherst Point"),          # engineering scan_name on a Mars target
    ("power_off", "arm stowed"),
    ("maze_2", "maze calibration"),
    ("meteorite", "ext cal meteorite"),
    ("Teflon_1", "Teflon calibration"),
    ("polycarbonate", "external calibration"),
    ("passive_diffusil_5ppp_extended", "passive diffusil"),
    ("laser_disabled_detail", ""),          # blank target -> engineering
]


@pytest.mark.parametrize("name,target", CAL_ENG_AT_THEIR_TARGETS)
def test_calibration_scan_type_never_contradicts_mars_target(name, target):
    # Each vector is a calibration/engineering family member: assert the geometry
    # axis calls it calibration FIRST (so drift away from the widened vocabulary
    # fails loudly rather than silently skipping the cross-axis check — Codex
    # PR #13 F1), then assert the purpose axis agrees it is not a Mars target.
    assert _st(name) == "calibration"
    assert classify_target_type(target, name) in ("cal_target", "engineering")


# (scan_name, cal/eng target, expected geometry) — orthogonality cases.
ORTHOGONAL = [
    ("HDR_500", "Calibration", "HDR"),       # HDR geometry of the cal target
    ("detail_1", "arm stowed", "detail"),    # detail geometry, engineering purpose
    ("survey_1296", None, "survey"),         # survey geometry, no target -> engineering
]


@pytest.mark.parametrize("name,target,expected_st", ORTHOGONAL)
def test_real_geometry_preserved_at_cal_eng_targets(name, target, expected_st):
    assert _st(name) == expected_st
    assert classify_target_type(target, name) in ("cal_target", "engineering")


def test_science_scan_agrees_on_both_axes():
    assert _st("detail_1") == "detail"
    assert classify_target_type("Amherst Point", "detail_1") == "mars_target"
