"""Default map classification windows match the fitter's assignments."""

import json
import math
from pathlib import Path

from sherloc_pipeline.core.fluor_id import FLUORESCENCE_RULES, assign_fluor_group
from sherloc_pipeline.core.mineral_id import (
    assign_min_id,
    classify_hydration_band,
    classify_organic_band,
    load_mineral_rules,
)

LIB = Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/web/frontend/src/lib"


def test_default_class_windows_match_fitter():
    rows = json.loads((LIB / "classWindows.json").read_text())
    windows = {(r["domain"], r["class_id"]): (r["lo"], r["hi"]) for r in rows}
    assert len(windows) == len(rows)

    minerals = load_mineral_rules()
    for rule in minerals:
        lo, hi = windows.pop(("minerals", rule.label))
        assert (lo, hi) == (rule.lo, rule.hi)
        assert assign_min_id((lo + hi) / 2, minerals) == rule.label

    for rule in FLUORESCENCE_RULES:
        lo, hi = windows.pop(("fluorescence", rule.label))
        assert (lo, hi) == (rule.lo, rule.hi)
        assert assign_fluor_group((lo + hi) / 2) == rule.label

    # These classifiers define their intervals in code. Check both inclusive
    # endpoints and the nearest representable value outside each endpoint.
    for domain, label, classify in [
        ("organics", "D_band", classify_organic_band),
        ("organics", "G_band", classify_organic_band),
        ("hydration", "OH_stretch", classify_hydration_band),
    ]:
        lo, hi = windows.pop((domain, label))
        assert lo < hi
        assert classify(lo) == classify((lo + hi) / 2) == classify(hi) == label
        assert classify(math.nextafter(lo, -math.inf)) != label
        assert classify(math.nextafter(hi, math.inf)) != label

    assert not windows


def test_class_window_edges_match_fitter():
    windows = json.loads((LIB / "classWindows.json").read_text())
    cases = json.loads((LIB / "classWindowEdges.json").read_text())
    endpoints = {(r["domain"], r[bound]) for r in windows for bound in ("lo", "hi")}
    assert [(c["domain"], c["value"]) for c in cases] == sorted(endpoints)

    minerals = load_mineral_rules()
    classify = {
        "minerals": lambda x: assign_min_id(x, minerals),
        "fluorescence": assign_fluor_group,
        "organics": classify_organic_band,
        "hydration": classify_hydration_band,
    }
    for case in cases:
        assert case["expected"] == classify[case["domain"]](case["value"]), case
