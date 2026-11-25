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


def test_default_class_windows_match_fitter():
    frontend = Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/web/frontend/src"
    rows = json.loads((frontend / "lib/classWindows.json").read_text())
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
