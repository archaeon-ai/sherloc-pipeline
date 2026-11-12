"""Build a minerals labeler manifest from two fit rules' components (local data only).

One row = one *disputed* component: persisted under one rule with no persisted component of the other
rule within ``MATCH_TOL_CM1`` at the same point (the comparison set's rule). The row shows that point's
mineral window, both fits, and the components each fit keeps.

Two sheets share this format:

- ``disputed``: rules ``aicc`` and ``ftest``; every disputed row of the input, in a seeded shuffle.
- ``gain2``: rules ``aicc`` and ``aicc_gain2`` (the AICc rule with a minimum gain of 2); only the
  components the gain threshold loses (kept under ``aicc``, unmatched under ``aicc_gain2``), sorted by
  SNR ascending (the SNR 3-4 rows first). "Kept" here has no point-R2 bar (ruling: r2 none), so no loss
  is hidden by a low R2. Input is the replay's component table in the same columns.

The input component table has one row per fitted component (persisted or not):
``scan, point, rule, center, amp, fwhm, snr, pass_sharpness, r2`` (extra columns ignored). Spectra are
read from ``<results>/<scan>/*_R1_normalized_baselined.csv`` (not the despiked file), columns
``raman_shift`` and one column per point index. Fit A / fit B is a seeded coin flip per row; the rule
behind each letter is in the manifest item, never served.

The manifest holds spectra: keep it on this machine, outside any repository or served directory.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

MATCH_TOL_CM1 = 10.0
PERSIST = {"snr_min": 3.0, "fwhm_min_cm1": 25.0, "r2_gt": 0.0, "pass_sharpness": True}
WINDOW_CM1 = (700.0, 1200.0)
# disputed reproduces the 2026-11-08 comparison set exactly (its persist rule had point R2 > 0).
SHEETS = {
    "disputed": {"rules": ("aicc", "ftest"), "order": "seeded shuffle", "r2_bar": True,
                 "directions": "both"},
    "gain2": {"rules": ("aicc", "aicc_gain2"), "order": "snr ascending", "r2_bar": False,
              "directions": "first rule only"},
}
INCOMPLETE_R2_GAP = 0.03
LABELS = {
    "1": ("present", "a mineral band is in the shaded region"),
    "2": ("absent", "no band there; the component fits noise or baseline"),
    "3": ("spike", "the feature is a spike or few-channel artifact, not a band"),
    "4": ("unsure", "cannot tell"),
}
DEFAULT_SEED = 20261112


def _truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_components(path):
    """Group a component table by (scan, point, rule)."""
    groups = {}
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            comp = {"center_cm1": float(row["center"]), "amplitude": float(row["amp"]),
                    "fwhm_cm1": float(row["fwhm"]), "snr": float(row["snr"]),
                    "pass_sharpness": _truthy(row["pass_sharpness"]), "r2": float(row["r2"])}
            groups.setdefault((row["scan"], int(row["point"]), row["rule"]), []).append(comp)
    return groups


def persisted(comp, r2_bar=True):
    """The persist rule: SNR >= 3, FWHM >= 25, pass_sharpness and, with r2_bar, point R2 > 0."""
    return (comp["snr"] >= PERSIST["snr_min"] and comp["fwhm_cm1"] >= PERSIST["fwhm_min_cm1"]
            and comp["pass_sharpness"] and (not r2_bar or comp["r2"] > PERSIST["r2_gt"]))


def disputed_rows(groups, rules, r2_bar=True, both_directions=True):
    """Components persisted under one rule with no persisted match under the other."""
    rule_x, rule_y = rules
    points = sorted({(scan, point) for scan, point, rule in groups if rule in rules})
    rows = []
    directions = ((rule_x, rule_y), (rule_y, rule_x)) if both_directions else ((rule_x, rule_y),)
    for scan, point in points:
        kept = {r: [c for c in groups.get((scan, point, r), []) if persisted(c, r2_bar)]
                for r in rules}
        for mine, other in directions:
            for comp in kept[mine]:
                if not any(abs(comp["center_cm1"] - o["center_cm1"]) <= MATCH_TOL_CM1
                           for o in kept[other]):
                    rows.append({"scan": scan, "point": point, "only_in": mine, "comp": comp})
    return rows


def read_window(results, scan, point, cache):
    """The point's normalized, baselined R1 spectrum cut to the mineral window."""
    if scan not in cache:
        files = [p for p in sorted((Path(results) / scan).glob("*_R1_normalized_baselined.csv"))
                 if "despiked" not in p.name]
        if len(files) != 1:
            raise ValueError(f"expected one R1 normalized baselined file for {scan}, found {len(files)}")
        with open(files[0], newline="", encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        cache[scan] = table
    x, y = [], []
    for row in cache[scan]:
        shift = float(row["raman_shift"])
        value = float(row[str(point)])
        if WINDOW_CM1[0] <= shift <= WINDOW_CM1[1] and math.isfinite(value):
            x.append(shift)
            y.append(value)
    return {"x": x, "y": y}


def read_mask_flags(path):
    """Optional review flags: scan, point, center_cm1, mask_px_in_fwhm."""
    flags = {}
    if path is None:
        return flags
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["scan"], int(row["point"]), round(float(row["center_cm1"]), 1))
            flags[key] = int(row["mask_px_in_fwhm"])
    return flags


def _fit(groups, scan, point, rule, r2_bar):
    return [{"center_cm1": c["center_cm1"], "amplitude": c["amplitude"], "fwhm_cm1": c["fwhm_cm1"],
             "snr": c["snr"], "kept": persisted(c, r2_bar)}
            for c in sorted(groups.get((scan, point, rule), []), key=lambda c: c["center_cm1"])]


def drawn_r2(window, fit):
    """R2 of the drawn line (sum of the output components) against the window spectrum."""
    x, y = window["x"], window["y"]
    model = [sum(c["amplitude"] * math.exp(-4 * math.log(2) * (v - c["center_cm1"]) ** 2
                                           / c["fwhm_cm1"] ** 2) for c in fit) for v in x]
    mean = sum(y) / len(y)
    total = sum((v - mean) ** 2 for v in y)
    return 1.0 - sum((a - b) ** 2 for a, b in zip(y, model)) / total if total else 0.0


def first_r2(groups, scan, point, rule):
    comps = groups.get((scan, point, rule), [])
    return comps[0]["r2"] if comps else None


def build_manifest(components, results, sheet="disputed", seed=DEFAULT_SEED, mask_flags=None):
    """Return the manifest document for one sheet."""
    if sheet not in SHEETS:
        raise ValueError(f"unknown sheet {sheet!r}")
    rules, r2_bar = SHEETS[sheet]["rules"], SHEETS[sheet]["r2_bar"]
    groups = read_components(components)
    rows = disputed_rows(groups, rules, r2_bar, SHEETS[sheet]["directions"] == "both")
    flags = read_mask_flags(mask_flags)
    rng = random.Random(seed)
    if sheet == "disputed":
        rng.shuffle(rows)
    else:
        rows.sort(key=lambda r: (r["comp"]["snr"], r["scan"], r["point"], r["comp"]["center_cm1"]))
    cache, items = {}, []
    for rank, row in enumerate(rows):
        scan, point, comp = row["scan"], row["point"], row["comp"]
        letters = dict(zip(("A", "B"), rules if rng.random() < 0.5 else rules[::-1]))
        mask = flags.get((scan, point, round(comp["center_cm1"], 1)), 0)
        window = read_window(results, scan, point, cache)
        fits = {letter: _fit(groups, scan, point, rule, r2_bar) for letter, rule in letters.items()}
        # A fit whose stored point R2 is well above the drawn line's R2 lost components to the
        # fitter's merge step; the UI says so for that fit.
        incomplete = [letter for letter, rule in letters.items() if fits[letter]
                      and first_r2(groups, scan, point, rule) is not None
                      and first_r2(groups, scan, point, rule) - drawn_r2(window, fits[letter])
                      > INCOMPLETE_R2_GAP]
        items.append({
            "unit_id": f"{sheet}:{scan}:p{point}:{row['only_in']}:{comp['center_cm1']:.2f}",
            "rank": rank, "scan": scan, "point": point, "only_in": row["only_in"],
            "fit_rules": letters, "snr": comp["snr"],
            "band": {"center_cm1": comp["center_cm1"], "fwhm_cm1": comp["fwhm_cm1"]},
            "window": window, "fits": fits,
            "flags": {"mask_px_in_band": mask, "incomplete_fits": incomplete},
        })
    header = {
        "sheet": sheet, "rules": list(rules), "order": SHEETS[sheet]["order"], "seed": seed,
        "n_items": len(items),
        "components_sha256": hashlib.sha256(Path(components).read_bytes()).hexdigest(),
        "persist_rule": PERSIST | {"r2_bar_applied": r2_bar},
        "directions": SHEETS[sheet]["directions"], "incomplete_r2_gap": INCOMPLETE_R2_GAP,
        "match_tol_cm1": MATCH_TOL_CM1, "window_cm1": list(WINDOW_CM1),
        "labels": {key: {"label": name, "meaning": meaning} for key, (name, meaning) in LABELS.items()},
        "public_check": "LOCAL ONLY: SHERLOC spectra from the local results tree; never commit, "
                        "upload or serve beyond loopback",
    }
    return {"header": header, "items": items}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--components", required=True, help="component table CSV")
    parser.add_argument("--results", required=True, help="results tree holding the R1 spectra")
    parser.add_argument("--out", required=True, help="manifest JSON to write (local only)")
    parser.add_argument("--sheet", choices=sorted(SHEETS), default="disputed")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mask-flags", help="optional CSV: scan, point, center_cm1, mask_px_in_fwhm")
    args = parser.parse_args(argv)
    document = build_manifest(args.components, args.results, args.sheet, args.seed, args.mask_flags)
    Path(args.out).write_text(json.dumps(document, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.sheet}: {document['header']['n_items']} rows -> {args.out}")


if __name__ == "__main__":
    main()
