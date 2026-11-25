"""Fluorescence feature identification by wavelength position.

Analogous to mineral_id.py for Raman peaks. Maps fluorescence peak centers
to group labels.

Group attributions: group 1a and 1b, the 304/325 nm doublet, Ce3+ in
anhydrite (Haney et al. 2026, JGR Planets, doi:10.1029/2025JE009385);
group 2, Ce3+ in phosphate, and group 3, silicate defect luminescence
(Scheller et al. 2024, Science Advances, doi:10.1126/sciadv.adm8241).
The group numbers follow Sharma et al. 2023, Nature,
doi:10.1038/s41586-023-06143-z.
This pipeline splits group 1 into 1a and 1b.

See docs/specs/FLUORESCENCE_FITTING_SPEC.md §4.1-4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class FluorescenceRule:
    """Wavelength range rule for fluorescence group assignment."""

    label: str
    lo: float  # inclusive lower bound (nm)
    hi: float  # inclusive upper bound (nm)


# Default rules — matches spec §4.1 and config.yaml fluorescence_rules
FLUORESCENCE_RULES: List[FluorescenceRule] = [
    FluorescenceRule("group3", 270.0, 295.0),  # Silicate defect
    FluorescenceRule("group1a", 300.0, 307.0),  # Ce3+ in anhydrite doublet (short-lambda)
    FluorescenceRule("group1b", 322.0, 329.0),  # Ce3+ in anhydrite doublet (long-lambda)
    FluorescenceRule("group2", 329.0, 355.0),  # Ce3+ in phosphate
]


def assign_fluor_group(center_nm: float) -> str:
    """Map fluorescence peak center to group label.

    Rules (inclusive bounds):
        group1a: 300-307 nm  (Ce3+ in anhydrite doublet, short-lambda)
        group1b: 322-329 nm  (Ce3+ in anhydrite doublet, long-lambda)
        group2:  329-355 nm  (Ce3+ in phosphate)
        group3:  270-295 nm  (Silicate defect)

    Returns 'unidentified' if the center falls outside all ranges.
    """
    for rule in FLUORESCENCE_RULES:
        if rule.lo <= center_nm <= rule.hi:
            return rule.label
    return "unidentified"


def classify_fluor_peaks(
    groups: List[str],
    peaks: List[Any],
) -> List[str]:
    """Apply post-classification rules to fluorescence group assignments.

    Currently implements one rule: orphan group1b reclassification.
    If a peak is labelled group1b but no group1a peak exists in the same
    spectrum **and** the peak center is above 328 nm, the group1b label is
    replaced with group2. Peaks at or below 328 nm are retained as group1b
    even without a group1a companion — 325 nm is solidly within the Ce3+
    anhydrite doublet range, while 328+ nm is ambiguous with group2.

    Args:
        groups: List of group labels (one per peak), modified in-place.
        peaks: List of peak objects with center_nm attribute.

    Returns:
        The modified groups list.
    """
    has_group1a = any(g == "group1a" for g in groups)
    if not has_group1a:
        for i, g in enumerate(groups):
            if g == "group1b" and peaks[i].center_nm > 328.0:
                groups[i] = "group2"
    return groups
