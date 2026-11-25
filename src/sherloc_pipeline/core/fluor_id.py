"""Fluorescence feature identification by wavelength position.

Analogous to mineral_id.py for Raman peaks. Maps fluorescence peak centers
to group labels. Also provides cross-modal
co-occurrence scoring between fluorescence groups and Raman mineral
assignments.

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

from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Cross-modal co-occurrence scoring
# ---------------------------------------------------------------------------

# sulf1_v1 is generally Ca-sulfate and sulf2_v1 Mg-sulfate, with overlap.
_SULFATE_LABELS = {"sulf1_v1", "sulf2_v1", "sulf_v3"}

# Raman labels treated as phosphate when scoring group 2
_PHOSPHATE_LABELS = {"phosphate"}

# Raman mineral labels that indicate perchlorate
_PERCHLORATE_LABELS = {"perchlorate"}

# Raman labels treated as silicate when scoring group 3
_SILICATE_LABELS = {"1050", "pyroxene", "silicate_hump"}


@dataclass
class CooccurrenceScore:
    """Cross-modal co-occurrence annotation for a fluorescence peak."""

    fluor_group: str
    raman_support: str  # "confirmed", "unsupported", "contradicted", "no_raman"
    phase_interpretation: str  # e.g. "Ce3+-bearing anhydrite"
    confidence_boost: float  # Stored as the peak's assignment_confidence.
    notes: List[str] = field(default_factory=list)


def score_cooccurrences(
    fluor_groups: List[str],
    raman_assignments: List[str],
) -> List[CooccurrenceScore]:
    """Score fluorescence-Raman co-occurrences for a single scan point.

    Takes the fluorescence group labels and Raman mineral assignments for
    the same scan point and returns per-fluorescence-peak co-occurrence
    annotations. This is a post-hoc annotation step — it does not modify
    the fitting results, only adds interpretive context.

    Co-occurrence rules:
    - A sulfate Raman label (sulf1_v1, sulf2_v1, sulf_v3) + group1a or group1b
      → "Ce3+-bearing anhydrite" (confirmed)
    - Phosphate Raman + Group 2 → "Ce3+-bearing phosphate" (confirmed)
    - No phosphate Raman + Group 2 → Ce3+ phosphate below Raman detection limit

    Args:
        fluor_groups: Group labels for each fluorescence peak at this point.
        raman_assignments: Mineral assignment labels for Raman peaks at this point.

    Returns:
        List of CooccurrenceScore, one per fluorescence peak.
    """
    raman_set = set(raman_assignments)
    has_sulfate = bool(raman_set & _SULFATE_LABELS)
    has_phosphate = bool(raman_set & _PHOSPHATE_LABELS)
    has_perchlorate = bool(raman_set & _PERCHLORATE_LABELS)
    has_any_raman = len(raman_assignments) > 0

    scores: List[CooccurrenceScore] = []

    for group in fluor_groups:
        if group in ("group1a", "group1b"):
            if has_sulfate:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="confirmed",
                    phase_interpretation="Ce3+-bearing anhydrite",
                    confidence_boost=1.3,
                ))
            elif has_any_raman:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="unsupported",
                    phase_interpretation="Group 1 without sulfate Raman",
                    confidence_boost=0.8,
                    notes=["No Ca-sulfate Raman detected at this point"],
                ))
            else:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="no_raman",
                    phase_interpretation="Group 1 (no Raman data)",
                    confidence_boost=1.0,
                ))

        elif group == "group2":
            if has_phosphate:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="confirmed",
                    phase_interpretation="Ce3+-bearing phosphate",
                    confidence_boost=1.3,
                ))
            elif has_perchlorate:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="contradicted",
                    phase_interpretation="Reassess: perchlorate Raman but Group 2 fluor suggests phosphate",
                    confidence_boost=0.7,
                    notes=["Perchlorate Raman + Group 2 fluor: likely phosphate, not perchlorate"],
                ))
            elif has_any_raman:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="unsupported",
                    phase_interpretation="Ce3+ phosphate below Raman detection limit",
                    confidence_boost=0.9,
                    notes=["No phosphate Raman detected; Ce3+ may be in trace phosphate"],
                ))
            else:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="no_raman",
                    phase_interpretation="Group 2 (no Raman data)",
                    confidence_boost=1.0,
                ))

        elif group == "group3":
            has_silicate = bool(raman_set & _SILICATE_LABELS)
            if has_silicate:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="confirmed",
                    phase_interpretation="Silicate defect luminescence",
                    confidence_boost=1.3,
                ))
            elif has_any_raman:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="unsupported",
                    phase_interpretation="Silicate defect luminescence",
                    confidence_boost=1.0,
                    notes=["No silicate Raman detected at this point"],
                ))
            else:
                scores.append(CooccurrenceScore(
                    fluor_group=group,
                    raman_support="no_raman",
                    phase_interpretation="Silicate defect luminescence",
                    confidence_boost=1.0,
                ))

        else:
            scores.append(CooccurrenceScore(
                fluor_group=group,
                raman_support="no_raman" if not has_any_raman else "unsupported",
                phase_interpretation="Unidentified fluorescence",
                confidence_boost=1.0,
            ))

    return scores
