"""Fluorescence feature identification by wavelength position.

Analogous to mineral_id.py for Raman peaks. Maps fluorescence peak centers
to group labels and detects Ce3+ doublets. Also provides cross-modal
co-occurrence scoring between fluorescence groups and Raman mineral
assignments.

See docs/specs/FLUORESCENCE_FITTING_SPEC.md §4.1-4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    FluorescenceRule("group2", 329.0, 355.0),  # Ce3+ in phosphate; predicted ~340/~360nm doublet, single asymmetric band
]


def assign_fluor_group(center_nm: float) -> str:
    """Map fluorescence peak center to group label.

    Rules (inclusive bounds):
        group1a: 300-307 nm  (Ce3+ in anhydrite doublet, short-lambda)
        group1b: 322-329 nm  (Ce3+ in anhydrite doublet, long-lambda)
        group2:  329-355 nm  (Ce3+ in phosphate; predicted ~340/~360nm doublet
                              observed as single asymmetric band, ~360nm unresolved)
        group3:  270-295 nm  (Silicate defect)

    Returns 'unidentified' if the center falls outside all ranges.
    """
    for rule in FLUORESCENCE_RULES:
        if rule.lo <= center_nm <= rule.hi:
            return rule.label
    return "unidentified"


@dataclass
class DoubletRecord:
    """A detected Ce3+ doublet pair."""

    peak_1a_idx: int  # index into the input peaks list
    peak_1b_idx: int
    center_1a_nm: float
    center_1b_nm: float
    amplitude_1a: float
    amplitude_1b: float
    fwhm_1a_nm: float
    fwhm_1b_nm: float
    snr_1a: float
    snr_1b: float
    separation_nm: float
    intensity_ratio: float  # amplitude_1a / amplitude_1b


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


def detect_doublets(
    peaks: List[Any],
    doublet_snr_threshold: float = 5.0,
    separation_range: Tuple[float, float] = (18.0, 29.0),
    doublet_ratio_range: Optional[Tuple[float, float]] = None,
) -> List[DoubletRecord]:
    """Identify Ce3+ doublets from co-located group1a + group1b peaks.

    Uses nearest-separation greedy pairing per spec §4.2:
    1. Compute all pairwise separations between group1a and group1b peaks
    2. Filter to pairs within separation_range (default 18-29 nm)
    3. Greedily assign pairs by smallest separation first (each peak used at most once)
    4. Filter by intensity ratio if doublet_ratio_range is set
    5. Tie-break: if two candidate pairs have identical separation, select the pair
       with higher combined SNR

    Rejected doublets' peaks are released — their group1b peaks will be
    reclassified to group2 by ``classify_fluor_peaks()`` (orphan reclassification).

    Args:
        peaks: List of peak objects with center_nm, amplitude, fwhm_nm, snr attributes.
        doublet_snr_threshold: Both peaks must exceed this SNR (default 5.0).
        separation_range: (min, max) nm for valid doublet separation.
        doublet_ratio_range: Optional (lo, hi) for intensity_ratio (amp_1a / amp_1b).
            Doublets outside this range are rejected.

    Returns:
        List of DoubletRecord objects for each identified doublet.
    """
    sep_lo, sep_hi = separation_range

    # Collect group1a and group1b peaks that meet SNR threshold
    group1a: List[Tuple[int, Any]] = []
    group1b: List[Tuple[int, Any]] = []

    for i, peak in enumerate(peaks):
        if peak.snr < doublet_snr_threshold:
            continue
        group = assign_fluor_group(peak.center_nm)
        if group == "group1a":
            group1a.append((i, peak))
        elif group == "group1b":
            group1b.append((i, peak))

    if not group1a or not group1b:
        return []

    # Compute all valid candidate pairs
    candidates: List[Tuple[float, float, int, Any, int, Any]] = []
    for idx_a, peak_a in group1a:
        for idx_b, peak_b in group1b:
            sep = abs(peak_b.center_nm - peak_a.center_nm)
            if sep_lo <= sep <= sep_hi:
                combined_snr = peak_a.snr + peak_b.snr
                candidates.append((sep, -combined_snr, idx_a, peak_a, idx_b, peak_b))

    # Sort by separation ascending, then by combined SNR descending (via negation)
    candidates.sort(key=lambda c: (c[0], c[1]))

    # Greedy pairing: each peak used at most once
    used_a: set = set()
    used_b: set = set()
    doublets: List[DoubletRecord] = []

    for sep, _neg_snr, idx_a, peak_a, idx_b, peak_b in candidates:
        if idx_a in used_a or idx_b in used_b:
            continue
        used_a.add(idx_a)
        used_b.add(idx_b)

        ratio = peak_a.amplitude / peak_b.amplitude if peak_b.amplitude > 0 else float("inf")

        doublets.append(
            DoubletRecord(
                peak_1a_idx=idx_a,
                peak_1b_idx=idx_b,
                center_1a_nm=peak_a.center_nm,
                center_1b_nm=peak_b.center_nm,
                amplitude_1a=peak_a.amplitude,
                amplitude_1b=peak_b.amplitude,
                fwhm_1a_nm=peak_a.fwhm_nm,
                fwhm_1b_nm=peak_b.fwhm_nm,
                snr_1a=peak_a.snr,
                snr_1b=peak_b.snr,
                separation_nm=sep,
                intensity_ratio=ratio,
            )
        )

    # Filter by intensity ratio if range specified
    if doublet_ratio_range is not None:
        ratio_lo, ratio_hi = doublet_ratio_range
        doublets = [
            d for d in doublets
            if ratio_lo <= d.intensity_ratio <= ratio_hi
        ]

    return doublets


# ---------------------------------------------------------------------------
# Cross-modal co-occurrence scoring
# ---------------------------------------------------------------------------

# Raman mineral labels that indicate Ca-sulfate (anhydrite/gypsum)
_SULFATE_LABELS = {"sulf1_v1", "sulf2_v1", "sulf_v3"}

# Raman mineral labels that indicate phosphate (apatite/merrillite)
_PHOSPHATE_LABELS = {"phosphate"}

# Raman mineral labels that indicate perchlorate
_PERCHLORATE_LABELS = {"perchlorate"}

# Raman mineral labels that indicate silicate minerals (feldspar, pyroxene, etc.)
_SILICATE_LABELS = {"1050", "pyroxene", "silicate_hump"}


@dataclass
class CooccurrenceScore:
    """Cross-modal co-occurrence annotation for a fluorescence peak."""

    fluor_group: str
    raman_support: str  # "confirmed", "unsupported", "contradicted", "no_raman"
    phase_interpretation: str  # e.g. "Ce3+-bearing anhydrite"
    confidence_boost: float  # multiplier on assignment_confidence (1.0 = no change)
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

    Co-occurrence rules (from plan):
    - Ca-sulfate Raman + Group 1 doublet → "Ce3+-bearing anhydrite" (confirmed)
    - Phosphate Raman + Group 2 → "Ce3+-bearing phosphate" (confirmed)
    - Phosphate Raman + no Group 2 → possibly perchlorate or Ce3+-free (unsupported)
    - No phosphate Raman + Group 2 → Ce3+ phosphate below Raman detection limit
    - Perchlorate Raman + Group 2 → reassess: more likely phosphate than perchlorate

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
