"""Classification service — re-assign peak labels with custom profiles.

The 'fast path' for classification: re-assign existing peaks to different
classes using modified spectral range boundaries from a custom profile.
No re-fitting required.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClassOverride:
    """A single class override in a classification profile."""

    domain: str  # "minerals" | "organics" | "hydration" | "fluorescence"
    class_id: str  # e.g., "hi_carb"
    label: str | None = None  # display name override
    center: float | None = None  # spectral center override (cm-1 or nm)
    range: float | None = None  # half-width override
    color: str | None = None  # hex color override
    snr_range: tuple[float, float] | None = None  # display range override
    disabled: bool = False  # exclude from fitting/display


@dataclass
class ClassificationProfile:
    """A named peak classification profile."""

    id: str  # UUID or "default"
    name: str
    base: str = "default"
    overrides: list[ClassOverride] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "name": self.name,
            "base": self.base,
            "overrides": [
                {k: v for k, v in o.__dict__.items() if v is not None and v is not False}
                for o in self.overrides
            ],
        })

    @classmethod
    def from_json(cls, data: str) -> "ClassificationProfile":
        d = json.loads(data)
        overrides = [ClassOverride(**o) for o in d.get("overrides", [])]
        return cls(
            id=d["id"],
            name=d["name"],
            base=d.get("base", "default"),
            overrides=overrides,
        )


def get_default_classification_rules() -> dict[str, list[dict]]:
    """Return default classification rules from config.yaml mineral rules."""
    # Read these from the existing config structure.
    # The mineral rules define: class_id, center_cm1, half_width, domain.
    # For E1, return hardcoded defaults matching config.yaml.
    return {
        "minerals": [
            {"class_id": "olivine", "label": "Olivine", "center": 840, "range": 30, "domain": "minerals"},
            {"class_id": "sulfate_v1", "label": "Sulfate ν1", "center": 1010, "range": 30, "domain": "minerals"},
            {"class_id": "hi_carb", "label": "Carbonate", "center": 1088, "range": 20, "domain": "minerals"},
            {"class_id": "silicate", "label": "Silicate", "center": 1010, "range": 50, "domain": "minerals"},
        ],
        "organics": [
            {"class_id": "D_band", "label": "D Band", "center": 1350, "range": 50, "domain": "organics"},
            {"class_id": "G_band", "label": "G Band", "center": 1580, "range": 50, "domain": "organics"},
        ],
        "hydration": [
            {"class_id": "OH_stretch", "label": "OH Stretch", "center": 3400, "range": 300, "domain": "hydration"},
        ],
        "fluorescence": [
            {"class_id": "group1a", "label": "Ce³⁺ 1a", "center": 304, "range": 12, "domain": "fluorescence"},
            {"class_id": "group1b", "label": "Ce³⁺ 1b", "center": 325, "range": 12, "domain": "fluorescence"},
            {"class_id": "group2", "label": "Ce³⁺ Phosphate", "center": 340, "range": 15, "domain": "fluorescence"},
            {"class_id": "group3", "label": "Silicate Defect", "center": 280, "range": 8, "domain": "fluorescence"},
        ],
    }


def reclassify_peaks(
    peaks: list[dict[str, Any]],
    profile: ClassificationProfile,
) -> list[dict[str, Any]]:
    """Re-assign peak labels using custom spectral ranges.

    Fast path — no re-fitting, just label reassignment based on
    modified center/range values from the profile overrides.

    Args:
        peaks: list of peak dicts with center_cm1/center_nm, snr, etc.
        profile: classification profile with overrides

    Returns:
        Same peaks with updated 'assignment' and 'assignment_label' fields.
    """
    defaults = get_default_classification_rules()

    # Build effective rules by applying overrides
    effective_rules: dict[str, list[dict]] = {}
    for domain, rules in defaults.items():
        effective_rules[domain] = []
        for rule in rules:
            # Check if this class has an override
            override = next(
                (
                    o
                    for o in profile.overrides
                    if o.domain == domain and o.class_id == rule["class_id"]
                ),
                None,
            )
            if override and override.disabled:
                continue  # Skip disabled classes

            effective_rule = dict(rule)
            if override:
                if override.label is not None:
                    effective_rule["label"] = override.label
                if override.center is not None:
                    effective_rule["center"] = override.center
                if override.range is not None:
                    effective_rule["range"] = override.range
            effective_rules[domain].append(effective_rule)

    # Re-assign each peak
    result = []
    for peak in peaks:
        peak = dict(peak)  # copy
        center = peak.get("center_cm1") or peak.get("center_nm")
        domain = peak.get("fit_modality", "minerals")

        if center is not None and domain in effective_rules:
            assigned = False
            for rule in effective_rules[domain]:
                if abs(center - rule["center"]) <= rule["range"]:
                    peak["assignment"] = rule["class_id"]
                    peak["assignment_label"] = rule["label"]
                    assigned = True
                    break
            if not assigned:
                peak["assignment"] = "unidentified"
                peak["assignment_label"] = "Unidentified"

        result.append(peak)

    return result


def compute_profile_hash(profile: ClassificationProfile) -> str:
    """SHA-256 hash of profile for cache matching."""
    import hashlib

    return hashlib.sha256(profile.to_json().encode()).hexdigest()
