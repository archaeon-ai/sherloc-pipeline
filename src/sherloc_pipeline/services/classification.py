"""Classification service — re-assign peak labels with custom profiles.

The 'fast path' for classification: re-assign existing peaks to different
classes using modified spectral range boundaries from a custom profile.
No re-fitting required.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

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

