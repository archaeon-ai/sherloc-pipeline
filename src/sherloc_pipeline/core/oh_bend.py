"""O-H bend versus G band: the stretch-to-bend rule for the organics window.

Molecular water has a bending mode near 1600-1650 cm-1, inside the window the
organics fit searches for the G band (1500-1700 cm-1). A point whose hydration
fit accepted an O-H stretch therefore predicts some bend in that window. The
rule here compares a G-window peak with the bend the same point's stretch
predicts:

* predicted bend = ``ratio x stretch height``, where ``ratio`` (bend height /
  stretch height) depends on the kind of stretch (its center and width) and
  stretch height is the maximum of the stretch's fitted model;
* the peak's noise is ``amplitude / snr`` (the fit's own noise estimate);
* ``amplitude <= predicted + bend_sigma x noise``  -> ``H2O_bend``;
* ``amplitude - predicted >= excess_sigma x noise`` -> ``G_band`` (the excess
  over the predicted bend is what the carbon label rests on);
* anything between -> ``G_band_or_H2O_bend`` (named as uncertain, not silently
  one or the other).

A point without an accepted stretch is not touched: the peak keeps the label
its center gives it. All numbers come from ``fitting.organics_oh_bend`` in
config.yaml; see docs/METHODS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

G_LABEL = "G_band"
BEND_LABEL = "H2O_bend"
UNCERTAIN_LABEL = "G_band_or_H2O_bend"

# Short labels used in the organics accepted-peaks table (``band`` column),
# which already uses "G" and "D" for the carbon bands.
SHORT_LABEL = {G_LABEL: "G", BEND_LABEL: BEND_LABEL, UNCERTAIN_LABEL: UNCERTAIN_LABEL}

_FWHM_TO_EXP = 4.0 * math.log(2.0)


@dataclass(frozen=True)
class StretchClass:
    """One kind of O-H stretch, matched on the largest stretch component."""

    name: str
    ratio: float
    center_min: float = -math.inf
    center_max: float = math.inf
    fwhm_min: float = 0.0
    fwhm_max: float = math.inf

    def matches(self, center: float, fwhm: float) -> bool:
        return (self.center_min <= center < self.center_max) and (self.fwhm_min <= fwhm < self.fwhm_max)


@dataclass(frozen=True)
class OhBendRule:
    classes: Tuple[StretchClass, ...]
    bend_sigma: float
    excess_sigma: float

    @classmethod
    def from_fitting_config(cls, fit_cfg: Mapping[str, Any]) -> Optional["OhBendRule"]:
        """Build the rule from ``fitting.organics_oh_bend``; None when the block is absent."""
        block = (fit_cfg or {}).get("organics_oh_bend")
        if not block:
            return None
        classes = []
        for i, entry in enumerate(block.get("stretch_classes") or []):
            classes.append(StretchClass(
                name=str(entry.get("name", f"class_{i}")),
                ratio=float(entry["ratio"]),
                center_min=float(entry.get("center_min", -math.inf)),
                center_max=float(entry.get("center_max", math.inf)),
                fwhm_min=float(entry.get("fwhm_min", 0.0)),
                fwhm_max=float(entry.get("fwhm_max", math.inf)),
            ))
        if not classes:
            raise ValueError("fitting.organics_oh_bend.stretch_classes is empty")
        bend_sigma = float(block["bend_sigma"])
        excess_sigma = float(block["excess_sigma"])
        if excess_sigma < bend_sigma:
            raise ValueError("fitting.organics_oh_bend: excess_sigma must be >= bend_sigma")
        return cls(classes=tuple(classes), bend_sigma=bend_sigma, excess_sigma=excess_sigma)

    def classify_stretch(self, components: Sequence[Tuple[float, float, float]]) -> Optional[StretchClass]:
        """The class of the largest-amplitude component; None when nothing matches."""
        if not components:
            return None
        center, _amp, fwhm = max(components, key=lambda c: c[1])
        for sc in self.classes:
            if sc.matches(center, fwhm):
                return sc
        return None

    def evaluate(
        self,
        amplitude: float,
        snr: float,
        components: Sequence[Tuple[float, float, float]],
    ) -> Optional["BendVerdict"]:
        """Verdict for one G-window peak; None when the point has no accepted stretch."""
        if not components:
            return None
        sc = self.classify_stretch(components)
        height = stretch_height(components)
        ratio = sc.ratio if sc is not None else 0.0
        predicted = ratio * height
        noise = amplitude / snr if snr and snr > 0 else math.inf
        excess_snr = (amplitude - predicted) / noise if math.isfinite(noise) and noise > 0 else 0.0
        if amplitude <= predicted + self.bend_sigma * noise:
            label = BEND_LABEL
        elif excess_snr >= self.excess_sigma:
            label = G_LABEL
        else:
            label = UNCERTAIN_LABEL
        return BendVerdict(
            label=label,
            stretch_class=sc.name if sc is not None else "unmatched",
            stretch_height=height,
            bend_predicted=predicted,
            bend_excess_snr=excess_snr,
        )


@dataclass(frozen=True)
class BendVerdict:
    label: str
    stretch_class: str
    stretch_height: float
    bend_predicted: float
    bend_excess_snr: float

    def as_row(self) -> Dict[str, Any]:
        return {
            "bend_rule": self.label,
            "oh_stretch_class": self.stretch_class,
            "oh_stretch_height": self.stretch_height,
            "bend_predicted": self.bend_predicted,
            "bend_excess_snr": self.bend_excess_snr,
        }


def stretch_height(components: Iterable[Tuple[float, float, float]], n: int = 2001) -> float:
    """Maximum of the summed Gaussian stretch model (components are (center, amplitude, fwhm))."""
    comps = [(float(c), float(a), float(w)) for c, a, w in components]
    if not comps:
        return 0.0
    lo = min(c - 2.0 * w for c, _a, w in comps)
    hi = max(c + 2.0 * w for c, _a, w in comps)
    best = 0.0
    for i in range(n):
        x = lo + (hi - lo) * i / (n - 1)
        y = sum(a * math.exp(-_FWHM_TO_EXP * (x - c) ** 2 / (w * w)) for c, a, w in comps if w > 0)
        best = max(best, y)
    return best


def load_hydration_stretches(rows: Iterable[Mapping[str, Any]]) -> Dict[int, List[Tuple[float, float, float]]]:
    """Group hydration accepted-peaks rows into {point: [(center, amplitude, fwhm), ...]}."""
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for r in rows:
        try:
            point = int(r["point"])
            comp = (float(r["center_cm1"]), float(r["amplitude_a"]), float(r["fwhm_cm1"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in comp):
            continue
        out.setdefault(point, []).append(comp)
    return out
