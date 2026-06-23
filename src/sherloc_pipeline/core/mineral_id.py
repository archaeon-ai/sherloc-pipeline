from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml


@dataclass
class MineralRule:
    label: str
    lo: float  # inclusive lower bound (cm^-1)
    hi: float  # inclusive upper bound (cm^-1)


def _load_default_rules_from_config() -> List[MineralRule]:
    """Load mineral rules from config.yaml (single source of truth).

    This function fails fast with clear error messages if configuration is missing
    or invalid, ensuring scientific data processing uses correct mineral ranges.

    Returns:
        List of MineralRule objects loaded from config.yaml

    Raises:
        RuntimeError: If config.yaml cannot be loaded or is missing mineral_rules
    """
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise RuntimeError(
            f"Configuration file not found: {config_path}\n"
            "Cannot load mineral classification rules. Ensure config.yaml exists in package."
        )

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse config.yaml: {e}\n"
            "Check YAML syntax in configuration file."
        )

    if not cfg or "fitting" not in cfg or "mineral_rules" not in cfg["fitting"]:
        raise RuntimeError(
            "Missing 'fitting.mineral_rules' in config.yaml\n"
            "Mineral classification rules are required for pipeline operation."
        )

    try:
        rules_data = cfg["fitting"]["mineral_rules"]
        return [MineralRule(str(d["label"]), float(d["lo"]), float(d["hi"])) for d in rules_data]
    except (KeyError, ValueError, TypeError) as e:
        raise RuntimeError(
            f"Invalid mineral_rules format in config.yaml: {e}\n"
            "Expected list of dicts with 'label', 'lo', and 'hi' keys."
        )

# Provenance: Mineral identification wavenumber ranges derived from published
# spectral libraries and validated against laboratory standards by domain expert.

# Load DEFAULT_RULES from config.yaml at module import time
# This makes config.yaml the single source of truth
DEFAULT_RULES: List[MineralRule] = _load_default_rules_from_config()


def load_mineral_rules(path: Optional[Path] = None, inline_rules: Optional[list] = None) -> List[MineralRule]:
    """Load mineral rules from inline config list or YAML/CSV path; fallback to defaults.

    YAML format:
    - label: olivine
      lo: 820
      hi: 860
    """
    # Prefer inline rules if provided
    if inline_rules:
        try:
            return [MineralRule(str(d["label"]), float(d["lo"]), float(d["hi"])) for d in inline_rules]
        except Exception:
            pass
    if path is None:
        return list(DEFAULT_RULES)
    try:
        p = Path(path)
        if not p.exists():
            return list(DEFAULT_RULES)
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(p.read_text()) or []
            rules = [MineralRule(str(d["label"]), float(d["lo"]), float(d["hi"])) for d in data]
            return rules
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            rules = [MineralRule(str(r["label"]), float(r["lo"]), float(r["hi"])) for _, r in df.iterrows()]
            return rules
        else:
            return list(DEFAULT_RULES)
    except Exception:
        return list(DEFAULT_RULES)


def assign_min_id(center_cm1: float, rules: List[MineralRule]) -> str:
    """Assign mineral ID using inclusive bounds with deterministic tie-breaker.

    If multiple rules match due to overlapping integer boundaries, choose the
    rule with the greatest lower bound (lo). This implements the policy that
    the higher-range interval owns the shared integer boundary.
    """
    try:
        x = float(center_cm1)
        matches = [r for r in rules if (r.lo <= x <= r.hi)]
        if not matches:
            return "unidentified"
        # Prefer the rule with the greatest lower bound when multiple match
        best = max(matches, key=lambda r: (r.lo, r.hi))
        return best.label
    except Exception:
        return "unidentified"


def map_min_id_series(center_series: pd.Series, rules: List[MineralRule]) -> pd.Series:
    return center_series.apply(lambda v: assign_min_id(v, rules))


def classify_organic_band(center_cm1: float) -> str:
    """Classify organic peak by wavenumber.

    D band: 1250-1450 cm-1 (disorder-induced)
    G band: 1500-1700 cm-1 (graphitic)
    """
    if 1250 <= center_cm1 <= 1450:
        return "D_band"
    elif 1500 <= center_cm1 <= 1700:
        return "G_band"
    return "unidentified"


def classify_hydration_band(center_cm1: float) -> str:
    """Classify hydration peak by wavenumber.

    OH stretch: 3000-4000 cm-1
    H2O bend: 1500-1700 cm-1
    """
    if 3000 <= center_cm1 <= 4000:
        return "OH_stretch"
    elif 1500 <= center_cm1 <= 1700:
        return "H2O_bend"
    return "unidentified"


