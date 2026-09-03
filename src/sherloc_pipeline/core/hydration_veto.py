"""Cosmic-ray veto for hydration (OH-stretch) peak fits.

Hydration fitting runs on the *non-despiked* spectrum by design (published-method
fidelity), so a cosmic ray inside the 2800-3900 cm-1 window can pass the R2 and
F-test gates and be reported as an OH-stretch feature with its FWHM pinned
against the lower bound.

Two independent signals are provided here:

**Mask veto** — run the existing classical despiker on the same spectrum that was
fit and reject/flag a candidate whose centre falls on masked spike bins, or whose
raw-to-despiked amplitude drop at the centre exceeds a ratio threshold. The spike
mask returned by :func:`despike_r1_spectrum` is the input; it is normally
discarded at call sites.

**Bound pinning** — a fit that converged within epsilon of the FWHM floor is
unreliable by construction: the optimiser wanted a narrower peak than the model
allows. This is a flag only, never a rejection.

Both mechanisms sit behind a feature flag that is DEFAULT OFF. The thresholds are
science decisions; the defaults below are proposals pending operator ratification
(see ``docs/reports/HYDRATION_CR_VETO_EVIDENCE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sherloc_pipeline.core.preprocessing import DespikeParams, despike_r1_spectrum

__all__ = [
    "FLAG_MASK_HIT",
    "FLAG_AMPLITUDE_DROP",
    "FLAG_FWHM_FLOOR_PINNED",
    "HydrationVetoConfig",
    "HydrationVetoResult",
    "despike_for_veto",
    "evaluate_hydration_peak",
]

# Flag vocabulary. Stable strings — they are written into peak rows/DTOs.
FLAG_MASK_HIT = "cr_mask_hit"
FLAG_AMPLITUDE_DROP = "cr_amplitude_drop"
FLAG_FWHM_FLOOR_PINNED = "fwhm_floor_pinned"

# Proposed defaults (NOT ratified — see the evidence report).
DEFAULT_CENTER_WINDOW_CM1 = 15.0
DEFAULT_AMPLITUDE_DROP_RATIO_MAX = 0.5
DEFAULT_MASK_MIN_DROP_RATIO = 0.10
DEFAULT_FWHM_FLOOR_EPSILON_CM1 = 0.5


@dataclass(frozen=True)
class HydrationVetoConfig:
    """Feature-flagged configuration for the hydration cosmic-ray veto.

    Attributes:
        enabled: Master feature flag. DEFAULT OFF — when False every call site
            behaves exactly as it did before this module existed.
        action: ``"reject"`` drops a cosmic-ray-implicated candidate from the
            accepted set; ``"flag"`` keeps it and only annotates it. Bound
            pinning is always flag-only regardless of this setting.
        center_window_cm1: Half-width of the window around a candidate centre
            searched for masked spike bins and for the raw-to-despiked drop.
        amplitude_drop_ratio_max: Maximum tolerated fraction of the candidate's
            local height that the despiker removed. Above this the candidate is
            spike-dominated.
        mask_min_drop_ratio: Minimum fraction of the candidate's local height a
            *masked* bin must account for before the mask signal counts. Guards
            against the despiker nicking a single noise channel on the apex of
            an authentic broad band.
        fwhm_floor_cm1: The FWHM lower bound the fit was run against.
        fwhm_floor_epsilon_cm1: A fit within this distance of the floor counts
            as converged at the bound.
    """

    enabled: bool = False
    action: str = "reject"
    center_window_cm1: float = DEFAULT_CENTER_WINDOW_CM1
    amplitude_drop_ratio_max: float = DEFAULT_AMPLITUDE_DROP_RATIO_MAX
    mask_min_drop_ratio: float = DEFAULT_MASK_MIN_DROP_RATIO
    fwhm_floor_cm1: float = 50.0
    fwhm_floor_epsilon_cm1: float = DEFAULT_FWHM_FLOOR_EPSILON_CM1

    @property
    def rejects(self) -> bool:
        """True when a cosmic-ray signal removes the candidate rather than annotating it."""
        return self.action == "reject"

    @classmethod
    def from_fitting_config(cls, fitting_cfg: Optional[Mapping[str, Any]]) -> "HydrationVetoConfig":
        """Build a config from the ``fitting`` section of ``config.yaml``.

        Reads the ``hydration_cr_veto`` sub-mapping, defaulting the FWHM floor to
        the same ``hydration_fwhm_min_cm1`` the fit is bounded by so the two can
        never drift apart.
        """
        cfg = dict(fitting_cfg or {})
        raw = cfg.get("hydration_cr_veto") or {}
        if not isinstance(raw, Mapping):
            raw = getattr(raw, "__dict__", {}) or {}
        floor = float(cfg.get("hydration_fwhm_min_cm1", 50.0))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            action=str(raw.get("action", "reject")),
            center_window_cm1=float(raw.get("center_window_cm1", DEFAULT_CENTER_WINDOW_CM1)),
            amplitude_drop_ratio_max=float(
                raw.get("amplitude_drop_ratio_max", DEFAULT_AMPLITUDE_DROP_RATIO_MAX)
            ),
            mask_min_drop_ratio=float(
                raw.get("mask_min_drop_ratio", DEFAULT_MASK_MIN_DROP_RATIO)
            ),
            fwhm_floor_cm1=float(raw.get("fwhm_floor_cm1", floor)),
            fwhm_floor_epsilon_cm1=float(
                raw.get("fwhm_floor_epsilon_cm1", DEFAULT_FWHM_FLOOR_EPSILON_CM1)
            ),
        )


@dataclass(frozen=True)
class HydrationVetoResult:
    """Per-candidate verdict from the hydration cosmic-ray veto."""

    vetoed: bool
    flags: Tuple[str, ...]
    mask_hit: bool
    bound_pinned: bool
    amplitude_drop_ratio: Optional[float]

    def as_row(self) -> Dict[str, Any]:
        """Serialise to the columns appended to hydration peak records."""
        return {
            "cr_vetoed": bool(self.vetoed),
            "cr_veto_flags": ";".join(self.flags),
            "cr_mask_hit": bool(self.mask_hit),
            "cr_amplitude_drop_ratio": self.amplitude_drop_ratio,
            "fwhm_floor_pinned": bool(self.bound_pinned),
        }


def despike_for_veto(
    x: np.ndarray,
    y: np.ndarray,
    params: Optional[DespikeParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Despike a spectrum purely to obtain its spike mask.

    The despiked spectrum is *not* what gets fit — it exists only so the veto can
    measure how much of a candidate's height the despiker would have removed.

    Args:
        x: Raman shift axis (cm-1).
        y: Intensity, as it was handed to the fitter (non-despiked).
        params: Classical despiker parameters; defaults to :class:`DespikeParams`.

    Returns:
        ``(despiked, spike_mask)`` as positional float/bool arrays aligned to ``y``.
        On despiker failure the input is returned with an all-False mask, so a
        veto can never manufacture rejections out of an error.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    try:
        series = pd.Series(y_arr, index=np.arange(len(y_arr)))
        despiked, mask = despike_r1_spectrum(
            series, params or DespikeParams(), raman_shift=x_arr
        )
        return (
            despiked.to_numpy(dtype=np.float64),
            mask.to_numpy(dtype=bool),
        )
    except Exception:
        return y_arr, np.zeros(y_arr.shape, dtype=bool)


def evaluate_hydration_peak(
    center_cm1: Optional[float],
    fwhm_cm1: Optional[float],
    x: np.ndarray,
    y_raw: np.ndarray,
    y_despiked: np.ndarray,
    spike_mask: np.ndarray,
    config: HydrationVetoConfig,
) -> HydrationVetoResult:
    """Score one hydration candidate against the mask, amplitude, and bound tests.

    Args:
        center_cm1: Fitted peak centre.
        fwhm_cm1: Fitted peak FWHM, used for the bound-pinning test and to size
            the local reference window.
        x: Raman shift axis covering the fitted spectrum.
        y_raw: The intensity that was fit (non-despiked).
        y_despiked: Output of :func:`despike_for_veto` on the same array.
        spike_mask: Spike mask from :func:`despike_for_veto`.
        config: Threshold configuration.

    Returns:
        A :class:`HydrationVetoResult`. ``vetoed`` is only ever True when a
        cosmic-ray signal fired AND ``config.action == "reject"``.
    """
    empty = HydrationVetoResult(False, (), False, False, None)
    if center_cm1 is None:
        return empty

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.size == 0:
        return empty
    y_arr = np.asarray(y_raw, dtype=np.float64)
    d_arr = np.asarray(y_despiked, dtype=np.float64)
    mask_arr = np.asarray(spike_mask, dtype=bool)

    window = float(config.center_window_cm1)
    near = np.abs(x_arr - float(center_cm1)) <= window
    if not near.any():
        # Centre outside the supplied axis (or a window narrower than the
        # channel spacing): fall back to the single nearest channel.
        near = np.zeros(x_arr.shape, dtype=bool)
        near[int(np.argmin(np.abs(x_arr - float(center_cm1))))] = True

    aligned = x_arr.size == y_arr.size == d_arr.size == mask_arr.size

    # Local reference level: the despiked spectrum one FWHM out from the centre.
    # Using the despiked trace keeps a neighbouring spike from inflating the
    # reference and hiding the drop.
    height = 0.0
    ratio: Optional[float] = None
    mask_ratio = 0.0
    if aligned:
        span = max(float(fwhm_cm1 or config.fwhm_floor_cm1), window)
        local = np.abs(x_arr - float(center_cm1)) <= span
        if not local.any():
            local = near
        reference = float(np.median(d_arr[local]))
        height = float(np.max(y_arr[near])) - reference
        removed = y_arr[near] - d_arr[near]
        if height > 0:
            ratio = max(0.0, min(1.0, float(np.max(removed)) / height))
            # Only positive-going masked bins implicate a cosmic ray: spikes are
            # additive, so a masked bin where the despiker RAISED the trace is a
            # downward outlier and says nothing about the candidate.
            masked_near = mask_arr[near] & (removed > 0.0)
            if masked_near.any():
                mask_ratio = max(0.0, min(1.0, float(np.max(removed[masked_near])) / height))

    # A masked bin only counts against the candidate when the despiker actually
    # took a material bite out of it. Without this floor the veto also fires on
    # authentic broad bands, where the rolling-median despiker routinely nicks a
    # single noise channel near the apex — a real false-positive risk measured
    # during development, not a hypothetical one.
    mask_hit = bool(mask_ratio > float(config.mask_min_drop_ratio))
    amplitude_hit = ratio is not None and ratio > float(config.amplitude_drop_ratio_max)

    bound_pinned = (
        fwhm_cm1 is not None
        and float(fwhm_cm1) <= config.fwhm_floor_cm1 + config.fwhm_floor_epsilon_cm1
    )

    flags = []
    if mask_hit:
        flags.append(FLAG_MASK_HIT)
    if amplitude_hit:
        flags.append(FLAG_AMPLITUDE_DROP)
    if bound_pinned:
        flags.append(FLAG_FWHM_FLOOR_PINNED)

    # Bound pinning is a second, independent signal — reported, never fatal.
    vetoed = bool((mask_hit or amplitude_hit) and config.rejects)

    return HydrationVetoResult(
        vetoed=vetoed,
        flags=tuple(flags),
        mask_hit=mask_hit,
        bound_pinned=bool(bound_pinned),
        amplitude_drop_ratio=ratio,
    )
