"""Which stored ``fitted_peaks`` rows the web views count as shown peaks.

Raman components (minerals, organics, hydration) are stored down to
``fitting.store_min_snr``; the views count a Raman row only when its SNR is at least the
display threshold ``fitting.min_snr``, the same value as the ``pass_snr`` flag and the
accepted-peaks files. With the two settings equal, as in the shipped configuration, every
stored Raman row passes and the views are unchanged. Rows of other modalities (fluorescence
has its own threshold at fit time) and rows without an SNR are counted as before.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import or_

from sherloc_pipeline.database.models import FittedPeakORM

#: The modalities whose components are stored down to ``fitting.store_min_snr``.
RAMAN_MODALITIES = ("minerals", "organics", "hydration")

#: The display threshold when no configuration is available.
DEFAULT_DISPLAY_MIN_SNR = 3.0


def display_min_snr(config: Any) -> float:
    """``fitting.min_snr`` of the app configuration (3.0 when absent)."""
    fitting = getattr(config, "fitting", None) if config is not None else None
    if not isinstance(fitting, dict):
        return DEFAULT_DISPLAY_MIN_SNR
    return float(fitting.get("min_snr", DEFAULT_DISPLAY_MIN_SNR))


def shown_peak_clause(min_snr: float):
    """SQL condition: the row is not a Raman component below ``min_snr``."""
    return or_(
        FittedPeakORM.fit_modality.notin_(RAMAN_MODALITIES),
        FittedPeakORM.snr.is_(None),
        FittedPeakORM.snr >= min_snr,
    )
