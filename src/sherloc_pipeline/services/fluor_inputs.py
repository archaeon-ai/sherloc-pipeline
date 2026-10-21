"""Fluorescence fit inputs: ML-despiked R1-R3 rows and laser normalization.

Fluorescence is fitted on the dark-subtracted R1-R3 spectra with the stored
ML cosmic-ray masks applied (Ken's ruling, 2026-10-21: despiked for organics,
hydration and fluorescence). The despiked spectra are persisted as their own
``spectra`` rows (``spectrum_type = "dark_sub_despiked"``) so the point and map
fitters read the same input, and an existing scan can be backfilled from its
stored masks without reprocessing.

Each region's despiked row is its dark-subtracted parent with the parent's
newest stored mask applied inside the certified detection window, exactly as
the web stored-mask toggle does (``web/routes/spectra.py``
``_stitch_point_r123_despiked``). Peaks and masks never attach to these rows:
fluorescence peaks stay linked to the dark-subtracted R2 row.

Laser normalization is a separate stage (``apply_fluorescence_laser_normalization``)
applied by the readers after the input is chosen, behind
``fluorescence_fitting.laser_normalize``. The per-point factor is the Loupe
convention, max photodiode / point photodiode (``core/laser_normalization.py``);
a scalar per point commutes with the masked interpolation, so masking first and
scaling second equals masking the normalized spectrum.
"""

from __future__ import annotations

import logging
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

DESPIKED_SPECTRUM_TYPE = "dark_sub_despiked"
PARENT_SPECTRUM_TYPE = "dark_subtracted"
DESPIKED_PROCESSING_LEVEL = "ml_despiked"
REGIONS: Tuple[str, ...] = ("R1", "R2", "R3")

#: Spectrum types that are instrument measurements (or their PDS processing
#: level). Derived rows such as ``dark_sub_despiked`` are excluded from counts.
MEASURED_SPECTRUM_TYPES: Tuple[str, ...] = (
    "active",
    "dark",
    "dark_subtracted",
    "laser_normalized",
)


def _as_aware(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def select_stored_mask(masks: Sequence[Any]) -> Optional[Any]:
    """Newest stored mask wins, ties broken by method (the web toggle's rule)."""
    if not masks:
        return None
    return max(masks, key=lambda m: (str(m.created_at or ""), m.method))


def _section(config: Any, name: str) -> Any:
    """A config section from a Config object or a plain dict (map mode)."""
    if isinstance(config, dict):
        return config.get(name) or {}
    return getattr(config, name, None) or {}


def interpolation_method(config: Any) -> str:
    """The despike replacement interpolation, as the pipeline and web use it."""
    pp = _section(config, "preprocessing")
    despike = pp.get("despike", {}) if hasattr(pp, "get") else {}
    method = (despike or {}).get("interpolation_method") if hasattr(despike, "get") else None
    return method or "linear"


def _decode(blob: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), dtype=np.float32)


def _encode(values: np.ndarray) -> bytes:
    return zlib.compress(np.asarray(values, dtype=np.float32).tobytes())


def delete_despiked_rows(session, scan_point_ids: Iterable[str]) -> int:
    """Remove the despiked rows of these scan points; returns the count."""
    from sherloc_pipeline.database.models import SpectrumORM

    ids = list(scan_point_ids)
    if not ids:
        return 0
    return (
        session.query(SpectrumORM)
        .filter(
            SpectrumORM.scan_point_id.in_(ids),
            SpectrumORM.spectrum_type == DESPIKED_SPECTRUM_TYPE,
        )
        .delete(synchronize_session=False)
    )


def rebuild_despiked_rows(session, scan_id: str, config: Any) -> Dict[str, int]:
    """Replace a scan's despiked rows from its stored masks.

    Deletes the scan's existing despiked rows first, then writes one row per
    (point, region) whose dark-subtracted parent has a stored mask. A parent
    without a stored mask gets no despiked row, so readers fall back for that
    point. Returns counts for metadata.
    """
    from sherloc_pipeline.core.mask_application import apply_stored_mask_to_array
    from sherloc_pipeline.database.models import ScanPointORM, SpectrumORM
    from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST
    from sherloc_pipeline.services.cr_masks import CRMaskService

    point_ids = [
        pid for (pid,) in session.query(ScanPointORM.id).filter(ScanPointORM.scan_id == scan_id)
    ]
    removed = delete_despiked_rows(session, point_ids)

    parents: Dict[Tuple[str, str], Any] = {}
    for row in (
        session.query(SpectrumORM)
        .filter(
            SpectrumORM.scan_point_id.in_(point_ids),
            SpectrumORM.spectrum_type == PARENT_SPECTRUM_TYPE,
            SpectrumORM.region.in_(REGIONS),
        )
        .order_by(SpectrumORM.created_at)
    ):
        parents.setdefault((row.scan_point_id, row.region), row)

    masks = CRMaskService.get_masks_for_spectra(session, [p.id for p in parents.values()])
    windows = DEFAULT_MANIFEST.region_windows
    interp = interpolation_method(config)

    written = no_mask = 0
    for (point_id, region), parent in parents.items():
        mask = select_stored_mask(masks.get(parent.id, []))
        if mask is None:
            no_mask += 1
            continue
        values = apply_stored_mask_to_array(
            _decode(parent.intensities).astype(float), mask.channel_indices,
            windows[region], interp,
        )
        session.add(SpectrumORM(
            id=str(uuid4()),
            scan_point_id=point_id,
            region=region,
            spectrum_type=DESPIKED_SPECTRUM_TYPE,
            processing_level=DESPIKED_PROCESSING_LEVEL,
            intensities=_encode(values),
            wavelengths=parent.wavelengths,
            wavenumbers=parent.wavenumbers,
            wavelength_source=parent.wavelength_source,
        ))
        written += 1
    session.flush()
    return {
        "despiked_rows_removed": int(removed),
        "despiked_rows_written": written,
        "regions_without_mask": no_mask,
    }


def load_fluorescence_inputs(
    session, scan_point_ids: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Choose each point's R1-R3 fluorescence input.

    Returns ``{scan_point_id: {"arrays": {region: ndarray}, "despiked": bool,
    "link_spectrum_id": str}}``. A point uses its despiked rows only when all
    three regions have one that is newer than both its parent spectrum and the
    parent's newest stored mask; otherwise it uses today's input
    (dark_subtracted, else active) for all three regions. Peaks always link to
    the point's measured R2 row. Points missing a region are omitted.
    """
    from sherloc_pipeline.database.models import SpectrumORM
    from sherloc_pipeline.services.cr_masks import CRMaskService

    ids = list(scan_point_ids)
    rows: Dict[Tuple[str, str, str], Any] = {}
    for row in (
        session.query(SpectrumORM)
        .filter(
            SpectrumORM.scan_point_id.in_(ids),
            SpectrumORM.region.in_(REGIONS),
            SpectrumORM.spectrum_type.in_(
                (DESPIKED_SPECTRUM_TYPE, PARENT_SPECTRUM_TYPE, "active")
            ),
        )
        .order_by(SpectrumORM.created_at)
    ):
        rows.setdefault((row.scan_point_id, row.region, row.spectrum_type), row)

    parent_ids = [
        r.id for (pid, reg, t), r in rows.items() if t == PARENT_SPECTRUM_TYPE
    ]
    masks = CRMaskService.get_masks_for_spectra(session, parent_ids)

    out: Dict[str, Dict[str, Any]] = {}
    for pid in ids:
        measured = {}
        for reg in REGIONS:
            row = rows.get((pid, reg, PARENT_SPECTRUM_TYPE)) or rows.get((pid, reg, "active"))
            if row is not None:
                measured[reg] = row
        if len(measured) != len(REGIONS):
            continue

        despiked = {}
        for reg in REGIONS:
            row = rows.get((pid, reg, DESPIKED_SPECTRUM_TYPE))
            parent = rows.get((pid, reg, PARENT_SPECTRUM_TYPE))
            if row is None or parent is None:
                break
            mask = select_stored_mask(masks.get(parent.id, []))
            built = _as_aware(row.created_at)
            newest_input = max(
                t for t in (
                    _as_aware(parent.updated_at), _as_aware(parent.created_at),
                    _as_aware(getattr(mask, "created_at", None)),
                ) if t is not None
            )
            if mask is None or built is None or built < newest_input:
                break
            despiked[reg] = row

        use_despiked = len(despiked) == len(REGIONS)
        try:
            arrays = {
                reg: _decode((despiked if use_despiked else measured)[reg].intensities)
                .astype(np.float64)
                for reg in REGIONS
            }
        except (zlib.error, ValueError) as exc:
            if not use_despiked:
                logger.warning("Unreadable spectrum for scan point %s: %s", pid, exc)
                continue
            # A bad despiked blob falls back to the measured rows.
            try:
                arrays = {reg: _decode(measured[reg].intensities).astype(np.float64)
                          for reg in REGIONS}
            except (zlib.error, ValueError) as exc2:
                logger.warning("Unreadable spectrum for scan point %s: %s", pid, exc2)
                continue
            use_despiked = False
        out[pid] = {
            "arrays": arrays,
            "despiked": use_despiked,
            "link_spectrum_id": measured["R2"].id,
        }
    return out


# ── laser normalization (one isolated stage; drop by config) ───────────────


def laser_normalization_factors(
    photodiode_means: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Per-point factor max_pd / pd for points with a positive photodiode mean."""
    valid = [v for v in photodiode_means.values() if v is not None and v > 0]
    if not valid:
        return {k: None for k in photodiode_means}
    max_pd = max(valid)
    return {
        k: (max_pd / v if v is not None and v > 0 else None)
        for k, v in photodiode_means.items()
    }


def apply_fluorescence_laser_normalization(
    arrays: Dict[str, np.ndarray], factor: Optional[float]
) -> Tuple[Dict[str, np.ndarray], bool]:
    """Scale one point's region arrays by its laser-normalization factor.

    Returns the (possibly unscaled) arrays and whether scaling was applied; a
    point without a factor is returned unchanged.
    """
    if factor is None:
        return arrays, False
    return {reg: a * factor for reg, a in arrays.items()}, True


def laser_normalize_enabled(config: Any) -> bool:
    """Whether the laser-normalization stage runs.

    Read from the caller's config; a partial config without the key (map-mode
    callers pass dicts) uses the packaged config.yaml, where it is set
    explicitly.
    """
    fluor_cfg = _section(config, "fluorescence_fitting")
    value = fluor_cfg.get("laser_normalize") if hasattr(fluor_cfg, "get") else None
    if value is None:
        from sherloc_pipeline.config import get_config

        value = (get_config().fluorescence_fitting or {})["laser_normalize"]
    return bool(value)


def summarize_inputs(chosen: Dict[str, Dict[str, Any]], normalized: List[str],
                     total_points: int) -> Dict[str, Any]:
    """Metadata recording which input each fit used."""
    despiked = sum(1 for v in chosen.values() if v["despiked"])
    return {
        "input_spectrum_type": DESPIKED_SPECTRUM_TYPE if despiked == len(chosen) and chosen
        else ("mixed" if despiked else PARENT_SPECTRUM_TYPE),
        "points_despiked": despiked,
        "points_fallback": len(chosen) - despiked,
        "points_laser_normalized": len(normalized),
        "points_total": total_points,
    }
