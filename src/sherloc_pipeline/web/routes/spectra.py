"""Spectrum endpoints: average, single point, subset."""

import logging
import zlib
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy.orm import Session

from sherloc_pipeline.core.badpix import load_badpix_channels
from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from sherloc_pipeline.core.mask_application import (
    apply_stored_mask_to_array,
    count_uncovered_contributor_channels,
)
from sherloc_pipeline.core.preprocessing import (
    DespikeParams,
    apply_mask_replacement,
    despike_r1_spectrum,
)
from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SpectrumORM
from sherloc_pipeline.services.cr_masks import PARENT_SPECTRUM_TYPE, CRMaskService
from sherloc_pipeline.web.adapters import numpy_to_list
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.schemas import (
    AverageSpectrumResponse,
    BadpixChannelItem,
    BadpixResponse,
    PointSpectrumResponse,
    ProvenanceInfo,
    SubsetRequest,
    SubsetResponse,
)

router = APIRouter(prefix="/api", tags=["spectra"])

VALID_REGIONS = {"R1", "R2", "R3", "R123"}

#: Allowed values of the coarse ``despike_method`` request selector
#: (issue #6). Distinct from the precise response ``despike_method``
#: provenance string the ml path returns.
VALID_DESPIKE_METHODS = {"none", "ml", "modz"}

#: Regions a live modz despike can cover. The modz algorithm is defined on
#: the R1 Raman plane only (``despike_r1_spectrum``); R2/R3/R123 are served
#: non-despiked with the un-covered region(s) disclosed.
MODZ_REGIONS = {"R1"}


def _resolve_despike_method(despike: bool, despike_method: Optional[str]) -> str:
    """Resolve the effective despike strategy (issue #6 method selector).

    Precedence: an explicit ``despike_method`` wins; when absent the legacy
    ``despike`` bool maps ``False -> "none"``, ``True -> "ml"`` (the
    stored-mask path that shipped in v4.2.x). The legacy bool keeps working
    forever (web API back-compat, docs/INVARIANTS.md).

    Raises:
        HTTPException(422): if ``despike_method`` is a non-empty value outside
            ``none | ml | modz``. (For the query-param endpoints this guards
            the manually-typed ``Optional[str]``; the subset body field is a
            pydantic ``Literal`` and is already 422'd before reaching here.)
    """
    if despike_method is not None:
        if despike_method not in VALID_DESPIKE_METHODS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid despike_method: {despike_method!r} "
                    f"(allowed: none, ml, modz)"
                ),
            )
        return despike_method
    return "ml" if despike else "none"

#: Regions contributing to the web R123 composite view (Loupe overlap
#: summation, constituent-first application — spec §4.6).
R123_CONTRIBUTORS = ("R1", "R2", "R3")


# ---------------------------------------------------------------------------
# Stored-mask despike toggle (spec §4.6, MLD-IFC-003/004, MLD-PER-002)
#
# The serving host never runs ML inference: ``?despike=true`` looks up the
# masks the pipeline already persisted (``cosmic_ray_masks`` table) and
# applies the *same* interpolation helper the pipeline uses. No onnxruntime
# and no ``ml_despike`` package is imported at web module load — the only
# ml_despike touch is a request-time import of the stdlib-only frozen
# manifest, reached solely on the despike branch (MLD-QUA-002 AC2).
# ---------------------------------------------------------------------------


def _despike_region_windows():
    """Certified per-region detection windows from the frozen manifest.

    Imported lazily (request-time) so importing the web app never imports
    ``ml_despike``; ``manifest.py`` is stdlib-only, so this never pulls
    onnxruntime even here (MLD-QUA-002 AC2). The manifest is the single
    source of truth for the windows — they are not re-hardcoded in ``web``.
    """
    from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

    return DEFAULT_MANIFEST.region_windows


def _despike_interpolation_method(config) -> str:
    """Resolve the despike replacement interpolation method from config.

    Mirrors the pipeline replacement (``DespikeParams.interpolation_method``,
    default ``"linear"``) so the web toggle replaces flagged channels the
    same way the processing path does.
    """
    pp = getattr(config, "preprocessing", None) or {}
    despike = pp.get("despike", {}) if hasattr(pp, "get") else {}
    method = (despike or {}).get("interpolation_method") if hasattr(despike, "get") else None
    return method or "linear"


def _modz_params_from_config(config) -> DespikeParams:
    """Build modz ``DespikeParams`` from the bundled config defaults.

    Resolves ``preprocessing.despike`` the same way the rest of the web tier
    reads config (``getattr`` then ``.get``, tolerant of the dataclass-style
    config and the dict-style test stub). Any key the config omits falls back
    to the ``DespikeParams`` field default, so a minimal config still yields a
    valid set of parameters — matching the CLI pipeline's legacy modz
    defaults.
    """
    pp = getattr(config, "preprocessing", None) or {}
    despike = pp.get("despike", {}) if hasattr(pp, "get") else {}
    if not hasattr(despike, "get"):
        despike = {}
    defaults = DespikeParams()

    def _get(key, fallback):
        val = despike.get(key, None)
        return fallback if val is None else val

    def _tuple(key, fallback):
        val = despike.get(key, None)
        return fallback if val is None else tuple(val)

    return DespikeParams(
        window_size=int(_get("window_size", defaults.window_size)),
        zscore_threshold=float(_get("zscore_threshold", defaults.zscore_threshold)),
        max_iterations=int(_get("max_iterations", defaults.max_iterations)),
        interpolation_method=str(
            _get("interpolation_method", defaults.interpolation_method)
        ),
        run_length_max=int(_get("run_length_max", defaults.run_length_max)),
        laser_window=_tuple("laser_window", defaults.laser_window),
        sulfate_center_window=_tuple(
            "sulfate_center_window", defaults.sulfate_center_window
        ),
        sulfate_guard_enable=bool(
            _get("sulfate_guard_enable", defaults.sulfate_guard_enable)
        ),
        sulfate_guard_search=_tuple(
            "sulfate_guard_search", defaults.sulfate_guard_search
        ),
        sulfate_guard_min_prominence=float(
            _get("sulfate_guard_min_prominence", defaults.sulfate_guard_min_prominence)
        ),
        sulfate_guard_min_halfwidth=float(
            _get("sulfate_guard_min_halfwidth", defaults.sulfate_guard_min_halfwidth)
        ),
        sulfate_guard_max_halfwidth=float(
            _get("sulfate_guard_max_halfwidth", defaults.sulfate_guard_max_halfwidth)
        ),
    )


def _modz_params_dict(params: DespikeParams) -> dict:
    """Serialize the modz params actually used into the response dict.

    JSON-friendly: tuples are emitted as lists. This is the additive
    ``despike_params_used`` field — present only on the modz path so the
    client can show exactly which thresholds produced the served array.
    """
    return {
        "window_size": params.window_size,
        "zscore_threshold": params.zscore_threshold,
        "max_iterations": params.max_iterations,
        "interpolation_method": params.interpolation_method,
        "run_length_max": params.run_length_max,
        "laser_window": list(params.laser_window),
        "sulfate_center_window": list(params.sulfate_center_window),
        "sulfate_guard_enable": params.sulfate_guard_enable,
        "sulfate_guard_search": list(params.sulfate_guard_search),
        "sulfate_guard_min_prominence": params.sulfate_guard_min_prominence,
        "sulfate_guard_min_halfwidth": params.sulfate_guard_min_halfwidth,
        "sulfate_guard_max_halfwidth": params.sulfate_guard_max_halfwidth,
    }


def _apply_modz_to_served_r1(
    intensity: np.ndarray,
    wavenumber: np.ndarray,
    params: DespikeParams,
):
    """Run the legacy modz despike on a SERVED R1 array (issue #6).

    Unlike the ``ml`` stored-mask path (per-constituent, applied before
    averaging), modz here runs on the *served* array — the averaged spectrum
    for the average/subset endpoints, the single point's spectrum for the
    point endpoint. This intentionally matches the legacy Workbench
    client-side modz step's display-level behavior, and differs from the CLI
    pipeline's per-spectrum pre-fit modz. R1 only.

    Args:
        intensity: The served R1 intensity array.
        wavenumber: The matching R1 Raman-shift axis (drives the laser /
            sulfate exclusion windows).
        params: Resolved modz ``DespikeParams``.

    Returns:
        ``(despiked_array, n_spikes_replaced)``.
    """
    series = pd.Series(np.asarray(intensity, dtype=float))
    despiked_series, spike_mask = despike_r1_spectrum(
        series, params, raman_shift=np.asarray(wavenumber, dtype=float)
    )
    return despiked_series.to_numpy(copy=True), int(spike_mask.to_numpy().sum())


def _select_stored_mask(masks):
    """Pick the stored mask record to apply for one spectrum.

    ``get_masks_for_spectra`` returns every method's record per spectrum;
    v1 persists a single method per (spectrum, method) so the list is length
    one in practice. Selection is deterministic regardless: newest
    ``created_at`` wins, ties broken by method name.
    """
    if not masks:
        return None
    return max(masks, key=lambda m: (str(m.created_at or ""), m.method))


def _despike_region_array(sel, selected_channels, channel_indices, window, interp):
    """Apply a region's stored mask to a region-selected intensity array.

    ``sel`` is the served (wavelength-window-selected) intensity array;
    ``selected_channels[row]`` is the absolute channel index of that row.
    Only mask channels that fall inside the region's certified window **and**
    inside the served selection are applicable (others are persisted but not
    applicable to this frame — spec §3.3).

    Returns ``(despiked_array, applied_absolute_channels_sorted)``.
    """
    lo, hi = window
    row_of_channel = {int(ch): row for row, ch in enumerate(selected_channels)}
    row_mask = np.zeros(len(sel), dtype=bool)
    applied = []
    for ch in channel_indices:
        ci = int(ch)
        if lo <= ci < hi:
            row = row_of_channel.get(ci)
            if row is not None:
                row_mask[row] = True
                applied.append(ci)
    if not row_mask.any():
        return np.asarray(sel, dtype=float), []
    out = apply_mask_replacement(
        pd.Series(np.asarray(sel, dtype=float)), row_mask, interp
    ).to_numpy(copy=True)
    return out, sorted(applied)

# Loupe stores dark-subtracted spectra; PDS stores laser-normalized.
_SPECTRUM_TYPE_BY_SOURCE = {
    "loupe": "dark_subtracted",
    "pds4": "laser_normalized",
}


def _resolve_spectrum_type(scan: ScanORM) -> str:
    """Return the appropriate spectrum_type filter for a scan's data source."""
    return _SPECTRUM_TYPE_BY_SOURCE.get(
        getattr(scan, "data_source", None) or "loupe",
        "dark_subtracted",
    )


def _get_session(request: Request) -> Session:
    return request.state.db


def _get_data_access(request: Request) -> DataAccessService:
    """Resolve the DataAccessService from app state."""
    access_mode = getattr(request.app.state, "access_mode", "internal")
    return DataAccessService(access_mode=access_mode)


def _get_wavelength_wavenumber(region: str):
    """Compute calibrated wavelength/wavenumber and region mask.

    For R123, returns the full 2148-channel arrays with an all-True mask.
    """
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
    if region == "R123":
        mask = np.ones(len(wavelength), dtype=bool)
    else:
        mask = get_region_wavelength_mask(wavelength, region)
    return wavelength, wavenumber, mask


def _extract_intensities(spectrum: SpectrumORM) -> np.ndarray:
    """Decode intensities from DB binary (zlib-compressed float32) to numpy array."""
    return np.frombuffer(zlib.decompress(spectrum.intensities), dtype=np.float32)


def _stitch_point_r123(
    session: Session, point_id: str, spectrum_type: str = "dark_subtracted"
) -> Optional[np.ndarray]:
    """Fetch R1, R2, R3 for a scan point and return stitched 2148-channel spectrum."""
    from sherloc_pipeline.core.r123_stitching import stitch_r123_spectrum

    parts = {}
    for reg in ("R1", "R2", "R3"):
        sp = (
            session.query(SpectrumORM)
            .filter(
                SpectrumORM.scan_point_id == point_id,
                SpectrumORM.region == reg,
                SpectrumORM.spectrum_type == spectrum_type,
            )
            .first()
        )
        if sp is None:
            return None
        parts[reg] = _extract_intensities(sp)

    try:
        return stitch_r123_spectrum(parts["R1"], parts["R2"], parts["R3"])
    except ValueError:
        return None


def _fetch_stored_masks(session: Session, spectrum_ids):
    """Map ``spectrum_id -> [CosmicRayMask]`` for the given rows (no-op on []).

    Thin wrapper over the shared read path so the route never re-implements
    the query and never imports ``ml_despike``.
    """
    ids = [sid for sid in spectrum_ids if sid]
    if not ids:
        return {}
    return CRMaskService.get_masks_for_spectra(session, ids)


def _stitch_point_r123_despiked(
    session: Session,
    point_id: str,
    spectrum_type: str,
    masks_by_spectrum: dict,
    region_windows,
    interp: str,
):
    """Constituent-first despiked R123 stitch (spec §4.6).

    Each region's full-plane DARK_SUBTRACTED constituent is despiked with
    its own stored mask (certified-window restricted) **before** the overlap
    summation — no mask is ever applied to an already-summed value on the
    serving path. Returns ``(stitched_2148_or_None, applied_abs_channels)``;
    a missing constituent row yields ``(None, [])``.
    """
    from sherloc_pipeline.core.r123_stitching import stitch_r123_spectrum

    parts = {}
    applied: set = set()
    for reg in R123_CONTRIBUTORS:
        sp = (
            session.query(SpectrumORM)
            .filter(
                SpectrumORM.scan_point_id == point_id,
                SpectrumORM.region == reg,
                SpectrumORM.spectrum_type == spectrum_type,
            )
            .first()
        )
        if sp is None:
            return None, []
        arr = _extract_intensities(sp).astype(float)
        mask = _select_stored_mask(masks_by_spectrum.get(sp.id, []))
        if mask is not None:
            lo, hi = region_windows[reg]
            in_window = [
                int(c) for c in mask.channel_indices if lo <= int(c) < hi and int(c) < len(arr)
            ]
            if in_window:
                arr = apply_stored_mask_to_array(
                    arr, mask.channel_indices, region_windows[reg], interp
                )
                applied.update(in_window)
        parts[reg] = arr
    try:
        stitched = stitch_r123_spectrum(parts["R1"], parts["R2"], parts["R3"])
    except ValueError:
        return None, []
    return stitched, sorted(applied)


def _method_of_masks(masks_by_spectrum) -> Optional[str]:
    """Provenance label of the first stored mask in a lookup result."""
    for recs in masks_by_spectrum.values():
        chosen = _select_stored_mask(recs)
        if chosen is not None:
            return chosen.method
    return None


def _r123_missing_regions(session: Session, point_ids, masks_by_spectrum) -> List[str]:
    """Regions lacking a stored mask on **any** involved point (all-or-none).

    A composite R123 view is despiked only if **every requested point** has a
    stored mask for **every** contributing region. A region is "missing" if
    even one involved point lacks a stored mask for it (or lacks the
    constituent row entirely); any missing region makes the whole composite
    render non-despiked, so a partially despiked composite is never labeled
    applied (spec §4.6). For a single-point request this is the per-point
    rule the point endpoint relies on.
    """
    point_ids = list(point_ids)
    have = {pid: {reg: False for reg in R123_CONTRIBUTORS} for pid in point_ids}
    rows = (
        session.query(SpectrumORM.id, SpectrumORM.scan_point_id, SpectrumORM.region)
        .filter(
            SpectrumORM.scan_point_id.in_(point_ids),
            SpectrumORM.region.in_(R123_CONTRIBUTORS),
            SpectrumORM.spectrum_type == PARENT_SPECTRUM_TYPE,
        )
        .all()
    )
    for sp_id, pt_id, reg in rows:
        if pt_id in have and masks_by_spectrum.get(sp_id):
            have[pt_id][reg] = True
    return [
        reg
        for reg in R123_CONTRIBUTORS
        if not all(have[pid][reg] for pid in point_ids)
    ]


def _channels_to_served_positions(
    abs_channels, selected_channels: np.ndarray
) -> List[int]:
    """Map absolute CCD channel indices to positions into the served array.

    ``selected_channels[pos]`` is the absolute channel index served at row
    ``pos`` (``np.where(region_mask)[0]``). This inverts that mapping so a
    despike that replaced absolute channel ``c`` is reported as the served
    position ``pos`` where ``selected_channels[pos] == c`` — the index the
    frontend places a marker at (issue #8). An absolute channel not present in
    the selection (defensive; the collectors only record applied channels that
    *are* in the selection) is skipped. Sorted, de-duplicated.

    For the ``R123`` composite ``selected_channels`` is ``0..2147`` (the full
    stitch is served), so the mapping is the identity and the served positions
    equal the absolute channels.
    """
    row_of_channel = {int(ch): pos for pos, ch in enumerate(selected_channels)}
    positions = {
        row_of_channel[int(c)]
        for c in abs_channels
        if int(c) in row_of_channel
    }
    return sorted(positions)


class _DespikeOutcome:
    """Per-request despike result feeding the additive response fields."""

    def __init__(self):
        self.applied = False
        self.method: Optional[str] = None
        self.masked_channels: List[int] = []
        # Positions into the SERVED array where the ml despike replaced
        # channels (issue #8). Populated alongside masked_channels on the ml
        # path; stays empty for none/modz.
        self.masked_positions: List[int] = []
        self.missing_regions: List[str] = []
        self.n_uncovered = 0


def _collect_single_region(
    session: Session,
    region: str,
    rows,
    mask: np.ndarray,
    selected_channels: np.ndarray,
    do_despike: bool,
    interp: str,
):
    """Collect (point_index, region-selected intensity) for single-region views.

    ``rows`` is an iterable of ``(SpectrumORM, point_index)``. With
    ``do_despike`` each point's served array is despiked with its own stored
    mask **before** the caller normalizes/averages (spec §4.6). A stored
    empty mask still marks the point "despiked".
    """
    outcome = _DespikeOutcome()
    indexed: List[tuple] = []
    applied_union: set = set()
    found = False
    masks_by_spectrum = {}
    window = None
    if do_despike:
        masks_by_spectrum = _fetch_stored_masks(session, [sp.id for sp, _ in rows])
        window = _despike_region_windows()[region]
    for sp, pt_idx in rows:
        raw = _extract_intensities(sp)
        if len(raw) < len(mask):
            continue
        sel = raw[mask]
        if do_despike:
            stored = _select_stored_mask(masks_by_spectrum.get(sp.id))
            if stored is not None:
                found = True
                if outcome.method is None:
                    outcome.method = stored.method
                sel, applied = _despike_region_array(
                    sel, selected_channels, stored.channel_indices, window, interp
                )
                applied_union.update(applied)
        indexed.append((pt_idx, sel))
    outcome.applied = do_despike and found
    outcome.masked_channels = sorted(applied_union)
    outcome.masked_positions = _channels_to_served_positions(
        applied_union, selected_channels
    )
    return indexed, outcome


def _collect_r123(
    session: Session,
    points,
    spec_type: str,
    mask: np.ndarray,
    do_despike: bool,
    interp: str,
):
    """Collect (point_index, region-windowed R123) for the composite view.

    ``points`` is a list of ``ScanPointORM``. With ``do_despike`` the view is
    despiked constituent-first and all-or-none: if any contributing region
    lacks a stored mask across the involved points, the whole composite
    renders non-despiked with ``despike_missing_regions`` populated
    (spec §4.6).
    """
    outcome = _DespikeOutcome()
    outcome.n_uncovered = (
        count_uncovered_contributor_channels("r123_summation", _despike_region_windows())
        if do_despike
        else 0
    )
    indexed: List[tuple] = []
    applied_union: set = set()

    if do_despike:
        region_windows = _despike_region_windows()
        point_ids = [pt.id for pt in points]
        constituent_ids = [
            sp_id
            for (sp_id,) in session.query(SpectrumORM.id).filter(
                SpectrumORM.scan_point_id.in_(point_ids),
                SpectrumORM.region.in_(R123_CONTRIBUTORS),
                SpectrumORM.spectrum_type == spec_type,
            )
        ]
        masks_by_spectrum = _fetch_stored_masks(session, constituent_ids)
        outcome.missing_regions = _r123_missing_regions(session, point_ids, masks_by_spectrum)
        if not outcome.missing_regions:
            outcome.method = _method_of_masks(masks_by_spectrum)
            for pt in points:
                stitched, applied = _stitch_point_r123_despiked(
                    session, pt.id, spec_type, masks_by_spectrum, region_windows, interp
                )
                if stitched is not None:
                    indexed.append((pt.point_index, stitched[mask]))
                    applied_union.update(applied)
            outcome.applied = len(indexed) > 0
            outcome.masked_channels = sorted(applied_union)
            # R123 serves the full 2148-channel stitch (all-True mask), a
            # position-preserving copy/sum, so served position == absolute
            # channel: positions and channels coincide (issue #8).
            outcome.masked_positions = _channels_to_served_positions(
                applied_union, np.where(mask)[0]
            )
            return indexed, outcome

    # Non-despiked composite (despike off, or an all-or-none miss).
    for pt in points:
        stitched = _stitch_point_r123(session, pt.id, spec_type)
        if stitched is not None:
            indexed.append((pt.point_index, stitched[mask]))
    return indexed, outcome


def _wavelength_filter_info(region: str) -> Optional[dict]:
    """Return wavelength filter info for provenance."""
    bounds = {"R1": (250.0, 282.0), "R2": (282.0, 337.8), "R3": (337.8, 357.4)}
    if region in bounds:
        lo, hi = bounds[region]
        return {"min_nm": lo, "max_nm": hi}
    return None


def _apply_laser_normalization(
    session: Session,
    scan_id: str,
    intensities_by_point: list[tuple[int, np.ndarray]],
) -> tuple[list[np.ndarray], bool, Optional[float]]:
    """Apply on-the-fly laser normalization to Loupe dark-subtracted spectra.

    Uses photodiode_mean from scan_points: norm = spectrum × max(pd) / pd[i].

    Args:
        session: DB session.
        scan_id: Scan UUID.
        intensities_by_point: List of (point_index, intensity_array) tuples.

    Returns:
        (normalized_arrays, was_applied, max_photodiode)
    """
    if not intensities_by_point:
        return [], False, None

    # Fetch photodiode values for all points in this scan
    points = (
        session.query(ScanPointORM.point_index, ScanPointORM.photodiode_mean)
        .filter(ScanPointORM.scan_id == scan_id)
        .all()
    )
    pd_map = {p.point_index: p.photodiode_mean for p in points if p.photodiode_mean is not None}

    if not pd_map:
        # No photodiode data — return unnormalized
        return [arr for _, arr in intensities_by_point], False, None

    max_pd = max(pd_map.values())
    if max_pd <= 0:
        return [arr for _, arr in intensities_by_point], False, None

    normalized = []
    for pt_idx, arr in intensities_by_point:
        pd_val = pd_map.get(pt_idx)
        if pd_val and pd_val > 0:
            normalized.append(arr * (max_pd / pd_val))
        else:
            normalized.append(arr)  # No photodiode for this point, pass through

    return normalized, True, max_pd


def _compute_average(
    stacked: np.ndarray,
    method: str,
    trim_pct_override: Optional[float],
    config,
) -> tuple:
    """Compute averaged spectrum using the requested method.

    Returns (avg_intensity, effective_pct, m_trimmed, method_used).
    """
    n_points = stacked.shape[0]

    if method == "mean":
        return stacked.mean(axis=0), 0.0, 0, "mean"

    if method == "median":
        return np.median(stacked, axis=0), 0.0, 0, "median"

    # trim_mean (default)
    if trim_pct_override is not None:
        trim_pct = trim_pct_override
    else:
        trim_pct = config.preprocessing.get("trim_mean_baseline_pct", 0.02)

    if n_points >= 3:
        effective_pct = max(trim_pct, (1 + 1e-9) / n_points)
        m_trimmed = int(np.floor(n_points * effective_pct))
        if m_trimmed < 1:
            m_trimmed = 1
            effective_pct = m_trimmed / n_points
    else:
        effective_pct = 0.0
        m_trimmed = 0

    if m_trimmed > 0 and n_points > 2 * m_trimmed:
        sorted_stack = np.sort(stacked, axis=0)
        trimmed = sorted_stack[m_trimmed : n_points - m_trimmed]
        avg_intensity = trimmed.mean(axis=0)
    else:
        avg_intensity = stacked.mean(axis=0)

    return avg_intensity, round(effective_pct, 6), m_trimmed, "trim_mean"


@lru_cache(maxsize=len(VALID_REGIONS))
def _badpix_items_for_region(region: str) -> tuple:
    """Curated known-noisy channels mapped to a region's served positions.

    Issue #9 annotation layer. The curated table (``core.badpix``) carries
    ABSOLUTE CCD channel indices; this maps each into the position INTO THE
    SERVED ARRAY for the requested region view (the same coordinate system as
    the issue-#8 ``masked_positions`` — reusing ``_channels_to_served_positions``
    keeps the marker placement identical to the ML spike markers). A channel
    outside the region's served selection is dropped (e.g. an R2 channel for a
    ``region=R1`` view). For ``R123`` the served array is the full
    2148-channel stitch, so position == channel (identity).

    Static + cached per region: the asset never changes at runtime, so this is
    computed once per region for the process lifetime. Returns a tuple of
    ``BadpixChannelItem`` (hashable cache value; the route copies into a list).
    """
    _, _, mask = _get_wavelength_wavenumber(region)
    selected_channels = np.where(mask)[0]
    # Position INTO the served array for each absolute channel; channels not in
    # this region's selection get no position (dropped).
    row_of_channel = {int(ch): pos for pos, ch in enumerate(selected_channels)}
    items = []
    for rec in load_badpix_channels():
        pos = row_of_channel.get(int(rec.channel))
        if pos is None:
            continue
        items.append(
            BadpixChannelItem(
                position=pos,
                channel=rec.channel,
                tier=rec.tier,
                source=rec.source,
            )
        )
    items.sort(key=lambda it: it.position)
    return tuple(items)


@router.get("/spectra/badpix", response_model=BadpixResponse)
def get_badpix_channels(region: str = Query("R1")) -> BadpixResponse:
    """Known-noisy detector channels for a region view (issue #9 annotation).

    Returns the curated known-noisy channels mapped to positions INTO the
    served array for ``region`` (``R1 | R2 | R3 | R123``; R123 = full-stitch
    identity). This is a STATIC list — no scan, no DB, no despike interaction
    whatsoever. It is an annotation surface only: the analyst toggles it on to
    see "known-noisy channel" markers, and it never masks or alters any
    spectral value (CR despiking is wholly separate, by design). The list was
    rebuilt 2026-06-17 on a dark-plane veto: a real defect must fire with the
    laser off, so active-only Raman/fluorescence bands (e.g. the carbonate nu1
    apex at channel 137) are NOT flagged.

    Each item is ``{position, channel, tier, source}``: ``position`` indexes
    the served wavenumber/intensity arrays for this region; ``channel`` is the
    absolute CCD index; ``tier`` 1 = CR-confusable elevated-noise, 2 = stable
    hot; ``source`` ∈ ``dark_veto | jb25 | both``.

    Raises:
        HTTPException(400): if ``region`` is not one of the valid regions
            (matches the spectra endpoints' invalid-region contract).
    """
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
    _, _, mask = _get_wavelength_wavenumber(region)
    return BadpixResponse(
        region=region,
        n_channels=int(mask.sum()),
        badpix=list(_badpix_items_for_region(region)),
    )


@router.get("/spectra/{scan_id}/average", response_model=AverageSpectrumResponse)
def get_average_spectrum(
    request: Request,
    scan_id: str,
    region: str = Query("R1"),
    baseline_corrected: bool = Query(False),
    averaging_method: str = Query("trim_mean"),
    trim_pct: Optional[float] = Query(None),
    despike: bool = Query(False),
    despike_method: Optional[str] = Query(None),
) -> AverageSpectrumResponse:
    """Retrieve the averaged spectrum for a scan.

    Despike is selected by the coarse ``despike_method`` param
    (``none | ml | modz``); when absent the legacy ``despike`` bool maps
    ``false -> none``, ``true -> ml`` (issue #6).

    - ``ml`` — each point's stored cosmic-ray mask is applied before
      averaging (no inference; spec §4.6).
    - ``modz`` — the legacy rolling-median modified-z-score despike is run on
      the **averaged** R1 array (display-level, matching the legacy Workbench
      client step; differs from the CLI's per-spectrum pre-fit modz). R1
      only: any other region is served non-despiked with the un-covered
      region named in ``despike_missing_regions``.
    """
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

    method = _resolve_despike_method(despike, despike_method)

    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    # Compute calibration
    wavelength, wavenumber, mask = _get_wavelength_wavenumber(region)
    wn_masked = wavenumber[mask]
    selected_channels = np.where(mask)[0]

    spec_type = _resolve_spectrum_type(scan)
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"

    # Stored ML masks live on DARK_SUBTRACTED rows only. modz runs live on the
    # served array (after averaging) and is handled separately below.
    do_despike = method == "ml" and spec_type == PARENT_SPECTRUM_TYPE
    interp = _despike_interpolation_method(request.app.state.config)

    # Collect (point_index, intensity) pairs for normalization
    if region == "R123":
        points = (
            session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == scan_id)
            .order_by(ScanPointORM.point_index)
            .all()
        )
        indexed_intensities, despike_outcome = _collect_r123(
            session, points, spec_type, mask, do_despike, interp
        )
    else:
        spectra = (
            session.query(SpectrumORM, ScanPointORM.point_index)
            .join(ScanPointORM)
            .filter(
                ScanPointORM.scan_id == scan_id,
                SpectrumORM.region == region,
                SpectrumORM.spectrum_type == spec_type,
            )
            .all()
        )
        indexed_intensities, despike_outcome = _collect_single_region(
            session, region, spectra, mask, selected_channels, do_despike, interp
        )

    if not indexed_intensities:
        raise HTTPException(status_code=404, detail="No valid spectra found")

    # Apply laser normalization for Loupe dark-subtracted spectra
    normalization_applied = False
    if is_loupe and spec_type == "dark_subtracted":
        all_intensities, normalization_applied, _ = _apply_laser_normalization(
            session, scan_id, indexed_intensities,
        )
    else:
        # PDS data is pre-normalized
        all_intensities = [arr for _, arr in indexed_intensities]
        normalization_applied = spec_type == "laser_normalized"

    n_points = len(all_intensities)
    stacked = np.stack(all_intensities)

    avg_intensity, effective_pct, m_trimmed, method_used = _compute_average(
        stacked, averaging_method, trim_pct, request.app.state.config,
    )

    # Live modz despike on the SERVED (averaged) array (issue #6). R1 only;
    # runs before the optional baseline correction (despike-then-baseline).
    modz_params_used: Optional[dict] = None
    n_masked_channels = len(despike_outcome.masked_channels)
    if method == "modz":
        if region in MODZ_REGIONS:
            modz_params = _modz_params_from_config(request.app.state.config)
            avg_intensity, n_masked_channels = _apply_modz_to_served_r1(
                avg_intensity, wn_masked, modz_params
            )
            despike_outcome.applied = True
            despike_outcome.method = "modz"
            despike_outcome.missing_regions = []
            modz_params_used = _modz_params_dict(modz_params)
        else:
            # modz is R1-only: serve non-despiked, disclosing the region.
            despike_outcome.applied = False
            despike_outcome.method = None
            n_masked_channels = 0
            despike_outcome.missing_regions = (
                list(R123_CONTRIBUTORS) if region == "R123" else [region]
            )

    # Optional baseline correction
    actually_corrected = False
    if baseline_corrected:
        try:
            import pandas as pd

            from sherloc_pipeline.core.baseline import BaselineParams, fit_baseline

            params = BaselineParams()
            series = pd.Series(avg_intensity, index=wn_masked)
            corrected, _bl = fit_baseline(series, params)
            avg_intensity = corrected.values
            actually_corrected = True
        except Exception as exc:
            logger.warning("Baseline correction failed for scan %s: %s", scan_id, exc)

    wl_masked = wavelength[mask]

    return AverageSpectrumResponse(
        scan_id=scan_id,
        region=region,
        n_points_averaged=n_points,
        effective_trim_pct_per_tail=round(effective_pct, 6),
        m_trimmed_per_tail=m_trimmed,
        baseline_corrected=actually_corrected,
        laser_normalized=normalization_applied,
        wavenumber=numpy_to_list(wn_masked),
        wavelength=numpy_to_list(wl_masked) if region != "R1" else None,
        intensity=numpy_to_list(avg_intensity),
        n_channels=len(wn_masked),
        provenance=ProvenanceInfo(
            averaging_method=method_used,
            wavelength_filter=_wavelength_filter_info(region),
        ),
        despike_applied=despike_outcome.applied,
        despike_method=despike_outcome.method,
        n_masked_channels=n_masked_channels,
        masked_positions=despike_outcome.masked_positions,
        despike_missing_regions=despike_outcome.missing_regions,
        n_uncovered_contributor_channels=despike_outcome.n_uncovered,
        despike_params_used=modz_params_used,
    )


@router.get("/spectra/{scan_id}/point/{idx}", response_model=PointSpectrumResponse)
def get_point_spectrum(
    request: Request,
    scan_id: str,
    idx: int,
    region: str = Query("R1"),
    spectrum_type: Optional[str] = Query(None),
    despike: bool = Query(False),
    despike_method: Optional[str] = Query(None),
) -> PointSpectrumResponse:
    """Retrieve the raw spectrum for a single measurement point.

    Despike is selected by the coarse ``despike_method`` param
    (``none | ml | modz``); when absent the legacy ``despike`` bool maps
    ``false -> none``, ``true -> ml`` (issue #6).

    - ``ml`` — the stored cosmic-ray mask for the served DARK_SUBTRACTED row
      is applied (no inference; spec §4.6). R123 is despiked constituent-first
      and all-or-none.
    - ``modz`` — the legacy rolling-median modified-z-score despike is run
      live on the served R1 array (display-level, matching the legacy
      Workbench client step). R1 only: any other region is served
      non-despiked with the un-covered region named in
      ``despike_missing_regions``.
    """
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

    method = _resolve_despike_method(despike, despike_method)

    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    # Resolve spectrum type from scan data source if not explicitly provided
    spec_type = spectrum_type or _resolve_spectrum_type(scan)

    point = (
        session.query(ScanPointORM)
        .filter(ScanPointORM.scan_id == scan_id, ScanPointORM.point_index == idx)
        .first()
    )
    if point is None:
        raise HTTPException(status_code=404, detail=f"Point index {idx} not found")

    wavelength, wavenumber, mask = _get_wavelength_wavenumber(region)
    selected_channels = np.where(mask)[0]

    # Stored ML masks attach to DARK_SUBTRACTED rows only; the ml despike is a
    # no-op on any other served representation (e.g. PDS laser_normalized).
    # modz runs live on the served R1 array and is handled separately below.
    do_despike = method == "ml" and spec_type == PARENT_SPECTRUM_TYPE
    interp = _despike_interpolation_method(request.app.state.config)
    despike_applied = False
    despike_method_out: Optional[str] = None
    masked_channels: List[int] = []
    # Positions into the served array for the ml red-triangle markers (issue
    # #8). Filled from masked_channels via the region selection just before the
    # response; stays empty on none/modz.
    masked_positions: List[int] = []
    missing_regions: List[str] = []
    modz_params_used: Optional[dict] = None
    n_uncovered = (
        count_uncovered_contributor_channels("r123_summation", _despike_region_windows())
        if method == "ml" and region == "R123"
        else 0
    )

    if region == "R123":
        if do_despike:
            region_windows = _despike_region_windows()
            constituent_ids = [
                sp_id
                for (sp_id,) in session.query(SpectrumORM.id).filter(
                    SpectrumORM.scan_point_id == point.id,
                    SpectrumORM.region.in_(R123_CONTRIBUTORS),
                    SpectrumORM.spectrum_type == spec_type,
                )
            ]
            masks_by_spectrum = _fetch_stored_masks(session, constituent_ids)
            missing_regions = _r123_missing_regions(session, [point.id], masks_by_spectrum)
            if missing_regions:
                stitched = _stitch_point_r123(session, point.id, spec_type)
            else:
                stitched, applied = _stitch_point_r123_despiked(
                    session, point.id, spec_type, masks_by_spectrum, region_windows, interp
                )
                if stitched is not None:
                    despike_applied = True
                    despike_method_out = _method_of_masks(masks_by_spectrum)
                    masked_channels = applied
        else:
            stitched = _stitch_point_r123(session, point.id, spec_type)
        if stitched is None:
            raise HTTPException(status_code=404, detail="Could not stitch R123 for this point")
        intensity = stitched[mask]
    else:
        spectrum = (
            session.query(SpectrumORM)
            .filter(
                SpectrumORM.scan_point_id == point.id,
                SpectrumORM.region == region,
                SpectrumORM.spectrum_type == spec_type,
            )
            .first()
        )
        if spectrum is None:
            raise HTTPException(status_code=404, detail="Spectrum not found for this point")
        raw = _extract_intensities(spectrum)
        fits = len(raw) >= len(mask)
        intensity = raw[mask] if fits else raw
        if do_despike and fits:
            stored = _select_stored_mask(_fetch_stored_masks(session, [spectrum.id]).get(spectrum.id))
            if stored is not None:
                window = _despike_region_windows()[region]
                intensity, masked_channels = _despike_region_array(
                    intensity, selected_channels, stored.channel_indices, window, interp
                )
                despike_applied = True
                despike_method_out = stored.method

    # Served positions of the ml-replaced channels, for the markers (issue
    # #8). Computed before normalization (positions are array-index based, so
    # normalization scaling does not move them). For R123 ``selected_channels``
    # is the full 0..2147 so positions equal absolute channels.
    masked_positions = _channels_to_served_positions(masked_channels, selected_channels)

    # Apply laser normalization for Loupe single-point spectra
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"
    if is_loupe and spec_type == "dark_subtracted":
        normed, was_normed, _ = _apply_laser_normalization(
            session, scan_id, [(idx, intensity)],
        )
        if was_normed:
            intensity = normed[0]

    # Live modz despike on the SERVED array (issue #6). R1 only; runs after
    # normalization, matching the average endpoint's served-array semantics.
    n_masked_channels = len(masked_channels)
    if method == "modz":
        if region in MODZ_REGIONS:
            modz_params = _modz_params_from_config(request.app.state.config)
            intensity, n_masked_channels = _apply_modz_to_served_r1(
                intensity, wavenumber[mask], modz_params
            )
            despike_applied = True
            despike_method_out = "modz"
            masked_channels = []  # modz reports a count, not absolute indices
            masked_positions = []  # markers come from the client's modz mask
            missing_regions = []
            modz_params_used = _modz_params_dict(modz_params)
        else:
            despike_applied = False
            despike_method_out = None
            n_masked_channels = 0
            masked_channels = []
            masked_positions = []
            missing_regions = (
                list(R123_CONTRIBUTORS) if region == "R123" else [region]
            )

    return PointSpectrumResponse(
        scan_id=scan_id,
        point_index=idx,
        region=region,
        spectrum_type=spec_type,
        wavenumber=numpy_to_list(wavenumber[mask]),
        wavelength=numpy_to_list(wavelength[mask]) if region != "R1" else None,
        intensity=numpy_to_list(intensity),
        n_channels=len(intensity),
        photodiode_mean=point.photodiode_mean,
        provenance=ProvenanceInfo(
            wavelength_filter=_wavelength_filter_info(region),
        ),
        despike_applied=despike_applied,
        despike_method=despike_method_out,
        n_masked_channels=n_masked_channels,
        masked_channels=masked_channels,
        masked_positions=masked_positions,
        despike_missing_regions=missing_regions,
        n_uncovered_contributor_channels=n_uncovered,
        despike_params_used=modz_params_used,
    )


@router.post("/spectra/{scan_id}/subset", response_model=SubsetResponse)
def get_subset_average(
    request: Request,
    scan_id: str,
    body: SubsetRequest,
) -> SubsetResponse:
    """Compute a trim-mean average over a subset of point indices.

    Despike is selected by the coarse ``despike_method`` body field
    (``none | ml | modz``); when absent the legacy ``despike`` bool maps
    ``false -> none``, ``true -> ml`` (issue #6). ``ml`` applies stored masks
    per point before averaging; ``modz`` runs the legacy despike on the
    averaged R1 array (R1 only; other regions served non-despiked with the
    un-covered region disclosed).
    """
    region = body.region
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

    # The body field is a pydantic Literal already 422'd; this also folds in
    # the legacy ``despike`` bool precedence.
    method = _resolve_despike_method(body.despike, body.despike_method)

    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    # Validate indices
    for idx in body.point_indices:
        if idx < 0 or idx >= scan.n_points:
            raise HTTPException(
                status_code=400, detail=f"Point index {idx} out of range [0, {scan.n_points})"
            )

    wavelength, wavenumber, mask = _get_wavelength_wavenumber(region)
    wn_masked = wavenumber[mask]
    selected_channels = np.where(mask)[0]

    spec_type = _resolve_spectrum_type(scan)
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"

    do_despike = method == "ml" and spec_type == PARENT_SPECTRUM_TYPE
    interp = _despike_interpolation_method(request.app.state.config)

    # Resolve the requested points (preserving the per-index 400 contract).
    points = []
    for idx in body.point_indices:
        point = (
            session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == scan_id, ScanPointORM.point_index == idx)
            .first()
        )
        if point is None:
            raise HTTPException(status_code=400, detail=f"Point index {idx} not found")
        points.append(point)

    if region == "R123":
        indexed_intensities, despike_outcome = _collect_r123(
            session, points, spec_type, mask, do_despike, interp
        )
    else:
        rows = []
        for point in points:
            spectrum = (
                session.query(SpectrumORM)
                .filter(
                    SpectrumORM.scan_point_id == point.id,
                    SpectrumORM.region == region,
                    SpectrumORM.spectrum_type == spec_type,
                )
                .first()
            )
            if spectrum is not None:
                rows.append((spectrum, point.point_index))
        indexed_intensities, despike_outcome = _collect_single_region(
            session, region, rows, mask, selected_channels, do_despike, interp
        )

    if not indexed_intensities:
        raise HTTPException(status_code=404, detail="No valid spectra found for subset")

    # Apply laser normalization for Loupe spectra
    if is_loupe and spec_type == "dark_subtracted":
        all_intensities, _, _ = _apply_laser_normalization(
            session, scan_id, indexed_intensities,
        )
    else:
        all_intensities = [arr for _, arr in indexed_intensities]

    n_points = len(all_intensities)
    stacked = np.stack(all_intensities)

    avg_intensity, effective_pct, m_trimmed, method_used = _compute_average(
        stacked, body.averaging_method, body.trim_pct, request.app.state.config,
    )

    # Live modz despike on the SERVED (averaged) array (issue #6). R1 only.
    modz_params_used: Optional[dict] = None
    n_masked_channels = len(despike_outcome.masked_channels)
    if method == "modz":
        if region in MODZ_REGIONS:
            modz_params = _modz_params_from_config(request.app.state.config)
            avg_intensity, n_masked_channels = _apply_modz_to_served_r1(
                avg_intensity, wn_masked, modz_params
            )
            despike_outcome.applied = True
            despike_outcome.method = "modz"
            despike_outcome.missing_regions = []
            modz_params_used = _modz_params_dict(modz_params)
        else:
            despike_outcome.applied = False
            despike_outcome.method = None
            n_masked_channels = 0
            despike_outcome.missing_regions = (
                list(R123_CONTRIBUTORS) if region == "R123" else [region]
            )

    wl_masked = wavelength[mask]

    return SubsetResponse(
        scan_id=scan_id,
        region=region,
        n_points_averaged=n_points,
        point_indices=body.point_indices,
        effective_trim_pct_per_tail=round(effective_pct, 6),
        m_trimmed_per_tail=m_trimmed,
        wavenumber=numpy_to_list(wn_masked),
        wavelength=numpy_to_list(wl_masked) if region != "R1" else None,
        intensity=numpy_to_list(avg_intensity),
        n_channels=len(wn_masked),
        provenance=ProvenanceInfo(
            averaging_method=method_used,
            wavelength_filter=_wavelength_filter_info(region),
        ),
        despike_applied=despike_outcome.applied,
        despike_method=despike_outcome.method,
        n_masked_channels=n_masked_channels,
        masked_positions=despike_outcome.masked_positions,
        despike_missing_regions=despike_outcome.missing_regions,
        n_uncovered_contributor_channels=despike_outcome.n_uncovered,
        despike_params_used=modz_params_used,
    )
