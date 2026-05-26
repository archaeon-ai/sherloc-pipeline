"""Spectrum endpoints: average, single point, subset."""

import logging
import zlib
from typing import List, Optional

logger = logging.getLogger(__name__)

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy.orm import Session

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SpectrumORM
from sherloc_pipeline.web.adapters import numpy_to_list
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.schemas import (
    AverageSpectrumResponse,
    PointSpectrumResponse,
    ProvenanceInfo,
    SubsetRequest,
    SubsetResponse,
)

router = APIRouter(prefix="/api", tags=["spectra"])

VALID_REGIONS = {"R1", "R2", "R3", "R123"}

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


@router.get("/spectra/{scan_id}/average", response_model=AverageSpectrumResponse)
def get_average_spectrum(
    request: Request,
    scan_id: str,
    region: str = Query("R1"),
    baseline_corrected: bool = Query(False),
    averaging_method: str = Query("trim_mean"),
    trim_pct: Optional[float] = Query(None),
) -> AverageSpectrumResponse:
    """Retrieve the averaged spectrum for a scan."""
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    # Compute calibration
    wavelength, wavenumber, mask = _get_wavelength_wavenumber(region)
    wn_masked = wavenumber[mask]

    spec_type = _resolve_spectrum_type(scan)
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"

    # Collect (point_index, intensity) pairs for normalization
    indexed_intensities: list[tuple[int, np.ndarray]] = []
    if region == "R123":
        points = (
            session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == scan_id)
            .order_by(ScanPointORM.point_index)
            .all()
        )
        for pt in points:
            stitched = _stitch_point_r123(session, pt.id, spec_type)
            if stitched is not None:
                indexed_intensities.append((pt.point_index, stitched[mask]))
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
        for sp, pt_idx in spectra:
            raw = _extract_intensities(sp)
            if len(raw) >= len(mask):
                indexed_intensities.append((pt_idx, raw[mask]))

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
    )


@router.get("/spectra/{scan_id}/point/{idx}", response_model=PointSpectrumResponse)
def get_point_spectrum(
    request: Request,
    scan_id: str,
    idx: int,
    region: str = Query("R1"),
    spectrum_type: Optional[str] = Query(None),
) -> PointSpectrumResponse:
    """Retrieve the raw spectrum for a single measurement point."""
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

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

    if region == "R123":
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
        intensity = raw[mask] if len(raw) >= len(mask) else raw

    # Apply laser normalization for Loupe single-point spectra
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"
    if is_loupe and spec_type == "dark_subtracted":
        normed, was_normed, _ = _apply_laser_normalization(
            session, scan_id, [(idx, intensity)],
        )
        if was_normed:
            intensity = normed[0]

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
    )


@router.post("/spectra/{scan_id}/subset", response_model=SubsetResponse)
def get_subset_average(
    request: Request,
    scan_id: str,
    body: SubsetRequest,
) -> SubsetResponse:
    """Compute a trim-mean average over a subset of point indices."""
    region = body.region
    if region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Invalid region: {region}")

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

    spec_type = _resolve_spectrum_type(scan)
    is_loupe = (getattr(scan, "data_source", None) or "loupe") == "loupe"

    # Fetch spectra for requested points (with point index for normalization)
    indexed_intensities: list[tuple[int, np.ndarray]] = []
    for idx in body.point_indices:
        point = (
            session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == scan_id, ScanPointORM.point_index == idx)
            .first()
        )
        if point is None:
            raise HTTPException(status_code=400, detail=f"Point index {idx} not found")
        if region == "R123":
            stitched = _stitch_point_r123(session, point.id, spec_type)
            if stitched is not None:
                indexed_intensities.append((idx, stitched[mask]))
            continue
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
            raw = _extract_intensities(spectrum)
            if len(raw) >= len(mask):
                indexed_intensities.append((idx, raw[mask]))

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
    )
