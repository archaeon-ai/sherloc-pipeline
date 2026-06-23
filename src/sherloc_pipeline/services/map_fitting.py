"""Map Mode fitting service -- thin orchestration wrapper.

Runs per-point spectral fitting across selected domains with
per-point callback for WebSocket streaming. Does NOT rewrite
fitting logic -- wraps existing core functions.

E1 design: sequential per-point fitting (no ProcessPoolExecutor).
Sequential is fine for up to ~200 points at ~0.3s/point = ~60 seconds.
Parallel execution can be added in a later phase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from sherloc_pipeline.core.baseline import BaselineParams, fit_baseline
from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from sherloc_pipeline.core.fitting import fit_spectrum
from sherloc_pipeline.core.mineral_id import (
    DEFAULT_RULES,
    MineralRule,
    assign_min_id,
    classify_hydration_band,
    classify_organic_band,
)
from sherloc_pipeline.core.preprocessing import (
    DespikeParams,
    build_weight_vector_from_windows,
    despike_r1_spectrum,
)
from sherloc_pipeline.database.models import ScanPointORM, SpectrumORM

logger = logging.getLogger(__name__)

# Minimum SNR to classify a peak as "measured" (matches map.py _SNR_THRESHOLD)
_SNR_THRESHOLD = 3.0

# How often to emit progress callbacks (every N points or this many seconds)
_PROGRESS_EVERY_N_POINTS = 5
_PROGRESS_EVERY_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DomainResult:
    """Per-domain fitting result for a single scan point."""

    status: str  # "measured" | "below_threshold" | "missing"
    peaks: list[dict[str, Any]] = field(default_factory=list)
    # Each peak dict contains:
    #   For Raman domains: center_cm1, snr, assignment, fwhm_cm1, amplitude, area
    #   For fluorescence: center_nm, snr, assignment, fwhm_nm, amplitude, area


@dataclass
class PointFitResult:
    """Fitting results for a single scan point across all requested domains."""

    point_index: int
    x: float  # ACI pixel coordinate
    y: float  # ACI pixel coordinate
    results: dict[str, DomainResult]  # domain name -> DomainResult


@dataclass
class MapFitSummary:
    """Summary statistics for a completed map fitting run."""

    total_points: int
    detections: dict[str, int]  # domain -> count of points with detections
    elapsed_s: float


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def compute_fit_signature(
    scan_id: str,
    domains: list[str],
    point_indices: list[int] | None,
    profile_id: str | None,
) -> str:
    """Canonical hash for deduplication of map fitting jobs."""
    key = json.dumps(
        {
            "scan_id": scan_id,
            "domains": sorted(domains),
            "points": sorted(point_indices) if point_indices else "all",
            "profile": profile_id or "default",
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Spectrum loading helper
# ---------------------------------------------------------------------------


def _load_point_spectra(
    session: Session,
    scan_point_ids: list[str],
    region: str = "R1",
) -> dict[str, np.ndarray]:
    """Load spectra for scan points from DB.

    Returns:
        {scan_point_id: intensity_array} for each point that has a spectrum.
    """
    spectra = (
        session.query(SpectrumORM)
        .filter(
            SpectrumORM.scan_point_id.in_(scan_point_ids),
            SpectrumORM.region == region,
            SpectrumORM.spectrum_type == "dark_subtracted",
        )
        .all()
    )
    result = {}
    for s in spectra:
        data = np.frombuffer(zlib.decompress(s.intensities), dtype=np.float32)
        result[s.scan_point_id] = data
    return result


# ---------------------------------------------------------------------------
# Domain fitting helpers
# ---------------------------------------------------------------------------


def _fit_raman_domain(
    wavenumber_r1: np.ndarray,
    intensity_r1: np.ndarray,
    config: dict,
    domain: str,
    *,
    baseline_params: BaselineParams | None = None,
    baseline_weights: np.ndarray | None = None,
) -> DomainResult:
    """Fit a single Raman domain (minerals, organics, or hydration) for one point.

    Args:
        wavenumber_r1: Calibrated wavenumber array (R1 region only, 523 channels).
        intensity_r1: Dark-subtracted intensity array (R1 region only).
        config: Full pipeline config dict.
        domain: One of "minerals", "organics", "hydration".
        baseline_params: asPLS baseline parameters (or None to skip).
        baseline_weights: Weight vector for asPLS (protect peak windows).

    Returns:
        DomainResult with fitted peaks or "missing"/"below_threshold" status.
    """
    fitting_cfg = config.get("fitting", {})

    # Despike the R1 intensity
    try:
        despike_params = DespikeParams()
        series = pd.Series(intensity_r1, index=wavenumber_r1)
        despiked, _ = despike_r1_spectrum(series, despike_params, raman_shift=wavenumber_r1)
        y = despiked.values.astype(np.float64)
    except Exception:
        # If despiking fails, use raw intensity
        y = intensity_r1.astype(np.float64)

    x = wavenumber_r1.astype(np.float64)

    # Baseline correction: asPLS removes broad spectral background
    if baseline_params is not None:
        try:
            bl_series = pd.Series(y, index=x)
            corrected, _ = fit_baseline(bl_series, baseline_params, weights=baseline_weights)
            y = corrected.values.astype(np.float64)
        except Exception:
            logger.debug("Baseline correction failed for domain=%s, using despiked data", domain)

    # Build domain-specific config
    if domain == "minerals":
        roi = tuple(fitting_cfg.get("r1_fit_range", [700, 1200]))
        cfg = {
            "r1_fit_range": list(roi),
            "fit_fwhm_min_initial_cm1": fitting_cfg.get("fit_fwhm_min_initial_cm1", 22),
            "filter_fwhm_min_cm1": fitting_cfg.get("filter_fwhm_min_cm1", 30),
            "fwhm_max_cm1": fitting_cfg.get("fwhm_max_cm1", 90),
            "slit_width_cm1_default": fitting_cfg.get("slit_width_cm1_default", 34.1),
            "slit_pref_weight": fitting_cfg.get("slit_pref_weight", 0.2),
            "low_fwhm_edge_penalty": fitting_cfg.get("low_fwhm_edge_penalty", 0.1),
            "max_peaks": fitting_cfg.get("max_peaks", 5),
            "peak_separation_cm1": fitting_cfg.get("peak_separation_cm1", 25),
            "r_squared_min": fitting_cfg.get("r_squared_min", 0.25),
            "min_snr": fitting_cfg.get("min_snr", 3.0),
            "min_seed_snr": fitting_cfg.get("min_seed_snr", 2.0),
            "min_display_snr": fitting_cfg.get("min_display_snr", 2.0),
            "min_amp_sigma_multiplier": fitting_cfg.get("min_amp_sigma_multiplier", 0.3),
            "noise_estimation": fitting_cfg.get("noise_estimation", {"window": [2000.0, 2100.0]}),
            "parsimony": fitting_cfg.get("parsimony", {}),
            "posthoc_filters": fitting_cfg.get("posthoc_filters", {}),
        }
        assign_fn = lambda center: assign_min_id(center, DEFAULT_RULES)

    elif domain == "organics":
        roi = tuple(fitting_cfg.get("organics_fit_range", [1250, 1850]))
        fwhm_bounds = fitting_cfg.get("organics_fwhm_bounds", [40, 200])
        cfg = {
            "r1_fit_range": list(roi),
            "fit_fwhm_min_initial_cm1": fwhm_bounds[0],
            "filter_fwhm_min_cm1": fwhm_bounds[0],
            "fwhm_max_cm1": fwhm_bounds[1],
            "slit_width_cm1_default": fitting_cfg.get("slit_width_cm1_default", 34.1),
            "slit_pref_weight": 0.0,  # No slit preference for organics
            "low_fwhm_edge_penalty": 0.0,
            "max_peaks": fitting_cfg.get("organics_max_peaks", 2),
            "peak_separation_cm1": fitting_cfg.get("peak_separation_cm1", 25),
            "r_squared_min": fitting_cfg.get("r_squared_min", 0.25),
            "min_snr": fitting_cfg.get("organics_min_snr", 2.0),
            "min_seed_snr": fitting_cfg.get("organics_min_snr", 2.0),
            "min_display_snr": fitting_cfg.get("min_display_snr", 2.0),
            "min_amp_sigma_multiplier": fitting_cfg.get("min_amp_sigma_multiplier", 0.3),
            "noise_estimation": fitting_cfg.get("noise_estimation", {"window": [2000.0, 2100.0]}),
            "parsimony": fitting_cfg.get("parsimony", {}),
            "posthoc_filters": fitting_cfg.get("posthoc_filters", {}),
        }
        assign_fn = classify_organic_band

    elif domain == "hydration":
        roi = tuple(fitting_cfg.get("hydration_fit_range", [2800, 3900]))
        cfg = {
            "r1_fit_range": list(roi),
            "fit_fwhm_min_initial_cm1": fitting_cfg.get("hydration_fwhm_min_cm1", 50),
            "filter_fwhm_min_cm1": fitting_cfg.get("hydration_fwhm_min_cm1", 50),
            "fwhm_max_cm1": fitting_cfg.get("hydration_fwhm_max_cm1", 300),
            "slit_width_cm1_default": fitting_cfg.get("slit_width_cm1_default", 34.1),
            "slit_pref_weight": 0.0,  # No slit preference for hydration
            "low_fwhm_edge_penalty": 0.0,
            "max_peaks": fitting_cfg.get("hydration_max_peaks", 2),
            "peak_separation_cm1": fitting_cfg.get("peak_separation_cm1", 25),
            "r_squared_min": fitting_cfg.get("hydration_r2_min", 0.25),
            "min_snr": fitting_cfg.get("hydration_min_snr", 3.0),
            "min_seed_snr": fitting_cfg.get("hydration_min_snr", 3.0),
            "min_display_snr": fitting_cfg.get("min_display_snr", 2.0),
            "min_amp_sigma_multiplier": fitting_cfg.get("min_amp_sigma_multiplier", 0.3),
            "noise_estimation": fitting_cfg.get("noise_estimation", {"window": [2000.0, 2100.0]}),
            "parsimony": {
                "model_selection": "ftest",
                "ftest_alpha": fitting_cfg.get("hydration_ftest_alpha", 0.01),
            },
            "posthoc_filters": fitting_cfg.get("posthoc_filters", {}),
        }
        assign_fn = classify_hydration_band

    else:
        return DomainResult(status="missing")

    # Run fitting
    try:
        fit_result, _ = fit_spectrum(x, y, cfg, roi=roi)
    except Exception as exc:
        logger.debug("Raman fit failed for domain=%s: %s", domain, exc)
        return DomainResult(status="missing")

    if not fit_result.peaks:
        return DomainResult(status="below_threshold")

    # Convert peaks to dicts with assignments
    peaks = []
    has_detection = False
    for p in fit_result.peaks:
        if not (p.pass_snr and p.pass_fwhm):
            continue
        assignment = assign_fn(p.m_cm1)
        peak_dict = {
            "center_cm1": round(p.m_cm1, 2),
            "snr": round(p.snr, 2),
            "assignment": assignment,
            "fwhm_cm1": round(p.fwhm, 2),
            "amplitude": round(p.a, 2),
            "area": round(p.area, 2),
            "r2": round(fit_result.r2, 4),
        }
        peaks.append(peak_dict)
        if p.snr >= _SNR_THRESHOLD:
            has_detection = True

    if not peaks:
        return DomainResult(status="below_threshold")

    status = "measured" if has_detection else "below_threshold"
    return DomainResult(status=status, peaks=peaks)


def _fit_fluorescence_domain(
    session: Session,
    scan_point_id: str,
    config: dict,
) -> DomainResult:
    """Fit fluorescence for a single scan point using R2+R3 spectra.

    Args:
        session: DB session for loading R2/R3 spectra.
        scan_point_id: UUID of the scan point.
        config: Full pipeline config dict.

    Returns:
        DomainResult with fitted fluorescence peaks.
    """
    from sherloc_pipeline.core.fluor_fitting import fit_fluorescence_spectrum
    from sherloc_pipeline.core.fluor_id import assign_fluor_group
    from sherloc_pipeline.core.r123_stitching import stitch_r123_spectrum

    fluor_cfg = config.get("fluorescence_fitting", {})

    # Load R1, R2, R3 and stitch
    parts = {}
    for region in ("R1", "R2", "R3"):
        sp = (
            session.query(SpectrumORM)
            .filter(
                SpectrumORM.scan_point_id == scan_point_id,
                SpectrumORM.region == region,
                SpectrumORM.spectrum_type == "dark_subtracted",
            )
            .first()
        )
        if sp is None:
            return DomainResult(status="missing")
        parts[region] = np.frombuffer(zlib.decompress(sp.intensities), dtype=np.float32)

    try:
        stitched = stitch_r123_spectrum(parts["R1"], parts["R2"], parts["R3"])
    except ValueError:
        return DomainResult(status="missing")

    # Get calibration for full 2148-channel spectrum
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)

    # Extract fluorescence range (R2+R3 region)
    # fit_fluorescence_spectrum handles the range internally, but we need
    # the full wavelength and stitched intensity arrays
    try:
        result = fit_fluorescence_spectrum(
            wavelength=wavelength,
            intensity=stitched.astype(np.float64),
            fit_range=tuple(fluor_cfg.get("fit_range", [276.0, 357.0])),
            position_bounds=tuple(fluor_cfg.get("position_bounds", [270.0, 357.0])),
            fwhm_range=tuple(fluor_cfg.get("fwhm_range", [7.0, 35.0])),
            min_peak_separation=fluor_cfg.get("min_peak_separation", 15.0),
            max_peaks=fluor_cfg.get("max_peaks", 4),
            snr_threshold=fluor_cfg.get("snr_threshold", 10.0),
            min_fwhm_nm=fluor_cfg.get("min_fwhm_nm", 8.0),
            noise_window=tuple(fluor_cfg.get("noise_window", [261.5, 262.3])),
            saturation_threshold=fluor_cfg.get("saturation_threshold", 60000.0),
            saturation_channel_limit=fluor_cfg.get("saturation_channel_limit", 5),
            overlap_exclusion=tuple(fluor_cfg.get("overlap_exclusion", [337.4, 338.4])),
        )
    except Exception as exc:
        logger.debug("Fluorescence fit failed: %s", exc)
        return DomainResult(status="missing")

    if not result.peaks or result.fit_skipped:
        return DomainResult(status="below_threshold")

    peaks = []
    has_detection = False
    for p in result.peaks:
        group = assign_fluor_group(p.center_nm)
        peak_dict = {
            "center_nm": round(p.center_nm, 2),
            "snr": round(p.snr, 2),
            "assignment": group,
            "fwhm_nm": round(p.fwhm_nm, 2),
            "amplitude": round(p.amplitude, 2),
            "area": round(p.area, 2),
        }
        peaks.append(peak_dict)
        has_detection = True  # fluorescence peaks already pass SNR threshold

    if not peaks:
        return DomainResult(status="below_threshold")

    status = "measured" if has_detection else "below_threshold"
    return DomainResult(status=status, peaks=peaks)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class MapFitService:
    """Orchestrates per-point spectral fitting for Map Mode.

    Sequential E1 implementation: fits points one at a time with callbacks
    for real-time WebSocket streaming.
    """

    def __init__(self, config: dict):
        self.config = config

    def run_map_fit(
        self,
        session: Session,
        scan_id: str,
        domains: list[str],
        point_indices: list[int] | None,
        point_coords: dict[int, tuple[float, float]],
        on_point_fitted: Callable[[PointFitResult], None],
        on_progress: Callable[[int, int, float, float], None],
        on_log: Callable[[int, str], None],
        cancel_event: threading.Event,
    ) -> MapFitSummary:
        """Run per-point fitting across the requested domains.

        Args:
            session: SQLAlchemy session (owned by the caller / fitting thread).
            scan_id: UUID of the scan to fit.
            domains: List of domains to fit (minerals, organics, hydration, fluorescence).
            point_indices: Subset of points to fit, or None for all points.
            point_coords: {point_index: (aci_x, aci_y)} from resolve_display_coordinates.
            on_point_fitted: Called after each point with the per-point result.
            on_progress: Called periodically with (fitted, total, elapsed_s, eta_s).
            on_log: Called per-point with (point_index, summary_message).
            cancel_event: Set by the WebSocket handler to request cancellation.

        Returns:
            MapFitSummary with aggregate statistics.
        """
        t_start = time.monotonic()

        # Query scan points (all or subset)
        query = (
            session.query(ScanPointORM)
            .filter(ScanPointORM.scan_id == scan_id)
            .order_by(ScanPointORM.point_index)
        )
        if point_indices is not None:
            query = query.filter(ScanPointORM.point_index.in_(point_indices))
        scan_points = query.all()

        total = len(scan_points)
        if total == 0:
            return MapFitSummary(total_points=0, detections={}, elapsed_s=0.0)

        # Pre-load R1 spectra in batch for efficiency
        raman_domains = [d for d in domains if d in ("minerals", "organics", "hydration")]
        r1_spectra: dict[str, np.ndarray] = {}
        if raman_domains:
            sp_ids = [sp.id for sp in scan_points]
            r1_spectra = _load_point_spectra(session, sp_ids, region="R1")

        # Calibration: compute once
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        r1_mask = get_region_wavelength_mask(wavelength, "R1")
        wavenumber_r1 = wavenumber[r1_mask]

        # --- Preprocessing setup (laser norm, background, baseline) ---
        # Laser normalization: load photodiode_mean per point
        pd_map: dict[int, float] = {}
        max_pd = 0.0
        if raman_domains:
            pd_rows = (
                session.query(ScanPointORM.point_index, ScanPointORM.photodiode_mean)
                .filter(ScanPointORM.scan_id == scan_id)
                .all()
            )
            pd_map = {
                p.point_index: p.photodiode_mean
                for p in pd_rows
                if p.photodiode_mean is not None and p.photodiode_mean > 0
            }
            if pd_map:
                max_pd = max(pd_map.values())
                print(f"[map-fit] Laser normalization: {len(pd_map)} points, max_pd={max_pd:.1f}", flush=True)

        # Baseline correction: asPLS params + weight vector
        bl_params: BaselineParams | None = None
        bl_weights: np.ndarray | None = None
        if raman_domains:
            pre_cfg = self.config.get("preprocessing", {})
            bl_cfg = pre_cfg.get("baseline", {})
            bl_params = BaselineParams(
                lam=bl_cfg.get("lam", 1e6),
                asymmetric_coef=bl_cfg.get("asymmetric_coef", 0.01),
                iters=bl_cfg.get("iters", 10),
                diff_order=bl_cfg.get("diff_order", 2),
                tol=bl_cfg.get("tol", 1e-3),
            )
            keep_windows = bl_cfg.get("keep_windows", [
                [600.0, 1130.0], [1300.0, 1720.0], [3000.0, 3800.0],
            ])
            keep_weight = bl_cfg.get("keep_weight", 0.01)
            bl_weights = build_weight_vector_from_windows(
                wavenumber_r1,
                keep_windows=[tuple(w) for w in keep_windows],
                default_weight=1.0,
                keep_weight=keep_weight,
            )

        # Emit preprocessing summary to terminal + WebSocket log
        if raman_domains:
            prep_parts = []
            if pd_map:
                prep_parts.append(f"laser_norm(max_pd={max_pd:.0f}, {len(pd_map)} pts)")
            if bl_params is not None:
                prep_parts.append(f"baseline(asPLS lam={bl_params.lam:.0e}, iters={bl_params.iters})")
            if prep_parts:
                prep_msg = f"Preprocessing: {', '.join(prep_parts)}"
            else:
                prep_msg = "Preprocessing: none (missing background/photodiode data)"
            print(f"[map-fit] {prep_msg}", flush=True)
            try:
                on_log(-1, prep_msg)
            except Exception:
                pass

        detections: dict[str, int] = {d: 0 for d in domains}
        last_progress_time = t_start
        fitted_count = 0

        for sp in scan_points:
            if cancel_event.is_set():
                logger.info("Map fit cancelled at point %d/%d", fitted_count, total)
                break

            point_index = sp.point_index
            coords = point_coords.get(point_index, (0.0, 0.0))
            domain_results: dict[str, DomainResult] = {}

            # Fit each requested Raman domain
            if sp.id in r1_spectra and raman_domains:
                raw_intensity = r1_spectra[sp.id].copy()

                # Laser normalization: correct for laser power variation
                pd_val = pd_map.get(point_index)
                if pd_val and max_pd > 0:
                    raw_intensity = raw_intensity * (max_pd / pd_val)

                # Apply R1 wavelength mask
                if len(raw_intensity) >= len(r1_mask):
                    intensity_r1 = raw_intensity[r1_mask]
                else:
                    intensity_r1 = raw_intensity

                for domain in raman_domains:
                    try:
                        result = _fit_raman_domain(
                            wavenumber_r1, intensity_r1, self.config, domain,
                            baseline_params=bl_params,
                            baseline_weights=bl_weights,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Raman domain %s failed for point %d: %s",
                            domain,
                            point_index,
                            exc,
                        )
                        result = DomainResult(status="missing")
                    domain_results[domain] = result
                    if result.status == "measured":
                        detections[domain] += 1
            else:
                # No R1 spectrum available
                for domain in raman_domains:
                    domain_results[domain] = DomainResult(status="missing")

            # Fit fluorescence if requested
            if "fluorescence" in domains:
                try:
                    fluor_result = _fit_fluorescence_domain(
                        session, sp.id, self.config
                    )
                except Exception as exc:
                    logger.debug(
                        "Fluorescence failed for point %d: %s", point_index, exc
                    )
                    fluor_result = DomainResult(status="missing")
                domain_results["fluorescence"] = fluor_result
                if fluor_result.status == "measured":
                    detections["fluorescence"] += 1

            # Build result and emit callbacks
            point_result = PointFitResult(
                point_index=point_index,
                x=coords[0],
                y=coords[1],
                results=domain_results,
            )

            try:
                on_point_fitted(point_result)
            except Exception:
                logger.debug("on_point_fitted callback failed for point %d", point_index)

            fitted_count += 1
            elapsed = time.monotonic() - t_start

            # Log summary — show ALL detected peaks, not just the best
            peak_parts = []
            r2_val = None
            for d, dr in domain_results.items():
                if dr.status == "measured" and dr.peaks:
                    for pk in sorted(dr.peaks, key=lambda p: p.get("snr", 0), reverse=True):
                        assignment = pk.get("assignment", "?")
                        center = pk.get("center_cm1") or pk.get("center_nm")
                        snr = pk.get("snr", 0)
                        fwhm = pk.get("fwhm_cm1") or pk.get("fwhm_nm")
                        unit = "cm⁻¹" if pk.get("center_cm1") else "nm"
                        fwhm_str = f" FWHM={fwhm:.1f}" if fwhm else ""
                        peak_parts.append(f"{assignment}({center:.0f}{unit} SNR={snr:.1f}{fwhm_str})")
                        if r2_val is None:
                            r2_val = pk.get("r2")
                elif dr.status == "below_threshold":
                    peak_parts.append(f"{d}:below_thresh")
            r2_str = f" R²={r2_val:.3f}" if r2_val is not None else ""
            if peak_parts:
                log_msg = f"Pt {point_index}/{total}: {', '.join(peak_parts)}{r2_str}"
            else:
                log_msg = f"Pt {point_index}/{total}: no detections"

            try:
                on_log(point_index, log_msg)
            except Exception:
                pass

            # Progress callback (every N points or every 2 seconds)
            now = time.monotonic()
            if (
                fitted_count % _PROGRESS_EVERY_N_POINTS == 0
                or (now - last_progress_time) >= _PROGRESS_EVERY_SECONDS
                or fitted_count == total
            ):
                if fitted_count > 0:
                    rate = fitted_count / elapsed if elapsed > 0 else 0
                    eta = (total - fitted_count) / rate if rate > 0 else 0.0
                else:
                    eta = 0.0

                try:
                    on_progress(fitted_count, total, elapsed, eta)
                except Exception:
                    pass
                last_progress_time = now

        elapsed_total = time.monotonic() - t_start
        return MapFitSummary(
            total_points=fitted_count,
            detections=detections,
            elapsed_s=round(elapsed_total, 2),
        )
