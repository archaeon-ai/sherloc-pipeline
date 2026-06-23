"""Fluorescence peak fitting for R2/R3 spectral regions.

Uses differential evolution with multi-Gaussian model to fit fluorescence
features in the 276-355 nm range. See docs/specs/FLUORESCENCE_FITTING_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from sherloc_pipeline.core.fitting import (
    _compute_aicc,
    compute_r2,
    fwhm_to_sigma,
    gaussian,
    multi_gaussian,
)


@dataclass
class FluorPeakFit:
    """Single fitted fluorescence peak."""

    center_nm: float
    amplitude: float
    fwhm_nm: float
    area: float
    snr: float


@dataclass
class FluorFitResult:
    """Result of fluorescence spectrum fitting."""

    peaks: List[FluorPeakFit]
    r2: float
    rss: float
    aicc: float
    n_peaks: int
    is_saturated: bool
    fit_skipped: bool
    n_saturated_channels: int
    n_masked_channels: int
    warnings: List[str] = field(default_factory=list)


def _fluor_objective(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_peak_separation: float,
) -> float:
    """Sum-of-squares objective with hard separation constraint.

    Returns inf if any pair of peaks is closer than min_peak_separation,
    otherwise returns sum(residuals^2). This prevents the optimizer from
    stacking Gaussians at the same location to approximate asymmetric peaks.
    """
    n_peaks = params.size // 3
    for i in range(n_peaks):
        for j in range(i + 1, n_peaks):
            if abs(params[i * 3] - params[j * 3]) < min_peak_separation:
                return np.inf

    y_model = multi_gaussian(x, params)
    return float(np.sum((y - y_model) ** 2))


def _fluor_objective_simple(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Sum-of-squares objective without separation constraint.

    Used for hypothesis-driven fitting where per-peak bounds already
    enforce feature separation via constrained center ranges.
    """
    y_model = multi_gaussian(x, params)
    return float(np.sum((y - y_model) ** 2))


def _fluor_objective_ratio_constrained(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    g1a_peak_idx: int,
    g1b_peak_idx: int,
    ratio_range: Tuple[float, float],
) -> float:
    """Sum-of-squares objective with amplitude ratio constraint on Ce3+ doublet.

    Returns inf if amp_1a / amp_1b falls outside ratio_range, enforcing
    physical Ce3+ branching ratio consistency. Used for hypothesis models
    containing both group1a and group1b (M3, M5, M6, M7).
    """
    amp_1a = params[g1a_peak_idx * 3 + 1]
    amp_1b = params[g1b_peak_idx * 3 + 1]
    if amp_1b <= 0:
        return np.inf
    ratio = amp_1a / amp_1b
    if ratio < ratio_range[0] or ratio > ratio_range[1]:
        return np.inf
    y_model = multi_gaussian(x, params)
    return float(np.sum((y - y_model) ** 2))


def _run_de_fit(
    bounds_list: List[Tuple[float, float]],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    min_peak_separation: float = 0.0,
    use_separation_constraint: bool = True,
    ratio_constraint: Optional[Tuple[int, int, Tuple[float, float]]] = None,
) -> Optional[Tuple[np.ndarray, float]]:
    """Run differential evolution and return (best_params, rss) or None.

    Args:
        ratio_constraint: If set, (g1a_peak_idx, g1b_peak_idx, (lo, hi))
            enforces amp_1a/amp_1b within [lo, hi] during optimization.
    """
    if not bounds_list:
        # M0 null model: RSS = sum(y^2), params empty
        rss = float(np.sum(y_fit ** 2))
        return np.array([]), rss

    try:
        if ratio_constraint is not None:
            g1a_idx, g1b_idx, ratio_range = ratio_constraint
            result = differential_evolution(
                _fluor_objective_ratio_constrained,
                bounds=bounds_list,
                args=(x_fit, y_fit, g1a_idx, g1b_idx, ratio_range),
                seed=42,
                maxiter=1000,
                tol=1e-8,
                polish=True,
            )
        elif use_separation_constraint and min_peak_separation > 0:
            result = differential_evolution(
                _fluor_objective,
                bounds=bounds_list,
                args=(x_fit, y_fit, min_peak_separation),
                seed=42,
                maxiter=1000,
                tol=1e-8,
                polish=True,
            )
        else:
            result = differential_evolution(
                _fluor_objective_simple,
                bounds=bounds_list,
                args=(x_fit, y_fit),
                seed=42,
                maxiter=1000,
                tol=1e-8,
                polish=True,
            )
        if not np.isfinite(result.fun):
            return None
        return result.x, result.fun
    except Exception:
        return None


def _agnostic_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    position_bounds: Tuple[float, float],
    fwhm_range: Tuple[float, float],
    min_peak_separation: float,
    max_peaks: int,
    y_max: float,
) -> Tuple[Optional[np.ndarray], float, float]:
    """Agnostic fitting: try 1..max_peaks, select best by AICc.

    Returns (best_params, best_aicc, best_rss).
    """
    best_aicc = float("inf")
    best_rss = float("inf")
    best_params: Optional[np.ndarray] = None
    n_data = x_fit.size

    for n in range(1, max_peaks + 1):
        bounds_list: list = []
        for _ in range(n):
            bounds_list.extend([
                position_bounds,
                (0.0, y_max * 1.5),
                fwhm_range,
            ])

        de_result = _run_de_fit(
            bounds_list, x_fit, y_fit,
            min_peak_separation=min_peak_separation,
            use_separation_constraint=True,
        )
        if de_result is None:
            continue

        params, rss = de_result
        aicc = _compute_aicc(n_data, rss, n * 3)
        if aicc < best_aicc:
            best_aicc = aicc
            best_rss = rss
            best_params = params

    return best_params, best_aicc, best_rss


def _hypothesis_fit(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    noise_std: float,
    y_max: float,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    agnostic_r2_threshold: float = 0.7,
    position_bounds: Tuple[float, float] = (270.0, 357.0),
    fwhm_range: Tuple[float, float] = (7.0, 35.0),
    min_peak_separation: float = 15.0,
    max_peaks: int = 4,
    doublet_ratio_range: Tuple[float, float] = (0.7, 1.3),
) -> Tuple[Optional[np.ndarray], float, float, List[str]]:
    """Hypothesis-driven fitting with feature scan and constrained models.

    Returns (best_params, best_aicc, best_rss, warnings).
    """
    from sherloc_pipeline.core.fluor_detection import (
        enumerate_candidate_models,
        scan_fluorescence_features,
    )

    n_data = x_fit.size
    warnings: List[str] = []

    # Phase 1: Feature scan
    scan = scan_fluorescence_features(wavelength, intensity, noise_std)

    # Phase 2: Enumerate candidate models
    candidates = enumerate_candidate_models(scan, y_max)

    # Phase 3: Fit each candidate, select by AICc
    best_aicc = float("inf")
    best_rss = float("inf")
    best_params: Optional[np.ndarray] = None

    for candidate in candidates:
        if candidate.n_peaks == 0:
            # M0 null model
            rss = float(np.sum(y_fit ** 2))
            aicc = _compute_aicc(n_data, rss, 0)
            if aicc < best_aicc:
                best_aicc = aicc
                best_rss = rss
                best_params = None  # will be handled as no-peak result
            continue

        # Build ratio constraint for models with both group1a + group1b
        ratio_constraint = None
        if "group1a" in candidate.features and "group1b" in candidate.features:
            g1a_idx = candidate.features.index("group1a")
            g1b_idx = candidate.features.index("group1b")
            ratio_constraint = (g1a_idx, g1b_idx, doublet_ratio_range)

        de_result = _run_de_fit(
            candidate.bounds, x_fit, y_fit,
            use_separation_constraint=False,  # bounds enforce separation
            ratio_constraint=ratio_constraint,
        )
        if de_result is None:
            continue

        params, rss = de_result
        aicc = _compute_aicc(n_data, rss, candidate.n_peaks * 3)
        if aicc < best_aicc:
            best_aicc = aicc
            best_rss = rss
            best_params = params

    # Phase 4: Agnostic diagnostic (conditional)
    if best_params is not None:
        y_model = multi_gaussian(x_fit, best_params)
        r2_hypothesis = compute_r2(y_fit, y_model)
    else:
        r2_hypothesis = 0.0

    if r2_hypothesis < agnostic_r2_threshold:
        agnostic_params, agnostic_aicc, agnostic_rss = _agnostic_fit(
            x_fit, y_fit, position_bounds, fwhm_range,
            min_peak_separation, max_peaks, y_max,
        )
        if agnostic_aicc < best_aicc and agnostic_params is not None:
            warnings.append("agnostic_fit_better")
            best_aicc = agnostic_aicc
            best_rss = agnostic_rss
            best_params = agnostic_params

    return best_params, best_aicc, best_rss, warnings


def fit_fluorescence_spectrum(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    fit_range: Tuple[float, float] = (276.0, 357.0),
    position_bounds: Tuple[float, float] = (270.0, 357.0),
    fwhm_range: Tuple[float, float] = (7.0, 35.0),
    min_peak_separation: float = 15.0,
    max_peaks: int = 4,
    snr_threshold: float = 10.0,
    min_fwhm_nm: float = 8.0,
    noise_std: Optional[float] = None,
    noise_window: Tuple[float, float] = (261.5, 262.3),
    saturation_threshold: float = 60000.0,
    saturation_channel_limit: int = 5,
    overlap_exclusion: Tuple[float, float] = (337.4, 338.4),
    strategy: str = "agnostic",
    agnostic_r2_threshold: float = 0.7,
    doublet_ratio_range: Tuple[float, float] = (0.7, 1.3),
) -> FluorFitResult:
    """Fit multi-Gaussian model to fluorescence spectrum.

    Supports two strategies:
    - "agnostic" (default): Discovery mode — tries 1 to max_peaks unconstrained,
      AICc selection. No amplitude ratio constraints; only rejects closely
      overlapping peaks.
    - "hypothesis": Feature scan → constrained candidate models → AICc selection.
      Enforces Ce3+ doublet amplitude ratio constraint for group1a+group1b.
      Falls back to agnostic fit if hypothesis R² < agnostic_r2_threshold.

    No baseline correction is applied. Saturated channels are masked
    or the fit is skipped if saturation is extensive. The R2/R3 overlap
    zone is excluded to avoid artificially doubled intensities.

    An early bail-out check skips the expensive differential-evolution loop
    when the maximum feature prominence (max − median of despiked data) is
    below ``snr_threshold × noise_std``. Because fluorescence data has no
    baseline subtraction, the median is used to remove the DC offset before
    comparing against the threshold. This is mathematically sound: if the
    tallest feature above the local level cannot exceed the SNR gate, no
    fitted Gaussian could pass the post-fit SNR filter.

    Args:
        wavelength: Wavelength array in nm (monotonically increasing).
        intensity: Dark-subtracted intensity array (counts).
        fit_range: Wavelength range for fitting (nm).
        position_bounds: Allowed peak center range (nm).
        fwhm_range: Hard FWHM bounds (nm) for optimizer.
        min_peak_separation: Minimum peak separation (nm), hard constraint.
        max_peaks: Maximum number of Gaussians to test.
        snr_threshold: Minimum SNR to retain a peak.
        min_fwhm_nm: Minimum FWHM (nm) to retain a peak. Rejects cosmic ray
            artifacts that hit the optimizer floor.
        noise_std: Pre-computed noise std. If None, estimated from noise_window.
        noise_window: Wavelength range for noise estimation (nm).
        saturation_threshold: CCD saturation level (counts).
        saturation_channel_limit: Channels at saturation before fit is skipped.
        overlap_exclusion: R2/R3 overlap zone to exclude (nm).
        strategy: "agnostic" (default) or "hypothesis".
        agnostic_r2_threshold: R² below which hypothesis triggers agnostic diagnostic.
        doublet_ratio_range: Allowed amp_1a/amp_1b ratio for Ce3+ doublet [lo, hi].

    Returns:
        FluorFitResult with peaks in wavelength (nm) units.
    """
    # --- Build fit mask: fit_range excluding overlap ---
    fit_mask = (
        (wavelength >= fit_range[0])
        & (wavelength <= fit_range[1])
        & ~(
            (wavelength >= overlap_exclusion[0])
            & (wavelength <= overlap_exclusion[1])
        )
    )
    x_fit = wavelength[fit_mask]
    y_fit = intensity[fit_mask]

    if x_fit.size == 0:
        return FluorFitResult(
            peaks=[],
            r2=0.0,
            rss=0.0,
            aicc=float("inf"),
            n_peaks=0,
            is_saturated=False,
            fit_skipped=False,
            n_saturated_channels=0,
            n_masked_channels=0,
            warnings=["no_data_in_fit_range"],
        )

    # --- Despike: rolling-median sigma-clip to remove cosmic rays ---
    _window = min(11, max(3, x_fit.size // 10) | 1)  # odd, >=3
    for _ in range(3):
        rolling_med = pd.Series(y_fit).rolling(
            window=_window, center=True, min_periods=1
        ).median().values
        residual = y_fit - rolling_med
        mad = np.median(np.abs(residual))
        robust_sigma = 1.4826 * mad if mad > 0 else np.std(residual)
        if robust_sigma == 0 or not np.isfinite(robust_sigma):
            break
        spike_mask = np.abs(residual) > 5.0 * robust_sigma
        if not np.any(spike_mask):
            break
        y_fit[spike_mask] = rolling_med[spike_mask]

    # --- Noise estimation ---
    if noise_std is None:
        noise_mask = (wavelength >= noise_window[0]) & (
            wavelength <= noise_window[1]
        )
        if np.any(noise_mask):
            noise_std = float(np.std(intensity[noise_mask]))
        else:
            # Fallback: use std of fit-range data
            noise_std = float(np.std(y_fit))

    # --- Saturation assessment (on fit-range data, excluding overlap) ---
    sat_level = 0.95 * saturation_threshold
    n_saturated = int(np.sum(y_fit > sat_level))
    is_saturated = bool(np.max(y_fit) >= saturation_threshold)

    # Count overlap channels for n_masked_channels
    overlap_mask = (
        (wavelength >= fit_range[0])
        & (wavelength <= fit_range[1])
        & (wavelength >= overlap_exclusion[0])
        & (wavelength <= overlap_exclusion[1])
    )
    n_overlap = int(np.sum(overlap_mask))

    # Three-tier saturation handling
    if is_saturated and n_saturated >= saturation_channel_limit:
        # Full saturation: skip fitting
        return FluorFitResult(
            peaks=[],
            r2=0.0,
            rss=0.0,
            aicc=float("inf"),
            n_peaks=0,
            is_saturated=True,
            fit_skipped=True,
            n_saturated_channels=n_saturated,
            n_masked_channels=n_overlap + n_saturated,
            warnings=["full_saturation_skip"],
        )

    # For partial saturation: mask out saturated channels
    n_sat_masked = 0
    if is_saturated:
        sat_keep = y_fit <= sat_level
        n_sat_masked = int(np.sum(~sat_keep))
        x_fit = x_fit[sat_keep]
        y_fit = y_fit[sat_keep]
        if x_fit.size < 10:
            return FluorFitResult(
                peaks=[],
                r2=0.0,
                rss=0.0,
                aicc=float("inf"),
                n_peaks=0,
                is_saturated=True,
                fit_skipped=True,
                n_saturated_channels=n_saturated,
                n_masked_channels=n_overlap + n_sat_masked,
                warnings=["insufficient_data_after_saturation_mask"],
            )

    n_masked_total = n_overlap + n_sat_masked
    y_max = float(np.max(y_fit)) if y_fit.size > 0 else 1.0
    n_data = x_fit.size
    fit_warnings: List[str] = []

    # --- Early bail-out: skip fitting if no channel exceeds SNR threshold ---
    # After despiking removes cosmic rays, check if the tallest feature
    # (max above median baseline) could possibly produce a peak above
    # the post-fit SNR filter. No baseline subtraction is applied to
    # fluorescence data, so we must subtract the median to measure
    # feature prominence rather than raw counts.
    y_prominence = y_max - float(np.median(y_fit))
    if noise_std > 0 and y_prominence / noise_std < snr_threshold:
        return FluorFitResult(
            peaks=[],
            r2=0.0,
            rss=float(np.sum(y_fit**2)),
            aicc=float("inf"),
            n_peaks=0,
            is_saturated=is_saturated,
            fit_skipped=True,
            n_saturated_channels=n_saturated,
            n_masked_channels=n_masked_total,
            warnings=["below_snr_threshold"],
        )

    # --- Model selection by strategy ---
    if strategy == "hypothesis":
        best_params, best_aicc, best_rss, hyp_warnings = _hypothesis_fit(
            x_fit, y_fit, noise_std, y_max,
            wavelength, intensity,
            agnostic_r2_threshold=agnostic_r2_threshold,
            position_bounds=position_bounds,
            fwhm_range=fwhm_range,
            min_peak_separation=min_peak_separation,
            max_peaks=max_peaks,
            doublet_ratio_range=doublet_ratio_range,
        )
        fit_warnings.extend(hyp_warnings)
    else:
        best_params, best_aicc, best_rss = _agnostic_fit(
            x_fit, y_fit, position_bounds, fwhm_range,
            min_peak_separation, max_peaks, y_max,
        )

    if best_params is None or best_params.size == 0:
        return FluorFitResult(
            peaks=[],
            r2=0.0,
            rss=best_rss if np.isfinite(best_rss) else float("inf"),
            aicc=best_aicc if np.isfinite(best_aicc) else float("inf"),
            n_peaks=0,
            is_saturated=is_saturated,
            fit_skipped=False,
            n_saturated_channels=n_saturated,
            n_masked_channels=n_masked_total,
            warnings=fit_warnings or ["fit_failed"],
        )

    # --- Compute fit quality and build peak list ---
    y_model = multi_gaussian(x_fit, best_params)
    r2 = compute_r2(y_fit, y_model)
    rss = float(np.sum((y_fit - y_model) ** 2))

    peaks: List[FluorPeakFit] = []
    for i in range(0, best_params.size, 3):
        center = float(best_params[i])
        amplitude = float(best_params[i + 1])
        fwhm = float(best_params[i + 2])
        sigma = fwhm_to_sigma(fwhm)
        area = float(amplitude * sigma * np.sqrt(2 * np.pi))
        snr = float(amplitude / max(noise_std, 1e-12))
        peaks.append(
            FluorPeakFit(
                center_nm=center,
                amplitude=amplitude,
                fwhm_nm=fwhm,
                area=area,
                snr=snr,
            )
        )

    # Filter by SNR threshold and minimum FWHM (rejects cosmic ray artifacts)
    peaks = [p for p in peaks if p.snr >= snr_threshold and p.fwhm_nm >= min_fwhm_nm]

    # Sort by center wavelength
    peaks.sort(key=lambda p: p.center_nm)

    # Hard separation filter: drop weaker peak when two are closer than min_peak_separation
    if min_peak_separation > 0 and len(peaks) > 1:
        filtered: List[FluorPeakFit] = [peaks[0]]
        for p in peaks[1:]:
            if abs(p.center_nm - filtered[-1].center_nm) < min_peak_separation:
                # Keep the stronger peak (higher SNR)
                if p.snr > filtered[-1].snr:
                    filtered[-1] = p
            else:
                filtered.append(p)
        peaks = filtered

    # Recompute R²/RSS/AICc from the peaks that survived filtering
    if peaks:
        retained_params = []
        for p in peaks:
            retained_params.extend([p.center_nm, p.amplitude, p.fwhm_nm])
        y_model_final = multi_gaussian(x_fit, np.array(retained_params))
        r2 = compute_r2(y_fit, y_model_final)
        rss = float(np.sum((y_fit - y_model_final) ** 2))
        best_aicc = _compute_aicc(n_data, rss, len(peaks) * 3)

    return FluorFitResult(
        peaks=peaks,
        r2=r2,
        rss=rss,
        aicc=best_aicc,
        n_peaks=len(peaks),
        is_saturated=is_saturated,
        fit_skipped=False,
        n_saturated_channels=n_saturated,
        n_masked_channels=n_masked_total,
        warnings=fit_warnings,
    )


