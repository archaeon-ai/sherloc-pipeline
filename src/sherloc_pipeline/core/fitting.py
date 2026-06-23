from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

from sherloc_pipeline.models.fitting import PeakFit, FitResult

# Provenance: Peak fitting parameters (slit-width 34.1 cm-1, FWHM 22-90 cm-1,
# R2 >= 0.25) and SNR thresholds (>=3 standard, >=2 organics with human filtering)
# established by domain expert from instrument specifications and literature.


def multi_gaussian(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Sum of Gaussians; params = [m1,a1,f1, m2,a2,f2, ...] where f is FWHM."""
    assert params.size % 3 == 0
    y = np.zeros_like(x, dtype=float)
    for i in range(0, params.size, 3):
        m_cm1, a, fwhm = params[i : i + 3]
        y += gaussian(x, m_cm1, a, fwhm)
    return y
def _compute_aicc(n_samples: int, rss: float, num_params: int) -> float:
    """Compute AICc for nonlinear least squares.

    AIC = n * ln(RSS/n) + 2k; AICc = AIC + 2k(k+1)/(n-k-1)
    where k is the number of free parameters.
    """
    n = float(max(n_samples, 1))
    k = float(num_params)
    # Guard against degenerate cases
    if rss <= 0:
        rss = 1e-12
    aic = n * np.log(rss / n) + 2.0 * k
    if n - k - 1 <= 0:
        return aic
    return aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)


def _f_test_pvalue(rss_reduced: float, rss_full: float, dof_reduced: int,
                   dof_full: int, delta_params: int) -> float:
    """p-value for F-test comparing nested Gaussian models.

    Tests whether the more complex model (more peaks) provides a statistically
    significant improvement over the simpler model.

    H0: simpler model is adequate
    H1: complex model is better
    F = ((RSS_reduced - RSS_full) / delta_params) / (RSS_full / dof_full)
    """
    if dof_full <= 0 or rss_full <= 0:
        return 1.0
    if rss_reduced <= rss_full:
        return 1.0
    f_stat = ((rss_reduced - rss_full) / delta_params) / (rss_full / dof_full)
    if f_stat <= 0:
        return 1.0
    return 1.0 - f_dist.cdf(f_stat, delta_params, dof_full)


def _penalty_terms(params: np.ndarray, fit_fwhm_min_initial_cm1: float, slit_width_cm1: float,
                   slit_pref_weight: float, low_fwhm_edge_penalty: float) -> float:
    penalty = 0.0
    # Scale penalties by relative amplitude so tiny/zero-amplitude components don't bias widths
    amps = params[1::3]
    max_a = float(np.max(amps)) if amps.size else 1.0
    amp_scale = lambda a: (float(a) / max(max_a, 1e-12)) ** 2
    for i in range(0, params.size, 3):
        a = params[i + 1]
        f = params[i + 2]
        scale = amp_scale(a)
        if f < fit_fwhm_min_initial_cm1:
            penalty += scale * low_fwhm_edge_penalty * (fit_fwhm_min_initial_cm1 - f) ** 2
        penalty += scale * slit_pref_weight * (f - slit_width_cm1) ** 2
    return penalty


def _residuals(params: np.ndarray, x: np.ndarray, y: np.ndarray,
               fit_fwhm_min_initial_cm1: float, slit_width_cm1: float,
               slit_pref_weight: float, low_fwhm_edge_penalty: float) -> np.ndarray:
    y_model = multi_gaussian(x, params)
    res = y_model - y
    # append penalty as extra residuals to be minimized
    pen = _penalty_terms(params, fit_fwhm_min_initial_cm1, slit_width_cm1, slit_pref_weight, low_fwhm_edge_penalty)
    # Always append a penalty term to keep residual length constant during numdiff
    return np.concatenate([res, np.array([np.sqrt(max(pen, 0.0))])])


def detect_initial_peaks(x: np.ndarray, y: np.ndarray, peak_separation_cm1: float,
                         max_peaks: int) -> List[int]:
    """Return indices of candidate peaks using simple prominence-based detection."""
    # Estimate a reasonable prominence as a fraction of dynamic range
    dyn = np.nanmax(y) - np.nanmin(y)
    prominence = max(dyn * 0.05, 1e-6)
    # Convert separation in cm^-1 to points using median spacing for robustness
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    distance = max(int(peak_separation_cm1 / max(dx, 1e-9)), 1)
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    # keep strongest by height, but preserve x-order when returning
    strongest = sorted(peaks, key=lambda idx: y[idx], reverse=True)[:max_peaks]
    return sorted(strongest)


def build_initial_params(x: np.ndarray, y: np.ndarray, peak_idxs: List[int],
                         initial_fwhm: float) -> np.ndarray:
    params = []
    for idx in peak_idxs:
        m_cm1 = float(x[idx])
        a = float(max(y[idx], 0.0))
        f = float(initial_fwhm)
        params.extend([m_cm1, a, f])
    return np.array(params, dtype=float)


def fit_spectrum(
    x_cm1: np.ndarray,
    y: np.ndarray,
    cfg: Dict,
    roi: Optional[Tuple[float, float]] = None,
    seed_centers: Optional[List[float]] = None,
    noise_std: Optional[float] = None,
) -> Tuple[FitResult, np.ndarray]:
    """Fit multiple Gaussians to a single spectrum with soft slit preference and low-FWHM penalty."""
    # ROI selection
    if roi is None:
        roi = tuple(cfg.get('r1_fit_range', [x_cm1.min(), x_cm1.max()]))
    mask = (x_cm1 >= roi[0]) & (x_cm1 <= roi[1])
    x = x_cm1[mask]
    y_roi = y[mask]
    # Config params
    fit_min = float(cfg.get('fit_fwhm_min_initial_cm1', 20))
    filt_min = float(cfg.get('filter_fwhm_min_cm1', 30))
    fwhm_max = float(cfg.get('fwhm_max_cm1', np.inf))
    slit_w = float(cfg.get('slit_width_cm1_default', 34.1))
    slit_wt = float(cfg.get('slit_pref_weight', 0.2))
    low_edge = float(cfg.get('low_fwhm_edge_penalty', 0.1))
    max_peaks = int(cfg.get('max_peaks', 8))
    peak_sep = float(cfg.get('peak_separation_cm1', 25))
    r2_min = float(cfg.get('r_squared_min', 0.25))
    snr_min = float(cfg.get('min_snr', 2.0))

    # Detect seed peaks and prefilter by seed SNR, unless explicit seed centers provided
    if noise_std is None:
        noise_std = compute_noise_std(x_cm1, y, cfg=cfg)
    if seed_centers is not None and len(seed_centers) > 0:
        # Map seed centers to nearest indices within ROI
        centers = sorted(seed_centers)
        seed_idxs_all = []
        for c in centers:
            idx = int(np.argmin(np.abs(x - float(c))))
            if idx not in seed_idxs_all:
                seed_idxs_all.append(idx)
        # Skip SNR prefiltering when explicit seeds are provided
    else:
        seed_idxs_all = detect_initial_peaks(x, y_roi, peak_separation_cm1=peak_sep, max_peaks=max_peaks)
        pre_snr_min = float(cfg.get('min_seed_snr', 2.0))
        seed_idxs_all = [i for i in seed_idxs_all if (y_roi[i] >= pre_snr_min * noise_std)]
        if not seed_idxs_all:
            return FitResult(peaks=[], r2=0.0, rss=float('inf'), dof=max(0, x.size - 1), warnings=["no_peaks_detected"]), np.zeros_like(y)

    # Model selection configuration
    parsimony_cfg = dict(cfg.get('parsimony', {}))
    model_selection = str(parsimony_cfg.get('model_selection', ''))
    if not model_selection:
        # Backward compat: use_aicc=true → "aicc", otherwise default to "ftest"
        model_selection = "aicc" if parsimony_cfg.get('use_aicc', False) else "ftest"
    ftest_alpha = float(parsimony_cfg.get('ftest_alpha', 0.01))
    aicc_min = int(parsimony_cfg.get('aicc_min_peaks', 1))
    aicc_max = int(parsimony_cfg.get('aicc_max_peaks', max_peaks))
    aicc_thresh = float(parsimony_cfg.get('aicc_improve_threshold', 0.0))

    # Prepare candidate peak sets
    if seed_centers is not None and len(seed_centers) > 0:
        # Use exactly provided seeds; skip AICc model-size search
        sel = sorted(seed_idxs_all)
        p0 = build_initial_params(x, y_roi, sel, initial_fwhm=max(fit_min, slit_w))
        lb, ub = [], []
        for _ in range(0, p0.size, 3):
            lb.extend([roi[0], 0.0, fit_min])
            ub.extend([roi[1], np.inf, fwhm_max])
        bounds = (np.array(lb, dtype=float), np.array(ub, dtype=float))
        res = least_squares(
            _residuals, p0, args=(x, y_roi, fit_min, slit_w, slit_wt, low_edge), method='trf', bounds=bounds, max_nfev=5000
        )
        params = res.x
        y_model = multi_gaussian(x, params)
        rss = float(np.sum((y_roi - y_model) ** 2))
        best = (_compute_aicc(n_samples=x.size, rss=rss, num_params=params.size), params, rss)
    else:
        seeds_sorted_by_height = sorted(seed_idxs_all, key=lambda i: y_roi[i], reverse=True)
        max_p = min(len(seeds_sorted_by_height), aicc_max, max_peaks)

        best = None  # tuple(score, params, rss)

        if model_selection == "ftest":
            # Sequential F-test: add peaks one at a time, each must pass significance test
            n_pts = x.size
            rss_prev = float(np.sum(y_roi ** 2))  # null model: y=0
            dof_prev = n_pts
            for p in range(1, max_p + 1):
                sel = sorted(seeds_sorted_by_height[:p])
                p0 = build_initial_params(x, y_roi, sel, initial_fwhm=max(fit_min, slit_w))
                lb, ub = [], []
                for _ in range(p):
                    lb.extend([roi[0], 0.0, fit_min])
                    ub.extend([roi[1], np.inf, fwhm_max])
                bounds = (np.array(lb, dtype=float), np.array(ub, dtype=float))
                try:
                    res = least_squares(
                        _residuals, p0, args=(x, y_roi, fit_min, slit_w, slit_wt, low_edge),
                        method='trf', bounds=bounds, max_nfev=5000,
                    )
                    params = res.x
                    y_model = multi_gaussian(x, params)
                    rss = float(np.sum((y_roi - y_model) ** 2))
                except Exception:
                    break
                dof = n_pts - 3 * p
                if dof <= 0:
                    break
                pval = _f_test_pvalue(rss_prev, rss, dof_prev, dof, delta_params=3)
                if pval < ftest_alpha:
                    best = (0.0, params, rss)
                    rss_prev = rss
                    dof_prev = dof
                else:
                    break
        else:
            # AICc model selection (legacy)
            min_p = min(aicc_min, max_p)
            for p in range(min_p, max_p + 1):
                sel = sorted(seeds_sorted_by_height[:p])
                p0 = build_initial_params(x, y_roi, sel, initial_fwhm=max(fit_min, slit_w))
                lb, ub = [], []
                for _ in range(0, p0.size, 3):
                    lb.extend([roi[0], 0.0, fit_min])
                    ub.extend([roi[1], np.inf, fwhm_max])
                bounds = (np.array(lb, dtype=float), np.array(ub, dtype=float))
                try:
                    res = least_squares(
                        _residuals, p0, args=(x, y_roi, fit_min, slit_w, slit_wt, low_edge), method='trf', bounds=bounds, max_nfev=5000
                    )
                    params = res.x
                    y_model = multi_gaussian(x, params)
                    rss = float(np.sum((y_roi - y_model) ** 2))
                    aicc = _compute_aicc(n_samples=x.size, rss=rss, num_params=params.size)
                    if best is None or aicc < best[0] - 1e-9:
                        best = (aicc, params, rss)
                except Exception:
                    continue

    # Fallback if nothing succeeded
    if best is None:
        return FitResult(peaks=[], r2=0.0, rss=float('inf'), dof=max(0, x.size - 1), warnings=["fit_failed"]), np.zeros_like(y)

    params = best[1]
    y_model = multi_gaussian(x, params)
    full_model = np.zeros_like(y)
    full_model[mask] = y_model

    r2 = compute_r2(y_roi, y_model)
    rss = best[2]
    dof = max(0, x.size - params.size)
    # noise estimate: use caller-provided value, else configurable window (default 2000–2100 cm^-1)
    if noise_std is None:
        noise_std = compute_noise_std(x_cm1, y, cfg=cfg)

    # Sharpness threshold for cosmic ray rejection
    sharpness_max = float(cfg.get('posthoc_filters', {}).get('sharpness_max', 3.0))

    peaks: List[PeakFit] = []
    for i in range(0, params.size, 3):
        m, a, f = params[i : i + 3]
        area = float(a * fwhm_to_sigma(f) * np.sqrt(2 * np.pi))
        snr = float(a / (noise_std + 1e-12))
        pass_fwhm = bool(f >= filt_min)
        pass_snr = bool(snr >= snr_min)
        pass_r2 = bool(r2 >= r2_min)
        # Sharpness ratio: data_at_center / amplitude (>>1 for cosmic rays)
        nearest_idx = int(np.argmin(np.abs(x - float(m))))
        data_at_center = float(y_roi[nearest_idx])
        sharpness_ratio = data_at_center / (float(a) + 1e-12)
        pass_sharpness = bool(sharpness_ratio < sharpness_max)
        peaks.append(PeakFit(m_cm1=float(m), a=float(a), fwhm=float(f), sigma=float(fwhm_to_sigma(f)), area=area,
                             snr=snr, pass_snr=pass_snr, pass_fwhm=pass_fwhm, pass_r2=pass_r2,
                             sharpness_ratio=sharpness_ratio, pass_sharpness=pass_sharpness))

    # Prune obviously spurious components: extremely low amplitude or below display SNR threshold
    min_display_snr = float(cfg.get('min_display_snr', 2.0))
    amp_sigma_mult = float(cfg.get('min_amp_sigma_multiplier', 0.5))
    min_amp = amp_sigma_mult * noise_std  # amplitude threshold relative to noise
    peaks = [p for p in peaks if (p.a >= min_amp and p.snr >= min_display_snr)]

    # Merge peaks that are closer than half the configured separation: keep the larger amplitude
    if len(peaks) > 1:
        peaks.sort(key=lambda p: p.m_cm1)
        merged: List[PeakFit] = []
        min_sep = 0.5 * peak_sep
        for p in peaks:
            if not merged:
                merged.append(p)
            else:
                if abs(p.m_cm1 - merged[-1].m_cm1) < min_sep:
                    if p.a > merged[-1].a:
                        merged[-1] = p
                else:
                    merged.append(p)
        peaks = merged

    # Final sort by position
    peaks.sort(key=lambda p: p.m_cm1)
    return FitResult(peaks=peaks, r2=float(r2), rss=rss, dof=dof, warnings=[]), full_model


def save_peak_table(peaks: List[PeakFit], output_csv_path: str) -> None:
    import pandas as pd
    rows = [
        {
            'center_cm1': p.m_cm1,
            'fwhm_cm1': p.fwhm,
            'amplitude_a': p.a,
            'area': p.area,
            'snr': p.snr,
            'pass_snr': p.pass_snr,
            'pass_fwhm': p.pass_fwhm,
            'pass_r2': p.pass_r2,
            'sharpness_ratio': p.sharpness_ratio,
            'pass_sharpness': p.pass_sharpness,
        }
        for p in peaks
    ]
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)




def fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma: float) -> float:
    return float(sigma) * (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian(x: np.ndarray, m_cm1: float, a: float, fwhm: float) -> np.ndarray:
    s = fwhm_to_sigma(fwhm)
    return a * np.exp(-0.5 * ((x - m_cm1) / s) ** 2)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_noise_std(
    x_cm1_full: np.ndarray, 
    y_full: np.ndarray,
    window: Optional[Tuple[float, float]] = None,
    cfg: Optional[Dict] = None,
) -> float:
    """Estimate noise as std of intensity in a configurable spectral window.
    
    The noise estimate is used for SNR calculation. By default, uses the 
    2000–2100 cm⁻¹ region which is typically featureless in SHERLOC spectra.
    
    Args:
        x_cm1_full: Full Raman shift array (cm⁻¹)
        y_full: Full intensity array
        window: Optional explicit window (min, max) in cm⁻¹. Takes precedence over cfg.
        cfg: Optional config dict. If provided and window is None, reads from
             cfg['noise_estimation']['window'].
    
    Returns:
        Noise standard deviation. Falls back to global std if window unavailable.
    
    Config example:
        fitting:
          noise_estimation:
            window: [2000.0, 2100.0]
    """
    # Determine window: explicit > config > default
    if window is None:
        if cfg is not None:
            noise_cfg = cfg.get('noise_estimation', {})
            window_list = noise_cfg.get('window', [2000.0, 2100.0])
            window = (float(window_list[0]), float(window_list[1]))
        else:
            window = (2000.0, 2100.0)  # Legacy default
    
    mask = (x_cm1_full >= window[0]) & (x_cm1_full <= window[1])
    if np.any(mask):
        noise = np.std(y_full[mask])
    else:
        # Fallback to global std
        noise = np.std(y_full)
    return float(noise)

"""
Gaussian fitting module for SHERLOC pipeline.

This module handles multi-peak Gaussian fitting of Raman spectra with quality control.
"""

