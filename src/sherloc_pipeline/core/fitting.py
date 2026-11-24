from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Sequence
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.stats import f as f_dist

from sherloc_pipeline.models.fitting import PeakFit, FitResult

# Fallback for the ``filter_fwhm_min_cm1`` fitting key when a config omits it;
# equal to the value in the shipped config.yaml.
FILTER_FWHM_MIN_CM1_DEFAULT = 25.0

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
    negative_spike_sigma: Optional[float] = None,
    cr_mask_cm1: Optional[Sequence[float]] = None,
) -> Tuple[FitResult, np.ndarray]:
    """Fit multiple Gaussians to a single spectrum with soft slit preference and low-FWHM penalty.

    negative_spike_sigma: when set, ROI channels below ``-negative_spike_sigma * noise_std``
    get zero weight: they are left out of seeding, the least-squares residual, RSS, AICc and R2.
    The returned model still covers every ROI channel.
    cr_mask_cm1: wavenumbers of cosmic-ray mask channels for this spectrum. When given, each
    returned peak carries ``cr_mask_review`` (a mask channel lies within center +/- FWHM/2).
    It is a review flag only: it never removes a peak and never changes the fit input.
    """
    # ROI selection
    if roi is None:
        roi = tuple(cfg.get('r1_fit_range', [x_cm1.min(), x_cm1.max()]))
    mask = (x_cm1 >= roi[0]) & (x_cm1 <= roi[1])
    x = x_cm1[mask]
    y_roi = y[mask]
    # Config params
    fit_min = float(cfg.get('fit_fwhm_min_initial_cm1', 20))
    filt_min = float(cfg.get('filter_fwhm_min_cm1', FILTER_FWHM_MIN_CM1_DEFAULT))
    fwhm_max = float(cfg.get('fwhm_max_cm1', np.inf))
    slit_w = float(cfg.get('slit_width_cm1_default', 34.1))
    slit_wt = float(cfg.get('slit_pref_weight', 0.2))
    low_edge = float(cfg.get('low_fwhm_edge_penalty', 0.1))
    max_peaks = int(cfg.get('max_peaks', 8))
    peak_sep = float(cfg.get('peak_separation_cm1', 25))
    r2_min = float(cfg.get('r_squared_min', 0.25))
    snr_min = float(cfg.get('min_snr', 2.0))
    min_display_snr = float(cfg.get('min_display_snr', 2.0))
    amp_sigma_mult = float(cfg.get('min_amp_sigma_multiplier', 0.5))

    # Detect seed peaks and prefilter by seed SNR, unless explicit seed centers provided
    if noise_std is None:
        noise_std = compute_noise_std(x_cm1, y, cfg=cfg)

    # Negative-spike channels get zero weight (they cannot be Raman signal).
    x_roi_all = x
    n_negative_masked = 0
    if negative_spike_sigma is not None and x.size:
        keep = y_roi >= -float(negative_spike_sigma) * float(noise_std)
        n_negative_masked = int(np.count_nonzero(~keep))
        if n_negative_masked:
            x, y_roi = x[keep], y_roi[keep]
    mask_warnings = [f"negative_spike_channels_masked:{n_negative_masked}"] if n_negative_masked else []
    if x.size == 0:
        # Every ROI channel was masked (or the ROI is empty): nothing to fit.
        return FitResult(peaks=[], r2=0.0, rss=float('inf'), dof=0, warnings=mask_warnings + ["no_peaks_detected"],
                         n_samples=0), np.zeros_like(y)
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
            return FitResult(peaks=[], r2=0.0, rss=float('inf'), dof=max(0, x.size - 1),
                             warnings=mask_warnings + ["no_peaks_detected"], n_samples=int(x.size)), np.zeros_like(y)

    # Model selection configuration
    parsimony_cfg = dict(cfg.get('parsimony', {}))
    model_selection = str(parsimony_cfg.get('model_selection', ''))
    if not model_selection:
        # Backward compat: use_aicc=true → "aicc", otherwise default to "ftest"
        model_selection = "aicc" if parsimony_cfg.get('use_aicc', False) else "ftest"
    ftest_alpha = float(parsimony_cfg.get('ftest_alpha', 0.01))
    aicc_min = int(parsimony_cfg.get('aicc_min_peaks', 1))
    aicc_max = int(parsimony_cfg.get('aicc_max_peaks', max_peaks))
    # A larger model must lower AICc by more than this to replace the current best (0 = strict minimum).
    aicc_thresh = max(float(parsimony_cfg.get('aicc_improve_threshold', 0.0)), 1e-9)
    # Skip a candidate the merge step below would shrink (two kept components closer than
    # half the seed separation): its AICc was earned by a component the output drops.
    reject_merged = bool(parsimony_cfg.get('reject_merged_candidates', False))
    min_amp = amp_sigma_mult * noise_std
    min_sep = 0.5 * peak_sep
    size_log: Optional[SizeLog] = None  # set by the AICc size search only

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
            size_rows: List[Dict] = []
            # Every size from 1 is fitted and logged, so the F-test pick and each component's
            # step can be read from the log; sizes below aicc_min_peaks never compete for AICc.
            for p in range(1, max_p + 1):
                eligible = p >= min_p
                sel = sorted(seeds_sorted_by_height[:p])
                # Slot j of the parameter vector holds seed sel[j]; its height rank (1 = tallest).
                ranks = [seeds_sorted_by_height.index(i) + 1 for i in sel]
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
                    # The log records every size, the merge-skip included; it decides nothing.
                    would_drop = _merge_would_drop(params, min_amp, min_display_snr, noise_std, min_sep)
                    size_rows.append(dict(p=p, seed_ranks=ranks, params=[float(v) for v in params],
                                          rss=rss, aicc=float(aicc), aicc_eligible=eligible,
                                          merge_skip=bool(reject_merged and would_drop), failed=False))
                    if not eligible or (reject_merged and would_drop):
                        continue
                    if best is None or aicc < best[0] - aicc_thresh:
                        best = (aicc, params, rss)
                except Exception:
                    size_rows.append(dict(p=p, seed_ranks=ranks, params=None, rss=None, aicc=None,
                                          aicc_eligible=eligible, merge_skip=False, failed=True))
                    continue
            size_log = SizeLog(n=int(x.size), rss0=float(np.sum(y_roi ** 2)), rows=size_rows,
                               ftest_alpha=ftest_alpha)

    gates = dict(noise_std=noise_std, filt_min=filt_min, snr_min=snr_min, r2_min=r2_min,
                 sharpness_max=float(cfg.get('posthoc_filters', {}).get('sharpness_max', 3.0)),
                 min_amp=min_amp, min_display_snr=min_display_snr, min_sep=min_sep, cr_mask_cm1=cr_mask_cm1)

    def _ftest_peaks(prm: np.ndarray) -> Tuple[List[PeakFit], float]:
        r2_f = float(compute_r2(y_roi, multi_gaussian(x, prm)))
        return _finalize_peaks(prm, x, y_roi, r2_f, gates, size_log.ranks_for(prm), size_log), r2_f

    # Fallback if nothing succeeded
    if best is None:
        selection = size_log.record(aicc_params=None, ftest_peaks_fn=_ftest_peaks) if size_log is not None else None
        return FitResult(peaks=[], r2=0.0, rss=float('inf'), dof=max(0, x.size - 1),
                         warnings=mask_warnings + ["fit_failed"], n_samples=int(x.size),
                         selection=selection), np.zeros_like(y)

    params = best[1]
    y_model = multi_gaussian(x, params)
    full_model = np.zeros_like(y)
    full_model[mask] = multi_gaussian(x_roi_all, params) if n_negative_masked else y_model

    r2 = compute_r2(y_roi, y_model)
    rss = best[2]
    dof = max(0, x.size - params.size)
    # noise estimate: use caller-provided value, else configurable window (default 2000–2100 cm^-1)
    if noise_std is None:
        noise_std = compute_noise_std(x_cm1, y, cfg=cfg)

    selection = None
    slot_ranks = None
    if size_log is not None:
        slot_ranks = size_log.ranks_for(params)
        selection = size_log.record(aicc_params=params, ftest_peaks_fn=_ftest_peaks)
    peaks = _finalize_peaks(params, x, y_roi, r2, gates, slot_ranks, size_log)
    return FitResult(peaks=peaks, r2=float(r2), rss=rss, dof=dof, warnings=list(mask_warnings),
                     n_samples=int(x.size), selection=selection), full_model


def _finalize_peaks(params: np.ndarray, x: np.ndarray, y_roi: np.ndarray, r2: float, gates: Dict,
                    slot_ranks: Optional[List[int]], size_log: Optional["SizeLog"]) -> List[PeakFit]:
    """One chosen model's output components: pass flags, then the prune and merge steps.

    The same steps for the AICc pick (what persists) and the F-test pick derived from the size
    log, so the two can be compared component for component. ``slot_ranks`` gives each parameter
    slot's seed height rank (None: no size search ran, and the marks stay None).
    """
    noise_std = gates['noise_std']
    peaks: List[PeakFit] = []
    for j, i in enumerate(range(0, params.size, 3)):
        m, a, f = params[i : i + 3]
        area = float(a * fwhm_to_sigma(f) * np.sqrt(2 * np.pi))
        snr = float(a / (noise_std + 1e-12))
        pass_fwhm = bool(f >= gates['filt_min'])
        pass_snr = bool(snr >= gates['snr_min'])
        pass_r2 = bool(r2 >= gates['r2_min'])
        # Sharpness ratio: data_at_center / amplitude (>>1 for cosmic rays)
        nearest_idx = int(np.argmin(np.abs(x - float(m))))
        data_at_center = float(y_roi[nearest_idx])
        sharpness_ratio = data_at_center / (float(a) + 1e-12)
        pass_sharpness = bool(sharpness_ratio < gates['sharpness_max'])
        cr_review = None
        if gates['cr_mask_cm1'] is not None:
            cr_review = bool(any(abs(float(c) - float(m)) <= 0.5 * float(f) for c in gates['cr_mask_cm1']))
        rank = slot_ranks[j] if slot_ranks is not None else None
        gain, step_p = size_log.step(rank) if (size_log is not None and rank is not None) else (None, None)
        peaks.append(PeakFit(m_cm1=float(m), a=float(a), fwhm=float(f), sigma=float(fwhm_to_sigma(f)), area=area,
                             snr=snr, pass_snr=pass_snr, pass_fwhm=pass_fwhm, pass_r2=pass_r2,
                             sharpness_ratio=sharpness_ratio, pass_sharpness=pass_sharpness,
                             cr_mask_review=cr_review, seed_rank=rank, aicc_gain=gain, step_p=step_p))

    # Prune obviously spurious components: extremely low amplitude or below display SNR threshold
    peaks = [p for p in peaks if (p.a >= gates['min_amp'] and p.snr >= gates['min_display_snr'])]

    # Merge peaks that are closer than half the configured separation: keep the larger amplitude
    if len(peaks) > 1:
        peaks.sort(key=lambda p: p.m_cm1)
        merged: List[PeakFit] = []
        for p in peaks:
            if not merged:
                merged.append(p)
            else:
                if abs(p.m_cm1 - merged[-1].m_cm1) < gates['min_sep']:
                    if p.a > merged[-1].a:
                        merged[-1] = p
                else:
                    merged.append(p)
        peaks = merged

    # Final sort by position
    peaks.sort(key=lambda p: p.m_cm1)
    return peaks


#: Sidecar format version; bump when a field changes meaning.
SIZE_LOG_VERSION = "fit-sizes/1"


class SizeLog:
    """Every model size the AICc search fitted at one point, and both rules' picks from it.

    Rows hold, per size p: the seed height rank of each parameter slot, the fitted parameters,
    RSS and AICc (None when the fit raised), whether the size may compete for AICc (False below
    ``aicc_min_peaks``) and whether ``reject_merged_candidates`` skipped it. Both rules choose from these same fits (the F-test path refits size p from the same
    start with the same bounds), so the F-test pick needs no extra fit: the longest run of sizes
    from 1 whose step p-value is below alpha, stopping at a failed fit or no degrees of freedom,
    as ``fit_spectrum``'s F-test branch does. Size 0 is the null model y = 0.
    """

    def __init__(self, n: int, rss0: float, rows: List[Dict], ftest_alpha: float):
        self.n, self.rss0, self.rows, self.alpha = n, rss0, rows, ftest_alpha
        self.by_p = {r['p']: r for r in rows}

    def _rss(self, p: int) -> Optional[float]:
        if p == 0:
            return self.rss0
        r = self.by_p.get(p)
        return None if r is None or r['failed'] else r['rss']

    def _aicc(self, p: int) -> Optional[float]:
        if p == 0:
            return float(_compute_aicc(n_samples=self.n, rss=self.rss0, num_params=0))
        r = self.by_p.get(p)
        return None if r is None or r['failed'] else r['aicc']

    def step(self, k: int) -> Tuple[Optional[float], Optional[float]]:
        """(AICc(k-1) - AICc(k), the F-test p-value of k-1 -> k peaks); None where a size is missing."""
        a0, a1 = self._aicc(k - 1), self._aicc(k)
        gain = None if a0 is None or a1 is None else float(a0 - a1)
        r0, r1 = self._rss(k - 1), self._rss(k)
        dof0, dof1 = self.n - 3 * (k - 1), self.n - 3 * k
        pval = None
        if r0 is not None and r1 is not None and dof1 > 0:
            pval = float(_f_test_pvalue(r0, r1, dof0, dof1, delta_params=3))
        return gain, pval

    def ftest_size(self) -> Optional[int]:
        """The size the sequential F-test keeps (0: none passes); None when size 1 was not fitted."""
        if 1 not in self.by_p:
            return None
        kept = 0
        p = 1
        while p in self.by_p:
            _, pval = self.step(p)
            if pval is None or not (pval < self.alpha):
                break
            kept = p
            p += 1
        return kept

    def ranks_for(self, params: np.ndarray) -> Optional[List[int]]:
        r = self.by_p.get(params.size // 3)
        return list(r['seed_ranks']) if r is not None else None

    def record(self, aicc_params: Optional[np.ndarray], ftest_peaks_fn) -> Dict:  # fn(params) -> (peaks, r2)
        """The sidecar record: every size with its step statistics, and both picks
        (``aicc_size`` None: no size was admissible, the fit returned no peaks)."""
        sizes = []
        for r in self.rows:
            gain, pval = self.step(r['p'])
            sizes.append({**r, 'step_aicc_gain': gain, 'step_p': pval})
        fsize = self.ftest_size()
        ftest_peaks = ftest_r2 = None
        if fsize:
            fpeaks, ftest_r2 = ftest_peaks_fn(np.asarray(self.by_p[fsize]['params'], dtype=float))
            ftest_peaks = [dict(center_cm1=p.m_cm1, fwhm_cm1=p.fwhm, amplitude_a=p.a, area=p.area, snr=p.snr,
                                pass_snr=p.pass_snr, pass_fwhm=p.pass_fwhm, pass_r2=p.pass_r2,
                                sharpness_ratio=p.sharpness_ratio, pass_sharpness=p.pass_sharpness,
                                seed_rank=p.seed_rank, aicc_gain=p.aicc_gain, step_p=p.step_p)
                           for p in fpeaks]
        elif fsize == 0:
            ftest_peaks = []
        return dict(version=SIZE_LOG_VERSION, n=self.n, rss0=self.rss0, aicc0=self._aicc(0),
                    ftest_alpha=self.alpha, sizes=sizes,
                    aicc_size=None if aicc_params is None else int(aicc_params.size // 3),
                    ftest_size=fsize, ftest_r2=ftest_r2, ftest_peaks=ftest_peaks)


def _merge_would_drop(params: np.ndarray, min_amp: float, min_display_snr: float,
                      noise_std: float, min_sep: float) -> bool:
    """True when fit_spectrum's prune + merge steps would drop a component of this candidate."""
    kept = sorted(
        float(params[i]) for i in range(0, params.size, 3)
        if params[i + 1] >= min_amp and params[i + 1] / (noise_std + 1e-12) >= min_display_snr
    )
    return any(b - a < min_sep for a, b in zip(kept, kept[1:]))


PEAK_TABLE_COLUMNS = [
    'center_cm1', 'fwhm_cm1', 'amplitude_a', 'area', 'snr',
    'pass_snr', 'pass_fwhm', 'pass_r2', 'sharpness_ratio', 'pass_sharpness', 'cr_mask_review',
    'aicc_gain', 'step_p',
]


def save_peak_table(peaks: List[PeakFit], output_csv_path: str) -> None:
    """Write a point's peak table. A point with no peaks gets a header-only
    table, so readers see zero rows instead of an unparseable empty file."""
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
            'cr_mask_review': p.cr_mask_review,
            'aicc_gain': p.aicc_gain,
            'step_p': p.step_p,
        }
        for p in peaks
    ]
    pd.DataFrame(rows, columns=PEAK_TABLE_COLUMNS).to_csv(output_csv_path, index=False)




def fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


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

