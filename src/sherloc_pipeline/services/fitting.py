"""
Fitting service for SHERLOC pipeline.

This module provides orchestration services for fitting SHERLOC scan data,
including mineral fitting, organics fitting, hydration fitting, and averages fitting.

All per-point fitting methods (minerals, hydration, organics, fluorescence) use a
three-phase parallel architecture:

  1. **Prepare** — build per-point input dicts (sequential)
  2. **Fit** — run per-point workers via ProcessPoolExecutor (parallel, configurable
     via ``fitting.parallel_workers`` for Raman domains or
     ``fluorescence_fitting.parallel_workers`` for fluorescence)
  3. **Aggregate** — collect results, write scan-level summary CSVs (sequential)

Worker functions (_fit_point_minerals, _fit_point_hydration, _fit_point_organics)
are module-level functions (required for pickling by ProcessPoolExecutor). Each
worker performs fitting + plot generation + CSV writes for a single point.

Worker count is resolved by ``core.utils.resolve_parallel_workers()``:
  - 0 = auto (half of CPU cores)
  - 1 = sequential (no multiprocessing overhead)
  - N > 1 = explicit worker count

Usage:
    from sherloc_pipeline.services.fitting import FittingService
    from rich.console import Console

    service = FittingService(console=Console())
    result = service.fit_minerals(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        data_dir=Path("../data/loupe"),
        results_dir=Path("../results"),
    )
    print(result.summary)
    for artifact in result.artifacts:
        print(f"  {artifact}")
"""

import copy
import logging
import math
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

from .base import ServiceResult
from .errors import FittingError, enrich
from .paths import resolve_scan_context
from .runtime import RuntimeContext

logger = logging.getLogger(__name__)

# Suppress harmless scipy optimizer warning during curve_fit Jacobian computation
warnings.filterwarnings(
    "ignore", message="invalid value encountered in subtract",
    category=RuntimeWarning, module=r"scipy\.optimize",
)


def _to_bool_safe(value) -> bool:
    """Convert CSV boolean values (True/False/1/0/strings) to Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "t", "yes")


# Domain-specific CSV filename patterns for Raman peak discovery
RAMAN_CSV_PATTERNS = {
    "minerals":  r"_point(\d+)_fit_peaks\.csv$",
    "organics":  r"_point(\d+)_organics_(?:dg|g)_peaks\.csv$",
    "hydration": r"_point(\d+)_(?:hydration|bend)_peaks\.csv$",
}

# Subdirectory names for each Raman domain's fit artifacts
RAMAN_SUBDIRS = {
    "minerals":  "minerals_fit",
    "organics":  "organics_fit",
    "hydration": "hydration_fit",
}


# ---------------------------------------------------------------------------
# Module-level worker functions for parallel per-point fitting.
# These MUST be module-level (not methods) so ProcessPoolExecutor can pickle them.
# ---------------------------------------------------------------------------

def _fit_point_minerals(
    point_data: Dict[str, Any],
    *,
    x: np.ndarray,
    fit_cfg: Dict,
    fit_roi: Tuple[float, float],
    plot_roi: Tuple[float, float],
    out_dir: str,
    sol: str,
    target: str,
    scan: str,
    region: str,
    snr_min_cfg: float,
    fwhm_min_cfg: float,
) -> Dict[str, Any]:
    """Fit minerals for a single point (picklable worker)."""
    from sherloc_pipeline.core.fitting import fit_spectrum, save_peak_table
    from sherloc_pipeline.visualization.fitting_plots import plot_fit_overlay

    point_idx = point_data['point_idx']
    y = point_data['y']
    out_path = Path(out_dir)
    artifacts: List[Path] = []
    warnings: List[str] = []

    try:
        result, y_model_full = fit_spectrum(x, y, fit_cfg, roi=fit_roi)
        mask = (x >= plot_roi[0]) & (x <= plot_roi[1])

        accepted_point = any(
            (p.pass_snr and p.pass_fwhm and p.pass_r2 and p.pass_sharpness)
            for p in (result.peaks or [])
        )

        reviewable_point = any(
            (getattr(p, 'snr', 0.0) >= snr_min_cfg and getattr(p, 'fwhm', 0.0) >= fwhm_min_cfg)
            for p in (result.peaks or [])
        )

        # Always save per-point peaks table
        try:
            peaks_csv = out_path / f"{sol}_{target}_{scan}_{region}_point{point_idx}_fit_peaks.csv"
            save_peak_table(result.peaks, str(peaks_csv))
            artifacts.append(peaks_csv)
        except Exception as e:
            warnings.append(f"Failed to save peaks table for point {point_idx}: {e}")

        # Export plot when strictly accepted OR reviewable
        if accepted_point or reviewable_point:
            png_path = out_path / f"{sol}_{target}_{scan}_{region}_point{point_idx}_fit.png"
            plot_fit_overlay(
                x, y, mask, result, y_model_full, str(png_path),
                title=None, xlim=plot_roi,
                sol=sol, target=target, scan=scan, point=point_idx, roi=fit_roi,
            )
            artifacts.append(png_path)

        # Collect accepted peaks
        accepted_peaks = []
        for p in (result.peaks or []):
            if p.pass_snr and p.pass_fwhm and p.pass_r2 and p.pass_sharpness:
                accepted_peaks.append({
                    'point': point_idx,
                    'center_cm1': p.m_cm1,
                    'amplitude_a': p.a,
                    'fwhm_cm1': p.fwhm,
                    'snr': p.snr,
                    'r2': result.r2,
                })

        # AICc summary
        fit_mask = (x >= fit_roi[0]) & (x <= fit_roi[1])
        n = int(fit_mask.sum())
        k = int(3 * len(result.peaks)) if result.peaks else 3
        rss = float(result.rss)
        if rss <= 0:
            rss = 1e-12
        aic = n * math.log(rss / max(n, 1)) + 2 * k
        aicc = aic if (n - k - 1) <= 0 else aic + (2 * k * (k + 1)) / (n - k - 1)

        summary_row = {
            'point': point_idx,
            'k_params': k,
            'num_peaks': len(result.peaks),
            'rss': result.rss,
            'r2': result.r2,
            'aicc': aicc,
        }

        return {
            'point_idx': point_idx,
            'summary_row': summary_row,
            'accepted_peaks': accepted_peaks,
            'artifacts': artifacts,
            'warnings': warnings,
            'count_accepted': 1 if accepted_point else 0,
        }

    except Exception as e:
        return {
            'point_idx': point_idx,
            'summary_row': None,
            'accepted_peaks': [],
            'artifacts': [],
            'warnings': [f"Fit failed for point {point_idx}: {e}"],
            'count_accepted': 0,
        }


def _fit_point_hydration(
    point_data: Dict[str, Any],
    *,
    x: np.ndarray,
    fit_cfg_oh: Dict,
    oh_roi: Tuple[float, float],
    oh_plot: Tuple[float, float],
    oh_mask: np.ndarray,
    plot_mask_oh: np.ndarray,
    n_edge: int,
    min_snr: float,
    r2_min: float,
    center_lo: float,
    center_hi: float,
    out_dir: str,
    sol: str,
    target: str,
    scan: str,
) -> Dict[str, Any]:
    """Fit hydration for a single point (picklable worker)."""
    from sherloc_pipeline.core.fitting import fit_spectrum, save_peak_table
    from sherloc_pipeline.visualization.fitting_plots import plot_fit_overlay

    point_idx = point_data['point_idx']
    y_full = point_data['y']
    out_path = Path(out_dir)
    artifacts: List[Path] = []
    warnings: List[str] = []

    # Extract hydration window
    x_oh = x[oh_mask]
    y_oh = y_full[oh_mask]

    if len(x_oh) < 10:
        return {
            'point_idx': point_idx,
            'summary_row': {'point': point_idx, 'oh_detected': False, 'oh_r2': '', 'oh_n_accepted': 0},
            'accepted_peaks': [],
            'artifacts': [],
            'warnings': [],
        }

    # Linear endpoint baseline
    left_avg = float(np.mean(y_oh[:n_edge]))
    right_avg = float(np.mean(y_oh[-n_edge:]))
    baseline = np.linspace(left_avg, right_avg, len(y_oh))
    y_bl = y_oh - baseline

    # Noise estimation from quiet 2800-3000 cm⁻¹ sub-window
    quiet_mask = (x_oh >= 2800.0) & (x_oh <= 3000.0)
    if np.sum(quiet_mask) >= 3:
        noise_std = float(np.std(y_bl[quiet_mask]))
    else:
        noise_std = float(np.std(y_bl))

    # Pre-fit SNR gate
    y_max = float(np.max(y_bl))
    if noise_std > 0 and (y_max / noise_std) < min_snr:
        return {
            'point_idx': point_idx,
            'summary_row': {'point': point_idx, 'oh_detected': False, 'oh_r2': '', 'oh_n_accepted': 0},
            'accepted_peaks': [],
            'artifacts': [],
            'warnings': [],
        }

    # Inject baselined hydration window for fit_spectrum
    y_for_fit = np.zeros_like(y_full)
    y_for_fit[oh_mask] = y_bl

    oh_result, oh_model = fit_spectrum(x, y_for_fit, fit_cfg_oh, roi=oh_roi, noise_std=noise_std)

    # R² quality gate
    if oh_result.r2 < r2_min:
        return {
            'point_idx': point_idx,
            'summary_row': {'point': point_idx, 'oh_detected': False, 'oh_r2': oh_result.r2, 'oh_n_accepted': 0},
            'accepted_peaks': [],
            'artifacts': [],
            'warnings': [],
        }

    # Accept peaks passing center range + sharpness filter
    accepted_peaks = []
    for p in oh_result.peaks:
        if p.m_cm1 < center_lo or p.m_cm1 > center_hi:
            continue
        if not p.pass_sharpness:
            continue
        accepted_peaks.append(p)

    oh_pass = len(accepted_peaks) > 0

    # Export if accepted
    if oh_pass:
        peaks_csv = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_hydration_peaks.csv"
        save_peak_table(accepted_peaks, str(peaks_csv))
        artifacts.append(peaks_csv)

        png_path = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_hydration_fit.png"
        plot_fit_overlay(
            x, y_for_fit, plot_mask_oh, oh_result, oh_model,
            str(png_path),
            sol=sol, target=target, scan=scan, point=point_idx,
            xlim=oh_plot, roi=oh_roi,
        )
        artifacts.append(png_path)

    # Build accepted rows
    accepted_rows = []
    for p in accepted_peaks:
        accepted_rows.append({
            'point': point_idx,
            'center_cm1': p.m_cm1,
            'amplitude_a': p.a,
            'fwhm_cm1': p.fwhm,
            'snr': p.snr,
            'band': 'OH',
            'r2': oh_result.r2,
            'sharpness_ratio': p.sharpness_ratio,
            'pass_sharpness': p.pass_sharpness,
        })

    summary_row = {
        'point': point_idx,
        'oh_detected': bool(oh_pass),
        'oh_r2': oh_result.r2,
        'oh_n_accepted': len(accepted_peaks),
    }

    return {
        'point_idx': point_idx,
        'summary_row': summary_row,
        'accepted_peaks': accepted_rows,
        'artifacts': artifacts,
        'warnings': warnings,
    }


def _fit_point_organics(
    point_data: Dict[str, Any],
    *,
    x: np.ndarray,
    fit_cfg_org: Dict,
    g_roi: Tuple[float, float],
    d_roi: Tuple[float, float],
    org_roi: Tuple[float, float],
    org_plot: Tuple[float, float],
    org_mask: np.ndarray,
    g_acc_lo: float,
    g_acc_hi: float,
    d_acc_lo: float,
    d_acc_hi: float,
    persist_min_snr: float,
    organics_fwhm_mins: Dict,
    use_norm_input: bool,
    rebaseline_cfg: Dict,
    out_dir: str,
    sol: str,
    target: str,
    scan: str,
) -> Dict[str, Any]:
    """Fit organics for a single point (picklable worker)."""
    from sherloc_pipeline.core.fitting import fit_spectrum, save_peak_table
    from sherloc_pipeline.visualization.fitting_plots import plot_fit_overlay
    from sherloc_pipeline.core.mineral_id import classify_organic_band

    point_idx = point_data['point_idx']
    y = point_data['y'].copy()
    out_path = Path(out_dir)
    artifacts: List[Path] = []
    warnings: List[str] = []

    # Optional local re-baseline
    rb = rebaseline_cfg
    if use_norm_input and rb and rb.get('enabled', False):
        from sherloc_pipeline.core.baseline import BaselineParams, fit_baseline_window
        rb_roi = tuple(rb.get('roi', [1250.0, 1850.0]))
        blp = BaselineParams(
            lam=rb.get('lam', 1e6),
            asymmetric_coef=rb.get('asymmetric_coef', 0.01),
            iters=rb.get('iters', 25),
            diff_order=rb.get('diff_order', 2),
            tol=1e-3,
        )
        w = float(rb.get('downweight', {}).get('weight', 0.2))
        weights_builder = (
            (float(rb.get('downweight', {}).get('g_center', 1605.0)), float(rb.get('downweight', {}).get('g_halfwidth', 35.0)), w),
            (float(rb.get('downweight', {}).get('d_center', 1350.0)), float(rb.get('downweight', {}).get('d_halfwidth', 50.0)), w),
        )
        method = str(rb.get('method', 'aspls')).lower()
        poly_degree = int(rb.get('poly_degree', 2))
        y = fit_baseline_window(x, y, rb_roi, blp, weights_builder, method=method, poly_degree=poly_degree)

    # G-band gate fit
    fit_cfg_g = {**fit_cfg_org, 'slit_pref_weight': 0.0}
    result_g, y_model_g = fit_spectrum(x, y, {**fit_cfg_g, 'max_peaks': 1}, roi=g_roi)
    g_accepted = [p for p in result_g.peaks if (p.pass_snr and p.pass_r2 and p.pass_sharpness and g_acc_lo <= p.fwhm <= g_acc_hi)]
    detected_g = len(g_accepted) > 0

    accepted_rows = []
    dg_accepted_list = []  # track for summary

    if detected_g:
        fit_cfg_dg = {**fit_cfg_org, 'slit_pref_weight': 0.0}
        result_dg, y_model_dg = fit_spectrum(
            x, y, {**fit_cfg_dg, 'max_peaks': 2, 'parsimony': {'use_aicc': False}, 'min_amp_sigma_multiplier': 0.2},
            roi=org_roi, seed_centers=[1410.0, 1605.0],
        )
        dg_accepted = [p for p in result_dg.peaks if (p.pass_snr and p.pass_fwhm and p.pass_r2 and p.pass_sharpness)]
        dg_accepted_list = dg_accepted

        if dg_accepted:
            peaks_csv = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_organics_dg_peaks.csv"
            save_peak_table(result_dg.peaks, str(peaks_csv))
            artifacts.append(peaks_csv)

            png_path = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_organics_dg_fit.png"
            plot_fit_overlay(
                x, y, org_mask, result_dg, y_model_dg,
                str(png_path),
                sol=sol, target=target, scan=scan, point=point_idx, xlim=org_plot, roi=org_roi,
            )
            artifacts.append(png_path)

            for p in dg_accepted:
                if p.snr < persist_min_snr:
                    continue
                label = 'D' if (d_roi[0] <= p.m_cm1 < d_roi[1]) else 'G'
                band_label = classify_organic_band(p.m_cm1)
                min_fwhm_for_band = float(organics_fwhm_mins.get(band_label, 0.0))
                if p.fwhm < min_fwhm_for_band:
                    continue
                accepted_rows.append({
                    'point': point_idx,
                    'center_cm1': p.m_cm1,
                    'amplitude_a': p.a,
                    'fwhm_cm1': p.fwhm,
                    'snr': p.snr,
                    'band': label,
                    'r2': result_dg.r2,
                    'sharpness_ratio': p.sharpness_ratio,
                    'pass_sharpness': p.pass_sharpness,
                })
        else:
            # Fall back to G-only
            try:
                peaks_csv = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_organics_g_peaks.csv"
                save_peak_table(result_g.peaks, str(peaks_csv))
                artifacts.append(peaks_csv)

                png_path = out_path / f"{sol}_{target}_{scan}_R1_point{point_idx}_organics_g_fit.png"
                plot_fit_overlay(
                    x, y, org_mask, result_g, y_model_g,
                    str(png_path),
                    sol=sol, target=target, scan=scan, point=point_idx, xlim=org_plot, roi=g_roi,
                )
                artifacts.append(png_path)
            except Exception:
                pass

            for p in g_accepted:
                if p.snr < persist_min_snr:
                    continue
                band_label = classify_organic_band(p.m_cm1)
                min_fwhm_for_band = float(organics_fwhm_mins.get(band_label, 0.0))
                if p.fwhm < min_fwhm_for_band:
                    continue
                accepted_rows.append({
                    'point': point_idx,
                    'center_cm1': p.m_cm1,
                    'amplitude_a': p.a,
                    'fwhm_cm1': p.fwhm,
                    'snr': p.snr,
                    'band': 'G',
                    'r2': result_g.r2,
                    'sharpness_ratio': p.sharpness_ratio,
                    'pass_sharpness': p.pass_sharpness,
                })

    summary_row = {
        'point': point_idx,
        'g_detected': bool(detected_g),
        'd_detected': False,
        'g_r2': result_g.r2,
    }

    return {
        'point_idx': point_idx,
        'summary_row': summary_row,
        'accepted_peaks': accepted_rows,
        'artifacts': artifacts,
        'warnings': warnings,
    }


class FittingService:
    """Service for orchestrating SHERLOC scan fitting operations.
    
    This service coordinates multiple fitting modalities:
    1. Minerals fitting (mineral-related peaks across all points)
    2. Organics fitting (D/G bands per point)
    3. Hydration fitting (O-H stretch and bending modes)
    4. Averages fitting (mean and trimmed mean spectra)
    
    The service maintains console output consistency by accepting an optional
    Console instance, allowing CLI commands to use their existing console while
    programmatic consumers can provide their own or use a default.
    
    Attributes:
        console: Rich Console instance for progress/output (defaults to new Console)
        
    Example:
        >>> service = FittingService()
        >>> result = service.fit_minerals("0921", "Amherst_Point", "detail_1")
        >>> print(result.summary)
        'Fitted minerals for scan 0921/Amherst_Point/detail_1 successfully'
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        *,
        context: Optional[RuntimeContext] = None,
        database_path: Optional[Path] = None
    ):
        """Initialize fitting service.

        Args:
            console: Optional Rich Console instance. If None, creates a new Console.
            context: Optional RuntimeContext providing resolved configuration and roots. If None,
                a new context is bootstrapped.
            database_path: Optional path to phase.db database for peak persistence. If None,
                persist_fitted_peaks() will raise an error.
        """
        self.console = console if console is not None else Console()
        self.context = context if context is not None else RuntimeContext.bootstrap()
        self._database_path = database_path
        self._engine = None

    def _run_parallel(self, worker, inputs, n_workers: int, label: str) -> list:
        """Run worker function over inputs with a Rich progress bar.

        Works for both parallel (ProcessPoolExecutor) and sequential modes.
        Returns results in the same order as inputs.
        """
        total = len(inputs)
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(label, total=total)
            if n_workers > 1:
                # Submit all, collect in input order
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(worker, inp): i for i, inp in enumerate(inputs)}
                    results = [None] * total
                    for future in as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()
                        progress.advance(task)
            else:
                results = []
                for inp in inputs:
                    results.append(worker(inp))
                    progress.advance(task)
        return results

    def fit_minerals(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        region: str = "R1",
        snr_min: Optional[float] = None,
        filter_fwhm_min: Optional[float] = None,
        fit_fwhm_initial_min: Optional[float] = None,
        r2_min: Optional[float] = None,
        slit_pref_weight: Optional[float] = None,
        roi_override: Optional[str] = None,
        max_peaks: Optional[int] = None,
        verbose: bool = False,
    ) -> ServiceResult:
        """Fit mineral-related peaks across all points in a scan.

        Per-point work (fit_spectrum + plot + CSV) runs in parallel via
        ``_fit_point_minerals`` workers. See ``fitting.parallel_workers`` config.

        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point"). Normalized (spaces→underscores).
            scan: Scan type (e.g., "detail_1", "line")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            region: Spectral region to fit (R1 only for now)
            snr_min: Override acceptance SNR threshold
            filter_fwhm_min: Post-fit FWHM minimum (cm^-1)
            fit_fwhm_initial_min: Initial/bounds FWHM minimum (cm^-1)
            r2_min: Override acceptance R^2 minimum
            slit_pref_weight: Soft preference weight toward slit width
            roi_override: Override fit ROI as 'lo,hi' cm^-1
            max_peaks: Maximum peaks to consider
            verbose: Enable verbose logging
        
        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata
            
        Raises:
            FittingError: If fitting fails or required files are missing
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config

        try:
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = FittingError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion

            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan
            )
            base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            out_dir = base / "minerals_fit"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Expect normalized baselined CSV for region
            input_csv = base / f"{sol}_{target}_{scan}_{region}_normalized_baselined.csv"
            if not input_csv.exists():
                error = FittingError(
                    f"Missing input: {input_csv}. Run process-scan with baseline first.",
                    exit_code=1,
                    context={"input_file": str(input_csv)}
                )
                raise enrich(error, sol=sol, target=target, scan=scan)
            
            df = pd.read_csv(input_csv)
            x_col = 'raman_shift' if 'raman_shift' in df.columns else df.columns[0]
            x = df[x_col].to_numpy(float)
            
            fit_cfg = copy.deepcopy(cfg.fitting or {})
            
            # Optionally estimate per-scan slit width and use it
            try:
                dyn = fit_cfg.get('dynamic_slit', {})
                if dyn and dyn.get('enabled', False):
                    r1_norm_csv = base / f"{sol}_{target}_{scan}_R1_normalized.csv"
                    if r1_norm_csv.exists():
                        r1_df = pd.read_csv(r1_norm_csv)
                        ingestion2 = DataIngestion(
                            base_data_dir=context.base_data_dir,
                            results_dir=context.results_dir,
                            sol=sol,
                            target=target,
                            scan=scan
                        )
                        est = ingestion2._estimate_scan_slit_from_laser(r1_df)
                        if est is not None:
                            fit_cfg['slit_width_cm1_default'] = float(est)
                            # Persist estimate
                            meta = {
                                'sol': sol,
                                'target': target,
                                'scan': scan,
                                'roi_lo': float(dyn.get('roi', [600.0, 700.0])[0]),
                                'roi_hi': float(dyn.get('roi', [600.0, 700.0])[1]),
                                'min_snr': float(dyn.get('min_snr', 10.0)),
                                'fwhm_lo': float(dyn.get('fwhm_bounds', [20.0, 80.0])[0]),
                                'fwhm_hi': float(dyn.get('fwhm_bounds', [20.0, 80.0])[1]),
                                'slit_width_estimate_cm1': float(est),
                            }
                            pd.DataFrame([meta]).to_csv(
                                base / f"{sol}_{target}_{scan}_R1_slit_estimate.csv",
                                index=False
                            )
                            self.console.print(f"[blue]Dynamic slit estimate used: {est:.2f} cm^-1[/blue]")
            except Exception:
                logger.debug("Dynamic slit integration skipped (non-fatal).", exc_info=True)
            
            # Apply CLI overrides
            if snr_min is not None:
                fit_cfg['min_snr'] = float(snr_min)
            if filter_fwhm_min is not None:
                fit_cfg['filter_fwhm_min_cm1'] = float(filter_fwhm_min)
            if fit_fwhm_initial_min is not None:
                fit_cfg['fit_fwhm_min_initial_cm1'] = float(fit_fwhm_initial_min)
            if r2_min is not None:
                fit_cfg['r_squared_min'] = float(r2_min)
            if slit_pref_weight is not None:
                fit_cfg['slit_pref_weight'] = float(slit_pref_weight)
            if max_peaks is not None and max_peaks > 0:
                fit_cfg['max_peaks'] = int(max_peaks)
            
            # ROI
            fit_roi = tuple(fit_cfg.get('r1_fit_range', [float(np.nanmin(x)), float(np.nanmax(x))]))
            plot_roi = tuple(fit_cfg.get('r1_plot_range', fit_roi))
            if roi_override:
                try:
                    lo, hi = [float(v) for v in str(roi_override).split(',')]
                    fit_roi = (lo, hi)
                    plot_roi = (lo, hi)
                except Exception:
                    logger.warning("Invalid --roi; expected 'lo,hi'")
            
            # Identify point columns
            point_cols = [c for c in df.columns if str(c).isdigit()]
            if not point_cols:
                error = FittingError(
                    "No point columns found to fit.",
                    exit_code=1,
                    context={"input_file": str(input_csv)}
                )
                raise enrich(error, sol=sol, target=target, scan=scan)
            
            try:
                snr_min_cfg = float(fit_cfg.get('min_snr', 3.0))
            except Exception:
                snr_min_cfg = 3.0
            fwhm_min_cfg = float(fit_cfg.get('reviewable_fwhm_min_cm1', 25.0))

            # --- Phase 1: Prepare per-point inputs ---
            point_inputs = [
                {'point_idx': int(col), 'y': df[col].to_numpy(float)}
                for col in point_cols
            ]

            # --- Phase 2: Parallel fitting + plotting + CSV writes ---
            from sherloc_pipeline.core.utils import resolve_parallel_workers
            n_workers = resolve_parallel_workers(
                int((cfg.fitting or {}).get("parallel_workers", 0)),
                len(point_inputs),
            )
            _worker = partial(
                _fit_point_minerals,
                x=x, fit_cfg=fit_cfg, fit_roi=fit_roi, plot_roi=plot_roi,
                out_dir=str(out_dir), sol=sol, target=target, scan=scan,
                region=region, snr_min_cfg=snr_min_cfg, fwhm_min_cfg=fwhm_min_cfg,
            )

            results = self._run_parallel(
                _worker, point_inputs, n_workers,
                f"Minerals {sol}/{target}/{scan}",
            )

            # --- Phase 3: Aggregate results ---
            artifacts: List[Path] = []
            warnings: List[str] = []
            summary_rows = []
            accepted_rows = []
            count = 0

            for r in results:
                artifacts.extend(r['artifacts'])
                warnings.extend(r['warnings'])
                if r['summary_row'] is not None:
                    summary_rows.append(r['summary_row'])
                accepted_rows.extend(r['accepted_peaks'])
                count += r.get('count_accepted', 0)

            # Emit scan-level AICc summary CSV
            try:
                summary_df = pd.DataFrame(summary_rows).sort_values('point')
                out_csv = base / f"{sol}_{target}_{scan}_{region}_fit_aicc_summary.csv"
                summary_df.to_csv(out_csv, index=False)
                artifacts.append(out_csv)
                self.console.print(f"[green]Wrote AICc summary to {out_csv}[/green]")
            except Exception as e:
                logger.warning(f"failed to write AICc summary: {e}")
                warnings.append(f"Failed to write AICc summary: {e}")
            
            # Emit accepted-peaks scan-level summary
            try:
                if accepted_rows:
                    acc_df = pd.DataFrame(accepted_rows).sort_values(['point', 'center_cm1'])
                    acc_csv = out_dir / f"{sol}_{target}_{scan}_{region}_accepted_peaks.csv"
                    acc_df.to_csv(acc_csv, index=False)
                    artifacts.append(acc_csv)
                    self.console.print(f"[green]Wrote accepted-peaks summary to {acc_csv}[/green]")
            except Exception as e:
                logger.warning(f"failed to write accepted-peaks summary: {e}")
                warnings.append(f"Failed to write accepted-peaks summary: {e}")
            
            self.console.print(f"[bold green]Done.[/bold green] Saved {count} overlays and peak tables to {out_dir}")
            
            metadata = {
                "sol": sol,
                "target": target,
                "scan": scan,
                "region": region,
                "points_fitted": len(point_cols),
                "points_accepted": count,
                "total_accepted_peaks": len(accepted_rows),
            }
            
            return ServiceResult(
                summary=f"Fitted minerals for {len(point_cols)} points, {count} accepted",
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )
            
        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to fit minerals: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
    
    def fit_scan(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        region: str = "R1",
        snr_min: Optional[float] = None,
        filter_fwhm_min: Optional[float] = None,
        fit_fwhm_initial_min: Optional[float] = None,
        r2_min: Optional[float] = None,
        slit_pref_weight: Optional[float] = None,
        roi_override: Optional[str] = None,
        max_peaks: Optional[int] = None,
        verbose: bool = False,
    ) -> ServiceResult:
        """Fit scan (deprecated alias for fit_minerals).
        
        This method is a deprecated alias that calls fit_minerals. It maintains
        backward compatibility with existing CLI commands.
        
        Args:
            See fit_minerals for parameter documentation.
        
        Returns:
            ServiceResult from fit_minerals
        """
        return self.fit_minerals(
            sol=sol,
            target=target,
            scan=scan,
            data_dir=data_dir,
            results_dir=results_dir,
            region=region,
            snr_min=snr_min,
            filter_fwhm_min=filter_fwhm_min,
            fit_fwhm_initial_min=fit_fwhm_initial_min,
            r2_min=r2_min,
            slit_pref_weight=slit_pref_weight,
            roi_override=roi_override,
            max_peaks=max_peaks,
            verbose=verbose,
        )
    
    def fit_hydration(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> ServiceResult:
        """Fit O-H stretch per point using pre-baseline Gaussian fitting.

        Per-point work (baseline + fit + plot + CSV) runs in parallel via
        ``_fit_point_hydration`` workers. See ``fitting.parallel_workers`` config.

        Uses R1_normalized.csv (laser-normalized, background-subtracted, NO asPLS
        baseline) with a linear endpoint baseline over the hydration window.
        No despiking — cosmic ray rejection relies on quality gates (R² ≥ 0.25,
        FWHM ≥ 50 cm⁻¹, F-test significance).

        Reference: Phua et al. (2024), JGR: Planets, 10.1029/2023JE008251.

        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan name (e.g., "detail_1")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            verbose: Enable verbose logging

        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata

        Raises:
            FittingError: If fitting fails or required files are missing
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config

        try:
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = FittingError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion

            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan
            )
            base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            multifits_dir = base / "hydration_fit"
            multifits_dir.mkdir(parents=True, exist_ok=True)

            fit_cfg = copy.deepcopy(cfg.fitting or {})

            # Select input: prefer pre-baseline (R1_normalized.csv) per audit findings
            use_prebaseline = bool(fit_cfg.get('hydration_use_prebaseline', True))
            input_csv_norm = base / f"{sol}_{target}_{scan}_R1_normalized.csv"
            input_csv_bl = base / f"{sol}_{target}_{scan}_R1_normalized_baselined.csv"

            if use_prebaseline and input_csv_norm.exists():
                input_csv = input_csv_norm
                self.console.print("[green]Using R1_normalized.csv (pre-baseline) for hydration fitting.[/green]")
            elif input_csv_bl.exists():
                input_csv = input_csv_bl
                self.console.print("[yellow]Falling back to R1_normalized_baselined.csv for hydration.[/yellow]")
            elif input_csv_norm.exists():
                input_csv = input_csv_norm
                self.console.print("[green]Using R1_normalized.csv for hydration fitting.[/green]")
            else:
                error = FittingError(
                    f"Missing input: {input_csv_norm} (or {input_csv_bl}). Run process-scan first.",
                    exit_code=1,
                    context={"input_file_norm": str(input_csv_norm), "input_file_bl": str(input_csv_bl)}
                )
                raise enrich(error, sol=sol, target=target, scan=scan)

            df = pd.read_csv(input_csv)
            x_col = 'raman_shift' if 'raman_shift' in df.columns else df.columns[0]
            x = df[x_col].to_numpy(float)

            # Hydration fitting config (Phua et al. 2024 aligned)
            oh_roi = tuple(fit_cfg.get('hydration_fit_range', [2800.0, 3900.0]))
            oh_plot = tuple(fit_cfg.get('hydration_plot_range', [2600.0, 4000.0]))
            max_peaks = int(fit_cfg.get('hydration_max_peaks', 2))
            min_snr = float(fit_cfg.get('hydration_min_snr', 3.0))
            fwhm_min = float(fit_cfg.get('hydration_fwhm_min_cm1', 50.0))
            fwhm_max = float(fit_cfg.get('hydration_fwhm_max_cm1', 300.0))
            r2_min = float(fit_cfg.get('hydration_r2_min', 0.25))
            ftest_alpha = float(fit_cfg.get('hydration_ftest_alpha', 0.01))
            center_lo, center_hi = fit_cfg.get('hydration_center_range', [3000.0, 3900.0])

            # Build fit config for hydration (passed to fit_spectrum)
            fit_cfg_oh = dict(fit_cfg)
            fit_cfg_oh['max_peaks'] = max_peaks
            fit_cfg_oh['fit_fwhm_min_initial_cm1'] = fwhm_min
            fit_cfg_oh['filter_fwhm_min_cm1'] = fwhm_min
            fit_cfg_oh['fwhm_max_cm1'] = fwhm_max
            fit_cfg_oh['min_snr'] = min_snr
            fit_cfg_oh['r_squared_min'] = r2_min
            fit_cfg_oh['parsimony'] = {
                'model_selection': 'ftest',
                'ftest_alpha': ftest_alpha,
            }

            oh_mask = (x >= oh_roi[0]) & (x <= oh_roi[1])
            plot_mask_oh = (x >= oh_plot[0]) & (x <= oh_plot[1])
            n_edge = 5  # points for endpoint averaging

            point_cols = [c for c in df.columns if str(c).isdigit()]

            # --- Phase 1: Prepare per-point inputs ---
            point_inputs = [
                {'point_idx': int(col), 'y': df[col].to_numpy(float)}
                for col in point_cols
            ]

            # --- Phase 2: Parallel fitting + plotting + CSV writes ---
            from sherloc_pipeline.core.utils import resolve_parallel_workers
            n_workers = resolve_parallel_workers(
                int((cfg.fitting or {}).get("parallel_workers", 0)),
                len(point_inputs),
            )
            _worker = partial(
                _fit_point_hydration,
                x=x, fit_cfg_oh=fit_cfg_oh, oh_roi=oh_roi, oh_plot=oh_plot,
                oh_mask=oh_mask, plot_mask_oh=plot_mask_oh, n_edge=n_edge,
                min_snr=min_snr, r2_min=r2_min, center_lo=center_lo,
                center_hi=center_hi, out_dir=str(multifits_dir),
                sol=sol, target=target, scan=scan,
            )

            results = self._run_parallel(
                _worker, point_inputs, n_workers,
                f"Hydration {sol}/{target}/{scan}",
            )

            # --- Phase 3: Aggregate results ---
            artifacts: List[Path] = []
            warnings: List[str] = []
            summary = []
            accepted_rows = []

            for r in results:
                artifacts.extend(r['artifacts'])
                warnings.extend(r['warnings'])
                summary.append(r['summary_row'])
                accepted_rows.extend(r['accepted_peaks'])

            summary_csv = base / f"{sol}_{target}_{scan}_R1_hydration_summary.csv"
            pd.DataFrame(summary).to_csv(summary_csv, index=False)
            artifacts.append(summary_csv)

            # Emit accepted-peaks summary
            try:
                if accepted_rows:
                    acc_df = pd.DataFrame(accepted_rows).sort_values(['point', 'center_cm1'])
                    acc_csv = multifits_dir / f"{sol}_{target}_{scan}_R1_hydration_accepted_peaks.csv"
                    acc_df.to_csv(acc_csv, index=False)
                    artifacts.append(acc_csv)
                    self.console.print(f"[green]Wrote accepted-peaks summary to {acc_csv}[/green]")
            except Exception as e:
                logger.warning(f"failed to write hydration accepted-peaks summary: {e}")
                warnings.append(f"Failed to write hydration accepted-peaks summary: {e}")

            self.console.print(f"[bold green]Hydration analysis complete.[/bold green] Artifacts in {multifits_dir}")

            metadata = {
                "sol": sol,
                "target": target,
                "scan": scan,
                "points_fitted": len(point_cols),
                "total_accepted_peaks": len(accepted_rows),
            }

            return ServiceResult(
                summary=f"Fitted hydration for {len(point_cols)} points, {len(accepted_rows)} accepted peaks",
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )

        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to fit hydration: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    def fit_organics(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> ServiceResult:
        """Fit organic D/G bands (1250–1850 cm⁻¹) per point (up to 2 peaks) and emit overlays/CSV + summary.

        Per-point work (G-band gate + D/G refinement + plot + CSV) runs in parallel
        via ``_fit_point_organics`` workers. See ``fitting.parallel_workers`` config.

        Args:
            sol: Sol number (e.g., "0712")
            target: Target name (e.g., "SAU008"). Normalized (spaces→underscores).
            scan: Scan name (e.g., "detail_1")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            verbose: Enable verbose logging

        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata

        Raises:
            FittingError: If fitting fails or required files are missing
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        
        try:
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = FittingError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
        
        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion

            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan
            )
            base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            multifits_dir = base / "organics_fit"
            multifits_dir.mkdir(parents=True, exist_ok=True)

            # Prefer despiked+baselined per-point spectra (legacy behavior), then baselined, then normalized
            input_csv_dsp_bl = base / f"{sol}_{target}_{scan}_R1_normalized_despiked_baselined.csv"
            input_csv_bl = base / f"{sol}_{target}_{scan}_R1_normalized_baselined.csv"
            input_csv_norm = base / f"{sol}_{target}_{scan}_R1_normalized.csv"
            use_norm_input = False
            if input_csv_dsp_bl.exists():
                input_csv = input_csv_dsp_bl
                self.console.print("[green]Using R1_normalized_despiked_baselined.csv for organics fitting.[/green]")
            elif input_csv_bl.exists():
                input_csv = input_csv_bl
                self.console.print("[green]Using R1_normalized_baselined.csv for organics fitting.[/green]")
            elif input_csv_norm.exists():
                input_csv = input_csv_norm
                use_norm_input = True
                self.console.print("[yellow]Using R1_normalized.csv (baselined not found). Local re-baseline is available.")
            else:
                error = FittingError(
                    f"Missing input: {input_csv_dsp_bl} (or {input_csv_bl} or {input_csv_norm}). Run process-scan first.",
                    exit_code=1,
                    context={
                        "input_file_dsp_bl": str(input_csv_dsp_bl),
                        "input_file_bl": str(input_csv_bl),
                        "input_file_norm": str(input_csv_norm)
                    }
                )
                raise enrich(error, sol=sol, target=target, scan=scan)

            df = pd.read_csv(input_csv)
            x_col = 'raman_shift' if 'raman_shift' in df.columns else df.columns[0]
            x = df[x_col].to_numpy(float)

            fit_cfg = copy.deepcopy(cfg.fitting or {})

            org_roi = tuple(fit_cfg.get('organics_fit_range', [1250.0, 1850.0]))
            org_plot = tuple(fit_cfg.get('organics_plot_range', org_roi))
            f_lo, f_hi = tuple(fit_cfg.get('organics_fwhm_bounds', [40.0, 120.0]))
            g_roi = tuple(fit_cfg.get('organics_g_roi', [1500.0, 1700.0]))
            d_roi = tuple(fit_cfg.get('organics_d_roi', [1250.0, 1500.0]))
            g_acc_lo, g_acc_hi = tuple(fit_cfg.get('organics_g_fwhm_accept', [40.0, 100.0]))
            d_acc_lo, d_acc_hi = tuple(fit_cfg.get('organics_d_fwhm_accept', [100.0, 200.0]))

            fit_cfg_org = dict(fit_cfg)
            fit_cfg_org['max_peaks'] = int(fit_cfg.get('organics_max_peaks', 2))
            fit_cfg_org['fit_fwhm_min_initial_cm1'] = float(max(fit_cfg_org.get('fit_fwhm_min_initial_cm1', 22.0), f_lo))
            fit_cfg_org['fwhm_max_cm1'] = float(f_hi)
            fit_cfg_org['min_snr'] = float(fit_cfg.get('organics_min_snr', fit_cfg.get('min_snr', 3.0)))

            # Post-hoc filters: match DB persistence thresholds
            posthoc = fit_cfg.get('posthoc_filters', {})
            persist_min_snr = float(fit_cfg.get('min_snr', 3.0))
            organics_fwhm_mins = posthoc.get('organics_fwhm', {})

            org_mask = (x >= org_roi[0]) & (x <= org_roi[1])
            point_cols = [c for c in df.columns if str(c).isdigit()]

            # --- Phase 1: Prepare per-point inputs ---
            point_inputs = [
                {'point_idx': int(col), 'y': df[col].to_numpy(float)}
                for col in point_cols
            ]

            # --- Phase 2: Parallel fitting + plotting + CSV writes ---
            from sherloc_pipeline.core.utils import resolve_parallel_workers
            n_workers = resolve_parallel_workers(
                int((cfg.fitting or {}).get("parallel_workers", 0)),
                len(point_inputs),
            )
            _worker = partial(
                _fit_point_organics,
                x=x, fit_cfg_org=fit_cfg_org, g_roi=g_roi, d_roi=d_roi,
                org_roi=org_roi, org_plot=org_plot, org_mask=org_mask,
                g_acc_lo=g_acc_lo, g_acc_hi=g_acc_hi, d_acc_lo=d_acc_lo,
                d_acc_hi=d_acc_hi, persist_min_snr=persist_min_snr,
                organics_fwhm_mins=dict(organics_fwhm_mins),
                use_norm_input=use_norm_input,
                rebaseline_cfg=dict(fit_cfg.get('organics_rebaseline', {})),
                out_dir=str(multifits_dir), sol=sol, target=target, scan=scan,
            )

            results = self._run_parallel(
                _worker, point_inputs, n_workers,
                f"Organics {sol}/{target}/{scan}",
            )

            # --- Phase 3: Aggregate results ---
            artifacts: List[Path] = []
            warnings: List[str] = []
            summary = []
            accepted_rows = []

            for r in results:
                artifacts.extend(r['artifacts'])
                warnings.extend(r['warnings'])
                summary.append(r['summary_row'])
                accepted_rows.extend(r['accepted_peaks'])

            summary_csv = base / f"{sol}_{target}_{scan}_R1_organics_summary.csv"
            pd.DataFrame(summary).sort_values('point').to_csv(summary_csv, index=False)
            artifacts.append(summary_csv)

            # Emit accepted-peaks summary
            try:
                if accepted_rows:
                    acc_df = pd.DataFrame(accepted_rows).sort_values(['point', 'center_cm1'])
                    acc_csv = multifits_dir / f"{sol}_{target}_{scan}_R1_organics_accepted_peaks.csv"
                    acc_df.to_csv(acc_csv, index=False)
                    artifacts.append(acc_csv)
                    self.console.print(f"[green]Wrote accepted-peaks summary to {acc_csv}[/green]")
            except Exception as e:
                logger.warning(f"failed to write organics accepted-peaks summary: {e}")
                warnings.append(f"Failed to write organics accepted-peaks summary: {e}")

            self.console.print(f"[bold green]Organics analysis complete.[/bold green] Artifacts in {multifits_dir}")

            metadata = {
                "sol": sol,
                "target": target,
                "scan": scan,
                "points_fitted": len(point_cols),
                "total_accepted_peaks": len(accepted_rows),
            }

            return ServiceResult(
                summary=f"Fitted organics for {len(point_cols)} points, {len(accepted_rows)} accepted peaks",
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )

        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to fit organics: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    def fit_averages(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        region: str = "R1",
        snr_min: Optional[float] = None,
        filter_fwhm_min: Optional[float] = None,
        fit_fwhm_initial_min: Optional[float] = None,
        r2_min: Optional[float] = None,
        slit_pref_weight: Optional[float] = None,
        roi_override: Optional[str] = None,
        max_peaks: Optional[int] = None,
        label_filter: Optional[str] = None,
        reviewed_only: bool = True,
        verbose: bool = False,
    ) -> ServiceResult:
        """Fit R1 average spectra (mean and trimmed mean) and write overlays/CSVs to averages_fit/.

        Args:
            sol: Sol number (e.g., "1613")
            target: Target name (e.g., "Nordoya"). Normalized (spaces→underscores).
            scan: Scan type (e.g., "detail", "line")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.
            region: Spectral region to fit (R1 only for now)
            snr_min: Override acceptance SNR threshold
            filter_fwhm_min: Post-fit FWHM minimum (cm^-1)
            fit_fwhm_initial_min: Initial/bounds FWHM minimum (cm^-1)
            r2_min: Override acceptance R^2 minimum
            slit_pref_weight: Soft preference weight toward slit width
            roi_override: Override fit ROI as 'lo,hi' cm^-1
            max_peaks: Maximum peaks to consider
            label_filter: Comma-separated label_id values to process only
            reviewed_only: Use only reviewed peaks (user_keep=True) for label averages
            verbose: Enable verbose logging

        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata

        Raises:
            FittingError: If fitting fails or required files are missing
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        run_context = self.context.with_overrides(
            data_dir=data_dir,
            results_dir=results_dir,
        )
        cfg = run_context.config
        
        try:
            context = resolve_scan_context(
                sol=sol,
                target=target,
                scan=scan,
                data_dir=data_dir,
                results_dir=results_dir,
                context=run_context,
            )
        except (FileNotFoundError, ValueError) as e:
            error = FittingError(
                f"Failed to resolve scan context: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)
        
        try:
            from sherloc_pipeline.core.data_ingestion import DataIngestion
            from sherloc_pipeline.core.fitting import fit_spectrum, save_peak_table, compute_r2
            from sherloc_pipeline.visualization.fitting_plots import plot_fit_overlay
            from sherloc_pipeline.models.fitting import FitResult
            
            ingestion = DataIngestion(
                base_data_dir=context.base_data_dir,
                results_dir=context.results_dir,
                sol=sol,
                target=target,
                scan=scan
            )
            base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            out_dir = base / "averages_fit"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Candidate inputs in preference order
            # Trim-mean filenames reflect the effective trim percentage (may differ
            # from configured 2% for small scans), so we glob for them.
            import re as _re_mod
            candidates: list[tuple[str, str]] = [
                ("mean", f"{sol}_{target}_{scan}_{region}_raw-n_mean_bkgsub_baselined.csv"),
            ]
            for _tm in sorted(base.glob(f"{sol}_{target}_{scan}_{region}_raw-n_*p_trim_mean_bkgsub_baselined.csv")):
                candidates.append(("trimmed_mean", _tm.name))
            candidates.append(
                ("mean", f"{sol}_{target}_{scan}_{region}_raw-n_mean_baselined.csv"),
            )
            for _tm in sorted(base.glob(f"{sol}_{target}_{scan}_{region}_raw-n_*p_trim_mean_baselined.csv")):
                if "_bkgsub_" not in _tm.name:
                    candidates.append(("trimmed_mean", _tm.name))
            
            # Load fitting config and apply overrides
            fit_cfg = copy.deepcopy(cfg.fitting or {})
            
            # Optionally estimate per-scan slit width
            try:
                dyn = fit_cfg.get('dynamic_slit', {})
                if dyn and dyn.get('enabled', False):
                    r1_norm_csv = base / f"{sol}_{target}_{scan}_R1_normalized.csv"
                    if r1_norm_csv.exists():
                        r1_df = pd.read_csv(r1_norm_csv)
                        ingestion2 = DataIngestion(
                            base_data_dir=context.base_data_dir,
                            results_dir=context.results_dir,
                            sol=sol,
                            target=target,
                            scan=scan
                        )
                        est = ingestion2._estimate_scan_slit_from_laser(r1_df)
                        if est is not None:
                            fit_cfg['slit_width_cm1_default'] = float(est)
            except Exception:
                logger.debug("Dynamic slit integration (averages) skipped.", exc_info=True)
            
            if snr_min is not None:
                fit_cfg['min_snr'] = float(snr_min)
            if filter_fwhm_min is not None:
                fit_cfg['filter_fwhm_min_cm1'] = float(filter_fwhm_min)
            if fit_fwhm_initial_min is not None:
                fit_cfg['fit_fwhm_min_initial_cm1'] = float(fit_fwhm_initial_min)
            if r2_min is not None:
                fit_cfg['r_squared_min'] = float(r2_min)
            if slit_pref_weight is not None:
                fit_cfg['slit_pref_weight'] = float(slit_pref_weight)
            if max_peaks is not None and max_peaks > 0:
                fit_cfg['max_peaks'] = int(max_peaks)
            
            artifacts: List[Path] = []
            warnings: List[str] = []
            summary_rows = []
            accepted_rows = []
            any_found = False
            
            for avg_kind, filename in candidates:
                input_csv = base / filename
                if not input_csv.exists():
                    continue
                any_found = True
                try:
                    self.console.print(f"[cyan]fit-averages {avg_kind} input:[/cyan] {input_csv}")
                except Exception:
                    pass
                
                df = pd.read_csv(input_csv)
                x_col = 'raman_shift' if 'raman_shift' in df.columns else df.columns[0]
                x = df[x_col].to_numpy(float)
                y = df['intensity'].to_numpy(float)
                
                def run_segment(roi, max_pk, f_lo, f_hi, slit_wt=fit_cfg.get('slit_pref_weight', 0.2)):
                    cfg = dict(fit_cfg)
                    cfg['r1_fit_range'] = list(roi)
                    cfg['max_peaks'] = int(max_pk)
                    cfg['fit_fwhm_min_initial_cm1'] = float(max(cfg.get('fit_fwhm_min_initial_cm1', 22.0), f_lo))
                    cfg['fwhm_max_cm1'] = float(f_hi)
                    cfg['slit_pref_weight'] = float(slit_wt)
                    res, y_model_full = fit_spectrum(x, y, cfg, roi=tuple(roi))
                    return res, y_model_full
                
                # Fit across multiple ROIs and combine components; plot full 500–4000 cm^-1
                full_plot = (500.0, 4000.0)
                if roi_override:
                    try:
                        lo, hi = [float(v) for v in str(roi_override).split(',')]
                        full_plot = (lo, hi)
                    except Exception:
                        logger.warning("Invalid --roi; expected 'lo,hi'")
                
                combined_peaks = []
                y_model_total = np.zeros_like(y)
                
                # Laser line (600–700), 1 peak
                res_laser, y_laser = run_segment((600.0, 700.0), 1, 20.0, 80.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                combined_peaks.extend(res_laser.peaks)
                y_model_total += y_laser
                
                # Minerals (700–1300), up to 5 peaks, FWHM 30–90
                res_min, y_min = run_segment((700.0, 1300.0), 5, 30.0, 90.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                combined_peaks.extend(res_min.peaks)
                y_model_total += y_min
                
                # G-band (broader in averages): widen ROI and allow larger FWHM, suppress slit preference
                res_g, y_g = run_segment((1570.0, 1635.0), 1, 40.0, 180.0, slit_wt=0.0)
                combined_peaks.extend(res_g.peaks)
                y_model_total += y_g
                
                # Hydration (3000–4000), up to 3 peaks, FWHM 40–4000
                res_h, y_h = run_segment((3000.0, 4000.0), 3, 40.0, 4000.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                combined_peaks.extend(res_h.peaks)
                y_model_total += y_h
                
                # Build combined result and overlay over full range
                full_mask = (x >= full_plot[0]) & (x <= full_plot[1])
                r2_full = compute_r2(y[full_mask], y_model_total[full_mask])
                result = FitResult(
                    peaks=combined_peaks,
                    r2=float(r2_full),
                    rss=float(np.sum((y[full_mask]-y_model_total[full_mask])**2)),
                    dof=max(0, int(full_mask.sum()) - 3*len(combined_peaks)),
                    warnings=[]
                )
                
                # Use explicit naming tokens and indicate if input was background-subtracted
                # Extract the actual trim label from the filename (e.g. "4p_trim_mean")
                if avg_kind == 'trimmed_mean':
                    _m = _re_mod.search(r'(\d+(?:\.\d+)?p_trim_mean)', filename)
                    kind_token = _m.group(1) if _m else '2p_trim_mean'
                else:
                    kind_token = 'mean'
                is_bgsub = ('bkgsub' in filename)
                suffix = ('_bkgsub' if is_bgsub else '')
                peaks_csv = out_dir / f"{sol}_{target}_{scan}_{region}_avg-{kind_token}{suffix}_fit_peaks.csv"
                save_peak_table(result.peaks, str(peaks_csv))
                artifacts.append(peaks_csv)
                
                png_path = out_dir / f"{sol}_{target}_{scan}_{region}_avg-{kind_token}{suffix}_fit.png"
                title = f"sol {sol} {target} {scan} R1 avg {kind_token} 500–4000 cm⁻¹ (R²={r2_full:.3f})"
                plot_mask = full_mask
                plot_fit_overlay(
                    x, y, plot_mask, result, y_model_total, str(png_path),
                    title=title, xlim=full_plot,
                    sol=sol, target=target, scan=scan, point=None, roi=full_plot
                )
                artifacts.append(png_path)
                
                # Accepted peaks rows
                for p in (result.peaks or []):
                    if p.pass_snr and p.pass_fwhm and p.pass_r2 and p.pass_sharpness:
                        accepted_rows.append({
                            'avg_kind': avg_kind,
                            'center_cm1': p.m_cm1,
                            'amplitude_a': p.a,
                            'fwhm_cm1': p.fwhm,
                            'snr': p.snr,
                            'r2': result.r2,
                        })
                
                # Summary row (use combined)
                n = int(full_mask.sum())
                k = int(3 * len(result.peaks)) if result.peaks else 3
                rss = float(np.sum((y_model_total[full_mask]-y[full_mask])**2))
                if rss <= 0:
                    rss = 1e-12
                aic = n * math.log(rss / max(n, 1)) + 2 * k
                aicc = aic if (n - k - 1) <= 0 else aic + (2 * k * (k + 1)) / (n - k - 1)
                summary_rows.append({
                    'avg_kind': avg_kind,
                    'k_params': k,
                    'num_peaks': len(result.peaks),
                    'rss': rss,
                    'r2': result.r2,
                    'aicc': aicc,
                })
            
            if not any_found:
                error = FittingError(
                    f"No suitable R1 average CSVs found under {base}. Run process-scan first.",
                    exit_code=1,
                    context={"base": str(base)}
                )
                raise enrich(error, sol=sol, target=target, scan=scan)
            
            # Write summaries
            try:
                if summary_rows:
                    summary_csv = base / f"{sol}_{target}_{scan}_{region}_averages_fit_summary.csv"
                    pd.DataFrame(summary_rows).sort_values('avg_kind').to_csv(summary_csv, index=False)
                    artifacts.append(summary_csv)
                if accepted_rows:
                    acc_csv = out_dir / f"{sol}_{target}_{scan}_{region}_averages_accepted_peaks.csv"
                    pd.DataFrame(accepted_rows).sort_values(['avg_kind', 'center_cm1']).to_csv(acc_csv, index=False)
                    artifacts.append(acc_csv)
            except Exception as e:
                logger.warning(f"failed to write averages summaries: {e}")
                warnings.append(f"Failed to write averages summaries: {e}")
            
            # Per-label averaged spectra from accepted points (optional)
            try:
                # Prefer reviewed scan-level table if present; else derive from per-modality CSVs
                scan_acc = base / f"{sol}_{target}_{scan}_accepted_peaks.csv"
                if scan_acc.exists():
                    acc = pd.read_csv(scan_acc)
                    if reviewed_only:
                        acc = acc[acc.get('user_keep', True) == True]
                else:
                    # Derive from per-modality CSVs
                    acc_rows = []
                    m_csv = base / "minerals_fit" / f"{sol}_{target}_{scan}_R1_accepted_peaks.csv"
                    if m_csv.exists():
                        mdf = pd.read_csv(m_csv)
                        try:
                            from sherloc_pipeline.core.mineral_id import load_mineral_rules, map_min_id_series
                            rules = load_mineral_rules(
                                Path(fit_cfg.get('library_path')) if fit_cfg.get('library_path') else None,
                                inline_rules=fit_cfg.get('mineral_rules')
                            )
                            if 'min_ID' not in mdf.columns or mdf['min_ID'].astype(str).str.strip().eq('').any():
                                mdf['min_ID'] = map_min_id_series(mdf['center_cm1'], rules)
                        except Exception:
                            pass
                        keep_thresh = float(fit_cfg.get('filter_fwhm_min_cm1', 30.0))
                        mdf = mdf[mdf['fwhm_cm1'] >= keep_thresh]
                        for _, r in mdf.iterrows():
                            acc_rows.append({
                                'modality': 'minerals',
                                'point': int(r['point']),
                                'label_id': str(r.get('min_ID', ''))
                            })
                    o_csv = base / "organics_fit" / f"{sol}_{target}_{scan}_R1_organics_accepted_peaks.csv"
                    if o_csv.exists():
                        odf = pd.read_csv(o_csv)
                        for _, r in odf.iterrows():
                            acc_rows.append({
                                'modality': 'organics',
                                'point': int(r['point']),
                                'label_id': str(r.get('band', ''))
                            })
                    h_csv = base / "hydration_fit" / f"{sol}_{target}_{scan}_R1_hydration_accepted_peaks.csv"
                    if h_csv.exists():
                        hdf = pd.read_csv(h_csv)
                        for _, r in hdf.iterrows():
                            acc_rows.append({
                                'modality': 'hydration',
                                'point': int(r['point']),
                                'label_id': str(r.get('band', ''))
                            })
                    acc = pd.DataFrame(acc_rows) if acc_rows else pd.DataFrame(columns=['modality', 'point', 'label_id'])
                
                # Optional label filter
                label_set = None
                if label_filter:
                    label_set = set([s.strip() for s in str(label_filter).split(',') if s.strip()]) or None
                
                if not acc.empty:
                    def _render_label_average(modality: str, label: str, pts: List[int], source_csv: Path):
                        if len(pts) < 2:
                            return
                        spec = pd.read_csv(source_csv)
                        x_col = 'raman_shift' if 'raman_shift' in spec.columns else spec.columns[0]
                        x = spec[x_col].to_numpy(float)
                        y_cols = [str(p) for p in pts if str(p) in spec.columns]
                        if len(y_cols) < 2:
                            return
                        y = spec[y_cols].to_numpy(float).mean(axis=1)
                        out_csv = out_dir / f"{sol}_{target}_{scan}_R1_label-{label}_mean.csv"
                        pd.DataFrame({x_col: x, 'intensity': y}).to_csv(out_csv, index=False)
                        artifacts.append(out_csv)
                        
                        # Use the same averages_fit segmentation
                        full_plot = (500.0, 4000.0)
                        y_model_total = np.zeros_like(y)
                        
                        def run_segment_label(roi, max_pk, f_lo, f_hi, slit_wt=fit_cfg.get('slit_pref_weight', 0.2)):
                            cfg = dict(fit_cfg)
                            cfg['r1_fit_range'] = list(roi)
                            cfg['max_peaks'] = int(max_pk)
                            cfg['fit_fwhm_min_initial_cm1'] = float(max(cfg.get('fit_fwhm_min_initial_cm1', 22.0), f_lo))
                            cfg['fwhm_max_cm1'] = float(f_hi)
                            cfg['slit_pref_weight'] = float(slit_wt)
                            res, y_model_full = fit_spectrum(x, y, cfg, roi=tuple(roi))
                            return res, y_model_full
                        
                        res_laser, y_laser = run_segment_label((600.0, 700.0), 1, 20.0, 80.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                        res_min, y_min = run_segment_label((700.0, 1300.0), 5, 30.0, 90.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                        res_g, y_g = run_segment_label((1570.0, 1635.0), 1, 40.0, 180.0, slit_wt=0.0)
                        res_h, y_h = run_segment_label((3000.0, 4000.0), 3, 40.0, 4000.0, slit_wt=fit_cfg.get('slit_pref_weight', 0.2))
                        y_model_total += y_laser + y_min + y_g + y_h
                        
                        mask = (x >= full_plot[0]) & (x <= full_plot[1])
                        r2_full = compute_r2(y[mask], y_model_total[mask])
                        result = FitResult(
                            peaks=(res_laser.peaks + res_min.peaks + res_g.peaks + res_h.peaks),
                            r2=float(r2_full),
                            rss=float(np.sum((y[mask]-y_model_total[mask])**2)),
                            dof=0,
                            warnings=[]
                        )
                        title = f"{label} (points: {', '.join(str(p) for p in sorted(pts))})"
                        png = out_dir / f"{sol}_{target}_{scan}_R1_label-{label}_mean_fit.png"
                        plot_fit_overlay(
                            x, y, mask, result, y_model_total, str(png),
                            title=title, xlim=full_plot,
                            sol=sol, target=target, scan=scan, point=None, roi=full_plot
                        )
                        artifacts.append(png)
                    
                    # Minerals
                    sel = acc[(acc['modality'] == 'minerals') & acc['label_id'].astype(str).str.strip().ne('')]
                    if label_set is not None:
                        sel = sel[sel['label_id'].astype(str).isin(label_set)]
                    if not sel.empty:
                        src = base / f"{sol}_{target}_{scan}_R1_normalized_baselined.csv"
                        for label, g in sel.groupby('label_id'):
                            _render_label_average('minerals', str(label), sorted(set(g['point'].astype(int))), src)
                    
                    # Organics
                    sel = acc[(acc['modality'] == 'organics') & acc['label_id'].astype(str).str.strip().ne('')]
                    if label_set is not None:
                        sel = sel[sel['label_id'].astype(str).isin(label_set)]
                    if not sel.empty:
                        src = base / f"{sol}_{target}_{scan}_R1_normalized_despiked_baselined.csv"
                        for label, g in sel.groupby('label_id'):
                            _render_label_average('organics', str(label), sorted(set(g['point'].astype(int))), src)
                    
                    # Hydration
                    sel = acc[(acc['modality'] == 'hydration') & acc['label_id'].astype(str).str.strip().ne('')]
                    if label_set is not None:
                        sel = sel[sel['label_id'].astype(str).isin(label_set)]
                    if not sel.empty:
                        src = base / f"{sol}_{target}_{scan}_R1_normalized_baselined.csv"
                        for label, g in sel.groupby('label_id'):
                            _render_label_average('hydration', str(label), sorted(set(g['point'].astype(int))), src)
            except Exception:
                logger.debug("Per-label averages generation skipped.", exc_info=True)
            
            metadata = {
                "sol": sol,
                "target": target,
                "scan": scan,
                "region": region,
                "averages_fitted": len(summary_rows),
                "total_accepted_peaks": len(accepted_rows),
            }
            
            return ServiceResult(
                summary=f"Fitted {len(summary_rows)} averages, {len(accepted_rows)} accepted peaks",
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
            )
            
        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to fit averages: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    def _discover_peak_csvs(self, fit_dir: Path, sol: str, target: str, scan: str, region: str, domain: str = "minerals") -> List[Tuple[Path, int]]:
        """Discover per-point peak CSV files and extract point indices.

        Args:
            fit_dir: Directory containing the domain-specific fit artifacts
            sol: Sol number
            target: Target name
            scan: Scan name
            region: Spectral region (e.g., "R1")
            domain: Fit domain ("minerals", "organics", "hydration")

        Returns:
            List of (csv_path, point_index) tuples
        """
        point_re = re.compile(RAMAN_CSV_PATTERNS[domain])

        results = []
        for csv_path in fit_dir.glob("*.csv"):
            match = point_re.search(csv_path.name)
            if match:
                point_idx = int(match.group(1))
                results.append((csv_path, point_idx))

        # For organics, prefer DG over G-only CSVs for the same point
        if domain == "organics":
            by_point: Dict[int, Tuple[Path, int]] = {}
            for csv_path, point_idx in results:
                is_dg = "_organics_dg_peaks" in csv_path.name
                if point_idx not in by_point or is_dg:
                    by_point[point_idx] = (csv_path, point_idx)
            results = list(by_point.values())

        return results

    def _write_peaks_to_db(self, session, peaks: list, scan_point_ids: list, domain: str) -> int:
        """Idempotent domain-filtered write: delete existing peaks for domain, insert new ones.

        Idempotency key: (scan_point_id, fit_modality). Region is NOT part of the
        delete scope -- all spectrum variants for the given scan points are covered.

        Args:
            session: SQLAlchemy session
            peaks: List of FittedPeakORM objects to insert
            scan_point_ids: List of scan_point_id strings to scope the delete
            domain: fit_modality value for domain-filtered delete

        Returns:
            Number of peaks inserted
        """
        from sherloc_pipeline.database.models import FittedPeakORM, SpectrumORM

        # Get ALL spectrum_ids for scan points (not just preferred ones)
        all_spectrum_ids = session.query(SpectrumORM.id).filter(
            SpectrumORM.scan_point_id.in_(scan_point_ids)
        ).all()

        if all_spectrum_ids:
            # Delete existing peaks for this domain across ALL spectrum variants
            deleted = session.query(FittedPeakORM).filter(
                FittedPeakORM.spectrum_id.in_([s[0] for s in all_spectrum_ids]),
                FittedPeakORM.fit_modality == domain
            ).delete(synchronize_session='fetch')
            if deleted > 0:
                logger.info(f"Deleted {deleted} existing {domain} peaks")
                self.console.print(f"[yellow]Deleted {deleted} existing {domain} peaks[/yellow]")

        # Bulk insert
        session.add_all(peaks)
        return len(peaks)

    def _load_r2_map(self, fit_dir: Path, results_base: Path, sol: str, target: str, scan: str, region: str, domain: str) -> Dict[int, float]:
        """Load point-level R-squared values from the appropriate source for each domain.

        Args:
            fit_dir: Domain-specific fit directory (e.g., results_base/minerals_fit)
            results_base: Base results directory
            sol: Sol number
            target: Target name
            scan: Scan name
            region: Spectral region
            domain: Fit domain

        Returns:
            Dict mapping point_index to R-squared value
        """
        r2_map: Dict[int, float] = {}

        if domain == "minerals":
            # AICc summary CSV at results_base level
            aicc_path = results_base / f"{sol}_{target}_{scan}_{region}_fit_aicc_summary.csv"
            if aicc_path.exists():
                try:
                    df = pd.read_csv(aicc_path)
                    if 'point' in df.columns and 'r2' in df.columns:
                        r2_map = dict(zip(df['point'].astype(int), df['r2']))
                except Exception as e:
                    logger.warning(f"Failed to load R² from AICc summary: {e}")
            else:
                logger.warning(f"AICc summary not found: {aicc_path}")
                self.console.print(f"[yellow]Warning: AICc summary not found, R² will be None[/yellow]")
        else:
            # Accepted-peaks CSV inside fit subdirectory
            acc_patterns = {
                "organics":  f"{sol}_{target}_{scan}_{region}_organics_accepted_peaks.csv",
                "hydration": f"{sol}_{target}_{scan}_{region}_hydration_accepted_peaks.csv",
            }
            acc_path = fit_dir / acc_patterns[domain]
            if acc_path.exists():
                try:
                    df = pd.read_csv(acc_path)
                    if 'point' in df.columns and 'r2' in df.columns:
                        # Take first R² per point (may vary for hydration OH vs bend)
                        for _, row in df.iterrows():
                            pt = int(row['point'])
                            if pt not in r2_map:
                                r2_map[pt] = float(row['r2'])
                except Exception as e:
                    logger.warning(f"Failed to load R² from accepted-peaks: {e}")
            else:
                logger.info(f"Accepted-peaks CSV not found for {domain}: {acc_path}")

        return r2_map

    def persist_raman_peaks(
        self,
        sol: str,
        target: str,
        scan: str,
        domain: str,
        region: str = "R1",
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> ServiceResult:
        """Persist Raman peaks from CSV artifacts into the phase.db database.

        Discovers per-point peak CSVs for the given domain, applies quality filters
        (SNR, FWHM), assigns feature labels via domain-specific classifiers, and
        writes peaks to the fitted_peaks table via _write_peaks_to_db().

        Idempotent: re-running for a domain replaces only that domain's peaks.
        Other domains' peaks are preserved.

        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan name (e.g., "detail_1")
            domain: Raman domain ("minerals", "organics", "hydration")
            region: Spectral region (default "R1")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.

        Returns:
            ServiceResult with summary, metadata including peak counts

        Raises:
            FittingError: If domain is invalid, database not configured, scan not
                found, or persistence fails
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        if domain not in ("minerals", "organics", "hydration"):
            raise FittingError(
                f"Invalid domain '{domain}'. Use 'minerals', 'organics', or 'hydration'. "
                "For fluorescence, use fit_fluorescence().",
                exit_code=1,
                context={"domain": domain}
            )

        try:
            from sherloc_pipeline.database.connection import get_engine, get_session
            from sherloc_pipeline.database.models import FittedPeakORM, ScanORM, ScanPointORM, SpectrumORM
            from sherloc_pipeline.core.mineral_id import (
                assign_min_id, load_mineral_rules,
                classify_organic_band, classify_hydration_band,
            )

            # Build domain-specific feature assigners
            fit_cfg = self.context.config.fitting or {}
            mineral_rules = load_mineral_rules(
                Path(fit_cfg.get('library_path')) if fit_cfg.get('library_path') else None,
                inline_rules=fit_cfg.get('mineral_rules'),
            )
            raman_assigners = {
                "minerals":  lambda center: assign_min_id(center, mineral_rules),
                "organics":  lambda center: classify_organic_band(center),
                "hydration": lambda center: classify_hydration_band(center),
            }

            # Validate database configuration
            if self._database_path is None:
                raise FittingError(
                    "No database_path configured for peak persistence",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan}
                )

            # Lazy-initialize engine
            if self._engine is None:
                self._engine = get_engine(self._database_path)

            # Resolve scan context to find results directory
            try:
                scan_ctx = resolve_scan_context(
                    sol=sol, target=target, scan=scan,
                    data_dir=data_dir, results_dir=results_dir,
                    context=self.context,
                )
            except (FileNotFoundError, ValueError) as e:
                raise FittingError(
                    f"Failed to resolve scan context: {e}",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan}
                )

            from sherloc_pipeline.core.data_ingestion import DataIngestion
            ingestion = DataIngestion(
                base_data_dir=scan_ctx.base_data_dir,
                results_dir=scan_ctx.results_dir,
                sol=sol, target=target, scan=scan,
            )
            results_base = ingestion.get_results_path(target=target, sol=sol, scan=scan)
            fit_dir = results_base / RAMAN_SUBDIRS[domain]
            if not fit_dir.exists():
                raise FittingError(
                    f"{domain.title()} fit directory not found: {fit_dir}. Run fitting first.",
                    exit_code=1,
                    context={"fit_dir": str(fit_dir)}
                )

            # Discover per-point peak CSVs
            peak_csvs = self._discover_peak_csvs(fit_dir, sol, target, scan, region, domain)
            if not peak_csvs:
                raise FittingError(
                    f"No fitted peak CSVs found in {fit_dir} for domain '{domain}'",
                    exit_code=1,
                    context={"domain": domain, "fit_dir": str(fit_dir)}
                )

            # Load R² values (domain-specific source)
            r2_map = self._load_r2_map(fit_dir, results_base, sol, target, scan, region, domain)

            # Get config thresholds
            min_snr = float(fit_cfg.get('min_snr', 3.0))
            fwhm_min = float(fit_cfg.get('reviewable_fwhm_min_cm1', 25.0))

            # Post-hoc quality filters from config
            posthoc = fit_cfg.get('posthoc_filters', {})
            r2_floor = float(posthoc.get('r2_min', 0.0))
            organics_fwhm_mins = posthoc.get('organics_fwhm', {})
            hydration_center_range = posthoc.get('hydration_center_range',
                                                  fit_cfg.get('hydration_center_range', [3000.0, 3900.0]))

            # Database session
            with get_session(self._engine) as session:
                # Look up scan -- target may use underscores (filesystem) or spaces (DB)
                db_target = target.replace('_', ' ')
                scan_orm = session.query(ScanORM).filter_by(
                    sol_number=int(sol), target=db_target, scan_name=scan
                ).first()
                if scan_orm is None and db_target != target:
                    scan_orm = session.query(ScanORM).filter_by(
                        sol_number=int(sol), target=target, scan_name=scan
                    ).first()
                if scan_orm is None:
                    raise FittingError(
                        f"Scan not found in database: sol={sol}, target={target}, scan={scan}",
                        exit_code=1,
                        context={"sol": sol, "target": target, "scan": scan}
                    )

                # Query scan points
                scan_points = session.query(ScanPointORM).filter_by(scan_id=scan_orm.id).all()

                # Build point_index -> spectrum_id lookup (prefer best available type)
                _spec_candidates = [
                    ("laser_normalized", "normalized"),
                    ("dark_subtracted", "normalized"),
                    ("active", "normalized"),
                    ("dark_subtracted", "raw"),
                    ("active", "raw"),
                ]
                point_to_spectrum: Dict[int, str] = {}
                for scan_point in scan_points:
                    spectrum = None
                    for spec_type, proc_level in _spec_candidates:
                        spectrum = session.query(SpectrumORM).filter_by(
                            scan_point_id=scan_point.id,
                            region=region,
                            spectrum_type=spec_type,
                            processing_level=proc_level,
                        ).first()
                        if spectrum is not None:
                            break
                    if spectrum is None:
                        logger.warning(f"No spectrum found for point {scan_point.point_index}")
                        continue
                    point_to_spectrum[scan_point.point_index] = spectrum.id

                # Build peaks from CSVs with feature assignment
                assigner = raman_assigners[domain]
                all_peaks = []
                warnings_list: List[str] = []
                total_reviewable = 0

                for csv_path, point_idx in peak_csvs:
                    if point_idx not in point_to_spectrum:
                        msg = f"Point {point_idx} not found in database, skipping CSV {csv_path.name}"
                        logger.warning(msg)
                        warnings_list.append(msg)
                        continue

                    spectrum_id = point_to_spectrum[point_idx]

                    # R² > 0 filter: skip entire point if fit has negative R²
                    point_r2 = r2_map.get(point_idx)
                    if point_r2 is not None and point_r2 <= r2_floor:
                        logger.info(f"Skipping point {point_idx}: R²={point_r2:.3f} ≤ {r2_floor}")
                        continue

                    try:
                        peaks_df = pd.read_csv(csv_path)
                    except Exception as e:
                        logger.warning(f"Failed to read {csv_path}: {e}")
                        continue

                    # Filter to reviewable peaks: SNR >= min_snr AND FWHM >= reviewable_fwhm_min_cm1
                    # Also apply sharpness filter if column present (cosmic ray rejection)
                    sharpness_mask = True
                    if 'pass_sharpness' in peaks_df.columns:
                        sharpness_mask = peaks_df['pass_sharpness'].apply(_to_bool_safe)
                    reviewable = peaks_df[
                        (peaks_df['snr'] >= min_snr) &
                        (peaks_df['fwhm_cm1'] >= fwhm_min) &
                        sharpness_mask
                    ]
                    total_reviewable += len(reviewable)

                    for _, row in reviewable.iterrows():
                        assignment = assigner(float(row['center_cm1']))

                        # Domain-specific post-hoc FWHM filter for organics
                        if domain == "organics" and organics_fwhm_mins:
                            min_fwhm_for_band = float(organics_fwhm_mins.get(assignment, 0.0))
                            if float(row['fwhm_cm1']) < min_fwhm_for_band:
                                logger.info(
                                    f"Filtered organics peak at {row['center_cm1']:.1f} cm⁻¹ "
                                    f"({assignment}): FWHM {row['fwhm_cm1']:.1f} < {min_fwhm_for_band}"
                                )
                                continue

                        # Hydration center range gate
                        if domain == "hydration" and hydration_center_range:
                            center = float(row['center_cm1'])
                            if center < hydration_center_range[0] or center > hydration_center_range[1]:
                                logger.info(
                                    f"Filtered hydration peak at {center:.1f} cm⁻¹: "
                                    f"outside [{hydration_center_range[0]}, {hydration_center_range[1]}]"
                                )
                                continue

                        peak = FittedPeakORM(
                            id=str(uuid4()),
                            spectrum_id=spectrum_id,
                            peak_type="gaussian",
                            fit_modality=domain,
                            center_cm1=float(row['center_cm1']),
                            amplitude=float(row['amplitude_a']),
                            fwhm_cm1=float(row['fwhm_cm1']),
                            area=float(row['area']) if pd.notna(row.get('area')) else None,
                            snr=float(row['snr']) if pd.notna(row.get('snr')) else None,
                            fit_quality=r2_map.get(point_idx),
                            mineral_assignment=assignment,
                            # assignment_confidence left as NULL (rule-based)
                        )
                        all_peaks.append(peak)

                # Write to DB via shared helper (handles idempotent delete+insert)
                scan_point_ids = [sp.id for sp in scan_points]
                inserted = self._write_peaks_to_db(session, all_peaks, scan_point_ids, domain)

                self.console.print(
                    f"[bold green]Persisted {inserted} {domain} peaks to database[/bold green]"
                )

                metadata = {
                    "sol": sol,
                    "target": target,
                    "scan": scan,
                    "region": region,
                    "domain": domain,
                    "peak_csvs_found": len(peak_csvs),
                    "total_reviewable_peaks": total_reviewable,
                    "peaks_inserted": inserted,
                    "min_snr_threshold": min_snr,
                    "fwhm_min_threshold": fwhm_min,
                }

                return ServiceResult(
                    summary=f"Persisted {inserted} {domain} peaks for scan {sol}/{target}/{scan}",
                    artifacts=[],
                    warnings=warnings_list,
                    metadata=metadata,
                )

        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to persist {domain} peaks: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan, "domain": domain}
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    def persist_fitted_peaks(
        self,
        sol: str,
        target: str,
        scan: str,
        region: str = "R1",
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> ServiceResult:
        """Persist fitted peaks (DEPRECATED: use persist_raman_peaks with domain='minerals').

        This method is a deprecated wrapper that calls persist_raman_peaks(domain='minerals').

        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan name (e.g., "detail_1")
            region: Spectral region (default "R1")
            data_dir: Base data directory. If None, uses config default.
            results_dir: Results directory. If None, uses config default.

        Returns:
            ServiceResult with summary, metadata including peak counts
        """
        import warnings as _warn_mod
        _warn_mod.warn(
            "persist_fitted_peaks() is deprecated; use persist_raman_peaks(domain='minerals')",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.persist_raman_peaks(
            sol=sol, target=target, scan=scan, domain='minerals',
            region=region, data_dir=data_dir, results_dir=results_dir,
        )

    def fit_fluorescence(
        self,
        sol: str,
        target: str,
        scan: str,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        generate_plots: bool = True,
    ) -> ServiceResult:
        """Fit fluorescence spectra for a scan and persist peaks to database.

        Queries R1/R2/R3 spectra from the database, stitches them via
        stitch_r123_spectrum() (Loupe overlap summation), fits multi-Gaussian
        fluorescence models via fit_fluorescence_spectrum(), assigns group labels
        via assign_fluor_group(), and persists peaks inline via _write_peaks_to_db().

        Execution is structured in three phases for parallelism:
          1. **Prepare** — sequential DB reads, spectrum decompression, R123 stitching
          2. **Fit** — parallel per-point fitting via ProcessPoolExecutor
             (configurable via ``fluorescence_fitting.parallel_workers``)
          3. **Post-process** — sequential group assignment, cross-modal annotation,
             DB persistence, and optional plot generation

        The fitting phase includes an early bail-out: if the maximum feature
        prominence (max − median, after despiking) is below ``snr_threshold ×
        noise_std``, the expensive differential-evolution loop is skipped entirely.

        When generate_plots is True (default), produces per-point fit overlay
        plots (PNG + PDF) and scan-level summary/accepted-peaks CSVs in a
        ``fluorescence_fit/`` subdirectory alongside Raman fit artifacts.

        Args:
            sol: Sol number (e.g., "0293")
            target: Target name (e.g., "Quartier")
            scan: Scan name (e.g., "HDR_1")
            data_dir: Base data directory (unused, kept for API consistency).
            results_dir: Results directory for plot/CSV output.
            generate_plots: If True (default), emit per-point fit plots and
                scan-level summary CSVs. Set False for faster parameter tuning.

        Returns:
            ServiceResult with summary and metadata including peak counts.

        Raises:
            FittingError: If database not configured, scan not found, or fitting fails.
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        try:
            import zlib

            from sherloc_pipeline.database.connection import get_engine, get_session
            from sherloc_pipeline.database.models import (
                FittedPeakORM, ScanORM, ScanPointORM, SpectrumORM,
            )
            from sherloc_pipeline.core.fluor_fitting import fit_fluorescence_spectrum
            from sherloc_pipeline.visualization.fitting_plots import plot_fluor_fit_overlay
            from sherloc_pipeline.core.fluor_id import assign_fluor_group, classify_fluor_peaks, score_cooccurrences
            from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
            from sherloc_pipeline.core.r123_stitching import stitch_r123_spectrum

            # Validate database configuration
            if self._database_path is None:
                raise FittingError(
                    "No database_path configured for fluorescence fitting",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan},
                )

            # Lazy-initialize engine
            if self._engine is None:
                self._engine = get_engine(self._database_path)

            # Load fluorescence config
            fluor_cfg = self.context.config.fluorescence_fitting or {}
            snr_threshold = float(fluor_cfg.get("snr_threshold", 2.0))
            fit_range = tuple(fluor_cfg.get("fit_range", [276.0, 355.0]))
            fwhm_range = tuple(fluor_cfg.get("fwhm_range", [10.0, 40.0]))
            max_peaks = int(fluor_cfg.get("max_peaks", 4))
            saturation_threshold = float(fluor_cfg.get("saturation_threshold", 60000.0))
            saturation_channel_limit = int(fluor_cfg.get("saturation_channel_limit", 5))
            strategy = str(fluor_cfg.get("strategy", "agnostic"))
            min_fwhm_nm = float(fluor_cfg.get("min_fwhm_nm", 8.0))

            # Pre-compute 2148-channel wavelength axis from Loupe calibration
            N_CHANNELS_FULL = 2148
            full_wavelength, _ = calculate_loupe_wavelength_wavenumber(N_CHANNELS_FULL)

            # Resolve output directory for plots and CSVs
            fluor_dir: Optional[Path] = None
            if generate_plots:
                results_base = (
                    (Path(results_dir) if results_dir else self.context.results_root)
                    / target / f"{sol}_{scan}"
                )
                try:
                    fluor_dir = results_base / "fluorescence_fit"
                    fluor_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    fluor_dir = None  # graceful skip if results dir doesn't exist

            with get_session(self._engine) as session:
                # Look up scan (target may use underscores in filesystem, spaces in DB)
                db_target = target.replace("_", " ")
                scan_orm = session.query(ScanORM).filter_by(
                    sol_number=int(sol), target=db_target, scan_name=scan
                ).first()
                if scan_orm is None and db_target != target:
                    scan_orm = session.query(ScanORM).filter_by(
                        sol_number=int(sol), target=target, scan_name=scan
                    ).first()
                if scan_orm is None:
                    raise FittingError(
                        f"Scan not found in database: sol={sol}, target={target}, scan={scan}",
                        exit_code=1,
                        context={"sol": sol, "target": target, "scan": scan},
                    )

                # Query scan points
                scan_points = session.query(ScanPointORM).filter_by(
                    scan_id=scan_orm.id
                ).all()

                if not scan_points:
                    raise FittingError(
                        f"No scan points found for scan {sol}/{target}/{scan}",
                        exit_code=1,
                        context={"sol": sol, "target": target, "scan": scan},
                    )

                all_peaks: List[FittedPeakORM] = []
                warnings_list: List[str] = []
                points_fitted = 0
                points_skipped = 0
                points_saturated = 0
                summary_rows: List[Dict[str, Any]] = []
                accepted_rows: List[Dict[str, Any]] = []
                artifacts: List[Path] = []

                # --- Phase 1: Prepare fitting inputs (sequential, DB reads) ---
                fit_inputs: List[Dict[str, Any]] = []
                for scan_point in scan_points:
                    point_idx = scan_point.point_index

                    # Query R1, R2, R3 spectra (prefer dark_subtracted, fallback active)
                    region_spectra = {}
                    for region in ("R1", "R2", "R3"):
                        for spec_type in ("dark_subtracted", "active"):
                            spec = session.query(SpectrumORM).filter_by(
                                scan_point_id=scan_point.id,
                                region=region,
                                spectrum_type=spec_type,
                            ).first()
                            if spec is not None:
                                region_spectra[region] = spec
                                break

                    # Skip point if any region missing (per spec §3.2)
                    missing = [r for r in ("R1", "R2", "R3") if r not in region_spectra]
                    if missing:
                        warnings_list.append(
                            f"Missing {', '.join(missing)} for point {point_idx}, skipping"
                        )
                        points_skipped += 1
                        summary_rows.append({
                            'point': point_idx, 'n_peaks': 0, 'r2': None,
                            'is_saturated': False, 'groups_detected': '',
                            'status': 'missing_regions',
                        })
                        continue

                    # Link peaks to R2 spectrum per spec §6.3
                    link_spectrum = region_spectra["R2"]

                    # Decompress each region to 2148-channel array and stitch
                    region_arrays = {}
                    for region in ("R1", "R2", "R3"):
                        raw = np.frombuffer(
                            zlib.decompress(region_spectra[region].intensities),
                            dtype=np.float32,
                        )
                        if len(raw) == N_CHANNELS_FULL:
                            region_arrays[region] = raw.astype(np.float64)
                        else:
                            # 716-ch region slice: pad to 2148 at correct offset
                            padded = np.zeros(N_CHANNELS_FULL, dtype=np.float64)
                            offset = {"R1": 0, "R2": 716, "R3": 1432}[region]
                            padded[offset:offset + len(raw)] = raw
                            region_arrays[region] = padded

                    r123 = stitch_r123_spectrum(
                        region_arrays["R1"], region_arrays["R2"], region_arrays["R3"]
                    )

                    fit_inputs.append({
                        'point_idx': point_idx,
                        'r123': r123,
                        'link_spectrum_id': link_spectrum.id,
                        'scan_point_id': scan_point.id,
                    })

                # --- Phase 2: Parallel fitting (CPU-bound, no DB) ---
                # Per-point fluorescence fitting is embarrassingly parallel:
                # each fit_fluorescence_spectrum() call is independent, CPU-bound
                # (differential evolution), and requires no DB access. We fan out
                # across a ProcessPoolExecutor, then collect results for sequential
                # post-processing (group assignment, cross-modal annotation, DB writes).
                #
                # Worker count is configurable via fluorescence_fitting.parallel_workers:
                #   0 = auto (half of CPU cores, balances throughput vs system responsiveness)
                #   1 = sequential (no multiprocessing overhead; safe for constrained envs)
                #   N = explicit worker count
                _fit_fn = partial(
                    fit_fluorescence_spectrum,
                    full_wavelength,
                    fit_range=fit_range,
                    fwhm_range=fwhm_range,
                    max_peaks=max_peaks,
                    snr_threshold=snr_threshold,
                    min_fwhm_nm=min_fwhm_nm,
                    saturation_threshold=saturation_threshold,
                    saturation_channel_limit=saturation_channel_limit,
                    strategy=strategy,
                )

                from sherloc_pipeline.core.utils import resolve_parallel_workers
                n_workers = resolve_parallel_workers(
                    int(fluor_cfg.get("parallel_workers", 0)),
                    len(fit_inputs),
                )

                r123_inputs = [inp['r123'] for inp in fit_inputs]
                fit_results = self._run_parallel(
                    _fit_fn, r123_inputs, n_workers,
                    f"Fluorescence {sol}/{target}/{scan}",
                )

                # --- Phase 3: Post-process results (sequential, DB writes) ---
                for inp, fit_result in zip(fit_inputs, fit_results):
                    point_idx = inp['point_idx']
                    link_spectrum_id = inp['link_spectrum_id']
                    scan_point_id = inp['scan_point_id']

                    if fit_result.fit_skipped:
                        if fit_result.is_saturated:
                            points_saturated += 1
                            status = 'saturated'
                        else:
                            points_skipped += 1
                            status = 'skipped'
                        summary_rows.append({
                            'point': point_idx, 'n_peaks': 0,
                            'r2': fit_result.r2 if fit_result.r2 is not None else None,
                            'is_saturated': fit_result.is_saturated,
                            'groups_detected': '', 'status': status,
                        })
                        continue

                    points_fitted += 1

                    # Assign group labels and apply post-classification rules
                    fluor_groups = [assign_fluor_group(p.center_nm) for p in fit_result.peaks]
                    fluor_groups = classify_fluor_peaks(fluor_groups, fit_result.peaks)

                    # Cross-modal annotation: query Raman peaks for same scan point
                    raman_peaks = session.query(FittedPeakORM).filter(
                        FittedPeakORM.spectrum_id.in_(
                            session.query(SpectrumORM.id).filter_by(
                                scan_point_id=scan_point_id
                            )
                        ),
                        FittedPeakORM.fit_modality.in_(["minerals", "organics", "hydration"]),
                    ).all()
                    raman_assignments = [
                        rp.mineral_assignment for rp in raman_peaks
                        if rp.mineral_assignment is not None
                    ]
                    cooccurrence_scores = score_cooccurrences(fluor_groups, raman_assignments)

                    # Construct FittedPeakORM objects with group assignment + co-occurrence
                    for i, peak in enumerate(fit_result.peaks):
                        group_label = fluor_groups[i]
                        confidence = cooccurrence_scores[i].confidence_boost if i < len(cooccurrence_scores) else None
                        orm_peak = FittedPeakORM(
                            id=str(uuid4()),
                            spectrum_id=link_spectrum_id,
                            peak_type="gaussian",
                            fit_modality="fluorescence",
                            center_nm=peak.center_nm,
                            fwhm_nm=peak.fwhm_nm,
                            amplitude=peak.amplitude,
                            area=peak.area,
                            snr=peak.snr,
                            fit_quality=fit_result.r2,
                            mineral_assignment=group_label,
                            assignment_confidence=confidence,
                            is_saturated=fit_result.is_saturated,
                            # center_cm1, fwhm_cm1 left as NULL for fluorescence
                        )
                        all_peaks.append(orm_peak)

                        # Collect accepted-peak row
                        accepted_rows.append({
                            'point': point_idx,
                            'center_nm': peak.center_nm,
                            'amplitude': peak.amplitude,
                            'fwhm_nm': peak.fwhm_nm,
                            'snr': peak.snr,
                            'group': group_label,
                            'r2': fit_result.r2,
                            'cooccurrence_confidence': confidence,
                        })

                    # Collect summary row for fitted point
                    detected_groups = sorted(set(fluor_groups))
                    summary_rows.append({
                        'point': point_idx,
                        'n_peaks': len(fit_result.peaks),
                        'r2': fit_result.r2,
                        'is_saturated': fit_result.is_saturated,
                        'groups_detected': ','.join(detected_groups),
                        'status': 'fitted',
                    })

                    # Per-point fit overlay plot
                    if fluor_dir is not None and len(fit_result.peaks) > 0:
                        try:
                            png_path = fluor_dir / f"{sol}_{target}_{scan}_point{point_idx}_fluor_fit.png"
                            plot_fluor_fit_overlay(
                                wavelength=full_wavelength,
                                intensity=inp['r123'],
                                result=fit_result,
                                output_png_path=str(png_path),
                                xlim=(fit_range[0] - 5, fit_range[1] + 5),
                                saturation_threshold=saturation_threshold,
                                sol=sol, target=target, scan=scan, point=point_idx,
                            )
                            artifacts.append(png_path)
                        except Exception as e:
                            logger.warning(f"Failed to generate fluor plot for point {point_idx}: {e}")

                # Persist via shared helper (idempotent domain-filtered delete+insert)
                scan_point_ids = [sp.id for sp in scan_points]
                inserted = self._write_peaks_to_db(
                    session, all_peaks, scan_point_ids, "fluorescence"
                )

                self.console.print(
                    f"[bold green]Persisted {inserted} fluorescence peaks to database[/bold green]"
                )

                # Write scan-level CSVs
                if fluor_dir is not None:
                    try:
                        if summary_rows:
                            summary_df = pd.DataFrame(summary_rows).sort_values('point')
                            summary_csv = fluor_dir / f"{sol}_{target}_{scan}_fluor_summary.csv"
                            summary_df.to_csv(summary_csv, index=False)
                            artifacts.append(summary_csv)
                        if accepted_rows:
                            acc_df = pd.DataFrame(accepted_rows).sort_values(['point', 'center_nm'])
                            acc_csv = fluor_dir / f"{sol}_{target}_{scan}_fluor_accepted_peaks.csv"
                            acc_df.to_csv(acc_csv, index=False)
                            artifacts.append(acc_csv)
                    except Exception as e:
                        logger.warning(f"Failed to write fluorescence CSVs: {e}")

                metadata = {
                    "sol": sol,
                    "target": target,
                    "scan": scan,
                    "total_scan_points": len(scan_points),
                    "points_fitted": points_fitted,
                    "points_skipped": points_skipped,
                    "points_saturated": points_saturated,
                    "peaks_inserted": inserted,
                    "snr_threshold": snr_threshold,
                    "plots_generated": len([a for a in artifacts if str(a).endswith('.png')]),
                    "output_dir": str(fluor_dir) if fluor_dir else None,
                }

                return ServiceResult(
                    summary=(
                        f"Fitted fluorescence for {points_fitted}/{len(scan_points)} points, "
                        f"{inserted} peaks persisted"
                    ),
                    artifacts=artifacts,
                    warnings=warnings_list,
                    metadata=metadata,
                )

        except FittingError:
            raise
        except Exception as e:
            error = FittingError(
                f"Failed to fit fluorescence: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    def extract_training_jsonl(
        self,
        output_path: Path,
        snr_threshold: float = 2.0,
    ) -> ServiceResult:
        """Extract unified training data as JSONL from fitted_peaks table.

        Queries all peaks with SNR >= snr_threshold, groups by scan point,
        builds cross-modal feature records with Raman + fluorescence peaks
        and doublet detection per spec §9.3.

        Args:
            output_path: Path to write JSONL output file.
            snr_threshold: Minimum SNR for peak inclusion (default 2.0).

        Returns:
            ServiceResult with record count and output path.
        """
        import json
        from collections import defaultdict
        from sherloc_pipeline.database.connection import get_engine, get_session
        from sherloc_pipeline.database.models import (
            FittedPeakORM, SpectrumORM, ScanPointORM, ScanORM,
        )
        from sherloc_pipeline.core.fluor_id import detect_doublets

        if self._database_path is None:
            raise FittingError(
                "Database path required for training data extraction",
                exit_code=1,
            )

        engine = get_engine(self._database_path)

        with get_session(engine) as session:
            # Query per spec §9.1
            rows = (
                session.query(
                    FittedPeakORM,
                    ScanPointORM.point_index,
                    ScanORM.sol_number,
                    ScanORM.target,
                    ScanORM.scan_name,
                    SpectrumORM.region,
                    SpectrumORM.spectrum_type,
                )
                .join(SpectrumORM, SpectrumORM.id == FittedPeakORM.spectrum_id)
                .join(ScanPointORM, ScanPointORM.id == SpectrumORM.scan_point_id)
                .join(ScanORM, ScanORM.id == ScanPointORM.scan_id)
                .filter(FittedPeakORM.snr >= snr_threshold)
                .order_by(
                    ScanORM.sol_number,
                    ScanORM.target,
                    ScanORM.scan_name,
                    ScanPointORM.point_index,
                    FittedPeakORM.fit_modality,
                )
                .all()
            )

            # Group by scan point
            groups: Dict[tuple, List[FittedPeakORM]] = defaultdict(list)
            for peak_orm, point_idx, sol_num, tgt, scan_name, _region, _spec_type in rows:
                key = (sol_num, tgt or "", scan_name, point_idx)
                groups[key].append(peak_orm)

            # Build JSONL records
            records: List[dict] = []
            total_doublets = 0

            for (sol_num, tgt, scan_name, point_idx), peaks in groups.items():
                raman_peaks = [
                    p for p in peaks
                    if p.fit_modality in ("minerals", "organics", "hydration")
                ]
                fluor_peaks = [
                    p for p in peaks if p.fit_modality == "fluorescence"
                ]

                if not raman_peaks and not fluor_peaks:
                    continue

                # Build input text per spec §9.3
                parts = [f"Point {point_idx}, Sol {sol_num} {tgt} {scan_name}."]

                # Raman peaks
                raman_descs = []
                for p in raman_peaks:
                    label = p.mineral_assignment or "unknown"
                    center = p.center_cm1 if p.center_cm1 is not None else 0.0
                    snr_val = p.snr if p.snr is not None else 0.0
                    raman_descs.append(
                        f"{label} {center:.1f} cm-1 (SNR {snr_val:.0f})"
                    )
                if raman_descs:
                    parts.append("Raman: " + ", ".join(raman_descs) + ".")

                # Fluorescence peaks
                fluor_descs = []
                for p in fluor_peaks:
                    group = p.mineral_assignment or "unidentified"
                    center = p.center_nm if p.center_nm is not None else 0.0
                    snr_val = p.snr if p.snr is not None else 0.0
                    fluor_descs.append(
                        f"{group} {center:.1f} nm (SNR {snr_val:.0f})"
                    )
                if fluor_descs:
                    parts.append("Fluor: " + ", ".join(fluor_descs) + ".")

                # Doublet detection
                doublets = detect_doublets(fluor_peaks) if fluor_peaks else []
                for d in doublets:
                    parts.append(
                        f"Doublet ratio {d.intensity_ratio:.2f}, "
                        f"sep {d.separation_nm:.1f} nm."
                    )
                    total_doublets += 1

                input_text = " ".join(parts)
                output_text = _build_phase_label(
                    raman_peaks, fluor_peaks, doublets
                )

                records.append({"input": input_text, "output": output_text})

        # Write JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return ServiceResult(
            summary=(
                f"Extracted {len(records)} training records to {output_path}"
            ),
            artifacts=[output_path],
            metadata={
                "total_records": len(records),
                "total_peaks_queried": len(rows),
                "total_doublets": total_doublets,
                "snr_threshold": snr_threshold,
            },
        )

    def query_co_occurrences(
        self,
        raman_modality: str = "minerals",
        raman_assignment_pattern: str = "sulf%",
        fluor_groups: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find scan points with co-occurring Raman and fluorescence detections.

        Implements spec §9.2 co-occurrence query. Uses the composite index
        ix_fitted_peaks_modality_assignment for efficient lookups.

        Args:
            raman_modality: Raman fit_modality to search (default 'minerals').
            raman_assignment_pattern: SQL LIKE pattern for mineral_assignment
                (default 'sulf%' for sulfates).
            fluor_groups: Fluorescence group names to match (default
                ['group1a', 'group1b'] for Ce3+).

        Returns:
            List of dicts with scan_point_id, sol_number, target, point_index.
        """
        from sherloc_pipeline.database.connection import get_engine, get_session
        from sherloc_pipeline.database.models import (
            FittedPeakORM, SpectrumORM, ScanPointORM, ScanORM,
        )

        if fluor_groups is None:
            fluor_groups = ["group1a", "group1b"]

        if self._database_path is None:
            raise FittingError(
                "Database path required for co-occurrence query",
                exit_code=1,
            )

        if self._engine is None:
            self._engine = get_engine(self._database_path)

        with get_session(self._engine) as session:
            # Subquery: scan point IDs with matching Raman peaks
            raman_subq = (
                session.query(ScanPointORM.id)
                .join(SpectrumORM, SpectrumORM.scan_point_id == ScanPointORM.id)
                .join(FittedPeakORM, FittedPeakORM.spectrum_id == SpectrumORM.id)
                .filter(
                    FittedPeakORM.fit_modality == raman_modality,
                    FittedPeakORM.mineral_assignment.like(raman_assignment_pattern),
                )
            )

            # Main query: scan points with fluorescence in those groups
            results = (
                session.query(
                    ScanPointORM.id.label("scan_point_id"),
                    ScanORM.sol_number,
                    ScanORM.target,
                    ScanPointORM.point_index,
                )
                .select_from(ScanPointORM)
                .join(ScanORM, ScanORM.id == ScanPointORM.scan_id)
                .join(SpectrumORM, SpectrumORM.scan_point_id == ScanPointORM.id)
                .join(FittedPeakORM, FittedPeakORM.spectrum_id == SpectrumORM.id)
                .filter(
                    ScanPointORM.id.in_(raman_subq.subquery().select()),
                    FittedPeakORM.fit_modality == "fluorescence",
                    FittedPeakORM.mineral_assignment.in_(fluor_groups),
                )
                .distinct()
                .all()
            )

            return [
                {
                    "scan_point_id": r.scan_point_id,
                    "sol_number": r.sol_number,
                    "target": r.target,
                    "point_index": r.point_index,
                }
                for r in results
            ]


def _build_phase_label(
    raman_peaks: list,
    fluor_peaks: list,
    doublets: list,
) -> str:
    """Auto-generate a phase label from peak detections for training data.

    Combines Raman mineral assignments and fluorescence group labels into
    a human-readable phase identification string.
    """
    evidence: List[str] = []
    phase_parts: List[str] = []

    # Collect unique Raman mineral assignments
    mineral_labels = set()
    for p in raman_peaks:
        if p.mineral_assignment:
            mineral_labels.add(p.mineral_assignment)

    for label in sorted(mineral_labels):
        if "sulf" in label:
            if "sulfate" not in phase_parts:
                phase_parts.append("sulfate")
            evidence.append(f"Ca-sulfate {label} Raman")
        elif "carb" in label:
            if "carbonate" not in phase_parts:
                phase_parts.append("carbonate")
            evidence.append(f"carbonate {label} Raman")
        elif "oliv" in label:
            if "olivine" not in phase_parts:
                phase_parts.append("olivine")
            evidence.append(f"olivine {label} Raman")
        elif "phos" in label:
            if "phosphate" not in phase_parts:
                phase_parts.append("phosphate")
            evidence.append(f"phosphate {label} Raman")
        elif "D_band" in label or "G_band" in label:
            if "organic carbon" not in phase_parts:
                phase_parts.append("organic carbon")
            evidence.append(f"organic {label}")
        elif "OH_stretch" in label or "H2O_bend" in label:
            if "hydrated phase" not in phase_parts:
                phase_parts.append("hydrated phase")
            evidence.append(f"hydration {label}")
        else:
            evidence.append(f"{label} Raman")

    # Fluorescence groups
    fluor_groups = set()
    for p in fluor_peaks:
        if p.mineral_assignment:
            fluor_groups.add(p.mineral_assignment)

    if doublets:
        phase_parts.insert(0, "Ce3+-bearing")
        evidence.append("Ce3+ fluorescent doublet")
    elif fluor_groups & {"group1a", "group1b"}:
        phase_parts.insert(0, "Ce3+-bearing")
        evidence.append("Ce3+ fluorescence")

    if "group2" in fluor_groups:
        evidence.append("phosphate Ce3+ fluorescence")
    if "group3" in fluor_groups:
        evidence.append("silicate defect fluorescence")

    phase = " ".join(phase_parts) if phase_parts else "unidentified phase"
    ev = " + ".join(evidence) if evidence else "no diagnostic features"

    return f"Phase: {phase}. Evidence: {ev}."
