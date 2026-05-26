from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pybaselines import Baseline

# Provenance: asPLS baseline parameters (lambda=1e6, asymmetry=0.01, protected
# mineral windows) developed by domain expert through iterative testing against
# known mineral standards. Not AI-generated or AI-tuned.


@dataclass
class BaselineParams:
    lam: float = 1e6
    asymmetric_coef: float = 0.01
    iters: int = 10
    diff_order: int = 2
    tol: float = 1e-3


def fit_baseline(series: pd.Series, params: BaselineParams, weights: Optional[np.ndarray] = None) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    y = series.astype(float).values
    bl = Baseline()
    baseline, _ = bl.aspls(
        y,
        lam=params.lam,
        diff_order=params.diff_order,
        max_iter=params.iters,
        tol=params.tol,
        asymmetric_coef=params.asymmetric_coef,
        weights=weights,
    )
    corrected = y - baseline
    return pd.Series(corrected, index=series.index), pd.Series(baseline, index=series.index)


def fit_baseline_window(
    x: np.ndarray,
    y: np.ndarray,
    roi: Tuple[float, float],
    params: BaselineParams,
    weights_builder: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    method: str = "aspls",
    poly_degree: int = 1,
) -> np.ndarray:
    mask = (x >= roi[0]) & (x <= roi[1])
    x_win = x[mask]
    y_win = y[mask]

    if method.lower() == "poly":
        # Build exclusion mask for protected bands; fit linear baseline on remaining points
        keep = np.ones_like(x_win, dtype=bool)
        if weights_builder is not None:
            for center, halfwidth, _ in weights_builder:
                keep &= ~((x_win >= center - halfwidth) & (x_win <= center + halfwidth))
        # Require sufficient points to fit; fallback to all if over-pruned
        if keep.sum() < 5:
            keep = np.ones_like(x_win, dtype=bool)
        deg = max(1, int(poly_degree))
        # Guard against singular matrices; fall back to lower degree
        try:
            coeffs = np.polyfit(x_win[keep], y_win[keep], deg=deg)
        except Exception:
            coeffs = np.polyfit(x_win[keep], y_win[keep], deg=1)
        baseline_win = np.polyval(coeffs, x_win)
        y_corr = pd.Series(y_win - baseline_win, index=pd.RangeIndex(len(y_win)))
    else:
        weights = None
        if weights_builder is not None:
            weights = np.ones_like(x_win, dtype=float)
            for center, halfwidth, w in weights_builder:
                weights[(x_win >= center - halfwidth) & (x_win <= center + halfwidth)] = w
        y_corr, _ = fit_baseline(pd.Series(y_win), params, weights)

    y_out = y.copy()
    y_out[mask] = y_corr.values
    return y_out


def baseline_aspls(intensity_series: pd.Series, params: BaselineParams) -> Tuple[pd.Series, pd.Series]:
    """Fit and subtract baseline using asPLS.

    Args:
        intensity_series: Intensity values as pandas Series
        params: BaselineParams with asPLS hyperparameters

    Returns:
        Tuple of (corrected_series, baseline_series)
    """
    if not isinstance(intensity_series, pd.Series):
        raise TypeError("intensity_series must be a pandas Series")

    y = intensity_series.astype(float).values
    bl = Baseline()
    baseline, _ = bl.aspls(
        y,
        lam=params.lam,
        diff_order=params.diff_order,
        max_iter=params.iters,
        tol=params.tol,
        asymmetric_coef=params.asymmetric_coef,
    )
    corrected = y - baseline
    return pd.Series(corrected, index=intensity_series.index), pd.Series(baseline, index=intensity_series.index)


def _baseline_aspls_with_weights(
    intensity_series: pd.Series, params: BaselineParams, weights: np.ndarray
) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(intensity_series, pd.Series):
        raise TypeError("intensity_series must be a pandas Series")
    y = intensity_series.astype(float).values
    bl = Baseline()
    baseline, _ = bl.aspls(
        y,
        lam=params.lam,
        diff_order=params.diff_order,
        max_iter=params.iters,
        tol=params.tol,
        asymmetric_coef=params.asymmetric_coef,
        weights=weights,
    )
    corrected = y - baseline
    return pd.Series(corrected, index=intensity_series.index), pd.Series(baseline, index=intensity_series.index)


def build_weight_vector_from_windows(
    raman_shift: np.ndarray,
    keep_windows: List[Tuple[float, float]] | Tuple[Tuple[float, float], ...],
    default_weight: float = 1.0,
    keep_weight: float = 0.01,
) -> np.ndarray:
    """Construct an asPLS penalty weight vector from explicit windows to preserve peaks.

    Any sample whose Raman shift lies inside a "keep window" is assigned the
    smaller ``keep_weight`` (reduces baseline influence), otherwise ``default_weight``.

    This mirrors the simpler notebook method ``create_custom_penalty_weight_vector``.
    """
    w = np.full_like(raman_shift, float(default_weight), dtype=float)
    for lo, hi in keep_windows:
        mask = (raman_shift >= float(lo)) & (raman_shift <= float(hi))
        w[mask] = np.minimum(w[mask], float(keep_weight))
    return w

