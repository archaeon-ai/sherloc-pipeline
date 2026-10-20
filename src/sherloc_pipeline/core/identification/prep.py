"""Versioned mask-aware preparation derived from prototype/model.py, without data I/O."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from sherloc_pipeline.models.identification import MethodRecipe


@dataclass
class Prepared:
    vector: np.ndarray | None
    mask: np.ndarray
    diagnostics: dict


def grid_for(recipe: MethodRecipe) -> np.ndarray:
    start, stop, step = recipe.grid
    return np.arange(start, stop + step / 2, step)


def baseline(y: np.ndarray, mask: np.ndarray, recipe: MethodRecipe) -> np.ndarray:
    d = diags([np.ones(len(y)-2), -2*np.ones(len(y)-2), np.ones(len(y)-2)],
              [0, 1, 2], shape=(len(y)-2, len(y)), format="csc")
    penalty = recipe.baseline_lambda * (d.T @ d)
    y = np.where(mask, y, 0.)
    weights = mask.astype(float)
    for _ in range(recipe.baseline_iterations):
        z = spsolve(diags(weights, format="csc") + penalty, weights * y)
        weights = np.where(mask, np.where(y > z, recipe.baseline_asymmetry, 1-recipe.baseline_asymmetry), 0.)
    return z


def prepare(x, y, recipe: MethodRecipe) -> Prepared:
    grid = grid_for(recipe)
    mask = np.zeros(len(grid), dtype=bool)

    def reject(reason, **details):
        return Prepared(None, mask, {"reason": reason, **details})

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) != len(y)
            or len(x) < recipe.min_samples or not np.isfinite(x).all()):
        return reject("invalid_axis")
    if not np.isfinite(y).all():
        return reject("nonfinite_intensity")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    y = np.bincount(inverse, weights=y) / counts
    if len(x) < recipe.min_samples:
        return reject("insufficient_unique_axis_values")
    native = (x >= grid.min()) & (x <= grid.max())
    if native.sum() < recipe.min_samples:
        return reject("insufficient_native_samples")
    margin = max(abs(shift) for shift in recipe.shifts)
    mask = (grid >= x.min() + margin) & (grid <= x.max() - margin)
    native &= (x >= x.min() + margin) & (x <= x.max() - margin)
    for i in np.flatnonzero(np.diff(x) > recipe.max_gap):
        mask &= ~((grid > x[i]) & (grid < x[i+1]))
    if mask.mean() < recipe.min_coverage:
        return reject("insufficient_valid_overlap", coverage=float(mask.mean()))
    for low, high in recipe.excluded_intervals:
        mask &= ~((grid >= low) & (grid <= high))
        native &= ~((x >= low) & (x <= high))
    indices = np.flatnonzero(native)
    breaks = np.flatnonzero((np.diff(indices) > 1) | (np.diff(x[indices]) > recipe.max_gap)) + 1
    segments = [segment for segment in np.split(indices, breaks) if len(segment) >= 2]
    if sum(map(len, segments)) < recipe.min_samples:
        return reject("insufficient_native_samples")
    z = np.full(len(grid), np.nan)
    for segment in segments:
        inside = (grid >= x[segment[0]]) & (grid <= x[segment[-1]])
        z[inside] = np.interp(grid[inside], x[segment], y[segment], left=np.nan, right=np.nan)
    mask &= np.isfinite(z)
    if mask.sum() < 2:
        return reject("insufficient_valid_overlap", coverage=float(mask.mean()))
    valid_y = np.concatenate([y[segment] for segment in segments])
    span = np.ptp(valid_y)
    if span <= max(1e-12, np.max(np.abs(valid_y)) * 1e-10):
        return reject("constant_signal")
    high = y >= valid_y.max() - max(1e-10, span * 1e-6)
    if any(np.any(np.convolve(high[segment].astype(int), np.ones(3), mode="valid") >= 3)
           for segment in segments if len(segment) >= 3):
        return reject("flat_top_possible_saturation")
    diffs = np.concatenate([np.diff(y[segment]) for segment in segments])
    noise = max(1e-12, 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2))
    signal = z - baseline(z, mask, recipe)
    signal[~mask] = 0
    norm = np.linalg.norm(signal[mask])
    if not np.isfinite(norm) or norm <= 1e-12:
        return reject("no_detectable_feature")
    return Prepared(signal / norm, mask, {
        "reason": None, "noise_native": float(noise),
        "snr_proxy": float(np.percentile(signal[mask], 95) / noise),
        "masked_bins": int((~mask).sum()), "native_points": len(x),
        "norm": float(norm), "cosmic_mask_available": False,
    })
