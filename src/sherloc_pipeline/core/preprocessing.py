"""
Data preprocessing module for SHERLOC pipeline.

This module handles the conversion of Loupe format data to standard spectral format,
including data restructuring, baseline correction, and cosmic ray removal.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from sherloc_pipeline.core.baseline import (
    BaselineParams,
    baseline_aspls,
    _baseline_aspls_with_weights,
    build_weight_vector_from_windows,
)

# Provenance: Cosmic ray detection algorithm (rolling-median + MAD with sulfate guard)
# developed by domain expert (K. Williford) from literature review and iterative
# experimental validation. Not AI-generated or AI-tuned.


@dataclass
class DespikeParams:
    """Configuration for spectral despiking.

    Attributes:
        window_size: Size of the rolling window used to compute the local median.
        zscore_threshold: Robust z-score threshold (relative to MAD-based sigma)
            used to classify a sample as a spike.
        max_iterations: Maximum number of despiking passes. Each pass re-computes
            the local statistics after interpolating previously flagged spikes.
        interpolation_method: Interpolation strategy used to fill flagged spikes.
            Typical values are "linear" and "spline" (order-specific methods
            may require additional dependencies/settings).
    """

    window_size: int = 7
    zscore_threshold: float = 6.0
    max_iterations: int = 1
    interpolation_method: str = "linear"
    run_length_max: int = 2
    # Exclusions and guards (R1, Raman shift cm^-1)
    laser_window: Tuple[float, float] = (600.0, 700.0)
    sulfate_center_window: Tuple[float, float] = (1014.0, 1020.0)
    sulfate_guard_enable: bool = True
    sulfate_guard_search: Tuple[float, float] = (990.0, 1050.0)
    sulfate_guard_min_prominence: float = 100.0
    sulfate_guard_min_halfwidth: float = 15.0
    sulfate_guard_max_halfwidth: float = 25.0


def despike_r1_spectrum(
    intensity_series: pd.Series,
    params: DespikeParams,
    raman_shift: Optional[np.ndarray] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Remove spike artifacts from a single R1 spectrum.

    Uses a robust rolling-median residual with a MAD-derived sigma to identify
    outliers and replaces them via interpolation. Designed to preserve genuine
    spectral peaks while eliminating narrow, high-amplitude spikes.

    Args:
        intensity_series: One-dimensional series of intensities indexed by
            spectral coordinate or integer position.
        params: DespikeParams controlling windowing and thresholds.

    Returns:
        A tuple of (despiked_series, spike_mask), where spike_mask is a boolean
        series indicating the locations that were considered spikes at any
        iteration.
    """

    if not isinstance(intensity_series, pd.Series):
        raise TypeError("intensity_series must be a pandas Series")

    if params.window_size < 3 or params.window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    working_series: pd.Series = intensity_series.astype(float).copy()
    any_spike_mask = pd.Series(False, index=working_series.index)

    # Build exclusion mask in spectral coordinates if provided
    exclude_mask: Optional[np.ndarray] = None
    if raman_shift is not None and len(raman_shift) == len(working_series):
        exclude_mask = np.zeros_like(raman_shift, dtype=bool)
        def apply_exclude(rng: Tuple[float, float]):
            exclude_mask[:] |= (raman_shift >= rng[0]) & (raman_shift <= rng[1])
        if params.laser_window:
            apply_exclude(params.laser_window)
        if params.sulfate_center_window:
            apply_exclude(params.sulfate_center_window)
        # Dynamic sulfate guard
        if params.sulfate_guard_enable and params.sulfate_guard_search:
            search = (raman_shift >= params.sulfate_guard_search[0]) & (raman_shift <= params.sulfate_guard_search[1])
            if np.any(search):
                y_sg = savgol_filter(working_series.values, max(5, (params.window_size//2)*2+1), 2, mode="interp")
                local_idx = np.where(search)[0]
                y_local = y_sg[local_idx]
                peaks, props = find_peaks(y_local, prominence=params.sulfate_guard_min_prominence)
                if peaks.size > 0:
                    # take the most prominent peak
                    k = int(peaks[np.argmax(props.get('prominences', np.zeros_like(peaks)))])
                    center = raman_shift[local_idx[k]]
                    halfw = float(params.sulfate_guard_min_halfwidth)
                    halfw = max(params.sulfate_guard_min_halfwidth, min(params.sulfate_guard_max_halfwidth, halfw))
                    apply_exclude((center - halfw, center + halfw))

    for _ in range(max(1, params.max_iterations)):
        rolling_median = (
            working_series.rolling(
                window=params.window_size, center=True, min_periods=1
            ).median()
        )

        residual = working_series - rolling_median
        mad = residual.abs().median()
        robust_sigma = 1.4826 * mad if mad > 0 else residual.abs().std(ddof=0)
        if robust_sigma == 0 or not np.isfinite(robust_sigma):
            # No variance; nothing to despike
            break

        spike_mask = residual.abs() > (params.zscore_threshold * robust_sigma)
        # Apply exclusion mask
        if exclude_mask is not None:
            spike_mask.loc[exclude_mask] = False
        # Run-length cap: allow only short, impulse-like spikes
        if spike_mask.any() and params.run_length_max > 0:
            mask_np = spike_mask.values.copy()
            # find runs of True
            diff = np.diff(np.concatenate(([0], mask_np.view(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                run_len = e - s
                if run_len > params.run_length_max:
                    mask_np[s:e] = False
            spike_mask = pd.Series(mask_np, index=spike_mask.index)
        if not spike_mask.any():
            break

        any_spike_mask = any_spike_mask | spike_mask
        working_series.loc[spike_mask] = np.nan

        # Interpolate flagged samples; use both directions to fill leading/trailing NaNs
        working_series = working_series.interpolate(
            method=params.interpolation_method, limit_direction="both"
        )

    return working_series, any_spike_mask


def despike_r1_dataframe(
    spectra_df: pd.DataFrame, params: DespikeParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply despiking to all point columns in an R1 spectra DataFrame.

    Expects a DataFrame shaped like the pipeline's R1 structure with a
    `raman_shift` column plus integer point columns (0..n-1). The
    `raman_shift` axis is preserved unchanged. Each point column is processed
    independently using `despike_r1_spectrum`.

    Args:
        spectra_df: Input DataFrame with columns: `raman_shift`, 0..N-1
        params: DespikeParams configuration

    Returns:
        Tuple of (despiked_df, spike_mask_df) where:
            - despiked_df preserves `raman_shift` and includes despiked point
              columns
            - spike_mask_df has boolean masks per point column
    """

    if 'raman_shift' not in spectra_df.columns:
        raise ValueError("Expected 'raman_shift' column in R1 spectra DataFrame")

    # Identify point columns as integers
    point_cols = [col for col in spectra_df.columns if isinstance(col, int)]
    if not point_cols:
        raise ValueError("No integer point columns found for despiking")

    raman_shift = spectra_df['raman_shift'].astype(float)
    raman = raman_shift.values

    cleaned_columns: dict[int, pd.Series] = {}
    mask_columns: dict[int, pd.Series] = {}

    for col in point_cols:
        cleaned_series, mask = despike_r1_spectrum(spectra_df[col], params, raman_shift=raman)
        cleaned_columns[col] = cleaned_series.astype(float)
        mask_columns[col] = pd.Series(mask, index=spectra_df.index)

    despiked_df = pd.concat(
        [pd.DataFrame({'raman_shift': raman_shift}), pd.DataFrame(cleaned_columns)], axis=1
    )
    spike_mask_df = pd.DataFrame(mask_columns, index=spectra_df.index)

    # Preserve column ordering: raman_shift then sorted point columns
    sorted_point_cols = sorted(point_cols)
    despiked_df = despiked_df[['raman_shift'] + sorted_point_cols]
    spike_mask_df = spike_mask_df[sorted_point_cols]

    return despiked_df, spike_mask_df


## baseline_aspls, _baseline_aspls_with_weights, build_weight_vector_from_windows
## are re-exported from sherloc_pipeline.core.baseline (imported above).


def baseline_r1_dataframe(
    spectra_df: pd.DataFrame, params: BaselineParams, weights: np.ndarray | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply asPLS baseline correction to all R1 point columns.

    Expects a DataFrame with `raman_shift` and integer point columns. Returns
    two DataFrames: the baseline-corrected spectra and the fitted baselines.

    Args:
        spectra_df: Input R1 DataFrame (normalized or despiked variant)
        params: BaselineParams hyperparameters

    Returns:
        Tuple of (corrected_df, baseline_df)
    """

    if 'raman_shift' not in spectra_df.columns:
        raise ValueError("Expected 'raman_shift' column for R1 baseline correction")

    point_cols = [col for col in spectra_df.columns if isinstance(col, int)]
    if not point_cols:
        raise ValueError("No integer point columns found for baseline correction")

    raman_shift = spectra_df['raman_shift'].astype(float)

    corrected_columns: dict[int, pd.Series] = {}
    baseline_columns: dict[int, pd.Series] = {}

    for col in point_cols:
        corrected, baseline = (
            baseline_aspls(spectra_df[col], params)
            if weights is None
            else _baseline_aspls_with_weights(spectra_df[col], params, weights)
        )
        corrected_columns[col] = corrected.astype(float)
        baseline_columns[col] = baseline.astype(float)

    corrected_df = pd.concat(
        [pd.DataFrame({'raman_shift': raman_shift}), pd.DataFrame(corrected_columns)], axis=1
    )
    baseline_df = pd.concat(
        [pd.DataFrame({'raman_shift': raman_shift}), pd.DataFrame(baseline_columns)], axis=1
    )

    sorted_point_cols = sorted(point_cols)
    corrected_df = corrected_df[['raman_shift'] + sorted_point_cols]
    baseline_df = baseline_df[['raman_shift'] + sorted_point_cols]

    return corrected_df, baseline_df






