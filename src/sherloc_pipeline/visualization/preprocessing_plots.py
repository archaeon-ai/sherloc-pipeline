"""Preprocessing verification and diagnostic plots."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_r1_despike_verification_single(
    r1_df: pd.DataFrame,
    r1_despiked_df: pd.DataFrame,
    spike_mask_df: pd.DataFrame,
    output_dir: Path,
    sol: str,
    target: str,
    scan: str,
    point: int,
    threshold: float,
) -> Path:
    """Create a single verification plot for raw vs despiked at a selected point."""

    output_dir.mkdir(parents=True, exist_ok=True)
    x = r1_df['raman_shift'].values

    y_raw = r1_df[point].values.astype(float)
    y_clean = r1_despiked_df[point].values.astype(float)
    spike_mask = spike_mask_df[point].values if point in spike_mask_df.columns else np.zeros_like(y_raw, dtype=bool)

    fig, ax = plt.subplots(figsize=(12, 6))
    order = np.argsort(x)
    x_ordered = x[order]
    y_raw_ordered = y_raw[order]
    y_clean_ordered = y_clean[order]
    spike_ordered = spike_mask[order]

    ax.plot(x_ordered, y_raw_ordered, color="#1f77b4", linewidth=1.2, alpha=0.8, label="Raw")
    ax.plot(x_ordered, y_clean_ordered, color="#2ca02c", linewidth=1.2, alpha=0.9, label="Despiked")
    if spike_ordered.any():
        ax.scatter(x_ordered[spike_ordered], y_raw_ordered[spike_ordered], s=12, color="#d62728", label="Spikes")

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"R1 Despike Verification - Point {point} - thresh={threshold}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    try:
        x0 = float(x_ordered[0])
    except Exception:
        x0 = 0.0
    ax.set_xlim([x0, 4000])
    plt.tight_layout()

    out_path = output_dir / f"{sol}_{target}_{scan}_{point}_thresh{int(threshold)}_R1_despike_test.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_r1_baseline_verification_single(
    r1_df: pd.DataFrame,
    r1_corrected_df: pd.DataFrame,
    r1_baseline_df: pd.DataFrame,
    output_dir: Path,
    sol: str,
    target: str,
    scan: str,
    point: int,
    variant: str = "raw",
) -> Path:
    """Create a 2×2 region-span verification plot for a single point.

    variant: "raw" for baseline on normalized-R1, "despiked" for baseline on
    despiked-R1. Filename: {sol}_{target}_{scan}_{point}_R1_{variant}_baseline_test.png
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    point_cols = [c for c in r1_df.columns if isinstance(c, int)]
    if point not in point_cols:
        # fallback to first available
        point = sorted(point_cols)[0]
    x = r1_df['raman_shift'].values
    saved_paths: List[Path] = []

    y_raw = r1_df[point].values.astype(float)
    y_corr = r1_corrected_df[point].values.astype(float)
    y_base = r1_baseline_df[point].values.astype(float)

    order = np.argsort(x)
    x_ord = x[order]
    y_raw_ord = y_raw[order]
    y_corr_ord = y_corr[order]
    y_base_ord = y_base[order]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"sol {sol} {target} {scan} point {point} asPLS baseline ({variant})")

    def plot_panel(ax, xvals, yraw, ybase, ycorr):
        ax.plot(xvals, yraw, color="#1f77b4", linewidth=1.2, alpha=0.85, label="Raw")
        ax.plot(xvals, ybase, color="#ff7f0e", linewidth=1.2, alpha=0.9, linestyle="--", label="Baseline")
        ax.plot(xvals, ycorr, color="#2ca02c", linewidth=1.2, alpha=0.9, label="Corrected")
        ax.axhline(0, alpha=0.5, color="black", lw=0.5)
        ax.grid(True, alpha=0.3)

    # Full range
    ax = axs[0, 0]
    plot_panel(ax, x_ord, y_raw_ord, y_base_ord, y_corr_ord)
    ax.set_title("full range")
    ax.set_xlim([500, 4000])

    # Sulf-carb region with spans
    ax = axs[0, 1]
    plot_panel(ax, x_ord, y_raw_ord, y_base_ord, y_corr_ord)
    ax.set_title("sulf-carb region")
    ax.set_xlim([900, 1300])
    ax.axvspan(998, 1038, alpha=0.3, color="yellow", label="sulf")
    ax.axvspan(1070, 1110, alpha=0.3, color="green", label="carb")

    # Organic region
    ax = axs[1, 0]
    plot_panel(ax, x_ord, y_raw_ord, y_base_ord, y_corr_ord)
    ax.set_title("organic region")
    ax.set_xlim([1200, 1800])
    ax.axvspan(1300, 1550, alpha=0.1, color="red")
    ax.axvspan(1550, 1650, alpha=0.3, color="red")

    # Hydration region
    ax = axs[1, 1]
    plot_panel(ax, x_ord, y_raw_ord, y_base_ord, y_corr_ord)
    ax.set_title("hydration region")
    ax.set_xlim([3000, 4000])

    for ax in axs.flat:
        ax.set(xlabel="Raman Shift (cm⁻¹)", ylabel="intensity")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.legend(labels=["raw", "aspls baseline", "processed"], loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=3)

    out_path = output_dir / f"{sol}_{target}_{scan}_{point}_R1_{variant}_baseline_test.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_average_bkgsub_baseline(
    x_axis: np.ndarray,
    avg_raw: np.ndarray,
    bg_interp: np.ndarray,
    avg_bkgsub: np.ndarray,
    avg_bkgsub_baselined: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """Plot standardized summary of background subtraction on averaged spectra.

    Shows: raw average (blue), background (orange), bkgsub (gray), bkgsub+baseline (green).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, avg_raw, color="#1f77b4", linewidth=1.2, label="raw avg")
    ax.plot(x_axis, bg_interp, color="#ff7f0e", linewidth=1.2, label="background")
    ax.plot(x_axis, avg_bkgsub, color="#7f7f7f", linewidth=1.2, label="bkgsub avg")
    ax.plot(x_axis, avg_bkgsub_baselined, color="#2ca02c", linewidth=1.2, label="bkgsub + baseline")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    try:
        x0 = float(x_axis[0])
    except Exception:
        x0 = 0.0
    ax.set_xlim([x0, 4000])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_average_baseline(
    x_axis: np.ndarray,
    avg_raw: np.ndarray,
    baseline: np.ndarray,
    avg_baselined: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    """Plot raw average, fitted baseline, and baseline-corrected average.

    Colors: raw (blue), baseline (orange dashed), baselined (green).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, avg_raw, color="#1f77b4", linewidth=1.2, label="raw avg")
    ax.plot(x_axis, baseline, color="#ff7f0e", linewidth=1.2, linestyle="--", label="baseline")
    ax.plot(x_axis, avg_baselined, color="#2ca02c", linewidth=1.2, label="baselined avg")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    try:
        x0 = float(x_axis[0])
    except Exception:
        x0 = 0.0
    ax.set_xlim([x0, 4000])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_average_single_series(
    x_axis: np.ndarray,
    y_series: np.ndarray,
    output_path: Path,
    title: str,
    color: str = "#2ca02c",
) -> Path:
    """Plot a single averaged spectrum (e.g., baselined average)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, y_series, color=color, linewidth=1.5)
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    try:
        x0 = float(x_axis[0])
    except Exception:
        x0 = 0.0
    ax.set_xlim([x0, 4000])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_average_bkgsub_comparison(
    x_axis: np.ndarray,
    avg_raw: np.ndarray,
    bg_unscaled_interp: np.ndarray,
    scale_used: float,
    output_path: Path,
    title: str,
) -> Path:
    # raw - (scale * bg) vs raw - bg, include background trace
    y_scaled = avg_raw - (scale_used * bg_unscaled_interp)
    y_unscaled = avg_raw - bg_unscaled_interp
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, avg_raw, color="#1f77b4", linewidth=1.2, label="raw avg")
    ax.plot(x_axis, bg_unscaled_interp, color="#000000", alpha=0.3, linewidth=1.2, label="background (unscaled)")
    ax.plot(x_axis, y_scaled, color="#2ca02c", linewidth=1.4, label=f"bkgsub scaled (s={scale_used:.3f})")
    ax.plot(x_axis, y_unscaled, color="#d62728", linewidth=1.2, linestyle="--", label="bkgsub unscaled (s=1)")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    try:
        x0 = float(x_axis[0])
    except Exception:
        x0 = 0.0
    ax.set_xlim([x0, 4000])
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
