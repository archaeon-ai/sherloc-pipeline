"""Fitting overlay visualization functions for Raman and fluorescence spectra.

These functions were moved from core/fitting.py and core/fluor_fitting.py
as part of the core/visualization layer separation (Public Release v3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sherloc_pipeline.models.fitting import FitResult
from sherloc_pipeline.core.fluor_fitting import FluorFitResult
from sherloc_pipeline.visualization.plotting import apply_plot_config, PlotConfig


# Group → color mapping per spec §7.2: group1a=amber, group1b=gold, group2=cyan, group3=violet
GROUP_COLORS = {
    "group1a": "#FF8F00",  # amber
    "group1b": "#FFD600",  # gold
    "group2": "#00BCD4",  # cyan
    "group3": "#7E57C2",  # violet
    "unidentified": "#9E9E9E",  # gray
}


def plot_fit_overlay(
    x_cm1: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    result: FitResult,
    y_model_full: np.ndarray,
    output_png_path: str,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = (700.0, 2000.0),
    sol: Optional[str] = None,
    target: Optional[str] = None,
    scan: Optional[str] = None,
    point: Optional[int] = None,
    roi: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot Raman spectrum with fitted Gaussian components overlaid.

    Moved from core/fitting.py. Callers should import from
    sherloc_pipeline.visualization.fitting_plots.
    """
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from sherloc_pipeline.core.fitting import gaussian

    # plotting ROI only for clarity
    x = x_cm1[mask]
    y_roi = y[mask]
    y_model = y_model_full[mask]
    fig, ax = plt.subplots(figsize=(12, 6))
    main_line, = ax.plot(x, y_roi, color='#1f77b4', linewidth=1.2)
    # Build a smooth plotting grid over the ROI to render smooth Gaussians
    x_smooth = np.linspace(x.min(), x.max(), max(2000, int(len(x) * 4)))
    # Recompute smooth model from fitted peaks for display
    y_model_smooth = np.zeros_like(x_smooth, dtype=float)
    for p in result.peaks:
        y_model_smooth += gaussian(x_smooth, p.m_cm1, p.a, p.fwhm)
    model_line, = ax.plot(x_smooth, y_model_smooth, color='#2ca02c', linewidth=1.2)
    handles = [Line2D([0],[0], color='#1f77b4', lw=1.2, label='baseline subtracted data'),
               Line2D([0],[0], color='#2ca02c', lw=1.2, label=f'model (R²={result.r2:.3f})')]
    text_colors = ['black', 'black']
    peaks_debug: List[Dict[str, object]] = []
    # components
    # Fixed colorblind-friendly palette for up to 5 peaks (avoid red, reserve for failing)
    cycle = ['#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
    ci = 0
    for idx, p in enumerate(result.peaks):
        y_comp = gaussian(x_smooth, p.m_cm1, p.a, p.fwhm)
        failing = (not p.pass_fwhm) or (not p.pass_snr) or (not p.pass_r2) or (not p.pass_sharpness)
        style = ':' if failing else '-'
        color = '#d62728' if failing else cycle[ci % len(cycle)]
        if not failing:
            ci += 1
        ax.plot(x_smooth, y_comp, linestyle=style, color=color, linewidth=1.0)
        label = f"m: {p.m_cm1:.1f}, a: {p.a:.1f}, FWHM: {p.fwhm:.1f}, SNR: {p.snr:.1f}"
        handles.append(Line2D([0],[0], color=color, lw=1.5, linestyle=style, label=label))
        text_colors.append('red' if failing else 'black')
        peaks_debug.append(
            {
                "peak_index": idx,
                "center_cm1": float(p.m_cm1),
                "amplitude": float(p.a),
                "fwhm_cm1": float(p.fwhm),
                "snr": float(p.snr),
                "pass_snr": bool(p.pass_snr),
                "pass_fwhm": bool(p.pass_fwhm),
                "pass_r2": bool(p.pass_r2),
                "sharpness_ratio": float(p.sharpness_ratio),
                "pass_sharpness": bool(p.pass_sharpness),
                "failing": failing,
                "line_style": style,
                "color": color,
            }
        )
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (counts)')
    if title is None and sol and target and scan and point is not None:
        r0, r1 = (xlim if roi is None else roi)
        title = f"sol {sol} {target} {scan} p{point} {int(r0)}-{int(r1)} cm⁻¹"
    elif title is None:
        title = f"R²={result.r2:.3f}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(list(xlim))
    # Place legend to the right, top-aligned with plot area
    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.85)
    # Colorize legend text for failing peaks
    for txt, col in zip(leg.get_texts(), text_colors):
        if col != 'black':
            txt.set_color(col)

    # Fitting plots need wide right margins for the legend with fit parameters
    fitting_margins = {
        "left": 0.08,
        "right": 0.62,
        "bottom": 0.12,
        "top": 0.94,
    }
    plot_config, bbox_inches = apply_plot_config(fig, margins_override=fitting_margins)

    metadata = {
        "title": title,
        "sol": sol,
        "target": target,
        "scan": scan,
        "point": point,
        "roi": list(roi) if roi is not None else None,
        "xlim": list(xlim) if xlim is not None else None,
        "x_range": [float(x.min()), float(x.max())],
        "deterministic_plots": plot_config.use_explicit_margins,
    }
    _maybe_write_overlay_debug(
        output_png_path,
        result,
        peaks_debug,
        fig,
        ax,
        leg,
        metadata=metadata,
        data_line_color='#1f77b4',
        model_line_color='#2ca02c',
        plot_config=plot_config,
    )
    # Use pad_inches=0.1 to add minimal whitespace around edges (prevents legends/text from touching border)
    fig.savefig(output_png_path, dpi=plot_config.savefig_dpi, bbox_inches=bbox_inches, pad_inches=0.1)
    plt.close(fig)


def _maybe_write_overlay_debug(
    output_png_path: str,
    result: FitResult,
    peaks_debug: List[Dict[str, object]],
    fig,
    ax,
    legend,
    *,
    metadata: Dict[str, object],
    data_line_color: str,
    model_line_color: str,
    plot_config: PlotConfig | None = None,
) -> None:
    if not os.getenv("SHERLOC_OVERLAY_DEBUG"):
        return
    try:
        from matplotlib import pyplot as plt
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bounds = ax.get_position().bounds
        legend_info = None
        if legend is not None:
            try:
                bbox = legend.get_window_extent(renderer=renderer)
                legend_info = {
                    "loc": getattr(legend, "_loc", None),
                    "bbox_inches": {
                        "x0": bbox.x0 / fig.dpi,
                        "y0": bbox.y0 / fig.dpi,
                        "width": bbox.width / fig.dpi,
                        "height": bbox.height / fig.dpi,
                    },
                }
            except Exception:
                legend_info = {"loc": getattr(legend, "_loc", None)}
        debug_data = {
            "metadata": metadata,
            "figure": {
                "size_inches": list(fig.get_size_inches()),
                "dpi": float(fig.get_dpi()),
            },
            "axes": {
                "position": {
                    "x0": axes_bounds[0],
                    "y0": axes_bounds[1],
                    "width": axes_bounds[2],
                    "height": axes_bounds[3],
                },
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
            },
            "legend": legend_info,
            "result": {
                "r2": float(result.r2),
                "rss": float(result.rss),
                "dof": int(result.dof),
                "warnings": list(result.warnings or []),
            },
            "lines": {
                "data_color": data_line_color,
                "model_color": model_line_color,
            },
            "peaks": peaks_debug,
        }
        if plot_config is not None:
            debug_data["plot_config"] = {
                "use_explicit_margins": plot_config.use_explicit_margins,
                "margins": plot_config.margins,
                "bbox_inches": plot_config.bbox_inches,
                "savefig_dpi": plot_config.savefig_dpi,
            }
        debug_path = Path(output_png_path).with_suffix(".json")
        debug_path.write_text(json.dumps(debug_data, indent=2))
    except Exception:
        # Debug capture should never break production runs
        pass


def plot_fluor_fit_overlay(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    result: FluorFitResult,
    output_png_path: str,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    saturation_threshold: float = 60000.0,
    sol: Optional[str] = None,
    target: Optional[str] = None,
    scan: Optional[str] = None,
    point: Optional[int] = None,
) -> None:
    """Plot fluorescence spectrum with fitted Gaussian components overlaid.

    Shows the dark-subtracted spectrum (blue), individual Gaussian components
    colored by fluorescence group, and the composite fit (green). Saturated
    channels are marked with red triangles. No baseline subtraction is applied.

    Moved from core/fluor_fitting.py. Callers should import from
    sherloc_pipeline.visualization.fitting_plots.

    Args:
        wavelength: Wavelength array in nm.
        intensity: Dark-subtracted intensity array (counts).
        result: FluorFitResult from fit_fluorescence_spectrum().
        output_png_path: Path for output PNG file (PDF also saved alongside).
        title: Optional figure title.
        xlim: Optional (min, max) wavelength limits for x-axis.
        saturation_threshold: CCD saturation level for marking channels.
        sol: Sol number for auto-title.
        target: Target name for auto-title.
        scan: Scan name for auto-title.
        point: Point index for auto-title.
    """
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    from sherloc_pipeline.core.fluor_id import assign_fluor_group
    from sherloc_pipeline.core.fitting import fwhm_to_sigma

    fig, ax = plt.subplots(figsize=(12, 6))

    # Despike for display: rolling-median sigma-clip
    plot_intensity = intensity.copy()
    _dw = min(11, max(3, len(plot_intensity) // 10) | 1)
    for _ in range(3):
        _rm = pd.Series(plot_intensity).rolling(window=_dw, center=True, min_periods=1).median().values
        _res = plot_intensity - _rm
        _mad = np.median(np.abs(_res))
        _rsig = 1.4826 * _mad if _mad > 0 else np.std(_res)
        if _rsig == 0 or not np.isfinite(_rsig):
            break
        _sm = np.abs(_res) > 5.0 * _rsig
        if not np.any(_sm):
            break
        plot_intensity[_sm] = _rm[_sm]

    # Plot despiked spectrum
    ax.plot(wavelength, plot_intensity, color="#1f77b4", linewidth=1.2, label="spectrum")

    # Mark saturated channels
    sat_mask = plot_intensity >= saturation_threshold
    if np.any(sat_mask):
        ax.scatter(
            wavelength[sat_mask],
            intensity[sat_mask],
            marker="v",
            color="#d62728",
            s=30,
            zorder=5,
            label=f"saturated ({int(np.sum(sat_mask))} ch)",
        )

    # Build legend handles
    handles = [
        Line2D([0], [0], color="#1f77b4", lw=1.2, label="spectrum"),
    ]
    if np.any(sat_mask):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="v",
                color="#d62728",
                lw=0,
                markersize=6,
                label=f"saturated ({int(np.sum(sat_mask))} ch)",
            )
        )

    # Plot fitted components and composite
    if result.peaks and not result.fit_skipped:
        # Smooth x-grid for Gaussian rendering
        wl_min = wavelength.min() if xlim is None else xlim[0]
        wl_max = wavelength.max() if xlim is None else xlim[1]
        x_smooth = np.linspace(wl_min, wl_max, max(2000, len(wavelength) * 4))

        # Composite model
        composite = np.zeros_like(x_smooth)
        for peak in result.peaks:
            sigma = fwhm_to_sigma(peak.fwhm_nm)
            component = peak.amplitude * np.exp(
                -0.5 * ((x_smooth - peak.center_nm) / sigma) ** 2
            )
            composite += component

        # Plot composite fit
        ax.plot(
            x_smooth,
            composite,
            color="#2ca02c",
            linewidth=1.2,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color="#2ca02c",
                lw=1.2,
                label=f"composite fit (R\u00b2={result.r2:.3f})",
            )
        )

        # Plot individual components colored by group
        for peak in result.peaks:
            group = assign_fluor_group(peak.center_nm)
            color = GROUP_COLORS.get(group, GROUP_COLORS["unidentified"])
            sigma = fwhm_to_sigma(peak.fwhm_nm)
            y_comp = peak.amplitude * np.exp(
                -0.5 * ((x_smooth - peak.center_nm) / sigma) ** 2
            )
            ax.plot(x_smooth, y_comp, color=color, linewidth=1.0, linestyle="--")
            label = (
                f"{group}: {peak.center_nm:.1f} nm, "
                f"A={peak.amplitude:.0f}, "
                f"FWHM={peak.fwhm_nm:.1f} nm, "
                f"SNR={peak.snr:.1f}"
            )
            handles.append(
                Line2D([0], [0], color=color, lw=1.5, linestyle="--", label=label)
            )
    elif result.fit_skipped:
        handles.append(
            Line2D(
                [0],
                [0],
                color="none",
                label=f"fit skipped ({', '.join(result.warnings)})",
            )
        )

    # Axis labels (fluorescence is wavelength domain — nm, not cm-1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (counts)")

    # Auto-title from scan metadata
    if title is None and sol and target and scan and point is not None:
        title = f"sol {sol} {target} {scan} p{point} fluorescence"
    elif title is None:
        title = f"Fluorescence fit (R\u00b2={result.r2:.3f})"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    if xlim is not None:
        ax.set_xlim(list(xlim))

    # Legend to the right (same pattern as Raman overlay)
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.85,
    )

    fitting_margins = {
        "left": 0.08,
        "right": 0.62,
        "bottom": 0.12,
        "top": 0.94,
    }
    plot_config, bbox_inches = apply_plot_config(fig, margins_override=fitting_margins)

    # Save PNG
    fig.savefig(
        output_png_path,
        dpi=plot_config.savefig_dpi,
        bbox_inches=bbox_inches,
        pad_inches=0.1,
    )

    plt.close(fig)
