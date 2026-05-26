"""Co-occurrence ACI overlay visualization.

Plots context images (ACI) with scan points colour-coded by co-occurrence
status: both Raman and fluorescence confirmed, Raman only, fluorescence
only, or neither detected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sherloc_pipeline.visualization.plotting import apply_plot_config

# Colour palette for co-occurrence overlay
_COLOR_CONFIRMED = (0.2, 0.8, 0.2, 1.0)   # green — both Raman + fluor
_COLOR_RAMAN_ONLY = (0.2, 0.4, 0.9, 1.0)  # blue
_COLOR_FLUOR_ONLY = (1.0, 0.6, 0.1, 1.0)  # orange
_COLOR_NEITHER = (0.6, 0.6, 0.6, 0.5)     # gray, semi-transparent


def plot_co_occurrence_overlay(
    context_image: np.ndarray,
    point_positions: List[Tuple[float, float]],
    co_occurrence_data: Dict,
    pattern_name: str,
    output_path: Path,
    *,
    marker_size: float = 50.0,
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 150,
) -> Path:
    """Plot ACI image with co-occurrence points highlighted.

    Colour coding:
    - Green: both Raman and fluorescence confirmed
    - Blue: Raman only
    - Orange: Fluorescence only
    - Gray: neither detected

    Args:
        context_image: 2-D (grayscale) or 3-D (RGB/RGBA) ACI array.
        point_positions: List of (x, y) pixel coordinates for every scan
            point.  Indexing must match the point indices used in
            *co_occurrence_data*.
        co_occurrence_data: Single pattern dict as returned by
            :func:`~sherloc_pipeline.services.pipeline.aggregate_co_occurrences`.
            Must contain ``point_indices`` (confirmed), ``n_points_raman_only``
            and ``n_points_fluor_only`` are for metadata only — the caller
            should also provide ``raman_only_indices`` and
            ``fluor_only_indices`` lists if available.
        pattern_name: Human-readable pattern name for the title.
        output_path: Destination file path (.png or .pdf).
        marker_size: Scatter marker size in points^2.
        figsize: Figure size in inches.
        dpi: Output resolution.

    Returns:
        The *output_path* that was written.
    """
    apply_plot_config()

    confirmed_set = set(co_occurrence_data.get("point_indices", []))
    raman_only_set = set(co_occurrence_data.get("raman_only_indices", []))
    fluor_only_set = set(co_occurrence_data.get("fluor_only_indices", []))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display the ACI
    if context_image.ndim == 2:
        ax.imshow(context_image, cmap="gray", origin="upper")
    else:
        ax.imshow(context_image, origin="upper")

    # Classify each point and scatter
    for category, color, label, zorder in [
        ("neither", _COLOR_NEITHER, "Neither", 1),
        ("fluor_only", _COLOR_FLUOR_ONLY, "Fluorescence only", 2),
        ("raman_only", _COLOR_RAMAN_ONLY, "Raman only", 3),
        ("confirmed", _COLOR_CONFIRMED, "Both confirmed", 4),
    ]:
        xs, ys = [], []
        for idx, (px, py) in enumerate(point_positions):
            if category == "confirmed" and idx in confirmed_set:
                xs.append(px); ys.append(py)
            elif category == "raman_only" and idx in raman_only_set:
                xs.append(px); ys.append(py)
            elif category == "fluor_only" and idx in fluor_only_set:
                xs.append(px); ys.append(py)
            elif category == "neither" and idx not in confirmed_set and idx not in raman_only_set and idx not in fluor_only_set:
                xs.append(px); ys.append(py)
        if xs:
            ax.scatter(xs, ys, s=marker_size, c=[color], label=label,
                       edgecolors="black", linewidths=0.3, zorder=zorder)

    ax.set_title(f"Co-occurrence: {pattern_name}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_axis_off()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path
