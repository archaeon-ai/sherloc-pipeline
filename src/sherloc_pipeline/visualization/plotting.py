"""Spectral and spatial plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib


@dataclass(frozen=True)
class PlotConfig:
    """Runtime plotting configuration for SHERLOC overlay figures."""

    use_explicit_margins: bool
    margins: Dict[str, float]
    bbox_inches: Optional[str]
    savefig_dpi: int
    rc_params: Dict[str, object]


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no"}


@lru_cache(maxsize=1)
def configure_matplotlib() -> PlotConfig:
    """Apply Matplotlib settings and return the effective plot configuration.

    This function standardizes matplotlib rcParams for consistent visual
    appearance across different environments (fonts, colors, line widths, etc.).
    Note that this does NOT guarantee byte-identical output across systems -
    minor rendering differences can still occur due to font rendering engines,
    antialiasing, and other platform-specific factors.

    Environment variables:
    - SHERLOC_LEGACY_PLOTS=1: Disable rcParams standardization, use tight_layout only
    - SHERLOC_DETERMINISTIC_PLOTS=0: Same as SHERLOC_LEGACY_PLOTS=1 (deprecated)

    Returns:
        PlotConfig with rcParams, margin settings, and save options.
    """

    if _env_flag_enabled("SHERLOC_LEGACY_PLOTS"):
        deterministic = False
    else:
        override = os.getenv("SHERLOC_DETERMINISTIC_PLOTS")
        if override is None:
            deterministic = True
        else:
            deterministic = _env_flag_enabled("SHERLOC_DETERMINISTIC_PLOTS")
    if not deterministic:
        return PlotConfig(
            use_explicit_margins=False,
            margins={},
            bbox_inches="tight",
            savefig_dpi=300,
            rc_params={},
        )

    rc_params = {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.linewidth": 0.75,
        "axes.edgecolor": "#424242",
        "axes.labelcolor": "#212121",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 300,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#e0e0e0",
        "lines.antialiased": True,
    }
    matplotlib.rcParams.update(rc_params)

    # Pre-configured margins for fitting plots with right-side peak parameter legends.
    # The wide right margin (0.62) accommodates multi-line legend text showing
    # peak centers, FWHMs, and amplitudes. These are available via margins_override
    # parameter in apply_plot_config() - not used by default.
    fitting_plot_margins = {
        "left": 0.08,
        "right": 0.62,
        "bottom": 0.12,
        "top": 0.94,
    }

    return PlotConfig(
        use_explicit_margins=False,  # Use tight_layout by default
        margins=fitting_plot_margins,  # For fitting plots via margins_override
        bbox_inches="tight",
        savefig_dpi=300,
        rc_params=rc_params,
    )


def apply_plot_config(
    fig: matplotlib.figure.Figure,
    *,
    tight_layout_kwargs: Optional[Dict[str, Any]] = None,
    margins_override: Optional[Dict[str, float]] = None,
    use_default_margins: bool = True,
    bbox_override: Optional[str] = None,
) -> Tuple[PlotConfig, Optional[str]]:
    """Apply the active PlotConfig to a Matplotlib figure and return it with bbox_inches.

    This function handles figure layout, applying either tight_layout or explicit
    margins depending on the plot type.

    Args:
        fig: Matplotlib figure to configure
        tight_layout_kwargs: If provided, passed to fig.tight_layout()
        margins_override: Explicit margins (left/right/top/bottom) for plots
            that need fixed spacing, e.g., fitting plots with right-side legends
        use_default_margins: If False, always use tight_layout
        bbox_override: Override bbox_inches for savefig

    Returns:
        Tuple of (PlotConfig, bbox_inches) for use with fig.savefig()

    Example:
        >>> config, bbox = apply_plot_config(fig)
        >>> fig.savefig(path, dpi=config.savefig_dpi, bbox_inches=bbox)

        >>> # For fitting plots with legends, use margins_override:
        >>> config, bbox = apply_plot_config(
        ...     fig, margins_override={"left": 0.08, "right": 0.62, "top": 0.94, "bottom": 0.12}
        ... )
    """

    plot_config = configure_matplotlib()

    bbox_inches: Optional[str] = None

    # Priority 1: Explicit margins (for fitting plots with legends)
    if margins_override is not None:
        fig.subplots_adjust(**margins_override)
        bbox_inches = bbox_override
    # Priority 2: Custom tight_layout or explicit tight_layout request
    elif tight_layout_kwargs is not None or not use_default_margins:
        if tight_layout_kwargs:
            fig.tight_layout(**tight_layout_kwargs)
        else:
            fig.tight_layout()
        bbox_inches = bbox_override if bbox_override is not None else "tight"
    # Priority 3: Use config's fixed margins (if enabled)
    elif plot_config.use_explicit_margins:
        fig.subplots_adjust(**plot_config.margins)
        bbox_inches = bbox_override if bbox_override is not None else plot_config.bbox_inches
    # Default: tight_layout for automatic spacing
    else:
        fig.tight_layout()
        bbox_inches = bbox_override if bbox_override is not None else (plot_config.bbox_inches or "tight")

    if bbox_inches is None:
        bbox_inches = plot_config.bbox_inches

    return plot_config, bbox_inches
