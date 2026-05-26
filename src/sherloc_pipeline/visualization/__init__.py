"""
Visualization package for PHASE spectral analysis.

Enhanced visualization capabilities for SHERLOC spectral data including
spatial spectrograms, interactive plots, and publication-quality figures.

Modules:
    plotting: Spectral and spatial plotting utilities (PlotConfig, configure_matplotlib, apply_plot_config)
    fitting_plots: Raman and fluorescence fit overlay plots (moved from core/fitting.py and core/fluor_fitting.py)
    spatial: Spatial ACI image overlay plots (moved from core/spatial.py)
    ingestion_plots: Average spectrum plots (moved from core/data_ingestion.py)
    preprocessing_plots: Preprocessing verification and diagnostic plots
    normalization_plots: Laser normalization diagnostic plots
    spectrograms: Enhanced spectrogram visualization pipeline for detail scans
"""

from .plotting import (
    PlotConfig,
    configure_matplotlib,
    apply_plot_config,
)

from .spectrograms import (
    SpectrogramVisualizationConfig,
    SpectrogramVisualizationPipeline,
    SpatialSpectralData,
)

from .fitting_plots import (
    plot_fit_overlay,
    plot_fluor_fit_overlay,
    GROUP_COLORS,
)

from .spatial import (
    overlay_points_on_aci,
    render_pointloc_full,
    render_pointloc_zoomed,
    render_pointloc_with_colorbar,
    build_combined_grid,
)

from .cooccurrence import plot_co_occurrence_overlay

__all__ = [
    "PlotConfig",
    "configure_matplotlib",
    "apply_plot_config",
    "SpectrogramVisualizationConfig",
    "SpectrogramVisualizationPipeline",
    "SpatialSpectralData",
    "plot_fit_overlay",
    "plot_fluor_fit_overlay",
    "GROUP_COLORS",
    "overlay_points_on_aci",
    "render_pointloc_full",
    "render_pointloc_zoomed",
    "render_pointloc_with_colorbar",
    "build_combined_grid",
    "plot_co_occurrence_overlay",
]