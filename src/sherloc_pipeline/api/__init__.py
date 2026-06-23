"""
SHERLOC Pipeline Python API.

This package provides notebook-friendly functions for spectral analysis,
enabling Jupyter workflows without CLI interaction.

Example:
    >>> from sherloc_pipeline.api.spectral import process_scan_average, plot_spectrum
    >>> 
    >>> # Process averaged spectrum
    >>> df, fit_result = process_scan_average(
    ...     sol="0921", target="Amherst_Point", scan="detail_1",
    ...     background="fs", baseline=True, fit=True
    ... )
    >>> 
    >>> # Generate plot
    >>> fig = plot_spectrum(df, fit_result=fit_result, xlim=(700, 1200))
    >>> fig.show()
"""

from .spectral import (
    process_scan_average,
    process_subset_average,
    process_point,
    load_point_spectrum,
    load_reference_spectrum,
    plot_spectrum,
    plot_overlay,
)

__all__ = [
    "process_scan_average",
    "process_subset_average",
    "process_point",
    "load_point_spectrum",
    "load_reference_spectrum",
    "plot_spectrum",
    "plot_overlay",
]


