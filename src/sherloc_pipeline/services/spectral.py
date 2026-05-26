"""
Spectral analysis service for SHERLOC pipeline.

This module provides the SpectralService for flexible spectral visualization
and analysis, supporting both averaged spectrum workflows (from Loupe data)
and point visualization (from existing pipeline outputs).

The service complements the existing full-pipeline and apply-review commands
by providing quick, one-off spectral analysis with configurable background
subtraction, baseline correction, and Gaussian fitting.

Usage:
    from sherloc_pipeline.services.spectral import SpectralService, SpectralPlotRequest
    
    service = SpectralService()
    request = SpectralPlotRequest(
        sol="0921",
        target="Amherst_Point",
        scan="detail_1",
        mode="averaged",
        background="fs",
        baseline=True,
        fit=True,
    )
    result = service.process(request)
    print(result.summary)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any
import json
import logging
import textwrap

import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from .base import ServiceResult
from .errors import SherlocServiceError, enrich
from .runtime import RuntimeContext
from sherloc_pipeline.visualization.plotting import configure_matplotlib, apply_plot_config

from sherloc_pipeline.core.baseline import (
    BaselineParams,
    _baseline_aspls_with_weights,
    build_weight_vector_from_windows,
)
from sherloc_pipeline.core.preprocessing import baseline_aspls
from sherloc_pipeline.core.fitting import fit_spectrum, gaussian
from sherloc_pipeline.core.utils import resolve_trim_proportion, format_trim_label
from sherloc_pipeline.models.fitting import FitResult, PeakFit

# Module-level logger
logger = logging.getLogger(__name__)


class SpectralPlotError(SherlocServiceError):
    """Error during spectral plotting operations.
    
    This error indicates a failure during spectral loading, processing,
    background subtraction, or plot generation.
    
    Example:
        >>> raise SpectralPlotError("Failed to load Loupe data", exit_code=1)
    """
    pass


@dataclass
class LoupeData:
    """Container for loaded Loupe spectral data.
    
    Attributes:
        spectra_df: DataFrame with raman_shift and point columns (0, 1, 2, ...)
        n_points: Number of spectral points in the scan
        ppp: Pulses per point (shots_per_spec from loupe.csv)
        working_dir: Path to the Loupe working directory
        metadata: Additional metadata from loupe.csv
    """
    spectra_df: pd.DataFrame
    n_points: int
    ppp: float
    working_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectralPlotRequest:
    """Request for spectral processing and plotting.
    
    This dataclass encapsulates all parameters needed for a spectral
    analysis operation, supporting both averaged spectrum workflows
    (from raw Loupe data) and point visualization (from pipeline outputs).
    
    Attributes:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan identifier (e.g., "detail_1", "line_1")
        mode: Processing mode - "averaged" for all points, "point" for single point,
              "subset" for averaging specific points
        point: Point index for point mode (required if mode="point")
        level: Processing level for point mode (required if mode="point")
        points: List of point indices for subset mode (required if mode="subset")
        avg_method: Averaging method for averaged/subset modes
        trim_pct: Trim percentage for trim-mean (ignored for other methods)
        background: Background type - "as" (arm stowed) or "fs" (fused silica)
        bgscale: Background scale factor - "auto" for PPP-based, or explicit float
        baseline: Whether to apply baseline correction
        fit: Whether to apply Gaussian fitting
        fit_range: Fit range as (min, max) in cm^-1
        single_peak_center: Optional center position for single-peak fitting mode
        n_peaks: Optional maximum number of peaks for n-peaks limit mode
        min_snr: Minimum SNR threshold for peak acceptance (overrides config)
        fwhm_min: Minimum FWHM in cm^-1 for peak acceptance (overrides config)
        fwhm_max: Maximum FWHM in cm^-1 for peak acceptance (overrides config)
        xlim: X-axis limits as (min, max) in cm^-1
        ylim: Y-axis limits as (min, max) in intensity units
        export: Export format - "csv", "png", or "both"
        
    Example:
        >>> request = SpectralPlotRequest(
        ...     sol="0921",
        ...     target="Amherst_Point",
        ...     scan="detail_1",
        ...     mode="averaged",
        ...     background="fs",
        ...     baseline=True,
        ...     fit=True,
        ... )
    """
    
    # Scan identification (always required)
    sol: str
    target: str
    scan: str
    
    # Mode selection (mutually exclusive workflows)
    mode: Literal["averaged", "point", "subset"] = "averaged"
    point: Optional[int] = None  # Required if mode="point"
    level: Optional[str] = None  # Required if mode="point"
    points: Optional[list[int]] = None  # Required if mode="subset"
    
    # Averaging options (mode="averaged" only)
    avg_method: Literal["mean", "median", "trim-mean"] = "trim-mean"
    trim_pct: float = 2.0  # Percentage for trim-mean (ignored for other methods)
    
    # Processing options (mode="averaged" only)
    background: Optional[Literal["as", "fs"]] = None
    bgscale: float | Literal["auto"] = "auto"
    baseline: bool = True
    fit: bool = False
    fit_range: Optional[Tuple[float, float]] = None
    single_peak_center: Optional[float] = None  # For single-peak fitting mode
    n_peaks: Optional[int] = None  # For n-peaks limit mode
    min_snr: Optional[float] = None  # Override config min_snr threshold
    fwhm_min: Optional[float] = None  # Override config filter_fwhm_min_cm1
    fwhm_max: Optional[float] = None  # Override config fwhm_max_cm1
    
    # Axis controls (both modes)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    
    # Domain selection (for plotting fitted peaks from DB)
    domain: Literal["raman", "fluor", "both"] = "raman"

    # Export options
    export: Literal["csv", "png", "both"] = "both"
    no_metadata: bool = False  # If True, skip JSON metadata export

    # Runtime (set during processing, not by user)
    n_points_averaged: Optional[int] = None  # Populated after data loading for effective trim label
    
    def __post_init__(self) -> None:
        """Validate request parameters."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate mode-specific requirements
        if self.mode == "point":
            if self.point is None:
                raise ValueError("Point mode requires --point to be specified")
            
            # level is optional - determines data source:
            # - With level: load from pre-processed pipeline outputs (legacy)
            # - Without level: process from raw Loupe data (new behavior)
            
            # If level is set with processing flags, warn that they'll be ignored
            if self.level is not None:
                if self.background or self.baseline or self.fit:
                    logger.warning(
                        "Processing flags (--background, --baseline, --fit) are ignored "
                        "when --level is specified. Use point mode without --level to "
                        "enable processing from Loupe data."
                    )
        
        if self.mode == "subset":
            if self.points is None or len(self.points) == 0:
                raise ValueError("Subset mode requires --points to be specified")
            if len(self.points) < 2:
                raise ValueError("Subset mode requires at least 2 points to average")
        
        # Validate trim_pct range
        if self.trim_pct < 0 or self.trim_pct > 50:
            raise ValueError(f"trim_pct must be between 0 and 50, got {self.trim_pct}")
        
        # Validate level if provided
        valid_levels = {
            "normalized",
            "normalized_baselined", 
            "normalized_despiked_baselined",
        }
        if self.level is not None and self.level not in valid_levels:
            raise ValueError(
                f"Invalid level '{self.level}'. "
                f"Valid options: {', '.join(sorted(valid_levels))}"
            )
        
        # Validate single_peak_center and n_peaks mutual exclusivity
        if self.single_peak_center is not None and self.n_peaks is not None:
            raise ValueError("--single-peak and --n-peaks are mutually exclusive")
        
        # Validate that single_peak_center requires --fit
        if self.single_peak_center is not None and not self.fit:
            raise ValueError("--single-peak requires --fit to be enabled")
        
        # Validate n_peaks requires --fit
        if self.n_peaks is not None and not self.fit:
            raise ValueError("--n-peaks requires --fit to be enabled")
        
        # Validate fitting threshold overrides require --fit
        if self.min_snr is not None and not self.fit:
            raise ValueError("--min-snr requires --fit to be enabled")
        if self.fwhm_min is not None and not self.fit:
            raise ValueError("--fwhm-min requires --fit to be enabled")
        if self.fwhm_max is not None and not self.fit:
            raise ValueError("--fwhm-max requires --fit to be enabled")
        
        # Validate threshold values are positive
        if self.min_snr is not None and self.min_snr <= 0:
            raise ValueError(f"--min-snr must be positive, got {self.min_snr}")
        if self.fwhm_min is not None and self.fwhm_min <= 0:
            raise ValueError(f"--fwhm-min must be positive, got {self.fwhm_min}")
        if self.fwhm_max is not None and self.fwhm_max <= 0:
            raise ValueError(f"--fwhm-max must be positive, got {self.fwhm_max}")
        
        # Validate fwhm_min < fwhm_max if both provided
        if self.fwhm_min is not None and self.fwhm_max is not None:
            if self.fwhm_min >= self.fwhm_max:
                raise ValueError(
                    f"--fwhm-min ({self.fwhm_min}) must be less than --fwhm-max ({self.fwhm_max})"
                )
        
        # Validate n_peaks is positive
        if self.n_peaks is not None and self.n_peaks < 1:
            raise ValueError("--n-peaks must be at least 1")


class SpectralService:
    """Service for spectral analysis and plotting.
    
    This service provides two main workflows:
    
    1. **Averaged mode**: Process raw Loupe data into an averaged spectrum
       with optional background subtraction, baseline correction, and fitting.
       
    2. **Point mode**: Load and visualize a single point from existing
       pipeline outputs with custom axis controls.
    
    The service uses the existing pipeline infrastructure (RuntimeContext,
    DataIngestion, etc.) and follows the ServiceResult pattern for outputs.
    
    Attributes:
        console: Rich Console instance for progress/output
        context: RuntimeContext for configuration and path resolution
        
    Example:
        >>> service = SpectralService()
        >>> request = SpectralPlotRequest(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     mode="averaged", background="fs", baseline=True
        ... )
        >>> result = service.process(request)
        >>> print(result.summary)
        'Processed averaged spectrum for 0921/Amherst_Point/detail_1'
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        *,
        context: Optional[RuntimeContext] = None,
    ):
        """Initialize spectral service.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new Console.
            context: Optional RuntimeContext providing resolved configuration and roots.
                If None, a new context is bootstrapped.
        """
        self.console = console if console is not None else Console()
        self.context = context if context is not None else RuntimeContext.bootstrap()
    
    def process(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process request and generate outputs.

        This is the main entry point for spectral processing. It dispatches
        to the appropriate workflow based on the request mode and domain.

        Args:
            request: SpectralPlotRequest with processing parameters

        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata

        Raises:
            SpectralPlotError: If processing fails

        Example:
            >>> result = service.process(request)
            >>> print(result.summary)
            >>> for artifact in result.artifacts:
            ...     print(f"  Created: {artifact}")
        """
        # Route based on domain
        if request.domain == "fluor":
            return self._process_fluor(request)
        elif request.domain == "both":
            return self._process_both(request)

        # Default: Raman domain (existing behavior)
        if request.mode == "averaged":
            return self._process_averaged(request)
        elif request.mode == "subset":
            return self._process_subset(request)
        else:
            return self._process_point(request)
    
    def _process_averaged(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process averaged spectrum from Loupe data.
        
        Mini-pipeline workflow:
        1. Load Loupe data (darkSubSpectra with laser normalization)
        2. Compute average (mean, median, or trim-mean)
        3. Apply background subtraction with PPP scaling (if requested)
        4. Apply baseline correction (if requested)
        5. Apply Gaussian fitting (if requested)
        6. Generate plot
        7. Export outputs (CSV, PNG)
        
        Args:
            request: SpectralPlotRequest with mode="averaged"
            
        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata
            
        Raises:
            SpectralPlotError: If processing fails
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for saving
        
        warnings: list[str] = []
        metadata: dict[str, Any] = {}
        
        # Step 1: Load Loupe data
        loupe_data = self._load_loupe_data(request.sol, request.target, request.scan)
        request.n_points_averaged = loupe_data.n_points
        metadata["n_points"] = loupe_data.n_points
        metadata["ppp"] = loupe_data.ppp
        metadata["working_dir"] = str(loupe_data.working_dir)

        # Step 2: Compute average
        spectrum_df = self._compute_average(
            loupe_data,
            method=request.avg_method,
            trim_pct=request.trim_pct,
        )
        metadata["avg_method"] = request.avg_method
        if request.avg_method == "trim-mean":
            metadata["trim_pct"] = request.trim_pct
        
        # Step 3: Apply background subtraction (if requested)
        bg_scale_used: Optional[float] = None
        if request.background is not None:
            # Calculate scale factor
            if request.bgscale == "auto":
                bg_scale_used = calculate_background_scale(
                    scan_ppp=loupe_data.ppp,
                    bg_ppp=900.0,
                )
            else:
                bg_scale_used = float(request.bgscale)
            
            spectrum_df = self._apply_background_subtraction(
                spectrum_df,
                bg_type=request.background,
                scale=bg_scale_used,
            )
            metadata["background"] = request.background
            metadata["bgscale"] = bg_scale_used
        
        # Step 4: Apply baseline correction (if requested)
        if request.baseline:
            spectrum_df = self._apply_baseline(spectrum_df)
            metadata["baseline"] = True
        
        # Step 5: Apply Gaussian fitting (if requested)
        fit_result = None
        model_array = None
        if request.fit:
            fit_result, model_array = self._apply_fitting(
                spectrum_df,
                fit_range=request.fit_range,
                single_peak_center=request.single_peak_center,
                n_peaks=request.n_peaks,
                min_snr=request.min_snr,
                fwhm_min=request.fwhm_min,
                fwhm_max=request.fwhm_max,
            )
            metadata["fit"] = True
            metadata["n_peaks"] = len(fit_result.peaks)
            metadata["r2"] = fit_result.r2
            
            # Add fitting warnings
            if fit_result.warnings:
                warnings.extend(fit_result.warnings)
        
        # Step 6: Generate plot
        fig = self._generate_plot(
            spectrum_df,
            request,
            fit_result=fit_result,
            model_array=model_array,
        )
        
        # Step 7: Export outputs
        artifacts = self._export(
            spectrum_df,
            fig,
            request,
            fit_result=fit_result,
            loupe_data=loupe_data,
            bg_scale_used=bg_scale_used,
        )
        
        # Close figure to free memory
        plt.close(fig)
        
        # Build summary
        processing_parts = []
        processing_parts.append(self._format_avg_method(request))
        if request.background is not None:
            processing_parts.append(f"{request.background} bg-subtracted")
        if request.baseline:
            processing_parts.append("baselined")
        if request.fit and fit_result is not None:
            processing_parts.append(f"fit ({len(fit_result.peaks)} peaks)")
        
        summary = (
            f"Processed averaged spectrum for {request.sol}/{request.target}/{request.scan} "
            f"({loupe_data.n_points} points, {loupe_data.ppp} PPP): "
            f"{', '.join(processing_parts)}"
        )
        
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _process_subset(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process subset-averaged spectrum from specified points.
        
        Similar to _process_averaged() but only averages the specified subset
        of point indices. This enables label-like averaging with ad-hoc point
        selection.
        
        Workflow:
        1. Load Loupe data
        2. Validate and filter to specified point indices
        3. Compute average across subset
        4. Apply background subtraction with PPP scaling (if requested)
        5. Apply baseline correction (if requested)
        6. Apply Gaussian fitting (if requested)
        7. Generate plot
        8. Export outputs
        
        Args:
            request: SpectralPlotRequest with mode="subset" and points list
            
        Returns:
            ServiceResult with summary, artifacts, warnings, and metadata
            
        Raises:
            SpectralPlotError: If processing fails or point indices invalid
        """
        import matplotlib
        matplotlib.use('Agg')
        
        warnings: list[str] = []
        metadata: dict[str, Any] = {}
        
        # Step 1: Load all Loupe data
        loupe_data = self._load_loupe_data(request.sol, request.target, request.scan)
        metadata["ppp"] = loupe_data.ppp
        metadata["working_dir"] = str(loupe_data.working_dir)
        metadata["total_points"] = loupe_data.n_points
        
        # Step 2: Validate and filter to specified points
        assert request.points is not None  # Validated in __post_init__
        requested_points = list(request.points)
        
        # Validate all point indices are in range
        invalid_points = [p for p in requested_points if p < 0 or p >= loupe_data.n_points]
        if invalid_points:
            raise SpectralPlotError(
                f"Invalid point indices: {invalid_points}. "
                f"Valid range is 0 to {loupe_data.n_points - 1}.",
                exit_code=1,
                context={
                    "invalid_points": invalid_points,
                    "n_points": loupe_data.n_points,
                    "requested_points": requested_points,
                }
            )
        
        # Filter spectra_df to only include requested points
        # Point columns are integers: 0, 1, 2, etc.
        df = loupe_data.spectra_df
        available_cols = [c for c in df.columns if isinstance(c, int)]
        
        # Keep only raman_shift and requested point columns
        cols_to_keep = ["raman_shift"] + [c for c in available_cols if c in requested_points]
        filtered_df = df[cols_to_keep].copy()
        
        # Create filtered LoupeData with subset
        subset_data = LoupeData(
            spectra_df=filtered_df,
            n_points=len(requested_points),
            ppp=loupe_data.ppp,
            working_dir=loupe_data.working_dir,
            metadata={**loupe_data.metadata, "subset_points": requested_points},
        )
        request.n_points_averaged = len(requested_points)
        metadata["n_points"] = len(requested_points)
        metadata["subset_points"] = requested_points

        # Step 3: Compute average over subset
        spectrum_df = self._compute_average(
            subset_data,
            method=request.avg_method,
            trim_pct=request.trim_pct,
        )
        metadata["avg_method"] = request.avg_method
        if request.avg_method == "trim-mean":
            metadata["trim_pct"] = request.trim_pct
        
        # Step 4: Apply background subtraction (if requested)
        bg_scale_used: Optional[float] = None
        if request.background is not None:
            if request.bgscale == "auto":
                bg_scale_used = calculate_background_scale(
                    scan_ppp=loupe_data.ppp,
                    bg_ppp=900.0,
                )
            else:
                bg_scale_used = float(request.bgscale)
            
            spectrum_df = self._apply_background_subtraction(
                spectrum_df,
                bg_type=request.background,
                scale=bg_scale_used,
            )
            metadata["background"] = request.background
            metadata["bgscale"] = bg_scale_used
        
        # Step 5: Apply baseline correction (if requested)
        if request.baseline:
            spectrum_df = self._apply_baseline(spectrum_df)
            metadata["baseline"] = True
        
        # Step 6: Apply Gaussian fitting (if requested)
        fit_result = None
        model_array = None
        if request.fit:
            fit_result, model_array = self._apply_fitting(
                spectrum_df,
                fit_range=request.fit_range,
                single_peak_center=request.single_peak_center,
                n_peaks=request.n_peaks,
                min_snr=request.min_snr,
                fwhm_min=request.fwhm_min,
                fwhm_max=request.fwhm_max,
            )
            metadata["fit"] = True
            metadata["n_peaks"] = len(fit_result.peaks)
            metadata["r2"] = fit_result.r2
            
            if fit_result.warnings:
                warnings.extend(fit_result.warnings)
        
        # Step 7: Generate plot
        fig = self._generate_plot(
            spectrum_df,
            request,
            fit_result=fit_result,
            model_array=model_array,
        )
        
        # Step 8: Export outputs
        artifacts = self._export(
            spectrum_df,
            fig,
            request,
            fit_result=fit_result,
            loupe_data=subset_data,  # Use subset_data to capture subset-specific info
            bg_scale_used=bg_scale_used,
        )
        
        plt.close(fig)
        
        # Build summary
        processing_parts = []
        processing_parts.append(f"subset ({len(requested_points)} pts)")
        processing_parts.append(self._format_avg_method(request))
        if request.background is not None:
            processing_parts.append(f"{request.background} bg-subtracted")
        if request.baseline:
            processing_parts.append("baselined")
        if request.fit and fit_result is not None:
            processing_parts.append(f"fit ({len(fit_result.peaks)} peaks)")
        
        summary = (
            f"Processed subset spectrum for {request.sol}/{request.target}/{request.scan} "
            f"({len(requested_points)} of {loupe_data.n_points} points, {loupe_data.ppp} PPP): "
            f"{', '.join(processing_parts)}"
        )
        
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _process_point_from_loupe(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process single point from raw Loupe data with full processing chain.
        
        This method provides point-level processing directly from Loupe data,
        enabling background subtraction, baseline correction, and fitting for
        individual points without first running the full pipeline.
        
        Workflow:
        1. Load Loupe data (darkSubSpectraN.csv)
        2. Validate point index is in range
        3. Extract single point as spectrum DataFrame
        4. Apply background subtraction (if --background)
        5. Apply baseline correction (if --baseline)
        6. Apply Gaussian fitting (if --fit)
        7. Generate plot
        8. Export outputs (CSV, PNG)
        
        Args:
            request: SpectralPlotRequest with mode="point" and level=None
            
        Returns:
            ServiceResult with processed spectrum and artifacts
            
        Raises:
            SpectralPlotError: If loading or processing fails
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for saving
        
        warnings: list[str] = []
        metadata: dict[str, Any] = {}
        
        # Step 1: Load Loupe data
        loupe_data = self._load_loupe_data(
            request.sol, request.target, request.scan
        )
        metadata["ppp"] = loupe_data.ppp
        metadata["total_points"] = loupe_data.n_points
        
        # Step 2: Validate point index
        if request.point is None:
            raise SpectralPlotError(
                "Point mode requires --point to be specified",
                exit_code=1
            )
        if request.point < 0 or request.point >= loupe_data.n_points:
            raise SpectralPlotError(
                f"Point {request.point} out of range (0-{loupe_data.n_points - 1})",
                exit_code=1
            )
        metadata["point"] = request.point
        
        # Step 3: Extract single point column as spectrum DataFrame
        # Point columns are integers (0, 1, 2, ...) not strings
        point_col = request.point
        spectrum_df = pd.DataFrame({
            'raman_shift': loupe_data.spectra_df['raman_shift'],
            'intensity': loupe_data.spectra_df[point_col].astype(float),
        })
        metadata["n_datapoints"] = len(spectrum_df)
        
        # Step 4: Apply background subtraction (if requested)
        bg_scale_used: Optional[float] = None
        if request.background is not None:
            if request.bgscale == "auto":
                bg_scale_used = calculate_background_scale(
                    scan_ppp=loupe_data.ppp,
                    bg_ppp=900.0,
                )
            else:
                bg_scale_used = float(request.bgscale)
            
            spectrum_df = self._apply_background_subtraction(
                spectrum_df,
                bg_type=request.background,
                scale=bg_scale_used,
            )
            metadata["background"] = request.background
            metadata["bgscale"] = bg_scale_used
        
        # Step 5: Apply baseline correction (if requested)
        if request.baseline:
            spectrum_df = self._apply_baseline(spectrum_df)
            metadata["baseline"] = True
        
        # Step 6: Apply Gaussian fitting (if requested)
        fit_result = None
        model_array = None
        if request.fit:
            fit_result, model_array = self._apply_fitting(
                spectrum_df,
                fit_range=request.fit_range,
                single_peak_center=request.single_peak_center,
                n_peaks=request.n_peaks,
                min_snr=request.min_snr,
                fwhm_min=request.fwhm_min,
                fwhm_max=request.fwhm_max,
            )
            metadata["fit"] = True
            metadata["n_peaks"] = len(fit_result.peaks)
            metadata["r2"] = fit_result.r2
            
            if fit_result.warnings:
                warnings.extend(fit_result.warnings)
        
        # Step 7: Generate plot
        fig = self._generate_plot(
            spectrum_df,
            request,
            fit_result=fit_result,
            model_array=model_array,
        )
        
        # Step 8: Export outputs
        artifacts = self._export(
            spectrum_df,
            fig,
            request,
            fit_result=fit_result,
            loupe_data=loupe_data,
            bg_scale_used=bg_scale_used,
        )
        
        plt.close(fig)
        
        # Build summary
        processing_parts = [f"point {request.point}"]
        if request.background is not None:
            processing_parts.append(f"{request.background} bg-subtracted")
        if request.baseline:
            processing_parts.append("baselined")
        if request.fit and fit_result is not None:
            processing_parts.append(f"fit ({len(fit_result.peaks)} peaks)")
        
        summary = (
            f"Processed point spectrum for {request.sol}/{request.target}/{request.scan} "
            f"(point {request.point}, {loupe_data.ppp} PPP): "
            f"{', '.join(processing_parts)}"
        )
        
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _process_point(self, request: SpectralPlotRequest) -> ServiceResult:
        """Dispatch to appropriate point processing method based on --level presence.
        
        This method acts as a dispatcher for point mode, routing to:
        - _process_point_from_loupe(): When --level is NOT provided (new behavior)
        - _process_point_from_results(): When --level IS provided (legacy behavior)
        
        Args:
            request: SpectralPlotRequest with mode="point"
            
        Returns:
            ServiceResult from the appropriate processing method
            
        Raises:
            SpectralPlotError: If point is not specified or processing fails
        """
        # Validate required parameters
        if request.point is None:
            raise SpectralPlotError(
                "Point mode requires --point to be specified",
                exit_code=1
            )
        
        # Dispatch based on whether level is provided
        if request.level is not None:
            # Legacy: load from pre-processed pipeline outputs
            return self._process_point_from_results(request)
        else:
            # New: process from raw Loupe data
            return self._process_point_from_loupe(request)
    
    # ------------------------------------------------------------------ #
    #  Fluorescence domain processing
    # ------------------------------------------------------------------ #

    def _load_fluor_loupe_data(
        self, sol: str, target: str, scan: str
    ) -> LoupeData:
        """Load R2/R3 fluorescence data from Loupe working directory.

        Returns a LoupeData whose ``spectra_df`` has a ``wavelength`` column
        (nm) instead of ``raman_shift`` (cm-1), plus integer point columns.
        """
        from sherloc_pipeline.core.data_ingestion import DataIngestion

        try:
            ingestion = DataIngestion(
                base_data_dir=self.context.data_root,
                results_dir=self.context.results_root,
                sol=sol,
                target=target,
                scan=scan,
            )
            working_dir = ingestion.find_working_directory(sol, scan)
            if working_dir is None:
                raise SpectralPlotError(
                    f"Working directory not found for sol {sol}, scan {scan}",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan},
                )
            metadata = ingestion.load_scan_metadata(working_dir)
            n_spectra = int(metadata.get("n_spectra", 0))
            ppp = float(metadata.get("shots_per_spec", 0))
            if n_spectra == 0:
                raise SpectralPlotError(
                    f"Invalid n_spectra in metadata: {n_spectra}",
                    exit_code=1,
                    context={"working_dir": str(working_dir)},
                )
            try:
                spectra_df = ingestion.load_laser_normalized_spectra(working_dir)
            except FileNotFoundError:
                spectra_df = ingestion.load_dark_subtracted_spectra(working_dir)

            fluor_df = ingestion.restructure_fluorescence_data(spectra_df, n_spectra)
            return LoupeData(
                spectra_df=fluor_df,
                n_points=n_spectra,
                ppp=ppp,
                working_dir=working_dir,
                metadata=metadata,
            )
        except SpectralPlotError:
            raise
        except Exception as e:
            raise SpectralPlotError(
                f"Failed to load fluorescence data: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            ) from e

    def _compute_fluor_average(
        self,
        loupe_data: LoupeData,
        method: Literal["mean", "median", "trim-mean"] = "trim-mean",
        trim_pct: float = 2.0,
        point_subset: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Compute average fluorescence spectrum (wavelength domain).

        Returns DataFrame with ``wavelength`` and ``intensity`` columns.
        """
        df = loupe_data.spectra_df
        point_cols = [c for c in df.columns if isinstance(c, int)]
        if not point_cols:
            raise SpectralPlotError(
                "No point columns found in fluorescence DataFrame",
                exit_code=1,
            )
        if point_subset is not None:
            point_cols = [c for c in point_cols if c in point_subset]
            if not point_cols:
                raise SpectralPlotError(
                    "No matching point columns after subset filter",
                    exit_code=1,
                )
        spectra_array = df[point_cols].values
        if method == "mean":
            avg_intensity = np.mean(spectra_array, axis=1)
        elif method == "median":
            avg_intensity = np.median(spectra_array, axis=1)
        else:
            n_pts = spectra_array.shape[1]
            baseline_pct = trim_pct / 100.0
            proportiontocut = resolve_trim_proportion(n_pts, baseline_pct)
            if proportiontocut != baseline_pct:
                logger.info(
                    "Fluor trim mean: dynamic adjustment for %d points "
                    "(baseline %.1f%% → effective %.1f%% per tail)",
                    n_pts, baseline_pct * 100, proportiontocut * 100,
                )
            avg_intensity = stats.trim_mean(spectra_array, proportiontocut, axis=1)
        return pd.DataFrame({
            "wavelength": df["wavelength"].values,
            "intensity": avg_intensity,
        })

    def _process_fluor(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process fluorescence domain for any mode (averaged/point/subset)."""
        if request.mode == "point":
            return self._process_fluor_point(request)
        elif request.mode == "subset":
            return self._process_fluor_averaged(request, subset=True)
        else:
            return self._process_fluor_averaged(request)

    def _process_fluor_averaged(
        self,
        request: SpectralPlotRequest,
        subset: bool = False,
    ) -> ServiceResult:
        """Process averaged fluorescence spectrum from Loupe R2/R3 data.

        Workflow: load R2/R3 → average → (optional) fit → plot → export.
        Background subtraction and baseline correction are not applicable
        to fluorescence and are skipped even if requested.
        """
        import matplotlib
        matplotlib.use("Agg")

        from sherloc_pipeline.core.fluor_fitting import (
            fit_fluorescence_spectrum,
            FluorFitResult,
        )
        from sherloc_pipeline.visualization.fitting_plots import plot_fluor_fit_overlay

        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Load fluorescence data
        loupe_data = self._load_fluor_loupe_data(
            request.sol, request.target, request.scan
        )
        n_avg = len(request.points) if subset and request.points else loupe_data.n_points
        request.n_points_averaged = n_avg
        metadata["n_points"] = loupe_data.n_points
        metadata["ppp"] = loupe_data.ppp
        metadata["domain"] = "fluor"

        # Warn if Raman-only options are set
        if request.background is not None:
            warnings.append("background subtraction not applicable to fluorescence, skipped")
        if request.baseline:
            warnings.append("baseline correction not applicable to fluorescence, skipped")

        # Compute average
        point_subset = request.points if subset else None
        spectrum_df = self._compute_fluor_average(
            loupe_data,
            method=request.avg_method,
            trim_pct=request.trim_pct,
            point_subset=point_subset,
        )
        metadata["avg_method"] = request.avg_method

        wavelength = spectrum_df["wavelength"].values
        intensity = spectrum_df["intensity"].values

        # Fit fluorescence if requested
        fluor_result: Optional[FluorFitResult] = None
        if request.fit:
            fluor_cfg = {}
            if hasattr(self.context.config, "fluorescence_fitting") and self.context.config.fluorescence_fitting:
                fluor_cfg = self.context.config.fluorescence_fitting
            fluor_result = fit_fluorescence_spectrum(
                wavelength,
                intensity,
                fit_range=tuple(fluor_cfg.get("fit_range", [276.0, 355.0])),
                fwhm_range=tuple(fluor_cfg.get("fwhm_range", [10.0, 40.0])),
                max_peaks=int(fluor_cfg.get("max_peaks", 4)),
                snr_threshold=float(fluor_cfg.get("snr_threshold", 2.0)),
                min_fwhm_nm=float(fluor_cfg.get("min_fwhm_nm", 8.0)),
                saturation_threshold=float(fluor_cfg.get("saturation_threshold", 60000.0)),
                saturation_channel_limit=int(fluor_cfg.get("saturation_channel_limit", 5)),
                strategy=str(fluor_cfg.get("strategy", "agnostic")),
            )
            metadata["fit"] = True
            metadata["n_peaks"] = fluor_result.n_peaks
            metadata["r2"] = fluor_result.r2
            if fluor_result.warnings:
                warnings.extend(fluor_result.warnings)

        # Generate plot
        output_dir = self.context.results_root / request.target / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = self._build_fluor_filename(request, fluor_result)
        png_path = output_dir / f"{base_filename}.png"

        if fluor_result is not None and not fluor_result.fit_skipped:
            plot_fluor_fit_overlay(
                wavelength,
                intensity,
                fluor_result,
                str(png_path),
                sol=request.sol,
                target=request.target,
                scan=request.scan,
            )
        else:
            # Simple fluorescence spectrum plot (no fit)
            self._generate_simple_fluor_plot(
                wavelength, intensity, request, str(png_path)
            )

        artifacts: list[Path] = [png_path]

        # Export CSV
        if request.export in ("csv", "both"):
            csv_path = output_dir / f"{base_filename}.csv"
            spectrum_df.to_csv(csv_path, index=False)
            artifacts.append(csv_path)

        # Build summary
        mode_label = "subset" if subset else "averaged"
        processing_parts = [self._format_avg_method(request)]
        if request.fit and fluor_result is not None:
            processing_parts.append(f"fit ({fluor_result.n_peaks} peaks)")
        summary = (
            f"Processed {mode_label} fluorescence spectrum for "
            f"{request.sol}/{request.target}/{request.scan} "
            f"({loupe_data.n_points} points, {loupe_data.ppp} PPP): "
            f"{', '.join(processing_parts)}"
        )
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )

    def _process_fluor_point(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process single-point fluorescence spectrum from Loupe R2/R3 data."""
        import matplotlib
        matplotlib.use("Agg")

        from sherloc_pipeline.core.fluor_fitting import (
            fit_fluorescence_spectrum,
            FluorFitResult,
        )
        from sherloc_pipeline.visualization.fitting_plots import plot_fluor_fit_overlay

        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        loupe_data = self._load_fluor_loupe_data(
            request.sol, request.target, request.scan
        )
        metadata["ppp"] = loupe_data.ppp
        metadata["total_points"] = loupe_data.n_points
        metadata["domain"] = "fluor"

        if request.point is None:
            raise SpectralPlotError(
                "Point mode requires --point to be specified", exit_code=1
            )
        if request.point < 0 or request.point >= loupe_data.n_points:
            raise SpectralPlotError(
                f"Point {request.point} out of range (0-{loupe_data.n_points - 1})",
                exit_code=1,
            )
        metadata["point"] = request.point

        point_col = request.point
        spectrum_df = pd.DataFrame({
            "wavelength": loupe_data.spectra_df["wavelength"],
            "intensity": loupe_data.spectra_df[point_col].astype(float),
        })

        wavelength = spectrum_df["wavelength"].values
        intensity = spectrum_df["intensity"].values

        # Fit fluorescence if requested
        fluor_result: Optional[FluorFitResult] = None
        if request.fit:
            fluor_cfg = {}
            if hasattr(self.context.config, "fluorescence_fitting") and self.context.config.fluorescence_fitting:
                fluor_cfg = self.context.config.fluorescence_fitting
            fluor_result = fit_fluorescence_spectrum(
                wavelength,
                intensity,
                fit_range=tuple(fluor_cfg.get("fit_range", [276.0, 355.0])),
                fwhm_range=tuple(fluor_cfg.get("fwhm_range", [10.0, 40.0])),
                max_peaks=int(fluor_cfg.get("max_peaks", 4)),
                snr_threshold=float(fluor_cfg.get("snr_threshold", 2.0)),
                min_fwhm_nm=float(fluor_cfg.get("min_fwhm_nm", 8.0)),
                saturation_threshold=float(fluor_cfg.get("saturation_threshold", 60000.0)),
                saturation_channel_limit=int(fluor_cfg.get("saturation_channel_limit", 5)),
                strategy=str(fluor_cfg.get("strategy", "agnostic")),
            )
            metadata["fit"] = True
            metadata["n_peaks"] = fluor_result.n_peaks
            metadata["r2"] = fluor_result.r2
            if fluor_result.warnings:
                warnings.extend(fluor_result.warnings)

        # Generate plot
        output_dir = self.context.results_root / request.target / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        base_filename = self._build_fluor_filename(request, fluor_result)
        png_path = output_dir / f"{base_filename}.png"

        if fluor_result is not None and not fluor_result.fit_skipped:
            plot_fluor_fit_overlay(
                wavelength,
                intensity,
                fluor_result,
                str(png_path),
                sol=request.sol,
                target=request.target,
                scan=request.scan,
                point=request.point,
            )
        else:
            self._generate_simple_fluor_plot(
                wavelength, intensity, request, str(png_path)
            )

        artifacts: list[Path] = [png_path]

        if request.export in ("csv", "both"):
            csv_path = output_dir / f"{base_filename}.csv"
            spectrum_df.to_csv(csv_path, index=False)
            artifacts.append(csv_path)

        processing_parts = [f"point {request.point}"]
        if request.fit and fluor_result is not None:
            processing_parts.append(f"fit ({fluor_result.n_peaks} peaks)")
        summary = (
            f"Processed fluorescence point spectrum for "
            f"{request.sol}/{request.target}/{request.scan} "
            f"(point {request.point}, {loupe_data.ppp} PPP): "
            f"{', '.join(processing_parts)}"
        )
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )

    def _process_both(self, request: SpectralPlotRequest) -> ServiceResult:
        """Process both Raman and fluorescence domains, returning combined artifacts."""
        from dataclasses import replace

        # Run Raman processing (existing path)
        raman_request = replace(request, domain="raman")
        raman_result = self.process(raman_request)

        # Run fluorescence processing
        fluor_request = replace(request, domain="fluor")
        fluor_result = self.process(fluor_request)

        # Combine results
        combined_artifacts = list(raman_result.artifacts) + list(fluor_result.artifacts)
        combined_warnings = list(raman_result.warnings) + list(fluor_result.warnings)
        combined_metadata = {
            "raman": raman_result.metadata,
            "fluor": fluor_result.metadata,
        }
        summary = (
            f"Processed both domains for {request.sol}/{request.target}/{request.scan}: "
            f"Raman ({raman_result.summary.split(': ', 1)[-1]}), "
            f"Fluorescence ({fluor_result.summary.split(': ', 1)[-1]})"
        )
        return ServiceResult(
            summary=summary,
            artifacts=combined_artifacts,
            warnings=combined_warnings,
            metadata=combined_metadata,
        )

    def _generate_simple_fluor_plot(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        request: SpectralPlotRequest,
        output_path: str,
    ) -> None:
        """Generate a simple fluorescence spectrum plot without fitting overlay."""
        from sherloc_pipeline.visualization.plotting import apply_plot_config

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wavelength, intensity, color="#1f77b4", linewidth=1.2, label="spectrum")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (counts)")
        title_parts = [f"sol {request.sol}", request.target, request.scan, "fluorescence"]
        if request.mode == "point" and request.point is not None:
            title_parts.append(f"p{request.point}")
        else:
            title_parts.append(f"avg {request.avg_method}")
        ax.set_title(" ".join(title_parts))
        ax.grid(True, alpha=0.3)
        if request.xlim is not None:
            ax.set_xlim(list(request.xlim))
        apply_plot_config(fig)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = str(Path(output_path).with_suffix(".pdf"))
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    def _build_fluor_filename(
        self,
        request: SpectralPlotRequest,
        fluor_result: Optional[object] = None,
    ) -> str:
        """Build output filename for fluorescence domain (uses 'fluor' instead of 'R1')."""
        if request.mode == "point":
            parts = [request.sol, request.target, request.scan, "fluor", f"p{request.point}"]
            if request.fit and fluor_result is not None:
                parts.append("fit")
            return "_".join(parts)

        # Averaged / subset
        avg_label = self._format_avg_method(request)
        parts = [request.sol, request.target, request.scan, "fluor"]
        if request.mode == "subset":
            if request.points and len(request.points) <= 10:
                pts_str = "-".join(str(p) for p in sorted(request.points))
                parts.append(f"subset-pts{pts_str}-{avg_label}")
            else:
                n = len(request.points) if request.points else 0
                parts.append(f"subset-{n}pts-{avg_label}")
        else:
            parts.append(f"avg-{avg_label}")
        if request.fit and fluor_result is not None:
            parts.append("fit")
        return "_".join(parts)

    def _process_point_from_results(self, request: SpectralPlotRequest) -> ServiceResult:
        """Load and visualize single point from pipeline outputs (legacy behavior).
        
        Workflow:
        1. Locate existing CSV in results/<target>/<sol>_<scan>/
        2. Load specified processing level
        3. Extract specified point column
        4. Apply axis controls
        5. Render and export plot
        
        Args:
            request: SpectralPlotRequest with mode="point" and level specified
            
        Returns:
            ServiceResult with visualization artifacts
            
        Raises:
            SpectralPlotError: If loading or visualization fails
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for saving
        
        warnings: list[str] = []
        metadata: dict[str, Any] = {}
        
        # Step 1: Load spectrum from pipeline outputs
        spectrum_df = self._load_pipeline_output(
            sol=request.sol,
            target=request.target,
            scan=request.scan,
            level=request.level,
            point=request.point,
        )
        metadata["point"] = request.point
        metadata["level"] = request.level
        metadata["n_datapoints"] = len(spectrum_df)
        
        # Step 2: Generate plot
        fig = self._generate_plot(
            spectrum_df,
            request,
            fit_result=None,  # Results mode doesn't support fitting
            model_array=None,
        )
        
        # Step 3: Export outputs
        artifacts = self._export(
            spectrum_df,
            fig,
            request,
            fit_result=None,
        )
        
        # Close figure to free memory
        plt.close(fig)
        
        # Build summary
        summary = (
            f"Visualized point {request.point} for {request.sol}/{request.target}/{request.scan} "
            f"({request.level})"
        )
        
        return ServiceResult(
            summary=summary,
            artifacts=artifacts,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _load_pipeline_output(
        self,
        sol: str,
        target: str,
        scan: str,
        level: str,
        point: int,
    ) -> pd.DataFrame:
        """Load single point spectrum from existing pipeline outputs.
        
        This method loads processed spectra from the pipeline's results directory
        and extracts a single point's spectrum for visualization.
        
        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan identifier (e.g., "detail_1")
            level: Processing level - one of:
                - "normalized": After laser normalization
                - "normalized_baselined": After baseline correction
                - "normalized_despiked_baselined": After despiking and baseline
            point: Point index (0-based) to extract
            
        Returns:
            DataFrame with two columns:
            - raman_shift: Raman shift values in cm^-1
            - intensity: Intensity values for the specified point
            
        Raises:
            SpectralPlotError: If file not found or point index out of range
            
        Example:
            >>> df = service._load_pipeline_output(
            ...     "0921", "Amherst_Point", "detail_1",
            ...     level="normalized_baselined", point=91
            ... )
            >>> print(df.columns.tolist())
            ['raman_shift', 'intensity']
        """
        # Validate level
        valid_levels = {
            "normalized",
            "normalized_baselined",
            "normalized_despiked_baselined",
        }
        if level not in valid_levels:
            raise SpectralPlotError(
                f"Invalid processing level: '{level}'. "
                f"Valid options: {', '.join(sorted(valid_levels))}",
                exit_code=1,
                context={"level": level, "valid_levels": list(valid_levels)}
            )
        
        # Build expected file path
        # Pattern: results/<target>/<sol>_<scan>/<sol>_<target>_<scan>_R1_<level>.csv
        results_dir = self.context.results_root
        scan_dir = results_dir / target / f"{sol}_{scan}"
        filename = f"{sol}_{target}_{scan}_R1_{level}.csv"
        csv_path = scan_dir / filename
        
        # Check if file exists
        if not csv_path.exists():
            raise SpectralPlotError(
                f"Pipeline output not found: {csv_path}",
                exit_code=1,
                context={
                    "sol": sol,
                    "target": target,
                    "scan": scan,
                    "level": level,
                    "expected_path": str(csv_path)
                }
            )
        
        try:
            # Load CSV - expect raman_shift as first column, then point columns as integers
            df = pd.read_csv(csv_path)
            
            # Validate structure
            if "raman_shift" not in df.columns:
                raise SpectralPlotError(
                    f"Missing 'raman_shift' column in {csv_path.name}",
                    exit_code=1,
                    context={"available_columns": list(df.columns)[:10]}
                )
            
            # Point columns should be integers (0, 1, 2, ...)
            # Convert column names from strings to check for point
            point_col = str(point)  # CSV columns are loaded as strings
            
            if point_col not in df.columns:
                # Get available point columns (numeric column names)
                available_points = [c for c in df.columns if c != "raman_shift" and c.isdigit()]
                max_point = max(int(p) for p in available_points) if available_points else -1
                
                raise SpectralPlotError(
                    f"Point {point} not found in {csv_path.name}. "
                    f"Available points: 0-{max_point}" if max_point >= 0 else "No point columns found",
                    exit_code=1,
                    context={
                        "point": point,
                        "max_available_point": max_point,
                        "n_points": len(available_points)
                    }
                )
            
            # Extract raman_shift and specified point as intensity
            result = pd.DataFrame({
                "raman_shift": df["raman_shift"].values,
                "intensity": df[point_col].values,
            })
            
            return result
            
        except SpectralPlotError:
            raise
        except Exception as e:
            raise SpectralPlotError(
                f"Failed to load pipeline output: {e}",
                exit_code=1,
                context={
                    "csv_path": str(csv_path),
                    "sol": sol,
                    "target": target,
                    "scan": scan,
                    "level": level,
                    "point": point
                }
            )
    
    def _load_loupe_data(self, sol: str, target: str, scan: str) -> LoupeData:
        """Load Loupe spectral data for averaging.
        
        This method locates the Loupe working directory, loads the laser-normalized
        spectra (darkSubSpectraN.csv), and restructures them into R1 Raman format.
        
        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan identifier (e.g., "detail_1")
            
        Returns:
            LoupeData with spectra DataFrame, metadata, and PPP value
            
        Raises:
            SpectralPlotError: If working directory not found or data cannot be loaded
            
        Example:
            >>> data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
            >>> print(f"Loaded {data.n_points} points at {data.ppp} PPP")
            Loaded 100 points at 500.0 PPP
        """
        from sherloc_pipeline.core.data_ingestion import DataIngestion
        
        try:
            # Create DataIngestion instance for the scan
            ingestion = DataIngestion(
                base_data_dir=self.context.data_root,
                results_dir=self.context.results_root,
                sol=sol,
                target=target,
                scan=scan,
            )
            
            # Find the working directory
            working_dir = ingestion.find_working_directory(sol, scan)
            if working_dir is None:
                raise SpectralPlotError(
                    f"Working directory not found for sol {sol}, scan {scan}",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan}
                )
            
            # Load metadata from loupe.csv
            metadata = ingestion.load_scan_metadata(working_dir)
            n_spectra = int(metadata.get('n_spectra', 0))
            ppp = float(metadata.get('shots_per_spec', 0))
            
            if n_spectra == 0:
                raise SpectralPlotError(
                    f"Invalid n_spectra in metadata: {n_spectra}",
                    exit_code=1,
                    context={"working_dir": str(working_dir)}
                )
            
            # Load laser-normalized spectra (preferred) or raw dark-subtracted
            try:
                spectra_df = ingestion.load_laser_normalized_spectra(working_dir)
            except FileNotFoundError:
                # Fall back to raw dark-subtracted if normalized not available
                spectra_df = ingestion.load_dark_subtracted_spectra(working_dir)
            
            # Restructure to R1 (Raman) format with raman_shift and point columns
            r1_df = ingestion.restructure_raman_data(spectra_df, n_spectra)
            
            return LoupeData(
                spectra_df=r1_df,
                n_points=n_spectra,
                ppp=ppp,
                working_dir=working_dir,
                metadata=metadata,
            )
            
        except SpectralPlotError:
            raise
        except FileNotFoundError as e:
            raise SpectralPlotError(
                f"File not found: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )
        except Exception as e:
            raise SpectralPlotError(
                f"Failed to load Loupe data: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan}
            )

    def _compute_average(
        self,
        loupe_data: LoupeData,
        method: Literal["mean", "median", "trim-mean"] = "trim-mean",
        trim_pct: float = 2.0,
    ) -> pd.DataFrame:
        """Compute average spectrum from loaded Loupe data.
        
        This method reduces multiple point spectra to a single averaged spectrum
        using the specified averaging method.
        
        Args:
            loupe_data: LoupeData containing spectra_df with raman_shift and point columns
            method: Averaging method - "mean", "median", or "trim-mean"
            trim_pct: Percentage to trim from each end for trim-mean (0-50).
                      Only used when method="trim-mean". Default is 2%.
                      
        Returns:
            DataFrame with two columns:
            - raman_shift: Raman shift values in cm^-1
            - intensity: Averaged intensity values
            
        Raises:
            SpectralPlotError: If averaging fails
            ValueError: If invalid method or trim_pct
            
        Example:
            >>> data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
            >>> avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
            >>> print(avg_df.columns.tolist())
            ['raman_shift', 'intensity']
        """
        if method not in ("mean", "median", "trim-mean"):
            raise ValueError(f"Invalid averaging method: {method}")
        
        if trim_pct < 0 or trim_pct > 50:
            raise ValueError(f"trim_pct must be between 0 and 50, got {trim_pct}")
        
        df = loupe_data.spectra_df
        
        # Extract point columns (integer column names)
        point_cols = [c for c in df.columns if isinstance(c, int)]
        if not point_cols:
            raise SpectralPlotError(
                "No point columns found in spectra DataFrame",
                exit_code=1,
                context={"available_columns": list(df.columns)[:10]}  # Show first 10
            )
        
        # Extract spectra values as 2D array (rows=wavelengths, cols=points)
        spectra_array = df[point_cols].values
        
        # Compute average based on method
        if method == "mean":
            avg_intensity = np.mean(spectra_array, axis=1)
        elif method == "median":
            avg_intensity = np.median(spectra_array, axis=1)
        else:  # trim-mean
            # scipy.stats.trim_mean takes proportiontocut (0-0.5)
            n_pts = spectra_array.shape[1]
            baseline_pct = trim_pct / 100.0
            proportiontocut = resolve_trim_proportion(n_pts, baseline_pct)
            if proportiontocut != baseline_pct:
                logger.info(
                    "Raman trim mean: dynamic adjustment for %d points "
                    "(baseline %.1f%% → effective %.1f%% per tail)",
                    n_pts, baseline_pct * 100, proportiontocut * 100,
                )
            avg_intensity = stats.trim_mean(spectra_array, proportiontocut, axis=1)
        
        # Build result DataFrame
        result = pd.DataFrame({
            "raman_shift": df["raman_shift"].values,
            "intensity": avg_intensity,
        })
        
        return result

    def _load_background(
        self,
        bg_type: Literal["as", "fs"],
    ) -> pd.DataFrame:
        """Load background spectrum from config path or fallback defaults.
        
        This method loads the appropriate background spectrum based on type:
        - "as": Arm Stowed post-anomaly (900 PPP)
        - "fs": Fused Silica Corning 7980 air-subtracted (900 PPP)
        
        Background paths and column mappings are read from config.yaml under
        preprocessing.background_subtraction.backgrounds. Falls back to
        hard-coded defaults if config section is missing.
        
        The returned DataFrame is normalized to have consistent column names
        (raman_shift, intensity) regardless of the source file format.
        
        Args:
            bg_type: Background type - "as" or "fs"
            
        Returns:
            DataFrame with columns:
            - raman_shift: Raman shift values in cm^-1
            - intensity: Background intensity values
            
        Raises:
            SpectralPlotError: If background type is invalid or file not found
            
        Example:
            >>> bg_df = service._load_background("fs")
            >>> print(bg_df.columns.tolist())
            ['raman_shift', 'intensity']
        """
        if bg_type not in ("as", "fs"):
            raise SpectralPlotError(
                f"Invalid background type: {bg_type}. Must be 'as' or 'fs'.",
                exit_code=1,
                context={"bg_type": bg_type}
            )
        
        # Hard-coded defaults (fallback if config missing)
        default_filenames = {
            "as": "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv",
            "fs": "Fused_Silica_Corning7980_Air_Subtracted-Bandwidth-35_SB-Pitt.csv",
        }
        default_column_mappings = {
            "as": {"raman_shift": "raman_shift", "intensity": "intensity"},
            "fs": {"raman_shift": "Raman shift (cm-1)", "intensity": "Intensity"},
        }
        
        # Try to read from config
        bg_config = self.context.config.preprocessing.get(
            "background_subtraction", {}
        ).get("backgrounds", {}).get(bg_type, {})
        
        # Use config values if available, otherwise fall back to defaults
        bg_filename = bg_config.get("file", default_filenames[bg_type])
        col_config = bg_config.get("columns", {})
        col_map = {
            "raman_shift": col_config.get("raman_shift", default_column_mappings[bg_type]["raman_shift"]),
            "intensity": col_config.get("intensity", default_column_mappings[bg_type]["intensity"]),
        }
        
        logger.debug(f"Loading background '{bg_type}': file={bg_filename}")
        
        # Try to find background file
        # 1. Check in data_root/../background/
        # 2. Check in data_root/../../background/
        # 3. Fall back to absolute path from config if available
        possible_paths = [
            self.context.data_root.parent / "background" / bg_filename,
            self.context.data_root.parent.parent / "background" / bg_filename,
            self.context.data_root / "background" / bg_filename,
        ]
        
        bg_path = None
        for path in possible_paths:
            if path.exists():
                bg_path = path
                break
        
        if bg_path is None:
            raise SpectralPlotError(
                f"Background file not found: {bg_filename}",
                exit_code=1,
                context={
                    "bg_type": bg_type,
                    "searched_paths": [str(p) for p in possible_paths],
                }
            )
        
        try:
            # Load background CSV
            bg_df = pd.read_csv(bg_path)
            
            # Normalize column names
            result = pd.DataFrame({
                "raman_shift": bg_df[col_map["raman_shift"]].values,
                "intensity": bg_df[col_map["intensity"]].values,
            })
            
            return result
            
        except KeyError as e:
            raise SpectralPlotError(
                f"Background file missing expected columns: {e}",
                exit_code=1,
                context={"bg_path": str(bg_path), "bg_type": bg_type}
            )
        except Exception as e:
            raise SpectralPlotError(
                f"Failed to load background: {e}",
                exit_code=1,
                context={"bg_path": str(bg_path), "bg_type": bg_type}
            )

    def _apply_background_subtraction(
        self,
        spectrum_df: pd.DataFrame,
        bg_type: Literal["as", "fs"],
        scale: float,
    ) -> pd.DataFrame:
        """Apply scaled background subtraction to spectrum.
        
        This method:
        1. Loads the background spectrum (AS or FS)
        2. Interpolates the background to match the spectrum's x-axis
        3. Scales the background by the provided scale factor
        4. Subtracts the scaled background from the spectrum
        
        Args:
            spectrum_df: DataFrame with 'raman_shift' and 'intensity' columns
            bg_type: Background type - "as" (arm stowed) or "fs" (fused silica)
            scale: Scale factor to apply to background before subtraction.
                   Typically calculated via calculate_background_scale().
                   
        Returns:
            DataFrame with same structure as input, background-subtracted intensity
            
        Raises:
            SpectralPlotError: If background cannot be loaded or applied
            ValueError: If spectrum_df is missing required columns
            
        Example:
            >>> avg_df = service._compute_average(data, method="trim-mean")
            >>> scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
            >>> subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        
        Notes:
            - Background is interpolated using linear interpolation
            - Points outside the background's x-range are extrapolated
            - For best results, ensure spectrum x-range is within background x-range
        """
        # Validate input
        required_cols = {"raman_shift", "intensity"}
        missing = required_cols - set(spectrum_df.columns)
        if missing:
            raise ValueError(f"spectrum_df missing required columns: {missing}")
        
        # Load background
        bg_df = self._load_background(bg_type)
        
        # Get background x-range for sanity check
        bg_x_min = bg_df["raman_shift"].min()
        bg_x_max = bg_df["raman_shift"].max()
        
        # Create interpolation function for background
        # Use fill_value="extrapolate" to handle edge cases where spectrum
        # extends beyond background range
        bg_interp = interpolate.interp1d(
            bg_df["raman_shift"].values,
            bg_df["intensity"].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        
        # Interpolate background to spectrum's x-axis
        spectrum_x = spectrum_df["raman_shift"].values
        
        # Sanity check: warn if spectrum exceeds background range by >5%
        spectrum_x_min = spectrum_x.min()
        spectrum_x_max = spectrum_x.max()
        bg_range = bg_x_max - bg_x_min
        
        extrapolation_low = max(0, bg_x_min - spectrum_x_min)
        extrapolation_high = max(0, spectrum_x_max - bg_x_max)
        total_extrapolation = extrapolation_low + extrapolation_high
        extrapolation_pct = (total_extrapolation / bg_range) * 100 if bg_range > 0 else 0
        
        if extrapolation_pct > 5:
            logger.warning(
                f"Spectrum x-range [{spectrum_x_min:.1f}, {spectrum_x_max:.1f}] extends beyond "
                f"background range [{bg_x_min:.1f}, {bg_x_max:.1f}] by {extrapolation_pct:.1f}%. "
                f"Extrapolated values may be inaccurate."
            )
        
        bg_interpolated = bg_interp(spectrum_x)
        
        # Apply scale and subtract
        scaled_bg = bg_interpolated * scale
        subtracted_intensity = spectrum_df["intensity"].values - scaled_bg
        
        # Build result DataFrame
        result = pd.DataFrame({
            "raman_shift": spectrum_x,
            "intensity": subtracted_intensity,
        })
        
        return result

    def _apply_baseline(
        self,
        spectrum_df: pd.DataFrame,
        params: Optional[BaselineParams] = None,
    ) -> pd.DataFrame:
        """Apply asPLS baseline correction to spectrum.
        
        This method uses the existing asPLS baseline algorithm to fit and
        subtract a smooth baseline from the spectrum, correcting for
        fluorescence and other broad spectral features.
        
        Args:
            spectrum_df: DataFrame with 'raman_shift' and 'intensity' columns
            params: Optional BaselineParams for asPLS fitting. If None, uses
                    config defaults or pipeline defaults:
                    - lam: 1e6 (smoothness parameter)
                    - diff_order: 2 (derivative order for smoothness)
                    
        Returns:
            DataFrame with same structure as input, baseline-corrected intensity
            
        Raises:
            SpectralPlotError: If baseline correction fails
            ValueError: If spectrum_df is missing required columns
            
        Example:
            >>> avg_df = service._compute_average(data, method="trim-mean")
            >>> corrected = service._apply_baseline(avg_df)
            >>> print(corrected['intensity'].min())  # Often near zero after correction
            
        Notes:
            - Uses asymmetric penalized least squares (asPLS) algorithm
            - The baseline tends to follow the lower envelope of the spectrum
            - Best results when applied after background subtraction
            - Config parameters are read from preprocessing.baseline section
        """
        # Validate input
        required_cols = {"raman_shift", "intensity"}
        missing = required_cols - set(spectrum_df.columns)
        if missing:
            raise ValueError(f"spectrum_df missing required columns: {missing}")
        
        # Get baseline parameters from config or use defaults
        if params is None:
            # Try to read from config if available
            try:
                config = self.context.config
                baseline_config = config.preprocessing.get("baseline", {})
                params = BaselineParams(
                    lam=baseline_config.get("lam", 1e6),
                    asymmetric_coef=baseline_config.get("asymmetric_coef", 0.01),
                    iters=baseline_config.get("iters", 10),
                    diff_order=baseline_config.get("diff_order", 2),
                    tol=baseline_config.get("tol", 1e-3),
                )
            except (AttributeError, KeyError):
                # Use sensible defaults if config not available
                params = BaselineParams(
                    lam=1e6,
                    diff_order=2,
                )
        
        try:
            # Create intensity Series with raman_shift as index for baseline function
            intensity_series = pd.Series(
                spectrum_df["intensity"].values,
                index=spectrum_df.index,
            )

            # Build weight vector from config keep_windows (same as pipeline)
            x_vals = spectrum_df["raman_shift"].values
            try:
                config = self.context.config
                bl_cfg = config.preprocessing.get("baseline", {})
                kw = bl_cfg.get("keep_windows", [])
                keep_windows = [tuple(map(float, w)) for w in kw] if kw else [
                    (600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)
                ]
                keep_weight = float(bl_cfg.get("keep_weight", 0.01))
            except (AttributeError, KeyError):
                keep_windows = [(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)]
                keep_weight = 0.01

            weights = build_weight_vector_from_windows(
                x_vals, keep_windows=keep_windows,
                default_weight=1.0, keep_weight=keep_weight,
            )
            corrected_series, baseline_series = _baseline_aspls_with_weights(
                intensity_series, params, weights,
            )

            # Build result DataFrame
            result = pd.DataFrame({
                "raman_shift": spectrum_df["raman_shift"].values,
                "intensity": corrected_series.values,
            })

            return result
            
        except Exception as e:
            raise SpectralPlotError(
                f"Baseline correction failed: {e}",
                exit_code=1,
                context={"error": str(e)}
            )

    def _apply_fitting(
        self,
        spectrum_df: pd.DataFrame,
        fit_range: Optional[Tuple[float, float]] = None,
        single_peak_center: Optional[float] = None,
        n_peaks: Optional[int] = None,
        min_snr: Optional[float] = None,
        fwhm_min: Optional[float] = None,
        fwhm_max: Optional[float] = None,
    ) -> Tuple[FitResult, np.ndarray]:
        """Apply Gaussian fitting to spectrum.
        
        This method uses the existing multi-Gaussian fitting algorithm to
        identify and fit peaks in the spectrum. It wraps the core `fit_spectrum`
        function with appropriate configuration defaults.
        
        Args:
            spectrum_df: DataFrame with 'raman_shift' and 'intensity' columns
            fit_range: Optional fit range as (min, max) in cm^-1. If None, uses
                       config default (typically 700-1200 for R1 mineral region).
            single_peak_center: Optional center position in cm^-1 for single-peak
                               fitting. When provided, exactly one Gaussian is fitted
                               near this position, bypassing automatic peak detection
                               and AICc model selection.
            n_peaks: Optional maximum number of peaks to fit. When provided, 
                    constrains the AICc model selection to at most n_peaks.
                    Mutually exclusive with single_peak_center.
            min_snr: Optional SNR threshold override (default from config: 3.0).
                    Peaks below this SNR are flagged as rejected.
            fwhm_min: Optional minimum FWHM override in cm^-1 (default from config: 30).
                     Peaks narrower than this are flagged as rejected.
            fwhm_max: Optional maximum FWHM override in cm^-1 (default from config: 90).
                     Peaks wider than this are flagged as rejected.
                       
        Returns:
            Tuple of:
            - FitResult: Contains peaks list, R², RSS, DOF, and warnings
            - np.ndarray: Model spectrum (same length as input)
            
        Raises:
            SpectralPlotError: If fitting fails
            ValueError: If spectrum_df is missing required columns, or if both
                       single_peak_center and n_peaks are provided
            
        Example:
            >>> avg_df = service._compute_average(data, method="trim-mean")
            >>> corrected = service._apply_baseline(avg_df)
            >>> fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
            >>> print(f"Found {len(fit_result.peaks)} peaks with R²={fit_result.r2:.3f}")
            Found 3 peaks with R²=0.987
            
            >>> # Single-peak fitting for carbonate at ~1090 cm^-1
            >>> fit_result, model = service._apply_fitting(
            ...     corrected, fit_range=(1000, 1200), single_peak_center=1090
            ... )
            >>> print(f"Fitted peak at {fit_result.peaks[0].m_cm1:.1f} cm^-1")
            Fitted peak at 1089.3 cm^-1
            
            >>> # Limit to at most 2 peaks
            >>> fit_result, model = service._apply_fitting(
            ...     corrected, fit_range=(700, 1200), n_peaks=2
            ... )
            >>> print(f"Found {len(fit_result.peaks)} peaks")
            Found 2 peaks
            
        Notes:
            - Uses multi-Gaussian fitting with automatic peak detection
            - Peak quality filters (SNR, FWHM, R²) are applied per config
            - The returned model array can be used for overlay plotting
            - For best results, apply background subtraction and baseline
              correction before fitting
            - When single_peak_center is provided, bypasses AICc model selection
              and fits exactly one Gaussian seeded at the specified position
            - When n_peaks is provided, AICc still selects optimal count (1 to n_peaks)
        """
        # Validate input
        required_cols = {"raman_shift", "intensity"}
        missing = required_cols - set(spectrum_df.columns)
        if missing:
            raise ValueError(f"spectrum_df missing required columns: {missing}")
        
        # Validate mutually exclusive options
        if single_peak_center is not None and n_peaks is not None:
            raise ValueError("single_peak_center and n_peaks are mutually exclusive")
        
        # Get fitting config from context or use defaults
        try:
            config = self.context.config
            fitting_config = config.fitting
        except (AttributeError, KeyError):
            # Use sensible defaults if config not available
            fitting_config = {}
        
        # Build cfg dict for fit_spectrum
        # Start with defaults and overlay with config
        cfg = {
            "r1_fit_range": fitting_config.get("r1_fit_range", [700, 1200]),
            "fit_fwhm_min_initial_cm1": fitting_config.get("fit_fwhm_min_initial_cm1", 22),
            "filter_fwhm_min_cm1": fitting_config.get("filter_fwhm_min_cm1", 30),
            "fwhm_max_cm1": fitting_config.get("fwhm_max_cm1", 90),
            "slit_width_cm1_default": fitting_config.get("slit_width_cm1_default", 34.1),
            "slit_pref_weight": fitting_config.get("slit_pref_weight", 0.2),
            "low_fwhm_edge_penalty": fitting_config.get("low_fwhm_edge_penalty", 0.1),
            "max_peaks": fitting_config.get("max_peaks", 5),
            "min_snr": fitting_config.get("min_snr", 3.0),
            "min_seed_snr": fitting_config.get("min_seed_snr", 2.0),
            "min_display_snr": fitting_config.get("min_display_snr", 2.0),
            "r_squared_min": fitting_config.get("r_squared_min", 0.25),
            "peak_separation_cm1": fitting_config.get("peak_separation_cm1", 25),
            "min_amp_sigma_multiplier": fitting_config.get("min_amp_sigma_multiplier", 0.3),
            "parsimony": fitting_config.get("parsimony", {
                "use_aicc": True,
                "aicc_min_peaks": 1,
                "aicc_max_peaks": 5,
                "aicc_improve_threshold": 0.0,
            }),
        }
        
        # For single-peak mode, override max_peaks to 1
        if single_peak_center is not None:
            cfg["max_peaks"] = 1
        
        # For n_peaks mode, constrain AICc model selection
        if n_peaks is not None:
            cfg["max_peaks"] = n_peaks
            cfg["parsimony"]["aicc_max_peaks"] = n_peaks
        
        # Apply threshold overrides if provided
        if min_snr is not None:
            cfg["min_snr"] = min_snr
            # Also update min_seed_snr and min_display_snr to be consistent
            # (these should be at or below min_snr to avoid filtering during detection)
            cfg["min_seed_snr"] = min(cfg["min_seed_snr"], min_snr)
            cfg["min_display_snr"] = min(cfg["min_display_snr"], min_snr)
        
        if fwhm_min is not None:
            cfg["filter_fwhm_min_cm1"] = fwhm_min
            # Also update initial FWHM min to be consistent
            cfg["fit_fwhm_min_initial_cm1"] = min(cfg["fit_fwhm_min_initial_cm1"], fwhm_min)
        
        if fwhm_max is not None:
            cfg["fwhm_max_cm1"] = fwhm_max
        
        try:
            # Extract x and y arrays
            x_cm1 = spectrum_df["raman_shift"].values.astype(float)
            y = spectrum_df["intensity"].values.astype(float)
            
            # Determine fit range
            roi = None
            if fit_range is not None:
                roi = fit_range
            # If not specified, fit_spectrum will use cfg["r1_fit_range"]
            
            # Prepare seed_centers for single-peak mode
            seed_centers = None
            if single_peak_center is not None:
                seed_centers = [single_peak_center]
            
            # Call core fitting function
            fit_result, model_array = fit_spectrum(
                x_cm1=x_cm1,
                y=y,
                cfg=cfg,
                roi=roi,
                seed_centers=seed_centers,
            )
            
            return fit_result, model_array
            
        except Exception as e:
            raise SpectralPlotError(
                f"Gaussian fitting failed: {e}",
                exit_code=1,
                context={"error": str(e), "fit_range": fit_range}
            )

    def apply_fitting(
        self,
        spectrum_df: pd.DataFrame,
        fit_range: Optional[Tuple[float, float]] = None,
        single_peak_center: Optional[float] = None,
        n_peaks: Optional[int] = None,
        min_snr: Optional[float] = None,
        fwhm_min: Optional[float] = None,
        fwhm_max: Optional[float] = None,
    ) -> Tuple[FitResult, np.ndarray]:
        """Apply Gaussian fitting to spectrum (public interface).

        Delegates to _apply_fitting. See _apply_fitting for full documentation.
        """
        return self._apply_fitting(
            spectrum_df,
            fit_range=fit_range,
            single_peak_center=single_peak_center,
            n_peaks=n_peaks,
            min_snr=min_snr,
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
        )

    def _generate_plot(
        self,
        spectrum_df: pd.DataFrame,
        request: SpectralPlotRequest,
        fit_result: Optional[FitResult] = None,
        model_array: Optional[np.ndarray] = None,
    ) -> Figure:
        """Generate spectral plot with optional fit overlay.
        
        This method creates a matplotlib Figure containing the processed spectrum
        with optional Gaussian fit overlay and peak parameter legend. The plot
        follows the existing pipeline styling conventions.
        
        Args:
            spectrum_df: DataFrame with 'raman_shift' and 'intensity' columns
            request: SpectralPlotRequest containing plot parameters (xlim, ylim, etc.)
            fit_result: Optional FitResult from Gaussian fitting
            model_array: Optional model spectrum array (same length as spectrum_df)
                        Required if fit_result is provided.
                        
        Returns:
            matplotlib Figure object ready for saving or display
            
        Raises:
            SpectralPlotError: If plot generation fails
            ValueError: If spectrum_df is missing required columns
            
        Example:
            >>> avg_df = service._compute_average(data, method="trim-mean")
            >>> corrected = service._apply_baseline(avg_df)
            >>> fig = service._generate_plot(corrected, request)
            >>> fig.savefig("spectrum.png", dpi=300)
            
        Notes:
            - Title format: "sol <sol> <target> <scan> R1 avg <method> [processing]"
            - When fitting is enabled, peak parameters are added to legend
            - Uses PlotConfig for consistent styling across pipeline
        """
        # Validate input
        required_cols = {"raman_shift", "intensity"}
        missing = required_cols - set(spectrum_df.columns)
        if missing:
            raise ValueError(f"spectrum_df missing required columns: {missing}")
        
        # Configure matplotlib with pipeline defaults
        configure_matplotlib()
        
        # Extract data
        x = spectrum_df["raman_shift"].values
        y = spectrum_df["intensity"].values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot main spectrum
        main_line, = ax.plot(x, y, color='#1f77b4', linewidth=1.2)
        
        # Build legend handles
        handles = [Line2D([0], [0], color='#1f77b4', lw=1.2, label='processed spectrum')]
        text_colors = ['black']
        
        # Add fit overlay if provided
        if fit_result is not None and model_array is not None:
            # Build a smooth plotting grid for Gaussian rendering
            x_smooth = np.linspace(x.min(), x.max(), max(2000, int(len(x) * 4)))
            
            # Recompute smooth model from fitted peaks for display
            y_model_smooth = np.zeros_like(x_smooth, dtype=float)
            for p in fit_result.peaks:
                y_model_smooth += gaussian(x_smooth, p.m_cm1, p.a, p.fwhm)
            
            # Plot model
            ax.plot(x_smooth, y_model_smooth, color='#2ca02c', linewidth=1.2)
            handles.append(
                Line2D([0], [0], color='#2ca02c', lw=1.2, 
                       label=f'model (R²={fit_result.r2:.3f})')
            )
            text_colors.append('black')
            
            # Plot individual peak components with parameters in legend
            cycle = ['#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf']
            ci = 0
            for p in fit_result.peaks:
                y_comp = gaussian(x_smooth, p.m_cm1, p.a, p.fwhm)
                failing = (not p.pass_fwhm) or (not p.pass_snr) or (not p.pass_r2)
                style = ':' if failing else '-'
                color = '#d62728' if failing else cycle[ci % len(cycle)]
                if not failing:
                    ci += 1
                ax.plot(x_smooth, y_comp, linestyle=style, color=color, linewidth=1.0)
                label = f"m: {p.m_cm1:.1f}, a: {p.a:.1f}, FWHM: {p.fwhm:.1f}, SNR: {p.snr:.1f}"
                handles.append(
                    Line2D([0], [0], color=color, lw=1.5, linestyle=style, label=label)
                )
                text_colors.append('red' if failing else 'black')
        
        # Set axis labels
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Intensity (counts)')
        
        # Build title: "sol <sol> <target> <scan> R1 avg <method> [processing]"
        # For subset mode, title may include multiple lines with points list
        title = self._build_plot_title(request, fit_result)
        
        # Count title lines to adjust figure layout
        title_lines = title.count('\n') + 1
        
        # Use suptitle for multi-line titles to avoid overlap with plot
        # Smaller font for multi-line titles to fit better
        title_fontsize = 10 if title_lines > 1 else 11
        fig.suptitle(title, fontsize=title_fontsize, y=0.98, 
                     verticalalignment='top', linespacing=1.3)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Apply axis limits
        if request.xlim is not None:
            ax.set_xlim(list(request.xlim))
            
            # Auto-scale Y to visible X range if ylim not explicitly set
            if request.ylim is None:
                xmin, xmax = request.xlim
                mask = (x >= xmin) & (x <= xmax)
                if mask.any():
                    y_visible = y[mask]
                    y_margin = (y_visible.max() - y_visible.min()) * 0.05
                    ax.set_ylim(y_visible.min() - y_margin, y_visible.max() + y_margin)
        
        if request.ylim is not None:
            ax.set_ylim(list(request.ylim))
        
        # Place legend
        if fit_result is not None and fit_result.peaks:
            # Fitting plots: legend to the right with peak parameters
            leg = ax.legend(
                handles=handles, 
                loc='upper left', 
                bbox_to_anchor=(1.02, 1.0), 
                borderaxespad=0.0, 
                framealpha=0.85
            )
            # Colorize legend text for failing peaks
            for txt, col in zip(leg.get_texts(), text_colors):
                if col != 'black':
                    txt.set_color(col)
            
            # Fitting plots need wide right margins for the legend
            # Adjust top margin based on title line count
            top_margin = 0.94 - (title_lines - 1) * 0.04  # Extra space per line
            fitting_margins = {
                "left": 0.08,
                "right": 0.62,
                "bottom": 0.12,
                "top": top_margin,
            }
            apply_plot_config(fig, margins_override=fitting_margins)
        else:
            # Simple spectrum: no legend needed (just one line)
            # Adjust top margin based on title line count
            if title_lines > 1:
                top_margin = 0.94 - (title_lines - 1) * 0.04
                margins = {
                    "left": 0.08,
                    "right": 0.95,
                    "bottom": 0.12,
                    "top": top_margin,
                }
                apply_plot_config(fig, margins_override=margins)
            else:
                apply_plot_config(fig)
        
        return fig

    def _build_plot_title(
        self,
        request: SpectralPlotRequest,
        fit_result: Optional[FitResult] = None,
    ) -> str:
        """Build plot title from request parameters with intelligent wrapping.
        
        Title format for averaged mode:
            "sol <sol> <target> <scan> R1 avg <method> [processing]"
        
        Title format for subset mode (includes full points list):
            Line 1: "sol <sol> <target> <scan> R1 subset (<n> pts) <method> [processing]"
            Line 2: "points: 21, 41, 49, 71, ..." (wrapped if needed)
        
        Title format for point mode:
            "sol <sol> <target> <scan> R1 point <point> <level>"
        
        Processing components (averaged/subset mode only):
        - Background type (as/fs) if applied
        - "baselined" if baseline correction applied
        - "fit" if fitting applied
        
        Args:
            request: SpectralPlotRequest with processing parameters
            fit_result: Optional FitResult (for R² in title if desired)
            
        Returns:
            Formatted title string (may contain newlines for multi-line titles)
            
        Example:
            >>> title = service._build_plot_title(request)
            "sol 0921 Amherst_Point detail_1 R1 avg 2p_trim_mean fs baselined"
            
            >>> title = service._build_plot_title(subset_request)
            "sol 0921 Amherst_Point detail_1 R1 subset (11 pts) 2p_trim_mean fs baselined\\npoints: 21, 41, 49, 71, 86, 87, 88, 90, 91, 92, 98"
            
            >>> title = service._build_plot_title(point_request)
            "sol 0921 Amherst_Point detail_1 R1 point 91 normalized_baselined"
        """
        # Point mode
        if request.mode == "point":
            if request.level is not None:
                # Legacy: results mode with processing level
                return (
                    f"sol {request.sol} {request.target} {request.scan} "
                    f"R1 point {request.point} {request.level}"
                )
            else:
                # New: Loupe mode with processing flags
                title_parts = [
                    f"sol {request.sol}",
                    request.target,
                    request.scan,
                    "R1",
                    f"point {request.point}",
                ]
                # Add processing indicators
                processing_parts = []
                if request.background is not None:
                    processing_parts.append(request.background)
                if request.baseline:
                    processing_parts.append("baselined")
                if request.fit and fit_result is not None:
                    processing_parts.append("fit")
                if processing_parts:
                    title_parts.append(" ".join(processing_parts))
                return " ".join(title_parts)
        
        # Averaged/Subset mode: format averaging method
        avg_label = self._format_avg_method(request)
        
        # Base title parts
        title_parts = [
            f"sol {request.sol}",
            request.target,
            request.scan,
            "R1",
        ]
        
        # Add mode-specific label
        if request.mode == "subset":
            n_points = len(request.points) if request.points else 0
            title_parts.append(f"subset ({n_points} pts) {avg_label}")
        else:
            title_parts.append(f"avg {avg_label}")
        
        # Add processing indicators
        processing_parts = []
        if request.background is not None:
            processing_parts.append(request.background)
        if request.baseline:
            processing_parts.append("baselined")
        if request.fit and fit_result is not None:
            processing_parts.append("fit")
        
        if processing_parts:
            title_parts.append(" ".join(processing_parts))
        
        main_title = " ".join(title_parts)
        
        # For subset mode, add points list on second line
        if request.mode == "subset" and request.points:
            points_str = ", ".join(str(p) for p in request.points)
            points_line = f"points: {points_str}"
            
            # Wrap points line if too long (target ~70 chars per line)
            if len(points_line) > 75:
                # Wrap just the points portion, keeping "points: " prefix
                wrapped_points = textwrap.fill(
                    points_str, 
                    width=70,
                    initial_indent="points: ",
                    subsequent_indent="  "
                )
                return f"{main_title}\n{wrapped_points}"
            else:
                return f"{main_title}\n{points_line}"
        
        return main_title

    def _format_avg_method(self, request: SpectralPlotRequest) -> str:
        """Format averaging method label for filenames and titles.

        Converts the averaging method parameters into a standardized label:
        - mean → "mean"
        - median → "median"
        - trim-mean → "Xp_trim_mean" where X is the effective trim percentage
          (accounts for dynamic trim adjustment on small scans)

        Args:
            request: SpectralPlotRequest containing avg_method, trim_pct,
                and optionally n_points_averaged (set during processing)

        Returns:
            Formatted method label string
        """
        if request.avg_method == "trim-mean":
            if request.n_points_averaged is not None:
                return format_trim_label(
                    request.n_points_averaged,
                    request.trim_pct / 100.0,
                )
            # Fallback when n_points unknown (e.g. unit tests)
            if request.trim_pct == int(request.trim_pct):
                pct_str = f"{int(request.trim_pct)}p"
            else:
                pct_str = f"{request.trim_pct}p"
            return f"{pct_str}_trim_mean"
        else:
            return request.avg_method

    def _build_filename(
        self,
        request: SpectralPlotRequest,
        fit_result: Optional[FitResult] = None,
    ) -> str:
        """Build output filename (without extension) from request parameters.
        
        Naming convention for averaged mode:
            <sol>_<target>_<scan>_R1_avg-<method>[_<bg>][_baselined][_fit]
        
        Naming convention for subset mode:
            If ≤10 points: <sol>_<target>_<scan>_R1_subset-pts<p1>-<p2>-...-<method>[_<bg>][_baselined][_fit]
            If >10 points: <sol>_<target>_<scan>_R1_subset-<n>pts-<method>[_<bg>][_baselined][_fit]
        
        Naming convention for point mode:
            <sol>_<target>_<scan>_R1_p<point>_<level>
        
        Args:
            request: SpectralPlotRequest with processing parameters
            fit_result: Optional FitResult (determines if "_fit" suffix is added)
            
        Returns:
            Filename string without extension
            
        Examples:
            >>> # Averaged mode
            >>> filename = service._build_filename(request)
            "0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined_fit"
            
            >>> # Subset mode with ≤10 points (includes point numbers)
            >>> filename = service._build_filename(subset_request_small)
            "0921_Amherst_Point_detail_1_R1_subset-pts21-41-49-2p_trim_mean_fs_baselined_fit"
            
            >>> # Subset mode with >10 points (count only)
            >>> filename = service._build_filename(subset_request_large)
            "0921_Amherst_Point_detail_1_R1_subset-11pts-2p_trim_mean_fs_baselined_fit"
            
            >>> # Point mode
            >>> filename = service._build_filename(point_request)
            "0921_Amherst_Point_detail_1_R1_p91_normalized_despiked_baselined"
        """
        if request.mode == "point":
            if request.level is not None:
                # Legacy: results mode naming: <sol>_<target>_<scan>_R1_p<point>_<level>
                return f"{request.sol}_{request.target}_{request.scan}_R1_p{request.point}_{request.level}"
            else:
                # New: Loupe mode naming with processing flags
                parts = [
                    request.sol,
                    request.target,
                    request.scan,
                    "R1",
                    f"p{request.point}",
                ]
                # Add processing indicators (same pattern as averaged/subset)
                if request.background is not None:
                    parts.append(request.background)
                if request.baseline:
                    parts.append("baselined")
                if request.fit and fit_result is not None:
                    parts.append("fit")
                # Add xlim range to filename if specified
                if request.xlim is not None:
                    xmin, xmax = request.xlim
                    parts.append(f"{int(xmin)}-{int(xmax)}")
                return "_".join(parts)
        
        # Averaged/Subset mode naming
        avg_label = self._format_avg_method(request)
        
        # Base filename parts
        parts = [
            request.sol,
            request.target,
            request.scan,
            "R1",
        ]
        
        # Add mode-specific label
        if request.mode == "subset":
            n_points = len(request.points) if request.points else 0
            # Include point numbers in filename if ≤10 points
            if request.points and n_points <= 10:
                # Format: subset-pts21-41-49-<method>
                pts_str = "-".join(str(p) for p in request.points)
                parts.append(f"subset-pts{pts_str}-{avg_label}")
            else:
                # Too many points, just use count
                parts.append(f"subset-{n_points}pts-{avg_label}")
        else:
            # Averaged mode: avg-<method>
            parts.append(f"avg-{avg_label}")
        
        # Add processing indicators
        if request.background is not None:
            parts.append(request.background)
        if request.baseline:
            parts.append("baselined")
        if request.fit and fit_result is not None:
            parts.append("fit")
        
        # Add xlim range to filename if specified (to distinguish zoomed plots)
        if request.xlim is not None:
            xmin, xmax = request.xlim
            parts.append(f"{int(xmin)}-{int(xmax)}")
        
        return "_".join(parts)

    def _build_metadata(
        self,
        request: SpectralPlotRequest,
        loupe_data: Optional[LoupeData] = None,
        fit_result: Optional[FitResult] = None,
        bg_scale_used: Optional[float] = None,
        output_dir: Optional[Path] = None,
        base_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build comprehensive metadata dictionary for JSON export.
        
        Captures all processing parameters and results for reproducibility
        and traceability. The metadata includes everything needed to understand
        and potentially reproduce the analysis.
        
        Args:
            request: SpectralPlotRequest with all processing parameters
            loupe_data: Optional LoupeData with source information
            fit_result: Optional FitResult with peak fitting results
            bg_scale_used: Actual background scale factor used
            output_dir: Output directory path
            base_filename: Base filename (without extension)
            
        Returns:
            Dictionary ready for JSON serialization
            
        Notes:
            - numpy types are converted to Python natives for JSON compatibility
            - Path objects are converted to strings
            - NaN/Inf values are replaced with None
        """
        def _safe_float(val: Any) -> Optional[float]:
            """Convert to float, handling numpy types and NaN/Inf."""
            if val is None:
                return None
            try:
                f = float(val)
                if np.isnan(f) or np.isinf(f):
                    return None
                return f
            except (TypeError, ValueError):
                return None
        
        def _safe_int(val: Any) -> Optional[int]:
            """Convert to int, handling numpy types."""
            if val is None:
                return None
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
        
        # Get pipeline version from pyproject.toml or hardcode
        pipeline_version = "2.0.0"
        
        metadata: Dict[str, Any] = {
            "metadata_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": pipeline_version,
            
            "scan": {
                "sol": request.sol,
                "target": request.target,
                "scan": request.scan,
            },
            
            "mode": request.mode,
        }
        
        # Source information
        source: Dict[str, Any] = {}
        if loupe_data is not None:
            source["working_directory"] = str(loupe_data.working_dir)
            source["n_spectra_total"] = _safe_int(loupe_data.n_points)
            source["ppp"] = _safe_float(loupe_data.ppp)
        if request.mode == "point":
            source["processing_level"] = request.level
        metadata["source"] = source
        
        # Selection information
        selection: Dict[str, Any] = {}
        if request.mode == "averaged":
            selection["type"] = "all"
            if loupe_data:
                selection["n_points_selected"] = _safe_int(loupe_data.n_points)
        elif request.mode == "subset":
            selection["type"] = "subset"
            selection["points"] = list(request.points) if request.points else []
            selection["n_points_selected"] = len(request.points) if request.points else 0
        elif request.mode == "point":
            selection["type"] = "single"
            selection["point"] = _safe_int(request.point)
            selection["n_points_selected"] = 1
        metadata["selection"] = selection
        
        # Averaging parameters (averaged/subset modes)
        if request.mode in ("averaged", "subset"):
            averaging: Dict[str, Any] = {
                "method": request.avg_method,
            }
            if request.avg_method == "trim-mean":
                averaging["trim_pct"] = _safe_float(request.trim_pct)
                n_avg = selection.get("n_points_selected", 0)
                if n_avg and n_avg > 0:
                    baseline_pct = request.trim_pct / 100.0
                    effective = resolve_trim_proportion(n_avg, baseline_pct)
                    averaging["effective_pct_per_tail"] = round(effective * 100, 2)
                    averaging["n_points_averaged"] = n_avg
                    averaging["m_trimmed_per_tail"] = int(effective * n_avg)
            metadata["averaging"] = averaging
        
        # Background subtraction
        if request.mode in ("averaged", "subset"):
            background: Dict[str, Any] = {"applied": request.background is not None}
            if request.background is not None:
                background["type"] = request.background
                background["scale_mode"] = "auto" if request.bgscale == "auto" else "manual"
                if bg_scale_used is not None:
                    background["scale_value"] = _safe_float(bg_scale_used)
                background["source_ppp"] = 900  # Both AS and FS are 900 PPP
                # Add source filenames
                bg_filenames = {
                    "as": "Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv",
                    "fs": "Fused_Silica_Corning7980_Air_Subtracted-Bandwidth-35_SB-Pitt.csv",
                }
                background["source_file"] = bg_filenames.get(request.background)
            metadata["background"] = background
        
        # Baseline correction
        if request.mode in ("averaged", "subset"):
            baseline_info: Dict[str, Any] = {"applied": request.baseline}
            if request.baseline:
                baseline_info["method"] = "aspls"
                # Get config defaults
                try:
                    config = self.context.config
                    baseline_config = config.preprocessing.get("baseline", {})
                    baseline_info["params"] = {
                        "lam": baseline_config.get("lam", 1e6),
                        "diff_order": baseline_config.get("diff_order", 2),
                    }
                except (AttributeError, KeyError):
                    baseline_info["params"] = {"lam": 1e6, "diff_order": 2}
            metadata["baseline"] = baseline_info
        
        # Fitting information
        if request.mode in ("averaged", "subset"):
            fitting: Dict[str, Any] = {"applied": request.fit}
            if request.fit:
                # Determine fitting mode
                if request.single_peak_center is not None:
                    fitting["mode"] = "single_peak"
                    fitting["single_peak_center"] = _safe_float(request.single_peak_center)
                elif request.n_peaks is not None:
                    fitting["mode"] = "n_peaks_limited"
                    fitting["n_peaks_limit"] = _safe_int(request.n_peaks)
                else:
                    fitting["mode"] = "auto"
                
                fitting["fit_range"] = list(request.fit_range) if request.fit_range else [700, 1200]
                
                if fit_result is not None:
                    results: Dict[str, Any] = {
                        "n_peaks_found": len(fit_result.peaks),
                        "r_squared": _safe_float(fit_result.r2),
                        "peaks": []
                    }
                    for p in fit_result.peaks:
                        peak_info = {
                            "center_cm1": _safe_float(p.m_cm1),
                            "amplitude": _safe_float(p.a),
                            "fwhm_cm1": _safe_float(p.fwhm),
                            "area": _safe_float(p.area),
                            "snr": _safe_float(p.snr),
                            "passed_qc": p.pass_fwhm and p.pass_snr and p.pass_r2,
                            "pass_fwhm": p.pass_fwhm,
                            "pass_snr": p.pass_snr,
                            "pass_r2": p.pass_r2,
                        }
                        results["peaks"].append(peak_info)
                    fitting["results"] = results
            metadata["fitting"] = fitting
        
        # Plot settings
        plot_info: Dict[str, Any] = {}
        if request.xlim is not None:
            plot_info["xlim"] = list(request.xlim)
        if request.ylim is not None:
            plot_info["ylim"] = list(request.ylim)
        plot_info["title"] = self._build_plot_title(request, fit_result)
        metadata["plot"] = plot_info
        
        # Output information
        if output_dir is not None and base_filename is not None:
            outputs: Dict[str, Any] = {
                "directory": str(output_dir),
                "basename": base_filename,
                "files": {}
            }
            if request.export in ("csv", "both"):
                outputs["files"]["csv"] = f"{base_filename}.csv"
            if request.export in ("png", "both"):
                outputs["files"]["png"] = f"{base_filename}.png"
            if not request.no_metadata:
                outputs["files"]["json"] = f"{base_filename}.json"
            metadata["outputs"] = outputs
        
        return metadata

    def _export(
        self,
        spectrum_df: pd.DataFrame,
        fig: Figure,
        request: SpectralPlotRequest,
        fit_result: Optional[FitResult] = None,
        loupe_data: Optional[LoupeData] = None,
        bg_scale_used: Optional[float] = None,
    ) -> list[Path]:
        """Export spectrum data, plot, and metadata to files.
        
        Creates output directory if needed and saves files based on request.export:
        - "csv": Save CSV only
        - "png": Save PNG only
        - "both": Save both CSV and PNG
        
        Additionally, exports JSON metadata unless request.no_metadata is True.
        The metadata file captures all processing parameters for reproducibility.
        
        Output directory: results/<target>/plots/
        
        Args:
            spectrum_df: DataFrame with 'raman_shift' and 'intensity' columns
            fig: matplotlib Figure to save as PNG
            request: SpectralPlotRequest with export settings
            fit_result: Optional FitResult (affects filename if fitting was applied)
            loupe_data: Optional LoupeData for source metadata
            bg_scale_used: Optional background scale factor actually used
            
        Returns:
            List of Path objects for created files
            
        Raises:
            SpectralPlotError: If export fails (JSON failures are warnings, not errors)
            ValueError: If spectrum_df is missing required columns
            
        Example:
            >>> paths = service._export(spectrum_df, fig, request)
            >>> for p in paths:
            ...     print(f"Created: {p}")
            Created: /path/to/results/Amherst_Point/plots/0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined.csv
            Created: /path/to/results/Amherst_Point/plots/0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined.png
            Created: /path/to/results/Amherst_Point/plots/0921_Amherst_Point_detail_1_R1_avg-2p_trim_mean_fs_baselined.json
        """
        # Validate input
        required_cols = {"raman_shift", "intensity"}
        missing = required_cols - set(spectrum_df.columns)
        if missing:
            raise ValueError(f"spectrum_df missing required columns: {missing}")
        
        # Build output directory: results/<target>/plots/
        output_dir = self.context.results_root / request.target / "plots"
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SpectralPlotError(
                f"Failed to create output directory: {output_dir}",
                exit_code=1,
                context={"error": str(e), "output_dir": str(output_dir)}
            )
        
        # Build base filename
        base_filename = self._build_filename(request, fit_result)
        
        # Track created artifacts
        artifacts: list[Path] = []
        
        try:
            # Export CSV if requested
            if request.export in ("csv", "both"):
                csv_path = output_dir / f"{base_filename}.csv"
                spectrum_df.to_csv(csv_path, index=False)
                artifacts.append(csv_path)
            
            # Export PNG if requested
            if request.export in ("png", "both"):
                png_path = output_dir / f"{base_filename}.png"
                fig.savefig(png_path, dpi=300, bbox_inches="tight")
                artifacts.append(png_path)
            
        except Exception as e:
            raise SpectralPlotError(
                f"Failed to export files: {e}",
                exit_code=1,
                context={
                    "error": str(e),
                    "output_dir": str(output_dir),
                    "base_filename": base_filename,
                }
            )
        
        # Export JSON metadata (unless disabled)
        # JSON export failures are warnings, not hard errors
        if not request.no_metadata:
            try:
                metadata = self._build_metadata(
                    request=request,
                    loupe_data=loupe_data,
                    fit_result=fit_result,
                    bg_scale_used=bg_scale_used,
                    output_dir=output_dir,
                    base_filename=base_filename,
                )
                json_path = output_dir / f"{base_filename}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                artifacts.append(json_path)
            except Exception as e:
                # JSON export failure is a warning, not a fatal error
                self.console.print(
                    f"[yellow]Warning: Failed to export metadata JSON: {e}[/yellow]"
                )
        
        return artifacts


def calculate_background_scale(
    scan_ppp: float,
    bg_ppp: float = 900.0,
    override: Optional[float] = None,
    scale_bounds: Tuple[float, float] = (0.1, 5.0),
) -> float:
    """Calculate background scaling factor.
    
    Default: PPP-based automatic scaling.
    - Both AS and FS backgrounds represent 900 PPP
    - Scale factor = scan_ppp / 900
    
    Override: User can specify explicit scale (e.g., 0.5) if over-subtraction
    is observed (anomalous dip at ~800 cm^-1 from instrument background).
    
    Args:
        scan_ppp: Pulses per point for the scan (from loupe.csv shots_per_spec)
        bg_ppp: Pulses per point for background spectrum (default 900)
        override: Explicit scale factor if provided
        scale_bounds: (min, max) bounds for warning if scale is unusual
        
    Returns:
        Scale factor to multiply background before subtraction
        
    Example:
        >>> calculate_background_scale(scan_ppp=500, bg_ppp=900)
        0.5555555555555556
        >>> calculate_background_scale(scan_ppp=900, bg_ppp=900)
        1.0
        >>> calculate_background_scale(scan_ppp=500, bg_ppp=900, override=0.5)
        0.5
    """
    if override is not None:
        return override
    
    # Sanity check: warn if scan_ppp is 0 or missing
    if scan_ppp is None or scan_ppp <= 0:
        logger.warning(
            f"scan_ppp is {scan_ppp}, which is invalid. Using scale=1.0 as fallback."
        )
        return 1.0
    
    scale = scan_ppp / bg_ppp
    
    # Warn if scale is outside expected bounds
    min_scale, max_scale = scale_bounds
    if scale < min_scale or scale > max_scale:
        logger.warning(
            f"Background scale {scale:.3f} is outside expected bounds [{min_scale}, {max_scale}]. "
            f"This may indicate unusual PPP values (scan_ppp={scan_ppp}, bg_ppp={bg_ppp})."
        )
    
    return scale


__all__ = [
    "SpectralService",
    "SpectralPlotRequest",
    "SpectralPlotError",
    "LoupeData",
    "BaselineParams",
    "FitResult",
    "PeakFit",
    "calculate_background_scale",
    "gaussian",
]

