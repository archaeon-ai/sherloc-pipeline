"""
Python API for spectral analysis workflows.

This module provides notebook-friendly functions for SHERLOC spectral analysis,
enabling Jupyter workflows without CLI interaction. All functions return clean
DataFrames and matplotlib Figures suitable for interactive exploration.

The API wraps the SpectralService with a functional interface optimized for
notebook usage, including multi-spectrum overlay capabilities.

Example:
    >>> from sherloc_pipeline.api.spectral import (
    ...     process_scan_average,
    ...     load_point_spectrum,
    ...     load_reference_spectrum,
    ...     plot_overlay,
    ... )
    >>> 
    >>> # Process an averaged spectrum with fitting
    >>> df, fit_result = process_scan_average(
    ...     sol="0921", target="Amherst_Point", scan="detail_1",
    ...     background="fs", baseline=True, fit=True
    ... )
    >>> 
    >>> # Load a reference spectrum for comparison
    >>> ref_df = load_reference_spectrum("forsterite")
    >>> 
    >>> # Overlay Mars spectrum with reference
    >>> fig = plot_overlay(
    ...     spectra=[
    ...         {"df": df, "label": "Amherst Point", "color": "#1f77b4"},
    ...         {"df": ref_df, "label": "Forsterite reference", 
    ...          "color": "#2ca02c", "linestyle": "--"},
    ...     ],
    ...     xlim=(700, 1200),
    ...     scale_to_peak=(820, 870),
    ...     title="Olivine comparison"
    ... )
    >>> fig.show()
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Literal, Union, TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from sherloc_pipeline.models.fitting import FitResult


def process_scan_average(
    sol: str,
    target: str,
    scan: str,
    *,
    avg_method: Literal["mean", "median", "trim-mean"] = "trim-mean",
    trim_pct: float = 2.0,
    background: Optional[Literal["as", "fs"]] = "fs",
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = True,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    min_snr: Optional[float] = None,
    fwhm_min: Optional[float] = None,
    fwhm_max: Optional[float] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional["FitResult"]]:
    """Process averaged spectrum from Loupe data.
    
    Loads raw Loupe spectra, computes the average across all points, and
    optionally applies background subtraction, baseline correction, and
    Gaussian fitting.
    
    This is the primary API function for spectral analysis workflows,
    providing the complete mini-pipeline processing chain in a single call.
    
    Args:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan identifier (e.g., "detail_1", "line_1")
        avg_method: Averaging method - "mean", "median", or "trim-mean".
            Default is "trim-mean" which is robust to outliers.
        trim_pct: Percentage to trim from each end for trim-mean (0-50).
            Only used when avg_method="trim-mean". Default is 2%.
        background: Background type for subtraction:
            - "as": Arm Stowed post-anomaly (900 PPP)
            - "fs": Fused Silica air-subtracted (900 PPP)
            - None: No background subtraction
            Default is "fs" (fused silica).
        bgscale: Background scale factor:
            - "auto": PPP-based automatic scaling (scan_ppp / 900)
            - float: Explicit scale factor (e.g., 0.5 for half-intensity)
            Default is "auto".
        baseline: Whether to apply asPLS baseline correction.
            Recommended after background subtraction. Default is True.
        fit: Whether to apply Gaussian peak fitting.
            Returns FitResult with peak parameters if True. Default is False.
        fit_range: Fit range as (min, max) in cm^-1.
            If None, uses default R1 mineral region (700-1200 cm^-1).
        single_peak_center: Center position in cm^-1 for single-peak fitting.
            When provided, fits exactly one Gaussian near this position.
            Mutually exclusive with n_peaks.
        n_peaks: Maximum number of peaks to fit.
            Constrains AICc model selection. Mutually exclusive with
            single_peak_center.
        data_dir: Optional path to Loupe data directory. If None, uses
            default from RuntimeContext.
        results_dir: Optional path to results directory. If None, uses
            default from RuntimeContext.
            
    Returns:
        Tuple of:
        - DataFrame with columns:
            - raman_shift: Raman shift values in cm^-1
            - intensity: Processed intensity values
        - Optional[FitResult]: Peak fitting results if fit=True, else None.
            Contains peaks list, R², RSS, DOF, and any warnings.
            
    Raises:
        ValueError: If invalid parameters (e.g., invalid avg_method)
        SpectralPlotError: If processing fails (file not found, etc.)
        
    Example:
        >>> # Basic usage with defaults
        >>> df, _ = process_scan_average("0921", "Amherst_Point", "detail_1")
        >>> print(df.head())
           raman_shift  intensity
        0   238.123456   12.345678
        1   238.987654   11.234567
        ...
        
        >>> # Full processing with fitting
        >>> df, fit = process_scan_average(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     background="fs", baseline=True, fit=True
        ... )
        >>> if fit:
        ...     for peak in fit.peaks:
        ...         print(f"Peak at {peak.m_cm1:.1f} cm^-1, FWHM={peak.fwhm:.1f}")
        Peak at 855.3 cm^-1, FWHM=45.2
        Peak at 1003.7 cm^-1, FWHM=38.9
        
        >>> # Single-peak fitting for carbonate
        >>> df, fit = process_scan_average(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     background="fs", baseline=True, fit=True,
        ...     fit_range=(1000, 1200), single_peak_center=1090
        ... )
        
    See Also:
        process_subset_average: For averaging specific point subsets
        load_point_spectrum: For loading individual point spectra
    """
    from sherloc_pipeline.services.spectral import (
        SpectralService,
        calculate_background_scale,
    )
    from sherloc_pipeline.services.runtime import RuntimeContext
    from sherloc_pipeline.models.fitting import FitResult
    
    # Bootstrap RuntimeContext with optional path overrides
    context = RuntimeContext.bootstrap(
        data_dir=data_dir,
        results_dir=results_dir,
    )
    
    # Create service instance
    service = SpectralService(context=context)
    
    # Step 1: Load Loupe data
    loupe_data = service._load_loupe_data(sol, target, scan)
    
    # Step 2: Compute average
    spectrum_df = service._compute_average(
        loupe_data,
        method=avg_method,
        trim_pct=trim_pct,
    )
    
    # Step 3: Apply background subtraction (if requested)
    if background is not None:
        if bgscale == "auto":
            scale = calculate_background_scale(
                scan_ppp=loupe_data.ppp,
                bg_ppp=900.0,
            )
        else:
            scale = float(bgscale)
        
        spectrum_df = service._apply_background_subtraction(
            spectrum_df,
            bg_type=background,
            scale=scale,
        )
    
    # Step 4: Apply baseline correction (if requested)
    if baseline:
        spectrum_df = service._apply_baseline(spectrum_df)
    
    # Step 5: Apply Gaussian fitting (if requested)
    fit_result: Optional[FitResult] = None
    if fit:
        fit_result, _ = service.apply_fitting(
            spectrum_df,
            fit_range=fit_range,
            single_peak_center=single_peak_center,
            n_peaks=n_peaks,
            min_snr=min_snr,
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
        )
    
    return spectrum_df, fit_result


def process_subset_average(
    sol: str,
    target: str,
    scan: str,
    points: List[int],
    *,
    avg_method: Literal["mean", "median", "trim-mean"] = "trim-mean",
    trim_pct: float = 2.0,
    background: Optional[Literal["as", "fs"]] = "fs",
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = True,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    min_snr: Optional[float] = None,
    fwhm_min: Optional[float] = None,
    fwhm_max: Optional[float] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional["FitResult"]]:
    """Process averaged spectrum from a subset of points.
    
    Similar to process_scan_average(), but only averages the specified
    subset of point indices. This enables label-like averaging with
    ad-hoc point selection.
    
    Args:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan identifier (e.g., "detail_1", "line_1")
        points: List of point indices to average (0-based).
            Must contain at least 2 points.
        avg_method: Averaging method. Default is "trim-mean".
        trim_pct: Trim percentage for trim-mean. Default is 2%.
        background: Background type ("as", "fs", or None). Default is "fs".
        bgscale: Background scale ("auto" or float). Default is "auto".
        baseline: Whether to apply baseline correction. Default is True.
        fit: Whether to apply Gaussian fitting. Default is False.
        fit_range: Fit range as (min, max) in cm^-1.
        single_peak_center: Center for single-peak fitting mode.
        n_peaks: Maximum number of peaks to fit.
        data_dir: Optional path to Loupe data directory.
        results_dir: Optional path to results directory.
            
    Returns:
        Tuple of (DataFrame, Optional[FitResult])
        
    Raises:
        ValueError: If points list is empty or contains invalid indices
        SpectralPlotError: If processing fails
        
    Example:
        >>> # Average specific points identified as high-carbonate
        >>> df, fit = process_subset_average(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     points=[21, 41, 49, 71, 86, 87, 88, 90, 91, 92, 98],
        ...     background="fs", baseline=True, fit=True
        ... )
        >>> print(f"Averaged {len(points)} points")
        Averaged 11 points
        
    See Also:
        process_scan_average: For averaging all points in a scan
    """
    from sherloc_pipeline.services.spectral import (
        SpectralService,
        SpectralPlotError,
        LoupeData,
        calculate_background_scale,
    )
    from sherloc_pipeline.services.runtime import RuntimeContext
    from sherloc_pipeline.models.fitting import FitResult
    
    # Validate points
    if not points or len(points) < 2:
        raise ValueError("Subset mode requires at least 2 points to average")
    
    # Bootstrap RuntimeContext with optional path overrides
    context = RuntimeContext.bootstrap(
        data_dir=data_dir,
        results_dir=results_dir,
    )
    
    # Create service instance
    service = SpectralService(context=context)
    
    # Step 1: Load all Loupe data
    loupe_data = service._load_loupe_data(sol, target, scan)
    
    # Step 2: Validate and filter to specified points
    invalid_points = [p for p in points if p < 0 or p >= loupe_data.n_points]
    if invalid_points:
        raise ValueError(
            f"Invalid point indices: {invalid_points}. "
            f"Valid range is 0 to {loupe_data.n_points - 1}."
        )
    
    # Filter spectra_df to only include requested points
    df = loupe_data.spectra_df
    available_cols = [c for c in df.columns if isinstance(c, int)]
    cols_to_keep = ["raman_shift"] + [c for c in available_cols if c in points]
    filtered_df = df[cols_to_keep].copy()
    
    # Create filtered LoupeData with subset
    subset_data = LoupeData(
        spectra_df=filtered_df,
        n_points=len(points),
        ppp=loupe_data.ppp,
        working_dir=loupe_data.working_dir,
        metadata={**loupe_data.metadata, "subset_points": list(points)},
    )
    
    # Step 3: Compute average over subset
    spectrum_df = service._compute_average(
        subset_data,
        method=avg_method,
        trim_pct=trim_pct,
    )
    
    # Step 4: Apply background subtraction (if requested)
    if background is not None:
        if bgscale == "auto":
            scale = calculate_background_scale(
                scan_ppp=loupe_data.ppp,
                bg_ppp=900.0,
            )
        else:
            scale = float(bgscale)
        
        spectrum_df = service._apply_background_subtraction(
            spectrum_df,
            bg_type=background,
            scale=scale,
        )
    
    # Step 5: Apply baseline correction (if requested)
    if baseline:
        spectrum_df = service._apply_baseline(spectrum_df)
    
    # Step 6: Apply Gaussian fitting (if requested)
    fit_result: Optional[FitResult] = None
    if fit:
        fit_result, _ = service.apply_fitting(
            spectrum_df,
            fit_range=fit_range,
            single_peak_center=single_peak_center,
            n_peaks=n_peaks,
            min_snr=min_snr,
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
        )
    
    return spectrum_df, fit_result


def process_point(
    sol: str,
    target: str,
    scan: str,
    point: int,
    *,
    background: Optional[Literal["as", "fs"]] = None,
    bgscale: Union[float, Literal["auto"]] = "auto",
    baseline: bool = False,
    fit: bool = False,
    fit_range: Optional[Tuple[float, float]] = None,
    single_peak_center: Optional[float] = None,
    n_peaks: Optional[int] = None,
    min_snr: Optional[float] = None,
    fwhm_min: Optional[float] = None,
    fwhm_max: Optional[float] = None,
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional["FitResult"]]:
    """Process single point spectrum from Loupe data.
    
    Loads raw Loupe spectra, extracts the specified point, and optionally
    applies background subtraction, baseline correction, and Gaussian fitting.
    
    This function provides point-level processing directly from Loupe data,
    enabling analysis of individual points with the full processing chain.
    
    Args:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan identifier (e.g., "detail_1", "line_1")
        point: Point index (0-based) to process.
        background: Background type for subtraction:
            - "as": Arm Stowed post-anomaly (900 PPP)
            - "fs": Fused Silica air-subtracted (900 PPP)
            - None: No background subtraction
            Default is None (no background subtraction).
        bgscale: Background scale factor:
            - "auto": PPP-based automatic scaling (scan_ppp / 900)
            - float: Explicit scale factor
            Default is "auto".
        baseline: Whether to apply asPLS baseline correction.
            Default is False.
        fit: Whether to apply Gaussian peak fitting.
            Returns FitResult with peak parameters if True. Default is False.
        fit_range: Fit range as (min, max) in cm^-1.
            If None, uses default R1 mineral region (700-1200 cm^-1).
        single_peak_center: Center position in cm^-1 for single-peak fitting.
            When provided, fits exactly one Gaussian near this position.
            Mutually exclusive with n_peaks.
        n_peaks: Maximum number of peaks to fit.
            Constrains AICc model selection. Mutually exclusive with
            single_peak_center.
        data_dir: Optional path to Loupe data directory.
        results_dir: Optional path to results directory.
            
    Returns:
        Tuple of:
        - DataFrame with columns:
            - raman_shift: Raman shift values in cm^-1
            - intensity: Processed intensity values
        - Optional[FitResult]: Peak fitting results if fit=True, else None.
            
    Raises:
        ValueError: If point index is out of range
        SpectralPlotError: If processing fails (file not found, etc.)
        
    Example:
        >>> # Process single point with fitting
        >>> df, fit = process_point(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     point=5, baseline=True, fit=True
        ... )
        >>> print(df.head())
           raman_shift  intensity
        0   238.123456   12.345678
        ...
        
        >>> # Single-peak carbonate fitting on specific point
        >>> df, fit = process_point(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     point=91, background="fs", baseline=True, fit=True,
        ...     fit_range=(1000, 1200), single_peak_center=1090
        ... )
        >>> if fit and fit.peaks:
        ...     print(f"Carbonate peak at {fit.peaks[0].m_cm1:.1f} cm^-1")
        
    See Also:
        process_scan_average: For averaging all points in a scan
        process_subset_average: For averaging specific point subsets
        load_point_spectrum: For loading from pre-processed pipeline outputs
    """
    from sherloc_pipeline.services.spectral import (
        SpectralService,
        SpectralPlotError,
        calculate_background_scale,
    )
    from sherloc_pipeline.services.runtime import RuntimeContext
    from sherloc_pipeline.models.fitting import FitResult
    
    # Bootstrap RuntimeContext with optional path overrides
    context = RuntimeContext.bootstrap(
        data_dir=data_dir,
        results_dir=results_dir,
    )
    
    # Create service instance
    service = SpectralService(context=context)
    
    # Step 1: Load Loupe data
    loupe_data = service._load_loupe_data(sol, target, scan)
    
    # Step 2: Validate point index
    if point < 0 or point >= loupe_data.n_points:
        raise ValueError(
            f"Point {point} out of range. "
            f"Valid range is 0 to {loupe_data.n_points - 1}."
        )
    
    # Step 3: Extract single point as spectrum DataFrame
    # Point columns are integers (0, 1, 2, ...)
    spectrum_df = pd.DataFrame({
        'raman_shift': loupe_data.spectra_df['raman_shift'],
        'intensity': loupe_data.spectra_df[point].astype(float),
    })
    
    # Step 4: Apply background subtraction (if requested)
    if background is not None:
        if bgscale == "auto":
            scale = calculate_background_scale(
                scan_ppp=loupe_data.ppp,
                bg_ppp=900.0,
            )
        else:
            scale = float(bgscale)
        
        spectrum_df = service._apply_background_subtraction(
            spectrum_df,
            bg_type=background,
            scale=scale,
        )
    
    # Step 5: Apply baseline correction (if requested)
    if baseline:
        spectrum_df = service._apply_baseline(spectrum_df)
    
    # Step 6: Apply Gaussian fitting (if requested)
    fit_result: Optional[FitResult] = None
    if fit:
        fit_result, _ = service.apply_fitting(
            spectrum_df,
            fit_range=fit_range,
            single_peak_center=single_peak_center,
            n_peaks=n_peaks,
            min_snr=min_snr,
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
        )
    
    return spectrum_df, fit_result


def load_point_spectrum(
    sol: str,
    target: str,
    scan: str,
    point: int,
    level: Literal[
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined"
    ],
    *,
    results_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load single point spectrum from pipeline outputs.
    
    Loads a pre-processed spectrum for a specific point from the
    results directory. This function is useful for examining
    individual point spectra after running the full pipeline.
    
    Args:
        sol: Sol number (e.g., "0921")
        target: Target name (e.g., "Amherst_Point")
        scan: Scan identifier (e.g., "detail_1", "line_1")
        point: Point index (0-based)
        level: Processing level to load:
            - "normalized": After laser normalization only
            - "normalized_baselined": After baseline correction
            - "normalized_despiked_baselined": After despiking and baseline
        results_dir: Optional path to results directory. If None, uses
            default from RuntimeContext.
            
    Returns:
        DataFrame with columns:
        - raman_shift: Raman shift values in cm^-1
        - intensity: Intensity values for the specified point
        
    Raises:
        SpectralPlotError: If file not found or point index out of range
        
    Example:
        >>> # Load point 91 from pipeline outputs
        >>> df = load_point_spectrum(
        ...     sol="0921", target="Amherst_Point", scan="detail_1",
        ...     point=91, level="normalized_despiked_baselined"
        ... )
        >>> print(f"Loaded {len(df)} data points")
        Loaded 523 data points
        
        >>> # Quick visualization
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(df['raman_shift'], df['intensity'])
        >>> plt.xlabel('Raman Shift (cm⁻¹)')
        >>> plt.show()
        
    See Also:
        process_scan_average: For processing averaged spectra from Loupe data
    """
    from sherloc_pipeline.services.spectral import SpectralService
    from sherloc_pipeline.services.runtime import RuntimeContext
    
    # Bootstrap RuntimeContext with optional path override
    context = RuntimeContext.bootstrap(
        results_dir=results_dir,
    )
    
    # Create service instance
    service = SpectralService(context=context)
    
    # Load the spectrum using the service's internal method
    spectrum_df = service._load_pipeline_output(
        sol=sol,
        target=target,
        scan=scan,
        level=level,
        point=point,
    )
    
    return spectrum_df


def load_reference_spectrum(
    mineral: str,
    *,
    library_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load reference mineral spectrum from library.
    
    Loads a reference spectrum for comparison with Mars data.
    Reference spectra are typically lab measurements of pure minerals
    at known conditions.
    
    Args:
        mineral: Mineral name (e.g., "forsterite", "gypsum").
            Case-insensitive. Uses fuzzy matching if exact match not found.
        library_path: Optional path to reference library directory.
            If None, uses default from configuration (tests/fixtures/reference).
            
    Returns:
        DataFrame with columns:
        - raman_shift: Raman shift values in cm^-1
        - intensity: Reference intensity values (typically normalized)
        
    Raises:
        FileNotFoundError: If mineral not found in library
        ValueError: If library contains no matching spectra
        
    Example:
        >>> # Load forsterite reference
        >>> ref_df = load_reference_spectrum("forsterite")
        >>> print(f"Reference spans {ref_df['raman_shift'].min():.0f} - "
        ...       f"{ref_df['raman_shift'].max():.0f} cm^-1")
        Reference spans 100 - 4000 cm^-1
        
        >>> # Compare with Mars spectrum
        >>> mars_df, _ = process_scan_average(
        ...     "0921", "Amherst_Point", "detail_1",
        ...     background="fs", baseline=True
        ... )
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(mars_df['raman_shift'], mars_df['intensity'], 
        ...         label='Mars', color='blue')
        >>> ax.plot(ref_df['raman_shift'], ref_df['intensity'] * scale_factor,
        ...         label='Forsterite', color='green', linestyle='--')
        >>> ax.legend()
        >>> plt.show()
        
    See Also:
        plot_overlay: For automated multi-spectrum overlay with scaling
    """
    # Determine library path
    if library_path is None:
        # Default to tests/fixtures/reference relative to package or cwd
        # Try multiple potential locations
        potential_paths = [
            Path("tests/fixtures/reference"),
            Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "reference",
        ]
        library_path = None
        for path in potential_paths:
            if path.exists():
                library_path = path
                break
        
        if library_path is None:
            raise FileNotFoundError(
                "Reference library not found. Provide library_path argument or "
                "ensure tests/fixtures/reference/ exists."
            )
    
    library_path = Path(library_path)
    if not library_path.exists():
        raise FileNotFoundError(f"Reference library not found: {library_path}")
    
    # Search for matching files (case-insensitive)
    mineral_lower = mineral.lower()
    matches = []
    
    for csv_file in library_path.glob("*.csv"):
        # Match if mineral name appears in filename (case-insensitive)
        if mineral_lower in csv_file.name.lower():
            matches.append(csv_file)
    
    if not matches:
        available = [f.stem for f in library_path.glob("*.csv")]
        raise FileNotFoundError(
            f"No reference spectrum found for mineral '{mineral}'. "
            f"Available: {', '.join(available) if available else 'none'}"
        )
    
    # Use the first match (or could implement priority/fuzzy logic later)
    reference_file = matches[0]
    
    try:
        # Load the CSV - forsterite format has 2 header rows
        # Row 0: metadata (starts with #)
        # Row 1: column names (also starts with #)
        # Try to auto-detect format by checking first line
        with open(reference_file) as f:
            first_line = f.readline()
        
        # If first line starts with #, likely has metadata headers
        if first_line.startswith("#"):
            df = pd.read_csv(reference_file, skiprows=2, header=None)
            # Forsterite format columns:
            # 0: wavelength (nm)
            # 1: Raman shift (cm-1)
            # 2: Average Intensity (counts) - includes background
            # 6: Baselined Intensity (counts) - background-subtracted
            # 8: Baselined Normalised Intensity - normalized to peak
            # Use column 6 (Baselined Intensity) for comparison with 
            # background-subtracted Mars spectra
            if len(df.columns) >= 7:
                df = pd.DataFrame({
                    "raman_shift": df.iloc[:, 1].values,  # Raman shift
                    "intensity": df.iloc[:, 6].values,    # Baselined Intensity
                })
            elif len(df.columns) >= 3:
                # Fallback to average intensity if baselined not available
                df = pd.DataFrame({
                    "raman_shift": df.iloc[:, 1].values,
                    "intensity": df.iloc[:, 2].values,
                })
            else:
                raise ValueError(f"Unexpected column count in {reference_file.name}")
        else:
            # Standard CSV format - try to find raman_shift column
            df = pd.read_csv(reference_file)
            
            # Try to identify columns
            rs_col = None
            int_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if "raman" in col_lower and "shift" in col_lower:
                    rs_col = col
                elif rs_col is None and ("cm-1" in col_lower or "cm^-1" in col_lower):
                    rs_col = col
                elif "intensity" in col_lower or "average" in col_lower:
                    int_col = col
            
            if rs_col is None or int_col is None:
                raise ValueError(
                    f"Could not identify raman_shift and intensity columns in {reference_file.name}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            df = pd.DataFrame({
                "raman_shift": df[rs_col].values,
                "intensity": df[int_col].values,
            })
        
        return df
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to load reference spectrum: {e}")


def plot_spectrum(
    df: pd.DataFrame,
    *,
    fit_result: Optional["FitResult"] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    color: str = "#1f77b4",
    linewidth: float = 1.2,
    linestyle: str = "-",
    show_grid: bool = True,
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Render single spectrum with optional fit overlay.
    
    Creates a publication-quality spectral plot with optional Gaussian
    fit overlay and peak parameter annotations.
    
    Args:
        df: DataFrame with 'raman_shift' and 'intensity' columns
        fit_result: Optional FitResult from Gaussian fitting.
            If provided, overlays fitted model and individual peaks.
        xlim: X-axis limits as (min, max) in cm^-1.
            If None, uses full data range.
        ylim: Y-axis limits as (min, max) in intensity units.
            If None, auto-scales to data (with margin).
        title: Plot title. If None, no title is added.
        color: Line color for the main spectrum. Default is matplotlib blue.
        linewidth: Line width for the spectrum. Default is 1.2.
        linestyle: Line style ("-", "--", ":", "-."). Default is solid.
        show_grid: Whether to show grid lines. Default is True.
        figsize: Figure size as (width, height) in inches. Default is (12, 6).
        
    Returns:
        matplotlib Figure object ready for display or saving.
        
    Raises:
        ValueError: If df is missing required columns
        
    Example:
        >>> # Basic spectrum plot
        >>> df, _ = process_scan_average("0921", "Amherst_Point", "detail_1")
        >>> fig = plot_spectrum(df, xlim=(700, 1200), title="Amherst Point")
        >>> fig.savefig("spectrum.png", dpi=300)
        
        >>> # With fit overlay
        >>> df, fit = process_scan_average(
        ...     "0921", "Amherst_Point", "detail_1",
        ...     background="fs", baseline=True, fit=True
        ... )
        >>> fig = plot_spectrum(df, fit_result=fit, xlim=(700, 1200))
        >>> fig.show()
        
        >>> # Custom styling
        >>> fig = plot_spectrum(
        ...     df, 
        ...     color="#e41a1c", 
        ...     linewidth=2.0,
        ...     linestyle="--",
        ...     title="Custom Styled Spectrum"
        ... )
        
    See Also:
        plot_overlay: For plotting multiple spectra together
    """
    import numpy as np
    from matplotlib.lines import Line2D
    from sherloc_pipeline.visualization.plotting import configure_matplotlib, apply_plot_config
    from sherloc_pipeline.services.spectral import gaussian
    from sherloc_pipeline.models.fitting import FitResult
    
    # Validate input
    required_cols = {"raman_shift", "intensity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    # Configure matplotlib with pipeline defaults
    configure_matplotlib()
    
    # Extract data
    x = df["raman_shift"].values
    y = df["intensity"].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main spectrum with custom styling
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle)
    
    # Build legend handles
    handles = [Line2D([0], [0], color=color, lw=linewidth, linestyle=linestyle, label='spectrum')]
    text_colors = ['black']
    
    # Add fit overlay if provided
    if fit_result is not None and hasattr(fit_result, 'peaks') and fit_result.peaks:
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
            peak_color = '#d62728' if failing else cycle[ci % len(cycle)]
            if not failing:
                ci += 1
            ax.plot(x_smooth, y_comp, linestyle=style, color=peak_color, linewidth=1.0)
            label = f"m: {p.m_cm1:.1f}, a: {p.a:.1f}, FWHM: {p.fwhm:.1f}, SNR: {p.snr:.1f}"
            handles.append(
                Line2D([0], [0], color=peak_color, lw=1.5, linestyle=style, label=label)
            )
            text_colors.append('red' if failing else 'black')
    
    # Set axis labels
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (counts)')
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    # Add grid
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Apply axis limits
    if xlim is not None:
        ax.set_xlim(list(xlim))
        
        # Auto-scale Y to visible X range if ylim not explicitly set
        if ylim is None:
            xmin, xmax = xlim
            mask = (x >= xmin) & (x <= xmax)
            if mask.any():
                y_visible = y[mask]
                y_margin = (y_visible.max() - y_visible.min()) * 0.05
                ax.set_ylim(y_visible.min() - y_margin, y_visible.max() + y_margin)
    
    if ylim is not None:
        ax.set_ylim(list(ylim))
    
    # Place legend
    if fit_result is not None and hasattr(fit_result, 'peaks') and fit_result.peaks:
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
        fitting_margins = {
            "left": 0.08,
            "right": 0.62,
            "bottom": 0.12,
            "top": 0.94,
        }
        apply_plot_config(fig, margins_override=fitting_margins)
    else:
        # Simple spectrum: no legend needed (just one line)
        apply_plot_config(fig)
    
    return fig


def plot_overlay(
    spectra: List[Dict[str, Any]],
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    scale_to_peak: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_grid: bool = True,
    figsize: Tuple[float, float] = (12, 6),
    legend_loc: str = "upper right",
) -> Figure:
    """Overlay multiple spectra with per-trace styling.
    
    Creates a multi-spectrum overlay plot, useful for comparing Mars
    spectra with reference minerals or comparing spectra from different
    locations.
    
    Each spectrum in the list can have individual styling (color,
    linestyle, linewidth) and an optional label for the legend.
    
    Args:
        spectra: List of spectrum dictionaries, each containing:
            - df: DataFrame with 'raman_shift' and 'intensity' columns (required)
            - label: Legend label (optional, default: no label)
            - color: Line color (optional, default: from color cycle)
            - linewidth: Line width (optional, default: 1.2)
            - linestyle: Line style (optional, default: "-")
        xlim: X-axis limits as (min, max) in cm^-1.
            If None, uses range spanning all spectra.
        ylim: Y-axis limits as (min, max) in intensity units.
            If None, auto-scales. Ignored if scale_to_peak is used.
        scale_to_peak: Range (min, max) in cm^-1 for auto-scaling.
            All spectra are normalized to have the same maximum value
            within this range. Useful for comparing peak shapes.
        title: Plot title. If None, no title is added.
        show_grid: Whether to show grid lines. Default is True.
        figsize: Figure size as (width, height) in inches. Default is (12, 6).
        legend_loc: Legend location (matplotlib location string).
            Default is "upper right".
            
    Returns:
        matplotlib Figure object ready for display or saving.
        
    Raises:
        ValueError: If spectra list is empty or dfs missing required columns
        
    Example:
        >>> # Compare two Mars targets
        >>> df1, _ = process_scan_average("0921", "Amherst_Point", "detail_1")
        >>> df2, _ = process_scan_average("0852", "Lake_Haiyaha", "detail_1")
        >>> 
        >>> fig = plot_overlay(
        ...     spectra=[
        ...         {"df": df1, "label": "Amherst Point", "color": "#1f77b4"},
        ...         {"df": df2, "label": "Lake Haiyaha", "color": "#ff7f0e"},
        ...     ],
        ...     xlim=(700, 1200),
        ...     title="Mars Target Comparison"
        ... )
        >>> fig.show()
        
        >>> # Compare Mars spectrum with reference mineral
        >>> mars_df, _ = process_scan_average("0921", "Amherst_Point", "detail_1")
        >>> ref_df = load_reference_spectrum("forsterite")
        >>> 
        >>> fig = plot_overlay(
        ...     spectra=[
        ...         {"df": mars_df, "label": "Mars (Amherst Point)", 
        ...          "color": "#1f77b4"},
        ...         {"df": ref_df, "label": "Forsterite reference", 
        ...          "color": "#2ca02c", "linestyle": "--"},
        ...     ],
        ...     xlim=(700, 1200),
        ...     scale_to_peak=(820, 870),  # Normalize to olivine doublet
        ...     title="Olivine Identification"
        ... )
        >>> fig.savefig("olivine_comparison.png", dpi=300)
        
        >>> # Multiple references with custom styling
        >>> fig = plot_overlay(
        ...     spectra=[
        ...         {"df": mars_df, "label": "Mars", "linewidth": 2.0},
        ...         {"df": forsterite_df, "label": "Forsterite", 
        ...          "linestyle": "--", "color": "#2ca02c"},
        ...         {"df": fayalite_df, "label": "Fayalite", 
        ...          "linestyle": ":", "color": "#d62728"},
        ...     ],
        ...     scale_to_peak=(820, 870),
        ...     legend_loc="upper left"
        ... )
        
    See Also:
        plot_spectrum: For plotting a single spectrum
        load_reference_spectrum: For loading reference spectra
    """
    import numpy as np
    from sherloc_pipeline.visualization.plotting import configure_matplotlib, apply_plot_config
    
    # Validate input
    if not spectra:
        raise ValueError("spectra list cannot be empty")
    
    # Validate each spectrum dict
    required_cols = {"raman_shift", "intensity"}
    for i, spec in enumerate(spectra):
        if "df" not in spec:
            raise ValueError(f"Spectrum {i} missing required 'df' key")
        df = spec["df"]
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Spectrum {i} DataFrame missing required columns: {missing}")
    
    # Default color cycle for unlabeled spectra
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Configure matplotlib with pipeline defaults
    configure_matplotlib()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate scale factors if scale_to_peak is specified
    scale_factors = []
    if scale_to_peak is not None:
        peak_min, peak_max = scale_to_peak
        max_values = []
        for spec in spectra:
            df = spec["df"]
            x = df["raman_shift"].values
            y = df["intensity"].values
            # Find max in the peak range
            mask = (x >= peak_min) & (x <= peak_max)
            if mask.any():
                max_val = np.max(y[mask])
            else:
                max_val = np.max(y)
            max_values.append(max_val)
        
        # Normalize to the first spectrum's max
        reference_max = max_values[0] if max_values[0] != 0 else 1.0
        scale_factors = [reference_max / mv if mv != 0 else 1.0 for mv in max_values]
    else:
        scale_factors = [1.0] * len(spectra)
    
    # Plot each spectrum
    handles = []
    for i, (spec, scale) in enumerate(zip(spectra, scale_factors)):
        df = spec["df"]
        x = df["raman_shift"].values
        y = df["intensity"].values * scale
        
        # Get styling options with defaults
        color = spec.get("color", default_colors[i % len(default_colors)])
        linewidth = spec.get("linewidth", 1.2)
        linestyle = spec.get("linestyle", "-")
        label = spec.get("label", None)
        
        # Plot the line
        line, = ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        
        if label:
            handles.append(line)
    
    # Set axis labels
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (counts)')
    
    # Set title if provided
    if title is not None:
        ax.set_title(title)
    
    # Add grid
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Apply axis limits
    if xlim is not None:
        ax.set_xlim(list(xlim))
        
        # Auto-scale Y based on visible data within xlim (not full dataset)
        if ylim is None:
            visible_y = []
            for spec, scale in zip(spectra, scale_factors):
                df = spec["df"]
                x = df["raman_shift"].values
                y = df["intensity"].values * scale
                xlim_mask = (x >= xlim[0]) & (x <= xlim[1])
                if xlim_mask.any():
                    visible_y.extend(y[xlim_mask])
            if visible_y:
                y_min, y_max = min(visible_y), max(visible_y)
                y_margin = (y_max - y_min) * 0.05
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    if ylim is not None:
        ax.set_ylim(list(ylim))
    
    # Add legend if any spectra have labels
    if handles:
        ax.legend(loc=legend_loc, framealpha=0.85)
    
    # Apply standard plot config
    apply_plot_config(fig)
    
    return fig


__all__ = [
    "process_scan_average",
    "process_subset_average",
    "load_point_spectrum",
    "load_reference_spectrum",
    "plot_spectrum",
    "plot_overlay",
]

