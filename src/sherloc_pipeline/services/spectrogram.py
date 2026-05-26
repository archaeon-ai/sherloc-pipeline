"""
Spectrogram rendering service for PHASE.

This module provides the SpectrogramService for generating spectrogram
visualizations from scan data. A spectrogram is a 2D heatmap showing
spectral intensity as a function of wavenumber (x-axis) and point index (y-axis).

The service supports:
- Generation from lists of Spectrum objects
- Generation from database scan queries
- Configurable normalization and color mapping
- Export to PNG, PDF, or SVG formats
- Difference spectrograms for comparing processing stages

Example:
    >>> from sherloc_pipeline.services.spectrogram import SpectrogramService
    >>> from sherloc_pipeline.models import SpectralRegion, ProcessingLevel
    >>>
    >>> service = SpectrogramService()
    >>> result = service.generate_from_scan(
    ...     scan_id=scan.id,
    ...     region=SpectralRegion.R1,
    ...     processing_level=ProcessingLevel.NORMALIZED,
    ... )
    >>> print(result.summary)

See Also:
    docs/PHASE_SPEC.md Section 5 for spectrogram specification.
    models/spectrogram.py for data models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import uuid

import numpy as np

from sherloc_pipeline.models import (
    SpectralRegion,
    ProcessingLevel,
    Spectrum,
)
from sherloc_pipeline.models.spectrogram import (
    ColorMapType,
    NormalizationType,
    InterpolationMethod,
    SpectrogramConfig,
    SpectrogramData,
    Spectrogram,
    DifferenceSpectrogram,
)
from sherloc_pipeline.services.base import ServiceResult


@dataclass
class SpectrogramRequest:
    """Request for spectrogram generation.

    Attributes:
        scan_id: UUID of the scan to generate spectrogram for
        region: Spectral region (R1, R2, R3, R123)
        processing_level: Processing level of source spectra
        config: Optional visualization configuration
        point_indices: Optional subset of points to include
        output_path: Optional path to save the rendered figure
        output_format: Output format ('png', 'pdf', 'svg')
    """

    scan_id: uuid.UUID
    region: SpectralRegion
    processing_level: ProcessingLevel
    config: Optional[SpectrogramConfig] = None
    point_indices: Optional[List[int]] = None
    output_path: Optional[Path] = None
    output_format: str = "png"


@dataclass
class SpectrogramResult:
    """Result from spectrogram generation.

    Attributes:
        spectrogram: The generated Spectrogram model
        figure_path: Path to saved figure (if output_path was provided)
        warnings: Any warnings during generation
    """

    spectrogram: Spectrogram
    figure_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


class SpectrogramService:
    """Service for generating spectrogram visualizations.

    The SpectrogramService provides methods for creating spectrogram
    2D heatmaps from spectral data. Spectrograms visualize intensity
    as a function of wavenumber and spatial position, enabling:

    - Detection of spatial variation in mineral signatures
    - Identification of outlier spectra
    - Quick overview of spectral features across a scan
    - Comparison of processing stages

    Example:
        >>> service = SpectrogramService()
        >>>
        >>> # Generate from spectra list
        >>> spectrogram = service.generate_from_spectra(
        ...     scan_id=scan.id,
        ...     spectra=spectra_list,
        ...     region=SpectralRegion.R1,
        ...     processing_level=ProcessingLevel.NORMALIZED,
        ... )
        >>>
        >>> # Render to file
        >>> result = service.render(spectrogram, Path("spectrogram.png"))
    """

    def __init__(self, default_config: Optional[SpectrogramConfig] = None):
        """Initialize the spectrogram service.

        Args:
            default_config: Default visualization configuration. If None,
                uses SpectrogramConfig() defaults.
        """
        self._default_config = default_config or SpectrogramConfig()

    def generate_from_spectra(
        self,
        scan_id: uuid.UUID,
        spectra: List[Spectrum],
        region: SpectralRegion,
        processing_level: ProcessingLevel,
        config: Optional[SpectrogramConfig] = None,
        wavenumbers: Optional[List[float]] = None,
    ) -> Spectrogram:
        """Generate a spectrogram from a list of Spectrum objects.

        Args:
            scan_id: UUID of the source scan
            spectra: List of Spectrum objects (should be same region/level)
            region: Spectral region
            processing_level: Processing level
            config: Visualization configuration (uses default if None)
            wavenumbers: Optional wavenumber array (uses first spectrum's if None)

        Returns:
            Spectrogram model instance

        Raises:
            ValueError: If spectra list is empty or spectra have mismatched lengths
        """
        if not spectra:
            raise ValueError("At least one spectrum is required")

        # Validate that all spectra have the same length
        first_len = len(spectra[0].intensity_values)
        for i, spectrum in enumerate(spectra[1:], 1):
            if len(spectrum.intensity_values) != first_len:
                raise ValueError(
                    f"Spectrum {i} has {len(spectrum.intensity_values)} channels, "
                    f"expected {first_len}"
                )

        # Stack intensities into matrix
        intensities = []
        for spectrum in spectra:
            intensities.append(spectrum.intensity_values)

        matrix = np.array(intensities, dtype=np.float32)
        n_points, n_channels = matrix.shape

        # Get wavenumber array
        if wavenumbers is not None:
            wn = np.array(wavenumbers, dtype=np.float32)
        else:
            # Try to get from first spectrum
            wn_values = spectra[0].wavenumber_values
            if wn_values is not None:
                wn = np.array(wn_values, dtype=np.float32)
            else:
                # Default to channel indices
                wn = np.arange(n_channels, dtype=np.float32)

        # Generate point labels
        point_labels = [f"p{i}" for i in range(n_points)]

        # Create data container
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(matrix),
            n_points=n_points,
            n_channels=n_channels,
            wavenumber_min=float(wn.min()),
            wavenumber_max=float(wn.max()),
            wavenumbers=SpectrogramData.compress_array(wn.tolist()),
            point_labels=point_labels,
            intensity_min=float(matrix.min()),
            intensity_max=float(matrix.max()),
        )

        # Create spectrogram
        return Spectrogram(
            scan_id=scan_id,
            region=region,
            processing_level=processing_level,
            config=config or self._default_config,
            data=data,
            point_indices=list(range(n_points)),
        )

    def generate_from_matrix(
        self,
        scan_id: uuid.UUID,
        matrix: np.ndarray,
        region: SpectralRegion,
        processing_level: ProcessingLevel,
        wavenumber_min: float,
        wavenumber_max: float,
        config: Optional[SpectrogramConfig] = None,
        wavenumbers: Optional[np.ndarray] = None,
        point_labels: Optional[List[str]] = None,
    ) -> Spectrogram:
        """Generate a spectrogram from a numpy array.

        Args:
            scan_id: UUID of the source scan
            matrix: 2D numpy array of shape (n_points, n_channels)
            region: Spectral region
            processing_level: Processing level
            wavenumber_min: Minimum wavenumber value
            wavenumber_max: Maximum wavenumber value
            config: Visualization configuration (uses default if None)
            wavenumbers: Optional explicit wavenumber array
            point_labels: Optional labels for each point

        Returns:
            Spectrogram model instance
        """
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {matrix.ndim}D")

        n_points, n_channels = matrix.shape

        # Ensure float32
        matrix = matrix.astype(np.float32)

        # Handle wavenumbers
        wn_compressed = None
        if wavenumbers is not None:
            wn_compressed = SpectrogramData.compress_array(wavenumbers.tolist())

        # Generate default point labels
        if point_labels is None:
            point_labels = [f"p{i}" for i in range(n_points)]

        # Create data container
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(matrix),
            n_points=n_points,
            n_channels=n_channels,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
            wavenumbers=wn_compressed,
            point_labels=point_labels,
            intensity_min=float(matrix.min()),
            intensity_max=float(matrix.max()),
        )

        return Spectrogram(
            scan_id=scan_id,
            region=region,
            processing_level=processing_level,
            config=config or self._default_config,
            data=data,
            point_indices=list(range(n_points)),
        )

    def render(
        self,
        spectrogram: Spectrogram,
        output_path: Optional[Path] = None,
        target: Optional[str] = None,
        sol: Optional[int] = None,
        dpi: Optional[int] = None,
    ) -> Tuple[object, Optional[Path]]:
        """Render a spectrogram to a matplotlib figure.

        Args:
            spectrogram: Spectrogram to render
            output_path: Optional path to save the figure
            target: Target name for title rendering
            sol: Sol number for title rendering
            dpi: Override DPI for saving (uses config.dpi if None)

        Returns:
            Tuple of (matplotlib Figure, saved path or None)

        Note:
            Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt
        from sherloc_pipeline.visualization.plotting import (
            configure_matplotlib,
            apply_plot_config,
        )

        # Configure matplotlib
        configure_matplotlib()

        config = spectrogram.config

        # Create figure
        fig, ax = plt.subplots(figsize=config.figure_size)

        # Get normalized matrix
        matrix = spectrogram.get_normalized_matrix()
        extent = spectrogram.data.get_extent()

        # Handle interpolation mapping (config values may be strings due to use_enum_values)
        interpolation = (
            config.interpolation.value
            if hasattr(config.interpolation, "value")
            else config.interpolation
        )
        if interpolation == "none":
            interpolation = "nearest"

        # Get colormap value
        colormap = (
            config.colormap.value
            if hasattr(config.colormap, "value")
            else config.colormap
        )

        # Render heatmap
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap=colormap,
            extent=extent,
            interpolation=interpolation,
            origin="upper",
        )

        # Set labels
        ax.set_xlabel(config.x_label)
        ax.set_ylabel(config.y_label)

        # Set title
        title = spectrogram.render_title(target=target, sol=sol)
        if title:
            ax.set_title(title)

        # Add colorbar
        if config.show_colorbar:
            cbar = fig.colorbar(im, ax=ax, label=config.colorbar_label)

        # Highlight specific points if requested
        if config.highlight_points:
            for point_idx in config.highlight_points:
                if 0 <= point_idx < spectrogram.data.n_points:
                    ax.axhline(
                        y=point_idx,
                        color="white",
                        linestyle="--",
                        linewidth=0.5,
                        alpha=0.7,
                    )

        # Apply plot config and get bbox
        plot_config, bbox_inches = apply_plot_config(fig)

        # Save if path provided
        saved_path = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_dpi = dpi if dpi is not None else config.dpi
            fig.savefig(output_path, dpi=save_dpi, bbox_inches=bbox_inches)
            saved_path = output_path

        return fig, saved_path

    def create_difference_spectrogram(
        self,
        spectrogram_a: Spectrogram,
        spectrogram_b: Spectrogram,
        operation: str = "subtract",
        config: Optional[SpectrogramConfig] = None,
    ) -> DifferenceSpectrogram:
        """Create a difference spectrogram from two spectrograms.

        Args:
            spectrogram_a: First (minuend) spectrogram
            spectrogram_b: Second (subtrahend) spectrogram
            operation: Operation type ('subtract', 'ratio', 'log_ratio')
            config: Optional configuration (defaults to diverging colormap)

        Returns:
            DifferenceSpectrogram model instance

        Raises:
            ValueError: If spectrograms have incompatible dimensions
        """
        # Validate dimensions
        if spectrogram_a.data.n_points != spectrogram_b.data.n_points:
            raise ValueError(
                f"Point count mismatch: {spectrogram_a.data.n_points} vs "
                f"{spectrogram_b.data.n_points}"
            )
        if spectrogram_a.data.n_channels != spectrogram_b.data.n_channels:
            raise ValueError(
                f"Channel count mismatch: {spectrogram_a.data.n_channels} vs "
                f"{spectrogram_b.data.n_channels}"
            )

        # Get matrices
        matrix_a = spectrogram_a.data.get_intensity_matrix()
        matrix_b = spectrogram_b.data.get_intensity_matrix()

        # Compute difference based on operation
        if operation == "subtract":
            diff_matrix = matrix_a - matrix_b
        elif operation == "ratio":
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                diff_matrix = matrix_a / np.where(matrix_b != 0, matrix_b, np.nan)
                diff_matrix = np.nan_to_num(diff_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        elif operation == "log_ratio":
            # Log ratio (log(a) - log(b))
            with np.errstate(divide="ignore", invalid="ignore"):
                log_a = np.log(np.where(matrix_a > 0, matrix_a, np.nan))
                log_b = np.log(np.where(matrix_b > 0, matrix_b, np.nan))
                diff_matrix = log_a - log_b
                diff_matrix = np.nan_to_num(diff_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        diff_matrix = diff_matrix.astype(np.float32)

        # Create data container
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(diff_matrix),
            n_points=spectrogram_a.data.n_points,
            n_channels=spectrogram_a.data.n_channels,
            wavenumber_min=spectrogram_a.data.wavenumber_min,
            wavenumber_max=spectrogram_a.data.wavenumber_max,
            wavenumbers=spectrogram_a.data.wavenumbers,
            point_labels=spectrogram_a.data.point_labels,
            intensity_min=float(diff_matrix.min()),
            intensity_max=float(diff_matrix.max()),
        )

        # Default config for difference uses diverging colormap
        if config is None:
            config = SpectrogramConfig(
                colormap=ColorMapType.COOLWARM,
                normalization=NormalizationType.ZSCORE,
            )

        return DifferenceSpectrogram(
            spectrogram_a_id=spectrogram_a.id,
            spectrogram_b_id=spectrogram_b.id,
            operation=operation,
            config=config,
            data=data,
        )

    def to_service_result(
        self,
        spectrogram: Spectrogram,
        output_path: Optional[Path] = None,
        warnings: Optional[List[str]] = None,
    ) -> ServiceResult:
        """Convert a spectrogram generation result to ServiceResult.

        Args:
            spectrogram: Generated spectrogram
            output_path: Path where figure was saved (if any)
            warnings: Any warnings generated

        Returns:
            ServiceResult for CLI consumption
        """
        # Handle enum values (may be strings due to use_enum_values config)
        region = (
            spectrogram.region.value
            if hasattr(spectrogram.region, "value")
            else spectrogram.region
        )
        processing_level = (
            spectrogram.processing_level.value
            if hasattr(spectrogram.processing_level, "value")
            else spectrogram.processing_level
        )

        artifacts = [output_path] if output_path else []
        return ServiceResult(
            summary=(
                f"Generated spectrogram for {region} "
                f"({spectrogram.data.n_points} points x "
                f"{spectrogram.data.n_channels} channels)"
            ),
            artifacts=artifacts,
            warnings=warnings or [],
            metadata={
                "spectrogram_id": str(spectrogram.id),
                "scan_id": str(spectrogram.scan_id),
                "region": region,
                "processing_level": processing_level,
                "n_points": spectrogram.data.n_points,
                "n_channels": spectrogram.data.n_channels,
                "wavenumber_min": spectrogram.data.wavenumber_min,
                "wavenumber_max": spectrogram.data.wavenumber_max,
            },
        )
