"""
Spectrogram data models for PHASE.

This module defines models for spectrogram visualization of SHERLOC spectral data.
A spectrogram is a 2D representation showing spectral intensity as a function of
wavenumber/wavelength (x-axis) and spatial position/point index (y-axis).

The spectrogram provides a powerful way to visualize patterns across multiple
measurement points in a scan, enabling:
- Detection of spatial variation in mineral signatures
- Identification of outlier spectra (cosmic rays, anomalies)
- Quick overview of spectral features across an entire scan
- Comparison of spectral intensity distributions

Classes:
    ColorMapType: Supported matplotlib color maps for visualization
    AxisScale: Linear or logarithmic scaling options
    NormalizationType: Per-spectrum or global normalization modes
    SpectrogramConfig: Configuration for spectrogram rendering
    SpectrogramData: Container for spectrogram matrix and metadata
    Spectrogram: Complete spectrogram with configuration and data

Example:
    >>> from sherloc_pipeline.models.spectrogram import (
    ...     Spectrogram, SpectrogramConfig, SpectrogramData, ColorMapType
    ... )
    >>> import numpy as np
    >>>
    >>> # Create configuration
    >>> config = SpectrogramConfig(
    ...     colormap=ColorMapType.VIRIDIS,
    ...     x_label="Raman Shift (cm^-1)",
    ...     y_label="Point Index",
    ... )
    >>>
    >>> # Create spectrogram data
    >>> data = SpectrogramData(
    ...     intensity_matrix=matrix_bytes,  # compressed 2D array
    ...     n_points=100,
    ...     n_channels=501,
    ...     wavenumber_min=200.0,
    ...     wavenumber_max=4000.0,
    ... )
    >>>
    >>> # Create complete spectrogram
    >>> spectrogram = Spectrogram(
    ...     scan_id=scan.id,
    ...     region=SpectralRegion.R1,
    ...     config=config,
    ...     data=data,
    ... )

See Also:
    docs/PHASE_SPEC.md Section 4 for spectrogram feature design.
"""

from enum import Enum
from typing import Optional, List, Tuple
import uuid
import zlib

from pydantic import Field, model_validator, field_validator, field_serializer
import base64

from sherloc_pipeline.models.base import (
    PHASEBaseModel,
    IdentifiableModel,
    ModelRegistry,
)
from sherloc_pipeline.models.spectra import SpectralRegion, ProcessingLevel


class ColorMapType(str, Enum):
    """Supported matplotlib color maps for spectrogram visualization.

    These color maps are chosen for scientific visualization best practices:
    - Perceptually uniform where possible
    - Accessible for color vision deficiency
    - Good contrast for spectral data

    Sequential color maps (low-to-high intensity):
        VIRIDIS: Default, perceptually uniform, accessible
        PLASMA: Warm colors, good for emphasizing peaks
        MAGMA: Dark background, good for publications
        INFERNO: High contrast, dark background
        CIVIDIS: Optimized for color vision deficiency

    Diverging color maps (for difference/ratio spectrograms):
        COOLWARM: Red-blue diverging
        SEISMIC: Red-white-blue, centered on zero
        RD_BU: Red-blue with neutral midpoint

    Traditional (for compatibility):
        JET: Rainbow (not recommended but familiar)
        HOT: Hot metal colors
        BONE: Grayscale with blue tint
        GRAY: Pure grayscale
    """
    # Perceptually uniform (recommended)
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"

    # Diverging (for difference spectrograms)
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"
    RD_BU = "RdBu"

    # Traditional
    JET = "jet"
    HOT = "hot"
    BONE = "bone"
    GRAY = "gray"


class AxisScale(str, Enum):
    """Scaling type for spectrogram axes.

    LINEAR: Standard linear scaling
    LOG: Logarithmic scaling (useful for wide dynamic range)
    SYMLOG: Symmetric log (handles negative values)
    """
    LINEAR = "linear"
    LOG = "log"
    SYMLOG = "symlog"


class NormalizationType(str, Enum):
    """Normalization mode for spectrogram intensity.

    NONE: Raw intensity values, no normalization
    GLOBAL: Normalize to global min/max across all spectra
    PER_SPECTRUM: Normalize each spectrum independently (0-1)
    PERCENTILE: Clip to percentile range (e.g., 1st-99th)
    ZSCORE: Standard score normalization per spectrum
    """
    NONE = "none"
    GLOBAL = "global"
    PER_SPECTRUM = "per_spectrum"
    PERCENTILE = "percentile"
    ZSCORE = "zscore"


class InterpolationMethod(str, Enum):
    """Interpolation method for spectrogram rendering.

    Controls how the 2D intensity matrix is rendered when the display
    resolution differs from the data resolution.

    NONE: No interpolation, nearest neighbor
    BILINEAR: Smooth bilinear interpolation
    BICUBIC: Smooth bicubic interpolation
    HANNING: Hanning windowed interpolation
    """
    NONE = "none"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    HANNING = "hanning"


class SpectrogramConfig(PHASEBaseModel):
    """Configuration for spectrogram visualization.

    Defines rendering parameters for creating spectrogram plots. This
    configuration is typically stored with the spectrogram and can be
    modified for different visualization contexts.

    Attributes:
        colormap: Color map for intensity mapping
        intensity_scale: Linear or log scaling for intensity
        normalization: How to normalize intensity values
        percentile_low: Lower percentile for clipping (if using percentile norm)
        percentile_high: Upper percentile for clipping (if using percentile norm)
        interpolation: Interpolation method for rendering
        x_label: Label for x-axis (wavenumber/wavelength)
        y_label: Label for y-axis (point index/distance)
        title_template: Template for plot title (can include {target}, {sol}, etc.)
        show_colorbar: Whether to display the colorbar
        colorbar_label: Label for the colorbar
        aspect_ratio: Aspect ratio for the plot (auto, equal, or numeric)
        figure_size: Figure size in inches (width, height)
        dpi: Resolution for saved figures

    Example:
        >>> config = SpectrogramConfig(
        ...     colormap=ColorMapType.PLASMA,
        ...     normalization=NormalizationType.PERCENTILE,
        ...     percentile_low=1.0,
        ...     percentile_high=99.0,
        ...     title_template="{target} Sol {sol} Spectrogram",
        ... )
    """

    colormap: ColorMapType = Field(
        default=ColorMapType.VIRIDIS,
        description="Color map for intensity visualization"
    )
    intensity_scale: AxisScale = Field(
        default=AxisScale.LINEAR,
        description="Scaling for intensity values"
    )
    normalization: NormalizationType = Field(
        default=NormalizationType.PERCENTILE,
        description="Normalization mode for intensity"
    )
    percentile_low: float = Field(
        default=1.0,
        ge=0.0,
        le=50.0,
        description="Lower percentile for clipping (if using percentile normalization)"
    )
    percentile_high: float = Field(
        default=99.0,
        ge=50.0,
        le=100.0,
        description="Upper percentile for clipping (if using percentile normalization)"
    )
    interpolation: InterpolationMethod = Field(
        default=InterpolationMethod.NONE,
        description="Interpolation method for rendering"
    )

    # Axis labels and title
    x_label: str = Field(
        default="Raman Shift (cm$^{-1}$)",
        description="Label for x-axis"
    )
    y_label: str = Field(
        default="Point Index",
        description="Label for y-axis"
    )
    title_template: Optional[str] = Field(
        default="{target} Sol {sol} - {region} Spectrogram",
        description="Template for plot title"
    )

    # Colorbar settings
    show_colorbar: bool = Field(
        default=True,
        description="Whether to display colorbar"
    )
    colorbar_label: str = Field(
        default="Intensity (a.u.)",
        description="Label for colorbar"
    )

    # Figure settings
    aspect_ratio: str = Field(
        default="auto",
        description="Aspect ratio: 'auto', 'equal', or numeric value"
    )
    figure_size: Tuple[float, float] = Field(
        default=(12.0, 8.0),
        description="Figure size in inches (width, height)"
    )
    dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="Resolution for saved figures"
    )

    # Overlay settings
    show_peak_markers: bool = Field(
        default=False,
        description="Overlay peak position markers"
    )
    show_mineral_bands: bool = Field(
        default=False,
        description="Overlay mineral identification bands"
    )
    highlight_points: Optional[List[int]] = Field(
        default=None,
        description="List of point indices to highlight"
    )

    @model_validator(mode="after")
    def validate_percentile_order(self) -> "SpectrogramConfig":
        """Ensure percentile_low < percentile_high."""
        if self.percentile_low >= self.percentile_high:
            raise ValueError(
                f"percentile_low ({self.percentile_low}) must be less than "
                f"percentile_high ({self.percentile_high})"
            )
        return self


class SpectrogramData(PHASEBaseModel):
    """Container for spectrogram intensity matrix and axes.

    Stores the 2D intensity array and associated axis information for
    a spectrogram. The intensity matrix is stored as compressed bytes
    for efficient storage and transmission.

    The matrix dimensions are [n_points, n_channels]:
    - Rows: spatial points (0 to n_points-1)
    - Columns: spectral channels (wavenumber or wavelength bins)

    Attributes:
        intensity_matrix: Compressed 2D float32 array (n_points x n_channels)
        n_points: Number of spatial points (rows)
        n_channels: Number of spectral channels (columns)
        wavenumber_min: Minimum wavenumber/wavelength value
        wavenumber_max: Maximum wavenumber/wavelength value
        wavenumbers: Optional explicit wavenumber array (if non-uniform)
        point_labels: Optional labels for each point (e.g., "p0", "p1", ...)
        intensity_min: Minimum intensity in the matrix
        intensity_max: Maximum intensity in the matrix

    Example:
        >>> import numpy as np
        >>> matrix = np.random.rand(100, 501).astype(np.float32)
        >>> compressed = zlib.compress(matrix.tobytes())
        >>> data = SpectrogramData(
        ...     intensity_matrix=compressed,
        ...     n_points=100,
        ...     n_channels=501,
        ...     wavenumber_min=200.0,
        ...     wavenumber_max=4000.0,
        ... )
    """

    intensity_matrix: bytes = Field(
        description="Compressed 2D float32 intensity array (n_points x n_channels)"
    )
    n_points: int = Field(
        gt=0,
        description="Number of spatial points (rows)"
    )
    n_channels: int = Field(
        gt=0,
        description="Number of spectral channels (columns)"
    )

    # X-axis (spectral) bounds
    wavenumber_min: float = Field(
        description="Minimum wavenumber/wavelength value"
    )
    wavenumber_max: float = Field(
        description="Maximum wavenumber/wavelength value"
    )
    wavenumbers: Optional[bytes] = Field(
        default=None,
        description="Compressed wavenumber array (if non-uniform spacing)"
    )

    # Y-axis (spatial) metadata
    point_labels: Optional[List[str]] = Field(
        default=None,
        description="Optional labels for each point"
    )

    # Intensity statistics
    intensity_min: Optional[float] = Field(
        default=None,
        description="Minimum intensity in matrix"
    )
    intensity_max: Optional[float] = Field(
        default=None,
        description="Maximum intensity in matrix"
    )

    @staticmethod
    def compress_matrix(matrix: "np.ndarray") -> bytes:
        """Compress a 2D numpy array to binary storage format.

        Args:
            matrix: 2D numpy array of float intensities

        Returns:
            Compressed bytes suitable for storage
        """
        import numpy as np
        arr = matrix.astype(np.float32)
        return zlib.compress(arr.tobytes())

    @staticmethod
    def decompress_matrix(data: bytes, shape: Tuple[int, int]) -> "np.ndarray":
        """Decompress binary data to a 2D numpy array.

        Args:
            data: Compressed bytes
            shape: Expected shape (n_points, n_channels)

        Returns:
            2D numpy array of float32 intensities
        """
        import numpy as np
        arr = np.frombuffer(zlib.decompress(data), dtype=np.float32)
        return arr.reshape(shape)

    @staticmethod
    def compress_array(values: List[float]) -> bytes:
        """Compress a 1D array to binary format."""
        import numpy as np
        arr = np.array(values, dtype=np.float32)
        return zlib.compress(arr.tobytes())

    @staticmethod
    def decompress_array(data: bytes) -> List[float]:
        """Decompress binary data to a list of floats."""
        import numpy as np
        arr = np.frombuffer(zlib.decompress(data), dtype=np.float32)
        return arr.tolist()

    def get_intensity_matrix(self) -> "np.ndarray":
        """Get the intensity matrix as a 2D numpy array.

        Returns:
            2D numpy array with shape (n_points, n_channels)
        """
        return self.decompress_matrix(
            self.intensity_matrix,
            (self.n_points, self.n_channels)
        )

    def get_wavenumbers(self) -> "np.ndarray":
        """Get wavenumber array for x-axis.

        If explicit wavenumbers are stored, returns those. Otherwise,
        generates a linearly spaced array from min to max.

        Returns:
            1D numpy array of wavenumber values
        """
        import numpy as np
        if self.wavenumbers is not None:
            return np.array(self.decompress_array(self.wavenumbers))
        return np.linspace(
            self.wavenumber_min,
            self.wavenumber_max,
            self.n_channels
        )

    def get_extent(self) -> Tuple[float, float, float, float]:
        """Get extent for matplotlib imshow.

        Returns:
            Tuple of (left, right, bottom, top) for imshow extent
        """
        return (
            self.wavenumber_min,
            self.wavenumber_max,
            self.n_points - 0.5,
            -0.5,
        )

    @classmethod
    def from_spectra(
        cls,
        spectra: List["Spectrum"],
        wavenumbers: Optional[List[float]] = None,
    ) -> "SpectrogramData":
        """Create SpectrogramData from a list of Spectrum objects.

        Args:
            spectra: List of Spectrum objects (should be same processing level)
            wavenumbers: Optional wavenumber array (uses first spectrum's if not provided)

        Returns:
            SpectrogramData instance with stacked intensity matrix
        """
        import numpy as np

        if not spectra:
            raise ValueError("At least one spectrum required")

        # Stack intensities into matrix
        intensities = []
        for spectrum in spectra:
            intensities.append(spectrum.intensity_values)

        matrix = np.array(intensities, dtype=np.float32)
        n_points, n_channels = matrix.shape

        # Get wavenumber bounds
        if wavenumbers is not None:
            wn = np.array(wavenumbers)
        else:
            # Try to get from first spectrum
            wn_values = spectra[0].wavenumber_values
            if wn_values is not None:
                wn = np.array(wn_values)
            else:
                # Default to channel indices
                wn = np.arange(n_channels).astype(float)

        # Generate point labels
        point_labels = [f"p{i}" for i in range(n_points)]

        compressed = cls.compress_matrix(matrix)
        wn_compressed = cls.compress_array(wn.tolist()) if len(wn) > 0 else None

        return cls(
            intensity_matrix=compressed,
            n_points=n_points,
            n_channels=n_channels,
            wavenumber_min=float(wn.min()) if len(wn) > 0 else 0.0,
            wavenumber_max=float(wn.max()) if len(wn) > 0 else float(n_channels - 1),
            wavenumbers=wn_compressed,
            point_labels=point_labels,
            intensity_min=float(matrix.min()),
            intensity_max=float(matrix.max()),
        )

    @model_validator(mode="after")
    def validate_wavenumber_order(self) -> "SpectrogramData":
        """Ensure wavenumber_min <= wavenumber_max."""
        if self.wavenumber_min > self.wavenumber_max:
            raise ValueError(
                f"wavenumber_min ({self.wavenumber_min}) must be <= "
                f"wavenumber_max ({self.wavenumber_max})"
            )
        return self

    @field_serializer("intensity_matrix", "wavenumbers")
    def serialize_bytes(self, value: Optional[bytes]) -> Optional[str]:
        """Serialize bytes fields to base64 for JSON compatibility."""
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")


@ModelRegistry.register
class Spectrogram(IdentifiableModel):
    """Complete spectrogram with configuration and data.

    A Spectrogram represents a 2D visualization of spectral data across
    multiple measurement points within a scan. It combines:
    - Reference to the source scan
    - Spectral region and processing level
    - Rendering configuration
    - The actual intensity matrix data

    Spectrograms enable:
    - Visual identification of spatial variation in spectra
    - Detection of outliers and anomalies
    - Overview of mineral distribution across a scan
    - Comparison between different processing levels

    Attributes:
        scan_id: UUID of the source Scan
        region: Spectral region (R1, R2, R3, or R123)
        processing_level: Processing level of source spectra
        config: Visualization configuration
        data: Intensity matrix and axis data
        point_indices: Indices of points included (if subset)
        title: Optional display title

    Example:
        >>> spectrogram = Spectrogram(
        ...     scan_id=scan.id,
        ...     region=SpectralRegion.R1,
        ...     processing_level=ProcessingLevel.NORMALIZED,
        ...     config=SpectrogramConfig(colormap=ColorMapType.VIRIDIS),
        ...     data=data,
        ... )
    """

    scan_id: uuid.UUID = Field(
        description="UUID of the source Scan"
    )
    region: SpectralRegion = Field(
        description="Spectral region (R1, R2, R3, or R123)"
    )
    processing_level: ProcessingLevel = Field(
        description="Processing level of source spectra"
    )
    config: SpectrogramConfig = Field(
        default_factory=SpectrogramConfig,
        description="Visualization configuration"
    )
    data: SpectrogramData = Field(
        description="Intensity matrix and axis metadata"
    )
    point_indices: Optional[List[int]] = Field(
        default=None,
        description="Point indices included (if subset of scan)"
    )
    title: Optional[str] = Field(
        default=None,
        description="Display title (overrides config template)"
    )

    def render_title(
        self,
        target: Optional[str] = None,
        sol: Optional[int] = None,
    ) -> str:
        """Render the display title using template and context.

        Args:
            target: Target name for template substitution
            sol: Sol number for template substitution

        Returns:
            Rendered title string
        """
        if self.title is not None:
            return self.title

        # Handle both enum and string values (due to use_enum_values config)
        region_str = self.region.value if hasattr(self.region, 'value') else self.region
        processing_str = (
            self.processing_level.value
            if hasattr(self.processing_level, 'value')
            else self.processing_level
        )

        if self.config.title_template is None:
            return f"{region_str} Spectrogram"

        return self.config.title_template.format(
            target=target or "Unknown",
            sol=sol or "?",
            region=region_str,
            processing=processing_str,
        )

    def get_normalized_matrix(self) -> "np.ndarray":
        """Get intensity matrix with normalization applied.

        Applies the normalization specified in config to the raw
        intensity matrix.

        Returns:
            Normalized 2D numpy array
        """
        import numpy as np

        matrix = self.data.get_intensity_matrix()
        norm_type = self.config.normalization

        if norm_type == NormalizationType.NONE:
            return matrix

        elif norm_type == NormalizationType.GLOBAL:
            vmin, vmax = matrix.min(), matrix.max()
            if vmax - vmin > 0:
                return (matrix - vmin) / (vmax - vmin)
            return matrix

        elif norm_type == NormalizationType.PER_SPECTRUM:
            result = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                row = matrix[i]
                vmin, vmax = row.min(), row.max()
                if vmax - vmin > 0:
                    result[i] = (row - vmin) / (vmax - vmin)
                else:
                    result[i] = row
            return result

        elif norm_type == NormalizationType.PERCENTILE:
            vmin = np.percentile(matrix, self.config.percentile_low)
            vmax = np.percentile(matrix, self.config.percentile_high)
            matrix = np.clip(matrix, vmin, vmax)
            if vmax - vmin > 0:
                return (matrix - vmin) / (vmax - vmin)
            return matrix

        elif norm_type == NormalizationType.ZSCORE:
            result = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                row = matrix[i]
                mean, std = row.mean(), row.std()
                if std > 0:
                    result[i] = (row - mean) / std
                else:
                    result[i] = row - mean
            return result

        return matrix

    @classmethod
    def from_scan_spectra(
        cls,
        scan_id: uuid.UUID,
        spectra: List["Spectrum"],
        region: SpectralRegion,
        processing_level: ProcessingLevel,
        config: Optional[SpectrogramConfig] = None,
        wavenumbers: Optional[List[float]] = None,
    ) -> "Spectrogram":
        """Create a Spectrogram from a list of spectra.

        Convenience constructor that builds a spectrogram from a list
        of Spectrum objects belonging to the same scan.

        Args:
            scan_id: UUID of the source scan
            spectra: List of Spectrum objects
            region: Spectral region
            processing_level: Processing level
            config: Optional visualization config
            wavenumbers: Optional wavenumber array

        Returns:
            New Spectrogram instance
        """
        data = SpectrogramData.from_spectra(spectra, wavenumbers)
        point_indices = [i for i in range(len(spectra))]

        return cls(
            scan_id=scan_id,
            region=region,
            processing_level=processing_level,
            config=config or SpectrogramConfig(),
            data=data,
            point_indices=point_indices,
        )


class DifferenceSpectrogram(IdentifiableModel):
    """Spectrogram showing difference between two spectrograms.

    Useful for visualizing:
    - Effect of processing steps (before/after baseline removal)
    - Spatial variation relative to a reference
    - Temporal changes in repeated scans

    Attributes:
        spectrogram_a_id: UUID of the first (minuend) spectrogram
        spectrogram_b_id: UUID of the second (subtrahend) spectrogram
        operation: Difference operation type
        config: Visualization config (should use diverging colormap)
        data: Difference intensity matrix

    Example:
        >>> diff = DifferenceSpectrogram(
        ...     spectrogram_a_id=after_baseline.id,
        ...     spectrogram_b_id=before_baseline.id,
        ...     operation="subtract",
        ...     config=SpectrogramConfig(colormap=ColorMapType.COOLWARM),
        ...     data=diff_data,
        ... )
    """

    spectrogram_a_id: uuid.UUID = Field(
        description="UUID of first spectrogram (minuend)"
    )
    spectrogram_b_id: uuid.UUID = Field(
        description="UUID of second spectrogram (subtrahend)"
    )
    operation: str = Field(
        default="subtract",
        description="Operation: 'subtract', 'ratio', 'log_ratio'"
    )
    config: SpectrogramConfig = Field(
        default_factory=lambda: SpectrogramConfig(
            colormap=ColorMapType.COOLWARM,
            normalization=NormalizationType.ZSCORE,
        ),
        description="Visualization config (diverging colormap recommended)"
    )
    data: SpectrogramData = Field(
        description="Difference/ratio intensity matrix"
    )

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {"subtract", "ratio", "log_ratio"}
        if v not in valid_ops:
            raise ValueError(f"operation must be one of {valid_ops}")
        return v
