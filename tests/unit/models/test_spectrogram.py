"""
Unit tests for PHASE spectrogram models (bd-3v5: WS4-A).

Tests the spectrogram visualization models defined in models/spectrogram.py:
- ColorMapType: Supported matplotlib color maps
- AxisScale: Linear/log scaling options
- NormalizationType: Intensity normalization modes
- InterpolationMethod: Rendering interpolation
- SpectrogramConfig: Visualization configuration
- SpectrogramData: Intensity matrix container
- Spectrogram: Complete spectrogram model
- DifferenceSpectrogram: Difference/ratio spectrograms
"""

import uuid
import zlib

import numpy as np
import pytest
from pydantic import ValidationError

from sherloc_pipeline.models import SpectralRegion, ProcessingLevel
from sherloc_pipeline.models.spectrogram import (
    ColorMapType,
    AxisScale,
    NormalizationType,
    InterpolationMethod,
    SpectrogramConfig,
    SpectrogramData,
    Spectrogram,
    DifferenceSpectrogram,
)


class TestColorMapType:
    """Tests for ColorMapType enum."""

    def test_perceptually_uniform_colormaps(self):
        """Perceptually uniform colormaps are available."""
        assert ColorMapType.VIRIDIS.value == "viridis"
        assert ColorMapType.PLASMA.value == "plasma"
        assert ColorMapType.MAGMA.value == "magma"
        assert ColorMapType.INFERNO.value == "inferno"
        assert ColorMapType.CIVIDIS.value == "cividis"

    def test_diverging_colormaps(self):
        """Diverging colormaps are available."""
        assert ColorMapType.COOLWARM.value == "coolwarm"
        assert ColorMapType.SEISMIC.value == "seismic"
        assert ColorMapType.RD_BU.value == "RdBu"

    def test_traditional_colormaps(self):
        """Traditional colormaps are available."""
        assert ColorMapType.JET.value == "jet"
        assert ColorMapType.HOT.value == "hot"
        assert ColorMapType.BONE.value == "bone"
        assert ColorMapType.GRAY.value == "gray"

    def test_colormap_count(self):
        """All expected colormaps are defined."""
        assert len(ColorMapType) == 12


class TestAxisScale:
    """Tests for AxisScale enum."""

    def test_scale_values(self):
        """All scale types have correct values."""
        assert AxisScale.LINEAR.value == "linear"
        assert AxisScale.LOG.value == "log"
        assert AxisScale.SYMLOG.value == "symlog"


class TestNormalizationType:
    """Tests for NormalizationType enum."""

    def test_normalization_values(self):
        """All normalization types have correct values."""
        assert NormalizationType.NONE.value == "none"
        assert NormalizationType.GLOBAL.value == "global"
        assert NormalizationType.PER_SPECTRUM.value == "per_spectrum"
        assert NormalizationType.PERCENTILE.value == "percentile"
        assert NormalizationType.ZSCORE.value == "zscore"


class TestInterpolationMethod:
    """Tests for InterpolationMethod enum."""

    def test_interpolation_values(self):
        """All interpolation methods have correct values."""
        assert InterpolationMethod.NONE.value == "none"
        assert InterpolationMethod.BILINEAR.value == "bilinear"
        assert InterpolationMethod.BICUBIC.value == "bicubic"
        assert InterpolationMethod.HANNING.value == "hanning"


class TestSpectrogramConfig:
    """Tests for SpectrogramConfig model."""

    def test_default_config(self):
        """Default configuration values are sensible."""
        config = SpectrogramConfig()

        assert config.colormap == ColorMapType.VIRIDIS
        assert config.intensity_scale == AxisScale.LINEAR
        assert config.normalization == NormalizationType.PERCENTILE
        assert config.percentile_low == 1.0
        assert config.percentile_high == 99.0
        assert config.interpolation == InterpolationMethod.NONE
        assert config.show_colorbar is True
        assert config.figure_size == (12.0, 8.0)
        assert config.dpi == 150

    def test_custom_config(self):
        """Custom configuration values are applied."""
        config = SpectrogramConfig(
            colormap=ColorMapType.PLASMA,
            normalization=NormalizationType.GLOBAL,
            figure_size=(10.0, 6.0),
            dpi=300,
            x_label="Wavelength (nm)",
            y_label="Spectrum Index",
        )

        assert config.colormap == ColorMapType.PLASMA
        assert config.normalization == NormalizationType.GLOBAL
        assert config.figure_size == (10.0, 6.0)
        assert config.dpi == 300
        assert config.x_label == "Wavelength (nm)"

    def test_percentile_validation_valid(self):
        """Valid percentile ranges pass validation."""
        config = SpectrogramConfig(
            percentile_low=5.0,
            percentile_high=95.0,
        )
        assert config.percentile_low == 5.0
        assert config.percentile_high == 95.0

    def test_percentile_validation_invalid(self):
        """Invalid percentile range raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SpectrogramConfig(
                percentile_low=60.0,
                percentile_high=40.0,
            )
        assert "percentile_low" in str(exc_info.value)

    def test_percentile_bounds(self):
        """Percentile bounds are enforced."""
        with pytest.raises(ValidationError):
            SpectrogramConfig(percentile_low=-1.0)

        with pytest.raises(ValidationError):
            SpectrogramConfig(percentile_low=51.0)

        with pytest.raises(ValidationError):
            SpectrogramConfig(percentile_high=49.0)

        with pytest.raises(ValidationError):
            SpectrogramConfig(percentile_high=101.0)

    def test_dpi_bounds(self):
        """DPI must be in valid range."""
        config = SpectrogramConfig(dpi=72)
        assert config.dpi == 72

        config = SpectrogramConfig(dpi=600)
        assert config.dpi == 600

        with pytest.raises(ValidationError):
            SpectrogramConfig(dpi=50)

        with pytest.raises(ValidationError):
            SpectrogramConfig(dpi=1000)

    def test_title_template(self):
        """Title template can include placeholders."""
        config = SpectrogramConfig(
            title_template="Sol {sol} - {target} ({region})"
        )
        assert "{sol}" in config.title_template
        assert "{target}" in config.title_template

    def test_overlay_settings(self):
        """Overlay settings can be configured."""
        config = SpectrogramConfig(
            show_peak_markers=True,
            show_mineral_bands=True,
            highlight_points=[0, 5, 10],
        )
        assert config.show_peak_markers is True
        assert config.show_mineral_bands is True
        assert config.highlight_points == [0, 5, 10]

    def test_serialization(self):
        """Config serializes to JSON correctly."""
        config = SpectrogramConfig(colormap=ColorMapType.PLASMA)
        data = config.model_dump()

        assert data["colormap"] == "plasma"
        assert data["intensity_scale"] == "linear"
        assert "figure_size" in data


class TestSpectrogramData:
    """Tests for SpectrogramData model."""

    @pytest.fixture
    def sample_matrix(self):
        """Create a sample intensity matrix."""
        np.random.seed(42)
        return np.random.rand(10, 100).astype(np.float32)

    @pytest.fixture
    def sample_data(self, sample_matrix):
        """Create sample SpectrogramData."""
        compressed = SpectrogramData.compress_matrix(sample_matrix)
        return SpectrogramData(
            intensity_matrix=compressed,
            n_points=10,
            n_channels=100,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

    def test_compress_decompress_matrix(self, sample_matrix):
        """Matrix compression round-trip preserves data."""
        compressed = SpectrogramData.compress_matrix(sample_matrix)
        assert isinstance(compressed, bytes)
        assert len(compressed) < sample_matrix.nbytes  # Should be smaller

        decompressed = SpectrogramData.decompress_matrix(
            compressed, sample_matrix.shape
        )
        np.testing.assert_array_almost_equal(sample_matrix, decompressed)

    def test_compress_decompress_array(self):
        """1D array compression round-trip preserves data."""
        original = [1.0, 2.5, 3.7, 4.2]
        compressed = SpectrogramData.compress_array(original)
        decompressed = SpectrogramData.decompress_array(compressed)

        np.testing.assert_array_almost_equal(original, decompressed)

    def test_basic_creation(self, sample_data):
        """SpectrogramData creates with required fields."""
        assert sample_data.n_points == 10
        assert sample_data.n_channels == 100
        assert sample_data.wavenumber_min == 200.0
        assert sample_data.wavenumber_max == 4000.0

    def test_get_intensity_matrix(self, sample_data, sample_matrix):
        """get_intensity_matrix returns correct array."""
        matrix = sample_data.get_intensity_matrix()

        assert matrix.shape == (10, 100)
        np.testing.assert_array_almost_equal(matrix, sample_matrix)

    def test_get_wavenumbers_linear(self, sample_data):
        """get_wavenumbers returns linear array when no explicit array."""
        wn = sample_data.get_wavenumbers()

        assert len(wn) == 100
        assert wn[0] == 200.0
        assert wn[-1] == 4000.0

    def test_get_wavenumbers_explicit(self, sample_matrix):
        """get_wavenumbers returns explicit array when provided."""
        wn_values = [200.0, 500.0, 1000.0, 2000.0, 4000.0]
        compressed = SpectrogramData.compress_matrix(sample_matrix[:, :5])
        wn_compressed = SpectrogramData.compress_array(wn_values)

        data = SpectrogramData(
            intensity_matrix=compressed,
            n_points=10,
            n_channels=5,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
            wavenumbers=wn_compressed,
        )

        result = data.get_wavenumbers()
        np.testing.assert_array_almost_equal(result, wn_values)

    def test_get_extent(self, sample_data):
        """get_extent returns correct imshow extent."""
        extent = sample_data.get_extent()

        # (left, right, bottom, top)
        assert extent[0] == 200.0  # wavenumber_min
        assert extent[1] == 4000.0  # wavenumber_max
        assert extent[2] == 9.5  # n_points - 0.5
        assert extent[3] == -0.5  # top

    def test_point_labels(self, sample_matrix):
        """Point labels are stored correctly."""
        compressed = SpectrogramData.compress_matrix(sample_matrix)
        labels = [f"point_{i}" for i in range(10)]

        data = SpectrogramData(
            intensity_matrix=compressed,
            n_points=10,
            n_channels=100,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
            point_labels=labels,
        )

        assert data.point_labels == labels

    def test_intensity_stats(self, sample_matrix):
        """Intensity min/max are stored."""
        compressed = SpectrogramData.compress_matrix(sample_matrix)

        data = SpectrogramData(
            intensity_matrix=compressed,
            n_points=10,
            n_channels=100,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
            intensity_min=float(sample_matrix.min()),
            intensity_max=float(sample_matrix.max()),
        )

        assert data.intensity_min == pytest.approx(sample_matrix.min())
        assert data.intensity_max == pytest.approx(sample_matrix.max())

    def test_wavenumber_validation(self):
        """Wavenumber order is validated."""
        matrix = np.random.rand(5, 50).astype(np.float32)
        compressed = SpectrogramData.compress_matrix(matrix)

        with pytest.raises(ValidationError) as exc_info:
            SpectrogramData(
                intensity_matrix=compressed,
                n_points=5,
                n_channels=50,
                wavenumber_min=4000.0,  # Greater than max!
                wavenumber_max=200.0,
            )
        assert "wavenumber_min" in str(exc_info.value)

    def test_positive_dimensions(self):
        """n_points and n_channels must be positive."""
        matrix = np.random.rand(5, 50).astype(np.float32)
        compressed = SpectrogramData.compress_matrix(matrix)

        with pytest.raises(ValidationError):
            SpectrogramData(
                intensity_matrix=compressed,
                n_points=0,
                n_channels=50,
                wavenumber_min=200.0,
                wavenumber_max=4000.0,
            )

        with pytest.raises(ValidationError):
            SpectrogramData(
                intensity_matrix=compressed,
                n_points=5,
                n_channels=0,
                wavenumber_min=200.0,
                wavenumber_max=4000.0,
            )


class TestSpectrogram:
    """Tests for Spectrogram model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample SpectrogramData."""
        matrix = np.random.rand(20, 200).astype(np.float32)
        compressed = SpectrogramData.compress_matrix(matrix)
        return SpectrogramData(
            intensity_matrix=compressed,
            n_points=20,
            n_channels=200,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

    @pytest.fixture
    def sample_spectrogram(self, sample_data):
        """Create a sample Spectrogram."""
        return Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            data=sample_data,
        )

    def test_basic_creation(self, sample_spectrogram):
        """Spectrogram creates with required fields."""
        assert sample_spectrogram.scan_id is not None
        assert sample_spectrogram.region == SpectralRegion.R1
        assert sample_spectrogram.processing_level == ProcessingLevel.NORMALIZED
        assert sample_spectrogram.config is not None
        assert sample_spectrogram.data is not None

    def test_has_uuid(self, sample_spectrogram):
        """Spectrogram has auto-generated UUID."""
        assert sample_spectrogram.id is not None
        assert isinstance(sample_spectrogram.id, uuid.UUID)

    def test_custom_config(self, sample_data):
        """Custom config is applied."""
        config = SpectrogramConfig(colormap=ColorMapType.PLASMA)
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=sample_data,
        )

        assert spectrogram.config.colormap == ColorMapType.PLASMA

    def test_render_title_with_custom(self, sample_data):
        """Custom title overrides template."""
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            data=sample_data,
            title="My Custom Title",
        )

        title = spectrogram.render_title(target="Test", sol=921)
        assert title == "My Custom Title"

    def test_render_title_with_template(self, sample_data):
        """Title template is rendered with context."""
        config = SpectrogramConfig(
            title_template="{target} Sol {sol} - {region}"
        )
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=sample_data,
        )

        title = spectrogram.render_title(target="Amherst_Point", sol=921)
        assert "Amherst_Point" in title
        assert "921" in title
        assert "R1" in title

    def test_point_indices(self, sample_data):
        """Point indices are stored for subset spectrograms."""
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            data=sample_data,
            point_indices=[0, 5, 10, 15],
        )

        assert spectrogram.point_indices == [0, 5, 10, 15]

    def test_get_normalized_matrix_none(self, sample_data):
        """NONE normalization returns raw matrix."""
        config = SpectrogramConfig(normalization=NormalizationType.NONE)
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=sample_data,
        )

        raw = sample_data.get_intensity_matrix()
        normalized = spectrogram.get_normalized_matrix()

        np.testing.assert_array_equal(raw, normalized)

    def test_get_normalized_matrix_global(self, sample_data):
        """GLOBAL normalization scales to 0-1."""
        config = SpectrogramConfig(normalization=NormalizationType.GLOBAL)
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=sample_data,
        )

        normalized = spectrogram.get_normalized_matrix()

        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_get_normalized_matrix_per_spectrum(self):
        """PER_SPECTRUM normalization normalizes each row."""
        # Create matrix with known values
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ], dtype=np.float32)
        compressed = SpectrogramData.compress_matrix(matrix)
        data = SpectrogramData(
            intensity_matrix=compressed,
            n_points=2,
            n_channels=3,
            wavenumber_min=0.0,
            wavenumber_max=2.0,
        )

        config = SpectrogramConfig(normalization=NormalizationType.PER_SPECTRUM)
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=data,
        )

        normalized = spectrogram.get_normalized_matrix()

        # Each row should be normalized to 0-1
        assert normalized[0, 0] == pytest.approx(0.0)
        assert normalized[0, 2] == pytest.approx(1.0)
        assert normalized[1, 0] == pytest.approx(0.0)
        assert normalized[1, 2] == pytest.approx(1.0)

    def test_get_normalized_matrix_percentile(self):
        """PERCENTILE normalization clips and scales."""
        # Create matrix with outliers
        matrix = np.array([
            [0.0, 50.0, 100.0],
            [1000.0, 50.0, 0.0],  # 1000 is an outlier
        ], dtype=np.float32)
        compressed = SpectrogramData.compress_matrix(matrix)
        data = SpectrogramData(
            intensity_matrix=compressed,
            n_points=2,
            n_channels=3,
            wavenumber_min=0.0,
            wavenumber_max=2.0,
        )

        config = SpectrogramConfig(
            normalization=NormalizationType.PERCENTILE,
            percentile_low=0.0,
            percentile_high=75.0,  # Clip at 75th percentile
        )
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=data,
        )

        normalized = spectrogram.get_normalized_matrix()

        # Max should be 1.0 (clipped)
        assert normalized.max() == pytest.approx(1.0)

    def test_serialization(self, sample_spectrogram):
        """Spectrogram serializes correctly."""
        data = sample_spectrogram.model_dump()

        assert "id" in data
        assert "scan_id" in data
        assert "region" in data
        assert "processing_level" in data
        assert "config" in data
        assert "data" in data

        # Enums serialize to values
        assert data["region"] == "R1"
        assert data["processing_level"] == "normalized"


class TestDifferenceSpectrogram:
    """Tests for DifferenceSpectrogram model."""

    @pytest.fixture
    def sample_diff_data(self):
        """Create sample difference data."""
        matrix = np.random.randn(10, 50).astype(np.float32)  # Can be negative
        compressed = SpectrogramData.compress_matrix(matrix)
        return SpectrogramData(
            intensity_matrix=compressed,
            n_points=10,
            n_channels=50,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

    def test_basic_creation(self, sample_diff_data):
        """DifferenceSpectrogram creates with required fields."""
        diff = DifferenceSpectrogram(
            spectrogram_a_id=uuid.uuid4(),
            spectrogram_b_id=uuid.uuid4(),
            data=sample_diff_data,
        )

        assert diff.operation == "subtract"
        assert diff.config.colormap == ColorMapType.COOLWARM

    def test_valid_operations(self, sample_diff_data):
        """Valid operations are accepted."""
        for op in ["subtract", "ratio", "log_ratio"]:
            diff = DifferenceSpectrogram(
                spectrogram_a_id=uuid.uuid4(),
                spectrogram_b_id=uuid.uuid4(),
                operation=op,
                data=sample_diff_data,
            )
            assert diff.operation == op

    def test_invalid_operation(self, sample_diff_data):
        """Invalid operation raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DifferenceSpectrogram(
                spectrogram_a_id=uuid.uuid4(),
                spectrogram_b_id=uuid.uuid4(),
                operation="invalid",
                data=sample_diff_data,
            )
        assert "operation" in str(exc_info.value)

    def test_default_config_diverging(self, sample_diff_data):
        """Default config uses diverging colormap."""
        diff = DifferenceSpectrogram(
            spectrogram_a_id=uuid.uuid4(),
            spectrogram_b_id=uuid.uuid4(),
            data=sample_diff_data,
        )

        assert diff.config.colormap == ColorMapType.COOLWARM
        assert diff.config.normalization == NormalizationType.ZSCORE

    def test_has_uuid(self, sample_diff_data):
        """DifferenceSpectrogram has auto-generated UUID."""
        diff = DifferenceSpectrogram(
            spectrogram_a_id=uuid.uuid4(),
            spectrogram_b_id=uuid.uuid4(),
            data=sample_diff_data,
        )

        assert diff.id is not None
        assert isinstance(diff.id, uuid.UUID)


class TestSpectrogramIntegration:
    """Integration tests for spectrogram models."""

    def test_full_workflow(self):
        """Test typical spectrogram creation workflow."""
        # 1. Create intensity matrix (simulating stacked spectra)
        n_points = 50
        n_channels = 501
        matrix = np.random.rand(n_points, n_channels).astype(np.float32)

        # Add some structure (peaks at certain positions)
        for i in range(n_points):
            matrix[i, 200:210] += 0.5  # Simulated peak

        # 2. Create data container
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(matrix),
            n_points=n_points,
            n_channels=n_channels,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
            intensity_min=float(matrix.min()),
            intensity_max=float(matrix.max()),
            point_labels=[f"p{i}" for i in range(n_points)],
        )

        # 3. Create configuration
        config = SpectrogramConfig(
            colormap=ColorMapType.VIRIDIS,
            normalization=NormalizationType.PERCENTILE,
            percentile_low=1.0,
            percentile_high=99.0,
            title_template="Sol {sol} - {target}",
            show_colorbar=True,
        )

        # 4. Create spectrogram
        scan_id = uuid.uuid4()
        spectrogram = Spectrogram(
            scan_id=scan_id,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            data=data,
        )

        # Verify everything works
        assert spectrogram.scan_id == scan_id
        assert spectrogram.data.n_points == n_points

        # Get normalized matrix
        normalized = spectrogram.get_normalized_matrix()
        assert normalized.shape == (n_points, n_channels)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Render title
        title = spectrogram.render_title(target="Amherst_Point", sol=921)
        assert "921" in title
        assert "Amherst_Point" in title

        # Serialize and deserialize
        json_str = spectrogram.model_dump_json()
        restored = Spectrogram.model_validate_json(json_str)
        assert restored.id == spectrogram.id
        assert restored.region == spectrogram.region

    def test_model_registry(self):
        """Spectrogram is registered in ModelRegistry."""
        from sherloc_pipeline.models.base import ModelRegistry

        # Import spectrogram module to ensure registration happens
        # (Registration occurs at module import time via @ModelRegistry.register)
        from sherloc_pipeline.models import spectrogram as _  # noqa: F401

        # Re-import to ensure fresh registration after any previous test cleanup
        import importlib
        from sherloc_pipeline.models import spectrogram
        importlib.reload(spectrogram)

        # Spectrogram should be registered
        assert ModelRegistry.get("Spectrogram") is not None
        assert ModelRegistry.get("Spectrogram").__name__ == "Spectrogram"

    def test_import_from_models_package(self):
        """All spectrogram types can be imported from models package."""
        from sherloc_pipeline.models import (
            ColorMapType,
            AxisScale,
            NormalizationType,
            InterpolationMethod,
            SpectrogramConfig,
            SpectrogramData,
            Spectrogram,
            DifferenceSpectrogram,
        )

        # All should be importable
        assert ColorMapType is not None
        assert AxisScale is not None
        assert NormalizationType is not None
        assert InterpolationMethod is not None
        assert SpectrogramConfig is not None
        assert SpectrogramData is not None
        assert Spectrogram is not None
        assert DifferenceSpectrogram is not None
