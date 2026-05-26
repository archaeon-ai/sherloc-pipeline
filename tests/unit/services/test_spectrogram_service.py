"""
Unit tests for SpectrogramService (bd-3mx: WS4-B).

Tests the spectrogram rendering service:
- Generation from Spectrum lists
- Generation from numpy matrices
- Normalization and rendering
- Difference spectrogram creation
- ServiceResult conversion
"""

from pathlib import Path
import tempfile
import uuid

import numpy as np
import pytest

from sherloc_pipeline.models import (
    SpectralRegion,
    ProcessingLevel,
    SpectrumType,
    Spectrum,
)
from sherloc_pipeline.models.spectrogram import (
    ColorMapType,
    NormalizationType,
    SpectrogramConfig,
    SpectrogramData,
    Spectrogram,
)
from sherloc_pipeline.services.spectrogram import (
    SpectrogramService,
    SpectrogramRequest,
    SpectrogramResult,
)


class TestSpectrogramService:
    """Tests for SpectrogramService."""

    @pytest.fixture
    def service(self):
        """Create a SpectrogramService instance."""
        return SpectrogramService()

    @pytest.fixture
    def sample_spectra(self):
        """Create sample Spectrum objects for testing."""
        scan_point_id = uuid.uuid4()
        spectra = []
        np.random.seed(42)

        for i in range(10):
            # Create intensity values with some structure
            intensity = np.random.rand(100).astype(np.float32)
            intensity[40:50] += 0.5  # Add a peak

            spectrum = Spectrum.from_values(
                scan_point_id=scan_point_id,
                region=SpectralRegion.R1,
                spectrum_type=SpectrumType.DARK_SUBTRACTED,
                processing_level=ProcessingLevel.NORMALIZED,
                intensity_values=intensity.tolist(),
                wavenumber_values=np.linspace(200, 4000, 100).tolist(),
            )
            spectra.append(spectrum)

        return spectra

    @pytest.fixture
    def scan_id(self):
        """Generate a sample scan UUID."""
        return uuid.uuid4()

    def test_service_creation(self, service):
        """Service creates with default configuration."""
        assert service._default_config is not None
        assert service._default_config.colormap == ColorMapType.VIRIDIS

    def test_service_custom_default_config(self):
        """Service accepts custom default configuration."""
        config = SpectrogramConfig(colormap=ColorMapType.PLASMA)
        service = SpectrogramService(default_config=config)
        assert service._default_config.colormap == ColorMapType.PLASMA

    def test_generate_from_spectra_basic(self, service, sample_spectra, scan_id):
        """Generate spectrogram from Spectrum list."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        assert spectrogram.scan_id == scan_id
        assert spectrogram.region == SpectralRegion.R1
        assert spectrogram.processing_level == ProcessingLevel.NORMALIZED
        assert spectrogram.data.n_points == 10
        assert spectrogram.data.n_channels == 100
        assert spectrogram.data.wavenumber_min == 200.0
        assert spectrogram.data.wavenumber_max == 4000.0

    def test_generate_from_spectra_with_config(self, service, sample_spectra, scan_id):
        """Generate with custom configuration."""
        config = SpectrogramConfig(
            colormap=ColorMapType.MAGMA,
            normalization=NormalizationType.GLOBAL,
        )

        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
        )

        assert spectrogram.config.colormap == ColorMapType.MAGMA
        assert spectrogram.config.normalization == NormalizationType.GLOBAL

    def test_generate_from_spectra_empty_raises(self, service, scan_id):
        """Empty spectra list raises ValueError."""
        with pytest.raises(ValueError, match="At least one spectrum is required"):
            service.generate_from_spectra(
                scan_id=scan_id,
                spectra=[],
                region=SpectralRegion.R1,
                processing_level=ProcessingLevel.NORMALIZED,
            )

    def test_generate_from_spectra_mismatched_lengths(self, service, scan_id):
        """Mismatched spectrum lengths raise ValueError."""
        scan_point_id = uuid.uuid4()

        # Create spectra with different lengths
        spectrum1 = Spectrum.from_values(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.NORMALIZED,
            intensity_values=[1.0, 2.0, 3.0],
        )
        spectrum2 = Spectrum.from_values(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.NORMALIZED,
            intensity_values=[1.0, 2.0, 3.0, 4.0],  # Different length
        )

        with pytest.raises(ValueError, match="Spectrum 1 has 4 channels"):
            service.generate_from_spectra(
                scan_id=scan_id,
                spectra=[spectrum1, spectrum2],
                region=SpectralRegion.R1,
                processing_level=ProcessingLevel.NORMALIZED,
            )

    def test_generate_from_matrix_basic(self, service, scan_id):
        """Generate spectrogram from numpy matrix."""
        matrix = np.random.rand(20, 200).astype(np.float32)

        spectrogram = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

        assert spectrogram.data.n_points == 20
        assert spectrogram.data.n_channels == 200
        assert spectrogram.data.wavenumber_min == 200.0
        assert spectrogram.data.wavenumber_max == 4000.0

    def test_generate_from_matrix_with_wavenumbers(self, service, scan_id):
        """Generate with explicit wavenumber array."""
        matrix = np.random.rand(10, 50).astype(np.float32)
        wavenumbers = np.linspace(500, 2000, 50)

        spectrogram = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=500.0,
            wavenumber_max=2000.0,
            wavenumbers=wavenumbers,
        )

        # Check wavenumbers are stored (use lower precision due to float32 storage)
        wn = spectrogram.data.get_wavenumbers()
        np.testing.assert_array_almost_equal(wn, wavenumbers, decimal=4)

    def test_generate_from_matrix_with_labels(self, service, scan_id):
        """Generate with custom point labels."""
        matrix = np.random.rand(5, 50).astype(np.float32)
        labels = ["A", "B", "C", "D", "E"]

        spectrogram = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
            point_labels=labels,
        )

        assert spectrogram.data.point_labels == labels

    def test_generate_from_matrix_invalid_dimensions(self, service, scan_id):
        """1D or 3D matrices raise ValueError."""
        with pytest.raises(ValueError, match="Matrix must be 2D"):
            service.generate_from_matrix(
                scan_id=scan_id,
                matrix=np.random.rand(100),  # 1D
                region=SpectralRegion.R1,
                processing_level=ProcessingLevel.NORMALIZED,
                wavenumber_min=200.0,
                wavenumber_max=4000.0,
            )

    def test_render_basic(self, service, sample_spectra, scan_id):
        """Render spectrogram creates figure."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        fig, saved_path = service.render(spectrogram)

        assert fig is not None
        assert saved_path is None  # No output path provided

        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_with_title_context(self, service, sample_spectra, scan_id):
        """Render with target and sol for title."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        fig, _ = service.render(
            spectrogram,
            target="Amherst_Point",
            sol=921,
        )

        # Check title was rendered
        assert fig is not None
        ax = fig.axes[0]
        assert "921" in ax.get_title()
        assert "Amherst_Point" in ax.get_title()

        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_render_saves_to_file(self, service, sample_spectra, scan_id):
        """Render saves figure to specified path."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_spectrogram.png"

            fig, saved_path = service.render(spectrogram, output_path=output_path)

            assert saved_path == output_path
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_render_creates_parent_directories(self, service, sample_spectra, scan_id):
        """Render creates parent directories if needed."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "test.png"

            fig, saved_path = service.render(spectrogram, output_path=output_path)

            assert output_path.exists()

            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_create_difference_spectrogram_subtract(self, service, scan_id):
        """Create difference spectrogram with subtraction."""
        # Create two spectrograms
        matrix_a = np.random.rand(10, 50).astype(np.float32) + 1.0
        matrix_b = np.random.rand(10, 50).astype(np.float32)

        spec_a = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix_a,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.BASELINED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spec_b = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix_b,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

        diff = service.create_difference_spectrogram(spec_a, spec_b, operation="subtract")

        assert diff.operation == "subtract"
        assert diff.spectrogram_a_id == spec_a.id
        assert diff.spectrogram_b_id == spec_b.id
        assert diff.config.colormap == ColorMapType.COOLWARM

        # Check difference is correct
        diff_matrix = diff.data.get_intensity_matrix()
        expected = matrix_a - matrix_b
        np.testing.assert_array_almost_equal(diff_matrix, expected)

    def test_create_difference_spectrogram_ratio(self, service, scan_id):
        """Create difference spectrogram with ratio."""
        matrix_a = np.random.rand(10, 50).astype(np.float32) + 1.0
        matrix_b = np.random.rand(10, 50).astype(np.float32) + 0.5

        spec_a = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix_a,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spec_b = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=matrix_b,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

        diff = service.create_difference_spectrogram(spec_a, spec_b, operation="ratio")

        assert diff.operation == "ratio"

    def test_create_difference_spectrogram_mismatched_points(self, service, scan_id):
        """Mismatched point counts raise ValueError."""
        spec_a = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=np.random.rand(10, 50).astype(np.float32),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spec_b = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=np.random.rand(15, 50).astype(np.float32),  # Different n_points
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

        with pytest.raises(ValueError, match="Point count mismatch"):
            service.create_difference_spectrogram(spec_a, spec_b)

    def test_create_difference_spectrogram_mismatched_channels(self, service, scan_id):
        """Mismatched channel counts raise ValueError."""
        spec_a = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=np.random.rand(10, 50).astype(np.float32),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spec_b = service.generate_from_matrix(
            scan_id=scan_id,
            matrix=np.random.rand(10, 60).astype(np.float32),  # Different n_channels
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )

        with pytest.raises(ValueError, match="Channel count mismatch"):
            service.create_difference_spectrogram(spec_a, spec_b)

    def test_to_service_result(self, service, sample_spectra, scan_id):
        """Convert spectrogram to ServiceResult."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        result = service.to_service_result(spectrogram)

        assert "Generated spectrogram" in result.summary
        assert "R1" in result.summary
        assert result.metadata["n_points"] == 10
        assert result.metadata["n_channels"] == 100
        assert result.metadata["region"] == "R1"
        assert result.metadata["scan_id"] == str(scan_id)

    def test_to_service_result_with_path(self, service, sample_spectra, scan_id):
        """ServiceResult includes output path when provided."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        output_path = Path("/tmp/test.png")
        result = service.to_service_result(spectrogram, output_path=output_path)

        assert len(result.artifacts) == 1
        assert result.artifacts[0] == output_path

    def test_to_service_result_with_warnings(self, service, sample_spectra, scan_id):
        """ServiceResult includes warnings."""
        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=sample_spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        warnings = ["Low signal in some points"]
        result = service.to_service_result(spectrogram, warnings=warnings)

        assert result.warnings == warnings


class TestSpectrogramRequest:
    """Tests for SpectrogramRequest dataclass."""

    def test_basic_request(self):
        """Create basic spectrogram request."""
        scan_id = uuid.uuid4()
        request = SpectrogramRequest(
            scan_id=scan_id,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
        )

        assert request.scan_id == scan_id
        assert request.region == SpectralRegion.R1
        assert request.processing_level == ProcessingLevel.NORMALIZED
        assert request.config is None
        assert request.point_indices is None
        assert request.output_path is None
        assert request.output_format == "png"

    def test_request_with_all_options(self):
        """Create request with all options."""
        scan_id = uuid.uuid4()
        config = SpectrogramConfig(colormap=ColorMapType.PLASMA)
        output_path = Path("/tmp/output.pdf")

        request = SpectrogramRequest(
            scan_id=scan_id,
            region=SpectralRegion.R2,
            processing_level=ProcessingLevel.BASELINED,
            config=config,
            point_indices=[0, 5, 10],
            output_path=output_path,
            output_format="pdf",
        )

        assert request.config.colormap == ColorMapType.PLASMA
        assert request.point_indices == [0, 5, 10]
        assert request.output_path == output_path
        assert request.output_format == "pdf"


class TestSpectrogramResult:
    """Tests for SpectrogramResult dataclass."""

    def test_basic_result(self):
        """Create basic spectrogram result."""
        # Create a minimal spectrogram
        matrix = np.random.rand(5, 50).astype(np.float32)
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(matrix),
            n_points=5,
            n_channels=50,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            data=data,
        )

        result = SpectrogramResult(spectrogram=spectrogram)

        assert result.spectrogram == spectrogram
        assert result.figure_path is None
        assert result.warnings == []

    def test_result_with_path_and_warnings(self):
        """Create result with path and warnings."""
        matrix = np.random.rand(5, 50).astype(np.float32)
        data = SpectrogramData(
            intensity_matrix=SpectrogramData.compress_matrix(matrix),
            n_points=5,
            n_channels=50,
            wavenumber_min=200.0,
            wavenumber_max=4000.0,
        )
        spectrogram = Spectrogram(
            scan_id=uuid.uuid4(),
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            data=data,
        )

        result = SpectrogramResult(
            spectrogram=spectrogram,
            figure_path=Path("/tmp/output.png"),
            warnings=["Warning 1", "Warning 2"],
        )

        assert result.figure_path == Path("/tmp/output.png")
        assert len(result.warnings) == 2


class TestSpectrogramIntegration:
    """Integration tests for spectrogram service."""

    def test_full_workflow(self):
        """Test complete spectrogram generation workflow."""
        # Setup
        service = SpectrogramService()
        scan_id = uuid.uuid4()
        scan_point_id = uuid.uuid4()

        # Create realistic spectral data
        np.random.seed(42)
        n_points = 25
        n_channels = 501

        spectra = []
        wavenumbers = np.linspace(200, 4000, n_channels).tolist()

        for i in range(n_points):
            # Create spectrum with some peaks
            intensity = np.random.rand(n_channels).astype(np.float32) * 0.1

            # Add peaks at known positions
            intensity[100:110] += 0.5 * np.exp(-0.5 * (np.arange(10) - 5) ** 2)
            intensity[300:310] += 0.3 * np.exp(-0.5 * (np.arange(10) - 5) ** 2)

            spectrum = Spectrum.from_values(
                scan_point_id=scan_point_id,
                region=SpectralRegion.R1,
                spectrum_type=SpectrumType.DARK_SUBTRACTED,
                processing_level=ProcessingLevel.NORMALIZED,
                intensity_values=intensity.tolist(),
                wavenumber_values=wavenumbers,
            )
            spectra.append(spectrum)

        # Generate spectrogram
        config = SpectrogramConfig(
            colormap=ColorMapType.VIRIDIS,
            normalization=NormalizationType.PERCENTILE,
            percentile_low=1.0,
            percentile_high=99.0,
        )

        spectrogram = service.generate_from_spectra(
            scan_id=scan_id,
            spectra=spectra,
            region=SpectralRegion.R1,
            processing_level=ProcessingLevel.NORMALIZED,
            config=config,
            wavenumbers=wavenumbers,
        )

        # Verify
        assert spectrogram.data.n_points == n_points
        assert spectrogram.data.n_channels == n_channels

        # Check normalized matrix
        normalized = spectrogram.get_normalized_matrix()
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Render to file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_workflow.png"
            fig, saved = service.render(
                spectrogram,
                output_path=output_path,
                target="Test_Target",
                sol=100,
            )

            assert output_path.exists()
            assert output_path.stat().st_size > 10000  # Reasonable file size

            # Create service result
            result = service.to_service_result(spectrogram, output_path=output_path)
            assert "Generated spectrogram" in result.summary

            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_import_from_services_package(self):
        """Service can be imported from services package."""
        from sherloc_pipeline.services import (
            SpectrogramService,
            SpectrogramRequest,
            SpectrogramResult,
        )

        assert SpectrogramService is not None
        assert SpectrogramRequest is not None
        assert SpectrogramResult is not None
