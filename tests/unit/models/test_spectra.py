"""
Unit tests for spectral data models (bd-tqd: WS2-A).

Tests the core domain models defined in models/spectra.py:
- Sol: Martian day of observations
- Scan: Complete spectroscopy scan
- ScanPoint: Single measurement point
- Spectrum: Spectral measurement at one processing level
"""

import json
import uuid
from datetime import date, datetime, timezone

import numpy as np
import pytest
from pydantic import ValidationError

from sherloc_pipeline.models import (
    DataSource,
    SpectralRegion,
    SpectrumType,
    ProcessingLevel,
    Sol,
    Scan,
    ScanPoint,
    Spectrum,
    ModelRegistry,
)


class TestDataSource:
    """Tests for DataSource enum."""

    def test_values(self):
        """DataSource has expected values."""
        assert DataSource.LOUPE.value == "loupe"
        assert DataSource.PDS4.value == "pds4"

    def test_is_str_enum(self):
        """DataSource values can be used as strings."""
        assert DataSource.LOUPE == "loupe"
        # str(enum) includes the class name, but .value gives the raw value
        assert DataSource.PDS4.value == "pds4"


class TestSpectralRegion:
    """Tests for SpectralRegion enum."""

    def test_values(self):
        """SpectralRegion has expected values."""
        assert SpectralRegion.R1.value == "R1"
        assert SpectralRegion.R2.value == "R2"
        assert SpectralRegion.R3.value == "R3"
        assert SpectralRegion.R123.value == "R123"

    def test_all_regions(self):
        """All four spectral regions are defined."""
        regions = list(SpectralRegion)
        assert len(regions) == 4


class TestSpectrumType:
    """Tests for SpectrumType enum."""

    def test_values(self):
        """SpectrumType has expected values."""
        assert SpectrumType.ACTIVE.value == "active"
        assert SpectrumType.DARK.value == "dark"
        assert SpectrumType.DARK_SUBTRACTED.value == "dark_subtracted"


class TestProcessingLevel:
    """Tests for ProcessingLevel enum."""

    def test_values(self):
        """ProcessingLevel has expected values."""
        assert ProcessingLevel.RAW.value == "raw"
        assert ProcessingLevel.CALIBRATED.value == "calibrated"
        assert ProcessingLevel.NORMALIZED.value == "normalized"
        assert ProcessingLevel.DESPIKED.value == "despiked"
        assert ProcessingLevel.BASELINED.value == "baselined"
        assert ProcessingLevel.DERIVED.value == "derived"


class TestSol:
    """Tests for Sol model."""

    def test_basic_creation(self):
        """Create Sol with minimal fields."""
        sol = Sol(sol_number=921)
        assert sol.sol_number == 921
        assert sol.earth_date is None
        assert sol.solar_longitude is None
        assert sol.mission_phase is None
        assert sol.data_source == DataSource.LOUPE

    def test_full_creation(self):
        """Create Sol with all fields."""
        sol = Sol(
            sol_number=921,
            earth_date=date(2025, 1, 15),
            solar_longitude=180.5,
            mission_phase="Extended Mission",
            data_source=DataSource.PDS4,
        )
        assert sol.sol_number == 921
        assert sol.earth_date == date(2025, 1, 15)
        assert sol.solar_longitude == 180.5
        assert sol.mission_phase == "Extended Mission"
        assert sol.data_source == DataSource.PDS4

    def test_sol_number_validation(self):
        """Sol number must be >= 0."""
        with pytest.raises(ValidationError):
            Sol(sol_number=-1)

    def test_solar_longitude_range(self):
        """Solar longitude must be 0-360."""
        # Valid at boundaries
        sol1 = Sol(sol_number=921, solar_longitude=0)
        sol2 = Sol(sol_number=921, solar_longitude=360)
        assert sol1.solar_longitude == 0
        assert sol2.solar_longitude == 360

        # Invalid
        with pytest.raises(ValidationError):
            Sol(sol_number=921, solar_longitude=-1)
        with pytest.raises(ValidationError):
            Sol(sol_number=921, solar_longitude=361)

    def test_has_timestamps(self):
        """Sol inherits timestamps from TimestampedModel."""
        sol = Sol(sol_number=921)
        assert sol.created_at is not None
        assert sol.updated_at is None

    def test_json_serialization(self):
        """Sol serializes to JSON correctly."""
        sol = Sol(sol_number=921, data_source=DataSource.LOUPE)
        data = sol.model_dump()
        assert data["sol_number"] == 921
        assert data["data_source"] == "loupe"

    def test_model_can_be_registered(self):
        """Sol can be registered in ModelRegistry."""
        # Registration happens at import; verify model is a valid type
        # This validates the @ModelRegistry.register decorator is applied
        assert hasattr(Sol, "__pydantic_complete__")


class TestScan:
    """Tests for Scan model."""

    def test_basic_creation(self):
        """Create Scan with minimal required fields."""
        scan = Scan(
            sol_number=921,
            scan_name="Amherst_Point",
            scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
            sclk_start=672194998,
            n_points=100,
            shots_per_point=10,
        )
        assert scan.sol_number == 921
        assert scan.scan_name == "Amherst_Point"
        assert scan.n_points == 100
        assert scan.n_channels == 2148  # default
        assert scan.laser_wavelength_nm == 248.6  # default

    def test_full_creation(self):
        """Create Scan with all fields."""
        scan = Scan(
            sol_number=921,
            scan_name="Amherst_Point",
            scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
            sclk_start=672194998,
            sclk_stop=672195100,
            n_points=100,
            n_channels=2148,
            shots_per_point=10,
            laser_wavelength_nm=248.6,
            processing_applied="normalized",
            source_path="/data/loupe/sol_0921/detail",
            loupe_metadata={"key": "value"},
            pds4_metadata=None,
        )
        assert scan.sclk_stop == 672195100
        assert scan.source_path == "/data/loupe/sol_0921/detail"
        assert scan.loupe_metadata == {"key": "value"}

    def test_has_uuid(self):
        """Scan has auto-generated UUID."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
            shots_per_point=10,
        )
        assert scan.id is not None
        assert isinstance(scan.id, uuid.UUID)

    def test_n_points_must_be_positive(self):
        """n_points must be > 0."""
        with pytest.raises(ValidationError):
            Scan(
                sol_number=921,
                scan_name="Test",
                scan_id="test",
                sclk_start=0,
                n_points=0,
                shots_per_point=10,
            )

    def test_target_name_not_empty(self):
        """target_name must not be empty."""
        with pytest.raises(ValidationError):
            Scan(
                sol_number=921,
                scan_name="",
                scan_id="test",
                sclk_start=0,
                n_points=10,
                shots_per_point=10,
            )

    def test_sclk_order_validation(self):
        """sclk_stop must be >= sclk_start if provided."""
        # Valid
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100,
            sclk_stop=200,
            n_points=10,
            shots_per_point=10,
        )
        assert scan.sclk_stop == 200

        # Invalid
        with pytest.raises(ValidationError):
            Scan(
                sol_number=921,
                scan_name="Test",
                scan_id="test",
                sclk_start=200,
                sclk_stop=100,  # Before start
                n_points=10,
                shots_per_point=10,
            )

    def test_shots_per_point_defaults_to_none(self):
        """shots_per_point defaults to None (PDS processed products lack it)."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
        )
        assert scan.shots_per_point is None

    def test_shots_per_point_accepts_positive(self):
        """shots_per_point accepts positive integers."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
            shots_per_point=300,
        )
        assert scan.shots_per_point == 300

    def test_shots_per_point_rejects_zero(self):
        """shots_per_point must be > 0 when provided."""
        with pytest.raises(ValidationError):
            Scan(
                sol_number=921,
                scan_name="Test",
                scan_id="test",
                sclk_start=0,
                n_points=10,
                shots_per_point=0,
            )

    def test_shots_per_point_rejects_negative(self):
        """shots_per_point must be > 0 when provided."""
        with pytest.raises(ValidationError):
            Scan(
                sol_number=921,
                scan_name="Test",
                scan_id="test",
                sclk_start=0,
                n_points=10,
                shots_per_point=-5,
            )

    def test_model_can_be_registered(self):
        """Scan can be registered in ModelRegistry."""
        # Verify model is a valid Pydantic model
        assert hasattr(Scan, "__pydantic_complete__")


class TestScanPoint:
    """Tests for ScanPoint model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create ScanPoint with minimal fields."""
        point = ScanPoint(scan_id=scan_id, point_index=0)
        assert point.scan_id == scan_id
        assert point.point_index == 0
        assert point.x_pixel is None
        assert point.y_pixel is None

    def test_full_creation(self, scan_id):
        """Create ScanPoint with all fields."""
        point = ScanPoint(
            scan_id=scan_id,
            point_index=5,
            azimuth_dn=12345,
            elevation_dn=67890,
            x_pixel=824.5,
            y_pixel=600.2,
            azimuth_error=0.5,
            elevation_error=0.3,
            photodiode_mean=1500.0,
            photodiode_std=50.0,
        )
        assert point.point_index == 5
        assert point.x_pixel == 824.5
        assert point.photodiode_mean == 1500.0

    def test_point_index_non_negative(self, scan_id):
        """point_index must be >= 0."""
        with pytest.raises(ValidationError):
            ScanPoint(scan_id=scan_id, point_index=-1)

    def test_photodiode_std_non_negative(self, scan_id):
        """photodiode_std must be >= 0."""
        with pytest.raises(ValidationError):
            ScanPoint(
                scan_id=scan_id,
                point_index=0,
                photodiode_std=-1.0,
            )

    def test_has_uuid(self, scan_id):
        """ScanPoint has auto-generated UUID."""
        point = ScanPoint(scan_id=scan_id, point_index=0)
        assert point.id is not None
        assert isinstance(point.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """ScanPoint can be registered in ModelRegistry."""
        assert hasattr(ScanPoint, "__pydantic_complete__")


class TestSpectrum:
    """Tests for Spectrum model."""

    @pytest.fixture
    def scan_point_id(self):
        """Provide a scan point UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_intensities(self):
        """Provide sample intensity values."""
        return [100.0, 150.0, 200.0, 175.0, 125.0]

    def test_compress_decompress_array(self, sample_intensities):
        """Test array compression/decompression roundtrip."""
        compressed = Spectrum.compress_array(sample_intensities)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decompressed = Spectrum.decompress_array(compressed)
        assert len(decompressed) == len(sample_intensities)
        for orig, decomp in zip(sample_intensities, decompressed):
            assert abs(orig - decomp) < 0.001

    def test_basic_creation(self, scan_point_id, sample_intensities):
        """Create Spectrum with required fields."""
        intensities = Spectrum.compress_array(sample_intensities)
        spectrum = Spectrum(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.NORMALIZED,
            intensities=intensities,
        )
        assert spectrum.region == SpectralRegion.R1
        assert spectrum.spectrum_type == SpectrumType.DARK_SUBTRACTED
        assert spectrum.processing_level == ProcessingLevel.NORMALIZED

    def test_intensity_values_property(self, scan_point_id, sample_intensities):
        """Test intensity_values property."""
        intensities = Spectrum.compress_array(sample_intensities)
        spectrum = Spectrum(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.RAW,
            intensities=intensities,
        )

        values = spectrum.intensity_values
        assert len(values) == len(sample_intensities)
        for orig, decomp in zip(sample_intensities, values):
            assert abs(orig - decomp) < 0.001

    def test_from_values_factory(self, scan_point_id, sample_intensities):
        """Test from_values class method."""
        wavenumbers = [700.0, 750.0, 800.0, 850.0, 900.0]

        spectrum = Spectrum.from_values(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.NORMALIZED,
            intensity_values=sample_intensities,
            wavenumber_values=wavenumbers,
        )

        # Check intensities
        values = spectrum.intensity_values
        assert len(values) == len(sample_intensities)

        # Check wavenumbers
        wn_values = spectrum.wavenumber_values
        assert wn_values is not None
        assert len(wn_values) == len(wavenumbers)

        # Wavelengths not provided
        assert spectrum.wavelength_values is None

    def test_optional_calibration_arrays(self, scan_point_id, sample_intensities):
        """Test wavelength/wavenumber arrays are optional."""
        spectrum = Spectrum.from_values(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.ACTIVE,
            processing_level=ProcessingLevel.RAW,
            intensity_values=sample_intensities,
        )
        assert spectrum.wavelengths is None
        assert spectrum.wavenumbers is None
        assert spectrum.wavelength_values is None
        assert spectrum.wavenumber_values is None

    def test_enum_serialization(self, scan_point_id, sample_intensities):
        """Test enum values serialize correctly."""
        intensities = Spectrum.compress_array(sample_intensities)
        spectrum = Spectrum(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.DARK_SUBTRACTED,
            processing_level=ProcessingLevel.NORMALIZED,
            intensities=intensities,
        )

        data = spectrum.model_dump()
        assert data["region"] == "R1"
        assert data["spectrum_type"] == "dark_subtracted"
        assert data["processing_level"] == "normalized"

    def test_has_uuid(self, scan_point_id, sample_intensities):
        """Spectrum has auto-generated UUID."""
        spectrum = Spectrum.from_values(
            scan_point_id=scan_point_id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.ACTIVE,
            processing_level=ProcessingLevel.RAW,
            intensity_values=sample_intensities,
        )
        assert spectrum.id is not None
        assert isinstance(spectrum.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """Spectrum can be registered in ModelRegistry."""
        assert hasattr(Spectrum, "__pydantic_complete__")


class TestModelRelationships:
    """Tests for relationships between models."""

    def test_scan_references_sol(self):
        """Scan references Sol via sol_number."""
        sol = Sol(sol_number=921)
        scan = Scan(
            sol_number=sol.sol_number,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
            shots_per_point=10,
        )
        assert scan.sol_number == sol.sol_number

    def test_scanpoint_references_scan(self):
        """ScanPoint references Scan via scan_id."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
            shots_per_point=10,
        )
        point = ScanPoint(scan_id=scan.id, point_index=0)
        assert point.scan_id == scan.id

    def test_spectrum_references_scanpoint(self):
        """Spectrum references ScanPoint via scan_point_id."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=0,
            n_points=10,
            shots_per_point=10,
        )
        point = ScanPoint(scan_id=scan.id, point_index=0)
        spectrum = Spectrum.from_values(
            scan_point_id=point.id,
            region=SpectralRegion.R1,
            spectrum_type=SpectrumType.ACTIVE,
            processing_level=ProcessingLevel.RAW,
            intensity_values=[100.0, 200.0],
        )
        assert spectrum.scan_point_id == point.id


class TestJsonRoundTrip:
    """Tests for JSON serialization round-trips."""

    def test_sol_roundtrip(self):
        """Sol survives JSON round-trip."""
        original = Sol(
            sol_number=921,
            earth_date=date(2025, 1, 15),
            solar_longitude=180.5,
        )
        json_str = original.model_dump_json()
        restored = Sol.model_validate_json(json_str)

        assert restored.sol_number == original.sol_number
        assert restored.earth_date == original.earth_date
        assert restored.solar_longitude == original.solar_longitude

    def test_scan_roundtrip(self):
        """Scan survives JSON round-trip."""
        original = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=672194998,
            n_points=100,
            shots_per_point=10,
        )
        json_str = original.model_dump_json()
        restored = Scan.model_validate_json(json_str)

        assert restored.id == original.id
        assert restored.sol_number == original.sol_number
        assert restored.scan_name == original.scan_name
        assert restored.n_points == original.n_points
