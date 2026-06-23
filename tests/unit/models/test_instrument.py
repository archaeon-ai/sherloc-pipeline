"""
Unit tests for instrument state models (bd-tqd: WS2-A).

Tests the instrument telemetry and configuration models:
- InstrumentState: State-of-health telemetry
- CCDConfiguration: CCD timing and region configuration
- ScannerCalibration: Scanner coordinate calibration
"""

import uuid

import pytest
from pydantic import ValidationError

from sherloc_pipeline.models import (
    InstrumentState,
    CCDConfiguration,
    ScannerCalibration,
    ModelRegistry,
)


class TestInstrumentState:
    """Tests for InstrumentState model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create InstrumentState with minimal fields."""
        state = InstrumentState(scan_id=scan_id)
        assert state.scan_id == scan_id
        assert state.ccd_temp_c is None
        assert state.laser_shot_counter is None

    def test_full_creation(self, scan_id):
        """Create InstrumentState with all fields."""
        state = InstrumentState(
            scan_id=scan_id,
            ccd_temp_c=-30.5,
            pcb_temp_c=25.0,
            laser_prt1_c=22.5,
            laser_prt2_c=23.0,
            laser_shot_counter=1500000,
            laser_misfire_counter=5,
            arc_event_counter=2,
            voltage_1_2v=1.204,
            voltage_3_3v=3.312,
            voltage_5v_dac=5.01,
            voltage_5v_adc=4.99,
            voltage_15v=15.1,
            voltage_neg_15v=-14.95,
            laser_int_time_us=20,
            laser_rep_rate_hz=80,
            laser_current_a=0.15,
            full_telemetry={"key": "value"},
        )
        assert state.ccd_temp_c == -30.5
        assert state.laser_shot_counter == 1500000
        assert state.voltage_1_2v == 1.204

    def test_counter_non_negative(self, scan_id):
        """Counter fields must be >= 0."""
        with pytest.raises(ValidationError):
            InstrumentState(scan_id=scan_id, laser_shot_counter=-1)

    def test_parse_temperature(self):
        """parse_temperature parses Loupe format correctly."""
        assert InstrumentState.parse_temperature("37.200 C") == pytest.approx(37.2)
        assert InstrumentState.parse_temperature("-30.5 C") == pytest.approx(-30.5)
        assert InstrumentState.parse_temperature("N/A") is None
        assert InstrumentState.parse_temperature("None") is None
        assert InstrumentState.parse_temperature("") is None
        assert InstrumentState.parse_temperature(None) is None

    def test_parse_voltage(self):
        """parse_voltage parses Loupe format correctly."""
        assert InstrumentState.parse_voltage("1.204 V") == pytest.approx(1.204)
        assert InstrumentState.parse_voltage("-14.95 V") == pytest.approx(-14.95)
        assert InstrumentState.parse_voltage("N/A") is None
        assert InstrumentState.parse_voltage("") is None

    def test_parse_time_value(self):
        """parse_time_value parses Loupe format correctly."""
        assert InstrumentState.parse_time_value("20 us") == 20
        assert InstrumentState.parse_time_value("80 Hz") == 80
        assert InstrumentState.parse_time_value("N/A") is None
        assert InstrumentState.parse_time_value("") is None

    def test_from_loupe_metadata(self, scan_id):
        """Create InstrumentState from Loupe metadata dict."""
        metadata = {
            "SE_CCD_TEMP_STAT_REG": "-30.5 C",
            "CNDH_PCB_TEMP_STAT_REG": "25.0 C",
            "SE_LASER_PRT1_STAT_REG": "22.5 C",
            "SE_LASER_PRT2_STAT_REG": "23.0 C",
            "laser_shot_counter": "1500000",
            "laser_misfire_counter": "5",
            "arc_event_counter": "2",
            "CNDH_1_2_V_STAT_REG": "1.204 V",
            "CNDH_3_3_V_STAT_REG": "3.312 V",
            "CNDH_5_V_DAC_STAT_REG": "5.01 V",
            "CNDH_5_V_ADC_STAT_REG": "4.99 V",
            "CNDH_15_V_STAT_REG": "15.1 V",
            "CNDH_NEG_15_V_STAT_REG": "-14.95 V",
            "LASER_INT_TIME": "20 us",
            "LASER_REP_RATE": "80 Hz",
        }

        state = InstrumentState.from_loupe_metadata(scan_id, metadata)

        assert state.ccd_temp_c == pytest.approx(-30.5)
        assert state.pcb_temp_c == pytest.approx(25.0)
        assert state.laser_shot_counter == 1500000
        assert state.laser_int_time_us == 20
        assert state.laser_rep_rate_hz == 80
        assert state.voltage_1_2v == pytest.approx(1.204)
        assert state.full_telemetry == metadata

    def test_from_loupe_metadata_with_na_values(self, scan_id):
        """Handle N/A values in Loupe metadata."""
        metadata = {
            "SE_CCD_TEMP_STAT_REG": "N/A",
            "laser_shot_counter": "N/A",
        }

        state = InstrumentState.from_loupe_metadata(scan_id, metadata)

        assert state.ccd_temp_c is None
        assert state.laser_shot_counter is None

    def test_has_uuid(self, scan_id):
        """InstrumentState has auto-generated UUID."""
        state = InstrumentState(scan_id=scan_id)
        assert state.id is not None
        assert isinstance(state.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """InstrumentState can be registered in ModelRegistry."""
        assert hasattr(InstrumentState, "__pydantic_complete__")


class TestCCDConfiguration:
    """Tests for CCDConfiguration model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create CCDConfiguration with minimal fields."""
        config = CCDConfiguration(scan_id=scan_id)
        assert config.scan_id == scan_id
        assert config.region_enable is None
        assert config.gain_2d is None

    def test_full_creation(self, scan_id):
        """Create CCDConfiguration with all fields."""
        config = CCDConfiguration(
            scan_id=scan_id,
            region_enable=7,  # All 3 regions
            gain_2d=1,
            mode_2d=0,
            vert_col1_low=0,
            vert_col1_high=500,
            vert_col2_low=501,
            vert_col2_high=1000,
            vert_col3_low=1001,
            vert_col3_high=2147,
            horz_clock_lim=2148,
        )
        assert config.region_enable == 7
        assert config.vert_col1_high == 500
        assert config.horz_clock_lim == 2148

    def test_non_negative_values(self, scan_id):
        """Integer fields must be >= 0."""
        with pytest.raises(ValidationError):
            CCDConfiguration(scan_id=scan_id, region_enable=-1)

        with pytest.raises(ValidationError):
            CCDConfiguration(scan_id=scan_id, vert_col1_low=-100)

    def test_from_loupe_metadata(self, scan_id):
        """Create CCDConfiguration from Loupe metadata dict."""
        metadata = {
            "REGION_ENABLE": "7",
            "CCD_GAIN_2D": "1",
            "MODE_2D": "0",
            "CCD_VERT_COL1_LOW": "0",
            "CCD_VERT_COL1_HIGH": "500",
            "CCD_VERT_COL2_LOW": "501",
            "CCD_VERT_COL2_HIGH": "1000",
            "CCD_VERT_COL3_LOW": "1001",
            "CCD_VERT_COL3_HIGH": "2147",
            "CCD_HORZ_CLOCK_LIM": "2148",
        }

        config = CCDConfiguration.from_loupe_metadata(scan_id, metadata)

        assert config.region_enable == 7
        assert config.gain_2d == 1
        assert config.vert_col1_high == 500
        assert config.horz_clock_lim == 2148

    def test_from_loupe_metadata_with_na_values(self, scan_id):
        """Handle N/A values in Loupe metadata."""
        metadata = {
            "REGION_ENABLE": "N/A",
            "CCD_GAIN_2D": "1",
        }

        config = CCDConfiguration.from_loupe_metadata(scan_id, metadata)

        assert config.region_enable is None
        assert config.gain_2d == 1

    def test_model_can_be_registered(self):
        """CCDConfiguration can be registered in ModelRegistry."""
        assert hasattr(CCDConfiguration, "__pydantic_complete__")


class TestScannerCalibration:
    """Tests for ScannerCalibration model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create ScannerCalibration with required fields."""
        cal = ScannerCalibration(
            scan_id=scan_id,
            az_scale=0.0285,
            el_scale=0.0285,
            laser_x=824,
            laser_y=600,
        )
        assert cal.az_scale == 0.0285
        assert cal.laser_x == 824
        assert cal.rotation_deg == 0.0  # default

    def test_with_rotation(self, scan_id):
        """Create ScannerCalibration with rotation."""
        cal = ScannerCalibration(
            scan_id=scan_id,
            az_scale=0.0285,
            el_scale=0.0285,
            laser_x=824,
            laser_y=600,
            rotation_deg=1.5,
        )
        assert cal.rotation_deg == 1.5

    def test_laser_position_non_negative(self, scan_id):
        """Laser position must be >= 0."""
        with pytest.raises(ValidationError):
            ScannerCalibration(
                scan_id=scan_id,
                az_scale=0.0285,
                el_scale=0.0285,
                laser_x=-1,
                laser_y=600,
            )

    def test_from_loupe_metadata(self, scan_id):
        """Create ScannerCalibration from Loupe metadata dict."""
        metadata = {
            "az_scale": "0.0285",
            "el_scale": "0.0290",
            "laser_x": "824",
            "laser_y": "600",
            "rotation": "1.5",
        }

        cal = ScannerCalibration.from_loupe_metadata(scan_id, metadata)

        assert cal.az_scale == pytest.approx(0.0285)
        assert cal.el_scale == pytest.approx(0.0290)
        assert cal.laser_x == 824
        assert cal.laser_y == 600
        assert cal.rotation_deg == pytest.approx(1.5)

    def test_from_loupe_metadata_missing_required(self, scan_id):
        """Raise error if required fields missing."""
        metadata = {
            "az_scale": "0.0285",
            # Missing el_scale, laser_x, laser_y
        }

        with pytest.raises(ValueError, match="Missing required"):
            ScannerCalibration.from_loupe_metadata(scan_id, metadata)

    def test_from_loupe_metadata_no_rotation(self, scan_id):
        """Handle missing rotation field."""
        metadata = {
            "az_scale": "0.0285",
            "el_scale": "0.0290",
            "laser_x": "824",
            "laser_y": "600",
            # No rotation
        }

        cal = ScannerCalibration.from_loupe_metadata(scan_id, metadata)

        assert cal.rotation_deg == 0.0

    def test_model_can_be_registered(self):
        """ScannerCalibration can be registered in ModelRegistry."""
        assert hasattr(ScannerCalibration, "__pydantic_complete__")


class TestInstrumentModelIntegration:
    """Integration tests for instrument models."""

    def test_all_instrument_models_from_loupe_metadata(self):
        """Create all instrument models from a single metadata dict."""
        scan_id = uuid.uuid4()

        metadata = {
            # Telemetry
            "SE_CCD_TEMP_STAT_REG": "-30.5 C",
            "laser_shot_counter": "1500000",
            # CCD config
            "REGION_ENABLE": "7",
            "CCD_GAIN_2D": "1",
            # Scanner cal
            "az_scale": "0.0285",
            "el_scale": "0.0285",
            "laser_x": "824",
            "laser_y": "600",
        }

        state = InstrumentState.from_loupe_metadata(scan_id, metadata)
        config = CCDConfiguration.from_loupe_metadata(scan_id, metadata)
        cal = ScannerCalibration.from_loupe_metadata(scan_id, metadata)

        # All reference same scan
        assert state.scan_id == scan_id
        assert config.scan_id == scan_id
        assert cal.scan_id == scan_id

        # Values parsed correctly
        assert state.ccd_temp_c == pytest.approx(-30.5)
        assert config.region_enable == 7
        assert cal.laser_x == 824
