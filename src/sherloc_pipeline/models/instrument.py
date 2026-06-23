"""
Instrument state and configuration models for PHASE.

This module defines models for SHERLOC instrument telemetry and configuration:
- InstrumentState: State-of-health telemetry for a scan
- CCDConfiguration: CCD timing and region configuration
- ScannerCalibration: Scanner coordinate calibration parameters

These models capture the instrument conditions during observations,
which are essential for proper data reduction and quality assessment.

Example:
    >>> from sherloc_pipeline.models.instrument import (
    ...     InstrumentState, CCDConfiguration, ScannerCalibration
    ... )
    >>>
    >>> state = InstrumentState(
    ...     scan_id=scan.id,
    ...     ccd_temp_c=-30.5,
    ...     laser_shot_counter=1500000,
    ... )
"""

from typing import Optional, Dict, Any
import uuid

from pydantic import Field, field_validator, model_validator

from sherloc_pipeline.models.base import (
    IdentifiableModel,
    ModelRegistry,
)


@ModelRegistry.register
class InstrumentState(IdentifiableModel):
    """State-of-health telemetry for a scan.

    InstrumentState captures the instrument conditions during a scan,
    including temperatures, voltages, and laser health metrics. This
    data is essential for quality assessment and long-term trending.

    Temperatures are stored in Celsius, voltages in Volts. The original
    Loupe format stores these as strings with units (e.g., "37.2 C"),
    which should be parsed before storing.

    Attributes:
        scan_id: UUID of parent Scan
        ccd_temp_c: CCD temperature in Celsius
        pcb_temp_c: PCB temperature in Celsius
        laser_prt1_c: Laser PRT1 temperature in Celsius
        laser_prt2_c: Laser PRT2 temperature in Celsius
        laser_shot_counter: Cumulative laser shots
        laser_misfire_counter: Cumulative misfires
        arc_event_counter: Arc event count
        voltage_1_2v: 1.2V rail voltage
        voltage_3_3v: 3.3V rail voltage
        voltage_5v_dac: 5V DAC voltage
        voltage_5v_adc: 5V ADC voltage
        voltage_15v: +15V rail voltage
        voltage_neg_15v: -15V rail voltage
        laser_int_time_us: Laser integration time in microseconds
        laser_rep_rate_hz: Laser repetition rate in Hz
        laser_current_a: Laser current in Amps
        full_telemetry: All SOH fields as JSON

    Example:
        >>> state = InstrumentState(
        ...     scan_id=scan.id,
        ...     ccd_temp_c=-30.5,
        ...     pcb_temp_c=25.0,
        ...     laser_shot_counter=1500000,
        ... )
        >>> state.ccd_temp_c
        -30.5
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )

    # Temperature readings (Celsius)
    ccd_temp_c: Optional[float] = Field(
        default=None,
        description="CCD temperature in Celsius"
    )
    pcb_temp_c: Optional[float] = Field(
        default=None,
        description="PCB temperature in Celsius"
    )
    laser_prt1_c: Optional[float] = Field(
        default=None,
        description="Laser PRT1 temperature in Celsius"
    )
    laser_prt2_c: Optional[float] = Field(
        default=None,
        description="Laser PRT2 temperature in Celsius"
    )

    # Laser health counters
    laser_shot_counter: Optional[int] = Field(
        default=None,
        ge=0,
        description="Cumulative laser shots"
    )
    laser_misfire_counter: Optional[int] = Field(
        default=None,
        ge=0,
        description="Cumulative laser misfires"
    )
    arc_event_counter: Optional[int] = Field(
        default=None,
        ge=0,
        description="Arc event count"
    )

    # Voltage readings (Volts)
    voltage_1_2v: Optional[float] = Field(
        default=None,
        description="1.2V rail voltage"
    )
    voltage_3_3v: Optional[float] = Field(
        default=None,
        description="3.3V rail voltage"
    )
    voltage_5v_dac: Optional[float] = Field(
        default=None,
        description="5V DAC voltage"
    )
    voltage_5v_adc: Optional[float] = Field(
        default=None,
        description="5V ADC voltage"
    )
    voltage_15v: Optional[float] = Field(
        default=None,
        description="+15V rail voltage"
    )
    voltage_neg_15v: Optional[float] = Field(
        default=None,
        description="-15V rail voltage"
    )

    # Laser settings
    laser_int_time_us: Optional[int] = Field(
        default=None,
        ge=0,
        description="Laser integration time in microseconds"
    )
    laser_rep_rate_hz: Optional[int] = Field(
        default=None,
        ge=0,
        description="Laser repetition rate in Hz"
    )
    laser_current_a: Optional[float] = Field(
        default=None,
        ge=0,
        description="Laser current in Amps"
    )

    # Full telemetry as JSON
    full_telemetry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="All SOH fields as JSON"
    )

    @staticmethod
    def parse_temperature(value: str) -> Optional[float]:
        """Parse temperature string from Loupe format.

        Args:
            value: Temperature string like "37.200 C" or "N/A"

        Returns:
            Temperature in Celsius, or None if not parseable

        Example:
            >>> InstrumentState.parse_temperature("37.200 C")
            37.2
            >>> InstrumentState.parse_temperature("N/A")
            None
        """
        if value in ("N/A", "None", "", None):
            return None
        try:
            return float(str(value).split()[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_voltage(value: str) -> Optional[float]:
        """Parse voltage string from Loupe format.

        Args:
            value: Voltage string like "1.204 V" or "N/A"

        Returns:
            Voltage in Volts, or None if not parseable

        Example:
            >>> InstrumentState.parse_voltage("1.204 V")
            1.204
            >>> InstrumentState.parse_voltage("N/A")
            None
        """
        if value in ("N/A", "None", "", None):
            return None
        try:
            return float(str(value).split()[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def parse_time_value(value: str) -> Optional[int]:
        """Parse time value string from Loupe format.

        Args:
            value: Time string like "20 us" or "80 Hz"

        Returns:
            Integer value, or None if not parseable

        Example:
            >>> InstrumentState.parse_time_value("20 us")
            20
            >>> InstrumentState.parse_time_value("80 Hz")
            80
        """
        if value in ("N/A", "None", "", None):
            return None
        try:
            return int(str(value).split()[0])
        except (ValueError, IndexError):
            return None

    @classmethod
    def from_loupe_metadata(
        cls,
        scan_id: uuid.UUID,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> "InstrumentState":
        """Create InstrumentState from Loupe metadata dictionary.

        This convenience constructor parses the raw loupe.csv metadata
        and extracts the relevant telemetry fields.

        Args:
            scan_id: UUID of parent Scan
            metadata: Dictionary from loupe.csv
            **kwargs: Additional fields

        Returns:
            New InstrumentState instance
        """
        return cls(
            scan_id=scan_id,
            ccd_temp_c=cls.parse_temperature(
                metadata.get("SE_CCD_TEMP_STAT_REG")
            ),
            pcb_temp_c=cls.parse_temperature(
                metadata.get("CNDH_PCB_TEMP_STAT_REG")
            ),
            laser_prt1_c=cls.parse_temperature(
                metadata.get("SE_LASER_PRT1_STAT_REG")
            ),
            laser_prt2_c=cls.parse_temperature(
                metadata.get("SE_LASER_PRT2_STAT_REG")
            ),
            laser_shot_counter=cls._safe_int(
                metadata.get("laser_shot_counter")
            ),
            laser_misfire_counter=cls._safe_int(
                metadata.get("laser_misfire_counter")
            ),
            arc_event_counter=cls._safe_int(
                metadata.get("arc_event_counter")
            ),
            voltage_1_2v=cls.parse_voltage(
                metadata.get("CNDH_1_2_V_STAT_REG")
            ),
            voltage_3_3v=cls.parse_voltage(
                metadata.get("CNDH_3_3_V_STAT_REG")
            ),
            voltage_5v_dac=cls.parse_voltage(
                metadata.get("CNDH_5_V_DAC_STAT_REG")
            ),
            voltage_5v_adc=cls.parse_voltage(
                metadata.get("CNDH_5_V_ADC_STAT_REG")
            ),
            voltage_15v=cls.parse_voltage(
                metadata.get("CNDH_15_V_STAT_REG")
            ),
            voltage_neg_15v=cls.parse_voltage(
                metadata.get("CNDH_NEG_15_V_STAT_REG")
            ),
            laser_int_time_us=cls.parse_time_value(
                metadata.get("LASER_INT_TIME")
            ),
            laser_rep_rate_hz=cls.parse_time_value(
                metadata.get("LASER_REP_RATE")
            ),
            full_telemetry=metadata,
            **kwargs,
        )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value in (None, "N/A", "None", ""):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


@ModelRegistry.register
class CCDConfiguration(IdentifiableModel):
    """CCD timing and region configuration.

    CCDConfiguration stores the CCD readout settings used during a scan,
    including region enables, gain settings, and column boundaries.

    Attributes:
        scan_id: UUID of parent Scan
        region_enable: Enabled regions bitmask
        gain_2d: 2D gain setting
        mode_2d: 2D mode setting
        vert_col1_low: Vertical column 1 low boundary
        vert_col1_high: Vertical column 1 high boundary
        vert_col2_low: Vertical column 2 low boundary
        vert_col2_high: Vertical column 2 high boundary
        vert_col3_low: Vertical column 3 low boundary
        vert_col3_high: Vertical column 3 high boundary
        horz_clock_lim: Horizontal clock limit

    Example:
        >>> config = CCDConfiguration(
        ...     scan_id=scan.id,
        ...     region_enable=7,  # All 3 regions enabled
        ...     gain_2d=1,
        ... )
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )
    region_enable: Optional[int] = Field(
        default=None,
        ge=0,
        description="Enabled regions bitmask"
    )
    gain_2d: Optional[int] = Field(
        default=None,
        ge=0,
        description="2D gain setting"
    )
    mode_2d: Optional[int] = Field(
        default=None,
        ge=0,
        description="2D mode setting"
    )
    vert_col1_low: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 1 low boundary"
    )
    vert_col1_high: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 1 high boundary"
    )
    vert_col2_low: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 2 low boundary"
    )
    vert_col2_high: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 2 high boundary"
    )
    vert_col3_low: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 3 low boundary"
    )
    vert_col3_high: Optional[int] = Field(
        default=None,
        ge=0,
        description="Vertical column 3 high boundary"
    )
    horz_clock_lim: Optional[int] = Field(
        default=None,
        ge=0,
        description="Horizontal clock limit"
    )

    @classmethod
    def from_loupe_metadata(
        cls,
        scan_id: uuid.UUID,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> "CCDConfiguration":
        """Create CCDConfiguration from Loupe metadata dictionary.

        Args:
            scan_id: UUID of parent Scan
            metadata: Dictionary from loupe.csv
            **kwargs: Additional fields

        Returns:
            New CCDConfiguration instance
        """
        return cls(
            scan_id=scan_id,
            region_enable=cls._safe_int(metadata.get("REGION_ENABLE")),
            gain_2d=cls._safe_int(metadata.get("CCD_GAIN_2D")),
            mode_2d=cls._safe_int(metadata.get("MODE_2D")),
            vert_col1_low=cls._safe_int(metadata.get("CCD_VERT_COL1_LOW")),
            vert_col1_high=cls._safe_int(metadata.get("CCD_VERT_COL1_HIGH")),
            vert_col2_low=cls._safe_int(metadata.get("CCD_VERT_COL2_LOW")),
            vert_col2_high=cls._safe_int(metadata.get("CCD_VERT_COL2_HIGH")),
            vert_col3_low=cls._safe_int(metadata.get("CCD_VERT_COL3_LOW")),
            vert_col3_high=cls._safe_int(metadata.get("CCD_VERT_COL3_HIGH")),
            horz_clock_lim=cls._safe_int(metadata.get("CCD_HORZ_CLOCK_LIM")),
            **kwargs,
        )

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value in (None, "N/A", "None", ""):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None


@ModelRegistry.register
class ScannerCalibration(IdentifiableModel):
    """Scanner coordinate calibration parameters.

    ScannerCalibration stores the parameters needed to convert scanner
    azimuth/elevation to ACI image pixel coordinates.

    Attributes:
        scan_id: UUID of parent Scan
        az_scale: Azimuth scale factor (pixels per DN)
        el_scale: Elevation scale factor (pixels per DN)
        laser_x: Laser center X pixel on ACI
        laser_y: Laser center Y pixel on ACI
        rotation_deg: Rotation angle in degrees

    Example:
        >>> cal = ScannerCalibration(
        ...     scan_id=scan.id,
        ...     az_scale=0.0285,
        ...     el_scale=0.0285,
        ...     laser_x=824,
        ...     laser_y=600,
        ...     rotation_deg=0.0,
        ... )
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )
    az_scale: float = Field(
        description="Azimuth scale factor (pixels per DN)"
    )
    el_scale: float = Field(
        description="Elevation scale factor (pixels per DN)"
    )
    laser_x: int = Field(
        ge=0,
        description="Laser center X pixel on ACI"
    )
    laser_y: int = Field(
        ge=0,
        description="Laser center Y pixel on ACI"
    )
    rotation_deg: float = Field(
        default=0.0,
        description="Rotation angle in degrees"
    )

    @classmethod
    def from_loupe_metadata(
        cls,
        scan_id: uuid.UUID,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> "ScannerCalibration":
        """Create ScannerCalibration from Loupe metadata dictionary.

        Args:
            scan_id: UUID of parent Scan
            metadata: Dictionary from loupe.csv
            **kwargs: Additional fields

        Returns:
            New ScannerCalibration instance

        Raises:
            ValueError: If required calibration fields are missing
        """
        az_scale = metadata.get("az_scale")
        el_scale = metadata.get("el_scale")
        laser_x = metadata.get("laser_x")
        laser_y = metadata.get("laser_y")
        rotation = metadata.get("rotation", 0.0)

        if any(v is None for v in [az_scale, el_scale, laser_x, laser_y]):
            raise ValueError("Missing required scanner calibration fields")

        return cls(
            scan_id=scan_id,
            az_scale=float(az_scale),
            el_scale=float(el_scale),
            laser_x=int(float(laser_x)),
            laser_y=int(float(laser_y)),
            rotation_deg=float(rotation) if rotation else 0.0,
            **kwargs,
        )
