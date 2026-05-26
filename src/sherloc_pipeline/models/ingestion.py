"""
Ingestion models for parsing Loupe working directories into PHASE domain models.

This module provides Pydantic models for data ingestion - transforming raw Loupe
CSV files into validated domain models. It implements the ingestion strategy
described in docs/schema/UNIFIED_SCHEMA.md.

Loupe Working Directory Structure:
    workspace/
        loupe.csv           - Scan metadata and instrument state
        spatial.csv         - Point positions (az, el, x, y, errors, currents)
        activeSpectra.csv   - Active (laser-illuminated) spectra
        darkSpectra.csv     - Dark frame spectra
        darkSubSpectra.csv  - Dark-subtracted spectra
        photodiodeRaw.csv   - Per-shot photodiode readings
        roi.csv             - Region of interest definitions
        img/                - Context images (ACI, WATSON)

Raw Models:
    RawLoupeMetadata: Parsed loupe.csv key-value pairs
    RawSpatialData: Parsed spatial.csv with point positions
    RawPhotodiodeData: Parsed photodiode statistics per point
    RawROI: Parsed roi.csv region definitions
    RawSpectraFile: Parsed spectra CSV (active/dark/darkSub)

Factory Methods:
    Each domain model gains a .from_loupe_working() class method that
    takes raw data and returns a validated domain model instance.

Example:
    >>> from pathlib import Path
    >>> from sherloc_pipeline.models.ingestion import LoupeWorkspaceParser
    >>>
    >>> parser = LoupeWorkspaceParser(Path("/data/loupe/sol_0921/detail_1/..."))
    >>> scan, points, spectra = parser.parse()
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator

from sherloc_pipeline.models.base import PHASEBaseModel
from sherloc_pipeline.models.spectra import (
    CoordinateFrame,
    DataSource,
    SpectralRegion,
    SpectrumType,
    ProcessingLevel,
    Sol,
    Scan,
    ScanPoint,
    Spectrum,
    classify_scan_class,
)
from sherloc_pipeline.models.instrument import (
    InstrumentState,
    CCDConfiguration,
    ScannerCalibration,
)
from sherloc_pipeline.models.context import (
    ContextImage,
    RegionOfInterest,
    ImageType,
)


# -----------------------------------------------------------------------------
# Raw Data Models - represent CSV file structures before normalization
# -----------------------------------------------------------------------------


class RawLoupeMetadata(PHASEBaseModel):
    """Raw metadata from loupe.csv file.

    loupe.csv is a key-value CSV with two columns: key and value.
    This model captures all fields before type conversion.

    Attributes:
        original_data_file: Original scan identifier (SCLK-based)
        human_readable_workspace: Target/workspace name
        n_spectra: Number of measurement points as string
        n_channels: Number of CCD channels as string
        laser_wavelength: Laser wavelength with units
        shots_per_spec: Shots per spectrum as string
        az_scale: Azimuth scale factor as string
        el_scale: Elevation scale factor as string
        laser_x: Laser center X pixel as string
        laser_y: Laser center Y pixel as string
        rotation: Rotation angle as string
        specProcessingApplied: Processing code or "None"
        raw_fields: All fields as raw key-value dict for full_telemetry

    Example:
        >>> metadata = RawLoupeMetadata.from_csv_path(Path("loupe.csv"))
        >>> metadata.original_data_file
        'SrlcSpecSpecSohRaw_0748731411-51550-1'
    """

    # Allow extra fields since loupe.csv has many telemetry values
    model_config = {"extra": "allow"}

    original_data_file: str = Field(description="Original scan identifier")
    human_readable_workspace: str = Field(description="Target/workspace name")
    n_spectra: str = Field(description="Number of measurement points")
    n_channels: str = Field(description="Number of CCD channels")
    laser_wavelength: str = Field(description="Laser wavelength with units")
    shots_per_spec: str = Field(description="Shots per spectrum")
    az_scale: str = Field(description="Azimuth scale factor")
    el_scale: str = Field(description="Elevation scale factor")
    laser_x: str = Field(description="Laser center X pixel")
    laser_y: str = Field(description="Laser center Y pixel")
    rotation: str = Field(description="Rotation angle")
    specProcessingApplied: str = Field(
        default="None", description="Processing code"
    )
    raw_fields: Dict[str, str] = Field(
        default_factory=dict, description="All raw fields"
    )

    @classmethod
    def from_csv_path(cls, csv_path: Path) -> "RawLoupeMetadata":
        """Parse loupe.csv from file path.

        Args:
            csv_path: Path to loupe.csv file

        Returns:
            Parsed RawLoupeMetadata instance

        Raises:
            FileNotFoundError: If csv_path does not exist
            ValueError: If required fields are missing
        """
        raw_fields = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key, value = row[0].strip(), row[1].strip()
                    raw_fields[key] = value

        return cls(
            original_data_file=raw_fields.get("original_data_file", ""),
            human_readable_workspace=raw_fields.get(
                "human_readable_workspace", ""
            ),
            n_spectra=raw_fields.get("n_spectra", "0"),
            n_channels=raw_fields.get("n_channels", "2148"),
            laser_wavelength=raw_fields.get("laser_wavelength", "248.6"),
            shots_per_spec=raw_fields.get("shots_per_spec", "0"),
            az_scale=raw_fields.get("az_scale", "0"),
            el_scale=raw_fields.get("el_scale", "0"),
            laser_x=raw_fields.get("laser_x", "0"),
            laser_y=raw_fields.get("laser_y", "0"),
            rotation=raw_fields.get("rotation", "0"),
            specProcessingApplied=raw_fields.get("specProcessingApplied", "None"),
            raw_fields=raw_fields,
        )

    def extract_sclk(self) -> int:
        """Extract spacecraft clock from original_data_file.

        Pattern: SrlcSpecSpecSohRaw_SCLK-XXXXX-X

        Returns:
            Spacecraft clock as integer

        Raises:
            ValueError: If SCLK cannot be extracted
        """
        # Pattern: SrlcSpecSpecSohRaw_0748731411-51550-1
        parts = self.original_data_file.split("_")
        if len(parts) >= 2:
            sclk_part = parts[1].split("-")[0]
            try:
                return int(sclk_part)
            except ValueError:
                pass
        raise ValueError(
            f"Cannot extract SCLK from: {self.original_data_file}"
        )

    def to_scan(
        self,
        sol_number: int,
        source_path: Optional[str] = None,
    ) -> Scan:
        """Convert to Scan domain model.

        Args:
            sol_number: Sol number for this scan
            source_path: Optional path to workspace directory

        Returns:
            Validated Scan instance
        """
        return Scan(
            sol_number=sol_number,
            scan_name=self.human_readable_workspace,
            scan_id=self.original_data_file,
            sclk_start=self.extract_sclk(),
            n_points=int(self.n_spectra),
            n_channels=int(self.n_channels),
            shots_per_point=int(self.shots_per_spec),
            laser_wavelength_nm=float(self.laser_wavelength),
            processing_applied=(
                self.specProcessingApplied
                if self.specProcessingApplied != "None"
                else None
            ),
            source_path=source_path,
            loupe_metadata=self.raw_fields,
            data_source=DataSource.LOUPE,
        )

    def to_instrument_state(self, scan_id: uuid.UUID) -> InstrumentState:
        """Convert to InstrumentState domain model.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            Validated InstrumentState instance
        """
        return InstrumentState.from_loupe_metadata(scan_id, self.raw_fields)

    def to_ccd_configuration(self, scan_id: uuid.UUID) -> CCDConfiguration:
        """Convert to CCDConfiguration domain model.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            Validated CCDConfiguration instance
        """
        return CCDConfiguration.from_loupe_metadata(scan_id, self.raw_fields)

    def to_scanner_calibration(
        self, scan_id: uuid.UUID
    ) -> Optional[ScannerCalibration]:
        """Convert to ScannerCalibration domain model.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            Validated ScannerCalibration instance, or None if fields missing
        """
        try:
            return ScannerCalibration.from_loupe_metadata(
                scan_id, self.raw_fields
            )
        except ValueError:
            return None


class RawSpatialPoint(PHASEBaseModel):
    """Single point from spatial.csv.

    Attributes:
        point_index: 0-based index in the scan
        az: Azimuth in DN (or None if not in file)
        el: Elevation in DN (or None if not in file)
        x: X pixel coordinate (or None)
        y: Y pixel coordinate (or None)
        az_err: Azimuth error (or None)
        el_err: Elevation error (or None)
        sum_current: Sum current reading (or None)
        diff_current: Difference current reading (or None)
    """

    point_index: int = Field(ge=0, description="0-based point index")
    az: Optional[int] = Field(default=None, description="Azimuth in DN")
    el: Optional[int] = Field(default=None, description="Elevation in DN")
    x: Optional[float] = Field(default=None, description="X pixel coordinate")
    y: Optional[float] = Field(default=None, description="Y pixel coordinate")
    az_err: Optional[int] = Field(default=None, description="Azimuth error")
    el_err: Optional[int] = Field(default=None, description="Elevation error")
    sum_current: Optional[int] = Field(
        default=None, description="Sum current reading"
    )
    diff_current: Optional[int] = Field(
        default=None, description="Difference current reading"
    )

    def to_scan_point(self, scan_id: uuid.UUID) -> ScanPoint:
        """Convert to ScanPoint domain model.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            Validated ScanPoint instance
        """
        return ScanPoint(
            scan_id=scan_id,
            point_index=self.point_index,
            azimuth_dn=self.az,
            elevation_dn=self.el,
            x_pixel=self.x,
            y_pixel=self.y,
            azimuth_error=float(self.az_err) if self.az_err is not None else None,
            elevation_error=float(self.el_err) if self.el_err is not None else None,
            coordinate_frame=CoordinateFrame.SCANNER_WORKSPACE,
        )


class RawSpatialData(PHASEBaseModel):
    """Parsed spatial.csv data for all points.

    spatial.csv has a multi-section format:
        az,el          # Header for section 1
        1041,726       # Point 0
        994,503        # Point 1
        ...
        x,y            # Header for section 2
        0.518,0.503    # Point 0
        ...
        az_err,el_err  # Header for section 3
        ...

    This model parses all sections and merges by point index.

    Attributes:
        points: List of RawSpatialPoint with merged data
    """

    points: List[RawSpatialPoint] = Field(
        default_factory=list, description="All points with spatial data"
    )

    @classmethod
    def from_csv_path(cls, csv_path: Path) -> "RawSpatialData":
        """Parse spatial.csv from file path.

        Args:
            csv_path: Path to spatial.csv file

        Returns:
            Parsed RawSpatialData instance
        """
        # Read all lines
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Parse sections
        sections: Dict[str, List[List[str]]] = {}
        current_header: Optional[str] = None
        current_data: List[List[str]] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a header line (contains letters)
            if any(c.isalpha() for c in line) and "," in line:
                # Save previous section
                if current_header is not None:
                    sections[current_header] = current_data
                # Start new section
                current_header = line
                current_data = []
            else:
                # Data line
                parts = line.split(",")
                current_data.append(parts)

        # Save last section
        if current_header is not None:
            sections[current_header] = current_data

        # Determine number of points from the largest section
        n_points = max(len(data) for data in sections.values()) if sections else 0

        # Build points by merging sections
        points = []
        for i in range(n_points):
            point_data = {"point_index": i}

            # Extract az,el
            if "az,el" in sections and i < len(sections["az,el"]):
                row = sections["az,el"][i]
                if len(row) >= 2:
                    point_data["az"] = _safe_int(row[0])
                    point_data["el"] = _safe_int(row[1])

            # Extract x,y
            if "x,y" in sections and i < len(sections["x,y"]):
                row = sections["x,y"][i]
                if len(row) >= 2:
                    point_data["x"] = _safe_float(row[0])
                    point_data["y"] = _safe_float(row[1])

            # Extract az_err,el_err
            if "az_err,el_err" in sections and i < len(sections["az_err,el_err"]):
                row = sections["az_err,el_err"][i]
                if len(row) >= 2:
                    point_data["az_err"] = _safe_int(row[0])
                    point_data["el_err"] = _safe_int(row[1])

            # Extract sum_current,diff_current
            if "sum_current,diff_current" in sections and i < len(
                sections["sum_current,diff_current"]
            ):
                row = sections["sum_current,diff_current"][i]
                if len(row) >= 2:
                    point_data["sum_current"] = _safe_int(row[0])
                    point_data["diff_current"] = _safe_int(row[1])

            points.append(RawSpatialPoint(**point_data))

        return cls(points=points)

    def to_scan_points(self, scan_id: uuid.UUID) -> List[ScanPoint]:
        """Convert all points to ScanPoint domain models.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            List of validated ScanPoint instances
        """
        return [point.to_scan_point(scan_id) for point in self.points]


class RawPhotodiodeStats(PHASEBaseModel):
    """Photodiode statistics for a single scan point.

    Computed from photodiodeRaw.csv which has one row per point,
    columns are shot_number_0 through shot_number_N.

    Attributes:
        point_index: 0-based point index
        mean: Mean photodiode reading
        std: Standard deviation of readings
        min_value: Minimum reading
        max_value: Maximum reading
    """

    point_index: int = Field(ge=0, description="0-based point index")
    mean: float = Field(description="Mean photodiode reading")
    std: float = Field(ge=0, description="Standard deviation")
    min_value: float = Field(description="Minimum reading")
    max_value: float = Field(description="Maximum reading")


class RawPhotodiodeData(PHASEBaseModel):
    """Parsed photodiode statistics from photodiodeRaw.csv.

    Attributes:
        stats: List of RawPhotodiodeStats per point
    """

    stats: List[RawPhotodiodeStats] = Field(
        default_factory=list, description="Photodiode stats per point"
    )

    @classmethod
    def from_csv_path(cls, csv_path: Path) -> "RawPhotodiodeData":
        """Parse photodiodeRaw.csv and compute statistics.

        Args:
            csv_path: Path to photodiodeRaw.csv file

        Returns:
            Parsed RawPhotodiodeData instance with computed stats
        """
        import numpy as np

        stats = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)

            for point_idx, row in enumerate(reader):
                if not row:
                    continue

                # Parse all values to floats
                values = []
                for v in row:
                    try:
                        values.append(float(v))
                    except ValueError:
                        continue

                if values:
                    arr = np.array(values)
                    stats.append(
                        RawPhotodiodeStats(
                            point_index=point_idx,
                            mean=float(np.mean(arr)),
                            std=float(np.std(arr)),
                            min_value=float(np.min(arr)),
                            max_value=float(np.max(arr)),
                        )
                    )

        return cls(stats=stats)

    def update_scan_points(
        self, scan_points: List[ScanPoint]
    ) -> List[ScanPoint]:
        """Update ScanPoints with photodiode statistics.

        Args:
            scan_points: List of ScanPoint instances to update

        Returns:
            Updated list (modifies in place and returns)
        """
        # Build lookup by point_index
        stats_by_idx = {s.point_index: s for s in self.stats}

        for point in scan_points:
            if point.point_index in stats_by_idx:
                s = stats_by_idx[point.point_index]
                point.photodiode_mean = s.mean
                point.photodiode_std = s.std

        return scan_points


class RawROI(PHASEBaseModel):
    """Raw region of interest from roi.csv.

    roi.csv format:
        ROI_Name
        #COLORHEX
        point_index_0
        point_index_1
        ...
        ENDROI
        (repeat for next ROI)

    Attributes:
        name: Display name for the ROI
        color: Color string (hex format)
        point_indices: List of point indices in this ROI
    """

    name: str = Field(min_length=1, description="ROI display name")
    color: str = Field(description="Color code (hex format)")
    point_indices: List[int] = Field(
        min_length=1, description="Point indices in this ROI"
    )

    def to_region_of_interest(self, scan_id: uuid.UUID) -> RegionOfInterest:
        """Convert to RegionOfInterest domain model.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            Validated RegionOfInterest instance
        """
        return RegionOfInterest.from_loupe_roi(
            scan_id=scan_id,
            name=self.name,
            color=self.color,
            points=self.point_indices,
        )


class RawROIData(PHASEBaseModel):
    """Parsed roi.csv file containing all ROIs.

    Attributes:
        rois: List of RawROI instances
    """

    rois: List[RawROI] = Field(
        default_factory=list, description="All ROIs from file"
    )

    @classmethod
    def from_csv_path(cls, csv_path: Path) -> "RawROIData":
        """Parse roi.csv from file path.

        Args:
            csv_path: Path to roi.csv file

        Returns:
            Parsed RawROIData instance
        """
        rois = []

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check if this is start of an ROI (name line)
            if line and not line.startswith("#") and line != "ENDROI":
                name = line
                i += 1

                # Next line should be color
                if i < len(lines):
                    color = lines[i].strip()
                    i += 1

                    # Collect point indices until ENDROI
                    point_indices = []
                    while i < len(lines):
                        idx_line = lines[i].strip()
                        if idx_line == "ENDROI":
                            i += 1
                            break
                        if idx_line:
                            try:
                                point_indices.append(int(idx_line))
                            except ValueError:
                                pass
                        i += 1

                    if point_indices:
                        rois.append(
                            RawROI(
                                name=name,
                                color=color,
                                point_indices=point_indices,
                            )
                        )
            else:
                i += 1

        return cls(rois=rois)

    def to_regions_of_interest(
        self, scan_id: uuid.UUID
    ) -> List[RegionOfInterest]:
        """Convert all ROIs to RegionOfInterest domain models.

        Args:
            scan_id: UUID of parent Scan

        Returns:
            List of validated RegionOfInterest instances
        """
        return [roi.to_region_of_interest(scan_id) for roi in self.rois]


class RawSpectraFile(PHASEBaseModel):
    """Parsed spectra CSV file (activeSpectra, darkSpectra, darkSubSpectra).

    Loupe spectra CSVs have a multi-section structure:
        - Row 0: R1 header, Rows 1-N: R1 data
        - Row N+1: R2 header, Rows N+2 to 2N+1: R2 data
        - Row 2N+2: R3 header, Rows 2N+3 to 3N+2: R3 data

    Each section has 2148 columns (CCD channels) and n_spectra rows.

    IMPORTANT: Prior to Sprint 4, only R1 data was read but mislabeled as R123.
    The database contains R1 Raman spectra only - R2/R3 fluorescence was never ingested.

    Attributes:
        file_type: Type of spectra file (active, dark, dark_subtracted)
        n_points: Number of points (rows)
        n_channels: Number of channels (columns)
        channel_names: List of column names
        section: Which spectral region this data represents (R1, R2, R3)
    """

    file_type: SpectrumType = Field(description="Type of spectra file")
    n_points: int = Field(ge=0, description="Number of points")
    n_channels: int = Field(ge=0, description="Number of channels")
    channel_names: List[str] = Field(
        default_factory=list, description="Column header names"
    )
    section: str = Field(default="R1", description="Spectral section (R1, R2, R3)")
    # Note: data is stored separately, not in Pydantic model for memory efficiency

    @classmethod
    def parse_header(
        cls, csv_path: Path, file_type: SpectrumType
    ) -> "RawSpectraFile":
        """Parse just the header to get channel count and names.

        Args:
            csv_path: Path to spectra CSV file
            file_type: Type of spectra (active, dark, dark_subtracted)

        Returns:
            RawSpectraFile with header info (no data loaded)
        """
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])

        return cls(
            file_type=file_type,
            n_points=0,  # Not yet counted
            n_channels=len(header),
            channel_names=header,
            section="R1",
        )

    @classmethod
    def _count_section_rows(cls, csv_path: Path) -> int:
        """Count the number of data rows in the first section (R1).

        Loupe CSVs have 3 sections, each with a header row followed by n_spectra
        data rows. We detect section boundaries by finding header rows (rows
        containing non-numeric first values).

        Args:
            csv_path: Path to spectra CSV

        Returns:
            Number of spectra (data rows per section)
        """
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip R1 header
            next(reader, [])

            n_spectra = 0
            for row in reader:
                if not row:
                    continue
                # Check if this is a header row (first value is not purely numeric)
                first_val = row[0].strip() if row else ""
                # Header rows have column names like "R2_Channel0" or contain letters
                if first_val and any(c.isalpha() for c in first_val):
                    # Hit R2 header, stop counting
                    break
                n_spectra += 1

        return n_spectra

    @classmethod
    def from_csv_path(
        cls, csv_path: Path, file_type: SpectrumType, section: str = "R1"
    ) -> Tuple["RawSpectraFile", List[List[float]]]:
        """Parse spectra CSV, reading specified section (R1, R2, or R3).

        Loupe CSVs have 3 sections with identical structure:
        - R1: skiprows=0, nrows=n_spectra (Raman region, 250-282 nm)
        - R2: skiprows=1+n_spectra, nrows=n_spectra (Fluorescence 1, 282-337.8 nm)
        - R3: skiprows=2*(1+n_spectra), nrows=n_spectra (Fluorescence 2, 337.8-357.4 nm)

        Args:
            csv_path: Path to spectra CSV file
            file_type: Type of spectra (active, dark, dark_subtracted)
            section: Which section to read ("R1", "R2", or "R3"). Default "R1".

        Returns:
            Tuple of (RawSpectraFile metadata, 2D list of intensities)

        Note:
            The data read is always the full 2148 channels. For ML training on R1,
            apply wavelength filtering (250-282 nm) to extract the meaningful 523
            channels and discard detector noise from unilluminated regions.
        """
        # First pass: count spectra in R1 section to know section boundaries
        n_spectra = cls._count_section_rows(csv_path)

        # Calculate skiprows based on section
        section_map = {
            "R1": 0,
            "R2": 1 + n_spectra,
            "R3": 2 * (1 + n_spectra),
        }
        skiprows = section_map.get(section.upper(), 0)

        # Read the specified section
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Skip to section start
            for _ in range(skiprows):
                next(reader, None)

            # Read header
            header = next(reader, [])

            # Read data rows up to n_spectra
            data = []
            rows_read = 0
            for row in reader:
                if rows_read >= n_spectra:
                    break
                if not row:
                    continue
                # Check if we hit next section header
                first_val = row[0].strip() if row else ""
                if first_val and any(c.isalpha() for c in first_val):
                    break
                values = [_safe_float(v) or 0.0 for v in row]
                data.append(values)
                rows_read += 1

        return (
            cls(
                file_type=file_type,
                n_points=len(data),
                n_channels=len(header),
                channel_names=header,
                section=section.upper(),
            ),
            data,
        )

    def iter_spectra(
        self,
        data: List[List[float]],
        scan_point_ids: List[uuid.UUID],
        processing_level: ProcessingLevel = ProcessingLevel.RAW,
    ) -> Iterator[Spectrum]:
        """Iterate over spectra, yielding Spectrum domain models.

        Uses the section field to determine the correct SpectralRegion label.

        Args:
            data: 2D intensity data from from_csv_path
            scan_point_ids: List of ScanPoint UUIDs in order
            processing_level: Processing level to assign

        Yields:
            Spectrum instances for each point
        """
        # Map section to SpectralRegion
        region_map = {
            "R1": SpectralRegion.R1,
            "R2": SpectralRegion.R2,
            "R3": SpectralRegion.R3,
        }
        region = region_map.get(self.section, SpectralRegion.R1)

        for point_idx, intensities in enumerate(data):
            if point_idx >= len(scan_point_ids):
                break

            yield Spectrum.from_values(
                scan_point_id=scan_point_ids[point_idx],
                region=region,
                spectrum_type=self.file_type,
                processing_level=processing_level,
                intensity_values=intensities,
            )


# -----------------------------------------------------------------------------
# Session File Parser - for .lpe files
# -----------------------------------------------------------------------------


class LoupeSessionEntry(PHASEBaseModel):
    """Entry from a Loupe session file (.lpe).

    Attributes:
        workspace_dict_name: Internal workspace identifier
        workspace_human_readable_name: Display name
        soff_path: Relative path to soff.xml file
    """

    workspace_dict_name: str = Field(description="Internal workspace ID")
    workspace_human_readable_name: str = Field(description="Display name")
    soff_path: str = Field(description="Relative path to soff.xml")


class LoupeSessionFile(PHASEBaseModel):
    """Parsed Loupe session file (.lpe).

    Session files list all workspaces for a sol.

    Attributes:
        sol_number: Sol number extracted from filename/path
        entries: List of workspace entries
    """

    sol_number: int = Field(ge=0, description="Sol number")
    entries: List[LoupeSessionEntry] = Field(
        default_factory=list, description="Workspace entries"
    )

    @classmethod
    def from_path(cls, session_path: Path) -> "LoupeSessionFile":
        """Parse .lpe session file.

        Args:
            session_path: Path to .lpe file

        Returns:
            Parsed LoupeSessionFile instance
        """
        # Extract sol number from parent directory name
        # Pattern: sol_XXXX or sol_XXXX_a/b
        sol_dir = session_path.parent.name
        match = re.match(r"sol_(\d+)", sol_dir)
        if match:
            sol_number = int(match.group(1))
        else:
            sol_number = 0

        entries = []
        with open(session_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header

            for row in reader:
                if len(row) >= 3:
                    entries.append(
                        LoupeSessionEntry(
                            workspace_dict_name=row[0].strip(),
                            workspace_human_readable_name=row[1].strip(),
                            soff_path=row[2].strip(),
                        )
                    )

        return cls(sol_number=sol_number, entries=entries)

    def workspace_paths(self, base_dir: Path) -> List[Path]:
        """Get full paths to workspace directories.

        Args:
            base_dir: Sol directory containing the .lpe file

        Returns:
            List of absolute paths to Loupe_working directories
        """
        paths = []
        for entry in self.entries:
            # soff_path is like: detail_1/SrlcSpec..._Loupe_working/soff.xml
            # We want the Loupe_working directory
            soff_full = base_dir / entry.soff_path
            if soff_full.exists():
                paths.append(soff_full.parent)
            else:
                # Try to find the workspace directory
                parts = entry.soff_path.split("/")
                if len(parts) >= 2:
                    workspace_dir = base_dir / parts[0] / parts[1]
                    if workspace_dir.exists():
                        paths.append(workspace_dir)

        return paths


# -----------------------------------------------------------------------------
# Main Workspace Parser
# -----------------------------------------------------------------------------


class LoupeWorkspaceParser:
    """Parser for a Loupe working directory.

    Orchestrates parsing of all CSV files in a workspace and produces
    validated domain model instances.

    Attributes:
        workspace_path: Path to the Loupe_working directory
        sol_number: Sol number (extracted or provided)

    Example:
        >>> parser = LoupeWorkspaceParser(
        ...     Path("/data/loupe/sol_0921/detail_1/SrlcSpec..._Loupe_working"),
        ...     sol_number=921,
        ... )
        >>> result = parser.parse()
        >>> result.scan.scan_name
        'detail_1'
    """

    def __init__(
        self,
        workspace_path: Path,
        sol_number: Optional[int] = None,
    ):
        """Initialize parser.

        Args:
            workspace_path: Path to Loupe_working directory
            sol_number: Sol number (extracted from path if not provided)
        """
        self.workspace_path = Path(workspace_path)

        if sol_number is not None:
            self.sol_number = sol_number
        else:
            # Try to extract from path
            self.sol_number = self._extract_sol_number()

    def _extract_sol_number(self) -> int:
        """Extract sol number from workspace path."""
        for part in self.workspace_path.parts:
            match = re.match(r"sol_(\d+)", part)
            if match:
                return int(match.group(1))
        return 0

    def parse(self) -> "LoupeWorkspaceResult":
        """Parse all files in the workspace.

        Returns:
            LoupeWorkspaceResult containing all parsed domain models
        """
        # Parse loupe.csv (required)
        loupe_path = self.workspace_path / "loupe.csv"
        if not loupe_path.exists():
            raise FileNotFoundError(f"loupe.csv not found in {self.workspace_path}")

        raw_metadata = RawLoupeMetadata.from_csv_path(loupe_path)

        # Create Scan
        scan = raw_metadata.to_scan(
            sol_number=self.sol_number,
            source_path=str(self.workspace_path),
        )

        # Loupe CSV sometimes truncates composite workspace names (e.g. "detail_"
        # instead of "detail_all"). If the directory name classifies as composite
        # but the CSV-derived name doesn't, prefer the directory name.
        dir_name = self.workspace_path.parent.name
        if (
            classify_scan_class(dir_name) == "composite"
            and classify_scan_class(scan.scan_name) != "composite"
        ):
            scan = scan.model_copy(update={"scan_name": dir_name})

        # Create instrument/configuration models
        instrument_state = raw_metadata.to_instrument_state(scan.id)
        ccd_config = raw_metadata.to_ccd_configuration(scan.id)
        scanner_cal = raw_metadata.to_scanner_calibration(scan.id)

        # Parse spatial.csv
        scan_points = []
        spatial_path = self.workspace_path / "spatial.csv"
        if spatial_path.exists():
            raw_spatial = RawSpatialData.from_csv_path(spatial_path)
            scan_points = raw_spatial.to_scan_points(scan.id)

            # Update with photodiode data if available
            photodiode_path = self.workspace_path / "photodiodeRaw.csv"
            if photodiode_path.exists():
                raw_photodiode = RawPhotodiodeData.from_csv_path(photodiode_path)
                raw_photodiode.update_scan_points(scan_points)

        # Parse ROIs
        rois = []
        roi_path = self.workspace_path / "roi.csv"
        if roi_path.exists():
            raw_rois = RawROIData.from_csv_path(roi_path)
            rois = raw_rois.to_regions_of_interest(scan.id)

        # Parse spectra (lazy - just return paths and metadata for now)
        spectra_files = {}
        for fname, stype in [
            ("activeSpectra.csv", SpectrumType.ACTIVE),
            ("darkSpectra.csv", SpectrumType.DARK),
            ("darkSubSpectra.csv", SpectrumType.DARK_SUBTRACTED),
        ]:
            fpath = self.workspace_path / fname
            if fpath.exists():
                spectra_files[stype] = fpath

        # Find context images
        context_images = []
        img_dir = self.workspace_path / "img"
        if img_dir.exists():
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".img"):
                    # Determine image type from filename
                    img_type = ImageType.ACI  # Default
                    if "WATSON" in img_file.name.upper():
                        img_type = ImageType.WATSON

                    context_images.append(
                        ContextImage(
                            scan_id=scan.id,
                            image_type=img_type,
                            file_path=str(img_file),
                        )
                    )

        return LoupeWorkspaceResult(
            scan=scan,
            instrument_state=instrument_state,
            ccd_configuration=ccd_config,
            scanner_calibration=scanner_cal,
            scan_points=scan_points,
            regions_of_interest=rois,
            context_images=context_images,
            spectra_files=spectra_files,
            workspace_path=self.workspace_path,
        )

    def parse_spectra(
        self,
        spectrum_type: SpectrumType,
        scan_point_ids: List[uuid.UUID],
        processing_level: ProcessingLevel = ProcessingLevel.RAW,
        section: str = "R1",
    ) -> List[Spectrum]:
        """Parse spectra of a specific type from a specific section.

        Args:
            spectrum_type: Which spectra file to parse
            scan_point_ids: List of ScanPoint UUIDs in order
            processing_level: Processing level to assign
            section: Spectral section to read ("R1", "R2", or "R3"). Default "R1".
                     R1 = Raman region (250-282 nm)
                     R2 = Fluorescence region 1 (282-337.8 nm)
                     R3 = Fluorescence region 2 (337.8-357.4 nm)

        Returns:
            List of Spectrum instances with correct region labeling
        """
        fname_map = {
            SpectrumType.ACTIVE: "activeSpectra.csv",
            SpectrumType.DARK: "darkSpectra.csv",
            SpectrumType.DARK_SUBTRACTED: "darkSubSpectra.csv",
        }

        fpath = self.workspace_path / fname_map[spectrum_type]
        if not fpath.exists():
            return []

        raw_file, data = RawSpectraFile.from_csv_path(fpath, spectrum_type, section)
        return list(
            raw_file.iter_spectra(data, scan_point_ids, processing_level)
        )


class LoupeWorkspaceResult(PHASEBaseModel):
    """Result of parsing a Loupe workspace directory.

    Contains all domain models extracted from the workspace.

    Attributes:
        scan: The Scan domain model
        instrument_state: Instrument telemetry
        ccd_configuration: CCD settings
        scanner_calibration: Scanner calibration (may be None)
        scan_points: List of ScanPoint instances
        regions_of_interest: List of ROI instances
        context_images: List of ContextImage instances
        spectra_files: Dict mapping SpectrumType to file paths
        workspace_path: Original workspace directory path
    """

    model_config = {"arbitrary_types_allowed": True}

    scan: Scan = Field(description="Parsed Scan model")
    instrument_state: InstrumentState = Field(
        description="Instrument telemetry"
    )
    ccd_configuration: CCDConfiguration = Field(description="CCD settings")
    scanner_calibration: Optional[ScannerCalibration] = Field(
        default=None, description="Scanner calibration"
    )
    scan_points: List[ScanPoint] = Field(
        default_factory=list, description="Scan points"
    )
    regions_of_interest: List[RegionOfInterest] = Field(
        default_factory=list, description="ROIs"
    )
    context_images: List[ContextImage] = Field(
        default_factory=list, description="Context images"
    )
    spectra_files: Dict[SpectrumType, Path] = Field(
        default_factory=dict, description="Paths to spectra files"
    )
    workspace_path: Path = Field(description="Original workspace path")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _safe_int(value: str) -> Optional[int]:
    """Safely convert string to int."""
    if not value or value in ("N/A", "None", ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _safe_float(value: str) -> Optional[float]:
    """Safely convert string to float."""
    if not value or value in ("N/A", "None", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_target_from_lpe(sol_dir: Path) -> Optional[str]:
    """Extract target name from .lpe filename in a sol directory.

    Convention: Sol_XXXX_<target>.lpe  ->  "Target Name"
    Example: Sol_1771_ Djuma.lpe -> "Djuma"
             Sol_0921_Amherst_Point.lpe -> "Amherst Point"
             Sol_1677_arm_stowed_dark.lpe -> "arm stowed dark"

    Args:
        sol_dir: Path to sol_XXXX directory containing a .lpe file.

    Returns:
        Target name with underscores replaced by spaces, or None if
        no .lpe file found or name cannot be parsed.
    """
    lpe_files = list(sol_dir.glob("*.lpe"))
    if not lpe_files:
        return None
    stem = lpe_files[0].stem  # e.g. "Sol_1771_ Djuma"
    match = re.match(r"Sol_\d+_\s*(.*)", stem)
    if not match or not match.group(1).strip():
        return None
    return match.group(1).replace("_", " ").strip()


def extract_sol_from_path(path: Path) -> Optional[int]:
    """Extract sol number from a path containing sol_XXXX directory.

    Args:
        path: Any path that may contain a sol directory

    Returns:
        Sol number if found, else None
    """
    for part in path.parts:
        match = re.match(r"sol_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def discover_workspaces(sol_dir: Path) -> List[Path]:
    """Discover all Loupe working directories in a sol directory.

    Args:
        sol_dir: Path to sol_XXXX directory

    Returns:
        List of paths to Loupe_working directories
    """
    workspaces = []

    # First try to find .lpe session file
    lpe_files = list(sol_dir.glob("*.lpe"))
    if lpe_files:
        session = LoupeSessionFile.from_path(lpe_files[0])
        return session.workspace_paths(sol_dir)

    # Otherwise, search for Loupe_working directories
    for subdir in sol_dir.iterdir():
        if subdir.is_dir():
            for working_dir in subdir.glob("*_Loupe_working"):
                if (working_dir / "loupe.csv").exists():
                    workspaces.append(working_dir)

    return workspaces
