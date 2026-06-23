"""
PDS4 parsers for SHERLOC products.

Parses PDS4 XML labels into PDSObservationMetadata Pydantic models,
RRS/RCS CSV files into spectral product metadata with numpy data arrays,
RMO CSV files into position/band intensity products, and simpler formats
(RLI, RCC, RLS) into their respective Pydantic product models.

Parsers:
    PDSLabelParser: XML label → PDSObservationMetadata (20+ metadata fields)
    PDSSpectralParser: RRS/RCS CSV → ParsedSpectralCSV (metadata + spectra arrays)
    PDSRMOParser: RMO CSV → PDSPositionProduct (positions + band intensities)
    PDSPhotodiodeParser: RLI CSV → PDSPhotodiodeProduct (avg photodiode per shot)
    PDSCalibrationParser: RCC CSV → PDSCalibrationProduct (multi-sol AlGaN drift)
    PDSCrossRefParser: RLS CSV → PDSCrossRefProduct (shot-spectrum-image mapping)

Usage:
    >>> from sherloc_pipeline.core.pds_parsers import PDSLabelParser, PDSSpectralParser
    >>> label_parser = PDSLabelParser()
    >>> metadata = label_parser.parse_label("./pds/sol_0921/.../file.xml")
    >>> spectral_parser = PDSSpectralParser()
    >>> result = spectral_parser.parse("./pds/sol_0921/.../file.csv")
    >>> result.product.n_spectra
    100
    >>> from sherloc_pipeline.core.pds_parsers import PDSRMOParser
    >>> rmo_parser = PDSRMOParser()
    >>> rmo = rmo_parser.parse("./pds/sol_0921/.../rmo.csv")
    >>> rmo.n_positions
    100
    >>> from sherloc_pipeline.core.pds_parsers import PDSPhotodiodeParser
    >>> pd_parser = PDSPhotodiodeParser()
    >>> rli = pd_parser.parse("./pds/sol_0921/.../rli.csv")
    >>> rli.n_shots
    100
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from sherloc_pipeline.models.pds import (
    CORE_PRODUCT_TYPES,
    PDSCalibrationProduct,
    PDSCalibrationRecord,
    PDSCrossRefProduct,
    PDSCrossRefRecord,
    PDSObservationMetadata,
    PDSPhotodiodeProduct,
    PDSPositionProduct,
    PDSPositionRecord,
    PDSProductId,
    PDSProductType,
    PDSSpectralProduct,
    PDSWavelengthRegion,
    PDS_EXPECTED_CHANNELS,
)

from sherloc_pipeline.core.utils import require_file

# PDS4 XML namespace URIs
_NS: Dict[str, str] = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "mars2020": "http://pds.nasa.gov/pds4/mission/mars2020/v1",
    "msn_surface": "http://pds.nasa.gov/pds4/msn_surface/v1",
    "geom": "http://pds.nasa.gov/pds4/geom/v1",
    "proc": "http://pds.nasa.gov/pds4/proc/v1",
}


class PDSLabelParser:
    """Parse PDS4 XML labels into PDSObservationMetadata.

    Extracts metadata fields defined in spec s7 from PDS4 XML labels,
    handling the multi-namespace structure of SHERLOC processed products.

    The parser extracts:
        - Identification: logical_identifier, version_id, title
        - Time: start/stop_date_time, local_mean_solar_time, solar_longitude
        - Mission: sol_number, spacecraft_clock_start/stop, mission_phase_name
        - Command: sequence_id (from Surface_Mission_Information)
        - Geometry: SITE/DRIVE from Motion_Counter, RSM azimuth/elevation
        - Processing: software_version, product_completion_status
        - Table structure: n_spectra, n_channels (from File_Area_Observational)

    Example:
        >>> parser = PDSLabelParser()
        >>> meta = parser.parse_label(Path("./pds/sol_0921/.../file.xml"))
        >>> meta.sol_number
        921
        >>> meta.spacecraft_clock_start
        '748731411.515'
    """

    def parse_label(self, xml_path: Path) -> PDSObservationMetadata:
        """Parse a PDS4 XML label file into PDSObservationMetadata.

        Args:
            xml_path: Path to the XML label file.

        Returns:
            Validated PDSObservationMetadata instance.

        Raises:
            FileNotFoundError: If the XML file doesn't exist.
            ET.ParseError: If the XML is malformed.
            ValueError: If required fields are missing from the label.
        """
        xml_path = Path(xml_path)
        require_file(xml_path, "XML label not found")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Parse product ID from the XML filename
        product_id = PDSProductId.from_filename(xml_path.name)

        # Extract fields from each XML section
        fields = self._extract_identification(root)
        fields.update(self._extract_time_coordinates(root))
        fields.update(self._extract_mission_area(root))
        fields.update(self._extract_surface_mission(root))
        fields.update(self._extract_geometry(root))
        fields.update(self._extract_processing(root))
        fields.update(self._extract_table_structure(root))

        fields["product_id"] = product_id

        return PDSObservationMetadata(**fields)

    def _extract_identification(self, root: ET.Element) -> Dict[str, Any]:
        """Extract fields from Identification_Area."""
        id_area = root.find("pds:Identification_Area", _NS)
        if id_area is None:
            raise ValueError("Missing Identification_Area in XML label")

        logical_id = _text(id_area, "pds:logical_identifier")
        if logical_id is None:
            raise ValueError("Missing logical_identifier in XML label")

        version_id = _text(id_area, "pds:version_id")
        if version_id is None:
            raise ValueError("Missing version_id in XML label")

        return {
            "logical_identifier": logical_id,
            "version_id": version_id,
            "title": _text(id_area, "pds:title"),
        }

    def _extract_time_coordinates(self, root: ET.Element) -> Dict[str, Any]:
        """Extract fields from Observation_Area/Time_Coordinates."""
        time_coords = root.find(
            "pds:Observation_Area/pds:Time_Coordinates", _NS
        )
        if time_coords is None:
            return {}

        solar_long = _text(time_coords, "pds:solar_longitude")

        return {
            "start_date_time": _text(time_coords, "pds:start_date_time"),
            "stop_date_time": _text(time_coords, "pds:stop_date_time"),
            "local_mean_solar_time": _text(
                time_coords, "pds:local_mean_solar_time"
            ),
            "solar_longitude": float(solar_long) if solar_long else None,
        }

    def _extract_mission_area(self, root: ET.Element) -> Dict[str, Any]:
        """Extract fields from Mission_Area (mars2020 namespace)."""
        obs_info = root.find(
            ".//mars2020:Observation_Information", _NS
        )
        if obs_info is None:
            return {}

        sol_text = _text(obs_info, "mars2020:sol_number")
        if sol_text is None:
            raise ValueError("Missing sol_number in XML label")

        return {
            "sol_number": int(sol_text),
            "spacecraft_clock_start": _text(
                obs_info, "mars2020:spacecraft_clock_start"
            ),
            "spacecraft_clock_stop": _text(
                obs_info, "mars2020:spacecraft_clock_stop"
            ),
            "mission_phase_name": _text(
                obs_info, "mars2020:mission_phase_name"
            ),
        }

    def _extract_surface_mission(self, root: ET.Element) -> Dict[str, Any]:
        """Extract fields from Surface_Mission_Information (msn_surface ns)."""
        result: Dict[str, Any] = {}

        # sequence_id from Command_Execution
        cmd_exec = root.find(
            ".//msn_surface:Command_Execution", _NS
        )
        if cmd_exec is not None:
            result["sequence_id"] = _text(
                cmd_exec, "msn_surface:sequence_id"
            )

        # product_completion_status from Telemetry
        telemetry = root.find(".//msn_surface:Telemetry", _NS)
        if telemetry is not None:
            result["product_completion_status"] = _text(
                telemetry, "msn_surface:product_completion_status"
            )

        return result

    def _extract_geometry(self, root: ET.Element) -> Dict[str, Any]:
        """Extract SITE/DRIVE from Motion_Counter and RSM angles."""
        result: Dict[str, Any] = {}

        # SITE and DRIVE from geom:Motion_Counter
        motion_counter = root.find(".//geom:Motion_Counter", _NS)
        if motion_counter is not None:
            for index_el in motion_counter.findall(
                "geom:Motion_Counter_Index", _NS
            ):
                index_id = _text(index_el, "geom:index_id")
                index_val = _text(index_el, "geom:index_value_number")
                if index_id == "SITE" and index_val is not None:
                    result["site"] = int(index_val)
                elif index_id == "DRIVE" and index_val is not None:
                    result["drive"] = int(index_val)

        # RSM azimuth/elevation from Articulation_Device_Parameters
        # Find the RSM device specifically
        for device_params in root.findall(
            ".//geom:Articulation_Device_Parameters", _NS
        ):
            device_id = _text(device_params, "geom:device_id")
            if device_id != "RSM":
                continue

            for angle_index in device_params.findall(
                ".//geom:Device_Angle_Index", _NS
            ):
                idx_id = _text(angle_index, "geom:index_id")
                idx_val = _text(angle_index, "geom:index_value_angle")
                if idx_val is None:
                    continue
                val = float(idx_val)
                # Skip sentinel values (1e+30 = not available)
                if val > 1e20:
                    continue
                if idx_id == "AZIMUTH FINAL-RESOLVER":
                    result["rsm_azimuth_rad"] = val
                elif idx_id == "ELEVATION FINAL-RESOLVER":
                    result["rsm_elevation_rad"] = val
            break  # Only process RSM device

        return result

    def _extract_processing(self, root: ET.Element) -> Dict[str, Any]:
        """Extract software version from Processing_Information."""
        program_version = root.find(
            ".//proc:Software_Program/proc:program_version", _NS
        )
        if program_version is not None and program_version.text:
            return {"software_version": program_version.text.strip()}
        return {}

    def _extract_table_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract n_spectra and n_channels from File_Area_Observational.

        For RRS/RCS: n_spectra from the first spectral region table's records,
        n_channels from the Group_Field_Delimited repetitions.
        """
        result: Dict[str, Any] = {}

        tables = root.findall(
            "pds:File_Area_Observational/pds:Table_Delimited", _NS
        )
        if not tables:
            return result

        # Find the first spectral region table (contains "REGION" in name)
        # or fall back to the first table
        spectral_table = None
        for table in tables:
            name = _text(table, "pds:name")
            if name and "REGION" in name.upper():
                spectral_table = table
                break

        if spectral_table is None:
            # Use first table for non-spectral products
            spectral_table = tables[0]

        records = _text(spectral_table, "pds:records")
        if records is not None:
            result["n_spectra"] = int(records)

        # n_channels from Group_Field_Delimited repetitions
        group = spectral_table.find(
            ".//pds:Group_Field_Delimited", _NS
        )
        if group is not None:
            reps = _text(group, "pds:repetitions")
            if reps is not None:
                result["n_channels"] = int(reps)

        return result


def _text(parent: ET.Element, tag: str) -> Optional[str]:
    """Get stripped text content of a child element, or None if missing."""
    el = parent.find(tag, _NS)
    if el is not None and el.text:
        return el.text.strip()
    return None


# --- RRS/RCS CSV section header patterns ---

_WAVELENGTH_HEADER = "WAVELENGTH (NM):"
_LASER_NORMALIZED_PREFIX = "LASER-NORMALIZED_SPECTRA:_REGION_"
_PROCESS_DATA_PREFIX = "PROCESS_DATA_SPECTRUM_REGION_"

# Known section headers for matching
_SECTION_PREFIXES = (
    _WAVELENGTH_HEADER,
    _LASER_NORMALIZED_PREFIX,
    _PROCESS_DATA_PREFIX,
)


class PDSZpzProductError(ValueError):
    """Raised when a CSV file is a zpz summary product that should be skipped.

    zpz products are detected via three reinforcing signals (spec s8):
    1. Filename contains 'zpz' in the processing suffix field
    2. Section headers use PROCESS_DATA format instead of LASER-NORMALIZED
    3. Exactly 2 spectra per region (summary, not point-by-point)

    If any signal is present, the product is rejected.
    """
    pass


@dataclass
class ParsedSpectralCSV:
    """Result of parsing an RRS/RCS CSV file.

    Separates metadata (in the Pydantic model) from spectral data arrays
    (in numpy) for memory efficiency — the Pydantic model can be stored
    without carrying large float arrays.

    Attributes:
        product: PDSSpectralProduct metadata (product_id, n_spectra, wavelengths, etc.)
        spectra: Dict mapping region name to (n_spectra, 2148) float64 arrays.
                 Keys are "R1", "R2", "R3".
    """
    product: PDSSpectralProduct
    spectra: Dict[str, np.ndarray]


class PDSSpectralParser:
    """Parse RRS/RCS CSV files into spectral product metadata and data arrays.

    RRS (Reduced Raw Spectra) and RCS (Reduced Calibrated Spectra) CSV files
    share the same 4-section structure (spec s7):

    1. WAVELENGTH (NM): — 1 row of 2148 wavelength calibration values
    2. LASER-NORMALIZED_SPECTRA:_REGION_1 — N rows × 2148 channels (R1 Raman)
    3. LASER-NORMALIZED_SPECTRA:_REGION_2 — N rows × 2148 channels (R2 Fluor.)
    4. LASER-NORMALIZED_SPECTRA:_REGION_3 — N rows × 2148 channels (R3 Fluor.)

    Each section has a header line followed by a column-name line (skipped),
    then data rows of comma-separated floats.

    zpz summary products use PROCESS_DATA_SPECTRUM_REGION_N headers and contain
    exactly 2 spectra per region. These are detected and rejected via
    PDSZpzProductError (spec s8).

    Example:
        >>> parser = PDSSpectralParser()
        >>> result = parser.parse(Path("./pds/sol_0921/.../rrs.csv"))
        >>> result.product.n_spectra
        100
        >>> result.spectra["R1"].shape
        (100, 2148)
    """

    def parse(self, csv_path: Path) -> ParsedSpectralCSV:
        """Parse an RRS/RCS CSV file into metadata and spectral data.

        Args:
            csv_path: Path to the RRS or RCS CSV file.

        Returns:
            ParsedSpectralCSV with product metadata and region data arrays.

        Raises:
            PDSZpzProductError: If the file is a zpz summary product.
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the file structure is invalid.
        """
        csv_path = Path(csv_path)
        require_file(csv_path, "CSV file not found")

        product_id = PDSProductId.from_filename(csv_path.name)

        # zpz signal tracking
        zpz_signals: List[str] = []

        # Signal 1: filename contains 'zpz' in processing suffix
        if "zpz" in product_id.middle.lower():
            zpz_signals.append(
                "filename contains 'zpz' in processing suffix"
            )

        # Read and split into sections
        sections = self._split_sections(csv_path)

        # Validate wavelength section present
        if _WAVELENGTH_HEADER not in sections:
            raise ValueError(
                f"Missing WAVELENGTH (NM): section in {csv_path.name}"
            )

        # Parse wavelength calibration array
        wavelengths = self._parse_wavelengths(sections[_WAVELENGTH_HEADER])

        # Parse spectral regions
        regions_present: List[str] = []
        spectra: Dict[str, np.ndarray] = {}

        for region_num in (1, 2, 3):
            region_name = f"R{region_num}"
            laser_key = f"{_LASER_NORMALIZED_PREFIX}{region_num}"
            process_key = f"{_PROCESS_DATA_PREFIX}{region_num}"

            if laser_key in sections:
                data = self._parse_data_rows(
                    sections[laser_key], region_name, csv_path.name
                )
                regions_present.append(region_name)
                spectra[region_name] = data
            elif process_key in sections:
                # Signal 2: PROCESS_DATA header format
                zpz_signals.append(
                    f"section header uses PROCESS_DATA format ({region_name})"
                )
                data = self._parse_data_rows(
                    sections[process_key], region_name, csv_path.name
                )
                regions_present.append(region_name)
                spectra[region_name] = data

        if not regions_present:
            raise ValueError(
                f"No spectral region sections found in {csv_path.name}"
            )

        # Signal 3: exactly 2 spectra per region
        if spectra and all(arr.shape[0] == 2 for arr in spectra.values()):
            zpz_signals.append("exactly 2 spectra per region")

        # Reject if any zpz signal present
        if zpz_signals:
            raise PDSZpzProductError(
                f"zpz summary product detected in {csv_path.name}: "
                + "; ".join(zpz_signals)
            )

        # Validate consistent spectra counts across regions
        counts = {name: arr.shape[0] for name, arr in spectra.items()}
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"Inconsistent spectra counts across regions in "
                f"{csv_path.name}: {counts}"
            )
        n_spectra = next(iter(unique_counts))

        product = PDSSpectralProduct(
            product_id=product_id,
            n_spectra=n_spectra,
            n_channels=len(wavelengths),
            wavelengths=wavelengths,
            regions_present=regions_present,
            source_path=str(csv_path),
        )

        return ParsedSpectralCSV(product=product, spectra=spectra)

    def _split_sections(
        self, csv_path: Path
    ) -> Dict[str, List[str]]:
        """Split a CSV file into named sections.

        Each section starts with a header line (matched by known prefixes),
        followed by a column-name line (skipped), then data lines.

        Returns:
            Dict mapping section header text to list of data lines.
        """
        sections: Dict[str, List[str]] = {}
        current_header: Optional[str] = None
        skip_column_header = False

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check if this line is a section header
                if self._is_section_header(line):
                    current_header = line
                    sections[current_header] = []
                    skip_column_header = True
                    continue

                # Skip the column-name line after each section header
                if skip_column_header:
                    skip_column_header = False
                    continue

                # Accumulate data lines
                if current_header is not None:
                    sections[current_header].append(line)

        return sections

    @staticmethod
    def _is_section_header(line: str) -> bool:
        """Check if a line is a known section header."""
        return any(line.startswith(prefix) for prefix in _SECTION_PREFIXES)

    @staticmethod
    def _parse_wavelengths(lines: List[str]) -> List[float]:
        """Parse the wavelength calibration row.

        The WAVELENGTH section has exactly 1 data row of 2148 float values.

        Returns:
            List of 2148 wavelength values in nm.

        Raises:
            ValueError: If the section is empty or has wrong channel count.
        """
        if not lines:
            raise ValueError("Empty WAVELENGTH section")
        values = [float(v) for v in lines[0].split(",")]
        if len(values) != PDS_EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {PDS_EXPECTED_CHANNELS} wavelength values, "
                f"got {len(values)}"
            )
        return values

    @staticmethod
    def _parse_data_rows(
        lines: List[str], region_name: str, filename: str
    ) -> np.ndarray:
        """Parse spectral data rows into a numpy array.

        Args:
            lines: Data lines (comma-separated floats).
            region_name: Region label for error messages (e.g., "R1").
            filename: Source filename for error messages.

        Returns:
            numpy array of shape (n_spectra, 2148), dtype float64.

        Raises:
            ValueError: If section is empty or has wrong channel count.
        """
        if not lines:
            raise ValueError(
                f"Empty {region_name} section in {filename}"
            )

        data = []
        for line in lines:
            row = line.split(",")
            data.append([float(v) for v in row])

        arr = np.array(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != PDS_EXPECTED_CHANNELS:
            raise ValueError(
                f"Expected {PDS_EXPECTED_CHANNELS} channels per row in "
                f"{region_name} of {filename}, got shape {arr.shape}"
            )
        return arr


# --- RMO CSV section headers ---

_RMO_POSITIONS_HEADER = "LASER_SHOT_POSITIONS"
_RMO_WAVELENGTH_HEADER = "WAVELENGTH_REGIONS"
_RMO_INTENSITY_HEADER = "SPECTRAL_INTENSITY"

_RMO_SECTION_HEADERS = (
    _RMO_POSITIONS_HEADER,
    _RMO_WAVELENGTH_HEADER,
    _RMO_INTENSITY_HEADER,
)

# Expected number of wavelength bands in RMO products.
_RMO_N_BANDS = 6


class PDSRMOParser:
    """Parse RMO (Reduced Measurement Overview) CSV files.

    RMO CSV files contain three tables (spec s7):

    1. LASER_SHOT_POSITIONS — Image_name, Position_index, x, y
       ACI pixel coordinates per laser shot position.
    2. WAVELENGTH_REGIONS — Column_index, Wavelength_start, Wavelength_stop
       6 band definitions used for integrated intensity.
    3. SPECTRAL_INTENSITY — Position_index, Spectral_intensity_0..5
       Per-position band intensities (one row per unique position).

    Survey scans reference 2 ACI images and have 2× position rows
    (same Position_index with different Image_name). De-duplication
    by Position_index keeps the first occurrence. All unique Image_name
    values are collected before de-duplication for ACI association.

    Example:
        >>> parser = PDSRMOParser()
        >>> rmo = parser.parse(Path("./pds/sol_0921/.../rmo.csv"))
        >>> rmo.n_positions
        100
        >>> len(rmo.image_names)  # 1 for detail, 2 for survey
        1
    """

    def parse(self, csv_path: Path) -> PDSPositionProduct:
        """Parse an RMO CSV file into a PDSPositionProduct.

        Args:
            csv_path: Path to the RMO CSV file.

        Returns:
            PDSPositionProduct with de-duplicated positions, band definitions,
            intensities, and all unique image names.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If required sections are missing or malformed.
        """
        csv_path = Path(csv_path)
        require_file(csv_path, "CSV file not found")

        product_id = PDSProductId.from_filename(csv_path.name)

        # Reject zpz summary products (same logic as spectral parser)
        if "zpz" in product_id.middle.lower():
            raise PDSZpzProductError(
                f"zpz summary product detected in {csv_path.name}: "
                "filename contains 'zpz' in processing suffix"
            )

        sections = self._split_sections(csv_path)

        # Validate all 3 required sections present
        for header in _RMO_SECTION_HEADERS:
            if header not in sections:
                raise ValueError(
                    f"Missing {header} section in {csv_path.name}"
                )

        # Parse each section
        image_names, positions = self._parse_positions(
            sections[_RMO_POSITIONS_HEADER], csv_path.name
        )
        wavelength_regions = self._parse_wavelength_regions(
            sections[_RMO_WAVELENGTH_HEADER], csv_path.name
        )
        band_intensities = self._parse_intensities(
            sections[_RMO_INTENSITY_HEADER], csv_path.name
        )

        # Validate position count matches intensity count
        if len(positions) != len(band_intensities):
            raise ValueError(
                f"Position count ({len(positions)}) != intensity count "
                f"({len(band_intensities)}) in {csv_path.name}"
            )

        return PDSPositionProduct(
            product_id=product_id,
            positions=positions,
            wavelength_regions=wavelength_regions,
            band_intensities=band_intensities,
            image_names=image_names,
            source_path=str(csv_path),
        )

    def _split_sections(
        self, csv_path: Path
    ) -> Dict[str, List[str]]:
        """Split an RMO CSV file into named sections.

        Each section starts with a header line (one of the 3 known RMO
        section names), followed by a column-name line (skipped), then
        data lines.

        Returns:
            Dict mapping section header to list of data lines.
        """
        sections: Dict[str, List[str]] = {}
        current_header: Optional[str] = None
        skip_column_header = False

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line in _RMO_SECTION_HEADERS:
                    current_header = line
                    sections[current_header] = []
                    skip_column_header = True
                    continue

                if skip_column_header:
                    skip_column_header = False
                    continue

                if current_header is not None:
                    sections[current_header].append(line)

        return sections

    @staticmethod
    def _parse_positions(
        lines: List[str], filename: str
    ) -> tuple:
        """Parse LASER_SHOT_POSITIONS rows.

        Collects all unique Image_name values (in order of first appearance)
        before de-duplicating by Position_index (keeps first occurrence).

        Args:
            lines: Data lines from the positions section.
            filename: Source filename for error messages.

        Returns:
            Tuple of (image_names: List[str], positions: List[PDSPositionRecord])
            where image_names contains all unique ACI names and positions
            are de-duplicated by Position_index.

        Raises:
            ValueError: If the section is empty or rows are malformed.
        """
        if not lines:
            raise ValueError(
                f"Empty LASER_SHOT_POSITIONS section in {filename}"
            )

        # Collect unique image names (ordered by first appearance)
        seen_images: Dict[str, None] = {}
        # De-duplicate positions by Position_index (keep first)
        seen_indices: Dict[int, PDSPositionRecord] = {}

        for line in lines:
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(
                    f"Expected 4 columns in LASER_SHOT_POSITIONS row, "
                    f"got {len(parts)} in {filename}: {line[:80]}"
                )

            image_name = parts[0]
            position_index = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])

            # Track unique image names in order
            if image_name not in seen_images:
                seen_images[image_name] = None

            # Keep first occurrence per Position_index
            if position_index not in seen_indices:
                seen_indices[position_index] = PDSPositionRecord(
                    image_name=image_name,
                    position_index=position_index,
                    x=x,
                    y=y,
                )

        image_names = list(seen_images.keys())
        # Sort positions by index for deterministic order
        positions = [
            seen_indices[idx] for idx in sorted(seen_indices.keys())
        ]

        return image_names, positions

    @staticmethod
    def _parse_wavelength_regions(
        lines: List[str], filename: str
    ) -> List[PDSWavelengthRegion]:
        """Parse WAVELENGTH_REGIONS rows (6 band definitions).

        Args:
            lines: Data lines from the wavelength regions section.
            filename: Source filename for error messages.

        Returns:
            List of 6 PDSWavelengthRegion instances.

        Raises:
            ValueError: If the section doesn't have exactly 6 rows.
        """
        if len(lines) != _RMO_N_BANDS:
            raise ValueError(
                f"Expected {_RMO_N_BANDS} wavelength regions in "
                f"{filename}, got {len(lines)}"
            )

        regions: List[PDSWavelengthRegion] = []
        for line in lines:
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 columns in WAVELENGTH_REGIONS row, "
                    f"got {len(parts)} in {filename}: {line[:80]}"
                )
            regions.append(PDSWavelengthRegion(
                column_index=int(parts[0]),
                wavelength_start=float(parts[1]),
                wavelength_stop=float(parts[2]),
            ))

        return regions

    @staticmethod
    def _parse_intensities(
        lines: List[str], filename: str
    ) -> List[List[float]]:
        """Parse SPECTRAL_INTENSITY rows (per-position 6-band intensities).

        Args:
            lines: Data lines from the intensity section.
            filename: Source filename for error messages.

        Returns:
            List of [intensity_0..5] lists, one per position, ordered by
            Position_index.

        Raises:
            ValueError: If the section is empty or rows are malformed.
        """
        if not lines:
            raise ValueError(
                f"Empty SPECTRAL_INTENSITY section in {filename}"
            )

        # Parse into (position_index, [6 intensities]) pairs
        indexed: List[tuple] = []
        for line in lines:
            parts = line.split(",")
            # Position_index + 6 intensity columns = 7
            if len(parts) != _RMO_N_BANDS + 1:
                raise ValueError(
                    f"Expected {_RMO_N_BANDS + 1} columns in "
                    f"SPECTRAL_INTENSITY row, got {len(parts)} in "
                    f"{filename}: {line[:80]}"
                )
            pos_idx = int(parts[0])
            intensities = [float(v) for v in parts[1:]]
            indexed.append((pos_idx, intensities))

        # Sort by position index for deterministic order
        indexed.sort(key=lambda t: t[0])

        return [intensities for _, intensities in indexed]


# --- RLI CSV section header ---

_RLI_HEADER = "LASER_PHOTODIODE_INTENSITY_MAP:"


class PDSPhotodiodeParser:
    """Parse RLI (Reduced Laser Intensity) CSV files.

    RLI CSVs contain a single-column table of average photodiode ADC counts,
    one value per laser shot position. The structure is:

    1. Section header: ``LASER_PHOTODIODE_INTENSITY_MAP:``
    2. Column header: ``avg_photodiode`` (skipped)
    3. Data rows: one float per line

    Calibration scans: 1 value. Detail scans: ~100 values.
    Survey scans: ~1296 values. zpz summary observations may have
    sentinel values of -1.0.

    Example:
        >>> parser = PDSPhotodiodeParser()
        >>> rli = parser.parse(Path("./pds/sol_0921/.../rli.csv"))
        >>> rli.n_shots
        100
    """

    def parse(self, csv_path: Path) -> PDSPhotodiodeProduct:
        """Parse an RLI CSV file into a PDSPhotodiodeProduct.

        Args:
            csv_path: Path to the RLI CSV file.

        Returns:
            PDSPhotodiodeProduct with per-shot photodiode intensities.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the file structure is invalid or has no data.
        """
        csv_path = Path(csv_path)
        require_file(csv_path, "CSV file not found")

        product_id = PDSProductId.from_filename(csv_path.name)

        intensities: List[float] = []
        found_header = False
        skip_column_header = False

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == _RLI_HEADER:
                    found_header = True
                    skip_column_header = True
                    continue

                if skip_column_header:
                    skip_column_header = False
                    continue

                if found_header:
                    intensities.append(float(line))

        if not found_header:
            raise ValueError(
                f"Missing {_RLI_HEADER} section in {csv_path.name}"
            )

        if not intensities:
            raise ValueError(
                f"No photodiode data rows in {csv_path.name}"
            )

        return PDSPhotodiodeProduct(
            product_id=product_id,
            intensities=intensities,
            source_path=str(csv_path),
        )


# --- RCC CSV section header ---

_RCC_HEADER = "CALIBRATION_FIT:"


class PDSCalibrationParser:
    """Parse RCC (Reduced Calibrated Compact) CSV files.

    RCC CSVs contain multi-sol calibration drift history with AlGaN 275 nm
    and laser reflection peak fit parameters. The structure is:

    1. Section header: ``CALIBRATION_FIT:``
    2. Column header: ``SOL,SCLK,laser_reflection_peak_location,...`` (skipped)
    3. Data rows: 6 CSV columns per row (typically ~47 records)

    Laser peak values of 0.0 indicate the laser reflection peak was not fit
    for that observation (the AlGaN peak is always present).

    Example:
        >>> parser = PDSCalibrationParser()
        >>> rcc = parser.parse(Path("./pds/sol_0921/.../rcc.csv"))
        >>> rcc.n_records
        47
        >>> rcc.records[-1].sol
        921
    """

    def parse(self, csv_path: Path) -> PDSCalibrationProduct:
        """Parse an RCC CSV file into a PDSCalibrationProduct.

        Args:
            csv_path: Path to the RCC CSV file.

        Returns:
            PDSCalibrationProduct with calibration records ordered by sol.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the file structure is invalid or has no data.
        """
        csv_path = Path(csv_path)
        require_file(csv_path, "CSV file not found")

        product_id = PDSProductId.from_filename(csv_path.name)

        records: List[PDSCalibrationRecord] = []
        found_header = False
        skip_column_header = False

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == _RCC_HEADER:
                    found_header = True
                    skip_column_header = True
                    continue

                if skip_column_header:
                    skip_column_header = False
                    continue

                if found_header:
                    parts = line.split(",")
                    if len(parts) != 6:
                        raise ValueError(
                            f"Expected 6 columns in CALIBRATION_FIT row, "
                            f"got {len(parts)} in {csv_path.name}: "
                            f"{line[:80]}"
                        )
                    records.append(PDSCalibrationRecord(
                        sol=int(parts[0]),
                        sclk=parts[1],
                        laser_peak_nm=float(parts[2]),
                        laser_fwhm_nm=float(parts[3]),
                        algan_peak_nm=float(parts[4]),
                        algan_fwhm_nm=float(parts[5]),
                    ))

        if not found_header:
            raise ValueError(
                f"Missing {_RCC_HEADER} section in {csv_path.name}"
            )

        if not records:
            raise ValueError(
                f"No calibration data rows in {csv_path.name}"
            )

        return PDSCalibrationProduct(
            product_id=product_id,
            records=records,
            source_path=str(csv_path),
        )


# --- RLS CSV section header ---

_RLS_HEADER = "LASER_SHOT_POSITION:"


class PDSCrossRefParser:
    """Parse RLS (Reduced Laser Shot) CSV files.

    RLS CSVs contain a shot-to-spectrum-to-image cross-reference table
    with ACI pixel coordinates. The structure is:

    1. Section header: ``LASER_SHOT_POSITION:``
    2. Column header: ``number,spec_name,image_name,samp,line`` (skipped)
    3. Data rows: 5 CSV columns per row

    Survey scans have 2× rows compared to unique positions (one per ACI
    image), providing full cross-reference without de-duplication.

    Example:
        >>> parser = PDSCrossRefParser()
        >>> rls = parser.parse(Path("./pds/sol_0921/.../rls.csv"))
        >>> rls.n_records
        100
        >>> rls.image_names
        ['SC3_0921_...ECM_...LMJ01.IMG']
    """

    def parse(self, csv_path: Path) -> PDSCrossRefProduct:
        """Parse an RLS CSV file into a PDSCrossRefProduct.

        Args:
            csv_path: Path to the RLS CSV file.

        Returns:
            PDSCrossRefProduct with cross-reference records.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If the file structure is invalid or has no data.
        """
        csv_path = Path(csv_path)
        require_file(csv_path, "CSV file not found")

        product_id = PDSProductId.from_filename(csv_path.name)

        records: List[PDSCrossRefRecord] = []
        found_header = False
        skip_column_header = False

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == _RLS_HEADER:
                    found_header = True
                    skip_column_header = True
                    continue

                if skip_column_header:
                    skip_column_header = False
                    continue

                if found_header:
                    parts = line.split(",")
                    if len(parts) != 5:
                        raise ValueError(
                            f"Expected 5 columns in LASER_SHOT_POSITION "
                            f"row, got {len(parts)} in {csv_path.name}: "
                            f"{line[:80]}"
                        )
                    records.append(PDSCrossRefRecord(
                        number=int(parts[0]),
                        spec_name=parts[1],
                        image_name=parts[2],
                        samp=float(parts[3]),
                        line=float(parts[4]),
                    ))

        if not found_header:
            raise ValueError(
                f"Missing {_RLS_HEADER} section in {csv_path.name}"
            )

        if not records:
            raise ValueError(
                f"No cross-reference data rows in {csv_path.name}"
            )

        return PDSCrossRefProduct(
            product_id=product_id,
            records=records,
            source_path=str(csv_path),
        )


# --- Observation grouping and classification (spec s8) ---

# Calibration sequence codes (spec s8 table)
_CALIBRATION_SEQUENCE_CODES = frozenset({"srlc10000", "srlc16000"})

# Spectra threshold separating detail from survey scans (spec s8 table)
_SURVEY_SPECTRA_THRESHOLD = 200


@dataclass
class PDSObservationGroup:
    """Grouped and classified PDS products from a single observation.

    Products sharing (sol, sclk, obs_id) belong to the same observation (spec s8).
    After grouping, zpz summary products are filtered out, latest versions are
    selected per product type, and the observation is classified.

    Attributes:
        observation_key: Unique key "SSSS_CCCCCCCCCC_NNN" (sol_sclk_obs_id)
        sol: Sol number
        sclk: Spacecraft clock value
        obs_id: 3-digit observation identifier
        sequence_code: SRLC sequence code (e.g., 'srlc10000')
        scan_type: Classified type: 'calibration', 'detail', 'survey', or
                   None if n_spectra unavailable for detail/survey distinction
        products: Dict of product_type string → PDSProductId (highest version).
                  Keys are lowercase type codes (e.g., 'rrs', 'rmo').
        filtered_zpz: List of PDSProductId instances filtered as zpz summaries
    """
    observation_key: str
    sol: int
    sclk: int
    obs_id: str
    sequence_code: str
    scan_type: Optional[str]
    products: Dict[str, PDSProductId]
    filtered_zpz: List[PDSProductId]


class PDSObservationGrouper:
    """Group PDS products by observation and classify them (spec s8).

    Implements the observation grouping pipeline:
    1. Discover CSV products in a sol directory
    2. Group by observation_key (sol, sclk, obs_id)
    3. Filter zpz summary products by filename
    4. Keep latest version per product type
    5. Verify RRS/RCS mutual exclusivity
    6. Classify: calibration (SRLC10000/16000), survey (>200), detail (≤200)

    Example:
        >>> grouper = PDSObservationGrouper()
        >>> groups = grouper.group_sol_directory(
        ...     Path("./pds/sol_0921/data_processed")
        ... )
        >>> len(groups)
        5
        >>> groups[0].scan_type
        'calibration'
    """

    CALIBRATION_CODES = _CALIBRATION_SEQUENCE_CODES
    SURVEY_THRESHOLD = _SURVEY_SPECTRA_THRESHOLD

    def group_sol_directory(
        self,
        data_dir: Path,
        label_parser: Optional[PDSLabelParser] = None,
    ) -> List[PDSObservationGroup]:
        """Discover, group, filter, and classify observations in a directory.

        Full pipeline: discover CSV filenames → group by observation → filter
        zpz → select latest versions → validate exclusivity → classify.

        Args:
            data_dir: Path to sol data directory containing CSV/XML file pairs.
            label_parser: Optional PDSLabelParser to read n_spectra from XML
                         labels for detail/survey classification. If None,
                         non-calibration observations get scan_type=None.

        Returns:
            List of PDSObservationGroup instances sorted by observation_key,
            excluding groups where all products were zpz-filtered.

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
            ValueError: If RRS and RCS both appear in the same observation.
        """
        products = self.discover_csv_products(data_dir)
        obs_groups = self.group_by_observation(products)

        result: List[PDSObservationGroup] = []

        for obs_key in sorted(obs_groups.keys()):
            obs_products = obs_groups[obs_key]

            # Filter zpz summary products
            clean, zpz_filtered = self.filter_zpz(obs_products)
            if not clean:
                continue  # All products were zpz — skip observation

            # Select latest version per product type
            best = self.select_latest_versions(clean)

            # Validate RRS/RCS mutual exclusivity
            self.validate_spectral_exclusivity(best)

            # Get metadata from first product (all share sol/sclk/obs_id)
            first = next(iter(best.values()))

            # Classify observation
            n_spectra = None
            if label_parser is not None:
                n_spectra = self._get_spectra_count(best, data_dir, label_parser)

            scan_type = self.classify(first.sequence_code, n_spectra)

            result.append(PDSObservationGroup(
                observation_key=obs_key,
                sol=first.sol,
                sclk=first.sclk,
                obs_id=first.obs_id,
                sequence_code=first.sequence_code,
                scan_type=scan_type,
                products=best,
                filtered_zpz=zpz_filtered,
            ))

        return result

    @staticmethod
    def discover_csv_products(data_dir: Path) -> List[PDSProductId]:
        """Parse all CSV filenames in a directory into PDSProductId instances.

        Silently skips files that don't match the PDS filename pattern.

        Args:
            data_dir: Directory containing PDS CSV files.

        Returns:
            List of parsed PDSProductId instances, sorted by filename.

        Raises:
            FileNotFoundError: If data_dir doesn't exist.
        """
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        products: List[PDSProductId] = []
        for csv_file in sorted(data_dir.glob("*.csv")):
            try:
                products.append(PDSProductId.from_filename(csv_file.name))
            except ValueError:
                continue  # Skip non-PDS filenames
        return products

    @staticmethod
    def group_by_observation(
        products: List[PDSProductId],
    ) -> Dict[str, List[PDSProductId]]:
        """Group products by observation_key (sol, sclk, obs_id).

        Products sharing the same (sol, sclk, observation_id) belong to
        the same observation (spec s8).

        Args:
            products: List of parsed PDS product identifiers.

        Returns:
            Dict mapping observation_key to list of products in that group.
        """
        groups: Dict[str, List[PDSProductId]] = {}
        for p in products:
            key = p.observation_key
            if key not in groups:
                groups[key] = []
            groups[key].append(p)
        return groups

    @staticmethod
    def filter_zpz(
        products: List[PDSProductId],
    ) -> tuple:
        """Separate zpz summary products from normal products.

        zpz products are identified by the presence of 'zpz' in the
        filename's middle section (processing suffix). This is signal 1
        from spec s8 — the other two signals (header format, spectra count)
        are checked at CSV parse time by the individual parsers.

        Args:
            products: Products from a single observation group.

        Returns:
            Tuple of (clean_products, zpz_products). Both lists preserve
            input ordering.
        """
        clean: List[PDSProductId] = []
        zpz: List[PDSProductId] = []
        for p in products:
            if "zpz" in p.middle.lower():
                zpz.append(p)
            else:
                clean.append(p)
        return clean, zpz

    @staticmethod
    def select_latest_versions(
        products: List[PDSProductId],
    ) -> Dict[str, PDSProductId]:
        """Keep the highest-version product per product type.

        Within an observation group, multiple versions of the same product
        type may exist (e.g., version 01 and 02 of RRS). This selects the
        highest version number for each product type (spec s8 / s6).

        Args:
            products: Products from a single observation (post zpz-filter).

        Returns:
            Dict mapping product_type string (e.g., 'rrs') to the
            highest-version PDSProductId for that type.
        """
        best: Dict[str, PDSProductId] = {}
        for p in products:
            # product_type is stored as string due to PHASEBaseModel use_enum_values
            pt = p.product_type if isinstance(p.product_type, str) else p.product_type.value
            if pt not in best or p.version > best[pt].version:
                best[pt] = p
        return best

    @staticmethod
    def classify(
        sequence_code: str,
        n_spectra: Optional[int] = None,
    ) -> Optional[str]:
        """Classify observation type from sequence code and spectra count.

        Classification rules (spec s8):
        - calibration: SRLC10000 or SRLC16000
        - detail: other SRLC codes with ≤200 spectra
        - survey: other SRLC codes with >200 spectra

        Args:
            sequence_code: SRLC sequence code (e.g., 'srlc10000').
            n_spectra: Number of spectra in the spectral product.
                      Required for detail/survey distinction.

        Returns:
            'calibration', 'detail', 'survey', or None if n_spectra
            is needed but not provided.
        """
        if sequence_code.lower() in _CALIBRATION_SEQUENCE_CODES:
            return "calibration"

        if n_spectra is None:
            return None

        if n_spectra > _SURVEY_SPECTRA_THRESHOLD:
            return "survey"
        return "detail"

    @staticmethod
    def validate_spectral_exclusivity(
        products: Dict[str, PDSProductId],
    ) -> None:
        """Verify RRS and RCS are mutually exclusive per observation (spec s8).

        An observation contains either RRS (Mars surface spectra) or RCS
        (calibration spectra), never both.

        Args:
            products: Dict of product_type → PDSProductId for one observation.

        Raises:
            ValueError: If both RRS and RCS are present.
        """
        has_rrs = "rrs" in products
        has_rcs = "rcs" in products
        if has_rrs and has_rcs:
            rrs = products["rrs"]
            rcs = products["rcs"]
            raise ValueError(
                f"RRS and RCS are mutually exclusive but both found in "
                f"observation {rrs.observation_key}: "
                f"{rrs.filename}, {rcs.filename}"
            )

    @staticmethod
    def _get_spectra_count(
        products: Dict[str, PDSProductId],
        data_dir: Path,
        label_parser: "PDSLabelParser",
    ) -> Optional[int]:
        """Get n_spectra from the spectral product's XML label.

        Tries RRS first (Mars surface), then RCS (calibration).

        Args:
            products: Product dict for one observation.
            data_dir: Directory containing the XML label files.
            label_parser: Parser instance for reading XML labels.

        Returns:
            Number of spectra, or None if no spectral XML label found.
        """
        for pt in ("rrs", "rcs"):
            if pt in products:
                xml_name = products[pt].xml_filename
                xml_path = data_dir / xml_name
                if xml_path.exists():
                    metadata = label_parser.parse_label(xml_path)
                    return metadata.n_spectra
        return None
