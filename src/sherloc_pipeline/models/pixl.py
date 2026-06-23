"""
Pydantic models for PIXL Pixlise data.

This module provides Pydantic models for PIXL XRF data exported from Pixlise,
enabling cross-instrument analysis between PIXL (XRF) and SHERLOC (Raman/UV).

The Pixlise exports contain AutoQuant-PDS quantification results with:
- 16 oxide wt% abundances (Na2O, MgO, Al2O3, SiO2, etc.)
- 16 corresponding uncertainties
- 16 intensity values (unique to Pixlise, not in PDS)
- Instrument metadata (chisq, livetime, total_counts)
- Beam location coordinates per image

Classes:
    PixliseTarget: PIXL observation target from Pixlise export
    PixliseQuantPoint: Single quantified point from AutoQuant-PDS
    PixliseBeamLocation: Beam location for a point on a specific image
    PixliseImage: Context image from Pixlise export
    PixliseExportParser: Parser for Pixlise zip exports

Example:
    >>> from sherloc_pipeline.models.pixl import PixliseTarget, PixliseQuantPoint
    >>>
    >>> target = PixliseTarget(
    ...     name="Bubenka",
    ...     rtt=59999001,
    ...     n_points=1723,
    ...     piquant_version="3.2.17",
    ... )
"""

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import re
import zipfile
import io
import csv

import numpy as np
from pydantic import Field, field_validator, computed_field

from sherloc_pipeline.models.base import (
    PHASEBaseModel,
    TimestampedModel,
    IdentifiableModel,
)


# Oxide column names in canonical order (matches Pixlise AutoQuant schema)
OXIDE_NAMES = [
    "Na2O",
    "MgO",
    "Al2O3",
    "SiO2",
    "P2O5",
    "SO3",
    "Cl",
    "K2O",
    "CaO",
    "TiO2",
    "Cr2O3",
    "MnO",
    "FeO-T",
    "NiO",
    "ZnO",
    "Br",
]

# Number of oxide columns
N_OXIDES = 16


class PixliseImageType(str, Enum):
    """Types of images in Pixlise exports."""

    PCW = "PCW"  # Context camera
    DTU_MSA = "DTU_MSA"  # MSA decorrelation
    SIF = "SIF"  # SIF images
    CSC_ACI = "CSC_ACI"  # ACI composites
    UNKNOWN = "unknown"


class PixliseTarget(IdentifiableModel):
    """PIXL observation target from Pixlise export.

    Represents a single PIXL observation target with metadata extracted
    from the Pixlise export directory and files.

    Attributes:
        name: Target name (e.g., "Bubenka")
        name_normalized: Lowercase, stripped target name for matching
        rtt: Round-trip time identifier from filename
        sol: Mars sol number (extracted if available)
        n_points: Number of quantified points
        export_date: Date of Pixlise export
        piquant_version: PIQUANT version used for quantification
        detector_config: PIQUANT detector configuration
        source_zip: Path to source zip file

    Example:
        >>> target = PixliseTarget(
        ...     name="Bubenka",
        ...     rtt=59999001,
        ...     n_points=1723,
        ...     piquant_version="3.2.17",
        ... )
    """

    name: str = Field(description="Target name from directory")
    name_normalized: str = Field(
        default="",
        description="Normalized target name (lowercase, stripped)",
    )
    rtt: int = Field(description="Round-trip time identifier")
    sol: Optional[int] = Field(
        default=None,
        description="Mars sol number (if extractable)",
    )
    n_points: int = Field(ge=0, description="Number of quantified points")
    export_date: Optional[date] = Field(
        default=None,
        description="Date of Pixlise export",
    )
    piquant_version: str = Field(
        default="",
        description="PIQUANT version (e.g., '3.2.17')",
    )
    detector_config: str = Field(
        default="",
        description="PIQUANT detector config (e.g., 'PIXL/PiquantConfigs/v7')",
    )
    source_zip: Optional[str] = Field(
        default=None,
        description="Path to source zip file",
    )

    @field_validator("name_normalized", mode="before")
    @classmethod
    def normalize_name(cls, v: str, info) -> str:
        """Auto-normalize name if not provided."""
        if v:
            return v
        # Try to get name from values being validated
        name = info.data.get("name", "")
        return name.strip().lower()

    def model_post_init(self, __context: Any) -> None:
        """Ensure name_normalized is set after init."""
        if not self.name_normalized:
            self.name_normalized = self.name.strip().lower()


class PixliseQuantPoint(IdentifiableModel):
    """Single quantified point from AutoQuant-PDS.

    Stores XRF quantification results for a single measurement point.
    Oxide values are stored as numpy arrays for efficient storage and
    computation.

    Attributes:
        target_id: Foreign key to PixliseTarget
        pmc: Point Motor Count (unique within observation)
        oxide_wt_pct: Oxide weight percentages (16 values)
        oxide_err: Oxide uncertainties (16 values)
        oxide_intensity: Oxide intensities (16 values, Pixlise-only)
        total_counts: Total detector counts
        livetime: Live time in seconds
        chisq: Chi-squared fit quality
        ev_start: Energy calibration start
        ev_per_ch: Energy per channel
        res: Resolution
        fit_iter: Fit iterations
        events: Event count
        triggers: Trigger count
        sclk: Spacecraft clock
        source_filename: Original source filename

    Example:
        >>> point = PixliseQuantPoint(
        ...     target_id=uuid.uuid4(),
        ...     pmc=42,
        ...     oxide_wt_pct=np.zeros(16),
        ...     oxide_err=np.zeros(16),
        ...     oxide_intensity=np.zeros(16),
        ...     total_counts=100000,
        ...     livetime=10.5,
        ...     chisq=1.2,
        ... )
    """

    target_id: str = Field(description="Foreign key to PixliseTarget (UUID string)")
    pmc: int = Field(description="Point Motor Count (unique within observation)")

    # Oxide data stored as bytes (serialized numpy arrays)
    oxide_wt_pct: bytes = Field(description="Oxide wt% as serialized float32 array")
    oxide_err: bytes = Field(description="Oxide errors as serialized float32 array")
    oxide_intensity: bytes = Field(
        description="Oxide intensities as serialized float32 array"
    )

    # Instrument metadata
    total_counts: Optional[int] = Field(default=None, description="Total detector counts")
    livetime: Optional[float] = Field(default=None, description="Live time (seconds)")
    chisq: Optional[float] = Field(default=None, description="Chi-squared fit quality")
    ev_start: Optional[float] = Field(default=None, description="Energy calibration start")
    ev_per_ch: Optional[float] = Field(default=None, description="Energy per channel")
    res: Optional[float] = Field(default=None, description="Resolution")
    fit_iter: Optional[int] = Field(default=None, description="Fit iterations")
    events: Optional[int] = Field(default=None, description="Event count")
    triggers: Optional[int] = Field(default=None, description="Trigger count")
    sclk: Optional[int] = Field(default=None, description="Spacecraft clock")
    source_filename: Optional[str] = Field(default=None, description="Source filename")

    @staticmethod
    def serialize_oxide_array(arr: np.ndarray) -> bytes:
        """Serialize a numpy array to bytes for storage.

        Args:
            arr: Numpy array of oxide values (16 elements)

        Returns:
            Bytes representation of the array
        """
        return arr.astype(np.float32).tobytes()

    @staticmethod
    def deserialize_oxide_array(data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array.

        Args:
            data: Bytes representation of oxide array

        Returns:
            Numpy array of float32 values
        """
        return np.frombuffer(data, dtype=np.float32)

    def get_oxide_wt_pct_array(self) -> np.ndarray:
        """Get oxide wt% as numpy array."""
        return self.deserialize_oxide_array(self.oxide_wt_pct)

    def get_oxide_err_array(self) -> np.ndarray:
        """Get oxide errors as numpy array."""
        return self.deserialize_oxide_array(self.oxide_err)

    def get_oxide_intensity_array(self) -> np.ndarray:
        """Get oxide intensities as numpy array."""
        return self.deserialize_oxide_array(self.oxide_intensity)


class PixliseBeamLocation(IdentifiableModel):
    """Beam location for a point on a specific image.

    Links a PMC measurement point to pixel coordinates on context images.

    Attributes:
        target_id: Foreign key to PixliseTarget
        image_id: Foreign key to PixliseImage
        pmc: Point Motor Count
        x: X coordinate in 3D space
        y: Y coordinate in 3D space
        z: Z coordinate in 3D space
        pixel_i: Pixel column coordinate on image
        pixel_j: Pixel row coordinate on image

    Example:
        >>> loc = PixliseBeamLocation(
        ...     target_id="...",
        ...     image_id="...",
        ...     pmc=42,
        ...     x=1.5, y=2.3, z=0.1,
        ...     pixel_i=512.0, pixel_j=384.0,
        ... )
    """

    target_id: str = Field(description="Foreign key to PixliseTarget (UUID string)")
    image_id: Optional[str] = Field(
        default=None,
        description="Foreign key to PixliseImage (UUID string)",
    )
    pmc: int = Field(description="Point Motor Count")
    x: float = Field(description="X coordinate in 3D space")
    y: float = Field(description="Y coordinate in 3D space")
    z: float = Field(description="Z coordinate in 3D space")
    pixel_i: Optional[float] = Field(default=None, description="Pixel column (i)")
    pixel_j: Optional[float] = Field(default=None, description="Pixel row (j)")
    image_name: Optional[str] = Field(
        default=None,
        description="Image name for reference",
    )


class PixliseImage(IdentifiableModel):
    """Context image from Pixlise export.

    Stores metadata for images exported with Pixlise data.

    Attributes:
        target_id: Foreign key to PixliseTarget
        filename: Image filename
        image_type: Type of image (PCW, DTU_MSA, SIF, CSC_ACI)
        file_path: Path to image file

    Example:
        >>> img = PixliseImage(
        ...     target_id="...",
        ...     filename="PCW_0921_bubenka.png",
        ...     image_type=PixliseImageType.PCW,
        ... )
    """

    target_id: str = Field(description="Foreign key to PixliseTarget (UUID string)")
    filename: str = Field(description="Image filename")
    image_type: PixliseImageType = Field(
        default=PixliseImageType.UNKNOWN,
        description="Type of image",
    )
    file_path: Optional[str] = Field(default=None, description="Full path to image file")

    def model_post_init(self, __context: Any) -> None:
        """Infer image type from filename if it's UNKNOWN."""
        if self.image_type == PixliseImageType.UNKNOWN and self.filename:
            if self.filename.startswith("PCW_"):
                object.__setattr__(self, "image_type", PixliseImageType.PCW)
            elif self.filename.startswith("DTU_MSA_"):
                object.__setattr__(self, "image_type", PixliseImageType.DTU_MSA)
            elif self.filename.startswith("SIF_"):
                object.__setattr__(self, "image_type", PixliseImageType.SIF)
            elif self.filename.startswith("CSC_ACI_"):
                object.__setattr__(self, "image_type", PixliseImageType.CSC_ACI)


class PixliseExportResult(PHASEBaseModel):
    """Result of parsing a Pixlise export zip file.

    Attributes:
        target: Parsed target metadata
        quant_points: List of quantified points
        beam_locations: List of beam locations
        images: List of context images
        warnings: Any warnings during parsing
    """

    target: PixliseTarget
    quant_points: List[PixliseQuantPoint] = Field(default_factory=list)
    beam_locations: List[PixliseBeamLocation] = Field(default_factory=list)
    images: List[PixliseImage] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class PixliseExportParser:
    """Parser for Pixlise zip exports.

    Parses Pixlise Data Export zip files and extracts:
    - Target metadata from directory structure
    - AutoQuant CSV quantification data
    - Beam location CSVs
    - Image file listings

    Example:
        >>> parser = PixliseExportParser()
        >>> result = parser.parse_zip("/nas/000_pixl/Pixlise Data Export 2026-01-28 (1).zip")
        >>> print(f"Parsed {result.target.name}: {len(result.quant_points)} points")
    """

    # Column mapping from Pixlise CSV to internal names
    OXIDE_COLUMN_MAP = {
        "Na2O_%": "Na2O",
        "MgO_%": "MgO",
        "Al2O3_%": "Al2O3",
        "SiO2_%": "SiO2",
        "P2O5_%": "P2O5",
        "SO3_%": "SO3",
        "Cl_%": "Cl",
        "K2O_%": "K2O",
        "CaO_%": "CaO",
        "TiO2_%": "TiO2",
        "Cr2O3_%": "Cr2O3",
        "MnO_%": "MnO",
        "FeO-T_%": "FeO-T",
        "NiO_%": "NiO",
        "ZnO_%": "ZnO",
        "Br_%": "Br",
    }

    def __init__(self):
        """Initialize the parser."""
        pass

    def parse_zip(self, zip_path: Path | str) -> PixliseExportResult:
        """Parse a Pixlise export zip file.

        Args:
            zip_path: Path to the zip file

        Returns:
            PixliseExportResult with parsed data

        Raises:
            ValueError: If the zip file is invalid or missing required data
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise ValueError(f"Zip file not found: {zip_path}")

        warnings: List[str] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the target directory and AutoQuant CSV
            target_name, rtt, autoquant_path = self._find_autoquant_csv(zf)

            # Parse PIQUANT version from header
            piquant_version, detector_config = self._parse_piquant_header(
                zf, autoquant_path
            )

            # Parse AutoQuant data
            quant_rows, quant_warnings = self._parse_autoquant_csv(zf, autoquant_path)
            warnings.extend(quant_warnings)

            # Parse export date from zip filename
            export_date = self._parse_export_date(zip_path.name)

            # Create target
            target = PixliseTarget(
                name=target_name,
                rtt=rtt,
                n_points=len(quant_rows),
                export_date=export_date,
                piquant_version=piquant_version,
                detector_config=detector_config,
                source_zip=str(zip_path),
            )

            # Create quant points
            quant_points = self._create_quant_points(quant_rows, str(target.id))

            # Parse images
            images = self._parse_images(zf, str(target.id))

            # Extract sol from image filenames
            sol = self._extract_sol_from_images(images)
            if sol is not None:
                target.sol = sol

            # Parse beam locations
            beam_locations, loc_warnings = self._parse_beam_locations(
                zf, target_name, str(target.id), images
            )
            warnings.extend(loc_warnings)

        return PixliseExportResult(
            target=target,
            quant_points=quant_points,
            beam_locations=beam_locations,
            images=images,
            warnings=warnings,
        )

    def _find_autoquant_csv(
        self, zf: zipfile.ZipFile
    ) -> tuple[str, int, str]:
        """Find the AutoQuant CSV file and extract target info.

        Returns:
            Tuple of (target_name, rtt, autoquant_path)
        """
        for name in zf.namelist():
            if "AutoQuant-PDS (Combined)-map-by-piquant.csv" in name:
                # Extract target name from path: Data Files/{TargetName}/{RTT}-AutoQuant...
                parts = name.split("/")
                if len(parts) >= 3 and parts[0] == "Data Files":
                    target_name = parts[1].strip()  # Fix leading space issue
                    # Extract RTT from filename
                    filename = parts[-1]
                    rtt_match = re.match(r"(\d+)-AutoQuant", filename)
                    if rtt_match:
                        rtt = int(rtt_match.group(1))
                        return target_name, rtt, name

        raise ValueError("AutoQuant-PDS CSV not found in zip file")

    def _parse_piquant_header(
        self, zf: zipfile.ZipFile, csv_path: str
    ) -> tuple[str, str]:
        """Parse PIQUANT version from CSV header line 0.

        Returns:
            Tuple of (piquant_version, detector_config)
        """
        with zf.open(csv_path) as f:
            first_line = f.readline().decode("utf-8").strip()
            # Parse: PIQUANT version: ghcr.io/pixlise/piquant:3.2.17 DetectorConfig: PIXL/PiquantConfigs/v7
            version_match = re.search(r"piquant:(\d+\.\d+\.\d+)", first_line)
            config_match = re.search(r"DetectorConfig:\s*(\S+)", first_line)

            version = version_match.group(1) if version_match else ""
            config = config_match.group(1) if config_match else ""

            return version, config

    def _parse_autoquant_csv(
        self, zf: zipfile.ZipFile, csv_path: str
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Parse AutoQuant CSV data (skip header line 0).

        Returns:
            Tuple of (rows, warnings)
        """
        warnings: List[str] = []
        rows: List[Dict[str, Any]] = []

        with zf.open(csv_path) as f:
            # Skip line 0 (PIQUANT version header)
            f.readline()

            # Read CSV from line 1 onward
            text_stream = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.DictReader(text_stream)

            for row in reader:
                # Strip whitespace from keys (CSV has spaces after commas)
                cleaned_row = {k.strip(): v for k, v in row.items()}
                rows.append(cleaned_row)

        return rows, warnings

    def _create_quant_points(
        self, rows: List[Dict[str, Any]], target_id: str
    ) -> List[PixliseQuantPoint]:
        """Create PixliseQuantPoint objects from parsed CSV rows."""
        points = []

        for row in rows:
            # Extract PMC
            pmc = int(row.get("PMC", 0))

            # Extract oxide values in canonical order
            oxide_wt_pct = np.zeros(N_OXIDES, dtype=np.float32)
            oxide_err = np.zeros(N_OXIDES, dtype=np.float32)
            oxide_intensity = np.zeros(N_OXIDES, dtype=np.float32)

            for i, oxide in enumerate(OXIDE_NAMES):
                # wt% column: {oxide}_%
                wt_col = f"{oxide}_%"
                oxide_wt_pct[i] = float(row.get(wt_col, 0.0) or 0.0)

                # error column: {oxide}_err
                err_col = f"{oxide}_err"
                oxide_err[i] = float(row.get(err_col, 0.0) or 0.0)

                # intensity column: {oxide}_int
                int_col = f"{oxide}_int"
                oxide_intensity[i] = float(row.get(int_col, 0.0) or 0.0)

            # Extract metadata
            total_counts = int(row.get("total_counts", 0) or 0)
            livetime = float(row.get("livetime", 0.0) or 0.0)
            chisq = float(row.get("chisq", 0.0) or 0.0)
            ev_start = float(row.get("eVstart", 0.0) or 0.0)
            ev_per_ch = float(row.get("eV/ch", 0.0) or 0.0)
            res = float(row.get("res", 0.0) or 0.0)
            fit_iter = int(row.get("iter", 0) or 0)
            events = int(row.get("Events", 0) or 0)
            triggers = int(row.get("Triggers", 0) or 0)
            sclk = int(row.get("SCLK", 0) or 0)
            filename = row.get("filename", "")

            point = PixliseQuantPoint(
                target_id=target_id,
                pmc=pmc,
                oxide_wt_pct=PixliseQuantPoint.serialize_oxide_array(oxide_wt_pct),
                oxide_err=PixliseQuantPoint.serialize_oxide_array(oxide_err),
                oxide_intensity=PixliseQuantPoint.serialize_oxide_array(oxide_intensity),
                total_counts=total_counts if total_counts else None,
                livetime=livetime if livetime else None,
                chisq=chisq if chisq else None,
                ev_start=ev_start if ev_start else None,
                ev_per_ch=ev_per_ch if ev_per_ch else None,
                res=res if res else None,
                fit_iter=fit_iter if fit_iter else None,
                events=events if events else None,
                triggers=triggers if triggers else None,
                sclk=sclk if sclk else None,
                source_filename=filename if filename else None,
            )
            points.append(point)

        return points

    def _parse_images(
        self, zf: zipfile.ZipFile, target_id: str
    ) -> List[PixliseImage]:
        """Parse image files from the Images/ directory."""
        images = []

        for name in zf.namelist():
            if name.startswith("Images/") and name.endswith(".png"):
                filename = Path(name).name
                image = PixliseImage(
                    target_id=target_id,
                    filename=filename,
                    file_path=name,
                )
                images.append(image)

        return images

    def _parse_beam_locations(
        self,
        zf: zipfile.ZipFile,
        target_name: str,
        target_id: str,
        images: List[PixliseImage],
    ) -> tuple[List[PixliseBeamLocation], List[str]]:
        """Parse beam location CSVs."""
        warnings: List[str] = []
        locations: List[PixliseBeamLocation] = []

        # Build image lookup by filename (without extension)
        image_lookup: Dict[str, PixliseImage] = {}
        for img in images:
            # Remove extension for matching
            base = Path(img.filename).stem
            image_lookup[base] = img

        # Find beam location files
        for name in zf.namelist():
            if "beam-locations.csv" in name and target_name in name:
                try:
                    with zf.open(name) as f:
                        text_stream = io.TextIOWrapper(f, encoding="utf-8")
                        reader = csv.DictReader(text_stream)

                        # Find the image name from the filename
                        # Format: {RTT}-{ImageName}-beam-locations.csv
                        basename = Path(name).name
                        match = re.match(r"\d+-(.+)-beam-locations\.csv", basename)
                        image_name = match.group(1) if match else None

                        # Find corresponding image
                        image = image_lookup.get(image_name) if image_name else None

                        for row in reader:
                            pmc = int(row.get("PMC", 0))
                            x = float(row.get("X", 0.0))
                            y = float(row.get("Y", 0.0))
                            z = float(row.get("Z", 0.0))

                            # Find i,j columns (variable naming)
                            pixel_i = None
                            pixel_j = None
                            for key in row.keys():
                                if key.endswith("_i") or key.endswith("_v3_i"):
                                    pixel_i = float(row[key]) if row[key] else None
                                elif key.endswith("_j") or key.endswith("_v3_j"):
                                    pixel_j = float(row[key]) if row[key] else None

                            loc = PixliseBeamLocation(
                                target_id=target_id,
                                image_id=str(image.id) if image else None,
                                pmc=pmc,
                                x=x,
                                y=y,
                                z=z,
                                pixel_i=pixel_i,
                                pixel_j=pixel_j,
                                image_name=image_name,
                            )
                            locations.append(loc)

                except Exception as e:
                    warnings.append(f"Failed to parse beam locations {name}: {e}")

        return locations, warnings

    def _parse_export_date(self, filename: str) -> Optional[date]:
        """Parse export date from zip filename."""
        # Format: Pixlise Data Export 2026-01-28 (N).zip
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if match:
            try:
                return date.fromisoformat(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_sol_from_images(self, images: List[PixliseImage]) -> Optional[int]:
        """Extract sol number from image filenames.

        Sol can be found in several filename patterns:
        - CSC_ACI_SOL1682_... (explicit SOL prefix)
        - PCW_1681_... (second field is sol)
        - DTU_MSA_1681_... (second field is sol)

        Returns:
            Sol number if found, None otherwise
        """
        for img in images:
            filename = img.filename

            # Pattern 1: Explicit SOL prefix (e.g., CSC_ACI_SOL1682_...)
            match = re.search(r"SOL(\d+)", filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

            # Pattern 2: PCW_{sol}_{sclk}_...
            if filename.startswith("PCW_"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    try:
                        sol = int(parts[1])
                        if 0 < sol < 5000:  # Reasonable sol range
                            return sol
                    except ValueError:
                        pass

            # Pattern 3: DTU_MSA_{sol}_...
            if filename.startswith("DTU_MSA_"):
                parts = filename.split("_")
                if len(parts) >= 3:
                    try:
                        sol = int(parts[2])
                        if 0 < sol < 5000:  # Reasonable sol range
                            return sol
                    except ValueError:
                        pass

        return None
