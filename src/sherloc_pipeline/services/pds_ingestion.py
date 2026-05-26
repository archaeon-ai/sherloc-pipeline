"""
PDS4 ingestion service for loading PDS data into phase_pds.db.

Two-engine approach per spec s5/s9:
  - Engine A (pds_engine): phase_pds.db for all writes
  - Engine B (loupe_engine): phase.db for read-only target name lookups

All data flows through Pydantic validation before ORM insertion.
Idempotency: PDS LID as scan_id, version comparison before skip.

Example:
    >>> from sherloc_pipeline.services.pds_ingestion import PDSIngestionService
    >>>
    >>> service = PDSIngestionService(pds_db_path="./phase_pds.db")
    >>> result = service.ingest_sol(Path("./pds/sol_0921/data_processed"))
    >>> print(result.summary)
    "Ingested sol 921: 5 observations, 1597 points, 4791 spectra"
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from sqlalchemy import select, func
from sqlalchemy.engine import Engine

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    init_pds_database,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    ContextImageORM,
)
from sherloc_pipeline.models.context import ContextImage, ImageType
from sherloc_pipeline.models.spectra import (
    CoordinateFrame,
    DataSource,
    ProcessingLevel,
    Scan,
    ScanPoint,
    ScanType,
    TargetType,
    Sol,
    Spectrum,
    SpectralRegion,
    SpectrumType,
    classify_target_type,
)
from sherloc_pipeline.core.pds_client import PDSDownloader
from sherloc_pipeline.core.pds_constants import PDS_DEFAULT_CACHE_DIR
from sherloc_pipeline.core.pds_parsers import (
    PDSLabelParser,
    PDSObservationGroup,
    PDSObservationGrouper,
    PDSRMOParser,
    PDSSpectralParser,
    PDSZpzProductError,
    ParsedSpectralCSV,
)
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError


logger = logging.getLogger(__name__)


class PDSIngestionError(SherlocServiceError):
    """Error during PDS data ingestion."""

    def __init__(
        self,
        message: str,
        sol: Optional[int] = None,
        observation_key: Optional[str] = None,
    ):
        super().__init__(message)
        self.sol = sol
        self.observation_key = observation_key


@dataclass
class PDSVersionUpdate:
    """Record of a version update for a single observation."""

    observation_key: str
    scan_id: str
    old_version: str
    new_version: str
    sol: int


@dataclass
class PDSIngestionStats:
    """Statistics from a PDS ingestion operation."""

    sols_processed: int = 0
    sols_skipped: int = 0
    observations_ingested: int = 0
    observations_skipped: int = 0
    observations_updated: int = 0
    points_ingested: int = 0
    spectra_ingested: int = 0
    context_images_ingested: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    version_updates: List[PDSVersionUpdate] = field(default_factory=list)

    def __add__(self, other: "PDSIngestionStats") -> "PDSIngestionStats":
        return PDSIngestionStats(
            sols_processed=self.sols_processed + other.sols_processed,
            sols_skipped=self.sols_skipped + other.sols_skipped,
            observations_ingested=self.observations_ingested + other.observations_ingested,
            observations_skipped=self.observations_skipped + other.observations_skipped,
            observations_updated=self.observations_updated + other.observations_updated,
            points_ingested=self.points_ingested + other.points_ingested,
            spectra_ingested=self.spectra_ingested + other.spectra_ingested,
            context_images_ingested=self.context_images_ingested + other.context_images_ingested,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            version_updates=self.version_updates + other.version_updates,
        )


class TargetNameResolver:
    """Resolves geological target names for PDS observations.

    Three-tier strategy (spec s10):
      1. SCLK cross-reference to Loupe DB (two-pass tolerance)
      2. Curated mapping table (configs/pds_target_mapping.json)
      3. NULL fallback

    SCLK cross-reference uses a two-pass approach:
      - Pass 1 (±3s): Tight window for high-confidence matches
      - Pass 2 (±5s): Wider window if Pass 1 finds nothing

    Tie-breaking for multiple candidates:
      1. Smallest absolute SCLK delta
      2. Same site_drive if still tied
    """

    PASS1_TOLERANCE = 3
    PASS2_TOLERANCE = 5

    def __init__(
        self,
        loupe_engine: Optional[Engine] = None,
        curated_path: Optional[Path] = None,
    ):
        self.loupe_engine = loupe_engine
        self._curated_map: Dict[str, str] = {}

        if curated_path is not None:
            curated = Path(curated_path)
            if curated.exists():
                try:
                    with open(curated) as f:
                        self._curated_map = json.load(f)
                    logger.info(
                        "Loaded %d curated target mappings from %s",
                        len(self._curated_map), curated,
                    )
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to load curated target mapping %s: %s",
                        curated, e,
                    )

    def resolve(
        self,
        sol: int,
        pds_sclk: int,
        site_drive: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve target name for a PDS observation.

        Args:
            sol: Sol number.
            pds_sclk: PDS spacecraft clock start (integer).
            site_drive: Optional site/drive string for tie-breaking.

        Returns:
            Target name string or None if unresolvable.
        """
        # Tier 1: SCLK cross-reference against Loupe DB
        if self.loupe_engine is not None:
            target = self._sclk_crossref(sol, pds_sclk, site_drive)
            if target is not None:
                return target

        # Tier 2: Curated mapping table
        # Key format: "{sol}_{sclk}" (matches spec convention)
        curated_key = f"{sol}_{pds_sclk}"
        if curated_key in self._curated_map:
            target = self._curated_map[curated_key]
            logger.info(
                "Target resolved via curated mapping: sol %d, SCLK %d → %s",
                sol, pds_sclk, target,
            )
            return target

        # Tier 3: NULL
        logger.info(
            "No target match: sol %d, PDS SCLK %d. Marked for manual curation.",
            sol, pds_sclk,
        )
        return None

    def _sclk_crossref(
        self,
        sol: int,
        pds_sclk: int,
        site_drive: Optional[str],
    ) -> Optional[str]:
        """Two-pass SCLK cross-reference against Loupe DB (spec s10)."""
        # Pass 1: ±3s
        target = self._sclk_query(sol, pds_sclk, site_drive, self.PASS1_TOLERANCE)
        if target is not None:
            return target

        # Pass 2: ±5s (only if Pass 1 found nothing)
        return self._sclk_query(sol, pds_sclk, site_drive, self.PASS2_TOLERANCE)

    def _sclk_query(
        self,
        sol: int,
        pds_sclk: int,
        site_drive: Optional[str],
        tolerance: int,
    ) -> Optional[str]:
        """Query Loupe DB for scans matching SCLK within tolerance.

        Returns:
            Target name or None if no match.
        """
        with get_session(self.loupe_engine) as session:
            rows = session.execute(
                select(
                    ScanORM.target,
                    ScanORM.sclk_start,
                    ScanORM.site_drive,
                ).where(
                    ScanORM.sol_number == sol,
                    ScanORM.target.isnot(None),
                    ScanORM.sclk_start >= pds_sclk - tolerance,
                    ScanORM.sclk_start <= pds_sclk + tolerance,
                )
            ).fetchall()

            if not rows:
                return None

            if len(rows) == 1:
                return rows[0].target

            # Multiple candidates — tie-break
            return self._tie_break(
                rows, sol, pds_sclk, site_drive, tolerance
            )

    def _tie_break(
        self,
        candidates: list,
        sol: int,
        pds_sclk: int,
        site_drive: Optional[str],
        tolerance: int,
    ) -> str:
        """Tie-break multiple SCLK candidates (spec s10).

        1. Smallest absolute SCLK delta (nearest match).
        2. Same site_drive if still tied.
        """
        # Sort by absolute SCLK delta
        sorted_candidates = sorted(
            candidates, key=lambda c: abs(c.sclk_start - pds_sclk)
        )
        best_delta = abs(sorted_candidates[0].sclk_start - pds_sclk)

        # Collect all tied at minimum delta
        tied = [c for c in sorted_candidates if abs(c.sclk_start - pds_sclk) == best_delta]

        if len(tied) == 1:
            selected = tied[0]
        elif site_drive is not None:
            # Prefer matching site_drive
            sd_matches = [c for c in tied if c.site_drive == site_drive]
            selected = sd_matches[0] if sd_matches else tied[0]
        else:
            selected = tied[0]

        logger.warning(
            "SCLK ambiguity: sol %d, PDS SCLK %d, %d candidates within ±%ds, "
            "selected delta=%ds.",
            sol, pds_sclk, len(candidates), tolerance,
            abs(selected.sclk_start - pds_sclk),
        )

        return selected.target


class PDSIngestionService:
    """Service for ingesting PDS4 data into phase_pds.db.

    Two-engine architecture (spec s5):
      - pds_engine: Writes to phase_pds.db (Engine A)
      - loupe_engine: Reads from phase.db for target name lookup (Engine B)

    All ingestion is idempotent: existing scans are skipped unless a newer
    PDS version is available or force=True.

    Attributes:
        pds_engine: SQLAlchemy engine for phase_pds.db (writes)
        loupe_engine: Optional SQLAlchemy engine for phase.db (target lookup)

    Example:
        >>> service = PDSIngestionService(
        ...     pds_db_path=Path("./phase_pds.db"),
        ...     loupe_db_path=Path("./phase.db"),
        ... )
        >>> result = service.ingest_sol(Path("./pds/sol_0921/data_processed"))
    """

    def __init__(
        self,
        pds_db_path: Optional[Path] = None,
        loupe_db_path: Optional[Path] = None,
    ):
        """Initialize the PDS ingestion service.

        Args:
            pds_db_path: Path to phase_pds.db. Defaults to ./phase_pds.db.
            loupe_db_path: Optional path to phase.db for target name cross-reference.
                If None, target names will not be resolved via SCLK lookup.
        """
        if pds_db_path is None:
            pds_db_path = Path("./phase_pds.db")

        self.pds_db_path = Path(pds_db_path)

        # Engine A: phase_pds.db for writes (with PDS-specific unique constraint)
        self.pds_engine: Engine = init_pds_database(self.pds_db_path)

        # Engine B: phase.db for target name lookups (optional)
        self.loupe_engine: Optional[Engine] = None
        if loupe_db_path is not None:
            loupe_path = Path(loupe_db_path)
            if loupe_path.exists():
                self.loupe_engine = get_engine(loupe_path)
            else:
                logger.warning(
                    "Loupe database not found at %s; target name lookup disabled",
                    loupe_path,
                )

        # Parsers
        self._label_parser = PDSLabelParser()
        self._spectral_parser = PDSSpectralParser()
        self._rmo_parser = PDSRMOParser()
        self._grouper = PDSObservationGrouper()

        # Target name resolver (spec s10).
        # Curated mapping is operator-supplied at
        # $SHERLOC_HOME/configs/pds_target_mapping.json (CWD-relative when
        # SHERLOC_HOME is unset, per §8.1). Missing file is treated as an
        # empty mapping (Tier 1 SCLK cross-reference + Tier 3 NULL fallback
        # still work). External users without Loupe access typically run
        # without this file. See PUBLIC_TOOLKIT_ARCHITECTURE_SPEC §11.4.
        sherloc_home = Path(os.environ.get("SHERLOC_HOME", "."))
        curated_path = sherloc_home / "configs" / "pds_target_mapping.json"
        self._target_resolver = TargetNameResolver(
            loupe_engine=self.loupe_engine,
            curated_path=curated_path if curated_path.exists() else None,
        )

    def ingest_sol(
        self,
        sol_dir: Path,
        force: bool = False,
    ) -> ServiceResult:
        """Ingest all observations for a sol directory.

        Discovers PDS products, groups by observation, and ingests each
        observation into phase_pds.db.

        Args:
            sol_dir: Path to sol data directory containing CSV/XML pairs
                (e.g., ./pds/sol_0921/data_processed).
            force: If True, re-ingest all observations regardless of version.

        Returns:
            ServiceResult with ingestion summary and statistics.

        Raises:
            PDSIngestionError: If discovery or ingestion fails critically.
        """
        sol_dir = Path(sol_dir)
        if not sol_dir.exists():
            raise PDSIngestionError(f"Sol directory not found: {sol_dir}")

        # Discover and group observations
        try:
            groups = self._grouper.group_sol_directory(
                sol_dir, label_parser=self._label_parser
            )
        except Exception as e:
            raise PDSIngestionError(
                f"Failed to discover observations in {sol_dir}: {e}"
            )

        if not groups:
            return ServiceResult(
                summary=f"No valid observations found in {sol_dir}",
                warnings=["No observations after zpz filtering"],
                metadata={"success": True, "observations": 0},
            )

        sol_number = groups[0].sol
        stats = PDSIngestionStats(sols_processed=1)

        # Ensure sol record exists in phase_pds.db
        self._ensure_sol(sol_number, groups)

        # Ingest each observation
        for group in groups:
            try:
                obs_stats = self._ingest_observation(group, sol_dir, force=force)
                stats = stats + obs_stats
            except Exception as e:
                stats.errors.append(
                    f"Observation {group.observation_key}: {e}"
                )
                logger.exception(
                    "Error ingesting observation %s", group.observation_key
                )

        # Build summary
        summary = (
            f"Ingested sol {sol_number}: "
            f"{stats.observations_ingested} observations, "
            f"{stats.points_ingested} points, "
            f"{stats.spectra_ingested} spectra, "
            f"{stats.context_images_ingested} context images"
        )
        if stats.observations_skipped:
            summary += f", {stats.observations_skipped} skipped"
        if stats.observations_updated:
            summary += f", {stats.observations_updated} updated"

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": len(stats.errors) == 0,
                "sol_number": sol_number,
                "observations_ingested": stats.observations_ingested,
                "observations_skipped": stats.observations_skipped,
                "observations_updated": stats.observations_updated,
                "points_ingested": stats.points_ingested,
                "spectra_ingested": stats.spectra_ingested,
                "context_images_ingested": stats.context_images_ingested,
                "errors": stats.errors,
                "version_updates": [
                    {
                        "observation_key": vu.observation_key,
                        "scan_id": vu.scan_id,
                        "old_version": vu.old_version,
                        "new_version": vu.new_version,
                    }
                    for vu in stats.version_updates
                ],
            },
        )

    def _ensure_sol(
        self,
        sol_number: int,
        groups: List[PDSObservationGroup],
    ) -> None:
        """Ensure a sol record exists in phase_pds.db.

        Creates the sol with data_source='pds4' if it doesn't exist.
        Enriches metadata from XML labels if available (spec s9 step 5).
        """
        with get_session(self.pds_engine) as session:
            existing = session.get(SolORM, sol_number)
            if existing is not None:
                return

            # Create sol stub with data_source='pds4'.
            # Metadata (earth_date, solar_longitude, mission_phase) is
            # enriched from XML labels in _ingest_observation → _enrich_sol_metadata.
            sol_orm = SolORM(
                sol_number=sol_number,
                data_source="pds4",
                created_at=datetime.now(timezone.utc),
            )
            session.add(sol_orm)

    def _ingest_observation(
        self,
        group: PDSObservationGroup,
        sol_dir: Path,
        force: bool = False,
    ) -> PDSIngestionStats:
        """Ingest a single PDS observation into phase_pds.db.

        Implements per-observation flow from spec s9:
        1. Idempotency check (LID as scan_id, version comparison)
        2. Target name resolution (spec s10, 3-tier lookup)
        3. Create ScanORM from XML metadata with resolved target
        4. Create ScanPointORM from RMO positions
        5. Create SpectrumORM (3 per point: R1, R2, R3)
        6. Enrich sol metadata

        Args:
            group: PDSObservationGroup with classified products.
            sol_dir: Path to sol data directory for file resolution.
            force: If True, re-ingest regardless of version.

        Returns:
            PDSIngestionStats for this observation.
        """
        stats = PDSIngestionStats()

        # Require spectral product (RRS or RCS)
        spectral_key = self._get_spectral_product_key(group)
        if spectral_key is None:
            stats.errors.append(
                f"{group.observation_key}: No RRS/RCS spectral product"
            )
            return stats

        spectral_product_id = group.products[spectral_key]

        # Parse XML label for metadata
        xml_path = sol_dir / spectral_product_id.xml_filename
        if not xml_path.exists():
            stats.errors.append(
                f"{group.observation_key}: XML label not found: {xml_path.name}"
            )
            return stats

        metadata = self._label_parser.parse_label(xml_path)

        # Step 1: Idempotency — PDS LID as scan_id
        scan_id = metadata.logical_identifier
        pds_version = metadata.version_id

        with get_session(self.pds_engine) as session:
            existing_scan = session.execute(
                select(ScanORM).where(ScanORM.scan_id == scan_id)
            ).scalar_one_or_none()

            if existing_scan is not None:
                if not force:
                    # Version comparison using numeric tuples (spec s9)
                    # "1.10" > "1.2" in tuple form: (1,10) > (1,2)
                    existing_meta = existing_scan.pds4_metadata or {}
                    existing_version_str = existing_meta.get("version", "1.0")
                    new_version = self._parse_version_tuple(pds_version)
                    old_version = self._parse_version_tuple(existing_version_str)

                    if new_version > old_version:
                        # Newer version: cascade delete and re-ingest
                        logger.info(
                            "Obs %s sol %s: version %s → %s, re-ingested.",
                            group.obs_id, group.sol,
                            existing_version_str, pds_version,
                        )
                        session.delete(existing_scan)
                        session.flush()
                        stats.observations_updated += 1
                        stats.version_updates.append(PDSVersionUpdate(
                            observation_key=group.observation_key,
                            scan_id=scan_id,
                            old_version=existing_version_str,
                            new_version=pds_version,
                            sol=group.sol,
                        ))
                    else:
                        stats.observations_skipped += 1
                        return stats
                else:
                    # force=True: cascade delete and re-ingest
                    logger.info(
                        "Force re-ingesting %s (version %s)",
                        scan_id, pds_version,
                    )
                    session.delete(existing_scan)
                    session.flush()
                    stats.observations_updated += 1

            # Step 2: Resolve target name (spec s10, 3-tier)
            resolved_target = self._target_resolver.resolve(
                sol=group.sol,
                pds_sclk=metadata.sclk_start_int,
                site_drive=metadata.site_drive_str,
            )

            # Step 3: Create Scan from metadata (synthesis rules, spec s9)
            scan = self._build_scan(group, metadata, target=resolved_target)
            scan_orm = ScanORM.from_pydantic(scan)
            session.add(scan_orm)
            session.flush()  # Ensure scan.id is available for children

            # Step 4: Create ScanPoints from RMO positions
            points = self._build_scan_points(group, sol_dir, scan)
            for point in points:
                point_orm = ScanPointORM.from_pydantic(point)
                session.add(point_orm)
            stats.points_ingested += len(points)

            # Step 5: Create Spectra (3 regions per point)
            n_spectra = self._build_spectra(
                group, sol_dir, scan, points, session
            )
            stats.spectra_ingested += n_spectra

            # Step 5b: ACI context image association (spec s11)
            context_images = self._build_context_images(group, sol_dir, scan)
            for img in context_images:
                img_orm = ContextImageORM.from_pydantic(img)
                session.add(img_orm)
            stats.context_images_ingested += len(context_images)

            # Step 6: Enrich sol metadata from XML
            self._enrich_sol_metadata(session, group.sol, metadata)

            # Target name was resolved in Step 2 (before scan creation)

            stats.observations_ingested += 1

        return stats

    def _get_spectral_product_key(
        self, group: PDSObservationGroup
    ) -> Optional[str]:
        """Get the spectral product type key ('rrs' or 'rcs') from group."""
        if "rrs" in group.products:
            return "rrs"
        if "rcs" in group.products:
            return "rcs"
        return None

    def _build_scan(
        self,
        group: PDSObservationGroup,
        metadata: "PDSObservationMetadata",
        target: Optional[str] = None,
    ) -> Scan:
        """Build a Scan Pydantic model from PDS observation data.

        Applies synthesis rules from spec s9:
        - scan_id: PDS LID (logical_identifier)
        - scan_name: "pds_{sol}_{sclk}_{obs_id}"
        - shots_per_point: NULL (not in processed PDS)
        - scan_type: From observation classification
        - data_source: 'pds4'
        - target: Resolved via TargetNameResolver (spec s10)
        """
        sclk_start = metadata.sclk_start_int
        sclk_stop = None
        if metadata.spacecraft_clock_stop is not None:
            try:
                sclk_stop = int(float(metadata.spacecraft_clock_stop))
            except (ValueError, TypeError):
                pass

        # Determine n_points from metadata or RMO
        n_spectra = metadata.n_spectra
        n_points = n_spectra if n_spectra is not None else 1

        # Map scan_type string to ScanType enum
        scan_type = None
        if group.scan_type is not None:
            scan_type = ScanType(group.scan_type)

        # Site/drive string from metadata
        site_drive = metadata.site_drive_str

        # Build pds4_metadata JSON blob
        pds4_meta = metadata.to_pds4_metadata_dict()

        scan_name = f"pds_{group.sol:04d}_{group.sclk}_{group.obs_id}"

        return Scan(
            sol_number=group.sol,
            scan_name=scan_name,
            target=target,
            scan_id=metadata.logical_identifier,
            sclk_start=sclk_start,
            sclk_stop=sclk_stop,
            n_points=n_points,
            n_channels=2148,
            shots_per_point=None,  # Not available in processed PDS products
            processing_applied="laser_normalized",
            source_path=str(
                Path("pds") / f"sol_{group.sol:04d}" / "data_processed"
            ),
            pds4_metadata=pds4_meta,
            data_source=DataSource.PDS4,
            site_drive=site_drive,
            sequence_id=group.sequence_code,
            scan_type=scan_type,
            target_type=TargetType(classify_target_type(target, scan_name)),
        )

    def _build_scan_points(
        self,
        group: PDSObservationGroup,
        sol_dir: Path,
        scan: Scan,
    ) -> List[ScanPoint]:
        """Build ScanPoint models from RMO positions.

        If RMO is available, positions come from the RMO CSV with
        coordinate_frame='aci_pixel'. If RMO is missing, creates
        placeholder points with position_index only (spec s9 fallback).
        """
        points: List[ScanPoint] = []

        if "rmo" in group.products:
            rmo_product_id = group.products["rmo"]
            rmo_csv = sol_dir / rmo_product_id.csv_filename

            try:
                rmo_result = self._rmo_parser.parse(rmo_csv)

                for pos in rmo_result.positions:
                    points.append(ScanPoint(
                        scan_id=scan.id,
                        point_index=pos.position_index,
                        x_pixel=pos.x,
                        y_pixel=pos.y,
                        coordinate_frame=CoordinateFrame.ACI_PIXEL,
                    ))

                return points
            except PDSZpzProductError:
                logger.warning(
                    "%s: RMO is zpz product, using index-only fallback",
                    group.observation_key,
                )
            except Exception as e:
                logger.warning(
                    "%s: RMO parse failed (%s), using index-only fallback",
                    group.observation_key, e,
                )
        else:
            # Expected for detail/survey but handled gracefully
            if group.scan_type in ("detail", "survey"):
                logger.warning(
                    "%s: RMO missing for %s scan, using index-only points",
                    group.observation_key, group.scan_type,
                )

        # Fallback: index-only points (spec s9 Missing RMO Fallback)
        n_points = scan.n_points
        for i in range(n_points):
            points.append(ScanPoint(
                scan_id=scan.id,
                point_index=i,
                coordinate_frame=None,
            ))

        return points

    def _build_spectra(
        self,
        group: PDSObservationGroup,
        sol_dir: Path,
        scan: Scan,
        points: List[ScanPoint],
        session: "Session",
    ) -> int:
        """Build and persist Spectrum ORM records from spectral CSV data.

        Creates 3 spectra per point (R1, R2, R3) from the RRS/RCS CSV.
        Uses PDS-embedded wavelength as wavelength_source.

        Returns:
            Number of spectra created.
        """
        spectral_key = self._get_spectral_product_key(group)
        if spectral_key is None:
            return 0

        product_id = group.products[spectral_key]
        csv_path = sol_dir / product_id.csv_filename

        try:
            parsed: ParsedSpectralCSV = self._spectral_parser.parse(csv_path)
        except PDSZpzProductError:
            logger.warning(
                "%s: Spectral product %s is zpz, skipping spectra",
                group.observation_key, spectral_key,
            )
            return 0
        except Exception as e:
            logger.error(
                "%s: Failed to parse spectral CSV %s: %s",
                group.observation_key, csv_path.name, e,
            )
            return 0

        # Compress wavelength array once for reuse
        # parsed.product.wavelengths is already a List[float] from Pydantic
        wavelength_bytes = Spectrum.compress_array(parsed.product.wavelengths)

        count = 0
        regions = [
            (SpectralRegion.R1, "R1"),
            (SpectralRegion.R2, "R2"),
            (SpectralRegion.R3, "R3"),
        ]

        for region_enum, region_key in regions:
            if region_key not in parsed.spectra:
                continue

            spectra_array = parsed.spectra[region_key]  # (N, 2148)
            n_rows = spectra_array.shape[0]

            if n_rows != len(points):
                logger.warning(
                    "%s: %s spectra count (%d) != point count (%d)",
                    group.observation_key, region_key, n_rows, len(points),
                )

            # Create one spectrum per point per region
            for i, point in enumerate(points):
                if i >= n_rows:
                    break

                intensities = Spectrum.compress_array(
                    spectra_array[i].tolist()
                )

                spectrum = Spectrum(
                    scan_point_id=point.id,
                    region=region_enum,
                    spectrum_type=SpectrumType.LASER_NORMALIZED,
                    processing_level=ProcessingLevel.NORMALIZED,
                    intensities=intensities,
                    wavelengths=wavelength_bytes,
                    wavelength_source="pds_embedded",
                )

                spectrum_orm = SpectrumORM.from_pydantic(spectrum)
                session.add(spectrum_orm)
                count += 1

        return count

    def _build_context_images(
        self,
        group: PDSObservationGroup,
        sol_dir: Path,
        scan: Scan,
    ) -> List[ContextImage]:
        """Build ContextImage models from RMO Image_name values (spec s11).

        Parses the RMO CSV for unique Image_name values (ACI filenames),
        constructs PDS4 LIDVIDs, and creates ContextImage records.

        Expected ACI counts per scan type:
        - Detail: 1 ACI (context ~100s before scan)
        - Survey: 2 ACI (pre-scan + post-scan)
        - Calibration: 1 ACI (calibration context)

        Args:
            group: PDSObservationGroup with classified products.
            sol_dir: Path to sol data directory.
            scan: Parent Scan model (for scan_id linkage).

        Returns:
            List of ContextImage models (may be empty if no RMO).
        """
        if "rmo" not in group.products:
            return []

        rmo_product_id = group.products["rmo"]
        rmo_csv = sol_dir / rmo_product_id.csv_filename

        try:
            rmo_result = self._rmo_parser.parse(rmo_csv)
        except PDSZpzProductError:
            logger.warning(
                "%s: RMO is zpz, no ACI images to associate",
                group.observation_key,
            )
            return []
        except Exception as e:
            logger.warning(
                "%s: RMO parse failed (%s), no ACI images",
                group.observation_key, e,
            )
            return []

        images: List[ContextImage] = []
        for image_name in rmo_result.image_names:
            lidvid = self._construct_aci_lidvid(image_name)
            images.append(ContextImage(
                scan_id=scan.id,
                image_type=ImageType.ACI,
                file_path=f"pds:{lidvid}",
                pds_lidvid=lidvid,
            ))

        return images

    @staticmethod
    def _construct_aci_lidvid(image_name: str) -> str:
        """Construct PDS4 LIDVID from RMO Image_name (spec s11).

        Algorithm:
        1. Strip .IMG extension
        2. Extract version suffix (last 2 characters)
        3. Lowercase remaining base name
        4. Prefix with ACI collection URI
        5. Append ::VID (major.0)

        Example:
            SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG
            → urn:nasa:pds:mars2020_imgops:data_aci_imgops:
              sc3_0921_0748731308_359ecm_n0450000srlc11374_0000lmj::1.0
        """
        PDS_ACI_COLLECTION = "urn:nasa:pds:mars2020_imgops:data_aci_imgops"

        name = image_name
        # Strip .IMG extension (case-insensitive)
        if name.upper().endswith(".IMG"):
            name = name[:-4]

        # Extract version suffix (last 2 chars) and convert to VID
        version_suffix = name[-2:]
        base_name = name[:-2]

        try:
            major_version = int(version_suffix)
        except ValueError:
            major_version = 1

        # Lowercase base name for LID product identifier
        lid_product = base_name.lower()

        return f"{PDS_ACI_COLLECTION}:{lid_product}::{major_version}.0"

    def _enrich_sol_metadata(
        self,
        session: "Session",
        sol_number: int,
        metadata: "PDSObservationMetadata",
    ) -> None:
        """Enrich sol record with metadata from XML label (spec s9 step 5).

        Only fills NULL fields to avoid overwriting existing values.
        """
        sol = session.get(SolORM, sol_number)
        if sol is None:
            return

        if sol.earth_date is None and metadata.earth_date is not None:
            try:
                from datetime import date as _date
                sol.earth_date = _date.fromisoformat(metadata.earth_date)
            except (ValueError, TypeError):
                pass

        if sol.solar_longitude is None and metadata.solar_longitude is not None:
            sol.solar_longitude = metadata.solar_longitude

        if sol.mission_phase is None and metadata.mission_phase_name is not None:
            sol.mission_phase = metadata.mission_phase_name

    def _resolve_xml_path(
        self,
        product_id: "PDSProductId",
        group: PDSObservationGroup,
        sol_dir: Path,
    ) -> Path:
        """Resolve XML label path for a product."""
        return sol_dir / product_id.xml_filename

    @staticmethod
    def _parse_version_tuple(version_str: Optional[str]) -> tuple:
        """Parse a PDS version string into a numeric tuple for comparison.

        Handles "1.0" -> (1, 0), "1.10" -> (1, 10), "2.3" -> (2, 3).
        Numeric tuple comparison avoids lexicographic errors where
        "1.10" < "1.2" would be incorrect as strings.

        Returns (0,) for None or unparseable values as a safe fallback.
        """
        if version_str is None:
            return (0,)
        try:
            return tuple(int(p) for p in version_str.split("."))
        except (ValueError, AttributeError):
            return (0,)

    def get_database_stats(self) -> Dict[str, int]:
        """Get current phase_pds.db statistics."""
        with get_session(self.pds_engine) as session:
            return {
                "sols": session.query(SolORM).count(),
                "scans": session.query(ScanORM).count(),
                "scan_points": session.query(ScanPointORM).count(),
                "spectra": session.query(SpectrumORM).count(),
                "context_images": session.query(ContextImageORM).count(),
            }

    def validate_database(self) -> Dict:
        """Run validation queries on phase_pds.db and return a structured report.

        Checks:
        - Spectra counts by region (R1/R2/R3)
        - Unique scan_ids
        - data_source='pds4' on all scans
        - Sol metadata completeness (earth_date, solar_longitude, mission_phase)

        Returns:
            Dict with keys: counts, spectra_by_region, scan_ids,
            data_source_check, sol_metadata, issues.
        """
        issues: List[str] = []

        with get_session(self.pds_engine) as session:
            # --- Table counts ---
            counts = {
                "sols": session.query(SolORM).count(),
                "scans": session.query(ScanORM).count(),
                "scan_points": session.query(ScanPointORM).count(),
                "spectra": session.query(SpectrumORM).count(),
                "context_images": session.query(ContextImageORM).count(),
            }

            # --- Spectra by region ---
            region_rows = (
                session.query(SpectrumORM.region, func.count(SpectrumORM.id))
                .group_by(SpectrumORM.region)
                .all()
            )
            spectra_by_region = {region: count for region, count in region_rows}

            # Validate R1/R2/R3 balance
            region_counts = list(spectra_by_region.values())
            if region_counts and len(set(region_counts)) > 1:
                issues.append(
                    f"Unbalanced spectra across regions: {spectra_by_region}"
                )

            # --- Unique scan_ids ---
            scan_id_rows = (
                session.query(ScanORM.scan_id).distinct().all()
            )
            unique_scan_ids = [row[0] for row in scan_id_rows]

            # Check for duplicates (unique constraint should prevent, but verify)
            total_scans = counts["scans"]
            if len(unique_scan_ids) != total_scans:
                issues.append(
                    f"Duplicate scan_ids: {total_scans} scans but "
                    f"{len(unique_scan_ids)} unique scan_ids"
                )

            # --- data_source check ---
            source_rows = (
                session.query(ScanORM.data_source, func.count(ScanORM.id))
                .group_by(ScanORM.data_source)
                .all()
            )
            data_source_counts = {src: cnt for src, cnt in source_rows}

            non_pds4 = {
                src: cnt for src, cnt in data_source_counts.items()
                if src != "pds4"
            }
            if non_pds4:
                issues.append(
                    f"Non-pds4 data_source found in scans: {non_pds4}"
                )

            # Also check sols.data_source
            sol_source_rows = (
                session.query(SolORM.data_source, func.count(SolORM.sol_number))
                .group_by(SolORM.data_source)
                .all()
            )
            sol_data_source_counts = {src: cnt for src, cnt in sol_source_rows}

            # --- Sol metadata completeness ---
            sol_rows = session.query(SolORM).all()
            sol_metadata = []
            for sol in sol_rows:
                entry = {
                    "sol_number": sol.sol_number,
                    "earth_date": str(sol.earth_date) if sol.earth_date else None,
                    "solar_longitude": sol.solar_longitude,
                    "mission_phase": sol.mission_phase,
                    "data_source": sol.data_source,
                }
                missing = []
                if sol.earth_date is None:
                    missing.append("earth_date")
                if sol.solar_longitude is None:
                    missing.append("solar_longitude")
                if sol.mission_phase is None:
                    missing.append("mission_phase")
                if missing:
                    issues.append(
                        f"Sol {sol.sol_number} missing metadata: {missing}"
                    )
                entry["missing_fields"] = missing
                sol_metadata.append(entry)

        report = {
            "counts": counts,
            "spectra_by_region": spectra_by_region,
            "scan_ids": unique_scan_ids,
            "data_source_check": {
                "scans": data_source_counts,
                "sols": sol_data_source_counts,
                "all_pds4": len(non_pds4) == 0,
            },
            "sol_metadata": sol_metadata,
            "issues": issues,
            "valid": len(issues) == 0,
        }

        logger.info(
            "Database validation: %s (%d issues)",
            "PASS" if report["valid"] else "FAIL",
            len(issues),
        )

        return report

    def check_for_updates(
        self,
        sol_dir: Path,
    ) -> Dict:
        """Check a sol directory for version updates against phase_pds.db.

        Compares the LIDVID version of each observation in the downloaded
        files against the version stored in pds4_metadata.version in the DB.
        Uses numeric tuple comparison (spec s6).

        Does NOT modify the database — read-only check.

        Args:
            sol_dir: Path to sol data directory (e.g., sol_0921/data_processed).

        Returns:
            Dict with keys:
              - new: list of observation_keys not yet in DB
              - updated: list of dicts with observation_key, scan_id, old/new version
              - current: list of observation_keys at latest version
              - errors: list of error strings
        """
        sol_dir = Path(sol_dir)
        if not sol_dir.exists():
            raise PDSIngestionError(f"Sol directory not found: {sol_dir}")

        try:
            groups = self._grouper.group_sol_directory(
                sol_dir, label_parser=self._label_parser
            )
        except Exception as e:
            raise PDSIngestionError(
                f"Failed to discover observations in {sol_dir}: {e}"
            )

        new_obs: List[Dict] = []
        updated_obs: List[Dict] = []
        current_obs: List[Dict] = []
        errors: List[str] = []

        for group in groups:
            spectral_key = self._get_spectral_product_key(group)
            if spectral_key is None:
                continue

            product_id = group.products[spectral_key]
            xml_path = sol_dir / product_id.xml_filename
            if not xml_path.exists():
                errors.append(
                    f"{group.observation_key}: XML label not found"
                )
                continue

            try:
                metadata = self._label_parser.parse_label(xml_path)
            except Exception as e:
                errors.append(
                    f"{group.observation_key}: XML parse failed: {e}"
                )
                continue

            scan_id = metadata.logical_identifier
            pds_version = metadata.version_id

            with get_session(self.pds_engine) as session:
                existing_scan = session.execute(
                    select(ScanORM).where(ScanORM.scan_id == scan_id)
                ).scalar_one_or_none()

                if existing_scan is None:
                    new_obs.append({
                        "observation_key": group.observation_key,
                        "scan_id": scan_id,
                        "version": pds_version,
                        "sol": group.sol,
                    })
                else:
                    existing_meta = existing_scan.pds4_metadata or {}
                    existing_version_str = existing_meta.get("version", "1.0")
                    new_ver = self._parse_version_tuple(pds_version)
                    old_ver = self._parse_version_tuple(existing_version_str)

                    if new_ver > old_ver:
                        updated_obs.append({
                            "observation_key": group.observation_key,
                            "scan_id": scan_id,
                            "old_version": existing_version_str,
                            "new_version": pds_version,
                            "sol": group.sol,
                        })
                        logger.info(
                            "Version update available: %s sol %d: %s → %s",
                            group.obs_id, group.sol,
                            existing_version_str, pds_version,
                        )
                    else:
                        current_obs.append({
                            "observation_key": group.observation_key,
                            "scan_id": scan_id,
                            "version": existing_version_str,
                            "sol": group.sol,
                        })

        return {
            "new": new_obs,
            "updated": updated_obs,
            "current": current_obs,
            "errors": errors,
            "has_updates": len(updated_obs) > 0 or len(new_obs) > 0,
        }
