"""
Cosmic-ray mask persistence service for the SHERLOC pipeline.

Persists the per-(point, region) channel-index masks produced by the ML
despike path into the ``cosmic_ray_masks`` satellite table (spec §4.5,
MLD-SYS-008/012), attached to each (scan_point, region)'s DARK_SUBTRACTED
spectrum row — the canonical serving representation (§3.4). Persistence
follows parent-row existence: masks whose parent spectrum row was never
ingested (e.g. a full-region run against an ``R1_only``-ingested database)
are skipped with a debug log while remaining in the run artifacts, so
provenance is never silently lost (MLD-SYS-008 AC3).

This module imports no ML runtime and nothing from ``ml_despike`` — it
consumes only the provenance metadata dict the despike step recorded
(MLD-QUA-002).

Usage:
    from sherloc_pipeline.services.cr_masks import CRMaskService

    service = CRMaskService()
    result = service.persist_masks(
        sol="0921", target="Amherst_Point", scan="detail_1",
        despike_metadata=pre_result.metadata["despike"],
        database_path=Path("./phase.db"),
    )
    print(result.metadata["masks_inserted"], result.metadata["regions_skipped"])
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from rich.console import Console

from .base import ServiceResult
from .errors import CRMaskError, enrich

logger = logging.getLogger(__name__)

#: Spectrum row the masks attach to (spec §3.4): the DARK_SUBTRACTED row
#: is required in every Loupe workspace and is what the web spectra
#: routes serve by default.
PARENT_SPECTRUM_TYPE = "dark_subtracted"


class CRMaskService:
    """Service for persisting and retrieving cosmic-ray despike masks.

    Follows the house service pattern: ServiceResult-returning methods,
    ``SherlocServiceError``-derived (``CRMaskError``) failures, idempotent
    writes (delete-then-insert per (spectrum, method)).
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console if console is not None else Console()

    def persist_masks(
        self,
        sol: str,
        target: str,
        scan: str,
        despike_metadata: Dict[str, Any],
        database_path: Path,
    ) -> ServiceResult:
        """Persist a run's despike masks to the database.

        Resolves the DARK_SUBTRACTED spectrum row for each (scan_point,
        region) carried by ``despike_metadata["masks"]``, deletes any
        existing rows for those spectra with the same method, then inserts
        fresh rows (idempotent re-run; the (spectrum_id, method) unique
        constraint is never violated). Every (point, region) entry is
        persisted — including empty masks — so the stored set reconstructs
        the in-run mask set exactly (MLD-SYS-008 AC2).

        Args:
            sol: Sol number (e.g., "0921")
            target: Target name (e.g., "Amherst_Point")
            scan: Scan name (e.g., "detail_1")
            despike_metadata: The despike provenance dict recorded by the
                preprocessing service (keys: method, model_sha256, tau,
                masks; spec §4.3 step 4).
            database_path: Path to the phase.db SQLite database.

        Returns:
            ServiceResult with ``masks_inserted`` / ``regions_skipped`` and
            the provenance identity in metadata.

        Raises:
            CRMaskError: If the database or scan is missing, or the
                metadata lacks the provenance/mask fields.
        """
        from sherloc_pipeline.core.data_ingestion import normalize_target_name
        target = normalize_target_name(target)

        for required in ("method", "model_sha256", "tau", "masks"):
            if required not in despike_metadata:
                raise CRMaskError(
                    f"despike metadata missing required field {required!r}; "
                    "cannot persist masks without full provenance",
                    exit_code=1,
                    context={"sol": sol, "target": target, "scan": scan},
                )

        database_path = Path(database_path)
        if not database_path.exists():
            raise CRMaskError(
                f"Database not found: {database_path}",
                exit_code=1,
                context={"database_path": str(database_path)},
            )

        method = despike_metadata["method"]
        model_sha256 = despike_metadata["model_sha256"]
        tau_by_region = despike_metadata["tau"]
        masks_by_point = despike_metadata["masks"]

        try:
            from pydantic import ValidationError

            from sherloc_pipeline.database.connection import get_engine, get_session
            from sherloc_pipeline.database.models import (
                CosmicRayMaskORM,
                ScanORM,
                ScanPointORM,
                SpectrumORM,
            )
            from sherloc_pipeline.models.spectra import CosmicRayMask

            engine = get_engine(database_path)
            with get_session(engine) as session:
                # Target may use underscores (filesystem) or spaces (DB) —
                # same dance as the Raman persistence precedent.
                db_target = target.replace('_', ' ')
                scan_orm = session.query(ScanORM).filter_by(
                    sol_number=int(sol), target=db_target, scan_name=scan
                ).first()
                if scan_orm is None and db_target != target:
                    scan_orm = session.query(ScanORM).filter_by(
                        sol_number=int(sol), target=target, scan_name=scan
                    ).first()
                if scan_orm is None:
                    raise CRMaskError(
                        f"Scan not found in database: sol={sol}, "
                        f"target={target}, scan={scan}",
                        exit_code=1,
                        context={"sol": sol, "target": target, "scan": scan},
                    )

                scan_points = session.query(ScanPointORM).filter_by(
                    scan_id=scan_orm.id
                ).all()
                point_by_index = {sp.point_index: sp for sp in scan_points}

                masks_inserted = 0
                regions_skipped = 0

                for point_key, region_masks in masks_by_point.items():
                    point_index = int(point_key)
                    scan_point = point_by_index.get(point_index)
                    for region, channels in region_masks.items():
                        spectrum = None
                        if scan_point is not None:
                            spectrum = (
                                session.query(SpectrumORM)
                                .filter_by(
                                    scan_point_id=scan_point.id,
                                    region=region,
                                    spectrum_type=PARENT_SPECTRUM_TYPE,
                                )
                                .order_by(SpectrumORM.created_at)
                                .first()
                            )
                        if spectrum is None:
                            # Persistence follows parent-row existence
                            # (MLD-SYS-008 AC3): skip with a debug log; the
                            # mask stays in the run artifacts.
                            logger.debug(
                                "CR mask persistence: no %s spectrum row for "
                                "point %s region %s — skipped (mask retained "
                                "in run artifacts)",
                                PARENT_SPECTRUM_TYPE, point_index, region,
                            )
                            regions_skipped += 1
                            continue

                        # The pydantic domain model is the write-boundary
                        # contract: channel range/ordering and n_flagged
                        # invariants are enforced before any row is created.
                        try:
                            mask_model = CosmicRayMask(
                                spectrum_id=UUID(spectrum.id),
                                method=method,
                                model_sha256=model_sha256,
                                tau=float(tau_by_region[region]),
                                channel_indices=[int(c) for c in channels],
                                n_flagged=len(channels),
                            )
                        except ValidationError as exc:
                            raise CRMaskError(
                                f"Invalid mask for point {point_index} region "
                                f"{region}: {exc.errors()[0].get('msg', exc)}",
                                exit_code=1,
                                context={
                                    "point": point_index,
                                    "region": region,
                                },
                            )

                        # Idempotent re-run: replace this (spectrum, method)
                        session.query(CosmicRayMaskORM).filter_by(
                            spectrum_id=spectrum.id, method=method
                        ).delete()
                        session.add(CosmicRayMaskORM.from_pydantic(mask_model))
                        masks_inserted += 1

                metadata = {
                    "sol": sol,
                    "target": target,
                    "scan": scan,
                    "method": method,
                    "model_sha256": model_sha256,
                    "masks_inserted": masks_inserted,
                    "regions_skipped": regions_skipped,
                }
                return ServiceResult(
                    summary=(
                        f"Persisted {masks_inserted} cosmic-ray masks for "
                        f"scan {sol}/{target}/{scan}"
                        + (f" ({regions_skipped} skipped: no parent spectrum row)"
                           if regions_skipped else "")
                    ),
                    artifacts=[],
                    warnings=[],
                    metadata=metadata,
                )

        except CRMaskError:
            raise
        except Exception as e:
            error = CRMaskError(
                f"Failed to persist cosmic-ray masks: {e}",
                exit_code=1,
                context={"sol": sol, "target": target, "scan": scan},
            )
            raise enrich(error, sol=sol, target=target, scan=scan)

    @staticmethod
    def get_masks_for_scan(
        sol: str,
        target: str,
        scan: str,
        database_path: Path,
        method: Optional[str] = None,
    ) -> Dict[str, Dict[int, List[int]]]:
        """Fetch a scan's stored masks keyed by region and point index.

        Read path for stored-mask consumers that address data the way the
        pipeline does — (sol, target, scan) + point index — rather than by
        spectrum row id (the ``plot`` despike path, MLD-IFC-007). Stored
        masks are optional at read time (the communicated-state contract,
        MLD-IFC-003 AC2 / §4.7): a missing database, scan, or mask set
        returns ``{}`` so callers render non-despiked and say so, rather
        than erroring.

        Args:
            sol: Sol number (e.g., "0921").
            target: Target name (underscores or spaces; normalized the
                same way as :meth:`persist_masks`).
            scan: Scan name (e.g., "detail_1").
            database_path: Path to the phase.db SQLite database.
            method: Optional method filter; None returns the newest stored
                record per (point, region) across methods.

        Returns:
            ``{region: {point_index: sorted channel indices}}`` for every
            (point, region) with a stored mask record. Records with zero
            flagged channels are included — a stored empty mask still
            means "this spectrum was screened".
        """
        from sqlalchemy.exc import DatabaseError, OperationalError

        from sherloc_pipeline.core.data_ingestion import normalize_target_name

        database_path = Path(database_path)
        if not database_path.exists():
            return {}
        target = normalize_target_name(target)

        from sherloc_pipeline.database.connection import get_engine, get_session
        from sherloc_pipeline.database.models import (
            CosmicRayMaskORM,
            ScanORM,
            ScanPointORM,
            SpectrumORM,
        )

        try:
            engine = get_engine(database_path)
            with get_session(engine) as session:
                db_target = target.replace('_', ' ')
                scan_orm = session.query(ScanORM).filter_by(
                    sol_number=int(sol), target=db_target, scan_name=scan
                ).first()
                if scan_orm is None and db_target != target:
                    scan_orm = session.query(ScanORM).filter_by(
                        sol_number=int(sol), target=target, scan_name=scan
                    ).first()
                if scan_orm is None:
                    return {}

                query = (
                    session.query(CosmicRayMaskORM, SpectrumORM, ScanPointORM)
                    .join(SpectrumORM, CosmicRayMaskORM.spectrum_id == SpectrumORM.id)
                    .join(ScanPointORM, SpectrumORM.scan_point_id == ScanPointORM.id)
                    .filter(
                        ScanPointORM.scan_id == scan_orm.id,
                        SpectrumORM.spectrum_type == PARENT_SPECTRUM_TYPE,
                    )
                )
                if method is not None:
                    query = query.filter(CosmicRayMaskORM.method == method)
                # Newest record wins per (region, point) when several methods
                # (or re-runs) are stored and no method filter is given.
                query = query.order_by(CosmicRayMaskORM.created_at)

                masks: Dict[str, Dict[int, List[int]]] = {}
                for mask_row, spectrum, point in query.all():
                    masks.setdefault(spectrum.region, {})[point.point_index] = [
                        int(c) for c in mask_row.channel_indices
                    ]
                return masks
        except (OperationalError, DatabaseError) as exc:
            # A pre-migration database (no cosmic_ray_masks table yet), or
            # a SQLite file that is not a SHERLOC database at all. Stored
            # masks are optional at read time — treat exactly like "no
            # masks" so consumers render non-despiked with the note
            # instead of erroring (MLD-IFC-007).
            logger.debug(
                "Stored-mask lookup against %s failed (%s) — treating as "
                "no stored masks",
                database_path, type(exc).__name__,
            )
            return {}

    @staticmethod
    def get_masks_for_spectra(
        session,
        spectrum_ids: Iterable[str],
        method: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Fetch stored masks for a set of spectrum rows (shared read path).

        Args:
            session: An open SQLAlchemy session.
            spectrum_ids: Spectrum row ids (string UUIDs).
            method: Optional method filter (e.g. ``ml_v1.3_tau_matched``);
                None returns masks of every method.

        Returns:
            ``{spectrum_id: [CosmicRayMask, ...]}`` (pydantic domain
            models); spectrum ids with no stored mask are absent.
        """
        from sherloc_pipeline.database.models import CosmicRayMaskORM

        ids = list(spectrum_ids)
        if not ids:
            return {}
        query = session.query(CosmicRayMaskORM).filter(
            CosmicRayMaskORM.spectrum_id.in_(ids)
        )
        if method is not None:
            query = query.filter(CosmicRayMaskORM.method == method)

        out: Dict[str, List[Any]] = {}
        for row in query.all():
            out.setdefault(row.spectrum_id, []).append(row.to_pydantic())
        return out
