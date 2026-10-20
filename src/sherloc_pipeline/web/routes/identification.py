"""One stored point, pinned references and a frozen provisional method."""

import re
import zlib
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field
from sqlalchemy import func
from sqlalchemy.orm import load_only

from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SpectrumORM
from sherloc_pipeline.models.identification import IdentificationModel, IdentificationRequest, PointSelection
from sherloc_pipeline.models.references import Digest
from sherloc_pipeline.services.identification import IdentificationService, MAX_NATIVE_POINTS, deny_source
from sherloc_pipeline.services.identification_runtime import ExecutionFailed, WorkRejected, run_process
from sherloc_pipeline.services.identification_source import prepare_stored_point
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.routes.references import reference_store, require_references, reject_query
from sherloc_pipeline.web.routes.spectra import _resolve_spectrum_type, _wavelength_filter_info

router = APIRouter(prefix="/api", tags=["identification"])


class StoredPoint(IdentificationModel):
    mode: Literal["point"]
    index: Annotated[int, Field(ge=0)]


class IdentifyBody(IdentificationModel):
    scan_id: Annotated[str, Field(min_length=1, max_length=200)]
    selection: StoredPoint
    region: Literal["R1"]
    spectrum_type: Literal["dark_subtracted", "laser_normalized"]
    expected_source_digest: Digest
    reference_release_id: Digest
    reference_subset_id: Digest
    method_release_id: Digest
    template_ids: list[Digest] = Field(min_length=1, max_length=10000)


class StoredPointResolver:
    def __init__(self, request, body):
        self.request, self.body = request, body
        self.provenance = None

    def resolve_bounded(self, selection, timeout):
        body, request = self.body, self.request
        query = request.state.db.query(SpectrumORM.id).join(ScanPointORM).filter(
            ScanPointORM.scan_id == body.scan_id, ScanPointORM.point_index == body.selection.index,
            SpectrumORM.region == body.region, SpectrumORM.spectrum_type == body.spectrum_type)
        ids = [row[0] for row in query.limit(2).all()]
        if len(ids) != 1:
            raise HTTPException(422, "Exactly one stored source spectrum is required")
        lengths = request.state.db.query(func.length(SpectrumORM.intensities), func.length(SpectrumORM.wavelengths)).filter(
            SpectrumORM.id == ids[0]).one()
        if any(size is not None and size > MAX_NATIVE_POINTS * 8 + 1024 for size in lengths):
            raise WorkRejected("source byte limit exceeded")
        row = request.state.db.query(SpectrumORM).options(load_only(
            SpectrumORM.id, SpectrumORM.scan_point_id, SpectrumORM.region, SpectrumORM.spectrum_type,
            SpectrumORM.processing_level, SpectrumORM.intensities, SpectrumORM.wavelengths, SpectrumORM.wavelength_source,
        )).filter(SpectrumORM.id == ids[0]).one()
        scan = request.state.db.get(ScanORM, body.scan_id)
        point = request.state.db.get(ScanPointORM, row.scan_point_id)
        max_pd = request.state.db.query(func.max(ScanPointORM.photodiode_mean)).filter(
            ScanPointORM.scan_id == body.scan_id, ScanPointORM.photodiode_mean > 0,
            ScanPointORM.photodiode_mean < float("inf")).scalar()
        fields = ("id", "sol_number", "target", "scan_name", "data_source", "pds4_metadata", "loupe_metadata",
                  "laser_wavelength_nm", "processing_applied", "processing_pipeline_version", "processing_config_hash", "n_points")
        detached = {"scan": {key: getattr(scan, key) for key in fields},
                    "point": {"point_index": point.point_index, "photodiode_mean": point.photodiode_mean},
                    "spectrum": {key: getattr(row, key) for key in
                                 ("id", "region", "spectrum_type", "processing_level", "intensities", "wavelengths", "wavelength_source")},
                    "max_photodiode": max_pd, "wavelength_filter": _wavelength_filter_info("R1")}
        x, y, self.provenance = run_process(prepare_stored_point, (detached, MAX_NATIVE_POINTS), timeout)
        if self.provenance["source_digest"] != body.expected_source_digest:
            raise HTTPException(409, "Source digest changed; reload the stored selection")
        return x, y



@router.post("/identify")
def identify(request: Request, body: IdentifyBody):
    reject_query(request, set())
    policy = require_references(request)
    if not policy.identification.enabled or not policy.identification.experimental_candidates:
        raise HTTPException(404, "Identification is disabled")
    runtime = request.app.state.identification_runtime
    try:
        runtime.admit_rate(request.client.host if request.client else "unknown")
        scan = request.state.db.get(ScanORM, body.scan_id)
        if scan is None:
            raise HTTPException(404, "Source scan unavailable")
        DataAccessService(request.app.state.access_mode).validate_scan_access(scan)
        if scan.n_channels > MAX_NATIVE_POINTS:
            raise WorkRejected("source point limit exceeded")
        if body.selection.index >= scan.n_points or body.spectrum_type != _resolve_spectrum_type(scan):
            raise HTTPException(422, "Unsupported stored selection")
        metadata = (scan.pds4_metadata if scan.data_source == "pds4" else scan.loupe_metadata) or {}
        products = list(dict.fromkeys(value for value in (metadata.get("product_id"), scan.scan_id, scan.id)
                                     if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value)))
        selection = PointSelection(
            product_id=products[0] if products else scan.id, sol=scan.sol_number, target=scan.target or scan.scan_name,
            point_index=body.selection.index, source_sha256=body.expected_source_digest,
            modality="raman", x_quantity="raman_shift", x_unit="cm-1",
            excitation_nm=float(scan.laser_wavelength_nm), excitation_evidence="scan_record_or_nominal_default")
        # An allowed metadata alias cannot hide a denied persisted identity.
        for product in products or [scan.id]:
            deny_source(selection.model_copy(update={"product_id": product}), policy.identification.source_denylist)
        resolver = StoredPointResolver(request, body)
        service = IdentificationService(request.state.db, reference_store(request), resolver, policy, "team", runtime)
        result = service.identify(IdentificationRequest(
            selection=selection, reference_release_id=body.reference_release_id,
            reference_subset_id=body.reference_subset_id, method_release_id=body.method_release_id,
            template_ids=body.template_ids))
        return {**result.model_dump(mode="json"), "source_digest": body.expected_source_digest,
                "input": resolver.provenance}
    except WorkRejected as exc:
        raise HTTPException(429, str(exc)) from exc
    except ExecutionFailed as exc:
        raise HTTPException(503, {"code": "execution_failed", "detail": str(exc)}) from exc
    except (ValueError, OSError, zlib.error) as exc:
        raise HTTPException(422, "Identification request or stored evidence is invalid") from exc
