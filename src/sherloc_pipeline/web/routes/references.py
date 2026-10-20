"""Authenticated, policy-filtered reference metadata and native traces."""

import base64
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from sherloc_pipeline.core.identification.recipe import recipe_for_release
from sherloc_pipeline.database.reference_models import ReferenceFileORM, ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.references import Digest, ReferenceTrace
from sherloc_pipeline.services.identification import MAX_NATIVE_POINTS, MAX_TOTAL_BYTES, load_reference
from sherloc_pipeline.services.identification_runtime import ExecutionFailed, WorkRejected, run_process
from sherloc_pipeline.services.reference_import import array_digest, digest
from sherloc_pipeline.services.reference_store import LocalReferenceStore
from sherloc_pipeline.web.reference_storage import R2ReferenceStore, can_serve_template, can_serve_trace

router = APIRouter(prefix="/api/references", tags=["references"])

MAX_BINDING_MEMBERS = 10_000
BINDING_QUERY_CHUNK = 500


def require_references(request):
    policy = request.app.state.feature_policy
    if request.app.state.access_mode != "internal" or policy.references.mode == "off":
        raise HTTPException(404, "References are disabled")
    return policy


def reject_query(request, allowed):
    if set(request.query_params) - allowed:
        raise HTTPException(422, "Unsupported query parameter")


def permitted_release(request, release_id):
    policy = require_references(request)
    if release_id != policy.references.release_id:
        raise HTTPException(404, "Reference release unavailable")
    release = request.state.db.get(ReferenceReleaseORM, release_id)
    if (release is None or release.status != "ready"
            or release.serving_policy_sha256 != policy.references.serving_policy_version):
        raise HTTPException(404, "Reference release unavailable")
    return policy, release


def reference_store(request):
    require_references(request)
    injected = getattr(request.app.state, "reference_store", None)
    if injected is not None:
        return injected
    if request.app.state.feature_policy.references.storage_backend == "r2":
        return R2ReferenceStore("team")
    root = os.environ.get("SHERLOC_REFERENCE_OBJECT_ROOT")
    if not root:
        raise HTTPException(503, "Reference storage unavailable")
    return LocalReferenceStore(Path(root))


def trace_metadata(row):
    metadata = ReferenceTrace.model_validate(row.metadata_json).model_dump(mode="json")
    # Array keys are internal storage details; identities and provenance suffice.
    for key in ("x_key", "y_key"):
        metadata.pop(key)
    return {"trace_revision_id": row.trace_revision_id, "release_id": row.release_id,
            "axis_sha256": row.axis_sha256, "intensity_sha256": row.intensity_sha256,
            "point_count": row.point_count, "provenance": metadata}


def identification_binding(request, policy, release):
    identify = policy.identification
    if not identify.enabled or not identify.experimental_candidates:
        return None
    ids = policy.references.template_ids
    if len(ids) > identify.max_templates:
        return None
    query = request.state.db.query(ReferenceTemplateORM).filter_by(release_id=release.release_id)
    if ids:
        query = query.filter(ReferenceTemplateORM.template_id.in_(ids))
    rows = query.order_by(ReferenceTemplateORM.template_id).limit(identify.max_templates + 1).all()
    if ids and [row.template_id for row in rows] != ids:
        return None
    ids = [row.template_id for row in rows]
    if not ids or len(ids) > identify.max_templates or digest(ids) != policy.references.subset_id:
        return None
    # Bound total authorization work before loading any member records.
    if any(len(row.member_trace_ids) > 100 for row in rows) or sum(
        len(row.member_trace_ids) for row in rows
    ) > MAX_BINDING_MEMBERS:
        return None
    member_ids = sorted({identity for row in rows for identity in row.member_trace_ids})
    members_by_id = {}
    for start in range(0, len(member_ids), BINDING_QUERY_CHUNK):
        members = request.state.db.query(ReferenceTraceORM).filter(
            ReferenceTraceORM.release_id == release.release_id,
            ReferenceTraceORM.trace_revision_id.in_(member_ids[start:start + BINDING_QUERY_CHUNK]),
        ).all()
        members_by_id.update((member.trace_revision_id, member) for member in members)
    for row in rows:
        members = [members_by_id.get(identity) for identity in row.member_trace_ids]
        if any(member is None for member in members) or not can_serve_template(row, members, release, policy, "team"):
            return None
    bindings = []
    for binding in identify.allowed_method_releases:
        if (binding.reference_release_id != release.release_id
                or binding.reference_subset_id != policy.references.subset_id):
            continue
        try:
            recipe_for_release(binding.method_release_id)
        except ValueError:
            continue
        bindings.append(binding)
    if len(bindings) != 1:
        return None
    binding = bindings[0]
    return {**binding.model_dump(mode="json"), "template_ids": ids}


@router.get("")
def list_references(request: Request, release_id: Digest,
                    modality: Literal["raman", "fluorescence", "unknown"] | None = None,
                    cursor: Annotated[str | None, Query(max_length=512)] = None,
                    limit: Annotated[int, Query(ge=1, le=100)] = 25):
    reject_query(request, {"release_id", "modality", "cursor", "limit"})
    policy, release = permitted_release(request, release_id)
    scope = digest(["team", policy.model_dump(mode="json"), release_id, modality])
    after = ""
    if cursor:
        try:
            token = json.loads(base64.urlsafe_b64decode(cursor))
            after = token["after"]
            if token["scope"] != scope or len(after) != 64 or any(c not in "0123456789abcdef" for c in after):
                raise ValueError
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, "Invalid reference cursor") from exc
    query = request.state.db.query(ReferenceTraceORM).filter(
        ReferenceTraceORM.release_id == release_id, ReferenceTraceORM.trace_revision_id > after,
        ReferenceTraceORM.rights_known.is_(True), ReferenceTraceORM.serving_status == "approved")
    if policy.references.mode == "allowlist":
        query = query.filter(ReferenceTraceORM.trace_revision_id.in_(policy.references.allowed_trace_ids))
    # Bound both the database page and returned page, including sparse permissions.
    rows = query.order_by(ReferenceTraceORM.trace_revision_id).limit(limit + 1).all()
    items = [trace_metadata(row) for row in rows[:limit]
             if can_serve_trace(row, release, policy, "team")
             and (modality is None or row.metadata_json.get("modality") == modality)]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = base64.urlsafe_b64encode(json.dumps({"scope": scope, "after": rows[limit - 1].trace_revision_id}).encode()).decode()
    return {"items": items, "next_cursor": next_cursor,
            "binding": identification_binding(request, policy, release)}


@router.get("/{trace_revision_id}")
def get_reference(request: Request, trace_revision_id: Digest, release_id: Digest):
    reject_query(request, {"release_id"})
    policy, release = permitted_release(request, release_id)
    row = request.state.db.get(ReferenceTraceORM, trace_revision_id)
    if row is None or not can_serve_trace(row, release, policy, "team"):
        raise HTTPException(404, "Reference trace unavailable")
    file = request.state.db.get(ReferenceFileORM, row.file_revision_id)
    if file is None or file.release_id != release_id:
        raise HTTPException(404, "Reference file unavailable")
    if row.point_count > MAX_NATIVE_POINTS or (file.metadata_json or {}).get("arrays_size", MAX_TOTAL_BYTES + 1) > MAX_TOTAL_BYTES:
        raise HTTPException(422, "Reference admission limit exceeded")
    key = ("team", digest(policy.model_dump(mode="json")), release_id, trace_revision_id,
           row.axis_sha256, row.intensity_sha256, file.array_sha256, None)
    cache = request.app.state.reference_cache
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        trace = ReferenceTrace.model_validate(row.metadata_json)
        slots = request.app.state.identification_runtime.slots
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Reference concurrency limit reached")
        try:
            arrays, _ = run_process(load_reference, (reference_store(request), file.array_locator, file.array_sha256, trace),
                                    policy.identification.timeout_seconds)
        finally:
            slots.release()
        x, y = arrays[trace.x_key], arrays[trace.y_key]
        if len(x) != row.point_count or array_digest(x) != row.axis_sha256 or array_digest(y) != row.intensity_sha256:
            raise ValueError("reference digest mismatch")
        result = {**trace_metadata(row), "x": x.tolist(), "y": y.tolist()}
    except WorkRejected as exc:
        raise HTTPException(429, str(exc)) from exc
    except (ExecutionFailed, ValueError, OSError) as exc:
        raise HTTPException(503, {"code": "execution_failed", "detail": "Reference execution failed"}) from exc
    cache.put(key, result)
    return result
