"""Bounded offline, one-point identification against pinned catalog templates."""

import hashlib
import io
import json
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy.orm import Session

from sherloc_pipeline.core.identification.compatibility import query_exclusion, reference_exclusion
from sherloc_pipeline.core.identification.decision import decide
from sherloc_pipeline.core.identification.inspection import inspection_fit
from sherloc_pipeline.core.identification.pearson import compare
from sherloc_pipeline.core.identification.prep import Prepared, grid_for, prepare
from sherloc_pipeline.core.identification.recipe import load_recipe
from sherloc_pipeline.core.identification.provenance import runtime_provenance
from sherloc_pipeline.database.reference_models import ReferenceFileORM, ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.identification import Candidate, IdentificationRequest, IdentificationResult, PointSelection
from sherloc_pipeline.models.references import ReferenceTemplate, ReferenceTrace, trace_key
from sherloc_pipeline.services.reference_import import array_archive_size, array_digest, digest, read_arrays
from sherloc_pipeline.services.reference_store import MAX_OBJECT_BYTES, ReferenceStore, verify_bytes
from sherloc_pipeline.services.identification_runtime import ExecutionFailed, IdentificationRuntime, WorkRejected, run_process
from sherloc_pipeline.models.feature_policy import FeaturePolicy, SourceDenylist
from sherloc_pipeline.services.reference_serving import can_serve_template

MAX_TEMPLATE_MEMBERS = 100
MAX_TOTAL_BYTES = MAX_OBJECT_BYTES
MAX_NATIVE_POINTS = 16384
MAX_TOTAL_POINTS = 262144


def deny_source(selection: PointSelection, denylist: SourceDenylist | None = None) -> None:
    """Apply the run's exclusions before opening a source, recipe or catalog."""
    denylist = denylist or SourceDenylist()
    normalize = lambda name: re.sub(r"[^a-z0-9]", "", name.casefold())
    product_sol = re.match(r"ss__(\d{4})_", selection.product_id.lower())
    if (selection.sol in denylist.sols
            or product_sol and int(product_sol[1]) in denylist.sols
            or selection.product_id.casefold() in {p.casefold() for p in denylist.product_ids}
            or normalize(selection.target) in {normalize(name) for name in denylist.target_names}):
        raise ValueError("source denied by run configuration")
    if product_sol and int(product_sol[1]) != selection.sol:
        raise ValueError("source sol identity mismatch")


class PointResolver(Protocol):
    def resolve_bounded(self, selection: PointSelection, timeout: float) -> tuple[np.ndarray, np.ndarray]: ...


class LocalPointResolver:
    """Read one explicit committed point NPZ, never a scan bag or browser array.

    Files are <product_id>-point-<index>.npz under the supplied root, with x, y
    and a JSON metadata scalar equal to the selection except source_sha256.
    """

    def __init__(self, root: Path, source_denylist: SourceDenylist | None = None):
        self.root = root.resolve()
        self.source_denylist = source_denylist or SourceDenylist()

    def resolve_bounded(self, selection, timeout):
        return run_process(_resolve_local_point, (self.root, self.source_denylist, selection), timeout)

    def resolve(self, selection: PointSelection):
        deny_source(selection, self.source_denylist)
        path = (self.root / f"{selection.product_id}-point-{selection.point_index}.npz").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("point source escapes root")
        with path.open("rb") as stream:
            data = verify_bytes(stream.read(MAX_OBJECT_BYTES + 1), selection.source_sha256)
        from types import SimpleNamespace

        if array_archive_size(data) > MAX_NATIVE_POINTS * 16 + 8192:
            raise WorkRejected("source point limit exceeded")
        arrays = read_arrays(data, [SimpleNamespace(x_key="x", y_key="y")])
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        if metadata != selection.model_dump(mode="json", exclude={"source_sha256"}):
            raise ValueError("point source metadata mismatch")
        return arrays["x"], arrays["y"]


def _resolve_local_point(root, denylist, selection):
    return LocalPointResolver(root, denylist).resolve(selection)


def load_reference(store, locator, sha256, trace):
    data = store.read_verified(locator, sha256)
    return read_arrays(data, [trace]), len(data)


class IdentificationService:
    def __init__(self, session: Session, store: ReferenceStore, resolver: PointResolver,
                 policy: FeaturePolicy, tier: str, runtime: IdentificationRuntime | None = None):
        self.session, self.store, self.resolver = session, store, resolver
        self.policy, self.tier = policy, tier
        self.runtime = runtime or IdentificationRuntime(policy.identification.max_concurrency)
        self._slots = self.runtime.slots

    def identify(self, request: IdentificationRequest) -> IdentificationResult:
        deny_source(request.selection, self.policy.identification.source_denylist)
        policy = self.policy
        if (self.tier != "team" or not policy.identification.enabled
                or not policy.identification.experimental_candidates or policy.references.mode == "off"):
            raise ValueError("experimental identification is disabled")
        if not self._slots.acquire(blocking=False):
            raise WorkRejected("identification concurrency limit reached")
        try:
            return self._identify(request)
        finally:
            self._slots.release()

    def _identify(self, request: IdentificationRequest) -> IdentificationResult:
        policy = self.policy
        started = time.monotonic()

        def check_time():
            if time.monotonic() - started > policy.identification.timeout_seconds:
                raise ExecutionFailed("identification time limit reached")

        if (request.reference_release_id != policy.references.release_id
                or request.reference_subset_id != policy.references.subset_id
                or not any(binding.method_release_id == request.method_release_id
                           and binding.reference_release_id == request.reference_release_id
                           and binding.reference_subset_id == request.reference_subset_id
                           for binding in policy.identification.allowed_method_releases)):
            raise ValueError("identification release binding mismatch")
        ids = request.template_ids
        if (ids != sorted(set(ids)) or len(ids) > policy.identification.max_templates
                or digest(ids) != request.reference_subset_id):
            raise ValueError("template subset identity or bound mismatch")
        recipe, recipe_sha256 = load_recipe()
        provenance = runtime_provenance()
        if recipe_sha256 != request.method_release_id:
            raise ValueError("method release is not the bundled frozen recipe")
        release = self.session.get(ReferenceReleaseORM, request.reference_release_id)
        if (release is None or release.status != "ready"
                or release.serving_policy_sha256 != policy.references.serving_policy_version):
            raise ValueError("reference release is unavailable")
        reason = query_exclusion(request.selection)
        if reason:
            raise ValueError(reason)
        # Authorize every catalog member before reading source or reference arrays.
        templates = []
        files = {}
        total_points = 0
        for template_id in ids:
            row = self.session.get(ReferenceTemplateORM, template_id)
            if row is None or len(row.member_trace_ids) > MAX_TEMPLATE_MEMBERS:
                raise ValueError("template missing or member bound exceeded")
            members = [self.session.get(ReferenceTraceORM, identity) for identity in row.member_trace_ids]
            if any(member is None for member in members) or not can_serve_template(
                row, members, release, policy, self.tier,
            ):
                raise ValueError("template is not permitted")
            metadata = ReferenceTemplate.model_validate(row.metadata_json)
            if len(metadata.members) != len(members) or any(
                member.source_key != trace_key(item.record_id, item.file_id, item.trace_index)
                or member.trace_index != item.trace_index
                for member, item in zip(members, metadata.members, strict=True)
            ):
                raise ValueError("template membership metadata mismatch")
            for member in members:
                if member.point_count > MAX_NATIVE_POINTS:
                    raise WorkRejected("reference point limit exceeded")
                total_points += member.point_count
                file = self.session.get(ReferenceFileORM, member.file_revision_id)
                if file is None or file.release_id != release.release_id:
                    raise ValueError("reference file is unavailable")
                files[file.file_revision_id] = file
            if total_points > MAX_TOTAL_POINTS:
                raise WorkRejected("identification point limit exceeded")
            templates.append((row, members))
            check_time()
        declared_bytes = sum((files[member.file_revision_id].metadata_json or {}).get("arrays_size", MAX_TOTAL_BYTES + 1)
                             for _, members in templates for member in members)
        if declared_bytes > MAX_TOTAL_BYTES:
            raise WorkRejected("identification byte limit exceeded")
        remaining = lambda: policy.identification.timeout_seconds - (time.monotonic() - started)
        x, y = self.resolver.resolve_bounded(request.selection, remaining())
        provenance_source = getattr(self.resolver, "provenance", None)
        if isinstance(provenance_source, dict):
            request = request.model_copy(update={"selection": request.selection.model_copy(update={
                "excitation_nm": provenance_source["excitation_nm"],
                "excitation_evidence": provenance_source["excitation_basis"]})})
        if len(x) > MAX_NATIVE_POINTS or len(y) > MAX_NATIVE_POINTS:
            raise WorkRejected("source point limit exceeded")
        key = (self.tier, digest(policy.model_dump(mode="json")), request.reference_release_id,
               request.reference_subset_id, request.method_release_id, request.selection.source_sha256,
               array_digest(x), array_digest(y), digest(request.model_dump(mode="json")), digest(provenance))
        # All members and the current source are rechecked even for a cached result.
        cached = self.runtime.cache.get(key)
        if cached is not None:
            return IdentificationResult.model_validate(cached)
        query = run_process(prepare, (np.array(x, copy=True), np.array(y, copy=True), recipe), remaining())
        detached = []
        for row, members in templates:
            detached_members = []
            metadata = ReferenceTemplate.model_validate(row.metadata_json)
            traces = [ReferenceTrace.model_validate(member.metadata_json) for member in members]
            compatible = (row.recipe_sha256 == recipe_sha256 and metadata.recipe_sha256 == recipe_sha256
                          and not metadata.compatibility_exclusions
                          and all(reference_exclusion(trace, request.selection) is None
                                  and trace.species_name == metadata.species_name for trace in traces))
            for member in members:
                trace = ReferenceTrace.model_validate(member.metadata_json)
                file = files[member.file_revision_id]
                detached_members.append({
                    "metadata_json": member.metadata_json, "source_key": member.source_key,
                    "trace_index": member.trace_index, "axis_sha256": member.axis_sha256,
                    "intensity_sha256": member.intensity_sha256,
                    "read_arrays": query.vector is not None and compatible,
                    "array_locator": file.array_locator, "array_sha256": file.array_sha256,
                    "point_count": member.point_count,
                })
            detached.append(({
                "template_id": row.template_id, "metadata_json": row.metadata_json,
                "recipe_sha256": row.recipe_sha256, "member_trace_ids": row.member_trace_ids,
            }, detached_members))
        read_store = self.store if any(member["read_arrays"] for _, members in detached for member in members) else None
        result = run_process(load_and_compute_identification,
                             (read_store, request, recipe, recipe_sha256, provenance, query, detached), remaining())
        self.runtime.cache.put(key, result.model_dump(mode="json"))
        return result


def load_and_compute_identification(store, request, recipe, recipe_sha256, provenance, query, templates):
    bytes_read = 0
    for _, members in templates:
        for member in members:
            member["arrays"] = {}
            if member["read_arrays"]:
                trace = ReferenceTrace.model_validate(member["metadata_json"])
                arrays, size = load_reference(store, member["array_locator"], member["array_sha256"], trace)
                bytes_read += size
                if bytes_read > MAX_TOTAL_BYTES:
                    raise WorkRejected("identification byte limit exceeded")
                if len(arrays[trace.x_key]) != member["point_count"]:
                    raise ValueError("reference point count mismatch")
                member["arrays"] = arrays
    return compute_identification(request, recipe, recipe_sha256, provenance, query, templates)


def compute_identification(request, recipe, recipe_sha256, provenance, query, templates):
    from types import SimpleNamespace

    ranking, matches, exclusions = [], {}, []
    for template_data, member_data in templates:
        row = SimpleNamespace(**template_data)
        members = [SimpleNamespace(**item) for item in member_data]
        metadata = ReferenceTemplate.model_validate(row.metadata_json)
        reason = None
        if (row.recipe_sha256 != recipe_sha256 or metadata.recipe_sha256 != recipe_sha256
                or metadata.compatibility_exclusions):
            reason = "incompatible_template_recipe"
        traces = [ReferenceTrace.model_validate(member.metadata_json) for member in members]
        reason = reason or next((r for trace in traces
                                if (r := reference_exclusion(trace, request.selection))), None)
        if any(trace.species_name != metadata.species_name for trace in traces):
            reason = reason or "mixed_template_species"
        if len(metadata.members) != len(members) or any(
            member.source_key != trace_key(item.record_id, item.file_id, item.trace_index)
            or member.trace_index != item.trace_index
            for member, item in zip(members, metadata.members, strict=True)
        ):
            raise ValueError("template membership metadata mismatch")
        prepared = []
        if not reason and query.vector is not None:
            for member, trace in zip(members, traces, strict=True):
                arrays = member.arrays
                if (array_digest(arrays[trace.x_key]) != member.axis_sha256
                        or array_digest(arrays[trace.y_key]) != member.intensity_sha256):
                    raise ValueError("reference trace digest mismatch")
                item = prepare(arrays[trace.x_key], arrays[trace.y_key], recipe)
                if item.vector is None:
                    reason = item.diagnostics["reason"]
                    break
                prepared.append(item)
        if reason:
            exclusions.append({"template_id": row.template_id, "reason": reason})
            continue
        if not prepared:
            continue
        mask = np.logical_and.reduce([item.mask for item in prepared])
        vector = np.average([item.vector for item in prepared], axis=0,
                            weights=[member.weight for member in metadata.members])
        reference = Prepared(vector, mask, {})
        match = compare(query, reference, recipe)
        if match is None:
            exclusions.append({"template_id": row.template_id, "reason": "insufficient_joint_overlap"})
            continue
        ranking.append(Candidate(template_id=row.template_id, species_name=metadata.species_name,
                                 supported_class=metadata.supported_class,
                                 trace_revision_ids=list(row.member_trace_ids),
                                 support=match["support"], shift_cm_1=match["shift_cm_1"],
                                 extra_fwhm_cm1=match["extra_fwhm_cm1"],
                                 citations=[{"trace_revision_id": identity, "citation_text": trace.citation_text,
                                             "citation_doi": trace.citation_doi, "citation_source": trace.citation_source,
                                             "license_identifier": trace.license_identifier,
                                             "license_text": trace.license_text, "license_source": trace.license_source}
                                            for identity, trace in zip(row.member_trace_ids, traces, strict=True)]))
        matches[row.template_id] = match
    # Freeze one best template per species, retaining signed support and tie identity.
    ranking.sort(key=lambda item: (-item.support, item.species_name, item.template_id))
    species = set()
    ranking = [item for item in ranking if not (item.species_name in species or species.add(item.species_name))]
    status, reasons = decide(ranking, query.diagnostics, recipe)
    evidence = {"grid": grid_for(recipe).tolist(), "mask": query.mask.tolist(),
                "prepared_trace": None if query.vector is None else query.vector.tolist(),
                "inspection": None}
    if ranking:
        match = matches[ranking[0].template_id]
        evidence["inspection"] = inspection_fit(query.vector, match["vector"], match["mask"])
    payload = {
        "status": status, "reasons": reasons,
        "method": {**recipe.model_dump(mode="json"), "recipe_sha256": recipe_sha256,
                   "release_id": recipe_sha256, **provenance},
        "selection": request.selection.model_dump(mode="json"),
        "reference_release_id": request.reference_release_id,
        "reference_subset_id": request.reference_subset_id,
        "effective_set_sha256": digest(sorted(matches)),
        "ranking": [item.model_dump(mode="json") for item in ranking],
        "candidate": ranking[0].species_name if status == "candidate" else None,
        "diagnostics": {**query.diagnostics, "excluded_references": exclusions},
        "evidence": evidence,
    }
    return IdentificationResult.model_validate({**payload, "result_digest": digest(payload)})
