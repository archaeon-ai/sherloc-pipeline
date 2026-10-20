"""Bounded offline, one-point identification against pinned catalog templates."""

import hashlib
import io
import json
import re
import threading
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
from sherloc_pipeline.services.reference_import import array_digest, digest, read_arrays
from sherloc_pipeline.services.reference_store import MAX_OBJECT_BYTES, ReferenceStore, verify_bytes
from sherloc_pipeline.models.feature_policy import FeaturePolicy, SourceDenylist
from sherloc_pipeline.services.reference_serving import can_serve_template

MAX_TEMPLATE_MEMBERS = 100
MAX_TOTAL_BYTES = MAX_OBJECT_BYTES


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
    def resolve(self, selection: PointSelection) -> tuple[np.ndarray, np.ndarray]: ...


class LocalPointResolver:
    """Read one explicit committed point NPZ, never a scan bag or browser array.

    Files are <product_id>-point-<index>.npz under the supplied root, with x, y
    and a JSON metadata scalar equal to the selection except source_sha256.
    """

    def __init__(self, root: Path, source_denylist: SourceDenylist | None = None):
        self.root = root.resolve()
        self.source_denylist = source_denylist or SourceDenylist()

    def resolve(self, selection: PointSelection):
        deny_source(selection, self.source_denylist)
        path = (self.root / f"{selection.product_id}-point-{selection.point_index}.npz").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("point source escapes root")
        with path.open("rb") as stream:
            data = verify_bytes(stream.read(MAX_OBJECT_BYTES + 1), selection.source_sha256)
        from types import SimpleNamespace

        arrays = read_arrays(data, [SimpleNamespace(x_key="x", y_key="y")])
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        if metadata != selection.model_dump(mode="json", exclude={"source_sha256"}):
            raise ValueError("point source metadata mismatch")
        return arrays["x"], arrays["y"]


class IdentificationService:
    def __init__(self, session: Session, store: ReferenceStore, resolver: PointResolver,
                 policy: FeaturePolicy, tier: str):
        self.session, self.store, self.resolver = session, store, resolver
        self.policy, self.tier = policy, tier
        self._slots = threading.BoundedSemaphore(policy.identification.max_concurrency)

    def identify(self, request: IdentificationRequest) -> IdentificationResult:
        deny_source(request.selection, self.policy.identification.source_denylist)
        policy = self.policy
        if (self.tier != "team" or not policy.identification.enabled
                or not policy.identification.experimental_candidates or policy.references.mode == "off"):
            raise ValueError("experimental identification is disabled")
        if not self._slots.acquire(blocking=False):
            raise ValueError("identification concurrency limit reached")
        try:
            return self._identify(request)
        finally:
            self._slots.release()

    def _identify(self, request: IdentificationRequest) -> IdentificationResult:
        policy = self.policy
        started = time.monotonic()

        def check_time():
            if time.monotonic() - started > policy.identification.timeout_seconds:
                raise ValueError("identification time limit reached")

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
        for template_id in ids:
            row = self.session.get(ReferenceTemplateORM, template_id)
            if row is None or len(row.member_trace_ids) > MAX_TEMPLATE_MEMBERS:
                raise ValueError("template missing or member bound exceeded")
            members = [self.session.get(ReferenceTraceORM, identity) for identity in row.member_trace_ids]
            if any(member is None for member in members) or not can_serve_template(
                row, members, release, policy, self.tier,
            ):
                raise ValueError("template is not permitted")
            templates.append((row, members))
            check_time()
        x, y = self.resolver.resolve(request.selection)
        query = prepare(x, y, recipe)
        check_time()
        ranking, matches, exclusions = [], {}, []
        bytes_read = 0
        for row, members in templates:
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
                    file = self.session.get(ReferenceFileORM, member.file_revision_id)
                    if file is None or file.release_id != release.release_id:
                        raise ValueError("reference file is unavailable")
                    data = self.store.read_verified(file.array_locator, file.array_sha256)
                    bytes_read += len(data)
                    if bytes_read > MAX_TOTAL_BYTES:
                        raise ValueError("identification byte limit exceeded")
                    arrays = read_arrays(data, [trace])
                    if (array_digest(arrays[trace.x_key]) != member.axis_sha256
                            or array_digest(arrays[trace.y_key]) != member.intensity_sha256):
                        raise ValueError("reference trace digest mismatch")
                    item = prepare(arrays[trace.x_key], arrays[trace.y_key], recipe)
                    if item.vector is None:
                        reason = item.diagnostics["reason"]
                        break
                    prepared.append(item)
                    check_time()
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
                                     extra_fwhm_cm1=match["extra_fwhm_cm1"]))
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
        check_time()
        return IdentificationResult.model_validate({**payload, "result_digest": digest(payload)})
