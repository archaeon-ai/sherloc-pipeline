"""Validate explicit local snapshots, stage objects, then commit a ready release."""

import hashlib
import io
import json
import math
import tokenize
import zipfile
import zlib
from functools import wraps
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from sherloc_pipeline.database.reference_models import ReferenceFileORM, ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.references import ReferenceSnapshot, ServingPolicy, known_rights, trace_key
from sherloc_pipeline.services.reference_store import MAX_OBJECT_BYTES, LocalReferenceStore, object_locator, verify_bytes

IMPORTER_VERSION = "reference-import-1"
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_RELEASE_BYTES = 256 * 1024 * 1024
REFERENCE_REVISION = "a13c4e5f6071"


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_local(root: Path, relative: str, expected: str, size: int) -> bytes:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("snapshot path must be relative and contained")
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("snapshot path escapes root or is not a file")
    if size > MAX_OBJECT_BYTES:
        raise ValueError("reference file too large")
    with path.open("rb") as stream:
        data = stream.read(MAX_OBJECT_BYTES + 1)
    if len(data) != size:
        raise ValueError("reference byte count mismatch")
    return verify_bytes(data, expected)



def _array_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (zipfile.BadZipFile, EOFError, OSError, KeyError, IndexError, TypeError,
                tokenize.TokenError, zlib.error, NotImplementedError, RuntimeError) as exc:
            raise ValueError("malformed reference array archive") from exc
    return checked


@_array_errors
def array_archive_size(data: bytes) -> int:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        size = sum(entry.file_size for entry in entries)
        if len(entries) > 20000 or size > MAX_OBJECT_BYTES:
            raise ValueError("reference array archive too large")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("duplicate array archive member")
        return size


@_array_errors
def read_arrays(data: bytes, traces) -> dict:
    array_archive_size(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported reference array header version")
                size = math.prod(shape) * dtype.itemsize
                if any(dimension < 0 for dimension in shape) or dtype.hasobject:
                    raise ValueError("invalid reference array header")
                if size > min(member.file_size - stream.tell(), MAX_OBJECT_BYTES, MAX_RELEASE_BYTES):
                    raise ValueError("reference array header exceeds byte limits")
    arrays = {}
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        for trace in traces:
            for key in (trace.x_key, trace.y_key):
                if key in arrays:
                    continue
                array = archive[key]
                if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                    raise ValueError("reference arrays must be finite nonempty numeric vectors")
                arrays[key] = array
            if len(arrays[trace.x_key]) != len(arrays[trace.y_key]):
                raise ValueError("reference axis/intensity length mismatch")
    return arrays


def array_digest(array) -> str:
    return digest({"dtype": array.dtype.str, "shape": list(array.shape),
                   "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()})


@dataclass
class ImportReport:
    release_id: str
    files: int
    traces: int
    templates: int
    excluded: int
    missing_citation: int
    unknown_rights: int
    team_permitted: int
    public_permitted: int = 0
    existing: bool = False
    revised_files: int = 0


def import_release(
    snapshot: Path, database: Path, objects: Path, serving_policy: Path | None = None,
    dry_run: bool = False,
) -> ImportReport:
    root = snapshot.resolve()
    manifest_path = root / "snapshot.json"
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError("snapshot manifest escapes root")
    with manifest_path.open("rb") as stream:
        raw_manifest = stream.read(MAX_SNAPSHOT_BYTES + 1)
    if len(raw_manifest) > MAX_SNAPSHOT_BYTES:
        raise ValueError("reference manifest too large")
    manifest = ReferenceSnapshot.model_validate_json(raw_manifest)
    policy = ServingPolicy()
    if serving_policy is not None:
        with serving_policy.open("rb") as stream:
            raw_policy = stream.read(MAX_SNAPSHOT_BYTES + 1)
        if len(raw_policy) > MAX_SNAPSHOT_BYTES:
            raise ValueError("reference policy too large")
        policy = ServingPolicy.model_validate_json(raw_policy)
    metadata = manifest.model_dump(mode="json")
    policy_data = policy.model_dump(mode="json")
    policy_sha = digest(policy_data)
    release_id = digest({"snapshot": metadata, "serving_policy": policy_data, "importer": IMPORTER_VERSION})
    keys = [trace_key(file.record_id, file.file_id, trace.trace_index)
            for file in manifest.files for trace in file.traces]
    file_keys = [(file.record_id, file.file_id) for file in manifest.files]
    if len(set(keys)) != len(keys) or len(set(file_keys)) != len(file_keys):
        raise ValueError("duplicate reference source identity")
    if len(set(policy.approved_trace_keys)) != len(policy.approved_trace_keys) or not set(policy.approved_trace_keys) <= set(keys):
        raise ValueError("serving policy has duplicate or unknown source keys")
    objects_to_stage = {}
    files, traces, templates = [], [], []
    trace_by_key = {}
    total_bytes = 0
    specimens = {trace.specimen_key for file in manifest.files for trace in file.traces
                 if trace.column_role == "measurement"}
    for file in manifest.files:
        raw = read_local(root, file.raw_path, file.raw_sha256, file.raw_size)
        array_bytes = read_local(root, file.arrays_path, file.arrays_sha256, file.arrays_size)
        total_bytes += len(raw) + len(array_bytes) + array_archive_size(array_bytes)
        if total_bytes > MAX_RELEASE_BYTES:
            raise ValueError("reference release too large")
        arrays = read_arrays(array_bytes, file.traces)
        array_hashes = {key: array_digest(array) for key, array in arrays.items()}
        file_metadata = file.model_dump(mode="json", exclude={"traces"})
        file_revision = digest({"release_id": release_id, "file": file_metadata})
        objects_to_stage[file.raw_sha256] = raw
        objects_to_stage[file.arrays_sha256] = array_bytes
        files.append(ReferenceFileORM(
            file_revision_id=file_revision, release_id=release_id, datatype_id=file.datatype_id,
            record_id=file.record_id, file_id=file.file_id,
            raw_locator=object_locator(file.raw_sha256), raw_sha256=file.raw_sha256, raw_size=file.raw_size,
            array_locator=object_locator(file.arrays_sha256), array_sha256=file.arrays_sha256,
            metadata_json=file_metadata,
        ))
        for trace in file.traces:
            key = trace_key(file.record_id, file.file_id, trace.trace_index)
            x_hash, y_hash = array_hashes[trace.x_key], array_hashes[trace.y_key]
            trace_metadata = trace.model_dump(mode="json")
            trace_id = digest({"file_revision": file_revision, "trace": trace_metadata,
                               "axis": x_hash, "intensity": y_hash})
            exclusions = list(trace.qa_flags)
            if trace.column_role == "uncertainty":
                exclusions.append("uncertainty_column")
            if trace.column_role == "source_mean" and trace.specimen_key in specimens:
                exclusions.append("source_mean_with_individual_measurements")
            if trace.x_unit == "unknown" or not trace.x_evidence or not trace.y_unit:
                exclusions.append("unknown_units_or_evidence")
            approved = key in policy.approved_trace_keys and known_rights(trace) and not exclusions
            row = ReferenceTraceORM(
                trace_revision_id=trace_id, release_id=release_id, file_revision_id=file_revision,
                trace_index=trace.trace_index, source_key=key, point_count=len(arrays[trace.x_key]),
                axis_sha256=x_hash, intensity_sha256=y_hash, rights_known=known_rights(trace),
                serving_status="excluded" if exclusions else "approved" if approved else "off",
                permitted_tiers=["team"] if approved else [], exclusions=exclusions,
                metadata_json=trace_metadata,
            )
            traces.append(row)
            trace_by_key[key] = row
    for template in manifest.templates:
        members = [trace_by_key.get(trace_key(member.record_id, member.file_id, member.trace_index))
                   for member in template.members]
        if any(member is None for member in members):
            raise ValueError("template member is not in this release")
        member_ids = [member.trace_revision_id for member in members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("duplicate template member")
        template_metadata = template.model_dump(mode="json")
        templates.append(ReferenceTemplateORM(
            template_id=digest({"release_id": release_id, "template": template_metadata,
                                "member_trace_ids": member_ids}),
            release_id=release_id, name=template.name, recipe_sha256=template.recipe_sha256,
            member_trace_ids=member_ids,
            permitted_tiers=["team"] if all("team" in member.permitted_tiers for member in members)
            and not template.compatibility_exclusions else [],
            metadata_json=template_metadata,
        ))
    if len({template.name for template in templates}) != len(templates):
        raise ValueError("duplicate template name")
    report = ImportReport(
        release_id, len(files), len(traces), len(templates), sum(bool(t.exclusions) for t in traces),
        sum(not (t.metadata_json["citation_text"] or t.metadata_json["citation_doi"]) for t in traces),
        sum(not t.rights_known for t in traces), sum("team" in t.permitted_tiers for t in traces),
    )
    # Opening SQLite through mode=ro keeps a dry run from creating a database.
    engine = None
    if database.exists():
        engine = create_engine(f"sqlite:///file:{database.resolve()}?mode=ro&uri=true")
        try:
            with Session(engine) as session:
                if "reference_releases" not in inspect(engine).get_table_names():
                    raise ValueError("reference catalog migration is required")
                existing = session.get(ReferenceReleaseORM, release_id)
                report.existing = existing is not None
                report.revised_files = sum(session.scalar(select(ReferenceFileORM.file_revision_id).where(
                    ReferenceFileORM.record_id == file.record_id, ReferenceFileORM.file_id == file.file_id,
                    ReferenceFileORM.raw_sha256 != file.raw_sha256,
                ).limit(1)) is not None for file in files)
                if existing is not None and existing.status != "ready":
                    raise ValueError("existing reference release is not ready")
        finally:
            engine.dispose()
    if dry_run:
        return report
    if not database.is_file():
        raise ValueError("reference catalog database must already be migrated")
    store = LocalReferenceStore(objects)
    for sha256, data in objects_to_stage.items():
        store.put_verified(data, sha256)
    if report.existing:
        return report
    engine = create_engine(f"sqlite:///{database.resolve()}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        with Session(engine) as session, session.begin():
            release = ReferenceReleaseORM(
                release_id=release_id, source_snapshot_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                metadata_sha256=digest(metadata), array_index_sha256=digest([
                    [t.trace_revision_id, t.axis_sha256, t.intensity_sha256] for t in traces
                ]), serving_policy_sha256=policy_sha, schema_version=manifest.schema_version,
                source_name=manifest.source_name, source_base_url=manifest.source_base_url,
                captured_at=manifest.captured_at, importer_version=IMPORTER_VERSION,
                parser_version=manifest.parser_version, qa_version=manifest.qa_version, status="staged",
            )
            session.add(release)
            session.flush()
            session.add_all(files)
            session.flush()
            session.add_all(traces)
            session.flush()
            session.add_all(templates)
            session.flush()
            release.status = "ready"
    finally:
        engine.dispose()
    return report
