"""Entirely synthetic local reference imports, verified reads and real migrations."""

import copy
import hashlib
import io
import json
import os
import shutil
import socket
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, event, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app as references_cli
from sherloc_pipeline.database.models import Base
from sherloc_pipeline.database.reference_models import ReferenceFileORM, ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.services.reference_import import REFERENCE_REVISION, import_release
from sherloc_pipeline.services.reference_store import LocalReferenceStore, object_locator
from sherloc_pipeline.web.feature_policy import FeaturePolicy
from sherloc_pipeline.web.reference_storage import R2ReferenceStore, can_serve_template, can_serve_trace

PREVIOUS_REVISION = "17db1a1940d6"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))


@pytest.fixture(scope="module")
def migrated_template(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    database = tmp_path_factory.mktemp("reference-catalog") / "catalog.db"
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    old_path = os.environ.get("PHASE_DATABASE_PATH")
    os.environ["PHASE_DATABASE_PATH"] = str(database)
    original = Base.metadata.create_all
    Base.metadata.create_all = Mock(side_effect=AssertionError("create_all cannot substitute for a migration"))
    try:
        command.upgrade(config, PREVIOUS_REVISION)
        engine = create_engine(f"sqlite:///{database}")
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == PREVIOUS_REVISION
            assert "reference_releases" not in inspect(connection).get_table_names()
        engine.dispose()
        command.upgrade(config, REFERENCE_REVISION)
    finally:
        Base.metadata.create_all = original
        if old_path is None:
            os.environ.pop("PHASE_DATABASE_PATH", None)
        else:
            os.environ["PHASE_DATABASE_PATH"] = old_path
    return database


@pytest.fixture
def database(migrated_template, tmp_path):
    database = tmp_path / "catalog.db"
    shutil.copyfile(migrated_template, database)
    return database


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    raw = b"synthetic source bytes; no scientific measurements"
    arrays = io.BytesIO()
    np.savez(arrays, x=np.array([3., 2., 2., 1.]), y=np.array([-2., 0., 3., 1.]))
    array_bytes = arrays.getvalue()
    (root / "source.bin").write_bytes(raw)
    (root / "arrays.npz").write_bytes(array_bytes)
    manifest = {
        "schema_version": "reference-snapshot-1", "source_name": "Synthetic Library",
        "source_base_url": "https://synthetic.invalid/unfetched",
        "captured_at": "2001-01-01T00:00:00Z", "parser_version": "synthetic-parser-1",
        "qa_version": "synthetic-qa-1", "files": [{
            "datatype_id": "synthetic-type", "record_id": "synthetic-0001", "file_id": "synthetic-file-1",
            "original_filename": "synthetic.bin", "source_url": "https://synthetic.invalid/unfetched/source",
            "media_type": "application/octet-stream", "raw_path": "source.bin",
            "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_size": len(raw),
            "arrays_path": "arrays.npz", "arrays_sha256": hashlib.sha256(array_bytes).hexdigest(),
            "arrays_size": len(array_bytes), "original_metadata": {"synthetic": True},
            "parse_adjustments": ["synthetic header removed"], "traces": [{
                "trace_index": 0, "source_label": "synthetic measurement", "x_key": "x", "y_key": "y",
                "modality": "raman", "x_quantity": "raman_shift", "x_unit": "cm-1",
                "x_evidence": "invented axis evidence", "y_unit": "synthetic counts",
                "species_name": "Syntheticite", "specimen_key": "synthetic-specimen-1",
                "citation_text": "Synthetic citation", "license_identifier": "synthetic-license",
                "license_source": "synthetic rights record",
                "rights_status": "known",
            }],
        }], "templates": [{
            "name": "synthetic-template", "species_name": "Syntheticite", "supported_class": "synthetic-class",
            "recipe_sha256": "a" * 64, "grouping_evidence": "synthetic specimen membership",
            "members": [{"record_id": "synthetic-0001", "file_id": "synthetic-file-1", "trace_index": 0}],
        }],
    }
    (root / "snapshot.json").write_text(json.dumps(manifest))
    policy = tmp_path / "serving.json"
    policy.write_text(json.dumps({"schema_version": "reference-serving-1",
                                  "approved_trace_keys": ["synthetic-0001:synthetic-file-1:0"]}))
    return root, manifest, policy


def save_manifest(root, manifest):
    (root / "snapshot.json").write_text(json.dumps(manifest))


def engine_for(database):
    engine = create_engine(f"sqlite:///{database}")

    @event.listens_for(engine, "connect")
    def foreign_keys(dbapi, record):
        dbapi.execute("PRAGMA foreign_keys=ON")

    return engine


def runtime_policy(release, traces, mode="all_approved"):
    return FeaturePolicy.model_validate({"references": {
        "mode": mode, "release_id": release.release_id, "subset_id": "e" * 64,
        "serving_policy_version": release.serving_policy_sha256,
        "allowed_trace_ids": [t.trace_revision_id for t in traces] if mode == "allowlist" else [],
    }})


def test_file_backed_migration_revision_indexes_constraints(database, monkeypatch):
    engine = engine_for(database)
    try:
        with engine.begin() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == REFERENCE_REVISION
            inspector = inspect(connection)
            for table in ("reference_releases", "reference_files", "reference_traces", "reference_templates"):
                assert table in inspector.get_table_names()
                expected = Base.metadata.tables[table]
                actual = {c["name"]: c["nullable"] for c in inspector.get_columns(table)}
                assert actual == {c.name: c.nullable for c in expected.columns}
                assert {i["name"] for i in inspector.get_indexes(table)} == {i.name for i in expected.indexes}
            assert inspector.get_foreign_keys("reference_traces")
            assert inspector.get_unique_constraints("reference_files")
            with pytest.raises(IntegrityError):
                connection.execute(text("""INSERT INTO reference_traces
                    (trace_revision_id, release_id, file_revision_id, trace_index, source_key,
                     point_count, axis_sha256, intensity_sha256, rights_known, serving_status,
                     permitted_tiers, exclusions, metadata_json)
                    VALUES (:id, :id, :id, 0, 'synthetic-key', 4, :id, :id, 0, 'off', '[]', '[]', '{}')"""),
                    {"id": "b" * 64})
    finally:
        engine.dispose()
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    monkeypatch.setenv("PHASE_DATABASE_PATH", str(database))
    command.downgrade(config, PREVIOUS_REVISION)
    engine = engine_for(database)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == PREVIOUS_REVISION
        assert "reference_releases" not in inspect(connection).get_table_names()
        assert "scans" in inspect(connection).get_table_names()
    engine.dispose()
    command.upgrade(config, REFERENCE_REVISION)


def test_import_repeat_native_arrays_and_policy(snapshot, database, tmp_path):
    root, manifest, policy = snapshot
    objects = tmp_path / "objects"
    first = import_release(root, database, objects, policy)
    second = import_release(root, database, objects, policy)
    assert not first.existing and second.existing
    assert first.team_permitted == 1 and first.public_permitted == 0
    engine = engine_for(database)
    try:
        with Session(engine) as session:
            assert session.query(ReferenceReleaseORM).count() == 1
            release = session.get(ReferenceReleaseORM, first.release_id)
            file = session.query(ReferenceFileORM).one()
            trace = session.query(ReferenceTraceORM).one()
            template = session.query(ReferenceTemplateORM).one()
            assert len(trace.trace_revision_id) == 64
            assert file.metadata_json["parse_adjustments"] == ["synthetic header removed"]
            assert trace.point_count == 4
            assert trace.metadata_json["species_name"] == "Syntheticite"
            data = LocalReferenceStore(objects).read_verified(file.array_locator, file.array_sha256)
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["x"], [3, 2, 2, 1])
                np.testing.assert_array_equal(arrays["y"], [-2, 0, 3, 1])
            config = runtime_policy(release, [trace])
            assert can_serve_trace(trace, release, config, "team")
            assert can_serve_template(template, [trace], release, config, "team")
            assert not can_serve_trace(trace, release, config, "public")
            assert not can_serve_template(template, [trace], release, config, "public")
            assert not can_serve_trace(trace, release, FeaturePolicy(), "team")
    finally:
        engine.dispose()


@pytest.mark.parametrize("change", ["bytes", "citation", "license", "qa", "parser", "grouping"])
def test_changes_make_new_release_without_deleting_old(snapshot, database, tmp_path, change):
    root, manifest, policy = snapshot
    objects = tmp_path / "objects"
    first = import_release(root, database, objects, policy)
    file = manifest["files"][0]
    trace = file["traces"][0]
    if change == "bytes":
        raw = b"synthetic replacement bytes under the same source ID"
        (root / file["raw_path"]).write_bytes(raw)
        file.update(raw_size=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
    elif change == "citation":
        trace["citation_text"] = "Synthetic replacement citation"
    elif change == "license":
        trace["license_identifier"] = "synthetic-license-2"
    elif change == "qa":
        trace["qa_flags"] = ["synthetic-exclusion"]
    elif change == "parser":
        manifest["parser_version"] = "synthetic-parser-2"
    else:
        manifest["templates"][0]["grouping_evidence"] = "synthetic regrouping evidence"
    save_manifest(root, manifest)
    second = import_release(root, database, objects, policy)
    assert first.release_id != second.release_id
    assert second.revised_files == (1 if change == "bytes" else 0)
    engine = engine_for(database)
    with Session(engine) as session:
        assert session.query(ReferenceReleaseORM).count() == 2
        assert session.query(ReferenceTraceORM).count() == 2
    engine.dispose()


@pytest.mark.parametrize("license_fields", [{}, {"license_identifier": "synthetic-license"},
                                           {"license_source": "synthetic-rights"}])
def test_missing_rights_and_citation_stay_unknown(snapshot, database, tmp_path, license_fields):
    root, manifest, policy = snapshot
    trace = manifest["files"][0]["traces"][0]
    for field in ("citation_text", "license_identifier", "license_source"):
        trace.pop(field)
    trace.update(license_fields)
    save_manifest(root, manifest)
    report = import_release(root, database, tmp_path / "objects", policy)
    assert report.missing_citation == 1 and report.unknown_rights == 1
    assert report.team_permitted == report.public_permitted == 0
    engine = engine_for(database)
    with Session(engine) as session:
        release = session.get(ReferenceReleaseORM, report.release_id)
        trace = session.query(ReferenceTraceORM).one()
        assert trace.metadata_json["citation_text"] is None
        assert not can_serve_trace(trace, release, runtime_policy(release, [trace]), "team")
    engine.dispose()


def test_uncertainty_mean_and_tier_limited_template(snapshot, database, tmp_path):
    root, manifest, policy = snapshot
    traces = manifest["files"][0]["traces"]
    for index, role in enumerate(("uncertainty", "source_mean"), 1):
        trace = copy.deepcopy(traces[0])
        trace.update(trace_index=index, column_role=role)
        traces.append(trace)
    manifest["templates"][0]["members"].append({"record_id": "synthetic-0001", "file_id": "synthetic-file-1", "trace_index": 1})
    save_manifest(root, manifest)
    report = import_release(root, database, tmp_path / "objects", policy)
    assert report.traces == 3 and report.excluded == 2
    engine = engine_for(database)
    with Session(engine) as session:
        template = session.query(ReferenceTemplateORM).one()
        assert template.permitted_tiers == []
        release = session.get(ReferenceReleaseORM, report.release_id)
        members = [session.get(ReferenceTraceORM, member) for member in template.member_trace_ids]
        assert not can_serve_template(template, members, release, runtime_policy(release, members), "team")
    engine.dispose()


def test_duplicate_content_keeps_aliases(snapshot, database, tmp_path):
    root, manifest, policy = snapshot
    alias = copy.deepcopy(manifest["files"][0])
    alias.update(record_id="synthetic-0002", file_id="synthetic-file-2")
    manifest["files"].append(alias)
    save_manifest(root, manifest)
    objects = tmp_path / "objects"
    report = import_release(root, database, objects, policy)
    assert report.files == 2 and report.traces == 2
    assert len(list((objects / "sherloc-references" / "sha256").iterdir())) == 2


@pytest.mark.parametrize("bad_path", ["../escape", "/synthetic/absolute", "https://synthetic.invalid/file", "folder\\escape"])
def test_path_traversal_rejected_before_writes(snapshot, database, tmp_path, bad_path):
    root, manifest, policy = snapshot
    manifest["files"][0]["raw_path"] = bad_path
    save_manifest(root, manifest)
    before = database.read_bytes()
    objects = tmp_path / "objects"
    with pytest.raises(ValueError):
        import_release(root, database, objects, policy)
    assert database.read_bytes() == before and not objects.exists()


def test_symlink_source_escape_rejected(snapshot, database, tmp_path):
    root, manifest, policy = snapshot
    outside = tmp_path / "outside.bin"
    outside.write_bytes((root / "source.bin").read_bytes())
    (root / "source.bin").unlink()
    (root / "source.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        import_release(root, database, tmp_path / "objects", policy)


@pytest.mark.parametrize("field,value", [("raw_sha256", "f" * 64), ("raw_size", 1), ("arrays_sha256", "f" * 64)])
def test_bad_hash_or_size_rejected(snapshot, database, tmp_path, field, value):
    root, manifest, policy = snapshot
    manifest["files"][0][field] = value
    save_manifest(root, manifest)
    with pytest.raises(ValueError):
        import_release(root, database, tmp_path / "objects", policy)
    assert not (tmp_path / "objects").exists()


@pytest.mark.parametrize("kind", ["pickle", "shape", "nonfinite", "length"])
def test_bad_native_arrays_rejected(snapshot, database, tmp_path, kind):
    root, manifest, policy = snapshot
    x = np.array([1., 2.])
    y = {"pickle": np.array([object(), object()]), "shape": np.array([[1., 2.]]),
         "nonfinite": np.array([1., np.nan]), "length": np.array([1.])}[kind]
    array_bytes = io.BytesIO()
    np.savez(array_bytes, x=x, y=y)
    data = array_bytes.getvalue()
    (root / "arrays.npz").write_bytes(data)
    manifest["files"][0].update(arrays_sha256=hashlib.sha256(data).hexdigest(), arrays_size=len(data))
    save_manifest(root, manifest)
    with pytest.raises(ValueError):
        import_release(root, database, tmp_path / "objects", policy)


def test_interrupted_staging_never_commits_release(snapshot, database, tmp_path, monkeypatch):
    root, _, policy = snapshot
    original = LocalReferenceStore.put_verified
    calls = []

    def interrupted(self, data, sha256):
        calls.append(sha256)
        if len(calls) == 2:
            raise OSError("synthetic interrupted staging")
        return original(self, data, sha256)

    monkeypatch.setattr(LocalReferenceStore, "put_verified", interrupted)
    with pytest.raises(OSError):
        import_release(root, database, tmp_path / "objects", policy)
    engine = engine_for(database)
    with Session(engine) as session:
        assert session.query(ReferenceReleaseORM).count() == 0
    engine.dispose()
    monkeypatch.setattr(LocalReferenceStore, "put_verified", original)
    assert import_release(root, database, tmp_path / "objects", policy).traces == 1


def test_dry_run_cli_no_database_or_objects_created(snapshot, tmp_path, database):
    root, _, policy = snapshot
    absent = tmp_path / "absent.db"
    objects = tmp_path / "objects"
    result = CliRunner().invoke(references_cli, ["references", "import", "--snapshot", str(root), "--database", str(absent),
                              "--objects", str(objects), "--serving-policy", str(policy), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["team_permitted"] == 1
    assert not absent.exists() and not objects.exists()
    before = database.read_bytes()
    dry = import_release(root, database, objects, policy, dry_run=True)
    assert database.read_bytes() == before and not objects.exists()
    actual = import_release(root, database, objects, policy)
    assert asdict(dry) == asdict(actual)


def test_storage_verification_and_mocked_r2(tmp_path):
    data = b"synthetic reference object"
    sha256 = hashlib.sha256(data).hexdigest()
    local = LocalReferenceStore(tmp_path / "objects")
    locator = local.put_verified(data, sha256)
    assert local.read_verified(locator, sha256) == data
    assert local.put_verified(data, sha256) == locator
    (local.root / locator).write_bytes(b"synthetic corruption")
    with pytest.raises(ValueError):
        local.read_verified(locator, sha256)
    reader = Mock(return_value=data)
    r2 = R2ReferenceStore("team", reader=reader)
    assert r2.read_verified(locator, sha256) == data
    reader.assert_called_once_with(locator)
    with pytest.raises(ValueError):
        r2.read_verified("../escape", sha256)
    assert reader.call_count == 1
    reader.return_value = b"synthetic wrong object"
    with pytest.raises(ValueError):
        r2.read_verified(locator, sha256)
    with pytest.raises(ValueError):
        R2ReferenceStore("public", reader=reader)


def test_revisions_immutable_and_retired_release_denied(snapshot, database, tmp_path):
    root, _, policy = snapshot
    report = import_release(root, database, tmp_path / "objects", policy)
    engine = engine_for(database)
    with Session(engine) as session:
        trace = session.query(ReferenceTraceORM).one()
        trace.point_count = 8
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()
        release = session.get(ReferenceReleaseORM, report.release_id)
        config = runtime_policy(release, [trace])
        session.commit()  # Expire the status before assigning the transition.
        release.status = "retired"
        session.commit()
        assert not can_serve_trace(trace, release, config, "team")
        release.status = "ready"
        with pytest.raises(ValueError, match="invalid reference release transition"):
            session.commit()
        session.rollback()
    engine.dispose()
    with pytest.raises(ValueError, match="not ready"):
        import_release(root, database, tmp_path / "objects", policy)


@pytest.mark.parametrize("record_id,file_id", [("synthetic-a:b", "c"), ("synthetic-a", "b:c")])
@pytest.mark.parametrize("location", ["file", "template"])
def test_delimited_source_identity_rejected_before_writes(snapshot, database, tmp_path, record_id, file_id, location):
    root, manifest, policy = snapshot
    identity = manifest["files"][0] if location == "file" else manifest["templates"][0]["members"][0]
    identity.update(record_id=record_id, file_id=file_id)
    policy.write_text(json.dumps({"approved_trace_keys": ["synthetic-a:b:c:0"]}))
    save_manifest(root, manifest)
    before = database.read_bytes()
    objects = tmp_path / "objects"
    with pytest.raises(ValueError, match="pattern"):
        import_release(root, database, objects, policy)
    assert database.read_bytes() == before and not objects.exists()


def test_compressed_archives_share_uncompressed_release_budget(snapshot, database, tmp_path, monkeypatch):
    from sherloc_pipeline.services import reference_import

    root, manifest, policy = snapshot
    files = manifest["files"]
    files.extend(copy.deepcopy(files[0]) for _ in range(2))
    for index, file in enumerate(files):
        arrays = io.BytesIO()
        np.savez_compressed(arrays, x=np.zeros(1024), y=np.zeros(1024))
        data = arrays.getvalue()
        file.update(record_id=f"synthetic-record-{index}", arrays_path=f"arrays-{index}.npz",
                    arrays_sha256=hashlib.sha256(data).hexdigest(), arrays_size=len(data))
        (root / file["arrays_path"]).write_bytes(data)
    manifest["templates"] = []
    policy.write_text(json.dumps({"approved_trace_keys": []}))
    save_manifest(root, manifest)
    expanded_size = reference_import.array_archive_size(data)
    assert len(data) * len(files) < expanded_size
    monkeypatch.setattr(reference_import, "MAX_RELEASE_BYTES", expanded_size * 2 + 2 * len(data) + 1024)
    original = reference_import.read_arrays
    reader = Mock(side_effect=original)
    monkeypatch.setattr(reference_import, "read_arrays", reader)
    before = database.read_bytes()
    objects = tmp_path / "objects"
    with pytest.raises(ValueError, match="release too large"):
        import_release(root, database, objects, policy)
    assert reader.call_count == 2
    assert database.read_bytes() == before and not objects.exists()


@pytest.mark.parametrize("model", [ReferenceReleaseORM, ReferenceFileORM, ReferenceTraceORM, ReferenceTemplateORM])
@pytest.mark.parametrize("method", ["instance", "query", "execute"])
def test_catalog_delete_rejected_and_reimport_preserves_rows(snapshot, database, tmp_path, model, method):
    root, _, policy = snapshot
    objects = tmp_path / "objects"
    report = import_release(root, database, objects, policy)
    engine = engine_for(database)
    try:
        with Session(engine) as session:
            with pytest.raises(ValueError, match="cannot be deleted"):
                if method == "instance":
                    session.delete(session.query(model).one())
                elif method == "query":
                    session.query(model).delete()
                else:
                    session.execute(delete(model))
                session.commit()
            session.rollback()
        repeated = import_release(root, database, objects, policy)
        assert repeated.existing and repeated.release_id == report.release_id
        with Session(engine) as session:
            for row_model in (ReferenceReleaseORM, ReferenceFileORM, ReferenceTraceORM, ReferenceTemplateORM):
                assert session.query(row_model).count() == 1
            release = session.get(ReferenceReleaseORM, report.release_id)
            assert release.status == "ready"
    finally:
        engine.dispose()
