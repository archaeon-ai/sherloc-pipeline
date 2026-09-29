"""Plain retries discover later workspaces and never process a stale zip extraction."""

import importlib
import zipfile
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import ScanORM, ScanPointORM
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.ingestion import IngestionService


cli = importlib.import_module("sherloc_pipeline.cli.app")


def add_workspace(sol_dir, number):
    workspace = sol_dir / f"detail_{number}" / f"scan_{number}_Loupe_working"
    workspace.mkdir(parents=True)
    (workspace / "loupe.csv").write_text(
        f"original_data_file,SrlcSpecSpecSohRaw_000000000{number}-00000-1\n"
        f"human_readable_workspace,detail_{number}\n"
        "n_spectra,1\nn_channels,2148\nlaser_wavelength,248.6\nshots_per_spec,1\n"
    )
    (workspace / "spatial.csv").write_text("az,el\n1,2\nx,y\n0.1,0.2\n")
    lpe = sol_dir / "Sol_0001_Synthetic_Rock.lpe"
    if not lpe.exists():
        lpe.write_text("workspaceDictName,workspaceHumanReadableName,soffPath\n")
    with lpe.open("a") as stream:
        stream.write(f"scan_{number},detail_{number},{workspace.relative_to(sol_dir)}/soff.xml\n")
    return workspace


def service_for(tmp_path):
    return IngestionService(
        database_path=tmp_path / "test.db", include_spectra=False,
        console=Console(quiet=True),
    )


def scan_snapshot(service):
    with get_session(service.engine) as session:
        return {
            scan.scan_name: (scan.id, tuple(
                point.id for point in session.query(ScanPointORM)
                .filter_by(scan_id=scan.id).order_by(ScanPointORM.point_index)
            ))
            for scan in session.query(ScanORM).all()
        }


def test_plain_retry_ingests_later_workspace_and_preserves_completed_rows(tmp_path):
    sol_dir = tmp_path / "sol_0001"
    add_workspace(sol_dir, 1)
    service = service_for(tmp_path)
    assert service.ingest_sol(sol_dir).metadata["scans_ingested"] == 1
    original = scan_snapshot(service)

    add_workspace(sol_dir, 2)
    result = service.ingest_sol(sol_dir, force=False)

    assert result.metadata["scans_ingested"] == 1
    assert result.metadata["scans_skipped"] == 1
    scans = scan_snapshot(service)
    assert set(scans) == {"detail_1", "detail_2"}
    assert scans["detail_1"] == original["detail_1"]
    assert len(scans["detail_2"][1]) == 1
    assert "skipped" not in result.summary


def test_plain_retry_recovers_workspace_parse_failure(tmp_path):
    sol_dir = tmp_path / "sol_0001"
    add_workspace(sol_dir, 1)
    bad_workspace = add_workspace(sol_dir, 2)
    csv_path = bad_workspace / "loupe.csv"
    valid_csv = csv_path.read_text()
    csv_path.write_text(valid_csv.replace("n_spectra,1", "n_spectra,broken"))
    service = service_for(tmp_path)

    assert service.ingest_sol(sol_dir).metadata["scans_ingested"] == 1
    original = scan_snapshot(service)
    assert set(original) == {"detail_1"}
    csv_path.write_text(valid_csv)

    result = service.ingest_sol(sol_dir, force=False)

    assert result.metadata["scans_ingested"] == 1
    assert result.metadata["scans_skipped"] == 1
    scans = scan_snapshot(service)
    assert set(scans) == {"detail_1", "detail_2"}
    assert scans["detail_1"] == original["detail_1"]


def test_unchanged_retry_is_idempotent(tmp_path):
    sol_dir = tmp_path / "sol_0001"
    add_workspace(sol_dir, 1)
    service = service_for(tmp_path)
    service.ingest_sol(sol_dir)
    original = scan_snapshot(service)

    result = service.ingest_sol(sol_dir)

    assert result.metadata["scans_ingested"] == 0
    assert result.metadata["scans_skipped"] == 1
    assert "skipped" in result.summary
    assert scan_snapshot(service) == original


def test_plain_retry_recovers_failure_after_workspace_rows_are_written(tmp_path, monkeypatch):
    sol_dir = tmp_path / "sol_0001"
    add_workspace(sol_dir, 1)
    add_workspace(sol_dir, 2)
    service = service_for(tmp_path)
    original_ingest = service._ingest_workspace_internal

    def fail_after_write(session, workspace_path, *args, **kwargs):
        stats = original_ingest(session, workspace_path, *args, **kwargs)
        if workspace_path.parent.name == "detail_2":
            session.flush()
            raise ValueError("synthetic failure after workspace write")
        return stats

    with monkeypatch.context() as patch:
        patch.setattr(service, "_ingest_workspace_internal", fail_after_write)
        first = service._ingest_sol_internal(sol_dir)
    assert len(first.errors) == 1
    assert first.scans_ingested == 1
    original = scan_snapshot(service)
    assert set(original) == {"detail_1"}

    result = service.ingest_sol(sol_dir, force=False)

    assert result.metadata["scans_ingested"] == 1
    assert result.metadata["scans_skipped"] == 1
    scans = scan_snapshot(service)
    assert set(scans) == {"detail_1", "detail_2"}
    assert scans["detail_1"] == original["detail_1"]


def test_process_new_plain_retry_discovers_later_scan(tmp_path, monkeypatch):
    sol_dir = tmp_path / "data" / "sol_0001"
    add_workspace(sol_dir, 1)
    pipeline = Mock()
    pipeline.return_value.run_full_pipeline.return_value = ServiceResult(summary="Synthetic fit")
    monkeypatch.setattr(cli, "PipelineService", pipeline)
    args = ["process-new", str(sol_dir), "--database", str(tmp_path / "test.db"),
            "--data-dir", str(tmp_path / "data"),
            "--results-dir", str(tmp_path / "results")]
    first = CliRunner().invoke(cli.app, args)
    assert first.exit_code == 0, first.output
    assert pipeline.return_value.run_full_pipeline.call_args.kwargs["scan"] == "detail_1"
    original = scan_snapshot(service_for(tmp_path))
    add_workspace(sol_dir, 2)
    pipeline.reset_mock()

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    calls = pipeline.return_value.run_full_pipeline.call_args_list
    assert {call.kwargs["scan"] for call in calls} == {"detail_1", "detail_2"}
    assert scan_snapshot(service_for(tmp_path))["detail_1"] == original["detail_1"]


def run_process_new(tmp_path, monkeypatch, path, *, dry_run=False):
    ingestion = Mock()
    ingestion.return_value.ingest_sol.return_value = ServiceResult(summary="Synthetic ingest")
    pipeline = Mock()
    pipeline.return_value.run_full_pipeline.return_value = ServiceResult(summary="Synthetic fit")
    selection = Mock(return_value=[(1, "Synthetic Rock", "detail_1")])
    monkeypatch.setattr(cli, "IngestionService", ingestion)
    monkeypatch.setattr(cli, "PipelineService", pipeline)
    monkeypatch.setattr(cli, "_select_fittable_scans", selection)
    # The scan selector is stubbed, but session setup still needs real temporary tables.
    service_for(tmp_path)
    args = ["process-new", str(path), "--database", str(tmp_path / "test.db"),
            "--data-dir", str(tmp_path / "data"),
            "--results-dir", str(tmp_path / "results")]
    if dry_run:
        args.append("--dry-run")
    return CliRunner().invoke(cli.app, args), ingestion, pipeline, selection


@pytest.mark.parametrize("difference", ["changed", "added", "missing", "extra"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_process_new_refuses_stale_extraction(tmp_path, monkeypatch, difference, dry_run):
    sol_dir = tmp_path / "data" / "sol_0001"
    sol_dir.mkdir(parents=True)
    existing = sol_dir / "source.txt"
    existing.write_bytes(b"old data")
    archive = tmp_path / "sol_0001.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sol_0001/source.txt", b"new data" if difference == "changed" else b"old data")
        if difference == "added":
            zf.writestr("sol_0001/later.txt", b"later workspace")
    if difference == "missing":
        existing.unlink()
    if difference == "extra":
        (sol_dir / "obsolete.txt").write_bytes(b"obsolete workspace")
    before = {str(p.relative_to(sol_dir)): p.read_bytes() for p in sol_dir.rglob("*") if p.is_file()}

    result, ingestion, pipeline, selection = run_process_new(
        tmp_path, monkeypatch, archive, dry_run=dry_run,
    )

    assert result.exit_code == 1, result.output
    assert "differs" in result.output
    ingestion.assert_not_called()
    pipeline.assert_not_called()
    selection.assert_not_called()
    assert {str(p.relative_to(sol_dir)): p.read_bytes() for p in sol_dir.rglob("*") if p.is_file()} == before


def test_process_new_accepts_identical_extraction(tmp_path, monkeypatch):
    sol_dir = tmp_path / "data" / "sol_0001"
    sol_dir.mkdir(parents=True)
    (sol_dir / "source.txt").write_bytes(b"same data")
    archive = tmp_path / "sol_0001.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sol_0001/source.txt", b"same data")

    result, ingestion, pipeline, _ = run_process_new(tmp_path, monkeypatch, archive)

    assert result.exit_code == 0, result.output
    ingestion.return_value.ingest_sol.assert_called_once_with(sol_dir, force=False)
    pipeline.return_value.run_full_pipeline.assert_called_once()


def test_process_new_extracts_fresh_archive(tmp_path, monkeypatch):
    archive = tmp_path / "sol_0001.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sol_0001/source.txt", b"fresh data")

    result, ingestion, pipeline, _ = run_process_new(tmp_path, monkeypatch, archive)

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "sol_0001" / "source.txt").read_bytes() == b"fresh data"
    ingestion.return_value.ingest_sol.assert_called_once()
    pipeline.return_value.run_full_pipeline.assert_called_once()


@pytest.mark.parametrize("member", ["other_sol/source.txt", "sol_0001/../source.txt"])
def test_process_new_refuses_archive_outside_expected_sol(tmp_path, monkeypatch, member):
    sol_dir = tmp_path / "data" / "sol_0001"
    sol_dir.mkdir(parents=True)
    (sol_dir / "source.txt").write_bytes(b"same data")
    archive = tmp_path / "sol_0001.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, b"same data")

    result, ingestion, pipeline, _ = run_process_new(tmp_path, monkeypatch, archive)

    assert result.exit_code == 1, result.output
    ingestion.assert_not_called()
    pipeline.assert_not_called()


def test_process_new_refuses_symlink_in_existing_extraction(tmp_path, monkeypatch):
    sol_dir = tmp_path / "data" / "sol_0001"
    sol_dir.mkdir(parents=True)
    source = tmp_path / "outside.txt"
    source.write_bytes(b"same data")
    (sol_dir / "source.txt").symlink_to(source)
    archive = tmp_path / "sol_0001.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sol_0001/source.txt", b"same data")

    result, ingestion, pipeline, _ = run_process_new(tmp_path, monkeypatch, archive)

    assert result.exit_code == 1, result.output
    ingestion.assert_not_called()
    pipeline.assert_not_called()
    assert source.read_bytes() == b"same data"
