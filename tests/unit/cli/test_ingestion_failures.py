"""Ingestion failures must reach callers before any scan processing starts."""

import importlib
import json
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sherloc_pipeline.models.ingestion import LoupeWorkspaceParser, LoupeWorkspaceResult
from sherloc_pipeline.models.instrument import CCDConfiguration, InstrumentState
from sherloc_pipeline.models.spectra import Scan
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services import ingestion


cli = importlib.import_module("sherloc_pipeline.cli.app")


@pytest.fixture
def synthetic_sol(tmp_path, monkeypatch):
    """Discover real workspace paths; parse synthetic scans into a temporary DB."""
    sol_dir = tmp_path / "sol_0001"
    for name in ("good", "bad"):
        workspace = sol_dir / name / f"{name}_Loupe_working"
        workspace.mkdir(parents=True)
        (workspace / "loupe.csv").write_text("synthetic parser input\n")

    def parse(parser):
        name = parser.workspace_path.parent.name
        scan = Scan(
            sol_number=1, scan_name=f"detail_{name}", target="Synthetic Rock",
            scan_id=f"synthetic_{name}", sclk_start=1, n_points=1,
        )
        return LoupeWorkspaceResult(
            scan=scan,
            instrument_state=InstrumentState(scan_id=scan.id),
            ccd_configuration=CCDConfiguration(scan_id=scan.id),
            workspace_path=parser.workspace_path,
        )

    monkeypatch.setattr(LoupeWorkspaceParser, "parse", parse)
    return sol_dir


def inject_failure(monkeypatch, failure):
    original_parse = LoupeWorkspaceParser.parse
    if failure in ("workspace", "both"):
        def fail_parse(parser):
            if parser.workspace_path.parent.name == "bad":
                raise ValueError("synthetic corrupt soff.xml")
            return original_parse(parser)
        monkeypatch.setattr(LoupeWorkspaceParser, "parse", fail_parse)
    if failure in ("finalization", "both"):
        def fail_finalize(*args):
            raise ValueError("synthetic finalization failure")
        monkeypatch.setattr(ingestion, "finalize_sol_scans", fail_finalize)


def assert_errors(errors, failure):
    assert len(errors) == (2 if failure == "both" else 1)
    if failure in ("workspace", "both"):
        assert any("Workspace bad_Loupe_working: synthetic corrupt soff.xml" == e
                   for e in errors)
    if failure in ("finalization", "both"):
        assert any("Sol 1 finalization: synthetic finalization failure" == e
                   for e in errors)


@pytest.mark.parametrize("failure", ["workspace", "finalization", "both"])
def test_ingest_sol_reports_collected_errors(synthetic_sol, tmp_path, monkeypatch, failure):
    inject_failure(monkeypatch, failure)
    service = ingestion.IngestionService(
        database_path=tmp_path / "test.db", console=Console(quiet=True),
    )

    result = service.ingest_sol(synthetic_sol)

    assert result.metadata["success"] is False
    assert_errors(result.metadata["errors"], failure)
    assert "incomplete" in result.summary.lower()
    # Reporting partial failure does not hide the successfully ingested scan.
    assert result.metadata["scans_ingested"] == (2 if failure == "finalization" else 1)
    assert service.get_database_stats()["scans"] == result.metadata["scans_ingested"]


def test_successful_and_skipped_sol_remain_successful(synthetic_sol, tmp_path):
    service = ingestion.IngestionService(
        database_path=tmp_path / "test.db", console=Console(quiet=True),
    )
    result = service.ingest_sol(synthetic_sol)
    assert result.metadata["success"] is True
    assert result.metadata["errors"] == []
    assert result.metadata["scans_ingested"] == 2
    assert result.summary.startswith("Ingested sol 1:")

    result = service.ingest_sol(synthetic_sol)
    assert result.metadata["success"] is True
    assert result.metadata["errors"] == []
    assert "skipped" in result.summary
    assert service.get_database_stats()["scans"] == 2


def test_process_new_continues_after_successful_ingestion(synthetic_sol, tmp_path, monkeypatch):
    pipeline = Mock()
    pipeline.return_value.run_full_pipeline.return_value = ServiceResult(summary="Synthetic fit")
    monkeypatch.setattr(cli, "PipelineService", pipeline)

    result = CliRunner().invoke(cli.app, [
        "process-new", str(synthetic_sol), "--database", str(tmp_path / "test.db"),
        "--data-dir", str(tmp_path), "--results-dir", str(tmp_path / "results"),
    ])

    assert result.exit_code == 0, result.output
    calls = pipeline.return_value.run_full_pipeline.call_args_list
    assert {call.kwargs["scan"] for call in calls} == {"detail_good", "detail_bad"}
    assert all(call.kwargs["target"] == "Synthetic Rock" for call in calls)
    assert "2 scan(s) processed" in result.output


@pytest.mark.parametrize("failure", ["workspace", "finalization", "both"])
@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
def test_process_new_stops_on_ingestion_errors(
    synthetic_sol, tmp_path, monkeypatch, failure, json_mode,
):
    inject_failure(monkeypatch, failure)
    select_scans = Mock(return_value=[])
    pipeline = Mock()
    monkeypatch.setattr(cli, "_select_fittable_scans", select_scans)
    monkeypatch.setattr(cli, "PipelineService", pipeline)
    args = ["--json"] if json_mode else []
    args += [
        "process-new", str(synthetic_sol), "--database", str(tmp_path / "test.db"),
        "--data-dir", str(tmp_path), "--results-dir", str(tmp_path / "results"),
    ]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 1, result.output
    select_scans.assert_not_called()
    pipeline.assert_not_called()
    if json_mode:
        error = json.loads(result.stderr.splitlines()[-1])
        assert error["error_type"] == "IngestionError"
        assert error["exit_code"] == 1
        assert error["context"]["sol"] == 1
        assert_errors(error["context"]["errors"], failure)
        assert result.stdout == ""
    else:
        assert "Process-new failed" in result.output
        if failure in ("workspace", "both"):
            assert "synthetic corrupt soff.xml" in result.output
        if failure in ("finalization", "both"):
            assert "synthetic finalization failure" in result.output
