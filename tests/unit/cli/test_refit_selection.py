"""Which scans `sherloc refit` runs (issue #170). Synthetic database; the pipeline is stubbed.

- A scan with no target cannot be run (full-pipeline needs one): it is
  reported as unrunnable, not run and not counted as failed.
- Scans sharing a (sol, target, scan_name) key (after the target's spaces
  become underscores, as refit runs it) run once per pass; the extra
  rows are reported as duplicates.
"""

import json
import uuid

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.database.connection import create_all_tables, get_engine, get_session
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.services import refit_journal
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.pipeline import PipelineService

SOL = 76543


def _scan(target, scan_name):
    return ScanORM(
        id=str(uuid.uuid4()), sol_number=SOL, target=target, target_type="mars_target",
        scan_name=scan_name, scan_id=f"synthetic-{uuid.uuid4().hex[:8]}",
        sclk_start=0, n_points=1, n_channels=2148,
    )


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
    path = tmp_path / "phase.db"
    engine = get_engine(path)
    create_all_tables(engine)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=SOL, data_source="loupe"))
        session.flush()
        session.add_all([
            _scan("Synthetic A", "detail_1"),
            _scan(None, "detail_2"),
            _scan("Synthetic B", "detail_3"),
            _scan("Synthetic_B", "detail_3"),
        ])
    engine.dispose()
    return path


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def run_full_pipeline(self, sol, target, scan, **kwargs):
        seen.append((sol, target, scan))
        return ServiceResult(summary="stubbed", metadata={})

    monkeypatch.setattr(PipelineService, "run_full_pipeline", run_full_pipeline)
    monkeypatch.setattr(refit_journal, "run_failures", lambda metadata: [])
    return seen


def _refit(db, tmp_path, *extra):
    result = CliRunner().invoke(app, [
        "--json", "refit", "--database", str(db), "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"), *extra,
    ])
    return result, json.loads(result.stdout)


def _journal_keys(path):
    return [tuple(json.loads(line)[k] for k in ("sol", "target", "scan"))
            for line in path.read_text().splitlines()[1:]]


def test_no_target_is_unrunnable_and_each_key_runs_once(db, calls, tmp_path):
    journal = tmp_path / "batch.jsonl"
    result, payload = _refit(db, tmp_path, "--journal", str(journal))

    assert result.exit_code == 0, result.output
    assert sorted(calls) == [(f"{SOL:04d}", "Synthetic_A", "detail_1"), (f"{SOL:04d}", "Synthetic_B", "detail_3")]
    summary = payload["result"]
    assert summary["processed"] == 2
    assert summary["failed"] == 0
    assert summary["unrunnable"] == [f"{SOL:04d} detail_2: no target"]
    assert summary["duplicates"] == [f"{SOL:04d} Synthetic_B detail_3"]
    assert sorted(_journal_keys(journal)) == [(SOL, "Synthetic_A", "detail_1"), (SOL, "Synthetic_B", "detail_3")]


def test_resume_does_not_rerun_unrunnable_scans_as_failures(db, calls, tmp_path):
    journal = tmp_path / "batch.jsonl"
    _refit(db, tmp_path, "--journal", str(journal))
    calls.clear()

    result, payload = _refit(db, tmp_path, "--resume", str(journal))

    assert result.exit_code == 0, result.output
    assert calls == []
    assert payload["result"]["failed"] == 0
    assert payload["result"]["skipped"] == 2
    assert payload["result"]["unrunnable"] == [f"{SOL:04d} detail_2: no target"]
