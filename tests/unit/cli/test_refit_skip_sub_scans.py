"""`sherloc refit --skip-sub-scans`: fit primary and composite scans. Synthetic database; the pipeline is stubbed.

A sub-scan (a downlink part of a scan) is left out when its parent scan is
stored, since the parent holds the same spectra. A sub-scan whose parent is
not stored is still run, and named in the journal and the result.
"""

import json
import uuid

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.services import refit_journal
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.pipeline import PipelineService

SOL = 76544


def _scan(target, scan_name, scan_class="primary", parent=None, sources=None):
    return ScanORM(
        id=str(uuid.uuid4()), sol_number=SOL, target=target, target_type="mars_target",
        scan_name=scan_name, scan_id=f"synthetic-{uuid.uuid4().hex[:8]}",
        sclk_start=0, n_points=1, n_channels=2148,
        scan_class=scan_class, parent_scan_id=parent, source_scan_ids=sources,
    )


@pytest.fixture
def make_db(tmp_path, monkeypatch):
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)

    def make(build):
        path = tmp_path / "phase.db"
        from sherloc_pipeline.database.connection import open_database
        engine = open_database(path, create_missing=True)
        with get_session(engine) as session:
            session.add(SolORM(sol_number=SOL, data_source="loupe"))
            session.flush()
            for row in build():
                session.add(row)
                session.flush()
        engine.dispose()
        return path

    return make


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def run_full_pipeline(self, sol, target, scan, **kwargs):
        seen.append(scan)
        return ServiceResult(summary="stubbed", metadata={})

    monkeypatch.setattr(PipelineService, "run_full_pipeline", run_full_pipeline)
    monkeypatch.setattr(refit_journal, "run_failures", lambda metadata: [])
    return seen


def _refit(db, tmp_path, *extra):
    result = CliRunner().invoke(app, [
        "--json", "refit", "--database", str(db), "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"), *extra,
    ])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["result"]


def _entries(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()[1:]]


def test_parent_with_two_sub_scans_fits_only_the_parent(make_db, calls, tmp_path):
    def build():
        parent = _scan("Synthetic A", "detail_1")
        return [
            parent,
            _scan("Synthetic A", "detail_1a", "sub_scan", parent=parent.id),
            _scan("Synthetic A", "detail_1b", "sub_scan", parent=parent.id),
        ]

    db = make_db(build)
    result = _refit(db, tmp_path, "--science", "--target", "Synthetic_A", "--skip-sub-scans")
    assert calls == ["detail_1"]
    assert result["sub_scans_without_parent"] == []


def test_without_the_option_every_scan_is_fitted(make_db, calls, tmp_path):
    def build():
        parent = _scan("Synthetic A", "detail_1")
        return [parent, _scan("Synthetic A", "detail_1a", "sub_scan", parent=parent.id)]

    db = make_db(build)
    result = _refit(db, tmp_path, "--science", "--target", "Synthetic_A")
    assert sorted(calls) == ["detail_1", "detail_1a"]
    assert "sub_scans_without_parent" not in result


def test_composites_are_fitted(make_db, calls, tmp_path):
    def build():
        line_1, line_2 = _scan("Synthetic A", "line_1"), _scan("Synthetic A", "line_2")
        return [
            line_1, line_2,
            _scan("Synthetic A", "lines_1_2", "composite", sources=[line_1.id, line_2.id]),
        ]

    db = make_db(build)
    _refit(db, tmp_path, "--skip-sub-scans")
    assert sorted(calls) == ["line_1", "line_2", "lines_1_2"]


def test_sub_scan_without_a_stored_parent_is_fitted_and_named(make_db, calls, tmp_path):
    db = make_db(lambda: [
        _scan("Synthetic A", "detail_2"),
        _scan("Synthetic A", "detail_3a", "sub_scan", parent=None),
    ])
    journal = tmp_path / "batch.jsonl"
    result = _refit(db, tmp_path, "--skip-sub-scans", "--journal", str(journal))
    assert sorted(calls) == ["detail_2", "detail_3a"]
    assert [s["scan"] for s in result["sub_scans_without_parent"]] == ["detail_3a"]
    named = {e["scan"]: e.get("sub_scan_without_parent") for e in _entries(journal)}
    assert named == {"detail_2": None, "detail_3a": True}


def test_journal_records_the_option_and_resume_requires_it(make_db, calls, tmp_path):
    db = make_db(lambda: [_scan("Synthetic A", "detail_1")])
    journal = tmp_path / "batch.jsonl"
    _refit(db, tmp_path, "--skip-sub-scans", "--journal", str(journal))
    header = json.loads(journal.read_text().splitlines()[0])
    assert header["selection"]["skip_sub_scans"] is True

    result = CliRunner().invoke(app, [
        "--json", "refit", "--database", str(db), "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"), "--resume", str(journal),
    ])
    assert result.exit_code == 1
    resumed = _refit(db, tmp_path, "--skip-sub-scans", "--resume", str(journal))
    assert resumed["skipped"] == 1

    plain = tmp_path / "plain.jsonl"
    _refit(db, tmp_path, "--journal", str(plain))
    assert "skip_sub_scans" not in json.loads(plain.read_text().splitlines()[0])["selection"]
