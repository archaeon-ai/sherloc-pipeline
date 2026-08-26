"""Tests for targetless-science ingest detection and remediation planning."""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import sys
from pathlib import Path

from sherloc_pipeline.database import ScanORM, get_session
from sherloc_pipeline.services.ingestion import IngestionService


SCRIPT = Path(__file__).parents[3] / "scripts" / "plan_missing_scan_targets.py"
SPEC = importlib.util.spec_from_file_location("plan_missing_scan_targets", SCRIPT)
assert SPEC and SPEC.loader
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def _make_audit_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE scans (
            id TEXT PRIMARY KEY,
            sol_number INTEGER NOT NULL,
            scan_id TEXT NOT NULL,
            scan_name TEXT NOT NULL,
            scan_type TEXT,
            n_points INTEGER NOT NULL,
            source_path TEXT,
            target TEXT,
            target_type TEXT NOT NULL
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "row-a", 100, "source-a", "detail_1", "detail", 100,
                None, None, "engineering",
            ),
            (
                "row-b", 101, "source-b", "survey_1", "survey", 400,
                None, None, "engineering",
            ),
            (
                "row-c", 101, "source-c", "detail_2", "detail", 100,
                None, "Context Rock", "mars_target",
            ),
            (
                "row-d", 102, "source-d", "line_1", "line", 20,
                None, None, "engineering",
            ),
        ],
    )
    connection.commit()
    connection.close()


def test_plan_lists_residue_and_never_writes(tmp_path):
    database = tmp_path / "audit.db"
    _make_audit_database(database)
    loupe_root = tmp_path / "loupe"
    sol_dir = loupe_root / "sol_0100"
    sol_dir.mkdir(parents=True)
    (sol_dir / "Sol_0100_Source_Rock.lpe").write_text("")

    with PLAN.open_read_only(database) as connection:
        scans = PLAN.find_missing_scans(connection)
        resolutions = [
            PLAN.resolve_target(connection, scan, loupe_root) for scan in scans
        ]
        output = PLAN.render_plan(scans, resolutions)

    assert [scan.scan_id for scan in scans] == ["source-a", "source-b"]
    assert [resolution.target for resolution in resolutions] == [
        "Source Rock",
        "Context Rock",
    ]
    assert output.startswith("targetless science scans: 2 across 2 sols\n")
    assert (
        "100\tsource-a\tdetail_1\tdetail\t100\tSource Rock\t.lpe filename"
        in output
    )
    assert "WHERE id = 'row-a' AND scan_id = 'source-a'" in output
    assert "target_type = 'mars_target'" in output

    # Opening, resolving, and rendering a plan must leave every row untouched.
    with sqlite3.connect(database) as connection:
        targets = connection.execute(
            "SELECT id, target, target_type FROM scans ORDER BY id"
        ).fetchall()
    assert targets[:2] == [
        ("row-a", None, "engineering"),
        ("row-b", None, "engineering"),
    ]


def test_plan_refuses_conflicting_sol_evidence(tmp_path):
    database = tmp_path / "audit.db"
    _make_audit_database(database)
    loupe_root = tmp_path / "loupe"
    sol_dir = loupe_root / "sol_0101"
    sol_dir.mkdir(parents=True)
    (sol_dir / "Sol_0101_Different_Rock.lpe").write_text("")

    with PLAN.open_read_only(database) as connection:
        scan = PLAN.find_missing_scans(connection)[1]
        resolution = PLAN.resolve_target(connection, scan, loupe_root)

    assert resolution.target is None
    assert resolution.source == (
        "conflicting sol evidence: Context Rock, Different Rock"
    )


def test_direct_workspace_ingest_uses_sol_lpe_target(fixtures_path, tmp_path):
    source = (
        fixtures_path
        / "loupe"
        / "sol_0921"
        / "detail_1"
        / "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
    )
    workspace = tmp_path / "sol_0921" / "detail_1" / source.name
    shutil.copytree(source, workspace)
    (tmp_path / "sol_0921" / "Sol_0921_Amherst_Point.lpe").write_text("")
    service = IngestionService(
        database_path=tmp_path / "ingest.db", include_spectra=False
    )

    result = service.ingest_workspace(workspace)

    assert not any("no geological target" in warning for warning in result.warnings)
    with get_session(service.engine) as session:
        scan = session.query(ScanORM).one()
        assert scan.target == "Amherst Point"
        assert scan.target_type == "mars_target"


def test_targetless_science_workspace_logs_error_and_returns_warning(
    fixtures_path, tmp_path, caplog
):
    workspace = (
        fixtures_path
        / "loupe"
        / "sol_0921"
        / "detail_1"
        / "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
    )
    service = IngestionService(
        database_path=tmp_path / "ingest.db", include_spectra=False
    )

    with caplog.at_level("ERROR", logger="sherloc_pipeline.services.ingestion"):
        result = service.ingest_workspace(workspace)

    assert any("no geological target" in warning for warning in result.warnings)
    assert any("no geological target" in record.message for record in caplog.records)
