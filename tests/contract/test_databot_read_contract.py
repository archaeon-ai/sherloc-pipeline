"""Contract test: the tables and columns phase-databot reads from this database (#91).

phase-databot reads the SHERLOC database directly, read-only. These are its reads
(phase-databot ``orchestrator/sherloc_db.py`` ``SERVED_READ_PATHS``, the probe the
refresh runs on a promoted copy, and ``orchestrator/served_inventory.py``'s two
schema probes). If a migration or a model change drops or renames one of these, the
refresh fails in production; this test makes it fail here first.

Both ways this package builds the tables are checked: the Alembic chain to head
(what ``sherloc init`` and every live database use) and ``create_all_tables``.
The Alembic upgrade runs directly, not through ``init_database``, which falls back
to ``create_all`` when a migration fails and would hide a broken chain.

When phase-databot starts reading another table or column, add it here.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from sherloc_pipeline.database import create_all_tables, get_engine

REPO_ROOT = Path(__file__).resolve().parents[2]

# table -> the columns phase-databot names. An empty set means the table is
# read whole (``SELECT COUNT(*)``), so only its existence is part of the contract.
DATABOT_READS: dict[str, frozenset[str]] = {
    "scans": frozenset({"id", "scan_id", "sol_number", "scan_name", "scan_type", "target_type"}),
    "scan_points": frozenset(),
    "fitted_peaks": frozenset(),
    "context_images": frozenset({"scan_id", "image_type", "r2_rel_key"}),
}

# The statements themselves, as phase-databot runs them.
DATABOT_STATEMENTS: tuple[str, ...] = (
    "SELECT COUNT(*) FROM scans",
    "SELECT scan_id, sol_number, scan_name FROM scans LIMIT 1",
    "SELECT COUNT(*) FROM scan_points",
    "SELECT COUNT(*) FROM fitted_peaks",
    "SELECT r2_rel_key FROM context_images LIMIT 1",
    "SELECT id, sol_number, scan_type, target_type FROM scans WHERE 0",
    "SELECT scan_id, image_type, r2_rel_key FROM context_images WHERE 0",
)


def _missing(db_path: Path) -> list[str]:
    """Every table or column in ``DATABOT_READS`` that ``db_path`` lacks."""
    with sqlite3.connect(db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        out = []
        for table, columns in DATABOT_READS.items():
            if table not in tables:
                out.append(table)
                continue
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            out += [f"{table}.{c}" for c in sorted(columns - have)]
    return out


# Built once per module (the Alembic chain takes several seconds); a test that
# changes the schema works on its own copy.
@pytest.fixture(scope="module")
def alembic_db(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("alembic") / "alembic.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PHASE_DATABASE_PATH", str(db_path))
        command.upgrade(Config(str(REPO_ROOT / "alembic.ini")), "head")
    return db_path


@pytest.fixture(scope="module")
def create_all_db(tmp_path_factory) -> Path:
    db_path = tmp_path_factory.mktemp("create_all") / "create_all.db"
    engine = get_engine(db_path)
    create_all_tables(engine)
    engine.dispose()
    return db_path


@pytest.mark.parametrize("built", ["alembic_db", "create_all_db"])
def test_every_table_and_column_databot_reads_exists(built, request):
    assert _missing(request.getfixturevalue(built)) == []


@pytest.mark.parametrize("built", ["alembic_db", "create_all_db"])
def test_databot_statements_execute(built, request):
    with sqlite3.connect(request.getfixturevalue(built)) as conn:
        for sql in DATABOT_STATEMENTS:
            conn.execute(sql).fetchall()


def test_every_statement_reads_a_listed_table():
    # Keeps the two lists in step: a statement may not read a table the contract omits.
    for sql in DATABOT_STATEMENTS:
        table = sql.split(" FROM ", 1)[1].split()[0]
        assert table in DATABOT_READS, sql


def test_alembic_version_is_readable(alembic_db):
    # phase-databot derives the expected revision from this checkout and compares it.
    with sqlite3.connect(alembic_db) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchall()


@pytest.mark.parametrize(
    ("ddl", "reported"),
    [
        ("ALTER TABLE context_images DROP COLUMN r2_rel_key", "context_images.r2_rel_key"),
        ("ALTER TABLE scans RENAME COLUMN scan_name TO name", "scans.scan_name"),
        ("DROP TABLE fitted_peaks", "fitted_peaks"),
    ],
)
def test_a_dropped_or_renamed_read_is_reported(alembic_db, tmp_path, ddl, reported):
    changed = shutil.copy(alembic_db, tmp_path / "changed.db")
    with sqlite3.connect(changed) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(ddl)
    assert reported in _missing(changed)
    assert _missing(alembic_db) == []
