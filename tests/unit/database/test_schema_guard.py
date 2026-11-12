"""#135: commands that read and write data never create or upgrade schema on an existing database."""

import json

import pytest
from sqlalchemy import inspect, text
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import (
    SchemaNotCurrentError,
    alembic_heads,
    create_all_tables,
    get_engine,
    init_pds_database,
    open_database,
    schema_revisions,
)
from sherloc_pipeline.database.reference_catalog_ddl import CATALOG_TABLES
from sherloc_pipeline.services.ingestion import IngestionService

OLD_REVISION = "17db1a1940d6"  # the revision before the reference-catalog tables


def _tables(path):
    engine = get_engine(path)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _revisions(path):
    engine = get_engine(path)
    try:
        return schema_revisions(engine)
    finally:
        engine.dispose()


@pytest.fixture
def current_db(tmp_path):
    path = tmp_path / "phase.db"
    open_database(path).dispose()
    return path


@pytest.fixture
def behind_db(current_db):
    """A database at an older revision: no reference-catalog tables, stamped OLD_REVISION."""
    engine = get_engine(current_db)
    with engine.begin() as conn:
        for table in reversed(CATALOG_TABLES):
            conn.execute(text(f"DROP TABLE {table}"))
        conn.execute(text("UPDATE alembic_version SET version_num = :rev"), {"rev": OLD_REVISION})
    engine.dispose()
    return current_db


@pytest.fixture
def unstamped_db(tmp_path):
    path = tmp_path / "unstamped.db"
    engine = get_engine(path)
    create_all_tables(engine)
    engine.dispose()
    return path


def test_new_database_is_created_and_stamped_at_head(tmp_path):
    path = tmp_path / "sub" / "new.db"
    open_database(path).dispose()
    assert _revisions(path) == alembic_heads()
    assert set(CATALOG_TABLES) <= _tables(path)


def test_database_at_head_opens_unchanged(current_db):
    before = _tables(current_db)
    open_database(current_db).dispose()
    assert _tables(current_db) == before
    assert _revisions(current_db) == alembic_heads()


def test_database_behind_head_is_refused_and_untouched(behind_db):
    before = _tables(behind_db)
    with pytest.raises(SchemaNotCurrentError) as excinfo:
        open_database(behind_db)
    message = str(excinfo.value)
    assert OLD_REVISION in message
    assert all(head in message for head in alembic_heads())
    assert f"PHASE_DATABASE_PATH={behind_db} alembic upgrade" in message
    assert _tables(behind_db) == before
    assert _revisions(behind_db) == (OLD_REVISION,)


def test_unstamped_tables_are_refused(unstamped_db):
    with pytest.raises(SchemaNotCurrentError, match="unstamped"):
        open_database(unstamped_db)
    assert "alembic_version" not in _tables(unstamped_db)


def test_ingestion_service_does_not_create_tables_on_an_old_database(behind_db):
    """The issue's case: process-new / ingest on a DB at the older revision."""
    with pytest.raises(SchemaNotCurrentError):
        IngestionService(database_path=behind_db, include_spectra=False)
    assert not set(CATALOG_TABLES) & _tables(behind_db)
    assert _revisions(behind_db) == (OLD_REVISION,)


def test_pds_database_is_not_migrated(behind_db):
    with pytest.raises(SchemaNotCurrentError):
        init_pds_database(behind_db)
    assert not set(CATALOG_TABLES) & _tables(behind_db)


def _db_stats(path):
    from sherloc_pipeline.cli.app import app

    return CliRunner().invoke(app, ["--json", "db-stats", "--database", str(path)])


def test_db_stats_reads_an_old_database_without_changing_it(behind_db):
    before = _tables(behind_db)
    result = _db_stats(behind_db)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["command"] == "db-stats"
    assert _tables(behind_db) == before
    assert _revisions(behind_db) == (OLD_REVISION,)


def test_db_stats_does_not_create_a_missing_database(tmp_path):
    path = tmp_path / "missing.db"
    result = _db_stats(path)
    assert result.exit_code != 0
    assert not path.exists()
