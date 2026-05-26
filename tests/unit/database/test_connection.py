"""
Tests for database connection utilities.

Tests cover:
- Engine creation
- Session management
- Table creation
- Foreign key enforcement
"""

import tempfile
from pathlib import Path

import pytest

from sherloc_pipeline.database.connection import (
    _default_pds_db_path,
    get_engine,
    get_session,
    create_all_tables,
    ensure_database_exists,
)
from sherloc_pipeline.database.models import Base, SolORM


class TestDefaultPdsDbPath:
    """Tests for _default_pds_db_path env-var resolution."""

    def test_uses_canonical_env_var(self, monkeypatch):
        monkeypatch.setenv("SHERLOC_PDS_DB", "/srv/canonical/phase_pds.db")
        monkeypatch.setenv("SHERLOC_PDS_DB_PATH", "/srv/legacy/phase_pds.db")
        assert _default_pds_db_path() == "/srv/canonical/phase_pds.db"

    def test_falls_back_to_legacy_env_var(self, monkeypatch):
        monkeypatch.delenv("SHERLOC_PDS_DB", raising=False)
        monkeypatch.setenv("SHERLOC_PDS_DB_PATH", "/srv/legacy/phase_pds.db")
        assert _default_pds_db_path() == "/srv/legacy/phase_pds.db"

    def test_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("SHERLOC_PDS_DB", raising=False)
        monkeypatch.delenv("SHERLOC_PDS_DB_PATH", raising=False)
        assert _default_pds_db_path() == "./phase_pds.db"

    def test_empty_canonical_falls_through(self, monkeypatch):
        monkeypatch.setenv("SHERLOC_PDS_DB", "")
        monkeypatch.setenv("SHERLOC_PDS_DB_PATH", "/srv/legacy/phase_pds.db")
        assert _default_pds_db_path() == "/srv/legacy/phase_pds.db"


class TestGetEngine:
    """Tests for get_engine function."""

    def test_in_memory_engine(self):
        """Test creating an in-memory database engine."""
        engine = get_engine(":memory:")
        assert engine is not None
        assert "memory" in str(engine.url)

    def test_file_engine(self, tmp_path):
        """Test creating a file-based database engine."""
        db_path = tmp_path / "test.db"
        engine = get_engine(db_path)
        assert engine is not None
        assert str(db_path) in str(engine.url)

    def test_engine_with_echo(self):
        """Test creating engine with SQL logging enabled."""
        engine = get_engine(":memory:", echo=True)
        assert engine is not None
        assert engine.echo is True


class TestGetSession:
    """Tests for get_session context manager."""

    def test_session_context_manager(self):
        """Test session creation and cleanup."""
        engine = get_engine(":memory:")
        create_all_tables(engine)

        with get_session(engine) as session:
            sol = SolORM(sol_number=921, data_source="loupe")
            session.add(sol)
            # Session should auto-commit on exit

        # Verify data was committed
        with get_session(engine) as session:
            retrieved = session.query(SolORM).filter_by(sol_number=921).first()
            assert retrieved is not None

    def test_session_rollback_on_error(self):
        """Test that session rolls back on error."""
        engine = get_engine(":memory:")
        create_all_tables(engine)

        try:
            with get_session(engine) as session:
                sol = SolORM(sol_number=921, data_source="loupe")
                session.add(sol)
                session.flush()
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify data was rolled back
        with get_session(engine) as session:
            retrieved = session.query(SolORM).filter_by(sol_number=921).first()
            assert retrieved is None


class TestCreateAllTables:
    """Tests for create_all_tables function."""

    def test_create_tables(self):
        """Test that all tables are created."""
        engine = get_engine(":memory:")
        create_all_tables(engine)

        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = [
            "sols",
            "scans",
            "scan_points",
            "spectra",
            "instrument_states",
            "ccd_configurations",
            "scanner_calibrations",
            "context_images",
            "regions_of_interest",
            "fitted_peaks",
        ]
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"

    def test_create_tables_idempotent(self):
        """Test that create_all_tables is idempotent."""
        engine = get_engine(":memory:")
        create_all_tables(engine)
        create_all_tables(engine)  # Should not raise

        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "sols" in tables


class TestEnsureDatabaseExists:
    """Tests for ensure_database_exists function."""

    def test_ensure_creates_file(self, tmp_path):
        """Test that ensure_database_exists creates the database file."""
        db_path = tmp_path / "subdir" / "test.db"
        engine = ensure_database_exists(db_path)

        assert db_path.exists()
        assert engine is not None

    def test_ensure_creates_tables(self, tmp_path):
        """Test that ensure_database_exists creates tables."""
        db_path = tmp_path / "test.db"
        engine = ensure_database_exists(db_path)

        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "sols" in tables


class TestForeignKeyEnforcement:
    """Tests for SQLite foreign key enforcement."""

    def test_foreign_key_violation_fails(self):
        """Test that foreign key violations are enforced."""
        engine = get_engine(":memory:")
        create_all_tables(engine)

        from sherloc_pipeline.database.models import ScanORM
        from sherloc_pipeline.database.connection import get_session_factory
        import uuid
        from sqlalchemy.exc import IntegrityError

        SessionLocal = get_session_factory(engine)
        session = SessionLocal()
        try:
            # Try to create a Scan without a Sol
            scan = ScanORM(
                id=str(uuid.uuid4()),
                sol_number=9999,  # Non-existent sol
                scan_name="Test",
                scan_id="test",
                sclk_start=100000,
                n_points=10,
                n_channels=2148,
                shots_per_point=10,
            )
            session.add(scan)
            with pytest.raises(IntegrityError):
                session.flush()
        finally:
            session.rollback()
            session.close()
