"""
Tests for Alembic migrations.

Tests cover:
- Migration can be applied to a fresh database
- Migration up/down cycle works
- All expected tables and indexes are created
"""

import os
import tempfile
from pathlib import Path

import pytest

from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, inspect


@pytest.fixture
def alembic_config(tmp_path):
    """Create an Alembic config for testing."""
    db_path = tmp_path / "test.db"

    # Get the project root
    project_root = Path(__file__).parent.parent.parent.parent

    # Create Alembic config
    config = Config(str(project_root / "alembic.ini"))

    # Override the database URL
    os.environ["PHASE_DATABASE_PATH"] = str(db_path)

    yield config, db_path

    # Cleanup
    del os.environ["PHASE_DATABASE_PATH"]


class TestAlembicMigrations:
    """Tests for Alembic migrations."""

    def test_upgrade_head(self, alembic_config):
        """Test running alembic upgrade head."""
        config, db_path = alembic_config

        # Run upgrade
        command.upgrade(config, "head")

        # Verify database was created
        assert db_path.exists()

        # Verify tables exist
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        expected_tables = [
            "alembic_version",
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
            assert table in tables, f"Table {table} not found after migration"

    def test_upgrade_then_downgrade(self, alembic_config):
        """Test upgrade then downgrade."""
        config, db_path = alembic_config

        # Upgrade
        command.upgrade(config, "head")

        # Verify tables exist
        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)
        tables_after_upgrade = inspector.get_table_names()
        assert "sols" in tables_after_upgrade

        # Downgrade
        command.downgrade(config, "base")

        # Refresh inspector
        inspector = inspect(engine)
        tables_after_downgrade = inspector.get_table_names()

        # Only alembic_version should remain (or be empty)
        for table in ["sols", "scans", "scan_points", "spectra"]:
            assert table not in tables_after_downgrade

    def test_indexes_created(self, alembic_config):
        """Test that indexes are created by migration."""
        config, db_path = alembic_config

        # Run upgrade
        command.upgrade(config, "head")

        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)

        # Check scan indexes
        scan_indexes = inspector.get_indexes("scans")
        index_names = [idx["name"] for idx in scan_indexes]

        assert "ix_scans_sol_number" in index_names
        assert "ix_scans_scan_name" in index_names
        assert "ix_scans_scan_id" in index_names
        assert "ix_scans_sol_scan_name" in index_names  # composite index

        # Check fitted_peaks indexes
        peak_indexes = inspector.get_indexes("fitted_peaks")
        peak_index_names = [idx["name"] for idx in peak_indexes]

        assert "ix_fitted_peaks_center_cm1" in peak_index_names or "ix_fitted_peaks_center_range" in peak_index_names
        assert "ix_fitted_peaks_mineral_assignment" in peak_index_names or "ix_fitted_peaks_mineral" in peak_index_names

    def test_foreign_keys_created(self, alembic_config):
        """Test that foreign keys are created by migration."""
        config, db_path = alembic_config

        # Run upgrade
        command.upgrade(config, "head")

        engine = create_engine(f"sqlite:///{db_path}")
        inspector = inspect(engine)

        # Check scans foreign keys (sols + self-referential parent_scan_id)
        scan_fks = inspector.get_foreign_keys("scans")
        assert len(scan_fks) == 2
        fk_tables = {fk["referred_table"] for fk in scan_fks}
        assert "sols" in fk_tables
        assert "scans" in fk_tables  # parent_scan_id self-FK

        # Check scan_points -> scans foreign key
        point_fks = inspector.get_foreign_keys("scan_points")
        assert len(point_fks) == 1
        assert point_fks[0]["referred_table"] == "scans"

        # Check spectra -> scan_points foreign key
        spectra_fks = inspector.get_foreign_keys("spectra")
        assert len(spectra_fks) == 1
        assert spectra_fks[0]["referred_table"] == "scan_points"

        # Check fitted_peaks -> spectra foreign key
        peak_fks = inspector.get_foreign_keys("fitted_peaks")
        assert len(peak_fks) == 1
        assert peak_fks[0]["referred_table"] == "spectra"
