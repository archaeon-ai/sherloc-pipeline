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
            "cosmic_ray_masks",
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
        for table in ["sols", "scans", "scan_points", "spectra", "cosmic_ray_masks"]:
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

        # Check cosmic_ray_masks indexes
        cr_mask_indexes = inspector.get_indexes("cosmic_ray_masks")
        cr_mask_index_names = [idx["name"] for idx in cr_mask_indexes]
        assert "ix_cosmic_ray_masks_spectrum_id" in cr_mask_index_names

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

        # Check cosmic_ray_masks -> spectra foreign key
        cr_mask_fks = inspector.get_foreign_keys("cosmic_ray_masks")
        assert len(cr_mask_fks) == 1
        assert cr_mask_fks[0]["referred_table"] == "spectra"


class TestR2RelKeyBackfill:
    """Migration 931df60632cb: r2_rel_key column + file_path backfill."""

    # (id, file_path, expected r2_rel_key) covering every file_path shape
    # observed in the production team + public databases at migration
    # time, plus the no-match case.
    CASES = [
        (
            "img-team-canonical",
            "/data/sherloc/data/loupe/sol_0921/detail_1/ws/img/a.PNG",
            "loupe/sol_0921/detail_1/ws/img/a.PNG",
        ),
        (
            "img-team-legacy-nas",
            "/nas/000_sherloc/data/loupe/sol_0852/detail_1/ws/img/b.PNG",
            "loupe/sol_0852/detail_1/ws/img/b.PNG",
        ),
        (
            "img-public-cache",
            "/data/sherloc/pds/sol_0712/data_aci/c.IMG",
            "sol_0712/data_aci/c.IMG",
        ),
        (
            "img-pds-sentinel",
            "pds:urn:nasa:pds:mars2020_imgops:data_aci_imgops:x::1.0",
            "pds:urn:nasa:pds:mars2020_imgops:data_aci_imgops:x::1.0",
        ),
        ("img-unknown-layout", "/somewhere/else/entirely/d.PNG", None),
        (
            "img-traversal",
            "/data/sherloc/data/loupe/../../../etc/passwd",
            None,
        ),
    ]

    def test_backfill_transforms(self, alembic_config):
        config, db_path = alembic_config

        # Build the schema as it existed before the locator migration.
        command.upgrade(config, "9b2e7c4a1f08")

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            from sqlalchemy import text

            for row_id, file_path, _ in self.CASES:
                conn.execute(
                    text(
                        "INSERT INTO context_images "
                        "(id, scan_id, image_type, file_path, created_at) "
                        "VALUES (:id, :scan_id, 'ACI', :file_path, "
                        "'2026-01-01 00:00:00')"
                    ),
                    {
                        "id": row_id,
                        "scan_id": "scan-under-test",
                        "file_path": file_path,
                    },
                )

        # Apply the locator migration.
        command.upgrade(config, "head")

        with engine.connect() as conn:
            from sqlalchemy import text

            for row_id, _, expected in self.CASES:
                got = conn.execute(
                    text(
                        "SELECT r2_rel_key FROM context_images "
                        "WHERE id = :id"
                    ),
                    {"id": row_id},
                ).scalar()
                assert got == expected, (
                    f"{row_id}: expected {expected!r}, got {got!r}"
                )

    def test_retry_after_partial_application(self, alembic_config):
        """Retrying after an interrupted run must not fail on duplicate column.

        Simulates the known swap/stamp hazard: the column was added but
        the revision was never stamped (e.g. deploy interrupted between
        DDL and stamp). A retried `upgrade head` must skip the ALTER and
        still complete the backfill (review F3).
        """
        config, db_path = alembic_config

        command.upgrade(config, "9b2e7c4a1f08")

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            from sqlalchemy import text

            # Manually pre-add the column: the partially-applied state.
            conn.execute(
                text("ALTER TABLE context_images ADD COLUMN r2_rel_key TEXT")
            )
            conn.execute(
                text(
                    "INSERT INTO context_images "
                    "(id, scan_id, image_type, file_path, created_at) "
                    "VALUES ('img-retry', 'scan-under-test', 'ACI', "
                    "'/data/sherloc/data/loupe/sol_0921/ws/img/r.PNG', "
                    "'2026-01-01 00:00:00')"
                )
            )

        # Retried upgrade: must not raise, and must still backfill.
        command.upgrade(config, "head")

        with engine.connect() as conn:
            from sqlalchemy import text

            got = conn.execute(
                text(
                    "SELECT r2_rel_key FROM context_images "
                    "WHERE id = 'img-retry'"
                )
            ).scalar()
            assert got == "loupe/sol_0921/ws/img/r.PNG"
