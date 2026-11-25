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
        still complete the backfill.
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


class TestRetainedPathCompatibilityEpoch:
    """The bounded pre-cleanup epoch preserves values and admits new writes."""

    REVISION = "b7e4f3a9c1d2"

    def test_retains_values_and_makes_legacy_path_nullable(self, alembic_config):
        config, db_path = alembic_config
        command.upgrade(config, "931df60632cb")

        engine = create_engine(f"sqlite:///{db_path}")
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO context_images "
                "(id,scan_id,image_type,file_path,r2_rel_key,created_at) "
                "VALUES ('existing','scan','ACI','/legacy/a.PNG',"
                "'loupe/a.PNG','2026-01-01 00:00:00')"
            ))

        command.upgrade(config, self.REVISION)

        inspector = inspect(engine)
        columns = {
            column["name"]: column
            for column in inspector.get_columns("context_images")
        }
        assert columns["file_path"]["nullable"] is True
        with engine.begin() as conn:
            assert conn.execute(text(
                "SELECT file_path FROM context_images WHERE id='existing'"
            )).scalar_one() == "/legacy/a.PNG"
            conn.execute(text(
                "INSERT INTO context_images "
                "(id,scan_id,image_type,r2_rel_key,created_at) "
                "VALUES ('new','scan','ACI','loupe/new.PNG',"
                "'2026-01-01 00:00:00')"
            ))
            revision = conn.execute(text(
                "SELECT version_num FROM alembic_version"
            )).scalar_one()
        assert revision == self.REVISION

    def test_final_cleanup_remains_a_separate_successor(self, alembic_config):
        config, db_path = alembic_config
        command.upgrade(config, self.REVISION)
        engine = create_engine(f"sqlite:///{db_path}")
        assert "file_path" in {
            column["name"]
            for column in inspect(engine).get_columns("context_images")
        }

        command.upgrade(config, "17db1a1940d6")
        assert "file_path" not in {
            column["name"]
            for column in inspect(engine).get_columns("context_images")
        }

    def test_downgrade_refuses_to_invent_missing_legacy_values(self, alembic_config):
        config, db_path = alembic_config
        command.upgrade(config, self.REVISION)
        engine = create_engine(f"sqlite:///{db_path}")
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO context_images "
                "(id,scan_id,image_type,r2_rel_key,created_at) "
                "VALUES ('new','scan','ACI','loupe/new.PNG',"
                "'2026-01-01 00:00:00')"
            ))
        with pytest.raises(RuntimeError, match="byte-exact predecessor snapshot"):
            command.downgrade(config, "931df60632cb")


class TestCrMaskReviewColumn:
    """4f638701cf7c: nullable fitted_peaks.cr_mask_review, existing rows NULL."""

    def test_adds_nullable_column_keeping_rows_and_downgrade_drops_it(self, alembic_config):
        import sqlite3

        config, db_path = alembic_config
        command.upgrade(config, "a13c4e5f6071")
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA foreign_keys=OFF")
        con.execute(
            "INSERT INTO fitted_peaks (id, spectrum_id, peak_type, amplitude, fit_modality, created_at) "
            "VALUES ('p1', 's1', 'gaussian', 1.0, 'minerals', '2026-11-12 00:00:00')"
        )
        con.commit()
        con.close()

        command.upgrade(config, "4f638701cf7c")
        con = sqlite3.connect(db_path)
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(fitted_peaks)")}
        assert "cr_mask_review" in cols and cols["cr_mask_review"][3] == 0  # nullable
        assert con.execute("SELECT id, cr_mask_review FROM fitted_peaks").fetchall() == [("p1", None)]
        con.close()

        command.downgrade(config, "a13c4e5f6071")
        con = sqlite3.connect(db_path)
        cols = {r[1] for r in con.execute("PRAGMA table_info(fitted_peaks)")}
        assert "cr_mask_review" not in cols
        assert con.execute("SELECT id FROM fitted_peaks").fetchall() == [("p1",)]
        con.close()

    def test_cr_mask_review_downgrade_preserves_fit_modality_guards(self, alembic_config, monkeypatch):
        import sqlite3
        from contextlib import closing

        def schema(conn):
            objects = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE tbl_name = 'fitted_peaks' ORDER BY type, name"
            )
            return (
                [(kind, name, table, " ".join(sql.split()) if kind == "table" else sql)
                 for kind, name, table, sql in objects],
                conn.execute("PRAGMA table_info(fitted_peaks)").fetchall(),
            )

        config, db_path = alembic_config
        insert_peak = (
            "INSERT INTO fitted_peaks (id, spectrum_id, peak_type, amplitude, fit_modality, created_at) "
            "VALUES (?, 's1', 'gaussian', 1.0, ?, '2026-11-12 00:00:00')"
        )
        command.upgrade(config, "4f638701cf7c")
        with closing(sqlite3.connect(db_path)) as conn:
            original = schema(conn)
            guards = [row for row in original[0] if row[0] == "trigger"]
            assert [row[1] for row in guards] == ["check_fit_modality_insert", "check_fit_modality_update"]
            conn.execute(insert_peak, ("p1", "minerals"))
            conn.commit()

        command.downgrade(config, "a13c4e5f6071")
        with closing(sqlite3.connect(db_path)) as conn:
            objects, columns = schema(conn)
            assert [row for row in objects if row[0] == "trigger"] == guards
            assert "cr_mask_review" not in {row[1] for row in columns}
            assert conn.execute("SELECT id, fit_modality FROM fitted_peaks").fetchall() == [("p1", "minerals")]
            with pytest.raises(sqlite3.IntegrityError, match="invalid fit_modality value"):
                conn.execute(insert_peak, ("p2", "invalid"))
            with pytest.raises(sqlite3.IntegrityError, match="invalid fit_modality value"):
                conn.execute("UPDATE fitted_peaks SET fit_modality = 'invalid' WHERE id = 'p1'")

        reference = db_path.with_name("reference.db")
        with monkeypatch.context() as patch:
            patch.setenv("PHASE_DATABASE_PATH", str(reference))
            command.upgrade(config, "a13c4e5f6071")
        with closing(sqlite3.connect(reference)) as conn:
            assert (objects, columns) == schema(conn)

        command.upgrade(config, "4f638701cf7c")
        with closing(sqlite3.connect(db_path)) as conn:
            assert schema(conn) == original


class TestSelectionMarksColumns:
    """a4d795f0a25e: nullable fitted_peaks.kept_by / aicc_gain / step_p, existing rows NULL."""

    def test_adds_nullable_columns_keeping_rows_and_downgrade_drops_them(self, alembic_config):
        import sqlite3

        config, db_path = alembic_config
        command.upgrade(config, "4f638701cf7c")
        con = sqlite3.connect(db_path)
        con.execute("PRAGMA foreign_keys=OFF")
        con.execute(
            "INSERT INTO fitted_peaks (id, spectrum_id, peak_type, amplitude, fit_modality, created_at) "
            "VALUES ('p1', 's1', 'gaussian', 1.0, 'minerals', '2026-11-23 00:00:00')"
        )
        con.commit()
        con.close()

        command.upgrade(config, "a4d795f0a25e")
        con = sqlite3.connect(db_path)
        cols = {r[1]: r for r in con.execute("PRAGMA table_info(fitted_peaks)")}
        for name in ("kept_by", "aicc_gain", "step_p"):
            assert name in cols and cols[name][3] == 0  # nullable
        assert con.execute("SELECT id, kept_by, aicc_gain, step_p FROM fitted_peaks").fetchall() == [
            ("p1", None, None, None)]
        con.close()

        command.downgrade(config, "4f638701cf7c")
        con = sqlite3.connect(db_path)
        cols = {r[1] for r in con.execute("PRAGMA table_info(fitted_peaks)")}
        assert not {"kept_by", "aicc_gain", "step_p"} & cols
        assert con.execute("SELECT id FROM fitted_peaks").fetchall() == [("p1",)]
        con.close()
