"""
Unit tests for processing state columns on ScanORM.

Tests that the 5 new processing state columns exist on ScanORM, have correct
default values (all None), and can be set to valid values.
"""

import hashlib
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text

from sherloc_pipeline.database.connection import get_engine
from sherloc_pipeline.database.models import Base, ScanORM, SolORM


@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = get_engine(":memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Database session with a default Sol row for FK requirements."""
    from sqlalchemy.orm import Session

    with Session(engine) as sess:
        sol = SolORM(sol_number=921, data_source="loupe")
        sess.add(sol)
        sess.commit()
        yield sess


def _make_scan(**overrides):
    """Return a minimal valid ScanORM kwargs dict."""
    kwargs = dict(
        id=str(uuid.uuid4()),
        sol_number=921,
        scan_name="detail_1",
        scan_id="test_scan_001",
        sclk_start=100000,
        n_points=50,
        n_channels=2148,
        laser_wavelength_nm=248.6,
        target_type="mars_target",
    )
    kwargs.update(overrides)
    return kwargs


class TestProcessingStateColumnsExist:
    """Verify the 5 new columns are present on ScanORM."""

    def test_processing_status_column_exists(self):
        cols = {c.key for c in ScanORM.__mapper__.columns}
        assert "processing_status" in cols

    def test_processed_at_column_exists(self):
        cols = {c.key for c in ScanORM.__mapper__.columns}
        assert "processed_at" in cols

    def test_processing_config_hash_column_exists(self):
        cols = {c.key for c in ScanORM.__mapper__.columns}
        assert "processing_config_hash" in cols

    def test_processing_pipeline_version_column_exists(self):
        cols = {c.key for c in ScanORM.__mapper__.columns}
        assert "processing_pipeline_version" in cols

    def test_processing_error_column_exists(self):
        cols = {c.key for c in ScanORM.__mapper__.columns}
        assert "processing_error" in cols


class TestProcessingStateDefaults:
    """Verify new columns default to None (NULL) for unprocessed scans."""

    def test_all_processing_fields_default_to_none(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()
        session.refresh(scan)

        assert scan.processing_status is None
        assert scan.processed_at is None
        assert scan.processing_config_hash is None
        assert scan.processing_pipeline_version is None
        assert scan.processing_error is None

    def test_unprocessed_scan_has_null_status(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()

        # NULL processing_status means unprocessed
        assert scan.processing_status is None


class TestProcessingStateSetCompleted:
    """Verify processing_status can be set to 'completed'."""

    def test_set_processing_status_completed(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()

        scan.processing_status = "completed"
        scan.processed_at = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
        scan.processing_pipeline_version = "3.0.0"
        session.commit()
        session.refresh(scan)

        assert scan.processing_status == "completed"
        assert scan.processing_pipeline_version == "3.0.0"
        assert scan.processed_at is not None

    def test_completed_scan_has_no_error(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()

        scan.processing_status = "completed"
        scan.processing_error = None
        session.commit()
        session.refresh(scan)

        assert scan.processing_status == "completed"
        assert scan.processing_error is None


class TestProcessingStateSetFailed:
    """Verify processing_status can be set to 'failed' with an error message."""

    def test_set_processing_status_failed(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()

        scan.processing_status = "failed"
        scan.processing_error = "ValueError: spectrum array has wrong shape (expected 523, got 0)"
        session.commit()
        session.refresh(scan)

        assert scan.processing_status == "failed"
        assert "ValueError" in scan.processing_error

    def test_failed_scan_error_persists_round_trip(self, session):
        error_msg = "RuntimeError: baseline subtraction failed at point 42"
        scan = ScanORM(**_make_scan())
        scan.processing_status = "failed"
        scan.processing_error = error_msg
        session.add(scan)
        session.commit()

        # Re-fetch from DB
        fetched = session.get(ScanORM, scan.id)
        assert fetched.processing_error == error_msg


class TestProcessingConfigHash:
    """Verify processing_config_hash stores a SHA256 string."""

    def test_config_hash_stores_sha256(self, session):
        config_str = '{"fitting": {"max_peaks": 5, "min_snr": 3.0}}'
        sha256 = hashlib.sha256(config_str.encode()).hexdigest()
        assert len(sha256) == 64

        scan = ScanORM(**_make_scan())
        scan.processing_config_hash = sha256
        session.add(scan)
        session.commit()
        session.refresh(scan)

        assert scan.processing_config_hash == sha256
        assert len(scan.processing_config_hash) == 64

    def test_config_hash_is_nullable(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()
        session.refresh(scan)

        assert scan.processing_config_hash is None


class TestProcessingStateInDatabase:
    """Verify column presence in the actual SQLite schema (not just ORM)."""

    def test_processing_columns_in_sqlite_schema(self, engine):
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("scans")}

        expected = {
            "processing_status",
            "processed_at",
            "processing_config_hash",
            "processing_pipeline_version",
            "processing_error",
        }
        missing = expected - columns
        assert not missing, f"Missing columns in DB schema: {missing}"


class TestProcessingStateFullWorkflow:
    """Simulate a full processing state lifecycle: unprocessed -> completed."""

    def test_lifecycle_unprocessed_to_completed(self, session):
        # Start: unprocessed
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()
        assert scan.processing_status is None

        # After pipeline run: completed
        config_hash = hashlib.sha256(b'{"max_peaks": 5}').hexdigest()
        scan.processing_status = "completed"
        scan.processed_at = datetime(2026, 3, 19, 15, 30, 0, tzinfo=timezone.utc)
        scan.processing_config_hash = config_hash
        scan.processing_pipeline_version = "3.0.0"
        scan.processing_error = None
        session.commit()
        session.refresh(scan)

        assert scan.processing_status == "completed"
        assert scan.processing_config_hash == config_hash
        assert scan.processing_pipeline_version == "3.0.0"
        assert scan.processing_error is None

    def test_lifecycle_unprocessed_to_failed_to_completed(self, session):
        scan = ScanORM(**_make_scan())
        session.add(scan)
        session.commit()

        # First attempt fails
        scan.processing_status = "failed"
        scan.processing_error = "FileNotFoundError: R1_normalized.csv not found"
        session.commit()
        assert scan.processing_status == "failed"

        # After fix: re-runs successfully
        scan.processing_status = "completed"
        scan.processing_error = None
        scan.processing_pipeline_version = "3.0.1"
        session.commit()
        session.refresh(scan)

        assert scan.processing_status == "completed"
        assert scan.processing_error is None
        assert scan.processing_pipeline_version == "3.0.1"
