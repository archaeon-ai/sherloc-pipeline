"""
Unit tests for segmentation service.

Tests cover:
- Schema creation
- Grain storage and retrieval
- Batch processing logic
- Statistics gathering
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import json

from sqlalchemy import text

from sherloc_pipeline.database import get_engine, get_session
from sherloc_pipeline.services.segmentation import (
    SegmentationService,
    BatchStats,
    CREATE_GRAIN_SEGMENTS_TABLE,
)
from sherloc_pipeline.vision.segmentation import (
    SegmentationConfig,
    SegmentationModel,
    GrainMask,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with required schema."""
    db_path = tmp_path / "test.db"

    # Create minimal schema for testing
    engine = get_engine(db_path)
    with get_session(engine) as session:
        # Create sols table
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS sols (
                sol_number INTEGER PRIMARY KEY
            )
        """))

        # Create scans table
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS scans (
                id VARCHAR(36) PRIMARY KEY,
                sol_number INTEGER,
                target_name TEXT,
                scan_id TEXT,
                sclk_start INTEGER,
                n_points INTEGER,
                shots_per_point INTEGER,
                n_channels INTEGER DEFAULT 2148,
                laser_wavelength_nm FLOAT DEFAULT 248.6,
                created_at DATETIME
            )
        """))

        # Create context_images table
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS context_images (
                id VARCHAR(36) PRIMARY KEY,
                scan_id VARCHAR(36),
                image_type VARCHAR(10),
                r2_rel_key TEXT,
                file_format VARCHAR(10),
                sol_number INTEGER,
                width_px INTEGER,
                height_px INTEGER,
                created_at DATETIME
            )
        """))

        session.commit()

    return db_path


@pytest.fixture
def service(temp_db):
    """Create a segmentation service with temp database."""
    config = SegmentationConfig(model=SegmentationModel.WATERSHED)
    return SegmentationService(
        database_path=temp_db,
        segmentation_config=config,
    )


class TestBatchStats:
    """Test BatchStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = BatchStats()

        assert stats.images_processed == 0
        assert stats.images_failed == 0
        assert stats.total_grains == 0
        assert stats.errors == []

    def test_addition(self):
        """Test combining stats."""
        stats1 = BatchStats(
            images_processed=5,
            total_grains=100,
            errors=["error1"],
        )
        stats2 = BatchStats(
            images_processed=3,
            total_grains=50,
            errors=["error2"],
        )

        combined = stats1 + stats2

        assert combined.images_processed == 8
        assert combined.total_grains == 150
        assert combined.errors == ["error1", "error2"]


class TestSegmentationService:
    """Test SegmentationService class."""

    def test_schema_creation(self, service, temp_db):
        """Test that schema is created on initialization."""
        engine = get_engine(temp_db)
        with get_session(engine) as session:
            # Check grain_segments table exists
            result = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='grain_segments'"
            )).fetchone()
            assert result is not None

            # Check segmentation_jobs table exists
            result = session.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='segmentation_jobs'"
            )).fetchone()
            assert result is not None

    def test_store_grain(self, service, temp_db):
        """Test storing a grain in the database."""
        # Create test image record first
        engine = get_engine(temp_db)
        with get_session(engine) as session:
            session.execute(text("""
                INSERT INTO context_images (id, scan_id, image_type, r2_rel_key, file_format, created_at)
                VALUES ('img-001', 'scan-001', 'ACI', 'sol_0001/data_aci/image.IMG', 'IMG', datetime('now'))
            """))
            session.commit()

        # Create grain mask
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True

        grain = GrainMask(
            segment_index=0,
            mask=mask,
            bbox=[30, 20, 20, 20],
            area=400,
            predicted_iou=0.95,
            stability_score=0.98,
            centroid=(40.0, 30.0),
            perimeter=80.0,
            aspect_ratio=1.0,
            circularity=0.79,
        )

        # Store grain
        with get_session(engine) as session:
            grain_id = service._store_grain(session, "img-001", grain, "watershed")
            session.commit()

            # Verify storage
            result = session.execute(text(
                "SELECT id, image_id, segment_index, area_px FROM grain_segments WHERE id = :id"
            ), {"id": grain_id}).fetchone()

            assert result is not None
            assert result[1] == "img-001"  # image_id
            assert result[2] == 0  # segment_index
            assert result[3] == 400  # area_px

    def test_get_grains_for_image(self, service, temp_db):
        """Test retrieving grains for an image."""
        engine = get_engine(temp_db)

        # Create test data
        with get_session(engine) as session:
            session.execute(text("""
                INSERT INTO context_images (id, scan_id, image_type, r2_rel_key, file_format, created_at)
                VALUES ('img-002', 'scan-001', 'ACI', 'sol_0001/data_aci/image2.IMG', 'IMG', datetime('now'))
            """))

            # Insert grains
            mask = np.zeros((50, 50), dtype=bool)
            mask[10:20, 10:20] = True

            for i in range(3):
                grain = GrainMask(
                    segment_index=i,
                    mask=mask,
                    bbox=[10, 10, 10, 10],
                    area=100 + i * 50,
                    predicted_iou=0.9,
                    stability_score=0.95,
                )
                service._store_grain(session, "img-002", grain, "watershed")

            session.commit()

        # Retrieve grains
        grains = service.get_grains_for_image("img-002")

        assert len(grains) == 3
        assert grains[0]["segment_index"] == 0
        assert grains[1]["segment_index"] == 1
        assert grains[2]["segment_index"] == 2

    def test_get_stats_empty(self, service):
        """Test stats with no segmentation data."""
        stats = service.get_stats()

        assert stats["total_grains"] == 0
        assert stats["images_with_segments"] == 0
        assert stats["coverage_pct"] == 0

    def test_get_stats_with_data(self, service, temp_db):
        """Test stats with segmentation data."""
        engine = get_engine(temp_db)

        with get_session(engine) as session:
            # Create test images
            session.execute(text("""
                INSERT INTO context_images (id, scan_id, image_type, r2_rel_key, file_format, created_at)
                VALUES
                    ('img-a', 'scan-001', 'ACI', 'sol_0001/data_aci/a.IMG', 'IMG', datetime('now')),
                    ('img-b', 'scan-001', 'ACI', 'sol_0001/data_aci/b.IMG', 'IMG', datetime('now'))
            """))

            # Create grains for first image
            mask = np.zeros((50, 50), dtype=bool)
            mask[10:20, 10:20] = True

            for i in range(5):
                grain = GrainMask(
                    segment_index=i,
                    mask=mask,
                    bbox=[10, 10, 10, 10],
                    area=100,
                    predicted_iou=0.9,
                    stability_score=0.95,
                )
                service._store_grain(session, "img-a", grain, "watershed")

            session.commit()

        stats = service.get_stats()

        assert stats["total_grains"] == 5
        assert stats["images_with_segments"] == 1
        assert stats["total_images"] == 2
        assert stats["coverage_pct"] == 50.0
        assert stats["by_model"]["watershed"] == 5


class TestJobManagement:
    """Test job creation and updating."""

    def test_create_job(self, service, temp_db):
        """Test creating a segmentation job."""
        engine = get_engine(temp_db)

        with get_session(engine) as session:
            job_id = service._create_job(session, total_images=100)
            session.commit()

            # Verify job created
            result = session.execute(text(
                "SELECT id, status, images_total FROM segmentation_jobs WHERE id = :id"
            ), {"id": job_id}).fetchone()

            assert result is not None
            assert result[1] == "running"  # status
            assert result[2] == 100  # images_total

    def test_update_job(self, service, temp_db):
        """Test updating job progress."""
        engine = get_engine(temp_db)

        with get_session(engine) as session:
            job_id = service._create_job(session, total_images=100)
            session.commit()

            # Update progress
            service._update_job(
                session, job_id,
                images_processed=50,
                images_failed=2,
                total_grains=500,
                last_image_id="img-050",
            )
            session.commit()

            # Verify update
            result = session.execute(text(
                "SELECT images_processed, images_failed, total_grains FROM segmentation_jobs WHERE id = :id"
            ), {"id": job_id}).fetchone()

            assert result[0] == 50
            assert result[1] == 2
            assert result[2] == 500

    def test_complete_job(self, service, temp_db):
        """Test completing a job."""
        engine = get_engine(temp_db)

        with get_session(engine) as session:
            job_id = service._create_job(session, total_images=10)
            session.commit()

            # Complete job
            service._update_job(
                session, job_id,
                images_processed=10,
                images_failed=0,
                total_grains=100,
                status="completed",
            )
            session.commit()

            # Verify completion
            result = session.execute(text(
                "SELECT status, completed_at FROM segmentation_jobs WHERE id = :id"
            ), {"id": job_id}).fetchone()

            assert result[0] == "completed"
            assert result[1] is not None  # completed_at should be set


class _FakeSegResult:
    grains = []
    n_grains = 0
    model_name = "fake"
    inference_time_s = 0.0


class _FakeSegmenter:
    """Stand-in for GrainSegmenter so the read path can be exercised without
    a real IMG or a real segmentation model."""

    last_image_path = None

    def __init__(self, _config):
        pass

    def segment(self, _image, *, image_id, image_path, compute_morphometry):
        _FakeSegmenter.last_image_path = image_path
        return _FakeSegResult()

    def unload_model(self):
        pass


class TestProcessImageLocatorResolution:
    """process_image() reads disk from the stored locator (r2_rel_key);
    the transitional file_path column was dropped (#7)."""

    def test_process_image_resolves_locator(self, temp_db, tmp_path, monkeypatch):
        """A well-formed locator resolves to ``<data_root>/<locator>`` and that
        resolved path is what reaches read_aci_image / the segmenter."""
        data_root = tmp_path / "data"
        locator = "loupe/sol_0921/detail_1/ws/img/SC3_0921.IMG"
        expected_path = str(data_root / locator)

        with get_session(get_engine(temp_db)) as session:
            session.execute(text("""
                INSERT INTO context_images
                    (id, scan_id, image_type, r2_rel_key, file_format, created_at)
                VALUES
                    ('img-loc', 'scan-1', 'ACI', :loc, 'IMG', datetime('now'))
            """), {"loc": locator})
            session.commit()

        read_calls = []

        def fake_read(path, validate_dimensions=False):
            read_calls.append(path)
            return np.zeros((8, 8), dtype=np.uint8), None

        monkeypatch.setattr(
            "sherloc_pipeline.services.segmentation.read_aci_image", fake_read
        )
        monkeypatch.setattr(
            "sherloc_pipeline.services.segmentation.GrainSegmenter", _FakeSegmenter
        )

        config = SegmentationConfig(model=SegmentationModel.WATERSHED)
        service = SegmentationService(
            database_path=temp_db,
            segmentation_config=config,
            data_root=data_root,
        )

        result = service.process_image("img-loc")

        assert result.metadata["success"] is True
        # The resolved disk path (data_root + locator) is what was read.
        assert read_calls == [expected_path]
        assert _FakeSegmenter.last_image_path == expected_path

    def test_process_image_null_locator_raises(self, temp_db, tmp_path, monkeypatch):
        """A NULL locator fails like a missing file, naming the locator (#7)."""
        from sherloc_pipeline.services.segmentation import SegmentationError

        with get_session(get_engine(temp_db)) as session:
            session.execute(text("""
                INSERT INTO context_images
                    (id, scan_id, image_type, r2_rel_key, file_format, created_at)
                VALUES
                    ('img-null', 'scan-1', 'ACI', NULL, 'IMG', datetime('now'))
            """))
            session.commit()

        # read_aci_image must NOT be reached for a null locator.
        def fail_read(*_a, **_k):
            raise AssertionError("read_aci_image should not be called")

        monkeypatch.setattr(
            "sherloc_pipeline.services.segmentation.read_aci_image", fail_read
        )
        monkeypatch.setattr(
            "sherloc_pipeline.services.segmentation.GrainSegmenter", _FakeSegmenter
        )

        service = SegmentationService(
            database_path=temp_db,
            segmentation_config=SegmentationConfig(model=SegmentationModel.WATERSHED),
            data_root=tmp_path / "data",
        )

        with pytest.raises(SegmentationError) as excinfo:
            service.process_image("img-null")
        msg = str(excinfo.value).lower()
        assert "resolve" in msg and "locator" in msg
