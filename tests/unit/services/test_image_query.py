"""
Tests for ImageQueryService.

Tests cover:
- Query by sol, scan, SCLK range, target
- Query filtering by format, camera, type
- Image counting and statistics
- Export functionality
- Image loading
- Scan point overlay retrieval
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import uuid
import json

import numpy as np

from sherloc_pipeline.services.image_query import (
    ImageQueryService,
    ImageQueryError,
    ImageInfo,
    ExportStats,
)
from sqlalchemy import select

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    create_all_tables,
    SolORM,
    ScanORM,
    ScanPointORM,
    ContextImageORM,
)


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_from_orm_basic(self):
        """Test creating ImageInfo from ORM without scan."""
        # Create a mock ORM object
        class MockImage:
            id = "test-id"
            scan_id = "scan-id"
            sol_number = 921
            sclk_start = 748731411
            file_path = "/path/to/image.IMG"
            file_format = "IMG"
            camera_id = "SC3"
            image_type = "ACI"
            product_id = "TEST_PRODUCT"
            width_px = 1648
            height_px = 1200
            image_time = datetime(2023, 5, 1, 12, 0, 0)
            focus_mode = "AUTOFOCUS"
            local_mean_solar_time = "12:30:00"

        info = ImageInfo.from_orm(MockImage())

        assert info.id == "test-id"
        assert info.sol_number == 921
        assert info.camera_id == "SC3"
        assert info.scan_target is None

    def test_from_orm_with_scan(self):
        """Test creating ImageInfo from ORM with scan info."""
        class MockImage:
            id = "test-id"
            scan_id = "scan-id"
            sol_number = 921
            sclk_start = 748731411
            file_path = "/path/to/image.IMG"
            file_format = "IMG"
            camera_id = "SC3"
            image_type = "ACI"
            product_id = "TEST_PRODUCT"
            width_px = 1648
            height_px = 1200
            image_time = None
            focus_mode = None
            local_mean_solar_time = None

        class MockScan:
            scan_name = "Garde_Abrasion_Patch"
            n_points = 100

        info = ImageInfo.from_orm(MockImage(), MockScan())

        assert info.scan_target == "Garde_Abrasion_Patch"
        assert info.scan_n_points == 100

    def test_to_dict(self):
        """Test converting ImageInfo to dictionary."""
        info = ImageInfo(
            id="test-id",
            scan_id="scan-id",
            sol_number=921,
            sclk_start=748731411,
            file_path="/path/to/image.IMG",
            file_format="IMG",
            camera_id="SC3",
            image_type="ACI",
            product_id="TEST",
            width_px=1648,
            height_px=1200,
            image_time=datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
            focus_mode="AUTOFOCUS",
            local_time="12:30:00",
        )

        d = info.to_dict()

        assert d["id"] == "test-id"
        assert d["sol_number"] == 921
        assert "2023-05-01" in d["image_time"]

    def test_to_dict_with_none_datetime(self):
        """Test to_dict handles None datetime."""
        info = ImageInfo(
            id="test-id",
            scan_id="scan-id",
            sol_number=None,
            sclk_start=None,
            file_path="/path/to/image.IMG",
            file_format="PNG",
            camera_id=None,
            image_type="ACI",
            product_id=None,
            width_px=None,
            height_px=None,
            image_time=None,
            focus_mode=None,
            local_time=None,
        )

        d = info.to_dict()
        assert d["image_time"] is None


class TestExportStats:
    """Tests for ExportStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = ExportStats()

        assert stats.images_requested == 0
        assert stats.images_exported == 0
        assert stats.errors == []

    def test_with_values(self):
        """Test with actual values."""
        stats = ExportStats(
            images_requested=10,
            images_exported=8,
            images_skipped=1,
            images_failed=1,
            total_bytes=1024,
            errors=["error1"],
        )

        assert stats.images_requested == 10
        assert stats.images_failed == 1
        assert len(stats.errors) == 1


@pytest.fixture
def db_with_images(tmp_path):
    """Create a test database with sample images."""
    db_path = tmp_path / "test.db"
    engine = get_engine(db_path)
    create_all_tables(engine)

    with get_session(engine) as session:
        # Create sols
        for sol_num in [921, 922, 1242]:
            sol = SolORM(
                sol_number=sol_num,
                data_source="loupe",
                created_at=datetime.now(timezone.utc),
            )
            session.add(sol)

        session.flush()

        # Create scans
        scan1_id = str(uuid.uuid4())
        scan1 = ScanORM(
            id=scan1_id,
            sol_number=921,
            scan_name="Garde_Abrasion_Patch",
            scan_id="SrlcSpecSpecSohRaw_0748731411-51550-1",
            sclk_start=748731411,
            n_points=100,
            n_channels=2148,
            shots_per_point=10,
            laser_wavelength_nm=248.6,
            created_at=datetime.now(timezone.utc),
        )
        session.add(scan1)

        scan2_id = str(uuid.uuid4())
        scan2 = ScanORM(
            id=scan2_id,
            sol_number=922,
            scan_name="Berry_Hollow",
            scan_id="SrlcSpecSpecSohRaw_0748831411-51550-1",
            sclk_start=748831411,
            n_points=50,
            n_channels=2148,
            shots_per_point=10,
            laser_wavelength_nm=248.6,
            created_at=datetime.now(timezone.utc),
        )
        session.add(scan2)

        session.flush()

        # Create scan points for scan1
        for i in range(5):
            point = ScanPointORM(
                id=str(uuid.uuid4()),
                scan_id=scan1_id,
                point_index=i,
                x_pixel=100.0 + i * 50,
                y_pixel=200.0 + i * 30,
                created_at=datetime.now(timezone.utc),
            )
            session.add(point)

        # Create images
        images = [
            # Sol 921, scan1, IMG format
            {
                "id": str(uuid.uuid4()),
                "scan_id": scan1_id,
                "sol_number": 921,
                "sclk_start": 748731397,
                "file_path": "/data/sol_921/image1.IMG",
                "file_format": "IMG",
                "camera_id": "SC3",
                "image_type": "ACI",
            },
            {
                "id": str(uuid.uuid4()),
                "scan_id": scan1_id,
                "sol_number": 921,
                "sclk_start": 748731400,
                "file_path": "/data/sol_921/image2.IMG",
                "file_format": "IMG",
                "camera_id": "SC2",
                "image_type": "ACI",
            },
            # Sol 921, PNG format
            {
                "id": str(uuid.uuid4()),
                "scan_id": scan1_id,
                "sol_number": 921,
                "sclk_start": None,
                "file_path": "/data/sol_921/image3.PNG",
                "file_format": "PNG",
                "camera_id": None,
                "image_type": "ACI",
            },
            # Sol 922, scan2
            {
                "id": str(uuid.uuid4()),
                "scan_id": scan2_id,
                "sol_number": 922,
                "sclk_start": 748831397,
                "file_path": "/data/sol_922/image4.IMG",
                "file_format": "IMG",
                "camera_id": "SC3",
                "image_type": "ACI",
            },
            # Sol 1242, WATSON image
            {
                "id": str(uuid.uuid4()),
                "scan_id": scan2_id,
                "sol_number": 1242,
                "sclk_start": 777229084,
                "file_path": "/data/sol_1242/watson.IMG",
                "file_format": "IMG",
                "camera_id": "SC1",
                "image_type": "WATSON",
            },
        ]

        for img_data in images:
            image = ContextImageORM(
                id=img_data["id"],
                scan_id=img_data["scan_id"],
                sol_number=img_data["sol_number"],
                sclk_start=img_data["sclk_start"],
                file_path=img_data["file_path"],
                file_format=img_data["file_format"],
                camera_id=img_data["camera_id"],
                image_type=img_data["image_type"],
                created_at=datetime.now(timezone.utc),
            )
            session.add(image)

        session.commit()

    return db_path, scan1_id, scan2_id


class TestQueryBySol:
    """Tests for query_by_sol method."""

    def test_query_basic(self, db_with_images):
        """Test basic sol query."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sol(921)

        assert len(images) == 3

    def test_query_with_format_filter(self, db_with_images):
        """Test sol query with format filter."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sol(921, file_format="IMG")

        assert len(images) == 2
        assert all(img.file_format == "IMG" for img in images)

    def test_query_with_camera_filter(self, db_with_images):
        """Test sol query with camera filter."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sol(921, camera_id="SC3")

        assert len(images) == 1
        assert images[0].camera_id == "SC3"

    def test_query_nonexistent_sol(self, db_with_images):
        """Test query for sol with no images."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sol(999)

        assert len(images) == 0

    def test_query_includes_scan_info(self, db_with_images):
        """Test that scan info is included by default."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sol(921)

        assert any(img.scan_target == "Garde_Abrasion_Patch" for img in images)


class TestQueryByScan:
    """Tests for query_by_scan method."""

    def test_query_basic(self, db_with_images):
        """Test basic scan query."""
        db_path, scan1_id, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_scan(scan1_id)

        assert len(images) == 3

    def test_query_with_format(self, db_with_images):
        """Test scan query with format filter."""
        db_path, scan1_id, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_scan(scan1_id, file_format="PNG")

        assert len(images) == 1

    def test_query_nonexistent_scan(self, db_with_images):
        """Test query for nonexistent scan."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_scan("nonexistent-id")

        assert len(images) == 0


class TestQueryBySCLKRange:
    """Tests for query_by_sclk_range method."""

    def test_query_basic(self, db_with_images):
        """Test basic SCLK range query."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sclk_range(748731395, 748731410)

        assert len(images) == 2

    def test_query_with_sol(self, db_with_images):
        """Test SCLK query with sol filter."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sclk_range(748731395, 748831400, sol_number=921)

        assert len(images) == 2
        assert all(img.sol_number == 921 for img in images)

    def test_query_no_results(self, db_with_images):
        """Test SCLK query with no matches."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_sclk_range(1, 100)

        assert len(images) == 0


class TestQueryByTarget:
    """Tests for query_by_target method."""

    def test_exact_match(self, db_with_images):
        """Test exact target match."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_target("Garde_Abrasion_Patch")

        assert len(images) == 3

    def test_partial_match(self, db_with_images):
        """Test partial target match."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_target("Garde", partial_match=True)

        assert len(images) == 3

    def test_no_match(self, db_with_images):
        """Test no matching target."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_target("Nonexistent")

        assert len(images) == 0


class TestQueryAll:
    """Tests for query_all method."""

    def test_basic(self, db_with_images):
        """Test query all images."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_all()

        assert len(images) == 5

    def test_with_limit(self, db_with_images):
        """Test query with limit."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_all(limit=2)

        assert len(images) == 2

    def test_with_offset(self, db_with_images):
        """Test query with offset."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_all(limit=2, offset=3)

        assert len(images) == 2

    def test_with_type_filter(self, db_with_images):
        """Test query with type filter."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_all(image_type="WATSON")

        assert len(images) == 1
        assert images[0].image_type == "WATSON"


class TestGetImageById:
    """Tests for get_image_by_id method."""

    def test_found(self, db_with_images):
        """Test getting existing image."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        # Get first image ID
        images = service.query_all(limit=1)
        image_id = images[0].id

        result = service.get_image_by_id(image_id)

        assert result is not None
        assert result.id == image_id

    def test_not_found(self, db_with_images):
        """Test getting nonexistent image."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        result = service.get_image_by_id("nonexistent-id")

        assert result is None


class TestCountImages:
    """Tests for count_images method."""

    def test_count_all(self, db_with_images):
        """Test counting all images."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        count = service.count_images()

        assert count == 5

    def test_count_with_filters(self, db_with_images):
        """Test counting with filters."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        count = service.count_images(file_format="IMG")
        assert count == 4

        count = service.count_images(camera_id="SC3")
        assert count == 2

        count = service.count_images(sol_number=921)
        assert count == 3


class TestGetAvailableSols:
    """Tests for get_available_sols method."""

    def test_basic(self, db_with_images):
        """Test getting available sols."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        sols = service.get_available_sols()

        assert 921 in sols
        assert 922 in sols
        assert 1242 in sols


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_basic(self, db_with_images):
        """Test getting statistics."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        stats = service.get_statistics()

        assert stats["total"] == 5
        assert stats["by_format"]["IMG"] == 4
        assert stats["by_format"]["PNG"] == 1
        assert "SC3" in stats["by_camera"]
        assert stats["by_type"]["ACI"] == 4
        assert stats["by_type"]["WATSON"] == 1


class TestGetScanPointsForImage:
    """Tests for get_scan_points_for_image method."""

    def test_basic(self, db_with_images):
        """Test getting scan points."""
        db_path, scan1_id, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_by_scan(scan1_id)
        points = service.get_scan_points_for_image(images[0])

        assert len(points) == 5
        assert all("x_pixel" in p for p in points)
        assert all("y_pixel" in p for p in points)
        assert all("point_index" in p for p in points)

    def test_no_scan(self, db_with_images):
        """Test with image that has no scan."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        # Create mock image without scan
        info = ImageInfo(
            id="test",
            scan_id=None,
            sol_number=None,
            sclk_start=None,
            file_path="/path",
            file_format="PNG",
            camera_id=None,
            image_type="ACI",
            product_id=None,
            width_px=None,
            height_px=None,
            image_time=None,
            focus_mode=None,
            local_time=None,
        )

        points = service.get_scan_points_for_image(info)

        assert len(points) == 0


class TestExportImages:
    """Tests for export_images method."""

    @pytest.fixture
    def service_with_real_images(self, db_with_images, tmp_path):
        """Create service with actual image files."""
        db_path, scan1_id, _ = db_with_images

        # Create a small test PNG
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        png_path = img_dir / "test.PNG"
        from PIL import Image
        img = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        img.save(png_path)

        # Update database to point to real file
        engine = get_engine(db_path)
        with get_session(engine) as session:
            # Get the PNG image and update its path
            images = session.execute(
                select(ContextImageORM).where(ContextImageORM.file_format == "PNG")
            ).scalars().all()

            for image in images:
                image.file_path = str(png_path)

            session.commit()

        service = ImageQueryService(database_path=db_path)
        return service, img_dir

    def test_export_png(self, service_with_real_images, tmp_path):
        """Test exporting to PNG format."""
        service, _ = service_with_real_images
        output_dir = tmp_path / "output"

        images = service.query_all(file_format="PNG")
        result = service.export_images(images, output_dir, format="png", show_progress=False)

        assert result.metadata["success"]
        assert result.metadata["images_exported"] >= 1
        assert (output_dir / "test.png").exists()

    def test_export_tiff(self, service_with_real_images, tmp_path):
        """Test exporting to TIFF format."""
        service, _ = service_with_real_images
        output_dir = tmp_path / "output"

        images = service.query_all(file_format="PNG")
        result = service.export_images(images, output_dir, format="tiff", show_progress=False)

        assert result.metadata["success"]
        assert (output_dir / "test.tiff").exists()

    def test_export_with_thumbnails(self, service_with_real_images, tmp_path):
        """Test exporting with thumbnails."""
        service, _ = service_with_real_images
        output_dir = tmp_path / "output"

        images = service.query_all(file_format="PNG")
        result = service.export_images(
            images,
            output_dir,
            format="png",
            create_thumbnails=True,
            show_progress=False,
        )

        assert result.metadata["success"]
        assert (output_dir / "thumbnails" / "test_thumb.png").exists()

    def test_export_skip_existing(self, service_with_real_images, tmp_path):
        """Test skipping existing files."""
        service, _ = service_with_real_images
        output_dir = tmp_path / "output"

        images = service.query_all(file_format="PNG")

        # Export first time
        service.export_images(images, output_dir, format="png", show_progress=False)

        # Export second time - should skip
        result = service.export_images(images, output_dir, format="png", show_progress=False)

        assert result.metadata["images_skipped"] >= 1

    def test_export_overwrite(self, service_with_real_images, tmp_path):
        """Test overwriting existing files."""
        service, _ = service_with_real_images
        output_dir = tmp_path / "output"

        images = service.query_all(file_format="PNG")

        # Export first time
        service.export_images(images, output_dir, format="png", show_progress=False)

        # Export with overwrite
        result = service.export_images(
            images,
            output_dir,
            format="png",
            overwrite=True,
            show_progress=False,
        )

        assert result.metadata["images_exported"] >= 1
        assert result.metadata["images_skipped"] == 0

    def test_export_missing_file(self, db_with_images, tmp_path):
        """Test handling missing source files."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)
        output_dir = tmp_path / "output"

        # Query images that point to nonexistent files
        images = service.query_all(file_format="IMG")
        result = service.export_images(images, output_dir, format="png", show_progress=False)

        assert result.metadata["images_failed"] > 0
        assert len(result.warnings) > 0

    def test_export_empty_list(self, db_with_images, tmp_path):
        """Test exporting empty list."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)
        output_dir = tmp_path / "output"

        result = service.export_images([], output_dir, format="png", show_progress=False)

        assert result.metadata["images_exported"] == 0


class TestLoadImage:
    """Tests for image loading methods."""

    @pytest.fixture
    def service_with_png(self, db_with_images, tmp_path):
        """Create service with actual PNG file."""
        db_path, _, _ = db_with_images

        # Create test PNG
        png_path = tmp_path / "test.PNG"
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        img.save(png_path)

        # Update database
        engine = get_engine(db_path)
        with get_session(engine) as session:
            images = session.execute(
                select(ContextImageORM).where(ContextImageORM.file_format == "PNG")
            ).scalars().all()

            image_id = None
            for image in images:
                image.file_path = str(png_path)
                image_id = image.id

            session.commit()

        service = ImageQueryService(database_path=db_path)
        return service, image_id

    def test_load_png(self, service_with_png):
        """Test loading PNG image."""
        service, image_id = service_with_png

        info = service.get_image_by_id(image_id)
        data, metadata = service.load_image(info)

        assert data.shape == (100, 100)
        assert metadata.product_id == "test"

    def test_load_by_id(self, service_with_png):
        """Test loading image by ID."""
        service, image_id = service_with_png

        data, metadata = service.load_image_by_id(image_id)

        assert data.shape == (100, 100)

    def test_load_nonexistent_id(self, db_with_images):
        """Test loading nonexistent image."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        with pytest.raises(ImageQueryError) as exc_info:
            service.load_image_by_id("nonexistent-id")

        assert "not found" in str(exc_info.value).lower()

    def test_load_missing_file(self, db_with_images):
        """Test loading when file is missing."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)

        images = service.query_all(limit=1)

        with pytest.raises(ImageQueryError) as exc_info:
            service.load_image(images[0])

        assert "not found" in str(exc_info.value).lower()


class TestExportBySol:
    """Tests for export_by_sol method."""

    def test_basic(self, db_with_images, tmp_path):
        """Test export by sol."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)
        output_dir = tmp_path / "output"

        # Won't actually export since files don't exist, but tests the flow
        result = service.export_by_sol(921, output_dir, show_progress=False)

        # Should try to export 3 images, but all fail (no files)
        assert result.metadata["images_failed"] == 3

    def test_no_images(self, db_with_images, tmp_path):
        """Test export for sol with no images."""
        db_path, _, _ = db_with_images
        service = ImageQueryService(database_path=db_path)
        output_dir = tmp_path / "output"

        result = service.export_by_sol(999, output_dir, show_progress=False)

        assert "No images found" in result.summary


# Skip integration tests if real data not available
@pytest.mark.skipif(
    not Path("./data/loupe").exists(),
    reason="Real ACI data not available"
)
class TestRealImageQuery:
    """Integration tests with real SHERLOC data."""

    def test_query_real_database(self):
        """Test querying the real database."""
        service = ImageQueryService()

        stats = service.get_statistics()
        assert stats["total"] > 0

    def test_query_available_sols(self):
        """Test getting available sols from real data."""
        service = ImageQueryService()

        sols = service.get_available_sols()
        assert len(sols) > 0

    def test_export_single_img(self, tmp_path):
        """Test exporting a single IMG file."""
        service = ImageQueryService()

        images = service.query_all(file_format="IMG", limit=1)

        if images:
            result = service.export_images(
                images,
                tmp_path / "output",
                format="png",
                show_progress=False,
            )

            assert result.metadata["images_exported"] == 1

    def test_load_real_img(self):
        """Test loading a real IMG file."""
        service = ImageQueryService()

        images = service.query_all(file_format="IMG", limit=1)

        if images:
            data, metadata = service.load_image(images[0])

            assert data.shape[0] > 0
            assert data.shape[1] > 0
