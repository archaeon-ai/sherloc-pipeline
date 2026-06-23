"""
Tests for ImageIngestionService.

Tests cover:
- SCLK parsing and extraction
- Image ingestion
- Scan linkage
- Idempotency
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
import uuid

from sherloc_pipeline.services.image_ingestion import (
    ImageIngestionService,
    ImageIngestionError,
    ImageIngestionStats,
    parse_sclk_string,
    extract_sclk_from_filename,
    extract_sol_from_path,
    extract_camera_id_from_filename,
)
from sherloc_pipeline.database import (
    get_engine,
    get_session,
    create_all_tables,
    SolORM,
    ScanORM,
    ContextImageORM,
)


class TestSCLKParsing:
    """Tests for SCLK parsing functions."""

    def test_parse_sclk_decimal(self):
        """Test parsing SCLK with decimal format."""
        assert parse_sclk_string("697951240.076") == 697951240

    def test_parse_sclk_with_subseconds(self):
        """Test parsing SCLK with subseconds format."""
        assert parse_sclk_string("0697951240-30092") == 697951240

    def test_parse_sclk_integer(self):
        """Test parsing plain integer SCLK."""
        assert parse_sclk_string("697951240") == 697951240

    def test_parse_sclk_empty(self):
        """Test parsing empty string."""
        assert parse_sclk_string("") is None
        assert parse_sclk_string(None) is None

    def test_parse_sclk_invalid(self):
        """Test parsing invalid SCLK string."""
        assert parse_sclk_string("invalid") is None


class TestFilenameExtraction:
    """Tests for filename extraction functions."""

    def test_extract_sclk_from_filename(self):
        """Test extracting SCLK from ACI filename."""
        filename = "SC3_0349_0697951235_031ECM_N0092982SRLC11360_0000LMJ01.IMG"
        assert extract_sclk_from_filename(filename) == 697951235

    def test_extract_sclk_sc2(self):
        """Test extracting SCLK from SC2 camera filename."""
        filename = "SC2_1242_0777229084_847ECM_N0563438SRLC11470_0000LMJ01.IMG"
        assert extract_sclk_from_filename(filename) == 777229084

    def test_extract_sclk_no_match(self):
        """Test extraction from non-matching filename."""
        assert extract_sclk_from_filename("random_file.img") is None

    def test_extract_sol_from_path(self):
        """Test extracting sol number from path."""
        path = Path("./data/loupe/sol_0921/detail_1/image.IMG")
        assert extract_sol_from_path(path) == 921

    def test_extract_sol_colorized(self):
        """Test extracting sol from colorized folder name."""
        path = Path("/data/loupe/sol_1242_colorized/detail_2/image.IMG")
        assert extract_sol_from_path(path) == 1242

    def test_extract_sol_no_match(self):
        """Test extraction from path without sol."""
        path = Path("/data/images/random/file.IMG")
        assert extract_sol_from_path(path) is None

    def test_extract_camera_id_sc3(self):
        """Test extracting SC3 camera ID."""
        assert extract_camera_id_from_filename("SC3_0349_...") == "SC3"

    def test_extract_camera_id_sc2(self):
        """Test extracting SC2 camera ID."""
        assert extract_camera_id_from_filename("SC2_1242_...") == "SC2"

    def test_extract_camera_id_sc1(self):
        """Test extracting SC1 camera ID."""
        assert extract_camera_id_from_filename("SC1_0100_...") == "SC1"

    def test_extract_camera_id_none(self):
        """Test extraction from non-matching filename."""
        assert extract_camera_id_from_filename("other_file.img") is None


class TestImageIngestionStats:
    """Tests for ImageIngestionStats."""

    def test_add_stats(self):
        """Test combining two stats objects."""
        stats1 = ImageIngestionStats(images_ingested=5, images_linked=3)
        stats2 = ImageIngestionStats(images_ingested=3, images_linked=2)

        combined = stats1 + stats2

        assert combined.images_ingested == 8
        assert combined.images_linked == 5

    def test_add_stats_with_errors(self):
        """Test combining stats with errors."""
        stats1 = ImageIngestionStats(errors=["error1"])
        stats2 = ImageIngestionStats(errors=["error2"])

        combined = stats1 + stats2

        assert len(combined.errors) == 2


class TestImageIngestionServiceInit:
    """Tests for ImageIngestionService initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Test initialization with default parameters."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)

        assert service.sclk_tolerance_min == ImageIngestionService.DEFAULT_SCLK_TOLERANCE_MIN
        assert service.sclk_tolerance_max == ImageIngestionService.DEFAULT_SCLK_TOLERANCE_MAX

    def test_init_with_custom_tolerance(self, tmp_path):
        """Test initialization with custom tolerance."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(
            database_path=db_path,
            sclk_tolerance_min=10,
            sclk_tolerance_max=120,
        )

        assert service.sclk_tolerance_min == 10
        assert service.sclk_tolerance_max == 120


class TestImageIngestion:
    """Tests for image ingestion functionality."""

    def test_ingest_nonexistent_file(self, tmp_path):
        """Test error when file doesn't exist."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)

        with pytest.raises(ImageIngestionError) as exc_info:
            service.ingest_image(tmp_path / "nonexistent.IMG")

        assert "not found" in str(exc_info.value).lower()

    def test_ingest_nonexistent_directory(self, tmp_path):
        """Test error when directory doesn't exist."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)

        with pytest.raises(ImageIngestionError) as exc_info:
            service.ingest_all_images(tmp_path / "nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_ingest_empty_directory(self, tmp_path):
        """Test ingesting an empty directory."""
        db_path = tmp_path / "test.db"
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        service = ImageIngestionService(database_path=db_path)
        result = service.ingest_all_images(empty_dir, show_progress=False)

        assert "No IMG files" in result.summary
        assert result.metadata["images_scanned"] == 0

    def test_get_stats_empty(self, tmp_path):
        """Test getting stats on empty database."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)

        # Need to create tables first
        create_all_tables(service.engine)

        stats = service.get_ingestion_stats()

        assert stats["total"] == 0


class TestScanLinkage:
    """Tests for SCLK-based scan linkage."""

    @pytest.fixture
    def service_with_scans(self, tmp_path):
        """Create service with pre-populated scans."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)
        create_all_tables(service.engine)

        # Add a test sol and scan
        with get_session(service.engine) as session:
            sol = SolORM(
                sol_number=921,
                data_source="loupe",
                created_at=datetime.now(timezone.utc),
            )
            session.add(sol)
            session.flush()

            scan = ScanORM(
                id=str(uuid.uuid4()),
                sol_number=921,
                scan_name="test_target",
                scan_id="SrlcSpecSpecSohRaw_0748731411-51550-1",
                sclk_start=748731411,  # Image SCLK should be ~14s before this
                n_points=100,
                n_channels=2148,
                shots_per_point=10,
                laser_wavelength_nm=248.6,
                created_at=datetime.now(timezone.utc),
            )
            session.add(scan)
            session.commit()

        return service

    def test_find_matching_scan(self, service_with_scans):
        """Test finding a matching scan by SCLK."""
        with get_session(service_with_scans.engine) as session:
            # Image SCLK ~14 seconds before scan
            image_sclk = 748731397
            scan_id = service_with_scans._find_matching_scan(session, image_sclk, 921)

            assert scan_id is not None

    def test_find_no_match_wrong_sol(self, service_with_scans):
        """Test no match when sol is wrong."""
        with get_session(service_with_scans.engine) as session:
            image_sclk = 748731397
            scan_id = service_with_scans._find_matching_scan(session, image_sclk, 999)

            assert scan_id is None

    def test_find_no_match_sclk_too_early(self, service_with_scans):
        """Test no match when SCLK is too early."""
        with get_session(service_with_scans.engine) as session:
            # Image SCLK 100 seconds before scan (outside tolerance)
            image_sclk = 748731311
            scan_id = service_with_scans._find_matching_scan(session, image_sclk, 921)

            assert scan_id is None

    def test_find_no_match_sclk_after(self, service_with_scans):
        """Test no match when image SCLK is after scan SCLK."""
        with get_session(service_with_scans.engine) as session:
            # Image SCLK after scan (invalid)
            image_sclk = 748731500
            scan_id = service_with_scans._find_matching_scan(session, image_sclk, 921)

            assert scan_id is None


# Skip integration tests if real data not available
@pytest.mark.skipif(
    not Path("./data/loupe").exists(),
    reason="Real ACI data not available"
)
class TestRealImageIngestion:
    """Integration tests with real SHERLOC data."""

    def test_ingest_single_real_image(self, tmp_path):
        """Test ingesting a real IMG file."""
        from sherloc_pipeline.vision.img_reader import scan_img_files

        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)
        create_all_tables(service.engine)

        # Find first available image
        files = scan_img_files("./data/loupe")
        assert len(files) > 0, "No IMG files found"

        # Create necessary sol/scan for linkage
        with get_session(service.engine) as session:
            # Extract sol from path
            sol_number = extract_sol_from_path(files[0])
            if sol_number:
                sol = SolORM(
                    sol_number=sol_number,
                    data_source="loupe",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(sol)
                session.commit()

        # This will likely not link (no matching scan) but should not error
        result = service.ingest_image(files[0])

        # Should succeed (either ingested or skipped due to no scan)
        assert result.metadata["success"]

    def test_batch_ingest_limit(self, tmp_path):
        """Test batch ingestion with limit."""
        db_path = tmp_path / "test.db"
        service = ImageIngestionService(database_path=db_path)
        create_all_tables(service.engine)

        result = service.ingest_all_images(
            Path("./data/loupe"),
            limit=5,
            show_progress=False,
        )

        assert result.metadata["images_scanned"] == 5
