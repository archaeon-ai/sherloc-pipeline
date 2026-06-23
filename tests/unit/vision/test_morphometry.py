"""
Unit tests for grain morphometry analysis.

Tests cover:
- MorphometryStats computation
- Size class classification
- Grain-spectrum linkage
- Report generation
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import json

from sqlalchemy import text

from sherloc_pipeline.database import get_engine, get_session
from sherloc_pipeline.vision.morphometry import (
    GrainMorphometryAnalyzer,
    MorphometryStats,
    SizeClass,
    GrainSpectralLink,
    WENTWORTH_SIZE_CLASSES,
    DEFAULT_PIXEL_SCALE_UM,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with required schema."""
    db_path = tmp_path / "test_morphometry.db"

    engine = get_engine(db_path)
    with get_session(engine) as session:
        # Create minimal schema
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS sols (sol_number INTEGER PRIMARY KEY)
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS scans (
                id VARCHAR(36) PRIMARY KEY,
                sol_number INTEGER,
                target_name TEXT,
                scan_id TEXT,
                sclk_start INTEGER,
                n_points INTEGER,
                shots_per_point INTEGER,
                n_channels INTEGER DEFAULT 2148
            )
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS context_images (
                id VARCHAR(36) PRIMARY KEY,
                scan_id VARCHAR(36),
                image_type VARCHAR(10),
                file_path TEXT,
                file_format VARCHAR(10),
                pixel_scale_um FLOAT
            )
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS scan_points (
                id VARCHAR(36) PRIMARY KEY,
                scan_id VARCHAR(36),
                point_index INTEGER,
                x_pixel FLOAT,
                y_pixel FLOAT
            )
        """))

        session.execute(text("""
            CREATE TABLE IF NOT EXISTS grain_segments (
                id VARCHAR(36) PRIMARY KEY,
                image_id VARCHAR(36),
                segment_index INTEGER,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_width INTEGER,
                bbox_height INTEGER,
                mask_rle TEXT,
                area_px INTEGER,
                perimeter_px FLOAT,
                aspect_ratio FLOAT,
                circularity FLOAT,
                centroid_x FLOAT,
                centroid_y FLOAT,
                model_name VARCHAR(100),
                confidence FLOAT,
                stability_score FLOAT,
                linked_point_indices TEXT,
                created_at DATETIME
            )
        """))

        session.commit()

    return db_path


@pytest.fixture
def analyzer(temp_db):
    """Create analyzer with temp database."""
    return GrainMorphometryAnalyzer(database_path=temp_db)


class TestMorphometryStats:
    """Test MorphometryStats dataclass."""

    def test_from_array_empty(self):
        """Test stats with empty array."""
        stats = MorphometryStats.from_array(np.array([]))
        assert stats.count == 0
        assert stats.mean == 0
        assert stats.std == 0

    def test_from_array_single(self):
        """Test stats with single value."""
        stats = MorphometryStats.from_array(np.array([100.0]))
        assert stats.count == 1
        assert stats.mean == 100.0
        assert stats.std == 0.0

    def test_from_array_normal(self):
        """Test stats with normal distribution."""
        np.random.seed(42)
        values = np.random.normal(100, 10, 1000)
        stats = MorphometryStats.from_array(values)

        assert stats.count == 1000
        assert abs(stats.mean - 100) < 1  # Within 1 of expected
        assert abs(stats.std - 10) < 1
        assert stats.min_val < stats.p25 < stats.median < stats.p75 < stats.max_val

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = MorphometryStats(
            count=100,
            mean=50.5,
            median=48.0,
            std=10.2,
            min_val=20.0,
            max_val=80.0,
            p25=40.0,
            p75=60.0,
            p95=75.0,
        )

        d = stats.to_dict()
        assert d["count"] == 100
        assert d["mean"] == 50.5
        assert d["std"] == 10.2


class TestSizeClass:
    """Test SizeClass dataclass."""

    def test_wentworth_classes_defined(self):
        """Test that Wentworth classes are properly defined."""
        assert len(WENTWORTH_SIZE_CLASSES) == 8

        # Check they are in decreasing size order
        for i in range(len(WENTWORTH_SIZE_CLASSES) - 1):
            assert WENTWORTH_SIZE_CLASSES[i].min_diameter_um > WENTWORTH_SIZE_CLASSES[i + 1].min_diameter_um

    def test_to_dict(self):
        """Test dictionary conversion."""
        sc = SizeClass(
            name="Test Class",
            min_diameter_um=100,
            max_diameter_um=200,
            count=50,
            percentage=25.0,
        )

        d = sc.to_dict()
        assert d["name"] == "Test Class"
        assert d["count"] == 50
        assert d["percentage"] == 25.0


class TestGrainSpectralLink:
    """Test GrainSpectralLink dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        link = GrainSpectralLink(
            grain_id="grain-001",
            image_id="img-001",
            scan_id="scan-001",
            point_indices=[1, 2, 3, 4, 5],
            n_points=5,
            grain_area_px=1000,
            grain_centroid=(100.0, 200.0),
        )

        d = link.to_dict()
        assert d["grain_id"] == "grain-001"
        assert d["n_points"] == 5
        assert d["grain_centroid"] == [100.0, 200.0]


class TestGrainMorphometryAnalyzer:
    """Test GrainMorphometryAnalyzer class."""

    def test_init_with_default_db(self, temp_db):
        """Test initialization."""
        analyzer = GrainMorphometryAnalyzer(database_path=temp_db)
        assert analyzer.pixel_scale_um == DEFAULT_PIXEL_SCALE_UM

    def test_pixels_to_um2(self, analyzer):
        """Test area conversion."""
        # 100 pixels at 10.1 um/pixel = 100 * 10.1^2 = 10,201 um^2
        area_um2 = analyzer._pixels_to_um2(100)
        assert abs(area_um2 - 10201.0) < 0.01

    def test_equivalent_diameter(self, analyzer):
        """Test equivalent diameter calculation."""
        # Area = pi * r^2, so for area = 10201 um^2
        # diameter = 2 * sqrt(area/pi)
        diameter = analyzer._equivalent_diameter_um(100)  # 100 pixels
        expected = 2 * np.sqrt(10201 / np.pi)
        assert abs(diameter - expected) < 0.01

    def test_get_all_grains_empty(self, analyzer):
        """Test get_all_grains with no data."""
        grains = analyzer.get_all_grains()
        assert len(grains) == 0

    def test_compute_statistics_empty(self, analyzer):
        """Test compute_statistics with no data."""
        stats = analyzer.compute_statistics()
        assert stats["total_grains"] == 0
        assert "error" in stats

    def test_compute_statistics_with_data(self, temp_db):
        """Test compute_statistics with grain data."""
        engine = get_engine(temp_db)

        # Create test data
        with get_session(engine) as session:
            # Create image
            session.execute(text("""
                INSERT INTO context_images (id, scan_id, image_type, file_path, pixel_scale_um)
                VALUES ('img-001', 'scan-001', 'ACI', '/test.IMG', 10.1)
            """))

            # Create grains with varying properties
            for i in range(10):
                session.execute(text("""
                    INSERT INTO grain_segments (
                        id, image_id, segment_index, bbox_x, bbox_y, bbox_width, bbox_height,
                        area_px, perimeter_px, aspect_ratio, circularity,
                        centroid_x, centroid_y, model_name, created_at
                    ) VALUES (
                        :id, 'img-001', :idx, 0, 0, 100, 100,
                        :area, :perim, :ar, :circ,
                        50, 50, 'watershed', datetime('now')
                    )
                """), {
                    "id": f"grain-{i:03d}",
                    "idx": i,
                    "area": 100 + i * 100,  # 100 to 1000 pixels
                    "perim": 40 + i * 10,   # 40 to 130 pixels
                    "ar": 1.0 + i * 0.1,    # 1.0 to 1.9
                    "circ": 0.8 - i * 0.05, # 0.8 to 0.35
                })

            session.commit()

        analyzer = GrainMorphometryAnalyzer(database_path=temp_db)
        stats = analyzer.compute_statistics()

        assert stats["total_grains"] == 10
        assert stats["images_analyzed"] == 1
        assert "area_px" in stats
        assert "circularity" in stats
        assert "size_classes" in stats

    def test_compute_size_classes(self, analyzer):
        """Test size class computation."""
        # Create test diameters spanning multiple classes
        # Very fine sand: 62.5-125 um
        # Fine sand: 125-250 um
        # Medium sand: 250-500 um
        # Coarse sand: 500-1000 um
        # Very coarse sand: 1000-2000 um
        diameters = np.array([
            70,    # Very fine sand (62.5-125)
            100,   # Very fine sand (62.5-125)
            200,   # Fine sand (125-250)
            400,   # Medium sand (250-500)
            700,   # Coarse sand (500-1000)
            1500,  # Very coarse sand (1000-2000)
        ])

        classes = analyzer._compute_size_classes(diameters)

        # Find specific classes
        very_coarse = next(c for c in classes if c.name == "Very coarse sand")
        coarse = next(c for c in classes if c.name == "Coarse sand")
        medium = next(c for c in classes if c.name == "Medium sand")
        fine = next(c for c in classes if c.name == "Fine sand")
        very_fine = next(c for c in classes if c.name == "Very fine sand")

        assert very_coarse.count == 1
        assert coarse.count == 1
        assert medium.count == 1
        assert fine.count == 1
        assert very_fine.count == 2

    def test_grain_spectrum_linkage_empty(self, analyzer):
        """Test linkage with no data."""
        links = analyzer.compute_grain_spectrum_linkage()
        assert len(links) == 0

    def test_generate_report(self, temp_db, tmp_path):
        """Test report generation."""
        analyzer = GrainMorphometryAnalyzer(database_path=temp_db)

        # Generate report (will show "no grains" message)
        report_path = tmp_path / "test_report.md"
        report = analyzer.generate_report(report_path)

        assert "Grain Morphometry Analysis Report" in report
        assert report_path.exists()

    def test_export_json(self, temp_db, tmp_path):
        """Test JSON export."""
        analyzer = GrainMorphometryAnalyzer(database_path=temp_db)

        json_path = tmp_path / "test_stats.json"
        analyzer.export_statistics_json(json_path)

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "statistics" in data


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_analyze_morphometry(self, temp_db, tmp_path):
        """Test analyze_morphometry function."""
        from sherloc_pipeline.vision.morphometry import analyze_morphometry

        result = analyze_morphometry(
            database_path=temp_db,
            output_dir=tmp_path,
        )

        assert "total_grains" in result
        assert (tmp_path / "GRAIN_MORPHOMETRY.md").exists()
        assert (tmp_path / "morphometry_stats.json").exists()

    def test_compute_grain_spectrum_linkage_function(self, temp_db):
        """Test compute_grain_spectrum_linkage function."""
        from sherloc_pipeline.vision.morphometry import compute_grain_spectrum_linkage

        links = compute_grain_spectrum_linkage(database_path=temp_db)
        assert isinstance(links, list)
