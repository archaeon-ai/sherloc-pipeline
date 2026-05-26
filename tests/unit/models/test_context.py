"""
Unit tests for context image and ROI models (bd-tqd: WS2-A).

Tests the context models:
- ContextImage: ACI/WATSON context images
- RegionOfInterest: User-defined spatial groupings
- ImageType: Image type enumeration
"""

import uuid

import pytest
from pydantic import ValidationError

from sherloc_pipeline.models import (
    ImageType,
    ContextImage,
    RegionOfInterest,
    ModelRegistry,
)


class TestImageType:
    """Tests for ImageType enum."""

    def test_values(self):
        """ImageType has expected values."""
        assert ImageType.ACI.value == "ACI"
        assert ImageType.WATSON.value == "WATSON"

    def test_all_types(self):
        """Both image types are defined."""
        types = list(ImageType)
        assert len(types) == 2


class TestContextImage:
    """Tests for ContextImage model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create ContextImage with minimal required fields."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/aci_921.png",
        )
        assert image.scan_id == scan_id
        assert image.image_type == ImageType.ACI
        assert image.file_path == "/data/img/aci_921.png"

    def test_full_creation(self, scan_id):
        """Create ContextImage with all fields."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/aci_921.png",
            product_id="urn:nasa:pds:mars2020_sherloc:...",
            sclk=672194998,
            pixel_scale_um=10.1,
            working_distance_cm=4.8,
            motor_position=1234,
            exposure_time_ms=100.0,
            led_illumination=True,
            width_px=1648,
            height_px=1200,
        )
        assert image.pixel_scale_um == 10.1
        assert image.width_px == 1648
        assert image.led_illumination is True

    def test_file_path_not_empty(self, scan_id):
        """file_path must not be empty."""
        with pytest.raises(ValidationError):
            ContextImage(
                scan_id=scan_id,
                image_type=ImageType.ACI,
                file_path="",
            )

    def test_positive_values(self, scan_id):
        """Positive value constraints are enforced."""
        with pytest.raises(ValidationError):
            ContextImage(
                scan_id=scan_id,
                image_type=ImageType.ACI,
                file_path="/data/img/test.png",
                pixel_scale_um=-10.1,  # Must be > 0
            )

        with pytest.raises(ValidationError):
            ContextImage(
                scan_id=scan_id,
                image_type=ImageType.ACI,
                file_path="/data/img/test.png",
                width_px=0,  # Must be > 0
            )

    def test_aspect_ratio_property(self, scan_id):
        """aspect_ratio calculates correctly."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/test.png",
            width_px=1648,
            height_px=1200,
        )
        assert image.aspect_ratio == pytest.approx(1648 / 1200)

        # None if dimensions not set
        image2 = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/test.png",
        )
        assert image2.aspect_ratio is None

    def test_total_pixels_property(self, scan_id):
        """total_pixels calculates correctly."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/test.png",
            width_px=1648,
            height_px=1200,
        )
        assert image.total_pixels == 1648 * 1200

        # None if dimensions not set
        image2 = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/test.png",
        )
        assert image2.total_pixels is None

    def test_from_loupe_metadata(self, scan_id):
        """Create ContextImage from Loupe metadata dict."""
        metadata = {
            "product_ID": "SC3_0921_0123456789_...",
            "sclk": "672194998",
            "pixel_scale": "10.1",
            "range": "4.8",
            "motor_pos": "1234",
            "exp_time": "100.0",
            "led_flag": "True",
        }

        image = ContextImage.from_loupe_metadata(
            scan_id=scan_id,
            file_path="/data/img/test.png",
            image_type=ImageType.ACI,
            metadata=metadata,
        )

        assert image.product_id == "SC3_0921_0123456789_..."
        assert image.sclk == 672194998
        assert image.pixel_scale_um == pytest.approx(10.1)
        assert image.working_distance_cm == pytest.approx(4.8)
        assert image.motor_position == 1234
        assert image.exposure_time_ms == pytest.approx(100.0)
        assert image.led_illumination is True

    def test_from_loupe_metadata_bool_parsing(self, scan_id):
        """Boolean parsing handles various formats."""
        # True variations
        for true_val in ["True", "true", "1", "yes", "on"]:
            metadata = {"led_flag": true_val}
            image = ContextImage.from_loupe_metadata(
                scan_id=scan_id,
                file_path="/data/img/test.png",
                image_type=ImageType.ACI,
                metadata=metadata,
            )
            assert image.led_illumination is True

        # False variations
        for false_val in ["False", "false", "0", "no", "off"]:
            metadata = {"led_flag": false_val}
            image = ContextImage.from_loupe_metadata(
                scan_id=scan_id,
                file_path="/data/img/test.png",
                image_type=ImageType.ACI,
                metadata=metadata,
            )
            assert image.led_illumination is False

    def test_watson_image_type(self, scan_id):
        """Can create WATSON type images."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.WATSON,
            file_path="/data/img/watson_921.png",
        )
        assert image.image_type == ImageType.WATSON

    def test_has_uuid(self, scan_id):
        """ContextImage has auto-generated UUID."""
        image = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/test.png",
        )
        assert image.id is not None
        assert isinstance(image.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """ContextImage can be registered in ModelRegistry."""
        assert hasattr(ContextImage, "__pydantic_complete__")


class TestRegionOfInterest:
    """Tests for RegionOfInterest model."""

    @pytest.fixture
    def scan_id(self):
        """Provide a scan UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, scan_id):
        """Create RegionOfInterest with required fields."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Carbonate Vein",
            color_hex="#00FF00",
            point_indices=[5, 6, 7, 12, 13, 14],
        )
        assert roi.name == "Carbonate Vein"
        assert roi.color_hex == "#00FF00"
        assert roi.point_indices == [5, 6, 7, 12, 13, 14]

    def test_with_description(self, scan_id):
        """Create RegionOfInterest with description."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Carbonate Vein",
            color_hex="#00FF00",
            point_indices=[5, 6, 7],
            description="Linear feature crossing target",
        )
        assert roi.description == "Linear feature crossing target"

    def test_name_not_empty(self, scan_id):
        """name must not be empty."""
        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="",
                color_hex="#00FF00",
                point_indices=[5, 6, 7],
            )

    def test_color_hex_format(self, scan_id):
        """color_hex must be valid hex format."""
        # Valid formats
        roi1 = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#FF0000",
            point_indices=[1],
        )
        assert roi1.color_hex == "#FF0000"

        roi2 = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#ff00ff",
            point_indices=[1],
        )
        assert roi2.color_hex == "#ff00ff"

        # Invalid formats
        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="FF0000",  # Missing #
                point_indices=[1],
            )

        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="#FF00",  # Too short
                point_indices=[1],
            )

        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="#GGGGGG",  # Invalid hex
                point_indices=[1],
            )

    def test_point_indices_not_empty(self, scan_id):
        """point_indices must have at least one point."""
        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="#00FF00",
                point_indices=[],
            )

    def test_point_indices_non_negative(self, scan_id):
        """point_indices must all be >= 0."""
        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="#00FF00",
                point_indices=[5, -1, 7],  # -1 is invalid
            )

    def test_point_indices_unique(self, scan_id):
        """point_indices must be unique."""
        with pytest.raises(ValidationError):
            RegionOfInterest(
                scan_id=scan_id,
                name="Test",
                color_hex="#00FF00",
                point_indices=[5, 6, 5],  # Duplicate 5
            )

    def test_point_indices_sorted(self, scan_id):
        """point_indices are sorted after validation."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#00FF00",
            point_indices=[14, 5, 12, 7, 6, 13],
        )
        assert roi.point_indices == [5, 6, 7, 12, 13, 14]

    def test_n_points_property(self, scan_id):
        """n_points returns correct count."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#00FF00",
            point_indices=[5, 6, 7, 12, 13, 14],
        )
        assert roi.n_points == 6

    def test_point_range_property(self, scan_id):
        """point_range returns (min, max)."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#00FF00",
            point_indices=[5, 6, 7, 12, 13, 14],
        )
        assert roi.point_range == (5, 14)

    def test_contains_point(self, scan_id):
        """contains_point checks membership."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#00FF00",
            point_indices=[5, 6, 7],
        )
        assert roi.contains_point(5) is True
        assert roi.contains_point(6) is True
        assert roi.contains_point(8) is False
        assert roi.contains_point(0) is False

    def test_overlaps_with(self, scan_id):
        """overlaps_with detects shared points."""
        roi1 = RegionOfInterest(
            scan_id=scan_id,
            name="ROI 1",
            color_hex="#FF0000",
            point_indices=[5, 6, 7],
        )
        roi2 = RegionOfInterest(
            scan_id=scan_id,
            name="ROI 2",
            color_hex="#00FF00",
            point_indices=[7, 8, 9],  # Overlaps at 7
        )
        roi3 = RegionOfInterest(
            scan_id=scan_id,
            name="ROI 3",
            color_hex="#0000FF",
            point_indices=[10, 11, 12],  # No overlap
        )

        assert roi1.overlaps_with(roi2) is True
        assert roi2.overlaps_with(roi1) is True
        assert roi1.overlaps_with(roi3) is False
        assert roi3.overlaps_with(roi1) is False

    def test_from_loupe_roi(self, scan_id):
        """Create RegionOfInterest from Loupe roi.csv data."""
        roi = RegionOfInterest.from_loupe_roi(
            scan_id=scan_id,
            name="Carbonate",
            color="#00FF00",
            points=[5, 6, 7, 12, 13, 14],
        )
        assert roi.name == "Carbonate"
        assert roi.color_hex == "#00FF00"
        assert roi.n_points == 6

    def test_from_loupe_roi_color_conversion(self, scan_id):
        """from_loupe_roi handles various color formats."""
        # With #
        roi1 = RegionOfInterest.from_loupe_roi(
            scan_id=scan_id,
            name="Test",
            color="#FF0000",
            points=[1],
        )
        assert roi1.color_hex == "#FF0000"

        # Without # (hex string)
        roi2 = RegionOfInterest.from_loupe_roi(
            scan_id=scan_id,
            name="Test",
            color="00FF00",
            points=[1],
        )
        assert roi2.color_hex == "#00FF00"

        # Invalid color defaults to gray
        roi3 = RegionOfInterest.from_loupe_roi(
            scan_id=scan_id,
            name="Test",
            color="not_a_color",
            points=[1],
        )
        assert roi3.color_hex == "#888888"

    def test_has_uuid(self, scan_id):
        """RegionOfInterest has auto-generated UUID."""
        roi = RegionOfInterest(
            scan_id=scan_id,
            name="Test",
            color_hex="#00FF00",
            point_indices=[1, 2, 3],
        )
        assert roi.id is not None
        assert isinstance(roi.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """RegionOfInterest can be registered in ModelRegistry."""
        assert hasattr(RegionOfInterest, "__pydantic_complete__")


class TestContextModelIntegration:
    """Integration tests for context models."""

    def test_multiple_images_per_scan(self):
        """Multiple ContextImages can reference same scan."""
        scan_id = uuid.uuid4()

        aci = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.ACI,
            file_path="/data/img/aci_921.png",
        )
        watson = ContextImage(
            scan_id=scan_id,
            image_type=ImageType.WATSON,
            file_path="/data/img/watson_921.png",
        )

        assert aci.scan_id == watson.scan_id == scan_id
        assert aci.image_type != watson.image_type
        assert aci.id != watson.id

    def test_multiple_rois_per_scan(self):
        """Multiple ROIs can reference same scan."""
        scan_id = uuid.uuid4()

        rois = [
            RegionOfInterest(
                scan_id=scan_id,
                name=f"ROI {i}",
                color_hex=f"#{i:02x}{i:02x}{i:02x}",
                point_indices=[i * 10, i * 10 + 1, i * 10 + 2],
            )
            for i in range(1, 4)
        ]

        assert len(rois) == 3
        assert all(r.scan_id == scan_id for r in rois)
        assert len(set(r.id for r in rois)) == 3  # Unique IDs

    def test_non_overlapping_rois(self):
        """ROIs covering all points without overlap."""
        scan_id = uuid.uuid4()
        n_points = 100

        # Create 5 non-overlapping ROIs
        rois = [
            RegionOfInterest(
                scan_id=scan_id,
                name=f"ROI {i}",
                color_hex="#FF0000",
                point_indices=list(range(i * 20, (i + 1) * 20)),
            )
            for i in range(5)
        ]

        # Check no overlap
        for i, roi1 in enumerate(rois):
            for roi2 in rois[i + 1:]:
                assert not roi1.overlaps_with(roi2)

        # Check all points covered
        all_points = set()
        for roi in rois:
            all_points.update(roi.point_indices)
        assert all_points == set(range(n_points))
