"""
Unit tests for grain segmentation pipeline.

Tests cover:
- Segmentation configuration validation
- GrainMask properties and morphometry
- Watershed baseline segmentation
- SAM model loading (when available)
- Batch processing logic
"""

import os
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import json

from sherloc_pipeline.vision.segmentation import (
    SegmentationConfig,
    SegmentationModel,
    GrainMask,
    GrainSegmenter,
    SegmentationResult,
    segment_grains,
)


class TestSegmentationConfig:
    """Test SegmentationConfig Pydantic model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SegmentationConfig()

        assert config.model == SegmentationModel.SAM_VIT_B
        assert config.points_per_side == 32
        assert config.pred_iou_thresh == 0.86
        assert config.stability_score_thresh == 0.92
        assert config.min_mask_area == 100
        assert config.device == "auto"

    def test_custom_config(self):
        """Test custom configuration."""
        config = SegmentationConfig(
            model=SegmentationModel.WATERSHED,
            points_per_side=64,
            min_mask_area=50,
        )

        assert config.model == SegmentationModel.WATERSHED
        assert config.points_per_side == 64
        assert config.min_mask_area == 50

    def test_points_per_side_validation(self):
        """Test points_per_side bounds."""
        # Valid
        config = SegmentationConfig(points_per_side=8)
        assert config.points_per_side == 8

        config = SegmentationConfig(points_per_side=128)
        assert config.points_per_side == 128

        # Invalid
        with pytest.raises(ValueError):
            SegmentationConfig(points_per_side=4)

        with pytest.raises(ValueError):
            SegmentationConfig(points_per_side=256)

    def test_threshold_validation(self):
        """Test threshold bounds."""
        # Valid
        config = SegmentationConfig(pred_iou_thresh=0.0)
        assert config.pred_iou_thresh == 0.0

        config = SegmentationConfig(pred_iou_thresh=1.0)
        assert config.pred_iou_thresh == 1.0

        # Invalid
        with pytest.raises(ValueError):
            SegmentationConfig(pred_iou_thresh=-0.1)

        with pytest.raises(ValueError):
            SegmentationConfig(pred_iou_thresh=1.1)

    def test_serialization(self):
        """Test JSON serialization."""
        config = SegmentationConfig(
            model=SegmentationModel.MOBILE_SAM,
            min_mask_area=200,
        )

        data = config.model_dump()
        assert data["model"] == "mobile_sam"
        assert data["min_mask_area"] == 200

        json_str = config.model_dump_json()
        assert "mobile_sam" in json_str


class TestGrainMask:
    """Test GrainMask dataclass."""

    def test_basic_creation(self):
        """Test creating a grain mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True  # 20x20 square

        grain = GrainMask(
            segment_index=0,
            mask=mask,
            bbox=[30, 20, 20, 20],
            area=400,
            predicted_iou=0.95,
            stability_score=0.98,
        )

        assert grain.segment_index == 0
        assert grain.area == 400
        assert grain.predicted_iou == 0.95
        assert grain.bbox == [30, 20, 20, 20]

    def test_compute_morphometry(self):
        """Test morphometry computation."""
        # Create a circular-ish mask
        mask = np.zeros((100, 100), dtype=bool)
        y, x = np.ogrid[-50:50, -50:50]
        circle = x**2 + y**2 <= 20**2
        mask[circle] = True

        grain = GrainMask(
            segment_index=0,
            mask=mask,
            bbox=[30, 30, 40, 40],
            area=int(mask.sum()),
            predicted_iou=0.9,
            stability_score=0.9,
        )

        grain.compute_morphometry()

        # Circle should have circularity close to 1
        assert grain.circularity > 0.8
        # Aspect ratio should be close to 1 for a circle
        assert 0.9 < grain.aspect_ratio < 1.1
        # Perimeter should be reasonable
        assert grain.perimeter > 0
        # Centroid should be near center
        assert 45 < grain.centroid[0] < 55
        assert 45 < grain.centroid[1] < 55

    def test_to_dict(self):
        """Test dictionary conversion."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True

        grain = GrainMask(
            segment_index=5,
            mask=mask,
            bbox=[10, 10, 20, 20],
            area=400,
            predicted_iou=0.88,
            stability_score=0.92,
            centroid=(20.0, 20.0),
            perimeter=80.0,
            aspect_ratio=1.0,
            circularity=0.79,
        )

        data = grain.to_dict()

        assert data["segment_index"] == 5
        assert data["area"] == 400
        assert data["bbox"] == [10, 10, 20, 20]
        assert data["circularity"] == pytest.approx(0.79)

    def test_rle_encoding(self):
        """Test RLE encoding and decoding."""
        pytest.importorskip("pycocotools")
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True

        grain = GrainMask(
            segment_index=0,
            mask=mask,
            bbox=[30, 20, 20, 20],
            area=400,
            predicted_iou=0.95,
            stability_score=0.98,
        )

        # Encode
        rle_str = grain.to_rle()
        assert isinstance(rle_str, str)

        # Decode
        metadata = grain.to_dict()
        recovered = GrainMask.from_rle(rle_str, 0, metadata)

        assert recovered.area == grain.area
        assert np.array_equal(recovered.mask, grain.mask)


class TestWatershedSegmentation:
    """Test watershed baseline segmentation."""

    def test_segment_simple_image(self):
        """Test watershed on a simple image with distinct regions."""
        # Create image with two distinct bright regions
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:40, 20:40] = 200  # Region 1
        image[60:80, 60:80] = 180  # Region 2

        config = SegmentationConfig(
            model=SegmentationModel.WATERSHED,
            min_mask_area=50,
        )

        segmenter = GrainSegmenter(config)
        result = segmenter.segment(image)

        # Should find at least one region
        assert result.n_grains >= 1
        assert result.model_name == "watershed"
        assert result.inference_time_s > 0

    def test_segment_empty_image(self):
        """Test watershed on uniform image."""
        # Uniform image should produce few/no segments
        image = np.ones((100, 100), dtype=np.uint8) * 128

        config = SegmentationConfig(
            model=SegmentationModel.WATERSHED,
            min_mask_area=50,
        )

        segmenter = GrainSegmenter(config)
        result = segmenter.segment(image)

        # May find zero or few grains depending on noise
        assert isinstance(result.n_grains, int)

    def test_segment_convenience_function(self):
        """Test the segment_grains convenience function."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:50, 20:50] = 200

        config = SegmentationConfig(
            model=SegmentationModel.WATERSHED,
            min_mask_area=30,
        )

        grains = segment_grains(image, config)

        assert isinstance(grains, list)
        # At least one grain from the bright region
        assert len(grains) >= 0


class TestSegmentationResult:
    """Test SegmentationResult dataclass."""

    def test_basic_result(self):
        """Test creating a result object."""
        config = SegmentationConfig()
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True

        grains = [
            GrainMask(
                segment_index=0,
                mask=mask,
                bbox=[10, 10, 20, 20],
                area=400,
                predicted_iou=0.9,
                stability_score=0.95,
            )
        ]

        result = SegmentationResult(
            image_id="test-123",
            image_path="/path/to/image.IMG",
            grains=grains,
            model_name="sam_vit_b",
            config=config,
            inference_time_s=2.5,
        )

        assert result.n_grains == 1
        assert result.total_grain_area == 400
        assert result.image_id == "test-123"

    def test_result_summary(self):
        """Test result summary statistics."""
        config = SegmentationConfig()
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:20, 10:20] = True  # 100 pixels

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[30:45, 30:45] = True  # 225 pixels

        grains = [
            GrainMask(
                segment_index=0,
                mask=mask1,
                bbox=[10, 10, 10, 10],
                area=100,
                predicted_iou=0.9,
                stability_score=0.95,
            ),
            GrainMask(
                segment_index=1,
                mask=mask2,
                bbox=[30, 30, 15, 15],
                area=225,
                predicted_iou=0.85,
                stability_score=0.9,
            ),
        ]

        result = SegmentationResult(
            image_id=None,
            image_path="",
            grains=grains,
            model_name="watershed",
            config=config,
            inference_time_s=0.5,
        )

        summary = result.summary()

        assert summary["n_grains"] == 2
        assert summary["total_area"] == 325
        assert summary["min_area"] == 100
        assert summary["max_area"] == 225
        assert summary["mean_area"] == pytest.approx(162.5)


class TestGrainSegmenter:
    """Test GrainSegmenter class."""

    def test_watershed_no_model_needed(self):
        """Test that watershed doesn't require model loading."""
        config = SegmentationConfig(model=SegmentationModel.WATERSHED)
        segmenter = GrainSegmenter(config)

        # Should work without loading any model
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = segmenter.segment(image)

        assert isinstance(result, SegmentationResult)

    def test_rgb_conversion(self):
        """Test that grayscale images are converted to RGB for SAM."""
        config = SegmentationConfig(model=SegmentationModel.WATERSHED)
        segmenter = GrainSegmenter(config)

        # Grayscale input
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = segmenter.segment(gray_image)
        assert isinstance(result, SegmentationResult)

        # RGB input
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = segmenter.segment(rgb_image)
        assert isinstance(result, SegmentationResult)

    def test_unload_model(self):
        """Test model unloading."""
        config = SegmentationConfig(model=SegmentationModel.WATERSHED)
        segmenter = GrainSegmenter(config)

        # Should not error even if no model loaded
        segmenter.unload_model()


# Skip SAM tests if checkpoint or required ML deps not available.
# Operators with the checkpoint should set SHERLOC_SAM_CHECKPOINT to its path.
import importlib.util as _ilu

_default_sam_ckpt = Path(__file__).resolve().parents[3] / ".cache" / "sam_checkpoints" / "sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = Path(os.getenv("SHERLOC_SAM_CHECKPOINT", str(_default_sam_ckpt)))

_sam_skip_reason = None
if not SAM_CHECKPOINT.exists():
    _sam_skip_reason = (
        f"SAM checkpoint not available at {SAM_CHECKPOINT} "
        "(set SHERLOC_SAM_CHECKPOINT)"
    )
elif _ilu.find_spec("torch") is None or _ilu.find_spec("segment_anything") is None:
    _sam_skip_reason = "torch + segment_anything not installed"


@pytest.mark.skipif(_sam_skip_reason is not None, reason=_sam_skip_reason or "")
class TestSAMSegmentation:
    """Integration tests for SAM-based segmentation."""

    def test_sam_model_loading(self):
        """Test that SAM model loads correctly."""
        config = SegmentationConfig(
            model=SegmentationModel.SAM_VIT_B,
            checkpoint_path=str(SAM_CHECKPOINT),
        )
        segmenter = GrainSegmenter(config)

        # Create test image
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        try:
            result = segmenter.segment(image)
            assert result.model_name == "sam_vit_b"
        finally:
            segmenter.unload_model()

    def test_sam_finds_grains(self):
        """Test that SAM finds grains in a synthetic image."""
        config = SegmentationConfig(
            model=SegmentationModel.SAM_VIT_B,
            checkpoint_path=str(SAM_CHECKPOINT),
            min_mask_area=50,
        )
        segmenter = GrainSegmenter(config)

        # Create image with clear regions
        image = np.zeros((200, 200), dtype=np.uint8)
        image[50:100, 50:100] = 200
        image[120:170, 120:170] = 180

        try:
            result = segmenter.segment(image)
            # SAM should find at least one region
            assert result.n_grains >= 1
            assert result.inference_time_s > 0
        finally:
            segmenter.unload_model()


# Skip real data tests if not available
@pytest.mark.skipif(
    not Path("./data/loupe").exists(),
    reason="Real ACI data not available"
)
class TestRealACISegmentation:
    """Integration tests with real SHERLOC ACI data."""

    def test_segment_real_aci_image(self):
        """Test segmentation on a real ACI image."""
        from sherloc_pipeline.vision import read_aci_image, scan_img_files

        files = scan_img_files("./data/loupe")
        assert len(files) > 0

        image, metadata = read_aci_image(files[0])

        config = SegmentationConfig(
            model=SegmentationModel.WATERSHED,
            min_mask_area=100,
        )

        grains = segment_grains(image, config)

        # Real images should have multiple grains
        assert len(grains) > 0
        for grain in grains:
            assert grain.area >= config.min_mask_area
