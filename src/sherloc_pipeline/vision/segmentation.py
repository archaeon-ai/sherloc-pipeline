"""
Grain Segmentation Pipeline for SHERLOC ACI Images.

This module provides automated grain segmentation using Meta's Segment Anything
Model (SAM) with fallback options for resource-constrained environments.

Key Features:
- SAM ViT-B primary segmentation (~54 masks/image, 2.2s inference)
- MobileSAM fallback for lightweight processing
- Traditional watershed baseline
- Batch processing with progress tracking and checkpointing
- Database integration for storing segmentation results

ACI Image Specifications:
- Resolution: 10.1 um/pixel
- Typical grain count: 30-100 per image
- Processing time: ~33 min for 909 images on RTX 3090 Ti

Usage:
    from sherloc_pipeline.vision.segmentation import (
        GrainSegmenter,
        SegmentationConfig,
        segment_grains,
    )

    # Quick segmentation
    segmenter = GrainSegmenter()
    grains = segmenter.segment(image)

    # With custom config
    config = SegmentationConfig(
        min_mask_area=100,
        points_per_side=32,
    )
    grains = segmenter.segment(image, config)

References:
    - SAM paper: Kirillov et al. (2023). "Segment Anything." ICCV 2023.
    - MobileSAM: Zhang et al. (2023). arXiv:2306.14289
    - SAM evaluation: docs/research/SAM_EVALUATION.md
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import json
import logging
import time
import uuid

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from sherloc_pipeline.models.base import PHASEBaseModel


logger = logging.getLogger(__name__)


# Default paths
DEFAULT_SAM_CHECKPOINT = Path.home() / ".cache/sherloc-pipeline/sam_checkpoints/sam_vit_b_01ec64.pth"
DEFAULT_MOBILE_SAM_CHECKPOINT = Path.home() / ".cache/sherloc-pipeline/sam_checkpoints/mobile_sam.pt"


class SegmentationModel(str, Enum):
    """Available segmentation models."""
    SAM_VIT_B = "sam_vit_b"
    SAM_VIT_L = "sam_vit_l"
    SAM_VIT_H = "sam_vit_h"
    MOBILE_SAM = "mobile_sam"
    WATERSHED = "watershed"


class SegmentationConfig(PHASEBaseModel):
    """Configuration for grain segmentation.

    Attributes:
        model: Segmentation model to use
        points_per_side: Grid density for SAM auto-mask generation
        pred_iou_thresh: Mask quality threshold (0.0-1.0)
        stability_score_thresh: Mask stability threshold (0.0-1.0)
        min_mask_area: Minimum grain size in pixels (~10 um^2 at ACI resolution)
        crop_n_layers: Number of multi-scale crop layers
        device: Compute device ('cuda', 'cpu', or 'auto')
        checkpoint_path: Path to model checkpoint (auto-detected if None)
    """

    model: SegmentationModel = Field(
        default=SegmentationModel.SAM_VIT_B,
        description="Segmentation model to use"
    )
    points_per_side: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Grid density for automatic mask generation"
    )
    pred_iou_thresh: float = Field(
        default=0.86,
        ge=0.0,
        le=1.0,
        description="Mask quality threshold"
    )
    stability_score_thresh: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Mask stability threshold"
    )
    min_mask_area: int = Field(
        default=100,
        ge=1,
        description="Minimum grain size in pixels"
    )
    crop_n_layers: int = Field(
        default=1,
        ge=0,
        le=4,
        description="Number of multi-scale crop layers"
    )
    crop_n_points_downscale_factor: int = Field(
        default=2,
        ge=1,
        description="Downscale factor for points in crops"
    )
    device: str = Field(
        default="auto",
        description="Compute device: 'cuda', 'cpu', or 'auto'"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to model checkpoint"
    )


@dataclass
class GrainMask:
    """Represents a single segmented grain.

    Attributes:
        segment_index: Index within the image (0-based)
        mask: Binary mask array (height x width)
        bbox: Bounding box [x, y, width, height]
        area: Area in pixels
        predicted_iou: Model's predicted quality (0-1)
        stability_score: Mask stability (0-1)
        centroid: Center point (x, y)
        perimeter: Perimeter in pixels (computed)
        aspect_ratio: Major/minor axis ratio (computed)
        circularity: 4*pi*area/perimeter^2 (computed)
    """

    segment_index: int
    mask: NDArray[np.bool_]
    bbox: List[int]  # [x, y, w, h]
    area: int
    predicted_iou: float
    stability_score: float
    centroid: Tuple[float, float] = field(default=(0.0, 0.0))
    perimeter: float = 0.0
    aspect_ratio: float = 1.0
    circularity: float = 0.0

    def compute_morphometry(self) -> None:
        """Compute morphometric properties from mask."""
        try:
            from skimage import measure
        except ImportError:
            logger.warning("scikit-image not available for morphometry")
            return

        if self.area < 4:
            return

        # Find contours
        contours = measure.find_contours(self.mask.astype(np.float64), 0.5)
        if not contours:
            return

        # Use the longest contour
        contour = max(contours, key=len)

        # Perimeter
        self.perimeter = float(measure.perimeter(self.mask))

        # Centroid
        props = measure.regionprops(self.mask.astype(np.int32))
        if props:
            self.centroid = (float(props[0].centroid[1]), float(props[0].centroid[0]))

            # Aspect ratio from fitted ellipse (use new API names with fallback)
            if hasattr(props[0], 'axis_major_length'):
                major = props[0].axis_major_length
                minor = props[0].axis_minor_length
            else:
                # Fallback for older scikit-image versions
                major = props[0].major_axis_length
                minor = props[0].minor_axis_length
            if major and minor > 0:
                self.aspect_ratio = major / minor

        # Circularity
        if self.perimeter > 0:
            self.circularity = 4 * np.pi * self.area / (self.perimeter ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            "segment_index": self.segment_index,
            "bbox": self.bbox,
            "area": self.area,
            "predicted_iou": float(self.predicted_iou),
            "stability_score": float(self.stability_score),
            "centroid": list(self.centroid),
            "perimeter": float(self.perimeter),
            "aspect_ratio": float(self.aspect_ratio),
            "circularity": float(self.circularity),
        }

    def to_rle(self) -> str:
        """Encode mask as run-length encoding (RLE)."""
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(self.mask.astype(np.uint8)))
        return json.dumps({
            "counts": rle["counts"].decode("utf-8"),
            "size": list(rle["size"]),
        })

    @classmethod
    def from_rle(cls, rle_str: str, segment_index: int, metadata: Dict[str, Any]) -> "GrainMask":
        """Create GrainMask from RLE-encoded string."""
        from pycocotools import mask as mask_utils
        rle_dict = json.loads(rle_str)
        rle = {
            "counts": rle_dict["counts"].encode("utf-8"),
            "size": rle_dict["size"],
        }
        mask = mask_utils.decode(rle).astype(bool)

        return cls(
            segment_index=segment_index,
            mask=mask,
            bbox=metadata.get("bbox", [0, 0, 0, 0]),
            area=metadata.get("area", int(mask.sum())),
            predicted_iou=metadata.get("predicted_iou", 0.0),
            stability_score=metadata.get("stability_score", 0.0),
            centroid=tuple(metadata.get("centroid", [0.0, 0.0])),
            perimeter=metadata.get("perimeter", 0.0),
            aspect_ratio=metadata.get("aspect_ratio", 1.0),
            circularity=metadata.get("circularity", 0.0),
        )


@dataclass
class SegmentationResult:
    """Result of grain segmentation on an image.

    Attributes:
        image_id: Database ID of the source image
        image_path: Path to the source image file
        grains: List of detected grain masks
        model_name: Name of the model used
        config: Configuration used for segmentation
        inference_time_s: Time taken for inference
        timestamp: When segmentation was performed
    """

    image_id: Optional[str]
    image_path: str
    grains: List[GrainMask]
    model_name: str
    config: SegmentationConfig
    inference_time_s: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def n_grains(self) -> int:
        """Number of detected grains."""
        return len(self.grains)

    @property
    def total_grain_area(self) -> int:
        """Total area covered by grains in pixels."""
        return sum(g.area for g in self.grains)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        areas = [g.area for g in self.grains]
        return {
            "n_grains": self.n_grains,
            "total_area": self.total_grain_area,
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
            "mean_area": np.mean(areas) if areas else 0,
            "inference_time_s": self.inference_time_s,
            "model": self.model_name,
        }


def _normalize_model(model: Union[SegmentationModel, str]) -> str:
    """Normalize model to string value for comparison."""
    if hasattr(model, 'value'):
        return model.value
    return str(model)


class GrainSegmenter:
    """Main class for grain segmentation.

    This class handles model loading, inference, and result processing.
    Supports SAM ViT-B (primary), MobileSAM (lightweight), and watershed (baseline).

    Example:
        >>> segmenter = GrainSegmenter()
        >>> result = segmenter.segment(image)
        >>> print(f"Found {result.n_grains} grains")
    """

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
    ):
        """Initialize the grain segmenter.

        Args:
            config: Segmentation configuration. If None, uses defaults.
        """
        self.config = config or SegmentationConfig()
        self._model = None
        self._mask_generator = None
        self._device = None

    def _get_device(self) -> str:
        """Determine the compute device to use."""
        import torch

        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _load_sam_model(self) -> Any:
        """Load SAM model and create mask generator."""
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        self._device = self._get_device()
        logger.info(f"Loading SAM model on {self._device}")

        # Determine model type and checkpoint
        model_str = _normalize_model(self.config.model)
        if model_str == SegmentationModel.SAM_VIT_B.value:
            model_type = "vit_b"
            checkpoint = self.config.checkpoint_path or str(DEFAULT_SAM_CHECKPOINT)
        elif model_str == SegmentationModel.SAM_VIT_L.value:
            model_type = "vit_l"
            checkpoint = self.config.checkpoint_path
            if not checkpoint:
                raise ValueError("SAM ViT-L requires explicit checkpoint path")
        elif model_str == SegmentationModel.SAM_VIT_H.value:
            model_type = "vit_h"
            checkpoint = self.config.checkpoint_path
            if not checkpoint:
                raise ValueError("SAM ViT-H requires explicit checkpoint path")
        else:
            raise ValueError(f"Unsupported SAM model: {self.config.model}")

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

        # Load model
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self._device)
        sam.eval()

        # Create mask generator with configured parameters
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=self.config.min_mask_area,
            crop_n_layers=self.config.crop_n_layers,
            crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
        )

        self._model = sam
        logger.info(f"SAM {model_type} loaded successfully")

        return self._mask_generator

    def _load_mobile_sam_model(self) -> Any:
        """Load MobileSAM model and create mask generator."""
        import torch

        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "MobileSAM not installed. Install with: "
                "pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )

        self._device = self._get_device()
        logger.info(f"Loading MobileSAM on {self._device}")

        checkpoint = self.config.checkpoint_path or str(DEFAULT_MOBILE_SAM_CHECKPOINT)

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {checkpoint}")

        # Load model
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.to(self._device)
        sam.eval()

        # Create mask generator
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=self.config.min_mask_area,
        )

        self._model = sam
        logger.info("MobileSAM loaded successfully")

        return self._mask_generator

    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._mask_generator is not None:
            return

        model_str = _normalize_model(self.config.model)
        sam_models = {
            SegmentationModel.SAM_VIT_B.value,
            SegmentationModel.SAM_VIT_L.value,
            SegmentationModel.SAM_VIT_H.value,
        }

        if model_str in sam_models:
            self._load_sam_model()
        elif model_str == SegmentationModel.MOBILE_SAM.value:
            self._load_mobile_sam_model()
        elif model_str == SegmentationModel.WATERSHED.value:
            pass  # No model needed
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

    def segment(
        self,
        image: NDArray[np.uint8],
        image_id: Optional[str] = None,
        image_path: Optional[str] = None,
        compute_morphometry: bool = True,
    ) -> SegmentationResult:
        """Segment grains in an image.

        Args:
            image: Grayscale or RGB image as numpy array
            image_id: Optional database ID for the image
            image_path: Optional path to the source image
            compute_morphometry: Whether to compute morphometric properties

        Returns:
            SegmentationResult with detected grains
        """
        start_time = time.time()

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image

        model_str = _normalize_model(self.config.model)
        if model_str == SegmentationModel.WATERSHED.value:
            grains = self._segment_watershed(image)
        else:
            grains = self._segment_sam(image_rgb)

        # Compute morphometry if requested
        if compute_morphometry:
            for grain in grains:
                grain.compute_morphometry()

        inference_time = time.time() - start_time

        return SegmentationResult(
            image_id=image_id,
            image_path=image_path or "",
            grains=grains,
            model_name=model_str,
            config=self.config,
            inference_time_s=inference_time,
        )

    def _segment_sam(self, image_rgb: NDArray[np.uint8]) -> List[GrainMask]:
        """Segment using SAM or MobileSAM."""
        self._ensure_model_loaded()

        # Generate masks
        masks = self._mask_generator.generate(image_rgb)

        # Convert to GrainMask objects
        grains = []
        for i, mask_data in enumerate(masks):
            grain = GrainMask(
                segment_index=i,
                mask=mask_data["segmentation"],
                bbox=list(mask_data["bbox"]),  # [x, y, w, h]
                area=mask_data["area"],
                predicted_iou=mask_data["predicted_iou"],
                stability_score=mask_data["stability_score"],
            )
            grains.append(grain)

        return grains

    def _segment_watershed(self, image: NDArray[np.uint8]) -> List[GrainMask]:
        """Segment using traditional watershed algorithm (baseline).

        This provides a fast baseline but typically over-segments significantly.
        """
        from skimage import filters, segmentation, measure
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi

        # Ensure grayscale
        if len(image.shape) == 3:
            image = image[:, :, 0]

        # Preprocess
        smoothed = filters.gaussian(image, sigma=1.5)
        thresh = filters.threshold_otsu(smoothed)
        binary = smoothed > (thresh * 0.8)

        # Distance transform + markers
        distance = ndi.distance_transform_edt(binary)
        coords = peak_local_max(
            distance,
            min_distance=20,
            labels=binary.astype(int),
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)

        # Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)

        # Convert regions to GrainMask objects
        grains = []
        props = measure.regionprops(labels)

        for i, prop in enumerate(props):
            if prop.area < self.config.min_mask_area:
                continue

            # Create binary mask for this region
            region_mask = labels == prop.label

            # Bounding box in [x, y, w, h] format
            minr, minc, maxr, maxc = prop.bbox
            bbox = [minc, minr, maxc - minc, maxr - minr]

            grain = GrainMask(
                segment_index=i,
                mask=region_mask,
                bbox=bbox,
                area=prop.area,
                predicted_iou=0.0,  # Not applicable for watershed
                stability_score=0.0,
                centroid=(prop.centroid[1], prop.centroid[0]),
            )
            grains.append(grain)

        return grains

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            import torch

            del self._model
            del self._mask_generator
            self._model = None
            self._mask_generator = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded")


def segment_grains(
    image: NDArray[np.uint8],
    config: Optional[SegmentationConfig] = None,
) -> List[GrainMask]:
    """Convenience function to segment grains in an image.

    Args:
        image: Grayscale or RGB image as numpy array
        config: Optional segmentation configuration

    Returns:
        List of detected grain masks
    """
    segmenter = GrainSegmenter(config)
    result = segmenter.segment(image)
    return result.grains
