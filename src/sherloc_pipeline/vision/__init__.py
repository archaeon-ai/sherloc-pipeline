"""
SHERLOC Vision Module - ACI Image Processing and Grain Analysis

This module provides tools for:
- Loading and parsing ACI (Autofocus Context Imager) images
- Grain segmentation using foundation models (SAM, etc.)
- Morphometric analysis of detected grains
- Linking grains to spectral measurement points

ACI Image Specifications:
- Resolution: 10.1 μm/pixel
- Format: VICAR/PDS3 (.IMG files)
- Total images: ~909 across 152 sols

Usage:
    from sherloc_pipeline.vision import read_aci_image, segment_grains

    # Load an ACI image
    image, metadata = read_aci_image(img_path)

    # Segment grains
    grains = segment_grains(image)

    # Analyze morphometry
    for grain in grains:
        grain.compute_morphometry()
        print(f"Grain {grain.segment_index}: area={grain.area}, circularity={grain.circularity:.2f}")
"""

from typing import TYPE_CHECKING

# Import types needed for __all__ exports
from .img_reader import ACIImageMetadata
from .segmentation import GrainMask, SegmentationConfig, SegmentationResult
from .morphometry import MorphometryStats, GrainSpectralLink

if TYPE_CHECKING:
    from .morphometry import GrainMorphometryAnalyzer

# Version
__version__ = "0.2.0"

# Lazy imports to avoid loading heavy dependencies unnecessarily
def read_aci_image(img_path, validate_dimensions=True):
    """Read a SHERLOC ACI image from VICAR format.

    See img_reader.read_aci_image for full documentation.
    """
    from .img_reader import read_aci_image as _read
    return _read(img_path, validate_dimensions)


def scan_img_files(directory, recursive=True):
    """Find all .IMG files in a directory.

    See img_reader.scan_img_files for full documentation.
    """
    from .img_reader import scan_img_files as _scan
    return _scan(directory, recursive)


def get_raw_vicar_label(img_path):
    """Extract the raw VICAR label as a dictionary.

    See img_reader.get_raw_vicar_label for full documentation.
    """
    from .img_reader import get_raw_vicar_label as _get
    return _get(img_path)


def segment_grains(image, config=None):
    """Segment grains in an ACI image using SAM.

    See segmentation.segment_grains for full documentation.

    Args:
        image: Grayscale or RGB numpy array
        config: Optional SegmentationConfig

    Returns:
        List of GrainMask objects
    """
    from .segmentation import segment_grains as _segment
    return _segment(image, config)


def GrainSegmenter(*args, **kwargs):
    """Create a grain segmenter instance.

    See segmentation.GrainSegmenter for full documentation.
    """
    from .segmentation import GrainSegmenter as _GrainSegmenter
    return _GrainSegmenter(*args, **kwargs)


def GrainMorphometryAnalyzer(*args, **kwargs):
    """Create a grain morphometry analyzer instance.

    See morphometry.GrainMorphometryAnalyzer for full documentation.
    """
    from .morphometry import GrainMorphometryAnalyzer as _Analyzer
    return _Analyzer(*args, **kwargs)


def analyze_morphometry(database_path=None, output_dir=None):
    """Run full morphometry analysis.

    See morphometry.analyze_morphometry for full documentation.
    """
    from .morphometry import analyze_morphometry as _analyze
    return _analyze(database_path, output_dir)


def compute_grain_spectrum_linkage(database_path=None, image_id=None):
    """Compute grain-spectrum linkage.

    See morphometry.compute_grain_spectrum_linkage for full documentation.
    """
    from .morphometry import compute_grain_spectrum_linkage as _compute
    return _compute(database_path, image_id)


__all__ = [
    "__version__",
    "read_aci_image",
    "scan_img_files",
    "get_raw_vicar_label",
    "segment_grains",
    "GrainSegmenter",
    "GrainMorphometryAnalyzer",
    "analyze_morphometry",
    "compute_grain_spectrum_linkage",
    "ACIImageMetadata",
    "GrainMask",
    "SegmentationConfig",
    "SegmentationResult",
    "MorphometryStats",
    "GrainSpectralLink",
]
