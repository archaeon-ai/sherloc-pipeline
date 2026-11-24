"""
SHERLOC Vision Module - ACI Image Reading

This module loads and parses ACI (Autofocus Context Imager) images:
VICAR/PDS3 (.IMG) files at 10.1 μm/pixel.

Usage:
    from sherloc_pipeline.vision import read_aci_image

    image, metadata = read_aci_image(img_path)
"""

# Import types needed for __all__ exports
from .img_reader import ACIImageMetadata

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


__all__ = [
    "__version__",
    "read_aci_image",
    "scan_img_files",
    "get_raw_vicar_label",
    "ACIImageMetadata",
]
