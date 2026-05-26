"""Core processing modules for SHERLOC pipeline."""

__all__ = [
    "calibration",
    "laser_normalization",
    "r1_extraction",
    "r123_stitching",
    "preprocessing",
    "baseline",
    "fitting",
    "mineral_id",
    "spatial",
    "pds_parsers",
    "pds_client",
    "data_ingestion",
    "accepted_assembler",
    "manifest",
    "utils",
    "voronoi",
    "coordinates",
]

from .laser_normalization import process_laser_normalization
from .r123_stitching import (
    stitch_r123_spectrum,
    stitch_r123_batch,
    r123_wavelength_axis,
    r123_wavenumber_axis,
)
