"""
SHERLOC Pipeline

Mars 2020 SHERLOC Raman/fluorescence data processing pipeline.
"""

__version__ = '5.4.0'
__author__ = "Ken Williford"
__email__ = "ken@bmsis.org"

# Import main components for easy access
from .core.laser_normalization import process_laser_normalization

__all__ = [
    "process_laser_normalization",
]
