"""
Simple configuration management for SHERLOC pipeline.

This module provides a straightforward way to load and access configuration
parameters from YAML files.

Path resolution order (§3.2.1):
    1. Environment variable (e.g., SHERLOC_DB_PATH)
    2. XDG data home (~/.local/share/sherloc/<subdir>) if the path exists
    3. Config value from config.yaml (relative or absolute)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from sherloc_pipeline.core.utils import require_file


def resolve_path(config_value: str, env_var: str, xdg_subdir: Optional[str] = None) -> Path:
    """Resolve a path using: env var > XDG > config value.

    Resolution order (§3.2.1):
        1. Environment variable ``env_var`` if set.
        2. XDG data home path ``~/.local/share/sherloc/<xdg_subdir>`` if it exists.
        3. ``config_value`` as-is (may be relative or absolute).

    Args:
        config_value: Default path string from config.yaml.
        env_var: Name of the environment variable that can override this path.
        xdg_subdir: Optional subdirectory under ``$XDG_DATA_HOME/sherloc/``.
            If None, the XDG step is skipped.

    Returns:
        Resolved :class:`pathlib.Path`.
    """
    # 1. Environment variable wins
    if env_var in os.environ:
        return Path(os.environ[env_var])

    # 2. XDG data home (only if subdir provided and path already exists)
    if xdg_subdir is not None:
        xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        xdg_path = Path(xdg_data) / "sherloc" / xdg_subdir
        if xdg_path.exists():
            return xdg_path

    # 3. Config value fallback
    return Path(config_value)


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all path entries in a loaded config dict.

    Applies :func:`resolve_path` to the ``paths``, ``database``, and ``pds``
    sections.  All other config keys are left untouched.

    Args:
        config: Raw dict loaded from config.yaml.

    Returns:
        New dict with path values resolved.  The input dict is not mutated.
    """
    import copy
    resolved = copy.deepcopy(config)

    # Resolve paths section
    paths = resolved.get("paths", {})
    if "data_root" in paths:
        paths["data_root"] = str(resolve_path(
            paths["data_root"], "SHERLOC_DATA_DIR", "data"
        ))
    if "results_root" in paths:
        paths["results_root"] = str(resolve_path(
            paths["results_root"], "SHERLOC_RESULTS_DIR", "results"
        ))
    if "background_dir" in paths:
        paths["background_dir"] = str(resolve_path(
            paths["background_dir"], "SHERLOC_BACKGROUND_DIR"
        ))
    resolved["paths"] = paths

    # Re-anchor preprocessing.background_subtraction.default_file under the
    # resolved paths.background_dir so SHERLOC_BACKGROUND_DIR is a single,
    # canonical override surface. The default config keeps the old absolute
    # form for the basename only — the directory comes from background_dir.
    bg_dir = paths.get("background_dir")
    pre = resolved.get("preprocessing", {})
    bs = pre.get("background_subtraction", {}) if isinstance(pre, dict) else {}
    default_file = bs.get("default_file") if isinstance(bs, dict) else None
    if bg_dir and default_file:
        bs["default_file"] = str(Path(bg_dir) / Path(default_file).name)

    # Resolve database section (may not exist in older configs)
    database = resolved.get("database", {})
    if "path" in database:
        database["path"] = str(resolve_path(
            database["path"], "SHERLOC_DB_PATH", "phase.db"
        ))
    if "pds_path" in database:
        database["pds_path"] = str(resolve_path(
            database["pds_path"], "SHERLOC_PDS_DB_PATH", "phase_pds.db"
        ))
    resolved["database"] = database

    # Resolve pds section
    pds = resolved.get("pds", {})
    if "cache_dir" in pds:
        pds["cache_dir"] = str(resolve_path(
            pds["cache_dir"], "SHERLOC_PDS_CACHE_DIR"
        ))
    resolved["pds"] = pds

    return resolved


@dataclass
class WavelengthCalibration:
    """Wavelength calibration parameters."""
    raman_coefficients: List[float]
    fluorescence_coefficients: List[float]
    cutoff_channel: int
    laser_wavelength: float
    n_channels: int


@dataclass
class ImageCalibration:
    """Image calibration parameters."""
    pixel_scale: float
    dimensions: List[int]
    default_upscale_factor: int = 3


@dataclass
class SpectralRegions:
    """Spectral region definitions."""
    r1_wavelength_min: float
    r1_wavelength_max: float
    r2_wavelength_min: float
    r2_wavelength_max: float
    r3_wavelength_min: float
    r3_wavelength_max: float


@dataclass
class PDSConfig:
    """PDS archive download configuration."""
    base_url: str = "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_sherloc"
    cache_dir: str = "./pds"
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 2.0


@dataclass
class Config:
    """Main configuration container."""
    wavelength: WavelengthCalibration
    image: ImageCalibration
    spectral_regions: SpectralRegions
    preprocessing: Dict[str, Any]
    fitting: Dict[str, Any]
    fluorescence_fitting: Dict[str, Any]
    spatial: Dict[str, Any]
    output: Dict[str, Any]
    pds: PDSConfig
    paths: Dict[str, Any]
    logging: Dict[str, Any]
    database: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.database is None:
            self.database = {}


def load_config(config_file: Path = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file. If None, uses default.
        
    Returns:
        Config object with all parameters.
    """
    if config_file is None:
        config_file = Path(__file__).parent / "config.yaml"
    
    require_file(config_file, "Configuration file not found")

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    # Resolve environment/XDG path overrides (§3.2.1)
    data = resolve_paths(data)

    # Create calibration objects
    wavelength = WavelengthCalibration(**data['wavelength'])
    image = ImageCalibration(**data['image'])
    spectral_regions = SpectralRegions(**data['spectral_regions'])

    # Create PDS config (with defaults if section missing)
    pds_data = data.get('pds', {})
    pds = PDSConfig(**pds_data) if pds_data else PDSConfig()

    # Create main config
    return Config(
        wavelength=wavelength,
        image=image,
        spectral_regions=spectral_regions,
        preprocessing=data.get('preprocessing', {}),
        fitting=data.get('fitting', {}),
        fluorescence_fitting=data.get('fluorescence_fitting', {}),
        spatial=data.get('spatial', {}),
        output=data.get('output', {}),
        pds=pds,
        paths=data.get('paths', {}),
        logging=data.get('logging', {}),
        database=data.get('database', {}),
    )


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)."""
    global _config
    _config = None
