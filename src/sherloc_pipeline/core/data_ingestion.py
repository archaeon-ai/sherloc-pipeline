"""
Data ingestion module for SHERLOC pipeline.

This module handles ingestion and restructuring of Loupe dataset data,
including spectral data processing, image handling, and results management.

Portions of this implementation are adapted from Loupe V5.1.5a
(Apache License 2.0, © 2022 California Institute of Technology / JPL),
particularly the wavelength conversion and spectral restructuring logic.

Key Features:
- Loupe-compatible spectral processing with exact algorithm replication
- Flexible scan discovery supporting any naming convention (detail, line, survey, etc.)
- Automatic working directory discovery without hardcoded names
- Data type standardization (string to integer point labels)
- ACI image handling with color/grayscale preservation
- Standardized file naming and results organization

Data Structure:
- Input: ../data/loupe/sol_XXXX/[scan]/SrlcSpecSpecSohRaw_XXXXX-XXXXX-X_Loupe_working/
- Output: ../results/[target]/[sol]_[target]_[scan]_[spectral_region]_[processing_level].*

Spectral Processing:
Each of R1, R2, R3 is a separate full-CCD readout (2148 channels). Only a subset
of channels in each readout contains meaningful signal; the rest is detector noise
from unilluminated regions. Regional masks extract the relevant channels:
- R1 (Raman):        channels 52-574   (523 ch, 250-282 nm)  → Raman shift (cm⁻¹)
- R2 (Fluorescence): channels 690-1668 (979 ch, 282-337.8 nm) → Wavelength (nm)
- R3 (Fluorescence): channels 1690-2147 (458 ch, 337.8-357.4 nm) → Wavelength (nm)
- R123 (Stitched):   2148 ch combined from R1+R2+R3 via Loupe overlap summation

See docs/schema/SPECTRAL_REGIONS.md for the canonical region definitions.

Image Processing:
- ACI Images: 1648×1200 pixels, 10.1 μm/pixel scale
- Supports both grayscale and color PNG images
- 3x upscaling capability for zoomed overlays
- Preserves original pixel scale and format metadata

Usage:
    from sherloc_pipeline.core.data_ingestion import DataIngestion
    
    # Initialize
    ingester = DataIngestion(base_data_dir, results_dir)
    
    # Discover and process scan
    scans = ingester.discover_scans("1613")
    working_dir = scans["detail"][0]
    
    # Load and restructure data
    metadata = ingester.load_scan_metadata(working_dir)
    dark_sub_df = ingester.load_dark_subtracted_spectra(working_dir)
    raman_df = ingester.restructure_raman_data(dark_sub_df, metadata['n_spectra'])
    
    # Save results
    raman_path = ingester.save_spectral_data(raman_df, "Nordoya", "1613", "detail", "R1", "raw")
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image

from ..config import get_config
from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.utils import require_file, resolve_trim_proportion

logger = logging.getLogger(__name__)


def normalize_target_name(target: str) -> str:
    """Normalize target name for consistent filesystem paths.

    The DB stores target names with spaces (from .lpe files), but filesystem
    paths use underscores.  This function ensures consistent directory and
    file naming regardless of whether the caller passes spaces or underscores.
    """
    return target.replace(" ", "_")


class DataIngestion:
    """Handles ingestion and restructuring of Loupe dataset data."""
    
    def __init__(self, base_data_dir: Path, results_dir: Path = None, config_file: Path = None,
                 sol: str = None, target: str = None, scan: str = None):
        self.base_data_dir = Path(base_data_dir)
        
        # Store explicit parameters for single-scan processing
        self.sol = sol
        self.target = normalize_target_name(target) if target else target
        self.scan = scan
        
        # Load configuration
        if config_file:
            from ..config import load_config
            self.config = load_config(config_file)
        else:
            self.config = get_config()
        
        # Set results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            try:
                base_results = self.config.paths.get('results_dir') if hasattr(self.config, 'paths') else None
            except Exception:
                base_results = None
            if base_results:
                self.results_dir = Path(base_results)
            else:
                # Fallback to repo-standard ../results relative to working dir
                self.results_dir = Path("../results")
        
        # Get calibration parameters
        self.aci_pixel_scale = self.config.image.pixel_scale
        self.aci_dimensions = self.config.image.dimensions
        
    def discover_loupe_datasets(self) -> List[Path]:
        """Discover all available Loupe datasets (sol_XXXX folders)."""
        datasets = []
        for item in self.base_data_dir.iterdir():
            if item.is_dir() and item.name.startswith('sol_'):
                datasets.append(item)
        return sorted(datasets)
    
    def discover_scans(self, sol: str) -> Dict[str, List[Path]]:
        """Discover all available scans for a given sol.
        
        This method automatically discovers all scan types in a sol directory,
        supporting flexible naming conventions like detail, line, survey, detail_1, etc.
        It finds working directories ending with '_Loupe_working' and validates
        that they contain all required files.
        
        Args:
            sol: Sol number (e.g., '1613')
            
        Returns:
            Dict mapping scan names to working directory paths
            Example: {'detail': [Path(...)], 'line': [Path(...)], 'survey': [Path(...)]}
            
        Raises:
            FileNotFoundError: If sol directory doesn't exist
            
        Note:
            Scan names are not hardcoded - any directory name is supported as long
            as it contains valid Loupe working directories.
        """
        sol_dir = self.base_data_dir / f"sol_{sol}"
        require_file(sol_dir, "Sol directory not found")
        
        scans = {}
        for item in sol_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                scan_name = item.name
                working_dirs = self._discover_working_directories(item)
                if working_dirs:
                    scans[scan_name] = working_dirs
        
        return scans
    
    def find_working_directory(self, sol: str, scan: str) -> Optional[Path]:
        """Find the working directory for a specific sol and scan.

        Tries direct directory lookup first (sol_XXXX/scan/), then falls back
        to Loupe manifest-based resolution for scans whose data directory name
        differs from the scan name (e.g. ``HDR_100`` vs ``HDR_100pulses``).

        Args:
            sol: Sol number (e.g., '1613')
            scan: Scan type (e.g., 'detail', 'line')

        Returns:
            Path to working directory, or None if not found
        """
        sol_dir = self.base_data_dir / f"sol_{sol}"
        if not sol_dir.exists():
            return None

        scan_dir = sol_dir / scan
        if scan_dir.exists():
            working_dirs = self._discover_working_directories(scan_dir)
            if working_dirs:
                if len(working_dirs) > 1:
                    raise RuntimeError(
                        "Multiple working directories discovered for "
                        f"sol '{sol}' scan '{scan}'. Remove extras or specify the intended scan."
                    )
                return working_dirs[0]

        # Fallback: manifest-based resolution
        from sherloc_pipeline.core.manifest import resolve_manifest_working_directory
        try:
            manifest_dir = resolve_manifest_working_directory(
                base_data_dir=self.base_data_dir, sol=sol, scan=scan,
            )
            if manifest_dir is not None:
                return manifest_dir
        except Exception:
            logger.debug("Manifest resolution failed for sol %s scan %s", sol, scan, exc_info=True)

        return None
    
    def discover_all_scans(self) -> List[Tuple[str, str, str, Path]]:
        """Discover all available scans across all sols for batch processing.
        
        Returns:
            List of tuples (sol, target, scan, working_dir)
        """
        all_scans = []
        
        # Discover all datasets
        datasets = self.discover_loupe_datasets()
        
        for dataset in datasets:
            sol = dataset.name.replace('sol_', '')
            
            # Discover scans for this sol
            scans = self.discover_scans(sol)
            
            for scan_name, working_dirs in scans.items():
                for working_dir in working_dirs:
                    # Extract target from working directory or use default
                    target = self._extract_target_from_working_dir(working_dir)
                    all_scans.append((sol, target, scan_name, working_dir))
        
        return all_scans
    
    def _extract_target_from_working_dir(self, working_dir: Path) -> str:
        """Extract target name from working directory path.
        
        This is a fallback method for batch processing when target isn't explicitly known.
        """
        # Try to extract from path structure
        path_parts = working_dir.parts
        for part in path_parts:
            if part in ['Nordoya', 'Bellegarde', 'Rochette']:
                return part
        
        # Fallback to explicit target if provided, otherwise 'unknown'
        if self.target:
            return self.target
        return 'unknown'
    
    def process_raw_spectra(self, working_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process raw spectra for a single scan.
        
        Args:
            working_dir: Path to working directory
            
        Returns:
            Dict with processed spectra DataFrames
        """
        # Load metadata
        metadata = self.load_scan_metadata(working_dir)
        n_spectra = int(metadata['n_spectra'])
        
        # Load dark subtracted spectra
        dark_sub_df = self.load_dark_subtracted_spectra(working_dir)
        
        # Process all spectral regions
        spectra = {}
        
        # R1 (Raman)
        spectra['R1'] = self.restructure_raman_data(dark_sub_df, n_spectra)
        
        # Fluorescence (R2+R3)
        spectra['fluorescence'] = self.restructure_fluorescence_data(dark_sub_df, n_spectra)
        
        # R123 (stitched)
        spectra['R123'] = self.create_r123_spectrum(dark_sub_df, n_spectra)
        
        return spectra
    
    def process_normalized_spectra(self, working_dir: Path) -> Dict[str, pd.DataFrame]:
        """Process normalized spectra for a single scan.
        
        Args:
            working_dir: Path to working directory
            
        Returns:
            Dict with processed normalized spectra DataFrames
        """
        # Load normalized data
        normalized_file = working_dir / "darkSubSpectraN.csv"
        require_file(normalized_file, "Normalized data file not found")
        
        normalized_df = pd.read_csv(normalized_file, dtype=str, low_memory=False)
        
        # Convert channel columns to numeric
        for col in normalized_df.columns:
            if col.startswith(('R1_Channel', 'R2_Channel', 'R3_Channel')):
                normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
        
        # Load metadata
        metadata = self.load_scan_metadata(working_dir)
        n_spectra = int(metadata['n_spectra'])
        
        # Process all spectral regions
        spectra = {}
        
        # R1 (Raman)
        spectra['R1'] = self.restructure_raman_data(normalized_df, n_spectra)
        
        # Fluorescence (R2+R3)
        spectra['fluorescence'] = self.restructure_fluorescence_data(normalized_df, n_spectra)
        
        # R123 (stitched)
        spectra['R123'] = self.create_r123_spectrum(normalized_df, n_spectra)
        
        return spectra
    
    def generate_statistical_analysis(self, sol: str, target: str, scan: str,
                                       working_dir: Optional[Path] = None) -> Dict[str, int]:
        """Generate statistical analysis for all processed spectra.

        Args:
            sol: Sol number
            target: Target name
            scan: Scan type
            working_dir: Pre-resolved working directory. If None, discovers via
                find_working_directory (may fail for manifest-resolved scans).

        Returns:
            Dict with file counts by type
        """
        # Create results structure (per-scan folders)
        self.create_results_structure(target, sol=sol, scan=scan)

        # Use pre-resolved working directory if provided, otherwise discover
        if working_dir is None:
            working_dir = self.find_working_directory(sol, scan)
            if not working_dir:
                raise FileNotFoundError(f"Working directory not found for sol {sol}, scan {scan}")
        
        # Get processed spectra
        raw_spectra = self.process_raw_spectra(working_dir)
        normalized_spectra = self.process_normalized_spectra(working_dir)
        
        file_counts = {'CSV': 0, 'PNG': 0}
        
        # Process normalized spectra (laser-normalized becomes our downstream "raw-n")
        for region, spectra_df in normalized_spectra.items():
            region_label = 'fluor' if region == 'fluorescence' else region
            # Save and plot both mean and 2% trimmed mean
            files = self.save_both_average_spectra(
                spectra_df, target, sol, scan, region_label, "raw-n"
            )
            plots = self.plot_both_average_spectra(
                spectra_df, target, sol, scan, region_label, "raw-n"
            )
            file_counts['CSV'] += len(files)
            file_counts['PNG'] += len(plots)
        
        # If dynamic slit is enabled, estimate per-scan slit from the laser line (R1 only)
        try:
            import yaml
            cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
            fit_cfg = yaml.safe_load(open(cfg_path, 'r')).get('fitting', {})
            dyn = fit_cfg.get('dynamic_slit', {})
            if dyn and dyn.get('enabled', False) and 'R1' in normalized_spectra:
                est = self._estimate_scan_slit_from_laser(normalized_spectra['R1'])
                if est is not None:
                    logger.info(f"Estimated per-scan slit width from laser line: {est:.2f} cm^-1")
                else:
                    logger.info("Dynamic slit estimation skipped or insufficient points.")
        except Exception:
            logger.debug("Dynamic slit estimation failed (non-fatal).", exc_info=True)

        return file_counts

    def _estimate_scan_slit_from_laser(self, r1_df: pd.DataFrame) -> Optional[float]:
        """Estimate slit width from the laser line (600–700 cm⁻1) across points.

        Fits a single Gaussian per point within 600–700 cm⁻1, requiring SNR >= configured
        threshold. Returns the median FWHM across accepted fits.
        """
        try:
            import numpy as np
            from sherloc_pipeline.core.fitting import fit_spectrum
            import yaml
            cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
            fit_cfg = yaml.safe_load(open(cfg_path, 'r')).get('fitting', {})
            dyn = fit_cfg.get('dynamic_slit', {})
            if not dyn:
                return None
            roi = tuple(dyn.get('roi', [600.0, 700.0]))
            snr_min = float(dyn.get('min_snr', 10.0))
            fwhm_lo, fwhm_hi = map(float, dyn.get('fwhm_bounds', [20.0, 80.0]))
            min_points = int(dyn.get('min_points', 5))

            x = r1_df['raman_shift'].to_numpy(float)
            cfg = dict(fit_cfg)
            cfg['max_peaks'] = 1
            cfg['fit_fwhm_min_initial_cm1'] = fwhm_lo
            cfg['fwhm_max_cm1'] = fwhm_hi
            cfg['min_seed_snr'] = snr_min
            cfg['min_display_snr'] = snr_min
            accepted = []
            for col in r1_df.columns:
                if col == 'raman_shift':
                    continue
                y = r1_df[col].to_numpy(float)
                result, _ = fit_spectrum(x, y, cfg, roi=roi)
                for p in result.peaks:
                    if p.snr >= snr_min and fwhm_lo <= p.fwhm <= fwhm_hi:
                        accepted.append(p.fwhm)
            if len(accepted) < min_points:
                return None
            return float(np.median(np.array(accepted)))
        except Exception:
            return None
    
    def process_single_scan(self, working_dir: Path, sol: str, target: str, scan: str, 
                           generate_plots: bool = True) -> Dict[str, int]:
        """Process a single scan with explicit parameters (for batch processing).
        
        Args:
            working_dir: Path to working directory
            sol: Sol number
            target: Target name
            scan: Scan type
            generate_plots: Whether to generate plots
            
        Returns:
            Dict with file counts by type
        """
        # Temporarily set parameters for this scan
        original_sol, original_target, original_scan = self.sol, self.target, self.scan
        self.sol, self.target, self.scan = sol, target, scan
        
        try:
            # Process the scan
            results = self.generate_statistical_analysis(sol, target, scan)
            return results
        finally:
            # Restore original parameters
            self.sol, self.target, self.scan = original_sol, original_target, original_scan
    
    def _discover_working_directories(self, scan_dir: Path) -> List[Path]:
        """Discover working directories within a scan folder.
        
        Args:
            scan_dir: Path to scan directory (e.g., detail/, line/, etc.)
            
        Returns:
            List of working directory paths
        """
        working_dirs = []
        for item in scan_dir.iterdir():
            if item.is_dir() and item.name.endswith('_Loupe_working'):
                # Validate that this is a proper working directory
                if self._validate_working_directory(item):
                    working_dirs.append(item)
        
        return sorted(working_dirs)
    
    def _validate_working_directory(self, working_dir: Path) -> bool:
        """Validate that a directory is a proper Loupe working directory.
        
        Args:
            working_dir: Path to potential working directory
            
        Returns:
            True if directory contains required files
        """
        required_files = [
            'loupe.csv',
            'photodiodeRaw.csv', 
            'darkSubSpectra.csv',
            'spatial.csv'
        ]
        
        for file_name in required_files:
            if not (working_dir / file_name).exists():
                return False
        
        return True
    
    def discover_aci_image(self, working_dir: Path) -> Path:
        """Discover ACI image for a given working directory.
        
        Args:
            working_dir: Path to working directory
            
        Returns:
            Path to ACI image file
        """
        img_dir = working_dir / "img"
        require_file(img_dir, "Image directory not found")
        
        # Look for PNG files in the img directory
        png_files = list(img_dir.glob("*.PNG")) + list(img_dir.glob("*.png"))
        
        if not png_files:
            raise FileNotFoundError(f"No PNG images found in: {img_dir}")
        
        if len(png_files) > 1:
            # If multiple PNG files, prefer the one that looks like an ACI image
            # (typically has specific naming pattern)
            aci_files = [f for f in png_files if 'SC2' in f.name and 'ECM' in f.name]
            if aci_files:
                return aci_files[0]
            else:
                # Fall back to first PNG file
                return png_files[0]
        
        return png_files[0]
    
    def load_aci_image(self, working_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load ACI image from PNG file.
        
        Args:
            working_dir: Path to working directory
            
        Returns:
            Tuple of (image_array, metadata_dict)
            - image_array: Image data (grayscale or color)
            - metadata: Dict with 'pixel_scale', 'dimensions', 'is_color', etc.
        """
        image_path = self.discover_aci_image(working_dir)
        
        # Load image using PIL to preserve color information
        pil_image = Image.open(image_path)
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Determine if image is color or grayscale
        is_color = len(image_array.shape) == 3 and image_array.shape[2] == 3
        
        # Validate dimensions
        if is_color:
            height, width, channels = image_array.shape
        else:
            height, width = image_array.shape
        
        # Validate/record image dimensions. If actual dims differ from config, prefer runtime detection.
        expected_dims = set([tuple(self.aci_dimensions), tuple(reversed(self.aci_dimensions))])
        actual_dims = (height, width)
        if actual_dims not in expected_dims:
            try:
                logger.warning(
                    "ACI dimensions %sx%s differ from configured %s; proceeding with runtime-detected dimensions.",
                    height, width, self.aci_dimensions
                )
            except Exception:
                pass
        
        # Create metadata
        metadata = {
            'pixel_scale': self.aci_pixel_scale,  # μm/pixel
            # Record runtime-detected dimensions
            'dimensions': (width, height),
            'is_color': is_color,
            'file_path': str(image_path),
            'original_format': pil_image.mode
        }
        
        return image_array, metadata
    
    def upscale_image(self, image: np.ndarray, factor: int = 3, 
                     preserve_color: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Upscale image for zoomed overlays.
        
        Args:
            image: Original image array
            factor: Upscaling factor
            preserve_color: Whether to preserve color information
            
        Returns:
            Tuple of (upscaled_image, metadata_dict)
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image, mode='L')
        
        # Calculate new dimensions
        original_size = pil_image.size
        new_size = (original_size[0] * factor, original_size[1] * factor)
        
        # Upscale using LANCZOS resampling
        upscaled_pil = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        upscaled_array = np.array(upscaled_pil)
        
        # Create metadata
        metadata = {
            'upscale_factor': factor,
            'original_dimensions': original_size,
            'upscaled_dimensions': new_size,
            'pixel_scale': self.aci_pixel_scale / factor,  # Adjusted pixel scale
            'is_color': len(upscaled_array.shape) == 3,
            'preserve_color': preserve_color
        }
        
        return upscaled_array, metadata
    
    def get_results_path(self, target: str, sol: str = None, scan: str = None,
                        spectral_region: str = None, processing_level: str = None,
                        file_extension: str = None) -> Path:
        """Generate standardized results file paths.

        Target names are normalized (spaces → underscores) for consistent
        directory and file naming.

        Args:
            target: Target name (e.g., 'Nordoya')
            sol: Sol number (e.g., '1613')
            scan: Scan type (e.g., 'detail')
            spectral_region: Spectral region (e.g., 'R1', 'R2', 'R3', 'R123')
            processing_level: Processing level (e.g., 'raw', 'normalized', 'baselined')
            file_extension: File extension (e.g., '.png', '.csv')

        Returns:
            Path object for the results file
        """
        target = normalize_target_name(target)
        if sol and scan and spectral_region and processing_level and file_extension:
            # Direct export under scan subfolder: ../results/[target]/[sol]_[scan]/[sol]_[target]_[scan]_[spectral_region]_[processing_level].[ext]
            filename = f"{sol}_{target}_{scan}_{spectral_region}_{processing_level}{file_extension}"
            return self.results_dir / target / f"{sol}_{scan}" / filename
        elif sol and scan:
            # Scan subfolder base: ../results/[target]/[sol]_[scan]/
            return self.results_dir / target / f"{sol}_{scan}"
        else:
            # Base target folder: ../results/[target]/
            return self.results_dir / target
    
    def create_results_structure(self, target: str, sol: str = None, scan: str = None) -> None:
        """Create standard results folder structure for a target (and optional scan subfolder)."""
        target = normalize_target_name(target)
        target_dir = self.results_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        # If sol/scan provided, create per-scan subfolder and its standard subfolders
        if sol and scan:
            scan_dir = target_dir / f"{sol}_{scan}"
            scan_dir.mkdir(exist_ok=True)
            (scan_dir / "test_plots").mkdir(exist_ok=True)
            (scan_dir / "spectra_plots").mkdir(exist_ok=True)
        else:
            # Backward-compat fallback at target root
            (target_dir / "test_plots").mkdir(exist_ok=True)
            (target_dir / "spectra_plots").mkdir(exist_ok=True)
        # multifits folders will be created as needed
    
    def load_scan_metadata(self, working_dir: Path) -> Dict[str, Any]:
        """Load metadata from loupe.csv file.
        
        Args:
            working_dir: Path to working directory
            
        Returns:
            Dict with metadata from loupe.csv
        """
        loupe_file = working_dir / "loupe.csv"
        require_file(loupe_file, "Loupe file not found")
        
        # Read loupe.csv as key-value pairs
        df = pd.read_csv(loupe_file, header=None, names=['key', 'value'])
        metadata = dict(zip(df['key'], df['value']))
        
        # Convert numeric values
        numeric_keys = ['n_spectra', 'n_channels', 'laser_wavelength', 'shots_per_spec']
        for key in numeric_keys:
            if key in metadata:
                try:
                    metadata[key] = float(metadata[key])
                except (ValueError, TypeError):
                    pass
        
        return metadata
    
    def load_dark_subtracted_spectra(self, working_dir: Path) -> pd.DataFrame:
        """Load darkSubSpectra.csv data (3N+3 rows, 2148 channels)."""
        spectra_file = working_dir / "darkSubSpectra.csv"
        require_file(spectra_file, "DarkSub spectra file not found")
        
        # Read CSV with proper data types to avoid mixed type warnings
        spectra_df = pd.read_csv(spectra_file, dtype=str, low_memory=False)
        
        # Convert all data columns to numeric, handling any non-numeric values
        for col in spectra_df.columns:
            if col.startswith(('R1_Channel', 'R2_Channel', 'R3_Channel')):
                spectra_df[col] = pd.to_numeric(spectra_df[col], errors='coerce')
        
        return spectra_df
    
    def load_laser_normalized_spectra(self, working_dir: Path) -> pd.DataFrame:
        """Load darkSubSpectraN.csv data if available."""
        normalized_file = working_dir / "darkSubSpectraN.csv"
        require_file(normalized_file, "Laser normalized spectra file not found")
        
        # Read with dtype=str and low_memory=False to avoid DtypeWarning on mixed columns
        return pd.read_csv(normalized_file, dtype=str, low_memory=False)
    
    def load_spatial_data(self, working_dir: Path) -> pd.DataFrame:
        """Load spatial.csv data for spatial mapping."""
        spatial_file = working_dir / "spatial.csv"
        require_file(spatial_file, "Spatial file not found")
        
        return pd.read_csv(spatial_file)
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to ensure point labels are integers.
        
        Args:
            df: DataFrame with potentially string column names
            
        Returns:
            DataFrame with standardized column names
        """
        # Create a mapping for column name conversion
        new_columns = {}
        
        for col in df.columns:
            if col == 'raman_shift' or col == 'wavelength':
                # Keep spectral axis columns as-is
                new_columns[col] = col
            else:
                # Try to convert point labels to integers
                try:
                    point_num = int(col)
                    new_columns[col] = point_num
                except ValueError:
                    # If conversion fails, keep original name
                    new_columns[col] = col
        
        # Rename columns
        df_renamed = df.rename(columns=new_columns)
        
        # Sort columns: spectral axis first, then point numbers
        spectral_cols = [col for col in df_renamed.columns if col in ['raman_shift', 'wavelength']]
        point_cols = [col for col in df_renamed.columns if isinstance(col, int)]
        other_cols = [col for col in df_renamed.columns if col not in spectral_cols + point_cols]
        
        # Sort point columns numerically
        point_cols_sorted = sorted(point_cols)
        
        # Reorder columns
        df_renamed = df_renamed[spectral_cols + point_cols_sorted + other_cols]
        
        return df_renamed
    
    def _validate_point_labels(self, df: pd.DataFrame, expected_n_points: int) -> bool:
        """Validate that point labels are correct integers.
        
        Args:
            df: DataFrame with point columns
            expected_n_points: Expected number of points
            
        Returns:
            True if validation passes
        """
        # Find point columns (integer column names)
        point_cols = [col for col in df.columns if isinstance(col, int)]
        
        if len(point_cols) != expected_n_points:
            return False
        
        # Check that point numbers are sequential starting from 0
        expected_points = list(range(expected_n_points))
        if sorted(point_cols) != expected_points:
            return False
        
        return True
    
    def restructure_raman_data(self, dark_sub_df: pd.DataFrame,
                             n_spectra: int) -> pd.DataFrame:
        """Convert R1 region to Raman shift format using Loupe's exact algorithms.

        This method processes the R1 region (250-282 nm wavelength, channels 52-574
        after filtering) from darkSubSpectra data and converts it to a standard
        format with Raman shift values and individual point columns. It uses Loupe's
        polynomial coefficients for wavelength/wavenumber calculation and ensures
        point labels are integers.
        
        Args:
            dark_sub_df: Raw darkSubSpectra data (3N+3 rows, 2148 columns)
            n_spectra: Number of spectra in scan
            
        Returns:
            DataFrame with columns: [raman_shift, 0, 1, 2, ..., n-1]
            - raman_shift: Raman shift values in cm⁻¹ (sorted increasing)
            - 0, 1, 2, ..., n-1: Intensity values for each scan point
            
        Raises:
            ValueError: If point label validation fails
            
        Note:
            Uses Loupe's polynomial coefficients for wavelength calculation:
            - Channels 0-500: Raman polynomial
            - Channels 501+: Fluorescence polynomial (not used for R1)
        """
        # Extract R1 region (first n_spectra rows)
        R1_data = dark_sub_df.iloc[:n_spectra].copy()
        
        # Calculate Raman shift values
        n_channels = len(R1_data.columns)
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels)

        # Restrict to R1 wavelength bounds only to avoid exporting non‑R1 channels
        r1_min = self.config.spectral_regions.r1_wavelength_min
        r1_max = self.config.spectral_regions.r1_wavelength_max
        r1_mask = (wavelength >= r1_min) & (wavelength <= r1_max)
        
        # Materialize point data in a single allocation to avoid pandas fragmentation
        raman_shift = pd.DataFrame({'raman_shift': wavenumber[r1_mask]})
        point_matrix = R1_data.to_numpy(dtype=float)[:, r1_mask].T
        point_columns = pd.DataFrame(point_matrix, columns=list(range(n_spectra)))

        # Combine raman shift with point intensities before standardizing column names
        raman_df = pd.concat([raman_shift, point_columns], axis=1)
        raman_df = self._standardize_column_names(raman_df)
        
        # Validate point labels
        if not self._validate_point_labels(raman_df, n_spectra):
            raise ValueError(f"Point label validation failed for {n_spectra} spectra")
        
        return raman_df
    
    def restructure_fluorescence_data(self, dark_sub_df: pd.DataFrame,
                                    n_spectra: int) -> pd.DataFrame:
        """Convert R2/R3 regions to wavelength format.

        Args:
            dark_sub_df: Raw darkSubSpectra data
            n_spectra: Number of spectra in scan

        Returns:
            DataFrame with columns: [wavelength, 0, 1, 2, ..., n-1]
        """
        # Extract R2 and R3 regions. The Loupe CSV layout repeats a
        # `R{1,2,3}_Channel*` header row at every section boundary; pandas
        # consumes the first one as the column header, so within the
        # DataFrame the remaining section headers sit at iloc[n_spectra]
        # and iloc[2*n_spectra+1]. Skip those header rows; otherwise the
        # `to_numpy(dtype=float)` below trips on the literal strings.
        R2_data = dark_sub_df.iloc[n_spectra+1:2*n_spectra+1].copy()
        R3_data = dark_sub_df.iloc[2*n_spectra+2:3*n_spectra+2].copy()
        
        # Calculate wavelength values
        n_channels = len(R2_data.columns)
        wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels)
        
        # Vectorise column construction to reduce DataFrame fragmentation
        fluorescence_shift = pd.DataFrame({'wavelength': wavelength})
        combined_matrix = R2_data.to_numpy(dtype=float) + R3_data.to_numpy(dtype=float)
        point_columns = pd.DataFrame(combined_matrix.T, columns=list(range(n_spectra)))

        fluorescence_df = pd.concat([fluorescence_shift, point_columns], axis=1)
        fluorescence_df = self._standardize_column_names(fluorescence_df)
        
        # Validate point labels
        if not self._validate_point_labels(fluorescence_df, n_spectra):
            raise ValueError(f"Point label validation failed for {n_spectra} spectra")
        
        return fluorescence_df
    
    def create_r123_spectrum(self, dark_sub_df: pd.DataFrame, 
                           n_spectra: int) -> pd.DataFrame:
        """Create stitched R123 fluorescence spectrum.
        
        Args:
            dark_sub_df: Raw darkSubSpectra data
            n_spectra: Number of spectra in scan
            
        Returns:
            DataFrame with columns: [wavelength, 0, 1, 2, ..., n-1]
        """
        # Extract all three regions. Section header rows sit at
        # iloc[n_spectra] and iloc[2*n_spectra+1] in the parsed DataFrame —
        # see restructure_fluorescence_data for the layout note.
        R1_data = dark_sub_df.iloc[:n_spectra].copy()
        R2_data = dark_sub_df.iloc[n_spectra+1:2*n_spectra+1].copy()
        R3_data = dark_sub_df.iloc[2*n_spectra+2:3*n_spectra+2].copy()
        
        # Calculate wavelength values
        n_channels = len(R1_data.columns)
        wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels)
        
        # Get wavelength boundaries from configuration
        r1_min = self.config.spectral_regions.r1_wavelength_min
        r1_max = self.config.spectral_regions.r1_wavelength_max
        r2_min = self.config.spectral_regions.r2_wavelength_min
        r2_max = self.config.spectral_regions.r2_wavelength_max
        r3_min = self.config.spectral_regions.r3_wavelength_min
        r3_max = self.config.spectral_regions.r3_wavelength_max
        
        # Find channel indices for each wavelength region
        r1_mask = (wavelength >= r1_min) & (wavelength <= r1_max)
        r2_mask = (wavelength >= r2_min) & (wavelength <= r2_max)
        r3_mask = (wavelength >= r3_min) & (wavelength <= r3_max)
        
        # Build stitched spectrum using vectorised ndarray operations to avoid
        # incremental DataFrame column inserts (which fragment pandas frames)
        R1_vals = R1_data.to_numpy(dtype=float)
        R2_vals = R2_data.to_numpy(dtype=float)
        R3_vals = R3_data.to_numpy(dtype=float)
        r123_matrix = np.zeros_like(R1_vals)

        r123_matrix[:, r1_mask] = R1_vals[:, r1_mask]
        r123_matrix[:, r2_mask] = R2_vals[:, r2_mask]
        r123_matrix[:, r3_mask] = R3_vals[:, r3_mask]

        wavelength_df = pd.DataFrame({'wavelength': wavelength})
        point_columns = pd.DataFrame(r123_matrix.T, columns=list(range(n_spectra)))

        r123_df = pd.concat([wavelength_df, point_columns], axis=1)
        r123_df = self._standardize_column_names(r123_df)
        
        # Validate point labels
        if not self._validate_point_labels(r123_df, n_spectra):
            raise ValueError(f"Point label validation failed for {n_spectra} spectra")
        
        return r123_df
    
    def calculate_average_spectrum(self, spectra_df: pd.DataFrame, 
                                 method: str = 'mean') -> pd.DataFrame:
        """Calculate mean spectrum across all points.
        
        Args:
            spectra_df: DataFrame with point columns
            method: 'mean' or 'trimmed_mean' (2% trimmed)
            
        Returns:
            DataFrame with columns: [raman_shift/wavelength, intensity]
        """
        # Find spectral axis column
        spectral_col = None
        for col in ['raman_shift', 'wavelength']:
            if col in spectra_df.columns:
                spectral_col = col
                break
        
        if spectral_col is None:
            raise ValueError("No spectral axis column found (raman_shift or wavelength)")
        
        # Find point columns (integer column names)
        point_cols = [col for col in spectra_df.columns if isinstance(col, int)]
        
        if not point_cols:
            raise ValueError("No point columns found")
        
        # Calculate average
        if method == 'mean':
            avg_intensity = spectra_df[point_cols].mean(axis=1)
        elif method == 'trimmed_mean':
            # Trimmed mean: remove baseline_pct from each tail, with dynamic
            # floor ensuring >= 1 point trimmed for scans with < 51 points.
            from scipy import stats
            from sherloc_pipeline.core.utils import resolve_trim_proportion
            baseline_pct = self.config.preprocessing.get(
                'trim_mean_baseline_pct', 0.02
            )
            n_pts = len(point_cols)
            adjusted = resolve_trim_proportion(n_pts, baseline_pct) != baseline_pct
            if adjusted:
                logger.info(
                    "Trim mean: dynamic adjustment for %d points "
                    "(baseline %.1f%% → effective %.1f%% per tail)",
                    n_pts,
                    baseline_pct * 100,
                    resolve_trim_proportion(n_pts, baseline_pct) * 100,
                )
            def _trimmed_ignore_nan(row: pd.Series) -> float:
                values = row.to_numpy(dtype=float)
                values = values[~np.isnan(values)]
                if values.size == 0:
                    return np.nan
                ptc = resolve_trim_proportion(values.size, baseline_pct)
                return float(stats.trim_mean(values, ptc))
            avg_intensity = spectra_df[point_cols].apply(_trimmed_ignore_nan, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            spectral_col: spectra_df[spectral_col],
            'intensity': avg_intensity
        })
        
        return result_df
    
    def calculate_both_average_spectra(self, spectra_df: pd.DataFrame, 
                                     method: str = 'mean') -> Dict[str, pd.DataFrame]:
        """Calculate both mean and 2% trimmed mean spectra.
        
        Args:
            spectra_df: DataFrame with point columns
            method: Primary method ('mean' or 'trimmed_mean')
            
        Returns:
            Dict with 'mean' and 'trimmed_mean' DataFrames
        """
        mean_spectrum = self.calculate_average_spectrum(spectra_df, method='mean')
        trimmed_spectrum = self.calculate_average_spectrum(spectra_df, method='trimmed_mean')
        
        return {
            'mean': mean_spectrum,
            'trimmed_mean': trimmed_spectrum
        }
    
    def plot_average_spectrum(self, spectra_df: pd.DataFrame,
                            target: str, sol: str, scan: str,
                            spectral_region: str, processing_level: str,
                            method: str = 'mean') -> Path:
        """Plot and save mean spectrum.

        Args:
            spectra_df: DataFrame with point columns
            target: Target name
            sol: Sol number
            scan: Scan type
            spectral_region: Spectral region (R1, R2, R3, R123)
            processing_level: Processing level (raw, normalized, etc.)
            method: Averaging method ('mean' or 'trimmed_mean')

        Returns:
            Path to saved plot file
        """
        from sherloc_pipeline.visualization.ingestion_plots import plot_spectrum

        # Calculate mean spectrum
        avg_df = self.calculate_average_spectrum(spectra_df, method=method)

        # Determine spectral axis and labels
        if 'raman_shift' in avg_df.columns:
            x_data = avg_df['raman_shift']
            x_label = 'Raman Shift (cm⁻¹)'
            title_suffix = 'Raman'
        else:
            x_data = avg_df['wavelength']
            x_label = 'Wavelength (nm)'
            title_suffix = 'Fluorescence'

        y_data = avg_df['intensity']
        title = f'Mean {spectral_region} {title_suffix} Spectrum - Sol {sol} {target} {scan} - {processing_level.title()}'

        # Generate output path
        output_path = self.get_results_path(
            target=target, sol=sol, scan=scan,
            spectral_region=spectral_region,
            processing_level=f"{processing_level}_mean",
            file_extension=".png"
        )

        output_path = plot_spectrum(
            x_data=x_data,
            y_data=y_data,
            x_label=x_label,
            title=title,
            spectral_region=spectral_region,
            output_path=output_path,
        )

        logger.info(f"Mean spectrum plot saved to: {output_path}")

        return output_path
    
    def plot_both_average_spectra(self, spectra_df: pd.DataFrame,
                                target: str, sol: str, scan: str,
                                spectral_region: str, processing_level: str) -> Dict[str, Path]:
        """Plot and save both mean and trimmed mean spectra.

        Args:
            spectra_df: DataFrame with point columns
            target: Target name
            sol: Sol number
            scan: Scan type
            spectral_region: Spectral region (R1, R2, R3, R123)
            processing_level: Processing level (raw, normalized, etc.)

        Returns:
            Dict with plot paths for both methods
        """
        from sherloc_pipeline.core.utils import format_trim_label
        from sherloc_pipeline.visualization.ingestion_plots import plot_spectrum

        # Calculate both spectra
        spectra = self.calculate_both_average_spectra(spectra_df)

        # Determine spectral axis and labels
        if 'raman_shift' in spectra['mean'].columns:
            x_label = 'Raman Shift (cm⁻¹)'
            title_suffix = 'Raman'
        else:
            x_label = 'Wavelength (nm)'
            title_suffix = 'Fluorescence'

        # Compute effective trim label for filenames
        n_pts = len([c for c in spectra_df.columns if isinstance(c, int)])
        baseline_pct = self.config.preprocessing.get('trim_mean_baseline_pct', 0.02)
        trim_label = format_trim_label(n_pts, baseline_pct)
        effective_pct = round(resolve_trim_proportion(n_pts, baseline_pct) * 100, 1)

        # Create plots for both methods
        plot_paths = {}

        for method, spectrum_df in spectra.items():
            x_data = spectrum_df[spectrum_df.columns[0]]  # First column is spectral axis
            y_data = spectrum_df['intensity']

            if method == 'mean':
                method_label = 'Mean'
                method_token = 'mean'
            else:
                pct_display = int(effective_pct) if effective_pct == int(effective_pct) else effective_pct
                method_label = f'{pct_display}% Trimmed Mean'
                method_token = trim_label

            title = f'{method_label} {spectral_region} {title_suffix} Spectrum - Sol {sol} {target} {scan} - {processing_level.title()}'

            # Generate output path
            output_path = self.get_results_path(
                target=target, sol=sol, scan=scan,
                spectral_region=spectral_region,
                processing_level=f"{processing_level}_{method_token}",
                file_extension=".png"
            )

            output_path = plot_spectrum(
                x_data=x_data,
                y_data=y_data,
                x_label=x_label,
                title=title,
                spectral_region=spectral_region,
                output_path=output_path,
            )

            plot_paths[method] = output_path
            logger.info(f"{method_label} spectrum plot saved to: {output_path}")

        return plot_paths
    
    def save_both_average_spectra(self, spectra_df: pd.DataFrame,
                                target: str, sol: str, scan: str,
                                spectral_region: str, processing_level: str) -> Dict[str, Path]:
        """Save both mean and trimmed mean spectra to CSV files.
        
        Args:
            spectra_df: DataFrame with point columns
            target: Target name
            sol: Sol number
            scan: Scan type
            spectral_region: Spectral region (R1, R2, R3, R123)
            processing_level: Processing level (raw, normalized, etc.)
            
        Returns:
            Dict with file paths for both methods
        """
        # Calculate both spectra
        spectra = self.calculate_both_average_spectra(spectra_df)

        # Compute effective trim label for filenames
        from sherloc_pipeline.core.utils import format_trim_label
        n_pts = len([c for c in spectra_df.columns if isinstance(c, int)])
        baseline_pct = self.config.preprocessing.get('trim_mean_baseline_pct', 0.02)
        trim_label = format_trim_label(n_pts, baseline_pct)

        # Save both spectra
        file_paths = {}

        for method, spectrum_df in spectra.items():
            method_token = trim_label if method == 'trimmed_mean' else 'mean'
            # Generate output path
            output_path = self.get_results_path(
                target=target, sol=sol, scan=scan,
                spectral_region=spectral_region,
                processing_level=f"{processing_level}_{method_token}",
                file_extension=".csv"
            )
            
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            spectrum_df.to_csv(output_path, index=False)
            
            file_paths[method] = output_path
            logger.info(f"{method.title()} spectrum saved to: {output_path}")
        
        return file_paths
    
    def save_spectral_data(self, spectra_df: pd.DataFrame, 
                          target: str, sol: str, scan: str, 
                          spectral_region: str, processing_level: str) -> Path:
        """Save spectral data to CSV file.
        
        Args:
            spectra_df: DataFrame with spectral data
            target: Target name
            sol: Sol number
            scan: Scan type
            spectral_region: Spectral region (R1, R2, R3, R123)
            processing_level: Processing level (raw, normalized, etc.)
            
        Returns:
            Path to saved CSV file
        """
        # Generate output path
        output_path = self.get_results_path(
            target=target, sol=sol, scan=scan, 
            spectral_region=spectral_region, 
            processing_level=processing_level,
            file_extension=".csv"
        )
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        spectra_df.to_csv(output_path, index=False, float_format='%.3f')
        
        logger.info(f"Spectral data saved to: {output_path}")
        
        return output_path
    
    def validate_scan_data(self, working_dir: Path) -> bool:
        """Validate that scan data is complete and properly formatted."""
        try:
            # Check required files
            required_files = ['loupe.csv', 'darkSubSpectra.csv', 'spatial.csv']
            for file_name in required_files:
                if not (working_dir / file_name).exists():
                    logger.error(f"Required file missing: {file_name}")
                    return False
            
            # Load and validate metadata
            metadata = self.load_scan_metadata(working_dir)
            if 'n_spectra' not in metadata or 'n_channels' not in metadata:
                logger.error("Missing required metadata")
                return False
            
            # Load and validate spectral data
            spectra_df = self.load_dark_subtracted_spectra(working_dir)
            expected_rows = 3 * metadata['n_spectra'] + 3
            actual_rows = len(spectra_df)
            if actual_rows != expected_rows:
                # Allow for small discrepancies (e.g., missing 1 row)
                if abs(actual_rows - expected_rows) <= 1:
                    logger.warning(f"Spectral data row count slight mismatch: expected {expected_rows}, got {actual_rows}")
                else:
                    logger.error(f"Spectral data row count mismatch: expected {expected_rows}, got {actual_rows}")
                    return False
            
            logger.info("Scan data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Scan data validation failed: {e}")
            return False
