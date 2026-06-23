"""Laser normalization: photodiode-based spectral intensity correction."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from sherloc_pipeline.core.utils import require_file

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)

logger = logging.getLogger(__name__)


def read_photodiode_data(photodiode_file: Path) -> pd.DataFrame:
    """
    Read photodiode data from CSV file.

    Args:
        photodiode_file: Path to photodiodeRaw.csv file

    Returns:
        DataFrame with photodiode data (spectra × shots)
    """
    df = pd.read_csv(photodiode_file)
    logger.info(f"Loaded photodiode data: {df.shape[0]} spectra, {df.shape[1]} shots")
    return df


def calculate_photodiode_summary(photodiode_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate photodiode summary by taking mean across shots for each spectrum.

    This matches Loupe's logic: photodiodeSummary = photodiodeAll.mean(axis=1)

    Args:
        photodiode_df: DataFrame with photodiode data (spectra × shots)

    Returns:
        Array of mean photodiode values for each spectrum
    """
    summary = photodiode_df.mean(axis=1).values
    logger.info(f"Calculated photodiode summary: {len(summary)} values, range {summary.min():.2f}-{summary.max():.2f}")
    return summary


def calculate_normalization_factors(photodiode_summary: np.ndarray) -> np.ndarray:
    """
    Calculate laser normalization factors using Loupe's method.

    Formula: norm_factors = max(pd) / pd_i for each spectrum
    This matches Loupe's logic: [max(_pd)/_pd_i for _pd_i in _pd]

    Args:
        photodiode_summary: Array of mean photodiode values

    Returns:
        Array of normalization factors
    """
    max_pd = np.max(photodiode_summary)
    norm_factors = max_pd / photodiode_summary
    logger.info(f"Calculated normalization factors: range {norm_factors.min():.3f}-{norm_factors.max():.3f}")
    return norm_factors


def process_laser_normalization(
    working_dir: Path,
    sol: str,
    target: str,
    scan: str,
    processing_flag: str = "N",
    generate_plots: bool = True,
    results_dir: Path = None
) -> None:
    """
    Process laser normalization for darkSubSpectra data only (matching Loupe behavior).

    This function performs the laser normalization workflow exactly like Loupe:
    1. Read photodiode data and calculate normalization factors
    2. Read darkSubSpectra data and split into R1, R2, R3 regions
    3. Apply normalization to each region separately
    4. Write normalized data back to file
    5. Generate comparison plots (optional)

    Args:
        working_dir: Path to the working directory containing the data files
        sol: Sol number for plot filenames (required)
        target: Target name for plot filenames (required)
        scan: Scan type for plot filenames (required)
        processing_flag: Processing flag to append to output filenames (default: "N")
        generate_plots: Whether to generate comparison plots (default: True)
        results_dir: Base results directory (default: None, will use config)
    """
    logger.info(f"Starting laser normalization for {working_dir}")

    # Read photodiode data and calculate normalization factors
    photodiode_file = working_dir / "photodiodeRaw.csv"
    require_file(photodiode_file, "Photodiode file not found")

    logger.info(f"Using photodiode data: {photodiode_file}")
    photodiode_df = read_photodiode_data(photodiode_file)
    photodiode_summary = calculate_photodiode_summary(photodiode_df)
    norm_factors = calculate_normalization_factors(photodiode_summary)

    # Read n_spectra from loupe.csv
    loupe_file = working_dir / "loupe.csv"
    require_file(loupe_file, "Loupe file not found")

    # Read loupe.csv as key-value pairs
    loupe_df = pd.read_csv(loupe_file, header=None, names=['key', 'value'])
    n_spectra = int(loupe_df[loupe_df['key'] == 'n_spectra']['value'].iloc[0])
    logger.info(f"n_spectra from loupe.csv: {n_spectra}")

    # Read darkSubSpectra data and split into regions (like Loupe)
    dark_sub_file = working_dir / "darkSubSpectra.csv"
    require_file(dark_sub_file, "DarkSub spectra file not found")

    logger.info("Loading darkSubSpectra data...")
    # Read the full file first (string dtype avoids pandas DtypeWarning on mixed rows)
    full_dark_sub = pd.read_csv(dark_sub_file, dtype=str, low_memory=False)
    logger.info(f"Dark-subtracted spectra: {len(full_dark_sub)} rows, {len(full_dark_sub.columns)} columns")

    # Split into R1, R2, R3 regions (matching Loupe's logic)
    # R1: skiprows=0, nrows=nSpectra (rows 1 to N)
    # R2: skiprows=1+nSpectra, nrows=nSpectra (rows N+1 to 2N)
    # R3: skiprows=2*(1+nSpectra), nrows=nSpectra (rows 2N+1 to 3N)

    R1 = full_dark_sub.iloc[0:n_spectra].copy()
    R2 = full_dark_sub.iloc[n_spectra:2*n_spectra].copy()
    R3 = full_dark_sub.iloc[2*n_spectra:3*n_spectra].copy()

    logger.info(f"Split into regions: R1={len(R1)} rows, R2={len(R2)} rows, R3={len(R3)} rows")

    # Apply normalization to each region (like Loupe)
    logger.info("Applying laser normalization to darkSubSpectra regions...")

    # Convert data to numeric before normalization (handle string values)
    R1_numeric = R1.apply(pd.to_numeric, errors='coerce')
    R2_numeric = R2.apply(pd.to_numeric, errors='coerce')
    R3_numeric = R3.apply(pd.to_numeric, errors='coerce')

    R1_norm = R1_numeric.multiply(norm_factors, axis='rows')
    R2_norm = R2_numeric.multiply(norm_factors, axis='rows')
    R3_norm = R3_numeric.multiply(norm_factors, axis='rows')

    # Write normalized data back (concatenate R1+R2+R3 like Loupe's writeSpectraRegions)
    output_file = working_dir / f"darkSubSpectra{processing_flag}.csv"
    logger.info(f"Writing normalized darkSubSpectra to {output_file}")

    # Write in the same format as Loupe's writeSpectraRegions
    R1_norm.to_csv(output_file, header=True, index=False, float_format='%.3f')
    R2_norm.to_csv(output_file, mode='a', header=True, index=False, float_format='%.3f')
    R3_norm.to_csv(output_file, mode='a', header=True, index=False, float_format='%.3f')

    # Update processing flag in loupe.csv
    update_loupe_csv_processing_flag(working_dir, processing_flag)

    # Generate comparison plots by default (can be disabled by caller via --no-plots)
    if generate_plots:
        logger.info("Generating comparison plots...")

        # Get n_channels for wavelength calculation
        n_channels = int(loupe_df[loupe_df['key'] == 'n_channels']['value'].iloc[0])

        # Calculate wavelength and wavenumber arrays
        wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels)

        # Create results directory for plots - use per-scan subfolder structure
        if results_dir is None:
            # Get the proper results directory from the config
            from ..config import get_config
            config = get_config()
            base_data_dir = working_dir.parent.parent.parent.parent.parent  # Go up to sherloc directory
            results_dir = base_data_dir / config.output.get('results_dir', 'results')

        # Create the test_plots subdirectory under [sol]_[scan]
        test_plots_dir = results_dir / target / f"{sol}_{scan}" / "test_plots"
        test_plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots for each region
        regions = ['R1', 'R2', 'R3']
        for region in regions:
            try:
                from sherloc_pipeline.visualization.normalization_plots import create_region_comparison_plot
                create_region_comparison_plot(
                    R1_numeric if region == 'R1' else R2_numeric if region == 'R2' else R3_numeric,
                    R1_norm if region == 'R1' else R2_norm if region == 'R2' else R3_norm,
                    wavelength, wavenumber, region, test_plots_dir, sol, target, scan
                )
            except Exception as e:
                logger.error(f"Failed to create {region} comparison plot: {e}")

        logger.info(f"Comparison plots saved to: {test_plots_dir}")

    logger.info("Laser normalization completed successfully")




def update_loupe_csv_processing_flag(working_dir: Path, processing_flag: str) -> None:
    """
    Update the processing flag in loupe.csv to indicate laser normalization was applied.

    Args:
        working_dir: Path to the working directory
        processing_flag: Processing flag to add
    """
    loupe_file = working_dir / "loupe.csv"
    if not loupe_file.exists():
        logger.warning(f"Loupe file not found: {loupe_file}")
        return

    # Read current loupe.csv as key-value pairs
    loupe_df = pd.read_csv(loupe_file, header=None, names=['key', 'value'])

    # Update processing flag (add 'N' if not already present)
    current_flag = str(loupe_df[loupe_df['key'] == 'specProcessingApplied']['value'].iloc[0])
    if processing_flag not in current_flag:
        new_flag = current_flag + processing_flag if current_flag != 'None' else processing_flag
        loupe_df.loc[loupe_df['key'] == 'specProcessingApplied', 'value'] = new_flag
        loupe_df.to_csv(loupe_file, header=False, index=False)
        logger.info(f"Updated processing flag: {current_flag} -> {new_flag}")
    else:
        logger.info(f"Processing flag {processing_flag} already present: {current_flag}")
