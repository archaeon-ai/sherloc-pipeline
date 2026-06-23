"""Laser normalization diagnostic plots."""

import logging

import numpy as np
import matplotlib.pyplot as plt

from sherloc_pipeline.core.calibration import get_region_wavelength_mask

logger = logging.getLogger(__name__)


def create_region_comparison_plot(original_data, normalized_data, wavelength, wavenumber,
                                 region, output_dir, sol, target, scan):
    """
    Create comparison plot for a specific region showing original vs normalized spectra.

    Args:
        original_data: Original region data
        normalized_data: Normalized region data
        wavelength: Wavelength array
        wavenumber: Wavenumber array
        region: Region name ('R1', 'R2', or 'R3')
        output_dir: Output directory for plots
        sol: Sol number
        target: Target name
        scan: Scan type
    """
    # Calculate average spectra for the region
    original_avg = original_data.mean(axis=0).values
    normalized_avg = normalized_data.mean(axis=0).values

    # Get wavelength mask for this region
    region_mask = get_region_wavelength_mask(wavelength, region)

    # Filter data to region-specific wavelength range
    region_wavelength = wavelength[region_mask]
    region_wavenumber = wavenumber[region_mask]
    region_original_avg = original_avg[region_mask]
    region_normalized_avg = normalized_avg[region_mask]

    # Create the plot based on region
    if region == 'R1':
        # R1: Only Raman shift plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Sort by wavenumber for increasing order
        sort_indices = np.argsort(region_wavenumber)
        region_wavenumber_sorted = region_wavenumber[sort_indices]
        region_original_sorted = region_original_avg[sort_indices]
        region_normalized_sorted = region_normalized_avg[sort_indices]

        ax.plot(region_wavenumber_sorted, region_original_sorted, color='#1f77b4', linewidth=1.5, alpha=0.8, label='Raw')
        ax.plot(region_wavenumber_sorted, region_normalized_sorted, color='#ff7f0e', linewidth=1.5, alpha=0.8, label='Normalized')
        ax.set_xlabel('Raman Shift (cm⁻¹)')
        ax.set_ylabel('Intensity (counts)')
        ax.set_title(f'Average {region} Spectrum - Sol {sol} {target} {scan} - Raw vs Normalized')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Limit to [first, 4000] cm^-1 for R1
        try:
            x0 = float(region_wavenumber_sorted[0])
        except Exception:
            x0 = 0.0
        ax.set_xlim([x0, 4000])

    else:
        # R2 and R3: Only wavelength plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(region_wavelength, region_original_avg, color='#1f77b4', linewidth=1.5, alpha=0.8, label='Raw')
        ax.plot(region_wavelength, region_normalized_avg, color='#ff7f0e', linewidth=1.5, alpha=0.8, label='Normalized')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (counts)')
        ax.set_title(f'Average {region} Spectrum - Sol {sol} {target} {scan} - Raw vs Normalized')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    output_file = output_dir / f"{sol}_{target}_{scan}_{region}_normalization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    logger.info(f"{region} comparison plot saved to: {output_file}")

    # Log statistics for this region
    logger.info(f"{region} - Wavelength range: {region_wavelength.min():.1f} - {region_wavelength.max():.1f} nm")
    logger.info(f"{region} - Original range: {region_original_avg.min():.1f} - {region_original_avg.max():.1f} counts")
    logger.info(f"{region} - Normalized range: {region_normalized_avg.min():.1f} - {region_normalized_avg.max():.1f} counts")
    logger.info(f"{region} - Mean intensity change: {((region_normalized_avg.mean() / region_original_avg.mean()) - 1) * 100:.1f}%")
