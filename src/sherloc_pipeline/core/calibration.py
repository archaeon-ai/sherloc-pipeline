"""Wavenumber calibration: CCD channel index to wavelength/wavenumber via Loupe polynomial coefficients."""

import numpy as np


def calculate_loupe_wavelength_wavenumber(n_channels: int, laser_wavelength: float = 248.5794):
    """
    Calculates wavelength and Raman shift arrays using Loupe's segmented polynomial parameters.

    Args:
        n_channels: Number of spectral channels
        laser_wavelength: Laser wavelength in nm (default: 248.5794)

    Returns:
        tuple: (wavelength, wavenumber) arrays
    """
    # Loupe's polynomial coefficients
    popt_R = [-7.85000e-06, 6.52400e-02, 2.46690e+02]
    popt_F = [-5.65724e-06, 6.33627e-02, 2.47474e+02]
    cutoff_channel = 500

    _pixels = list(range(n_channels))
    wavelength = list(np.polyval(popt_R, _pixels[0:cutoff_channel+1])) + \
                 list(np.polyval(popt_F, _pixels[cutoff_channel+1:]))

    wavenumber = [(10**7) * ((1/laser_wavelength) - (1/w)) for w in wavelength]

    return np.array(wavelength), np.array(wavenumber)


def get_region_wavelength_mask(wavelength, region):
    """
    Get wavelength mask for specific region based on wavelength ranges.

    Args:
        wavelength: Array of wavelengths
        region: Region ('R1', 'R2', or 'R3')

    Returns:
        numpy.array: Boolean mask for the region
    """
    if region == 'R1':
        return (wavelength >= 250.0) & (wavelength <= 282.0)
    elif region == 'R2':
        return (wavelength >= 282.0) & (wavelength <= 337.8)
    elif region == 'R3':
        return (wavelength >= 337.8) & (wavelength <= 357.4)
    else:
        raise ValueError(f"Invalid region: {region}")
