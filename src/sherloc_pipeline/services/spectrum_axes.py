"""Resolve native spectrum axes without changing detector-channel order."""

import hashlib
import zlib
from dataclasses import dataclass

import numpy as np

from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)


@dataclass
class SpectrumAxes:
    wavelength: np.ndarray
    wavenumber: np.ndarray
    mask: np.ndarray
    source: str
    calibration_version: str | None
    excitation_nm: float
    excitation_basis: str
    sha256: str


def resolve_spectrum_axes(scan, spectra, region: str) -> SpectrumAxes:
    """Require identical axes for every aggregate/stitch contributor.

    Stored axes use compressed float32 arrays, like intensities. Rounded
    duplicates are valid; the Loupe polynomial's known segment break is
    permitted only for axes explicitly labeled loupe_polynomial.
    """
    if not spectra:
        raise ValueError("No spectrum axes available")
    resolved = []
    for sp in spectra:
        n_channels = len(np.frombuffer(zlib.decompress(sp.intensities), dtype=np.float32))
        if sp.wavelengths is not None:
            wavelength = np.frombuffer(zlib.decompress(sp.wavelengths), dtype=np.float32).astype(float)
            source = sp.wavelength_source or "stored"
            if (
                len(wavelength) != n_channels
                or not len(wavelength)
                or not np.isfinite(wavelength).all()
                or (wavelength <= 0).any()
            ):
                raise ValueError("Invalid stored wavelength array")
            differences = np.diff(wavelength)
            if source == "loupe_polynomial" and len(differences) > 500:
                differences = np.delete(differences, 500)
            if (differences < 0).any() or not (differences > 0).any():
                raise ValueError("Invalid stored wavelength ordering")
            excitation = scan.laser_wavelength_nm
            # Ingestion may have populated this field from the model default;
            # the stored scan value alone cannot establish measured excitation.
            basis = "scan_record_or_nominal_default"
            calibration = "loupe_v5.1.5a" if source == "loupe_polynomial" else None
            if excitation is None or not np.isfinite(excitation) or excitation <= 0:
                raise ValueError("Invalid recorded excitation wavelength")
            wavenumber = 1e7 * (1 / excitation - 1 / wavelength)
        else:
            if scan.data_source == "pds4" or sp.wavelength_source == "pds_embedded":
                raise ValueError("PDS spectrum is missing its embedded wavelength array")
            wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels)
            source = "loupe_polynomial"
            calibration = "loupe_v5.1.5a"
            excitation = 248.5794
            basis = "loupe_calibration_default"
        if region == "R123" and n_channels != 2148:
            raise ValueError("R123 requires full 2148-channel readouts")
        identity = (source, calibration, excitation, basis)
        if resolved:
            previous, _, previous_identity = resolved[0]
            if identity != previous_identity or not np.array_equal(wavelength, previous):
                raise ValueError("Incompatible spectrum axes; resampling is not supported")
        resolved.append((wavelength, wavenumber, identity))
    wavelength, wavenumber, (source, calibration, excitation, basis) = resolved[0]
    mask = (
        np.ones(len(wavelength), dtype=bool)
        if region == "R123"
        else get_region_wavelength_mask(wavelength, region)
    )
    if not mask.any():
        raise ValueError("Spectrum axis has no channels in the requested region")
    digest = hashlib.sha256(wavelength.astype("<f8").tobytes())
    digest.update(wavenumber.astype("<f8").tobytes())
    return SpectrumAxes(wavelength, wavenumber, mask, source, calibration, excitation, basis, digest.hexdigest())
