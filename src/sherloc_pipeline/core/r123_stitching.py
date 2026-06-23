"""
R123 spectrum stitching module for SHERLOC pipeline.

Implements the Loupe R123 stitching algorithm with overlap summation.
Each of R1, R2, R3 is a separate full-CCD readout (2148 channels).
This module combines them into a single 2148-channel stitched spectrum.

Loupe reference: file_IO.py lines 714-739

Channel layout (overlap SUMMATION, not averaging):
    Channel 0-564:     R1 only       (565 channels, copy)
    Channel 565-689:   R1 + R2       (125 channels, sum)
    Channel 690-1667:  R2 only       (978 channels, copy)
    Channel 1668-1689: R2 + R3       (22 channels, sum)
    Channel 1690-2147: R3 only       (458 channels, copy)

Total: 565 + 125 + 978 + 22 + 458 = 2148 channels
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: Loupe stitching boundaries (inclusive channel indices)
# ---------------------------------------------------------------------------

N_CHANNELS = 2148
"""Total CCD channels per readout."""

# Region boundaries (Python slice endpoints are exclusive)
_R1_ONLY_END = 565       # channels [0, 565)
_OVERLAP1_END = 690      # channels [565, 690) -> R1+R2
_R2_ONLY_END = 1668      # channels [690, 1668) -> R2 only
_OVERLAP2_END = 1690     # channels [1668, 1690) -> R2+R3
_R3_ONLY_END = N_CHANNELS  # channels [1690, 2148) -> R3 only

# Widths (for documentation/validation)
_OVERLAP1_WIDTH = _OVERLAP1_END - _R1_ONLY_END   # 125
_OVERLAP2_WIDTH = _OVERLAP2_END - _R2_ONLY_END    # 22


def stitch_r123_spectrum(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    *,
    nan_to_zero: bool = True,
) -> np.ndarray:
    """Stitch R1, R2, R3 readouts into a combined R123 spectrum.

    Matches the Loupe algorithm with overlap **summation** (not averaging).

    Parameters
    ----------
    r1 : np.ndarray
        R1 readout, shape ``(2148,)``. Meaningful data in channels 0-574;
        channels 565-689 overlap with R2.
    r2 : np.ndarray
        R2 readout, shape ``(2148,)``. Meaningful data in channels 565-1668;
        channels 565-689 overlap with R1, channels 1668-1689 overlap with R3.
    r3 : np.ndarray
        R3 readout, shape ``(2148,)``. Meaningful data in channels 1668-2147;
        channels 1668-1689 overlap with R2.
    nan_to_zero : bool, default True
        If *True*, replace NaN values in inputs with 0 before stitching.
        This prevents NaN propagation into non-NaN regions during summation.

    Returns
    -------
    np.ndarray
        Stitched 2148-channel spectrum, dtype ``float64``.

    Raises
    ------
    ValueError
        If any input does not have exactly 2148 elements.
    """
    # --- validation ---
    for name, arr in [("r1", r1), ("r2", r2), ("r3", r3)]:
        if arr.shape != (N_CHANNELS,):
            raise ValueError(
                f"{name} must have shape ({N_CHANNELS},), got {arr.shape}"
            )

    # Work on float64 copies to avoid mutating caller data
    _r1 = np.asarray(r1, dtype=np.float64)
    _r2 = np.asarray(r2, dtype=np.float64)
    _r3 = np.asarray(r3, dtype=np.float64)

    if nan_to_zero:
        np.nan_to_num(_r1, copy=False, nan=0.0)
        np.nan_to_num(_r2, copy=False, nan=0.0)
        np.nan_to_num(_r3, copy=False, nan=0.0)

    r123 = np.zeros(N_CHANNELS, dtype=np.float64)

    # Region 1: R1 only (channels 0-564)
    r123[0:_R1_ONLY_END] = _r1[0:_R1_ONLY_END]

    # Overlap 1: R1 + R2 (channels 565-689) -- SUMMATION
    r123[_R1_ONLY_END:_OVERLAP1_END] = (
        _r1[_R1_ONLY_END:_OVERLAP1_END] + _r2[_R1_ONLY_END:_OVERLAP1_END]
    )

    # Region 2: R2 only (channels 690-1667)
    r123[_OVERLAP1_END:_R2_ONLY_END] = _r2[_OVERLAP1_END:_R2_ONLY_END]

    # Overlap 2: R2 + R3 (channels 1668-1689) -- SUMMATION
    r123[_R2_ONLY_END:_OVERLAP2_END] = (
        _r2[_R2_ONLY_END:_OVERLAP2_END] + _r3[_R2_ONLY_END:_OVERLAP2_END]
    )

    # Region 3: R3 only (channels 1690-2147)
    r123[_OVERLAP2_END:_R3_ONLY_END] = _r3[_OVERLAP2_END:_R3_ONLY_END]

    return r123


def stitch_r123_batch(
    r1_batch: np.ndarray,
    r2_batch: np.ndarray,
    r3_batch: np.ndarray,
    *,
    nan_to_zero: bool = True,
) -> np.ndarray:
    """Stitch batches of R1, R2, R3 readouts into R123 spectra (vectorised).

    Parameters
    ----------
    r1_batch, r2_batch, r3_batch : np.ndarray
        Arrays of shape ``(N, 2148)`` where *N* is the number of spectra.
    nan_to_zero : bool, default True
        Replace NaN values with 0 before stitching.

    Returns
    -------
    np.ndarray
        Stitched spectra, shape ``(N, 2148)``, dtype ``float64``.

    Raises
    ------
    ValueError
        If inputs have mismatched leading dimensions or wrong channel count.
    """
    # --- validation ---
    for name, arr in [("r1_batch", r1_batch), ("r2_batch", r2_batch),
                      ("r3_batch", r3_batch)]:
        if arr.ndim != 2 or arr.shape[1] != N_CHANNELS:
            raise ValueError(
                f"{name} must have shape (N, {N_CHANNELS}), got {arr.shape}"
            )

    if not (r1_batch.shape[0] == r2_batch.shape[0] == r3_batch.shape[0]):
        raise ValueError(
            f"Batch sizes must match: r1={r1_batch.shape[0]}, "
            f"r2={r2_batch.shape[0]}, r3={r3_batch.shape[0]}"
        )

    n = r1_batch.shape[0]

    # Work on float64 copies
    _r1 = np.array(r1_batch, dtype=np.float64)
    _r2 = np.array(r2_batch, dtype=np.float64)
    _r3 = np.array(r3_batch, dtype=np.float64)

    if nan_to_zero:
        np.nan_to_num(_r1, copy=False, nan=0.0)
        np.nan_to_num(_r2, copy=False, nan=0.0)
        np.nan_to_num(_r3, copy=False, nan=0.0)

    r123 = np.zeros((n, N_CHANNELS), dtype=np.float64)

    # Region 1: R1 only
    r123[:, 0:_R1_ONLY_END] = _r1[:, 0:_R1_ONLY_END]

    # Overlap 1: R1 + R2 -- SUMMATION
    r123[:, _R1_ONLY_END:_OVERLAP1_END] = (
        _r1[:, _R1_ONLY_END:_OVERLAP1_END] + _r2[:, _R1_ONLY_END:_OVERLAP1_END]
    )

    # Region 2: R2 only
    r123[:, _OVERLAP1_END:_R2_ONLY_END] = _r2[:, _OVERLAP1_END:_R2_ONLY_END]

    # Overlap 2: R2 + R3 -- SUMMATION
    r123[:, _R2_ONLY_END:_OVERLAP2_END] = (
        _r2[:, _R2_ONLY_END:_OVERLAP2_END] + _r3[:, _R2_ONLY_END:_OVERLAP2_END]
    )

    # Region 3: R3 only
    r123[:, _OVERLAP2_END:_R3_ONLY_END] = _r3[:, _OVERLAP2_END:_R3_ONLY_END]

    return r123


def r123_wavelength_axis() -> np.ndarray:
    """Return the calibrated 2148-element wavelength axis for stitched R123 spectra.

    Uses the Loupe V5.1.5a segmented polynomial calibration -- NEVER np.linspace.

    Returns
    -------
    np.ndarray
        Wavelength array in nm, shape ``(2148,)``.
    """
    from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber

    wavelength, _ = calculate_loupe_wavelength_wavenumber(N_CHANNELS)
    return wavelength


def r123_wavenumber_axis() -> np.ndarray:
    """Return the calibrated 2148-element wavenumber axis for stitched R123 spectra.

    Uses the Loupe V5.1.5a segmented polynomial calibration -- NEVER np.linspace.

    Returns
    -------
    np.ndarray
        Wavenumber (Raman shift) array in cm^-1, shape ``(2148,)``.
    """
    from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber

    _, wavenumber = calculate_loupe_wavelength_wavenumber(N_CHANNELS)
    return wavenumber
