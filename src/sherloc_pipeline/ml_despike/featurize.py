"""Certified 8-channel featurization for the ML CR detector (v1.1).

Clean-room port of the certified reference implementation (maintained in
the research tree, not tracked here). The formulas below ARE the certified
observable's definition (MLD-DET-002): per-frame robust normalization of
the raw ACTIVE and DARK planes over the region's certified detection
window, region one-hot, and log-scale context channels. Any change here
voids the certification evidence base unless re-verified through the
parity gate (MLD-QUA-003).

Import surface is numpy-only (plus the stdlib manifest module) — no
scipy, no pandas, no onnxruntime, no experiment paths.
"""

from typing import Sequence

import numpy as np

from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

#: Spectral regions in canonical order (matches the one-hot encoding).
REGIONS = ("R1", "R2", "R3")

_REGION_ID = {"R1": 0, "R2": 1, "R3": 2}
_REGION_ONEHOT = {r: np.eye(3, dtype=np.float32)[_REGION_ID[r]] for r in REGIONS}

N_CHANNELS = DEFAULT_MANIFEST.n_channels
_REGION_WINDOWS = DEFAULT_MANIFEST.region_windows


def _validate_frame(plane: np.ndarray, plane_name: str) -> np.ndarray:
    arr = np.asarray(plane)
    if arr.ndim != 1 or arr.shape[0] != N_CHANNELS:
        raise ValueError(
            f"{plane_name} plane must be a 1-D array of length {N_CHANNELS}, "
            f"got shape {arr.shape}"
        )
    return arr


def featurize(active: np.ndarray, dark: np.ndarray, region: str) -> np.ndarray:
    """Certified 8-plane input from one raw ACTIVE/DARK frame pair.

    Per plane, over the region's certified window ``[lo, hi)``:
    ``med = median(plane[lo:hi])``, ``mad = median(|plane[lo:hi] - med|)``,
    ``scale = 1.4826 * mad + 1.0``. Channels (float32, shape
    ``(8, N_CHANNELS)``):

    - ``x0 = (active - med_a) / scale_a``
    - ``x1 = (dark - med_d) / scale_d``
    - ``x2..x4`` = region one-hot broadcast
    - ``x5 = log10(scale_a) / 4.0``
    - ``x6 = log10(scale_d) / 4.0``
    - ``x7 = log10(1 + |median(active[lo:hi])|) / 4.0``

    Args:
        active: Raw ACTIVE plane, 1-D length-2148, raw DN
            (pre-normalization).
        dark: Raw DARK plane, same shape and units.
        region: One of ``"R1"``, ``"R2"``, ``"R3"``.

    Returns:
        float32 array of shape ``(8, 2148)``.
    """
    if region not in _REGION_WINDOWS:
        raise ValueError(
            f"unknown region {region!r}; expected one of {', '.join(REGIONS)}"
        )
    active = _validate_frame(active, "active")
    dark = _validate_frame(dark, "dark")

    lo, hi = _REGION_WINDOWS[region]
    x = np.empty((8, N_CHANNELS), dtype=np.float32)
    for k, plane in enumerate((active, dark)):
        win = plane[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        scale = 1.4826 * mad + 1.0
        x[k] = (plane - med) / scale
        x[5 + k] = np.log10(scale) / 4.0
    oh = _REGION_ONEHOT[region]
    x[2] = oh[0]
    x[3] = oh[1]
    x[4] = oh[2]
    x[7] = np.log10(1.0 + abs(float(np.median(active[lo:hi])))) / 4.0
    return x


def featurize_batch(
    actives: Sequence[np.ndarray],
    darks: Sequence[np.ndarray],
    regions: Sequence[str],
) -> np.ndarray:
    """Stack per-frame featurizations to ``(N, 8, 2148)`` float32.

    Inputs are parallel sequences: ``actives[i]``/``darks[i]`` is the raw
    plane pair of frame *i* and ``regions[i]`` its region. Order is
    preserved.
    """
    if not (len(actives) == len(darks) == len(regions)):
        raise ValueError(
            "actives, darks, and regions must have equal lengths; got "
            f"{len(actives)}, {len(darks)}, {len(regions)}"
        )
    if len(actives) == 0:
        return np.empty((0, 8, N_CHANNELS), dtype=np.float32)
    return np.stack([featurize(a, d, r) for a, d, r in zip(actives, darks, regions)])
