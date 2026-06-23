"""Lightweight fluorescence feature detection for hypothesis-driven fitting.

Scans for known fluorescence features (Group 1 doublet, Group 2, Group 3)
using scipy.signal.find_peaks with group-specific windows and prominence
thresholds. Results feed into candidate model enumeration for constrained
hypothesis fitting.

See docs/specs/FLUORESCENCE_FITTING_SPEC.md §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


@dataclass
class FeatureDetection:
    """A detected feature within a group window."""

    group: str
    center_nm: float
    prominence: float
    estimated_fwhm_nm: float
    estimated_amplitude: float


@dataclass
class FeatureScanResult:
    """Result of scanning a fluorescence spectrum for known features."""

    group3_detected: bool = False
    group1_detected: bool = False  # True only if both 1a and 1b found
    group2_detected: bool = False
    detections: List[FeatureDetection] = field(default_factory=list)

    @property
    def group1a_detected(self) -> bool:
        return any(d.group == "group1a" for d in self.detections)

    @property
    def group1b_detected(self) -> bool:
        return any(d.group == "group1b" for d in self.detections)


@dataclass
class CandidateModel:
    """A candidate fluorescence model to fit.

    Specifies which features to include and per-peak bounds for
    constrained differential evolution.
    """

    model_id: str  # M0..M7
    features: List[str]  # e.g. ["group3", "group1a", "group1b"]
    n_peaks: int
    bounds: List[Tuple[float, float]]  # flattened [center, amp, fwhm] per peak
    description: str = ""


# Default scan parameters per group
_GROUP_WINDOWS = {
    "group3": (270.0, 295.0),
    "group1a": (295.0, 315.0),  # wider than assignment range for detection
    "group1b": (315.0, 332.0),  # wider than assignment range for detection
    "group2": (329.0, 357.0),
}

# Minimum prominence (relative to noise) to consider a feature detected
_DEFAULT_PROMINENCE_FACTOR = 3.0


def scan_fluorescence_features(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    noise_std: float,
    prominence_factor: float = _DEFAULT_PROMINENCE_FACTOR,
    group_windows: Optional[dict] = None,
) -> FeatureScanResult:
    """Scan for known fluorescence features using peak prominence.

    This is a cheap binary classifier per feature (~0.2 ms/point). It answers
    "is there significant signal here?" and estimates initial parameters as
    seeds for constrained fitting.

    A detection requires both sufficient prominence AND sufficient peak
    amplitude relative to noise, to reject noise spikes that happen to
    have high local prominence.

    Args:
        wavelength: Wavelength array in nm (monotonically increasing).
        intensity: Dark-subtracted intensity array (despiked).
        noise_std: Noise standard deviation for prominence thresholding.
        prominence_factor: Multiplier on noise_std for minimum prominence.
        group_windows: Override group detection windows {group: (lo, hi)}.

    Returns:
        FeatureScanResult with per-group detection flags and initial estimates.
    """
    windows = group_windows or _GROUP_WINDOWS
    min_prominence = prominence_factor * max(noise_std, 1.0)
    # Amplitude must be well above noise floor to be a real feature
    min_amplitude = prominence_factor * 2.0 * max(noise_std, 1.0)

    result = FeatureScanResult()

    for group, (lo, hi) in windows.items():
        mask = (wavelength >= lo) & (wavelength <= hi)
        if np.sum(mask) < 3:
            continue

        x_win = wavelength[mask]
        y_win = intensity[mask]

        # find_peaks on the intensity within this window
        peak_indices, properties = find_peaks(
            y_win,
            prominence=min_prominence,
            width=3,  # minimum width in samples (reject single-sample spikes)
        )

        if len(peak_indices) == 0:
            continue

        # Take the most prominent peak in this window
        prominences = properties["prominences"]
        best_prom_idx = int(np.argmax(prominences))
        best_idx = peak_indices[best_prom_idx]
        best_prominence = float(prominences[best_prom_idx])

        center = float(x_win[best_idx])
        amplitude = float(y_win[best_idx])

        # Reject if amplitude is too low (noise spike with high local prominence)
        if amplitude < min_amplitude:
            continue

        # Estimate FWHM from width at half-prominence
        widths = properties["widths"]
        best_width_samples = float(widths[best_prom_idx])
        # Convert sample width to nm
        dx = float(np.median(np.diff(x_win))) if len(x_win) > 1 else 0.2
        estimated_fwhm = best_width_samples * dx

        detection = FeatureDetection(
            group=group,
            center_nm=center,
            prominence=best_prominence,
            estimated_fwhm_nm=max(estimated_fwhm, 5.0),  # floor at 5 nm
            estimated_amplitude=amplitude,
        )
        result.detections.append(detection)

    # Set group flags
    result.group3_detected = any(d.group == "group3" for d in result.detections)
    result.group2_detected = any(d.group == "group2" for d in result.detections)
    # Group 1: detecting group1a implies the doublet — Ce3+ 5d→2F5/2
    # (group1a) always has a 5d→2F7/2 partner (group1b). The feature scan
    # often misses the broader/weaker group1b, but constrained DE will find
    # it when the doublet model (M3) is enumerated.
    has_1a = any(d.group == "group1a" for d in result.detections)
    result.group1_detected = has_1a

    return result


# Feature bounds for constrained hypothesis fitting
_FEATURE_BOUNDS = {
    "group3": {
        "center": (270.0, 295.0),
        "fwhm": (10.0, 25.0),
    },
    "group1a": {
        "center": (300.0, 307.0),
        "fwhm": (8.0, 18.0),  # empirical: 13.4 ± 1.5 nm (Berry Hollow N=75, ±3σ)
    },
    "group1b": {
        "center": (322.0, 329.0),
        "fwhm": (11.0, 29.0),  # empirical: 20.1 ± 3.0 nm (Berry Hollow N=78, ±3σ)
    },
    "group2": {
        "center": (329.0, 355.0),
        "fwhm": (15.0, 30.0),
    },
}


def enumerate_candidate_models(
    scan: FeatureScanResult,
    y_max: float,
) -> List[CandidateModel]:
    """Generate candidate models from feature scan results.

    Always includes M0 (null baseline). Only includes models whose
    constituent features were detected in the scan.

    Args:
        scan: Result from scan_fluorescence_features().
        y_max: Maximum intensity for amplitude bounds.

    Returns:
        List of CandidateModel objects to fit.
    """
    amp_bound = (0.0, y_max * 1.5)
    models: List[CandidateModel] = []

    # M0: null model (always included as baseline)
    models.append(CandidateModel(
        model_id="M0",
        features=[],
        n_peaks=0,
        bounds=[],
        description="null baseline",
    ))

    def _make_bounds(features: List[str]) -> List[Tuple[float, float]]:
        bounds: List[Tuple[float, float]] = []
        for feat in features:
            fb = _FEATURE_BOUNDS[feat]
            bounds.extend([fb["center"], amp_bound, fb["fwhm"]])
        return bounds

    # M1: Group 3 only
    if scan.group3_detected:
        features = ["group3"]
        models.append(CandidateModel(
            model_id="M1",
            features=features,
            n_peaks=1,
            bounds=_make_bounds(features),
            description="Group 3 only (silicate defect)",
        ))

    # M2: Group 2 only
    if scan.group2_detected:
        features = ["group2"]
        models.append(CandidateModel(
            model_id="M2",
            features=features,
            n_peaks=1,
            bounds=_make_bounds(features),
            description="Group 2 only (Ce3+ phosphate)",
        ))

    # M3: Group 1 doublet
    if scan.group1_detected:
        features = ["group1a", "group1b"]
        models.append(CandidateModel(
            model_id="M3",
            features=features,
            n_peaks=2,
            bounds=_make_bounds(features),
            description="Group 1 doublet (Ce3+ anhydrite)",
        ))

    # M4: Group 3 + Group 2
    if scan.group3_detected and scan.group2_detected:
        features = ["group3", "group2"]
        models.append(CandidateModel(
            model_id="M4",
            features=features,
            n_peaks=2,
            bounds=_make_bounds(features),
            description="Group 3 + Group 2",
        ))

    # M5: Group 3 + Group 1 doublet
    if scan.group3_detected and scan.group1_detected:
        features = ["group3", "group1a", "group1b"]
        models.append(CandidateModel(
            model_id="M5",
            features=features,
            n_peaks=3,
            bounds=_make_bounds(features),
            description="Group 3 + Group 1 doublet",
        ))

    # M6: Group 2 + Group 1 doublet
    if scan.group2_detected and scan.group1_detected:
        features = ["group2", "group1a", "group1b"]
        models.append(CandidateModel(
            model_id="M6",
            features=features,
            n_peaks=3,
            bounds=_make_bounds(features),
            description="Group 2 + Group 1 doublet",
        ))

    # M7: All three groups
    if scan.group3_detected and scan.group2_detected and scan.group1_detected:
        features = ["group3", "group1a", "group1b", "group2"]
        models.append(CandidateModel(
            model_id="M7",
            features=features,
            n_peaks=4,
            bounds=_make_bounds(features),
            description="All groups (Group 3 + Group 1 doublet + Group 2)",
        ))

    return models
