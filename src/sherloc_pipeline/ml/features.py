"""
Feature extraction utilities for spectral data.

This module provides feature extractors for transforming SHERLOC spectra
into feature vectors suitable for ML analysis.

Classes:
    FeatureExtractor: Base class for feature extractors
    PeakFeatureExtractor: Extract peak-based features
    StatisticalFeatureExtractor: Extract statistical summary features
    SpectralFeatureExtractor: Combined spectral feature extraction

Functions:
    extract_peak_features: Quick function for peak feature extraction

Example:
    >>> from sherloc_pipeline.ml.features import SpectralFeatureExtractor
    >>> import numpy as np
    >>>
    >>> spectrum = np.random.randn(500)
    >>> extractor = SpectralFeatureExtractor()
    >>> features = extractor.extract(spectrum)
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from sherloc_pipeline.ml.base import Transformer


@dataclass
class FeatureVector:
    """Container for extracted features with metadata.

    Attributes:
        values: Feature values as numpy array
        names: Feature names corresponding to values
        metadata: Additional extraction metadata
    """

    values: np.ndarray
    names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of name: value pairs."""
        return {name: float(val) for name, val in zip(self.names, self.values)}


class FeatureExtractor(Transformer):
    """Base class for spectral feature extractors.

    Feature extractors transform raw spectra into fixed-length
    feature vectors suitable for ML algorithms.

    Subclasses must implement:
        - extract(spectrum): Extract features from single spectrum
    """

    def __init__(self):
        super().__init__()
        self.feature_names_: List[str] = []

    @abstractmethod
    def extract(self, spectrum: ArrayLike) -> FeatureVector:
        """Extract features from a single spectrum.

        Args:
            spectrum: 1D array of spectral values

        Returns:
            FeatureVector with extracted features
        """
        pass

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> "FeatureExtractor":
        """Fit the extractor (most extractors don't need fitting).

        Args:
            X: Training spectra of shape (n_samples, n_features)
            y: Ignored

        Returns:
            self
        """
        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform multiple spectra to feature vectors.

        Args:
            X: Input spectra of shape (n_samples, n_wavelengths)

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        features = []
        for spectrum in X:
            fv = self.extract(spectrum)
            features.append(fv.values)
            if not self.feature_names_:
                self.feature_names_ = fv.names
        return np.array(features)


class PeakFeatureExtractor(FeatureExtractor):
    """Extract features based on spectral peaks.

    Identifies prominent peaks and extracts their positions,
    heights, widths, and relative intensities.

    Attributes:
        n_peaks: Maximum number of peaks to extract
        min_height: Minimum peak height (relative to max)
        min_prominence: Minimum peak prominence
        width_rel_height: Relative height for width calculation

    Example:
        >>> extractor = PeakFeatureExtractor(n_peaks=5)
        >>> features = extractor.extract(spectrum)
    """

    def __init__(
        self,
        n_peaks: int = 10,
        min_height: float = 0.1,
        min_prominence: float = 0.05,
        width_rel_height: float = 0.5,
    ):
        """Initialize peak feature extractor.

        Args:
            n_peaks: Maximum number of peaks to extract
            min_height: Minimum relative height threshold
            min_prominence: Minimum prominence threshold
            width_rel_height: Relative height for width calculation (0.5 = FWHM)
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.min_height = min_height
        self.min_prominence = min_prominence
        self.width_rel_height = width_rel_height

    def extract(self, spectrum: ArrayLike) -> FeatureVector:
        """Extract peak-based features from spectrum.

        Features include:
        - Peak positions (normalized to [0, 1])
        - Peak heights (normalized to max)
        - Peak widths (in channels)
        - Number of peaks found

        Args:
            spectrum: 1D spectral array

        Returns:
            FeatureVector with peak features
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        n_channels = len(spectrum)

        # Normalize spectrum for threshold comparison
        spec_min = spectrum.min()
        spec_max = spectrum.max()
        spec_range = spec_max - spec_min if spec_max > spec_min else 1.0
        spec_norm = (spectrum - spec_min) / spec_range

        # Simple peak detection (local maxima)
        peaks = []
        for i in range(1, n_channels - 1):
            if spec_norm[i] > spec_norm[i - 1] and spec_norm[i] > spec_norm[i + 1]:
                if spec_norm[i] >= self.min_height:
                    # Compute prominence (height above local baseline)
                    left_min = np.min(spec_norm[max(0, i - 10):i])
                    right_min = np.min(spec_norm[i + 1:min(n_channels, i + 11)])
                    prominence = spec_norm[i] - max(left_min, right_min)

                    if prominence >= self.min_prominence:
                        # Estimate width at relative height
                        target_height = spec_norm[i] - prominence * self.width_rel_height
                        left_idx, right_idx = i, i

                        while left_idx > 0 and spec_norm[left_idx] > target_height:
                            left_idx -= 1
                        while right_idx < n_channels - 1 and spec_norm[right_idx] > target_height:
                            right_idx += 1

                        width = right_idx - left_idx

                        peaks.append({
                            "position": i,
                            "height": spec_norm[i],
                            "prominence": prominence,
                            "width": width,
                        })

        # Sort by prominence and take top n_peaks
        peaks = sorted(peaks, key=lambda p: p["prominence"], reverse=True)[:self.n_peaks]

        # Sort by position for consistent ordering
        peaks = sorted(peaks, key=lambda p: p["position"])

        # Build feature vector
        features = []
        names = []

        # Number of peaks
        features.append(len(peaks))
        names.append("n_peaks")

        # Pad to n_peaks with zeros
        for i in range(self.n_peaks):
            if i < len(peaks):
                p = peaks[i]
                features.extend([
                    p["position"] / n_channels,  # Normalized position
                    p["height"],  # Already normalized
                    p["width"] / n_channels,  # Normalized width
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

            names.extend([
                f"peak_{i}_position",
                f"peak_{i}_height",
                f"peak_{i}_width",
            ])

        return FeatureVector(
            values=np.array(features),
            names=names,
            metadata={"n_peaks_found": len(peaks)},
        )


class StatisticalFeatureExtractor(FeatureExtractor):
    """Extract statistical summary features from spectra.

    Computes various statistical measures that characterize
    the overall shape and distribution of spectral values.

    Features include:
    - Mean, std, min, max, range
    - Skewness, kurtosis
    - Percentiles (25, 50, 75)
    - Zero-crossing rate
    - Energy metrics

    Example:
        >>> extractor = StatisticalFeatureExtractor()
        >>> features = extractor.extract(spectrum)
    """

    def __init__(self, percentiles: List[int] = None):
        """Initialize statistical feature extractor.

        Args:
            percentiles: List of percentiles to compute (default: [25, 50, 75])
        """
        super().__init__()
        self.percentiles = percentiles or [25, 50, 75]

    def extract(self, spectrum: ArrayLike) -> FeatureVector:
        """Extract statistical features from spectrum.

        Args:
            spectrum: 1D spectral array

        Returns:
            FeatureVector with statistical features
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)

        features = []
        names = []

        # Basic statistics
        features.append(np.mean(spectrum))
        names.append("mean")

        features.append(np.std(spectrum))
        names.append("std")

        features.append(np.min(spectrum))
        names.append("min")

        features.append(np.max(spectrum))
        names.append("max")

        features.append(np.max(spectrum) - np.min(spectrum))
        names.append("range")

        # Percentiles
        for p in self.percentiles:
            features.append(np.percentile(spectrum, p))
            names.append(f"percentile_{p}")

        # Higher moments
        mean = np.mean(spectrum)
        std = np.std(spectrum)
        if std > 0:
            normalized = (spectrum - mean) / std
            skewness = np.mean(normalized ** 3)
            kurtosis = np.mean(normalized ** 4) - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        features.append(skewness)
        names.append("skewness")

        features.append(kurtosis)
        names.append("kurtosis")

        # Zero-crossing rate (how often signal changes sign)
        centered = spectrum - np.mean(spectrum)
        zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
        zero_crossing_rate = zero_crossings / len(spectrum)

        features.append(zero_crossing_rate)
        names.append("zero_crossing_rate")

        # Energy metrics
        energy = np.sum(spectrum ** 2)
        features.append(energy)
        names.append("energy")

        # RMS
        rms = np.sqrt(np.mean(spectrum ** 2))
        features.append(rms)
        names.append("rms")

        # Spectral centroid (center of mass)
        indices = np.arange(len(spectrum))
        if np.sum(np.abs(spectrum)) > 0:
            centroid = np.sum(indices * np.abs(spectrum)) / np.sum(np.abs(spectrum))
        else:
            centroid = len(spectrum) / 2
        features.append(centroid / len(spectrum))
        names.append("centroid")

        # Spectral spread (variance around centroid)
        if np.sum(np.abs(spectrum)) > 0:
            spread = np.sqrt(
                np.sum(((indices - centroid) ** 2) * np.abs(spectrum))
                / np.sum(np.abs(spectrum))
            )
        else:
            spread = 0.0
        features.append(spread / len(spectrum))
        names.append("spread")

        return FeatureVector(
            values=np.array(features),
            names=names,
        )


class SpectralFeatureExtractor(FeatureExtractor):
    """Combined spectral feature extraction.

    Combines peak-based and statistical features into a single
    comprehensive feature vector for ML analysis.

    Attributes:
        peak_extractor: PeakFeatureExtractor instance
        stat_extractor: StatisticalFeatureExtractor instance
        include_derivatives: Whether to include derivative features

    Example:
        >>> extractor = SpectralFeatureExtractor(n_peaks=5)
        >>> features = extractor.extract(spectrum)
    """

    def __init__(
        self,
        n_peaks: int = 10,
        include_derivatives: bool = True,
        include_stats: bool = True,
    ):
        """Initialize combined feature extractor.

        Args:
            n_peaks: Maximum peaks for peak extraction
            include_derivatives: Include 1st/2nd derivative features
            include_stats: Include statistical features
        """
        super().__init__()
        self.n_peaks = n_peaks
        self.include_derivatives = include_derivatives
        self.include_stats = include_stats
        self.peak_extractor = PeakFeatureExtractor(n_peaks=n_peaks)
        self.stat_extractor = StatisticalFeatureExtractor()

    def extract(self, spectrum: ArrayLike) -> FeatureVector:
        """Extract comprehensive features from spectrum.

        Args:
            spectrum: 1D spectral array

        Returns:
            FeatureVector with combined features
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)

        all_features = []
        all_names = []

        # Peak features
        peak_fv = self.peak_extractor.extract(spectrum)
        all_features.extend(peak_fv.values)
        all_names.extend([f"peak_{n}" for n in peak_fv.names])

        # Statistical features
        if self.include_stats:
            stat_fv = self.stat_extractor.extract(spectrum)
            all_features.extend(stat_fv.values)
            all_names.extend([f"stat_{n}" for n in stat_fv.names])

        # Derivative features
        if self.include_derivatives:
            # First derivative
            d1 = np.gradient(spectrum)
            d1_fv = self.stat_extractor.extract(d1)
            all_features.extend(d1_fv.values)
            all_names.extend([f"d1_{n}" for n in d1_fv.names])

            # Second derivative
            d2 = np.gradient(d1)
            d2_fv = self.stat_extractor.extract(d2)
            all_features.extend(d2_fv.values)
            all_names.extend([f"d2_{n}" for n in d2_fv.names])

        return FeatureVector(
            values=np.array(all_features),
            names=all_names,
        )


def extract_peak_features(
    spectrum: ArrayLike,
    n_peaks: int = 10,
    min_height: float = 0.1,
) -> Dict[str, float]:
    """Quick function to extract peak features from a spectrum.

    Convenience function for simple peak extraction without
    creating an extractor instance.

    Args:
        spectrum: 1D spectral array
        n_peaks: Maximum number of peaks
        min_height: Minimum relative peak height

    Returns:
        Dictionary of feature name to value
    """
    extractor = PeakFeatureExtractor(n_peaks=n_peaks, min_height=min_height)
    fv = extractor.extract(spectrum)
    return fv.to_dict()
