"""
Unit tests for ML feature extraction.

Tests the feature extraction utilities in ml/features.py:
- PeakFeatureExtractor
- StatisticalFeatureExtractor
- SpectralFeatureExtractor
- extract_peak_features function
"""

import numpy as np
import pytest

from sherloc_pipeline.ml.features import (
    FeatureVector,
    FeatureExtractor,
    PeakFeatureExtractor,
    StatisticalFeatureExtractor,
    SpectralFeatureExtractor,
    extract_peak_features,
)


@pytest.fixture
def synthetic_spectrum():
    """Create synthetic spectrum with known peaks."""
    n_channels = 500
    x = np.arange(n_channels)

    # Create spectrum with 3 clear peaks
    spectrum = (
        100 * np.exp(-((x - 100) ** 2) / 100) +  # Peak at 100
        200 * np.exp(-((x - 250) ** 2) / 50) +   # Peak at 250 (tallest)
        150 * np.exp(-((x - 400) ** 2) / 80) +   # Peak at 400
        10 * np.random.randn(n_channels)         # Noise
    )
    return spectrum


@pytest.fixture
def flat_spectrum():
    """Create flat spectrum with no peaks."""
    return np.ones(500) * 100 + np.random.randn(500) * 0.1


class TestFeatureVector:
    """Tests for FeatureVector dataclass."""

    def test_basic_creation(self):
        """Create FeatureVector with values and names."""
        values = np.array([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]

        fv = FeatureVector(values=values, names=names)

        assert len(fv.values) == 3
        assert len(fv.names) == 3

    def test_to_dict(self):
        """Convert FeatureVector to dictionary."""
        values = np.array([1.0, 2.0, 3.0])
        names = ["a", "b", "c"]

        fv = FeatureVector(values=values, names=names)
        d = fv.to_dict()

        assert d == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_metadata(self):
        """FeatureVector can store metadata."""
        fv = FeatureVector(
            values=np.array([1.0]),
            names=["x"],
            metadata={"source": "test", "version": 1}
        )

        assert fv.metadata["source"] == "test"
        assert fv.metadata["version"] == 1


class TestPeakFeatureExtractor:
    """Tests for PeakFeatureExtractor."""

    def test_extract_known_peaks(self, synthetic_spectrum):
        """Extract peaks from spectrum with known peaks."""
        extractor = PeakFeatureExtractor(n_peaks=5, min_height=0.1)
        fv = extractor.extract(synthetic_spectrum)

        # Should find at least 3 peaks
        n_peaks_found = fv.metadata.get("n_peaks_found", 0)
        assert n_peaks_found >= 3

    def test_feature_vector_structure(self, synthetic_spectrum):
        """Feature vector has expected structure."""
        n_peaks = 5
        extractor = PeakFeatureExtractor(n_peaks=n_peaks)
        fv = extractor.extract(synthetic_spectrum)

        # Should have: 1 (n_peaks) + 3 * n_peaks (position, height, width per peak)
        expected_features = 1 + 3 * n_peaks
        assert len(fv.values) == expected_features
        assert len(fv.names) == expected_features

    def test_normalized_positions(self, synthetic_spectrum):
        """Peak positions are normalized to [0, 1]."""
        extractor = PeakFeatureExtractor(n_peaks=5)
        fv = extractor.extract(synthetic_spectrum)

        # Position features should be in [0, 1]
        d = fv.to_dict()
        for key, value in d.items():
            if "position" in key:
                assert 0 <= value <= 1

    def test_flat_spectrum(self, flat_spectrum):
        """Handles flat spectrum gracefully."""
        extractor = PeakFeatureExtractor(n_peaks=5)
        fv = extractor.extract(flat_spectrum)

        # Should still produce valid feature vector
        assert len(fv.values) > 0
        assert not np.any(np.isnan(fv.values))

    def test_n_peaks_limit(self, synthetic_spectrum):
        """Respects n_peaks limit."""
        extractor = PeakFeatureExtractor(n_peaks=2)
        fv = extractor.extract(synthetic_spectrum)

        # Structure should match n_peaks=2
        expected_features = 1 + 3 * 2
        assert len(fv.values) == expected_features


class TestStatisticalFeatureExtractor:
    """Tests for StatisticalFeatureExtractor."""

    def test_basic_extraction(self, synthetic_spectrum):
        """Extract statistical features from spectrum."""
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(synthetic_spectrum)

        # Check expected features exist
        d = fv.to_dict()
        assert "mean" in d
        assert "std" in d
        assert "min" in d
        assert "max" in d
        assert "range" in d

    def test_percentiles(self, synthetic_spectrum):
        """Extracts specified percentiles."""
        extractor = StatisticalFeatureExtractor(percentiles=[10, 50, 90])
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert "percentile_10" in d
        assert "percentile_50" in d
        assert "percentile_90" in d

    def test_higher_moments(self, synthetic_spectrum):
        """Extracts skewness and kurtosis."""
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert "skewness" in d
        assert "kurtosis" in d

    def test_energy_metrics(self, synthetic_spectrum):
        """Extracts energy-related features."""
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert "energy" in d
        assert "rms" in d

    def test_spectral_moments(self, synthetic_spectrum):
        """Extracts spectral centroid and spread."""
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert "centroid" in d
        assert "spread" in d

    def test_values_are_finite(self, synthetic_spectrum):
        """All extracted values are finite."""
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(synthetic_spectrum)

        assert np.all(np.isfinite(fv.values))

    def test_constant_spectrum(self):
        """Handles constant spectrum."""
        spectrum = np.ones(100) * 50.0
        extractor = StatisticalFeatureExtractor()
        fv = extractor.extract(spectrum)

        d = fv.to_dict()
        assert d["mean"] == pytest.approx(50.0)
        assert d["std"] == pytest.approx(0.0)
        assert d["skewness"] == pytest.approx(0.0)


class TestSpectralFeatureExtractor:
    """Tests for SpectralFeatureExtractor."""

    def test_combines_features(self, synthetic_spectrum):
        """Combines peak and statistical features."""
        extractor = SpectralFeatureExtractor(
            n_peaks=3,
            include_derivatives=False,
            include_stats=True
        )
        fv = extractor.extract(synthetic_spectrum)

        # Should have both peak and stat features
        d = fv.to_dict()
        assert any("peak_" in key for key in d.keys())
        assert any("stat_" in key for key in d.keys())

    def test_derivative_features(self, synthetic_spectrum):
        """Includes derivative-based features when enabled."""
        extractor = SpectralFeatureExtractor(
            n_peaks=2,
            include_derivatives=True,
            include_stats=True
        )
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert any("d1_" in key for key in d.keys())  # First derivative
        assert any("d2_" in key for key in d.keys())  # Second derivative

    def test_no_derivatives(self, synthetic_spectrum):
        """Can exclude derivative features."""
        extractor = SpectralFeatureExtractor(
            n_peaks=2,
            include_derivatives=False
        )
        fv = extractor.extract(synthetic_spectrum)

        d = fv.to_dict()
        assert not any("d1_" in key for key in d.keys())
        assert not any("d2_" in key for key in d.keys())

    def test_transform_batch(self, synthetic_spectrum):
        """Transform multiple spectra at once."""
        spectra = np.vstack([synthetic_spectrum, synthetic_spectrum * 0.5])

        extractor = SpectralFeatureExtractor(n_peaks=3)
        extractor.fit(spectra)
        features = extractor.transform(spectra)

        assert features.shape[0] == 2
        assert features.shape[1] > 0


class TestExtractPeakFeaturesFunction:
    """Tests for extract_peak_features convenience function."""

    def test_returns_dict(self, synthetic_spectrum):
        """Returns dictionary of features."""
        features = extract_peak_features(synthetic_spectrum)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_n_peaks_parameter(self, synthetic_spectrum):
        """Respects n_peaks parameter."""
        features_3 = extract_peak_features(synthetic_spectrum, n_peaks=3)
        features_5 = extract_peak_features(synthetic_spectrum, n_peaks=5)

        # More peaks = more features
        assert len(features_5) > len(features_3)

    def test_min_height_parameter(self, synthetic_spectrum):
        """Respects min_height parameter."""
        # Very high threshold should find fewer peaks
        features_low = extract_peak_features(synthetic_spectrum, min_height=0.1)
        features_high = extract_peak_features(synthetic_spectrum, min_height=0.9)

        # Check n_peaks found
        assert features_low["n_peaks"] >= features_high["n_peaks"]


class TestFeatureExtractorFitTransform:
    """Tests for fit/transform pattern on extractors."""

    def test_fit_transform(self, synthetic_spectrum):
        """fit_transform works correctly."""
        spectra = np.vstack([synthetic_spectrum] * 5)

        extractor = SpectralFeatureExtractor(n_peaks=3)
        features = extractor.fit_transform(spectra)

        assert features.shape[0] == 5
        assert extractor._is_fitted

    def test_feature_names_stored(self, synthetic_spectrum):
        """Feature names are stored after fitting."""
        spectra = np.vstack([synthetic_spectrum] * 3)

        extractor = SpectralFeatureExtractor(n_peaks=2)
        extractor.fit_transform(spectra)

        assert len(extractor.feature_names_) > 0
