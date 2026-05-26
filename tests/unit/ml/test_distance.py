"""
Unit tests for ML distance metrics.

Tests the distance/similarity metrics defined in ml/distance.py:
- Cosine similarity/distance
- Euclidean distance
- Correlation distance
- Manhattan distance
- Spectral angle distance
- Pairwise distance computation
"""

import numpy as np
import pytest

from sherloc_pipeline.ml.distance import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    manhattan_distance,
    correlation_distance,
    spectral_angle_distance,
    compute_pairwise_distances,
    get_distance_function,
    DistanceMetric,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([-1.0, -2.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_scaled_vectors(self):
        """Cosine similarity is invariant to scaling."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([10.0, 20.0, 30.0])
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

    def test_symmetric(self):
        """Cosine similarity is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        assert cosine_similarity(v1, v2) == pytest.approx(cosine_similarity(v2, v1))

    def test_accepts_lists(self):
        """Accepts Python lists as input."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        result = cosine_similarity(v1, v2)
        assert 0 <= result <= 1

    def test_zero_vector_raises(self):
        """Zero vector raises ValueError."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero vectors"):
            cosine_similarity(v1, v2)

    def test_different_lengths_raises(self):
        """Different length vectors raise ValueError."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity(v1, v2)


class TestCosineDistance:
    """Tests for cosine_distance function."""

    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have distance 1.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_distance(v1, v2) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        """Opposite vectors have distance 2.0."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_distance(v1, v2) == pytest.approx(2.0)


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        """Test with known 3-4-5 triangle."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)

    def test_1d_distance(self):
        """Test 1D distance."""
        v1 = np.array([0.0])
        v2 = np.array([5.0])
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)

    def test_symmetric(self):
        """Euclidean distance is symmetric."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        assert euclidean_distance(v1, v2) == pytest.approx(euclidean_distance(v2, v1))

    def test_non_negative(self):
        """Euclidean distance is always non-negative."""
        v1 = np.array([1.0, -2.0, 3.0])
        v2 = np.array([-4.0, 5.0, -6.0])
        assert euclidean_distance(v1, v2) >= 0


class TestManhattanDistance:
    """Tests for manhattan_distance function."""

    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert manhattan_distance(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        """Test with known distance."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert manhattan_distance(v1, v2) == pytest.approx(7.0)

    def test_symmetric(self):
        """Manhattan distance is symmetric."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([4.0, 5.0])
        assert manhattan_distance(v1, v2) == pytest.approx(manhattan_distance(v2, v1))


class TestCorrelationDistance:
    """Tests for correlation_distance function."""

    def test_identical_vectors(self):
        """Identical vectors have distance 0.0."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert correlation_distance(v, v) == pytest.approx(0.0)

    def test_perfect_correlation(self):
        """Linearly scaled vectors have distance 0.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([2.0, 4.0, 6.0])
        assert correlation_distance(v1, v2) == pytest.approx(0.0)

    def test_linear_transform_invariance(self):
        """Correlation is invariant to linear transformation."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([10.0 + 5 * 1.0, 10.0 + 5 * 2.0, 10.0 + 5 * 3.0])
        assert correlation_distance(v1, v2) == pytest.approx(0.0)

    def test_negative_correlation(self):
        """Negatively correlated vectors have distance near 2.0."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([3.0, 2.0, 1.0])
        assert correlation_distance(v1, v2) == pytest.approx(2.0)

    def test_constant_vector_raises(self):
        """Constant vector raises ValueError."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="constant vectors"):
            correlation_distance(v1, v2)


class TestSpectralAngleDistance:
    """Tests for spectral_angle_distance function."""

    def test_identical_vectors(self):
        """Identical vectors have angle 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert spectral_angle_distance(v, v) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have angle pi/2."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert spectral_angle_distance(v1, v2) == pytest.approx(np.pi / 2)

    def test_opposite_vectors(self):
        """Opposite vectors have angle pi."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert spectral_angle_distance(v1, v2) == pytest.approx(np.pi)

    def test_scale_invariance(self):
        """Spectral angle is invariant to scaling."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([100.0, 200.0, 300.0])
        assert spectral_angle_distance(v1, v2) == pytest.approx(0.0)


class TestDistanceMetricEnum:
    """Tests for DistanceMetric enum."""

    def test_enum_values(self):
        """Enum has expected values."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.CORRELATION.value == "correlation"
        assert DistanceMetric.MANHATTAN.value == "manhattan"
        assert DistanceMetric.SPECTRAL_ANGLE.value == "spectral_angle"

    def test_get_distance_function(self):
        """get_distance_function returns correct functions."""
        assert get_distance_function("cosine") == cosine_distance
        assert get_distance_function("euclidean") == euclidean_distance
        assert get_distance_function(DistanceMetric.MANHATTAN) == manhattan_distance


class TestComputePairwiseDistances:
    """Tests for compute_pairwise_distances function."""

    def test_self_distances_zero(self):
        """Diagonal of distance matrix should be zero."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        distances = compute_pairwise_distances(X)
        assert np.allclose(np.diag(distances), 0.0)

    def test_symmetric_matrix(self):
        """Distance matrix should be symmetric."""
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        distances = compute_pairwise_distances(X)
        assert np.allclose(distances, distances.T)

    def test_correct_shape(self):
        """Output shape matches input samples."""
        X = np.array([[0.0] * 10 for _ in range(5)])
        distances = compute_pairwise_distances(X)
        assert distances.shape == (5, 5)

    def test_known_distances(self):
        """Test with known Euclidean distances."""
        X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        distances = compute_pairwise_distances(X, metric="euclidean")
        assert distances[0, 1] == pytest.approx(3.0)
        assert distances[0, 2] == pytest.approx(4.0)
        assert distances[1, 2] == pytest.approx(5.0)

    def test_with_second_array(self):
        """Test pairwise distances between two arrays."""
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        Y = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
        distances = compute_pairwise_distances(X, Y, metric="euclidean")
        assert distances.shape == (2, 3)
        assert distances[0, 0] == pytest.approx(1.0)  # (0,0) to (0,1)
        assert distances[1, 2] == pytest.approx(1.0)  # (1,0) to (2,0)

    def test_different_metrics(self):
        """Different metrics produce different results."""
        X = np.array([[1.0, 2.0], [2.0, 4.0], [1.0, 3.0]])

        eucl = compute_pairwise_distances(X, metric="euclidean")
        cos = compute_pairwise_distances(X, metric="cosine")

        # Euclidean: scaled vectors have non-zero distance
        assert eucl[0, 1] > 0

        # Cosine: scaled vectors have zero distance
        assert cos[0, 1] == pytest.approx(0.0, abs=1e-10)
