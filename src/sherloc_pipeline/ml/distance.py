"""
Distance and similarity metrics for spectral comparison.

This module provides various distance/similarity metrics optimized for
comparing SHERLOC spectra, including cosine similarity, Euclidean distance,
and correlation-based measures.

Functions:
    cosine_similarity: Cosine similarity between spectra
    cosine_distance: 1 - cosine_similarity
    euclidean_distance: L2 norm distance
    correlation_distance: 1 - Pearson correlation
    manhattan_distance: L1 norm distance
    spectral_angle_distance: Spectral angle mapper (SAM)
    compute_pairwise_distances: Distance matrix computation

Example:
    >>> import numpy as np
    >>> from sherloc_pipeline.ml.distance import cosine_similarity, euclidean_distance
    >>>
    >>> spectrum1 = np.array([100, 150, 200, 180, 120])
    >>> spectrum2 = np.array([110, 145, 195, 175, 125])
    >>>
    >>> similarity = cosine_similarity(spectrum1, spectrum2)
    >>> print(f"Cosine similarity: {similarity:.4f}")
"""

from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike


class DistanceMetric(str, Enum):
    """Available distance metric types.

    Attributes:
        COSINE: Cosine distance (1 - cosine_similarity)
        EUCLIDEAN: Euclidean (L2) distance
        CORRELATION: Correlation distance (1 - Pearson r)
        MANHATTAN: Manhattan (L1) distance
        SPECTRAL_ANGLE: Spectral angle distance
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    CORRELATION = "correlation"
    MANHATTAN = "manhattan"
    SPECTRAL_ANGLE = "spectral_angle"


def _validate_vectors(
    v1: ArrayLike,
    v2: ArrayLike,
    allow_different_lengths: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert input vectors.

    Args:
        v1: First vector
        v2: Second vector
        allow_different_lengths: If False, vectors must have same length

    Returns:
        Tuple of numpy arrays

    Raises:
        ValueError: If vectors have incompatible shapes
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError(
            f"Vectors must be 1-dimensional. Got shapes {v1.shape} and {v2.shape}"
        )

    if not allow_different_lengths and len(v1) != len(v2):
        raise ValueError(
            f"Vectors must have same length. Got {len(v1)} and {len(v2)}"
        )

    return v1, v2


def cosine_similarity(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction). For non-negative
    spectra, values range from 0 to 1.

    Formula: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Cosine similarity value in [-1, 1]

    Raises:
        ValueError: If either vector is zero-length or all zeros

    Example:
        >>> spectrum1 = [100, 150, 200]
        >>> spectrum2 = [100, 150, 200]
        >>> cosine_similarity(spectrum1, spectrum2)
        1.0
    """
    v1, v2 = _validate_vectors(v1, v2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute cosine similarity for zero vectors")

    return float(np.dot(v1, v2) / (norm1 * norm2))


def cosine_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute cosine distance between two vectors.

    Cosine distance is 1 - cosine_similarity, giving a distance metric
    that ranges from 0 (identical) to 2 (opposite).

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Cosine distance value in [0, 2]

    Example:
        >>> spectrum1 = [100, 150, 200]
        >>> spectrum2 = [100, 150, 200]
        >>> cosine_distance(spectrum1, spectrum2)
        0.0
    """
    return 1.0 - cosine_similarity(v1, v2)


def euclidean_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    The Euclidean distance is the straight-line distance between
    two points in n-dimensional space.

    Formula: sqrt(sum((v1 - v2)^2))

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Euclidean distance (non-negative)

    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    v1, v2 = _validate_vectors(v1, v2)
    return float(np.linalg.norm(v1 - v2))


def manhattan_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute Manhattan (L1) distance between two vectors.

    The Manhattan distance is the sum of absolute differences,
    also known as taxicab or city-block distance.

    Formula: sum(|v1 - v2|)

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Manhattan distance (non-negative)

    Example:
        >>> manhattan_distance([0, 0], [3, 4])
        7.0
    """
    v1, v2 = _validate_vectors(v1, v2)
    return float(np.sum(np.abs(v1 - v2)))


def correlation_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute correlation distance between two vectors.

    Correlation distance is 1 - Pearson correlation coefficient.
    Values range from 0 (perfect correlation) to 2 (perfect anti-correlation).

    This metric is invariant to linear scaling and offset, making it
    useful for comparing spectral shapes regardless of intensity.

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Correlation distance in [0, 2]

    Raises:
        ValueError: If either vector has zero variance

    Example:
        >>> correlation_distance([1, 2, 3], [2, 4, 6])
        0.0  # Perfect correlation
    """
    v1, v2 = _validate_vectors(v1, v2)

    # Compute Pearson correlation
    v1_centered = v1 - np.mean(v1)
    v2_centered = v2 - np.mean(v2)

    std1 = np.std(v1)
    std2 = np.std(v2)

    if std1 == 0 or std2 == 0:
        raise ValueError("Cannot compute correlation for constant vectors")

    correlation = np.dot(v1_centered, v2_centered) / (len(v1) * std1 * std2)
    return float(1.0 - correlation)


def spectral_angle_distance(v1: ArrayLike, v2: ArrayLike) -> float:
    """Compute Spectral Angle Mapper (SAM) distance.

    SAM measures the angle between two spectra in n-dimensional space,
    treating each spectrum as a vector. The result is in radians [0, pi].

    This metric is invariant to overall intensity scaling, making it
    robust for comparing spectra with different brightness levels.

    Formula: arccos(cosine_similarity)

    Args:
        v1: First spectrum/vector
        v2: Second spectrum/vector

    Returns:
        Spectral angle in radians [0, pi]

    Example:
        >>> spectral_angle_distance([1, 0], [0, 1])  # 90 degrees
        1.5707963267948966  # pi/2 radians
    """
    similarity = cosine_similarity(v1, v2)
    # Clip to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(np.arccos(similarity))


def get_distance_function(metric: Union[str, DistanceMetric]) -> Callable:
    """Get distance function by name or enum.

    Args:
        metric: Distance metric name or DistanceMetric enum

    Returns:
        Distance function

    Raises:
        ValueError: If metric is not recognized
    """
    if isinstance(metric, str):
        metric = DistanceMetric(metric.lower())

    metric_functions = {
        DistanceMetric.COSINE: cosine_distance,
        DistanceMetric.EUCLIDEAN: euclidean_distance,
        DistanceMetric.CORRELATION: correlation_distance,
        DistanceMetric.MANHATTAN: manhattan_distance,
        DistanceMetric.SPECTRAL_ANGLE: spectral_angle_distance,
    }

    if metric not in metric_functions:
        raise ValueError(f"Unknown distance metric: {metric}")

    return metric_functions[metric]


def compute_pairwise_distances(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
) -> np.ndarray:
    """Compute pairwise distance matrix between samples.

    If Y is None, computes distances between all pairs of samples in X.
    If Y is provided, computes distances between samples in X and Y.

    Args:
        X: First set of samples, shape (n_samples_X, n_features)
        Y: Optional second set of samples, shape (n_samples_Y, n_features)
        metric: Distance metric to use

    Returns:
        Distance matrix of shape (n_samples_X, n_samples_X) or
        (n_samples_X, n_samples_Y) if Y is provided

    Example:
        >>> X = np.array([[0, 0], [1, 0], [0, 1]])
        >>> distances = compute_pairwise_distances(X, metric="euclidean")
        >>> distances.shape
        (3, 3)
        >>> distances[0, 0]  # Self-distance is 0
        0.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    symmetric = Y is None

    if symmetric:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))

    n_x = X.shape[0]
    n_y = Y.shape[0]

    distance_func = get_distance_function(metric)

    # Pre-allocate distance matrix
    distances = np.zeros((n_x, n_y), dtype=np.float64)

    for i in range(n_x):
        start_j = i if symmetric else 0
        for j in range(start_j, n_y):
            dist = distance_func(X[i], Y[j])
            distances[i, j] = dist
            if symmetric and i != j:
                distances[j, i] = dist

    return distances
