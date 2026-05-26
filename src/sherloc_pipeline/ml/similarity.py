"""
Spectral similarity comparison module.

This module provides high-level APIs for comparing SHERLOC spectra,
optimized for batch processing of large datasets (>10k spectra).

Features:
    - Vectorized similarity computations using numpy broadcasting
    - Memory-efficient chunked processing for large datasets
    - Multiple similarity metrics (cosine, euclidean, correlation)
    - k-nearest neighbor search functionality
    - Similarity matrix computation with optional filtering

Classes:
    SimilarityResult: Container for similarity search results
    SpectralSimilarity: Main class for batch similarity operations
    SimilarityConfig: Configuration for similarity computations

Functions:
    batch_cosine_similarity: Vectorized cosine similarity
    batch_euclidean_distance: Vectorized Euclidean distance
    batch_correlation: Vectorized Pearson correlation
    find_similar_spectra: Find k most similar spectra

Example:
    >>> import numpy as np
    >>> from sherloc_pipeline.ml.similarity import SpectralSimilarity
    >>>
    >>> # Create similarity engine
    >>> similarity = SpectralSimilarity(metric="cosine")
    >>>
    >>> # Build index from reference spectra
    >>> reference = np.random.randn(1000, 500)
    >>> similarity.fit(reference)
    >>>
    >>> # Find similar spectra for queries
    >>> queries = np.random.randn(10, 500)
    >>> results = similarity.find_similar(queries, k=5)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sherloc_pipeline.ml.base import MLBaseModel, Transformer
from sherloc_pipeline.ml.distance import DistanceMetric


class SimilarityMetric(str, Enum):
    """Similarity metric types for spectral comparison.

    Attributes:
        COSINE: Cosine similarity (dot product of normalized vectors)
        CORRELATION: Pearson correlation coefficient
        EUCLIDEAN: Negative Euclidean distance (higher = more similar)
        SPECTRAL_ANGLE: Negative spectral angle (higher = more similar)
    """

    COSINE = "cosine"
    CORRELATION = "correlation"
    EUCLIDEAN = "euclidean"
    SPECTRAL_ANGLE = "spectral_angle"


@dataclass
class SimilarityResult:
    """Container for similarity search results.

    Attributes:
        indices: Indices of similar spectra in reference set
        scores: Similarity scores (higher = more similar)
        distances: Optional distance values (lower = more similar)
        metadata: Additional result metadata
    """

    indices: np.ndarray
    scores: np.ndarray
    distances: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of query results."""
        return len(self.indices)

    @property
    def top_matches(self) -> List[Tuple[int, float]]:
        """Get list of (index, score) tuples for top matches."""
        if self.indices.ndim == 1:
            return list(zip(self.indices.tolist(), self.scores.tolist()))
        # For multiple queries, return first query's results
        return list(zip(self.indices[0].tolist(), self.scores[0].tolist()))


class SimilarityConfig(MLBaseModel):
    """Configuration for spectral similarity computations.

    Attributes:
        metric: Similarity metric to use
        chunk_size: Batch size for chunked processing (memory control)
        normalize: Whether to L2-normalize spectra before comparison
        min_similarity: Minimum similarity threshold for filtering
        n_jobs: Number of parallel workers (1 = sequential)

    Example:
        >>> config = SimilarityConfig(metric="cosine", chunk_size=1000)
        >>> similarity = SpectralSimilarity(config=config)
    """

    metric: SimilarityMetric = SimilarityMetric.COSINE
    chunk_size: int = 2000  # Process 2000 spectra at a time
    normalize: bool = True
    min_similarity: Optional[float] = None
    n_jobs: int = 1

    @property
    def metric_enum(self) -> SimilarityMetric:
        """Get metric as SimilarityMetric enum (handles use_enum_values=True)."""
        if isinstance(self.metric, SimilarityMetric):
            return self.metric
        return SimilarityMetric(self.metric)

    @property
    def metric_value(self) -> str:
        """Get metric value as string."""
        if isinstance(self.metric, SimilarityMetric):
            return self.metric.value
        return str(self.metric)



def _normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row of a matrix.

    Args:
        X: Input matrix of shape (n_samples, n_features)
        eps: Small epsilon to avoid division by zero

    Returns:
        Row-normalized matrix with same shape
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)  # Avoid division by zero
    return X / norms


def _center_rows(X: np.ndarray) -> np.ndarray:
    """Center each row to zero mean.

    Args:
        X: Input matrix of shape (n_samples, n_features)

    Returns:
        Row-centered matrix with same shape
    """
    return X - X.mean(axis=1, keepdims=True)


def batch_cosine_similarity(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute cosine similarity matrix using vectorized operations.

    Efficiently computes all pairwise cosine similarities between
    spectra in X and Y (or X and X if Y is None).

    Formula: cos(X, Y) = (X @ Y.T) / (||X|| * ||Y||)

    Args:
        X: Query spectra of shape (n_queries, n_features)
        Y: Reference spectra of shape (n_refs, n_features), or None for self-comparison
        normalize: If True, L2-normalize vectors (set False if already normalized)

    Returns:
        Similarity matrix of shape (n_queries, n_refs) with values in [-1, 1]

    Example:
        >>> X = np.array([[1, 0, 0], [0, 1, 0]])
        >>> Y = np.array([[1, 0, 0], [0, 0, 1]])
        >>> sim = batch_cosine_similarity(X, Y)
        >>> sim[0, 0]  # X[0] identical to Y[0]
        1.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))

    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))

    if normalize:
        X = _normalize_rows(X)
        Y = _normalize_rows(Y)

    # Matrix multiplication gives all pairwise dot products
    similarity = X @ Y.T

    # Clip to handle numerical errors
    return np.clip(similarity, -1.0, 1.0)


def batch_euclidean_distance(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute Euclidean distance matrix using vectorized operations.

    Uses the identity: ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2*X@Y.T

    Args:
        X: Query spectra of shape (n_queries, n_features)
        Y: Reference spectra of shape (n_refs, n_features), or None for self-comparison

    Returns:
        Distance matrix of shape (n_queries, n_refs) with non-negative values

    Example:
        >>> X = np.array([[0, 0], [3, 4]])
        >>> distances = batch_euclidean_distance(X)
        >>> distances[0, 1]  # Distance from origin to (3,4)
        5.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))

    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))

    # Use ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2*X@Y.T
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n_queries, 1)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T  # (1, n_refs)
    XY = X @ Y.T  # (n_queries, n_refs)

    # Compute squared distances
    sq_distances = X_sq + Y_sq - 2 * XY

    # Handle numerical errors (small negative values)
    sq_distances = np.maximum(sq_distances, 0.0)

    return np.sqrt(sq_distances)


def batch_correlation(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute Pearson correlation matrix using vectorized operations.

    Pearson correlation measures linear relationship between spectra,
    independent of scale and offset. Good for comparing spectral shapes.

    Formula: r = cov(X, Y) / (std(X) * std(Y))

    Args:
        X: Query spectra of shape (n_queries, n_features)
        Y: Reference spectra of shape (n_refs, n_features), or None for self-comparison

    Returns:
        Correlation matrix of shape (n_queries, n_refs) with values in [-1, 1]

    Example:
        >>> X = np.array([[1, 2, 3]])
        >>> Y = np.array([[2, 4, 6]])  # Perfectly correlated
        >>> corr = batch_correlation(X, Y)
        >>> corr[0, 0]
        1.0
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))

    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))

    # Center the data
    X_centered = _center_rows(X)
    Y_centered = _center_rows(Y)

    # Normalize to unit variance (this makes dot product = correlation)
    X_norm = _normalize_rows(X_centered)
    Y_norm = _normalize_rows(Y_centered)

    # Matrix multiplication gives correlations
    correlation = X_norm @ Y_norm.T

    return np.clip(correlation, -1.0, 1.0)


def batch_spectral_angle(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Compute spectral angle distances using vectorized operations.

    Spectral Angle Mapper (SAM) measures the angle between spectra,
    invariant to intensity scaling. Result is in radians [0, pi].

    Args:
        X: Query spectra of shape (n_queries, n_features)
        Y: Reference spectra of shape (n_refs, n_features)

    Returns:
        Angle matrix of shape (n_queries, n_refs) in radians [0, pi]
    """
    cos_sim = batch_cosine_similarity(X, Y, normalize=True)
    return np.arccos(cos_sim)


def find_similar_spectra(
    query: ArrayLike,
    reference: ArrayLike,
    k: int = 5,
    metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
    return_distances: bool = False,
) -> SimilarityResult:
    """Find the k most similar spectra in a reference set.

    Convenience function for simple similarity searches without
    building a SpectralSimilarity instance.

    Args:
        query: Query spectrum(s) of shape (n_features,) or (n_queries, n_features)
        reference: Reference spectra of shape (n_refs, n_features)
        k: Number of similar spectra to return
        metric: Similarity metric to use
        return_distances: If True, include distance values in result

    Returns:
        SimilarityResult with indices and scores of k most similar spectra

    Example:
        >>> query = np.random.randn(500)
        >>> reference = np.random.randn(1000, 500)
        >>> result = find_similar_spectra(query, reference, k=5)
        >>> print(f"Top match: index {result.indices[0]}, score {result.scores[0]:.3f}")
    """
    query = np.atleast_2d(np.asarray(query, dtype=np.float64))
    reference = np.atleast_2d(np.asarray(reference, dtype=np.float64))

    if isinstance(metric, str):
        metric = SimilarityMetric(metric.lower())

    # Compute similarities/distances
    if metric == SimilarityMetric.COSINE:
        similarities = batch_cosine_similarity(query, reference)
    elif metric == SimilarityMetric.CORRELATION:
        similarities = batch_correlation(query, reference)
    elif metric == SimilarityMetric.EUCLIDEAN:
        distances = batch_euclidean_distance(query, reference)
        # Convert distance to similarity (negative distance)
        similarities = -distances
    elif metric == SimilarityMetric.SPECTRAL_ANGLE:
        angles = batch_spectral_angle(query, reference)
        # Convert angle to similarity (negative angle)
        similarities = -angles
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Find top-k for each query
    k = min(k, reference.shape[0])

    if query.shape[0] == 1:
        # Single query - return 1D arrays
        top_indices = np.argsort(similarities[0])[-k:][::-1]
        top_scores = similarities[0, top_indices]
        distances = -top_scores if metric in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.SPECTRAL_ANGLE) else None
    else:
        # Multiple queries - return 2D arrays
        top_indices = np.zeros((query.shape[0], k), dtype=np.int64)
        top_scores = np.zeros((query.shape[0], k), dtype=np.float64)

        for i in range(query.shape[0]):
            idx = np.argsort(similarities[i])[-k:][::-1]
            top_indices[i] = idx
            top_scores[i] = similarities[i, idx]

        distances = -top_scores if metric in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.SPECTRAL_ANGLE) else None

    return SimilarityResult(
        indices=top_indices,
        scores=top_scores,
        distances=distances if return_distances else None,
        metadata={"metric": metric.value, "k": k, "n_queries": query.shape[0]},
    )


class SpectralSimilarity(Transformer):
    """Batch spectral similarity engine optimized for large datasets.

    Provides efficient similarity search and matrix computation for
    datasets with >10k spectra using chunked processing.

    Features:
        - Builds normalized reference index for fast queries
        - Memory-efficient chunked processing for large datasets
        - Multiple similarity metrics with unified interface
        - Configurable thresholding and filtering

    Attributes:
        config: SimilarityConfig with computation parameters
        reference_: Fitted reference spectra (normalized if applicable)
        n_features_: Number of features per spectrum

    Example:
        >>> similarity = SpectralSimilarity(metric="cosine", chunk_size=1000)
        >>> similarity.fit(reference_spectra)  # 10000 x 500 matrix
        >>> results = similarity.find_similar(query_spectra, k=10)
    """

    def __init__(
        self,
        metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE,
        chunk_size: int = 2000,
        normalize: bool = True,
        min_similarity: Optional[float] = None,
        config: Optional[SimilarityConfig] = None,
    ):
        """Initialize spectral similarity engine.

        Args:
            metric: Similarity metric to use
            chunk_size: Batch size for chunked processing
            normalize: Whether to L2-normalize spectra
            min_similarity: Minimum similarity threshold
            config: Optional full configuration (overrides other args)
        """
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = SimilarityConfig(
                metric=metric,
                chunk_size=chunk_size,
                normalize=normalize,
                min_similarity=min_similarity,
            )

        self.reference_: Optional[np.ndarray] = None
        self.reference_centered_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> "SpectralSimilarity":
        """Build reference index from spectra.

        Preprocesses and stores reference spectra for efficient similarity
        queries. For large datasets, considers memory-efficient storage.

        Args:
            X: Reference spectra of shape (n_samples, n_features)
            y: Ignored (for API compatibility)

        Returns:
            self: Fitted similarity engine
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.n_samples_, self.n_features_ = X.shape

        # Preprocess reference for efficient similarity computation
        if self.config.normalize and self.config.metric_enum in (
            SimilarityMetric.COSINE,
            SimilarityMetric.SPECTRAL_ANGLE,
        ):
            self.reference_ = _normalize_rows(X)
        else:
            self.reference_ = X.copy()

        # For correlation, also store centered version
        if self.config.metric_enum == SimilarityMetric.CORRELATION:
            centered = _center_rows(X)
            self.reference_centered_ = _normalize_rows(centered)

        self._is_fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Compute similarity matrix between X and reference.

        For large datasets, uses chunked processing to manage memory.

        Args:
            X: Query spectra of shape (n_queries, n_features)

        Returns:
            Similarity matrix of shape (n_queries, n_reference)
        """
        self._check_is_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Query has {X.shape[1]} features, expected {self.n_features_}"
            )

        return self._compute_similarity_matrix(X)

    def _compute_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute similarity matrix with optional chunking.

        Args:
            X: Query spectra

        Returns:
            Similarity matrix
        """
        n_queries = X.shape[0]
        n_refs = self.n_samples_

        # Decide whether to use chunking
        # Chunk if total matrix size exceeds threshold (~100M elements)
        matrix_size = n_queries * n_refs
        use_chunking = matrix_size > 100_000_000

        if not use_chunking:
            return self._compute_similarity_batch(X)

        # Chunked computation
        chunk_size = self.config.chunk_size
        similarity = np.zeros((n_queries, n_refs), dtype=np.float64)

        for i in range(0, n_queries, chunk_size):
            end_i = min(i + chunk_size, n_queries)
            X_chunk = X[i:end_i]
            similarity[i:end_i] = self._compute_similarity_batch(X_chunk)

        return similarity

    def _compute_similarity_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute similarity for a batch of queries.

        Args:
            X: Query batch

        Returns:
            Similarity scores for the batch
        """
        metric = self.config.metric_enum

        if metric == SimilarityMetric.COSINE:
            if self.config.normalize:
                X_norm = _normalize_rows(X)
                return X_norm @ self.reference_.T
            return batch_cosine_similarity(X, self.reference_, normalize=True)

        elif metric == SimilarityMetric.CORRELATION:
            X_centered = _center_rows(X)
            X_norm = _normalize_rows(X_centered)
            return X_norm @ self.reference_centered_.T

        elif metric == SimilarityMetric.EUCLIDEAN:
            distances = batch_euclidean_distance(X, self.reference_)
            return -distances  # Negate for similarity ordering

        elif metric == SimilarityMetric.SPECTRAL_ANGLE:
            if self.config.normalize:
                X_norm = _normalize_rows(X)
                cos_sim = X_norm @ self.reference_.T
            else:
                cos_sim = batch_cosine_similarity(X, self.reference_)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            return -np.arccos(cos_sim)  # Negate for similarity ordering

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_similar(
        self,
        X: ArrayLike,
        k: int = 5,
        threshold: Optional[float] = None,
    ) -> SimilarityResult:
        """Find k most similar reference spectra for each query.

        Efficiently finds nearest neighbors using the fitted reference index.
        For very large k or when threshold is used, computes full similarity
        matrix first.

        Args:
            X: Query spectra of shape (n_features,) or (n_queries, n_features)
            k: Number of similar spectra to return per query
            threshold: Optional similarity threshold (overrides config)

        Returns:
            SimilarityResult with indices and scores of similar spectra

        Example:
            >>> results = similarity.find_similar(query, k=10)
            >>> for idx, score in results.top_matches:
            ...     print(f"Match: {idx}, Score: {score:.4f}")
        """
        self._check_is_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Query has {X.shape[1]} features, expected {self.n_features_}"
            )

        k = min(k, self.n_samples_)
        threshold = threshold if threshold is not None else self.config.min_similarity

        # Compute similarities
        similarities = self._compute_similarity_matrix(X)

        # Find top-k for each query
        n_queries = X.shape[0]

        if n_queries == 1:
            # Single query optimization
            sims = similarities[0]

            if threshold is not None:
                # Filter by threshold first
                valid_mask = sims >= threshold
                valid_indices = np.where(valid_mask)[0]
                valid_sims = sims[valid_mask]
                sorted_order = np.argsort(valid_sims)[-k:][::-1]
                top_indices = valid_indices[sorted_order]
                top_scores = valid_sims[sorted_order]
            else:
                top_indices = np.argsort(sims)[-k:][::-1]
                top_scores = sims[top_indices]

        else:
            # Multiple queries
            top_indices = np.zeros((n_queries, k), dtype=np.int64)
            top_scores = np.zeros((n_queries, k), dtype=np.float64)

            for i in range(n_queries):
                sims = similarities[i]

                if threshold is not None:
                    valid_mask = sims >= threshold
                    valid_indices = np.where(valid_mask)[0]
                    valid_sims = sims[valid_mask]
                    actual_k = min(k, len(valid_indices))
                    sorted_order = np.argsort(valid_sims)[-actual_k:][::-1]

                    top_indices[i, :actual_k] = valid_indices[sorted_order]
                    top_scores[i, :actual_k] = valid_sims[sorted_order]
                    # Pad with -1 for indices and 0 for scores if fewer than k
                    if actual_k < k:
                        top_indices[i, actual_k:] = -1
                else:
                    idx = np.argsort(sims)[-k:][::-1]
                    top_indices[i] = idx
                    top_scores[i] = sims[idx]

        # Compute distances if using distance-based metric
        if self.config.metric_enum in (SimilarityMetric.EUCLIDEAN, SimilarityMetric.SPECTRAL_ANGLE):
            distances = -top_scores
        else:
            distances = None

        return SimilarityResult(
            indices=top_indices,
            scores=top_scores,
            distances=distances,
            metadata={
                "metric": self.config.metric_value,
                "k": k,
                "n_queries": n_queries,
                "threshold": threshold,
            },
        )

    def pairwise_similarity(
        self,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        """Compute full pairwise similarity matrix for reference set.

        Memory-efficient computation for large datasets using chunking.

        Args:
            chunk_size: Override config chunk_size for this computation

        Returns:
            Symmetric similarity matrix of shape (n_reference, n_reference)

        Example:
            >>> sim_matrix = similarity.pairwise_similarity()
            >>> np.allclose(sim_matrix, sim_matrix.T)  # Symmetric
            True
        """
        self._check_is_fitted()

        chunk_size = chunk_size or self.config.chunk_size
        n = self.n_samples_

        # Pre-allocate result matrix
        similarity = np.zeros((n, n), dtype=np.float64)

        # Compute in chunks, exploiting symmetry
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)

            for j in range(i, n, chunk_size):
                end_j = min(j + chunk_size, n)

                # Compute chunk
                if self.config.metric_enum == SimilarityMetric.COSINE:
                    chunk = self.reference_[i:end_i] @ self.reference_[j:end_j].T
                elif self.config.metric_enum == SimilarityMetric.CORRELATION:
                    chunk = self.reference_centered_[i:end_i] @ self.reference_centered_[j:end_j].T
                elif self.config.metric_enum == SimilarityMetric.EUCLIDEAN:
                    chunk = -batch_euclidean_distance(
                        self.reference_[i:end_i],
                        self.reference_[j:end_j]
                    )
                elif self.config.metric_enum == SimilarityMetric.SPECTRAL_ANGLE:
                    cos_chunk = self.reference_[i:end_i] @ self.reference_[j:end_j].T
                    cos_chunk = np.clip(cos_chunk, -1.0, 1.0)
                    chunk = -np.arccos(cos_chunk)
                else:
                    raise ValueError(f"Unknown metric: {self.config.metric_enum}")

                similarity[i:end_i, j:end_j] = chunk

                # Copy to symmetric position if not on diagonal
                if i != j:
                    similarity[j:end_j, i:end_i] = chunk.T

        return similarity

    def similarity_histogram(
        self,
        bins: int = 50,
        sample_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute histogram of pairwise similarities.

        Useful for understanding the distribution of similarities in the
        reference set and setting appropriate thresholds.

        Args:
            bins: Number of histogram bins
            sample_size: If set, use random sampling for large datasets

        Returns:
            Tuple of (counts, bin_edges) like np.histogram
        """
        self._check_is_fitted()

        n = self.n_samples_

        if sample_size is not None and n > sample_size:
            # Random sampling for large datasets
            indices = np.random.choice(n, sample_size, replace=False)
            subset = self.reference_[indices]

            if self.config.metric_enum == SimilarityMetric.CORRELATION:
                subset_centered = self.reference_centered_[indices]
                sim_matrix = subset_centered @ subset_centered.T
            else:
                sim_matrix = self._compute_similarity_batch(subset)
        else:
            sim_matrix = self.pairwise_similarity()

        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(sim_matrix.shape[0], k=1)
        similarities = sim_matrix[upper_indices]

        return np.histogram(similarities, bins=bins)

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters for JSON serialization."""
        return {
            "class": self.__class__.__name__,
            "metric": self.config.metric_value,
            "chunk_size": self.config.chunk_size,
            "normalize": self.config.normalize,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
        }
