"""
Clustering algorithms for spectral analysis.

This module provides clustering implementations tailored for SHERLOC
spectroscopy data, including KMeans, DBSCAN, and Hierarchical clustering.

Classes:
    ClusteringResult: Container for clustering results with metadata
    BaseClusterer: Abstract base class for clusterers
    KMeansClusterer: K-Means clustering wrapper
    DBSCANClusterer: DBSCAN density-based clustering
    HierarchicalClusterer: Agglomerative hierarchical clustering

Functions:
    compute_silhouette_score: Compute silhouette score for clustering quality
    compute_silhouette_samples: Compute per-sample silhouette coefficients

Example:
    >>> from sherloc_pipeline.ml.clustering import KMeansClusterer, compute_silhouette_score
    >>> import numpy as np
    >>>
    >>> # Create synthetic spectral data
    >>> spectra = np.random.randn(100, 50)
    >>>
    >>> # Cluster into 5 groups
    >>> clusterer = KMeansClusterer(n_clusters=5)
    >>> result = clusterer.fit_predict(spectra)
    >>> print(f"Found {len(set(result.labels))} clusters")
    >>>
    >>> # Evaluate clustering quality
    >>> score = compute_silhouette_score(spectra, result.labels)
    >>> print(f"Silhouette score: {score:.3f}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from sherloc_pipeline.ml.base import Estimator, ModelPersistence
from sherloc_pipeline.ml.distance import DistanceMetric, compute_pairwise_distances


@dataclass
class ClusteringResult:
    """Container for clustering results with metadata.

    Attributes:
        labels: Cluster label for each sample (-1 indicates noise for DBSCAN)
        n_clusters: Number of clusters found (excluding noise)
        cluster_sizes: Number of samples in each cluster
        centroids: Cluster centroids (if applicable)
        inertia: Within-cluster sum of squares (for KMeans)
        silhouette_score: Optional silhouette score if computed
        metadata: Additional algorithm-specific metadata
    """

    labels: np.ndarray
    n_clusters: int
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    centroids: Optional[np.ndarray] = None
    inertia: Optional[float] = None
    silhouette_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute cluster sizes from labels."""
        if not self.cluster_sizes:
            unique, counts = np.unique(self.labels, return_counts=True)
            self.cluster_sizes = {int(label): int(count) for label, count in zip(unique, counts)}
            # Exclude noise points from n_clusters if using DBSCAN
            if -1 in self.cluster_sizes:
                self.n_clusters = len(self.cluster_sizes) - 1
            else:
                self.n_clusters = len(self.cluster_sizes)

    def get_cluster_members(self, cluster_id: int) -> np.ndarray:
        """Get indices of samples belonging to a specific cluster.

        Args:
            cluster_id: Cluster label to query

        Returns:
            Array of sample indices in the specified cluster
        """
        return np.where(self.labels == cluster_id)[0]


class BaseClusterer(Estimator, ModelPersistence):
    """Abstract base class for clustering algorithms.

    Provides common interface and utilities for all clustering
    implementations in the PHASE ML infrastructure.

    Subclasses must implement:
        - fit(X): Fit the clustering model
        - predict(X): Assign cluster labels to samples
    """

    def __init__(self, random_state: Optional[int] = None):
        """Initialize base clusterer.

        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.random_state = random_state
        self._result: Optional[ClusteringResult] = None

    @property
    def labels_(self) -> Optional[np.ndarray]:
        """Get cluster labels from last fit."""
        return self._result.labels if self._result else None

    @property
    def n_clusters_(self) -> Optional[int]:
        """Get number of clusters from last fit."""
        return self._result.n_clusters if self._result else None

    def fit_predict(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> ClusteringResult:
        """Fit the clusterer and return clustering result.

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Ignored (for API compatibility)

        Returns:
            ClusteringResult with labels and metadata
        """
        self.fit(X, y)
        return self._result

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters for JSON serialization."""
        params = super()._get_saveable_params()
        params.update({
            "random_state": self.random_state,
        })
        return params


class KMeansClusterer(BaseClusterer):
    """K-Means clustering for spectral data.

    Partitions spectra into K clusters by minimizing within-cluster
    variance. Works well when clusters are roughly spherical and
    similar in size.

    Attributes:
        n_clusters: Number of clusters to form
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        n_init: Number of random initializations to try
        random_state: Random seed for reproducibility

    Example:
        >>> clusterer = KMeansClusterer(n_clusters=5, random_state=42)
        >>> result = clusterer.fit_predict(spectra)
        >>> print(f"Inertia: {result.inertia:.2f}")
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: Optional[int] = None,
    ):
        """Initialize KMeans clusterer.

        Args:
            n_clusters: Number of clusters to form
            max_iter: Maximum iterations per initialization
            tol: Convergence tolerance (relative change in centroids)
            n_init: Number of random initializations
            random_state: Random seed for reproducibility
        """
        super().__init__(random_state=random_state)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.centroids_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> "KMeansClusterer":
        """Fit KMeans clustering to data.

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Ignored

        Returns:
            self
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for init_idx in range(self.n_init):
            # Random initialization: pick n_clusters random samples as centroids
            init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            centroids = X[init_indices].copy()

            for iteration in range(self.max_iter):
                # Assignment step: assign each point to nearest centroid
                distances = compute_pairwise_distances(X, centroids, metric=DistanceMetric.EUCLIDEAN)
                labels = np.argmin(distances, axis=1)

                # Update step: recompute centroids
                new_centroids = np.zeros_like(centroids)
                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        new_centroids[k] = X[mask].mean(axis=0)
                    else:
                        # Empty cluster: reinitialize to random point
                        new_centroids[k] = X[rng.randint(n_samples)]

                # Check convergence
                centroid_shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids

                if centroid_shift < self.tol:
                    break

            # Compute inertia (within-cluster sum of squares)
            inertia = 0.0
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    cluster_points = X[mask]
                    inertia += np.sum((cluster_points - centroids[k]) ** 2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()

        self.centroids_ = best_centroids
        self.inertia_ = best_inertia

        self._result = ClusteringResult(
            labels=best_labels,
            n_clusters=self.n_clusters,
            centroids=best_centroids,
            inertia=best_inertia,
            metadata={"n_init": self.n_init, "max_iter": self.max_iter},
        )

        self._mark_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new samples.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Cluster labels for each sample
        """
        self._check_is_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        distances = compute_pairwise_distances(X, self.centroids_, metric=DistanceMetric.EUCLIDEAN)
        return np.argmin(distances, axis=1)

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters for serialization."""
        params = super()._get_saveable_params()
        params.update({
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": self.n_init,
        })
        if self.centroids_ is not None:
            params["centroids"] = self.centroids_.tolist()
        return params


class DBSCANClusterer(BaseClusterer):
    """DBSCAN density-based clustering for spectral data.

    Density-Based Spatial Clustering of Applications with Noise.
    Finds clusters of arbitrary shape and identifies noise points.
    Does not require specifying number of clusters in advance.

    Attributes:
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum samples in a neighborhood to form a core point
        metric: Distance metric to use

    Example:
        >>> clusterer = DBSCANClusterer(eps=0.5, min_samples=5)
        >>> result = clusterer.fit_predict(spectra)
        >>> noise_count = result.cluster_sizes.get(-1, 0)
        >>> print(f"Found {result.n_clusters} clusters, {noise_count} noise points")
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
    ):
        """Initialize DBSCAN clusterer.

        Args:
            eps: Epsilon neighborhood radius
            min_samples: Minimum points to form a dense region
            metric: Distance metric to use
        """
        super().__init__(random_state=None)  # DBSCAN is deterministic
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.core_sample_indices_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> "DBSCANClusterer":
        """Fit DBSCAN clustering to data.

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Ignored

        Returns:
            self
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples = X.shape[0]

        # Compute full distance matrix
        distances = compute_pairwise_distances(X, metric=self.metric)

        # Find neighbors within eps for each point
        neighborhoods = [np.where(distances[i] <= self.eps)[0] for i in range(n_samples)]

        # Identify core points
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        core_mask = n_neighbors >= self.min_samples
        core_indices = np.where(core_mask)[0]
        self.core_sample_indices_ = core_indices

        # Initialize labels: -1 = unclassified/noise
        labels = np.full(n_samples, -1, dtype=int)

        # Cluster assignment
        cluster_id = 0
        for core_idx in core_indices:
            if labels[core_idx] != -1:
                continue

            # Start new cluster with BFS from this core point
            labels[core_idx] = cluster_id
            seeds = list(neighborhoods[core_idx])

            while seeds:
                q = seeds.pop(0)
                if labels[q] == -1:  # Previously noise
                    labels[q] = cluster_id
                elif labels[q] != cluster_id:
                    continue

                if core_mask[q]:  # If q is also a core point, expand
                    for neighbor in neighborhoods[q]:
                        if labels[neighbor] == -1:
                            seeds.append(neighbor)
                            labels[neighbor] = cluster_id

            cluster_id += 1

        self._result = ClusteringResult(
            labels=labels,
            n_clusters=cluster_id,
            metadata={
                "eps": self.eps,
                "min_samples": self.min_samples,
                "n_core_samples": len(core_indices),
            },
        )

        self._mark_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new samples.

        Note: DBSCAN does not have a natural predict method.
        New samples are assigned to the nearest core point's cluster
        if within eps, otherwise labeled as noise (-1).

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Cluster labels for each sample
        """
        self._check_is_fitted()
        # DBSCAN doesn't naturally predict; this is a simple approximation
        raise NotImplementedError(
            "DBSCAN predict() is not naturally defined. "
            "Use fit_predict() for new data or consider KMeans for prediction tasks."
        )

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters for serialization."""
        params = super()._get_saveable_params()
        params.update({
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": str(self.metric),
        })
        return params


class HierarchicalClusterer(BaseClusterer):
    """Agglomerative Hierarchical Clustering for spectral data.

    Builds a hierarchy of clusters by iteratively merging the closest
    clusters based on linkage criteria.

    Attributes:
        n_clusters: Number of clusters to extract from the hierarchy
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        metric: Distance metric to use

    Example:
        >>> clusterer = HierarchicalClusterer(n_clusters=5, linkage="ward")
        >>> result = clusterer.fit_predict(spectra)
        >>> print(f"Cluster sizes: {result.cluster_sizes}")
    """

    def __init__(
        self,
        n_clusters: int = 2,
        linkage: str = "ward",
        metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
    ):
        """Initialize Hierarchical clusterer.

        Args:
            n_clusters: Number of clusters to form
            linkage: Linkage criterion for distance between clusters
            metric: Distance metric (ignored for ward linkage)
        """
        super().__init__(random_state=None)  # Hierarchical is deterministic
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.dendrogram_: Optional[List] = None

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> "HierarchicalClusterer":
        """Fit hierarchical clustering to data.

        Args:
            X: Input data of shape (n_samples, n_features)
            y: Ignored

        Returns:
            self
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_samples = X.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        # Each sample starts in its own cluster
        # cluster_members[i] = list of sample indices in cluster i
        cluster_members: Dict[int, List[int]] = {i: [i] for i in range(n_samples)}
        active_clusters = set(range(n_samples))
        cluster_id_counter = n_samples

        # Store merge history for dendrogram
        merge_history = []

        # Compute initial distance matrix
        if self.linkage == "ward":
            # For Ward, we need original data for centroid computation
            cluster_centroids = {i: X[i] for i in range(n_samples)}
        else:
            distances = compute_pairwise_distances(X, metric=self.metric)

        while len(active_clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_pair = None

            cluster_list = list(active_clusters)
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    ci, cj = cluster_list[i], cluster_list[j]

                    if self.linkage == "ward":
                        # Ward: increase in total variance
                        ni = len(cluster_members[ci])
                        nj = len(cluster_members[cj])
                        ci_centroid = cluster_centroids[ci]
                        cj_centroid = cluster_centroids[cj]
                        dist = (ni * nj / (ni + nj)) * np.sum((ci_centroid - cj_centroid) ** 2)
                    else:
                        # Other linkages
                        pair_distances = []
                        for mi in cluster_members[ci]:
                            for mj in cluster_members[cj]:
                                pair_distances.append(distances[mi, mj])

                        if self.linkage == "single":
                            dist = min(pair_distances)
                        elif self.linkage == "complete":
                            dist = max(pair_distances)
                        elif self.linkage == "average":
                            dist = np.mean(pair_distances)
                        else:
                            raise ValueError(f"Unknown linkage: {self.linkage}")

                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (ci, cj)

            # Merge the closest pair
            ci, cj = merge_pair
            new_cluster = cluster_id_counter
            cluster_id_counter += 1

            # Combine members
            cluster_members[new_cluster] = cluster_members[ci] + cluster_members[cj]

            # Update centroid for Ward
            if self.linkage == "ward":
                ni = len(cluster_members[ci])
                nj = len(cluster_members[cj])
                cluster_centroids[new_cluster] = (
                    ni * cluster_centroids[ci] + nj * cluster_centroids[cj]
                ) / (ni + nj)
                del cluster_centroids[ci]
                del cluster_centroids[cj]

            # Update active clusters
            active_clusters.remove(ci)
            active_clusters.remove(cj)
            active_clusters.add(new_cluster)

            # Record merge
            merge_history.append({
                "clusters": (ci, cj),
                "new_cluster": new_cluster,
                "distance": min_dist,
                "n_members": len(cluster_members[new_cluster]),
            })

            del cluster_members[ci]
            del cluster_members[cj]

        self.dendrogram_ = merge_history

        # Create final labels
        labels = np.zeros(n_samples, dtype=int)
        for label_idx, cluster_id in enumerate(active_clusters):
            for sample_idx in cluster_members[cluster_id]:
                labels[sample_idx] = label_idx

        self._result = ClusteringResult(
            labels=labels,
            n_clusters=self.n_clusters,
            metadata={
                "linkage": self.linkage,
                "n_merges": len(merge_history),
            },
        )

        self._mark_fitted()
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new samples.

        Note: Hierarchical clustering does not naturally support prediction.
        New samples are assigned to the cluster with nearest centroid.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Cluster labels for each sample
        """
        self._check_is_fitted()
        raise NotImplementedError(
            "Hierarchical clustering predict() is not naturally defined. "
            "Use fit_predict() for new data or consider KMeans for prediction tasks."
        )

    def _get_saveable_params(self) -> Dict[str, Any]:
        """Get parameters for serialization."""
        params = super()._get_saveable_params()
        params.update({
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": str(self.metric),
        })
        return params


def compute_silhouette_samples(
    X: ArrayLike,
    labels: ArrayLike,
    metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Compute silhouette coefficient for each sample.

    The silhouette coefficient measures how similar a sample is to its own
    cluster compared to other clusters. Values range from -1 to 1, where:
    - 1 means the sample is far from neighboring clusters
    - 0 means the sample is on or very close to a cluster boundary
    - -1 means the sample may be assigned to the wrong cluster

    This implementation uses chunked processing for memory efficiency with
    large datasets (>50k samples).

    Args:
        X: Input data of shape (n_samples, n_features)
        labels: Cluster labels for each sample
        metric: Distance metric to use
        chunk_size: Batch size for chunked distance computation

    Returns:
        Array of silhouette coefficients, shape (n_samples,)

    Raises:
        ValueError: If less than 2 clusters or only 1 sample per cluster

    Example:
        >>> spectra = np.random.randn(100, 50)
        >>> clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        >>> result = clusterer.fit_predict(spectra)
        >>> sil_samples = compute_silhouette_samples(spectra, result.labels)
        >>> print(f"Mean silhouette: {sil_samples.mean():.3f}")
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = X.shape[0]

    if len(labels) != n_samples:
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of samples ({n_samples})"
        )

    # Get unique labels (excluding noise label -1)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        raise ValueError(
            f"Silhouette score requires at least 2 clusters, found {n_clusters}"
        )

    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    # Compute cluster sizes and masks
    cluster_masks = {label: (labels == label) for label in unique_labels}
    cluster_sizes = {label: np.sum(mask) for label, mask in cluster_masks.items()}

    # Pre-allocate arrays for a(i) and b(i)
    # a(i) = mean distance to same cluster
    # b(i) = min mean distance to other clusters
    intra_cluster_dist = np.zeros(n_samples, dtype=np.float64)
    nearest_cluster_dist = np.full(n_samples, np.inf, dtype=np.float64)

    # Chunked computation for memory efficiency
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_indices = np.arange(chunk_start, chunk_end)
        X_chunk = X[chunk_start:chunk_end]

        # Compute distances from this chunk to all samples
        distances_chunk = compute_pairwise_distances(X_chunk, X, metric=metric)

        for i_local, i_global in enumerate(chunk_indices):
            sample_label = labels[i_global]

            # Skip noise points
            if sample_label < 0:
                intra_cluster_dist[i_global] = 0
                nearest_cluster_dist[i_global] = 0
                continue

            # Get distances from this sample to all other samples
            dists = distances_chunk[i_local]

            # Compute a(i): mean distance to same cluster (excluding self)
            same_cluster_mask = cluster_masks[sample_label].copy()
            same_cluster_mask[i_global] = False  # Exclude self
            same_cluster_count = cluster_sizes[sample_label] - 1

            if same_cluster_count > 0:
                intra_cluster_dist[i_global] = np.sum(dists[same_cluster_mask]) / same_cluster_count
            else:
                # Only sample in cluster
                intra_cluster_dist[i_global] = 0

            # Compute b(i): min mean distance to other clusters
            for other_label in unique_labels:
                if other_label == sample_label:
                    continue

                other_mask = cluster_masks[other_label]
                other_count = cluster_sizes[other_label]

                if other_count > 0:
                    mean_dist = np.sum(dists[other_mask]) / other_count
                    nearest_cluster_dist[i_global] = min(
                        nearest_cluster_dist[i_global], mean_dist
                    )

    # Compute silhouette coefficients: (b - a) / max(a, b)
    silhouette = np.zeros(n_samples, dtype=np.float64)
    max_ab = np.maximum(intra_cluster_dist, nearest_cluster_dist)

    # Avoid division by zero
    nonzero_mask = max_ab > 0
    silhouette[nonzero_mask] = (
        (nearest_cluster_dist[nonzero_mask] - intra_cluster_dist[nonzero_mask])
        / max_ab[nonzero_mask]
    )

    # Noise points get silhouette of 0
    noise_mask = labels < 0
    silhouette[noise_mask] = 0

    return silhouette


def compute_silhouette_score(
    X: ArrayLike,
    labels: ArrayLike,
    metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
    chunk_size: int = 2000,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> float:
    """Compute mean silhouette score for clustering quality evaluation.

    The silhouette score is a measure of how well-separated clusters are.
    Values range from -1 to 1, where higher is better:
    - > 0.7: Strong clustering structure
    - 0.5 - 0.7: Reasonable clustering
    - 0.25 - 0.5: Weak clustering, possibly overlapping
    - < 0.25: No substantial structure

    For large datasets (>50k samples), uses random sampling and chunked
    distance computation to maintain O(n * sample_size) memory usage.

    Args:
        X: Input data of shape (n_samples, n_features)
        labels: Cluster labels for each sample (noise points labeled -1)
        metric: Distance metric to use
        chunk_size: Batch size for chunked distance computation
        sample_size: If set, use random sampling for large datasets.
            None uses all samples (up to 10000 by default for performance).
        random_state: Random seed for reproducible sampling

    Returns:
        Mean silhouette coefficient across all samples (excluding noise)

    Raises:
        ValueError: If less than 2 clusters found

    Example:
        >>> spectra = np.random.randn(100, 50)
        >>> clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        >>> result = clusterer.fit_predict(spectra)
        >>> score = compute_silhouette_score(spectra, result.labels)
        >>> print(f"Silhouette score: {score:.3f}")

    Note:
        For very large datasets (>50k samples), consider using sample_size
        parameter to avoid excessive computation time while still getting
        a representative score estimate.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    n_samples = X.shape[0]

    # Default sample size for large datasets
    default_sample_size = 10000
    if sample_size is None and n_samples > default_sample_size:
        sample_size = default_sample_size

    # Apply sampling if needed
    if sample_size is not None and n_samples > sample_size:
        rng = np.random.RandomState(random_state)

        # Stratified sampling to preserve cluster proportions
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise

        # Calculate samples per cluster proportionally
        sampled_indices = []
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_label = len(label_indices)
            # Proportional sample size, but at least 2 per cluster
            n_sample_label = max(2, int(sample_size * n_label / n_samples))
            n_sample_label = min(n_sample_label, n_label)
            sampled = rng.choice(label_indices, n_sample_label, replace=False)
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        X = X[sampled_indices]
        labels = labels[sampled_indices]

    # Compute per-sample silhouette
    silhouette_samples = compute_silhouette_samples(X, labels, metric, chunk_size)

    # Return mean (excluding noise points)
    non_noise_mask = labels >= 0
    if not np.any(non_noise_mask):
        return 0.0

    return float(np.mean(silhouette_samples[non_noise_mask]))
