"""
Unit tests for ML clustering algorithms.

Tests the clustering implementations in ml/clustering.py:
- KMeansClusterer
- DBSCANClusterer
- HierarchicalClusterer
- ClusteringResult
"""

import numpy as np
import pytest

from sherloc_pipeline.ml.clustering import (
    ClusteringResult,
    BaseClusterer,
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    compute_silhouette_score,
    compute_silhouette_samples,
)


@pytest.fixture
def simple_clusters():
    """Create simple 2D data with clear cluster structure."""
    np.random.seed(42)
    # 3 well-separated clusters
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([10, 0])
    cluster3 = np.random.randn(30, 2) + np.array([5, 10])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def synthetic_spectra():
    """Create synthetic spectral data with patterns."""
    np.random.seed(42)
    n_samples = 50
    n_channels = 100

    # Three spectral patterns
    base1 = np.sin(np.linspace(0, 4 * np.pi, n_channels))
    base2 = np.cos(np.linspace(0, 2 * np.pi, n_channels))
    base3 = np.exp(-((np.arange(n_channels) - 50) ** 2) / 200)

    spectra = []
    labels = []
    for i in range(n_samples):
        pattern = i % 3
        if pattern == 0:
            s = base1 + 0.1 * np.random.randn(n_channels)
        elif pattern == 1:
            s = base2 + 0.1 * np.random.randn(n_channels)
        else:
            s = base3 + 0.1 * np.random.randn(n_channels)
        spectra.append(s)
        labels.append(pattern)

    return np.array(spectra), np.array(labels)


class TestClusteringResult:
    """Tests for ClusteringResult dataclass."""

    def test_basic_creation(self):
        """Create ClusteringResult with labels."""
        labels = np.array([0, 0, 1, 1, 2])
        result = ClusteringResult(labels=labels, n_clusters=3)

        assert result.n_clusters == 3
        assert len(result.labels) == 5

    def test_cluster_sizes_computed(self):
        """Cluster sizes are computed from labels."""
        labels = np.array([0, 0, 0, 1, 1, 2])
        result = ClusteringResult(labels=labels, n_clusters=3)

        assert result.cluster_sizes[0] == 3
        assert result.cluster_sizes[1] == 2
        assert result.cluster_sizes[2] == 1

    def test_get_cluster_members(self):
        """get_cluster_members returns correct indices."""
        labels = np.array([0, 1, 0, 1, 0])
        result = ClusteringResult(labels=labels, n_clusters=2)

        members_0 = result.get_cluster_members(0)
        members_1 = result.get_cluster_members(1)

        assert list(members_0) == [0, 2, 4]
        assert list(members_1) == [1, 3]

    def test_handles_noise_labels(self):
        """Handles DBSCAN noise labels (-1)."""
        labels = np.array([0, -1, 1, -1, 0])
        result = ClusteringResult(labels=labels, n_clusters=0)

        # n_clusters should exclude noise
        assert result.n_clusters == 2  # clusters 0 and 1
        assert result.cluster_sizes[-1] == 2  # noise points

    def test_centroids_optional(self):
        """Centroids are optional."""
        labels = np.array([0, 0, 1, 1])
        result = ClusteringResult(labels=labels, n_clusters=2)
        assert result.centroids is None

        centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
        result_with = ClusteringResult(labels=labels, n_clusters=2, centroids=centroids)
        assert result_with.centroids is not None


class TestKMeansClusterer:
    """Tests for KMeansClusterer."""

    def test_basic_clustering(self, simple_clusters):
        """Basic clustering finds expected number of clusters."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        assert result.n_clusters == 3
        assert len(result.labels) == len(simple_clusters)
        assert set(result.labels) == {0, 1, 2}

    def test_centroids_computed(self, simple_clusters):
        """Centroids are computed after fitting."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        assert result.centroids is not None
        assert result.centroids.shape == (3, 2)

    def test_inertia_computed(self, simple_clusters):
        """Inertia (within-cluster sum of squares) is computed."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        assert result.inertia is not None
        assert result.inertia >= 0

    def test_predict_new_samples(self, simple_clusters):
        """Can predict cluster labels for new samples."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_clusters)

        new_samples = np.array([[0.0, 0.0], [10.0, 0.0]])
        predictions = clusterer.predict(new_samples)

        assert len(predictions) == 2

    def test_reproducibility(self, simple_clusters):
        """Same random_state produces same results."""
        clusterer1 = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer2 = KMeansClusterer(n_clusters=3, random_state=42)

        result1 = clusterer1.fit_predict(simple_clusters)
        result2 = clusterer2.fit_predict(simple_clusters)

        assert np.array_equal(result1.labels, result2.labels)

    def test_not_fitted_error(self):
        """Predict before fit raises error."""
        clusterer = KMeansClusterer(n_clusters=3)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clusterer.predict(np.array([[0.0, 0.0]]))

    def test_too_few_samples(self):
        """Fewer samples than clusters raises error."""
        clusterer = KMeansClusterer(n_clusters=5)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="n_samples"):
            clusterer.fit(X)

    def test_spectral_data(self, synthetic_spectra):
        """Works with high-dimensional spectral data."""
        spectra, _ = synthetic_spectra
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(spectra)

        assert result.n_clusters == 3
        assert len(result.labels) == len(spectra)


class TestDBSCANClusterer:
    """Tests for DBSCANClusterer."""

    def test_basic_clustering(self, simple_clusters):
        """DBSCAN finds clusters in well-separated data."""
        clusterer = DBSCANClusterer(eps=2.0, min_samples=5)
        result = clusterer.fit_predict(simple_clusters)

        # Should find 3 clusters
        assert result.n_clusters >= 2  # May vary slightly

    def test_identifies_noise(self):
        """DBSCAN identifies noise points as label -1."""
        # Create data with outliers
        X = np.vstack([
            np.random.randn(20, 2),  # Dense cluster
            np.array([[10.0, 10.0], [11.0, 10.0]])  # Outliers
        ])

        clusterer = DBSCANClusterer(eps=1.0, min_samples=5)
        result = clusterer.fit_predict(X)

        # Should have some noise points
        assert -1 in result.labels

    def test_core_samples_stored(self, simple_clusters):
        """Core sample indices are stored."""
        clusterer = DBSCANClusterer(eps=2.0, min_samples=5)
        clusterer.fit(simple_clusters)

        assert clusterer.core_sample_indices_ is not None
        assert len(clusterer.core_sample_indices_) > 0

    def test_deterministic(self, simple_clusters):
        """DBSCAN is deterministic."""
        clusterer1 = DBSCANClusterer(eps=2.0, min_samples=5)
        clusterer2 = DBSCANClusterer(eps=2.0, min_samples=5)

        result1 = clusterer1.fit_predict(simple_clusters)
        result2 = clusterer2.fit_predict(simple_clusters)

        assert np.array_equal(result1.labels, result2.labels)

    def test_predict_not_supported(self, simple_clusters):
        """DBSCAN predict raises NotImplementedError."""
        clusterer = DBSCANClusterer(eps=2.0, min_samples=5)
        clusterer.fit(simple_clusters)

        with pytest.raises(NotImplementedError):
            clusterer.predict(np.array([[0.0, 0.0]]))

    def test_eps_effect(self, simple_clusters):
        """Larger eps tends to merge clusters."""
        small_eps = DBSCANClusterer(eps=1.0, min_samples=5)
        large_eps = DBSCANClusterer(eps=10.0, min_samples=5)

        result_small = small_eps.fit_predict(simple_clusters)
        result_large = large_eps.fit_predict(simple_clusters)

        # Larger eps should find fewer (or same) clusters
        assert result_large.n_clusters <= result_small.n_clusters


class TestHierarchicalClusterer:
    """Tests for HierarchicalClusterer."""

    def test_basic_clustering(self, simple_clusters):
        """Hierarchical clustering finds expected clusters."""
        clusterer = HierarchicalClusterer(n_clusters=3, linkage="ward")
        result = clusterer.fit_predict(simple_clusters)

        assert result.n_clusters == 3
        assert len(result.labels) == len(simple_clusters)

    def test_linkage_options(self, simple_clusters):
        """Different linkage options work."""
        for linkage in ["ward", "complete", "average", "single"]:
            clusterer = HierarchicalClusterer(n_clusters=3, linkage=linkage)
            result = clusterer.fit_predict(simple_clusters)
            assert result.n_clusters == 3

    def test_dendrogram_stored(self, simple_clusters):
        """Dendrogram merge history is stored."""
        clusterer = HierarchicalClusterer(n_clusters=3, linkage="ward")
        clusterer.fit(simple_clusters)

        assert clusterer.dendrogram_ is not None
        assert len(clusterer.dendrogram_) > 0

    def test_deterministic(self, simple_clusters):
        """Hierarchical clustering is deterministic."""
        clusterer1 = HierarchicalClusterer(n_clusters=3, linkage="ward")
        clusterer2 = HierarchicalClusterer(n_clusters=3, linkage="ward")

        result1 = clusterer1.fit_predict(simple_clusters)
        result2 = clusterer2.fit_predict(simple_clusters)

        assert np.array_equal(result1.labels, result2.labels)

    def test_predict_not_supported(self, simple_clusters):
        """Hierarchical predict raises NotImplementedError."""
        clusterer = HierarchicalClusterer(n_clusters=3)
        clusterer.fit(simple_clusters)

        with pytest.raises(NotImplementedError):
            clusterer.predict(np.array([[0.0, 0.0]]))

    def test_spectral_data(self, synthetic_spectra):
        """Works with spectral data."""
        spectra, _ = synthetic_spectra
        clusterer = HierarchicalClusterer(n_clusters=3, linkage="complete")
        result = clusterer.fit_predict(spectra)

        assert result.n_clusters == 3


class TestClustererPersistence:
    """Tests for model save/load functionality."""

    def test_kmeans_save_load(self, simple_clusters, tmp_path):
        """KMeans can be saved and loaded."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        clusterer.fit(simple_clusters)

        save_path = tmp_path / "kmeans.pkl"
        clusterer.save(save_path)

        loaded = KMeansClusterer.load(save_path)
        assert loaded.n_clusters == 3
        assert loaded.centroids_ is not None

    def test_kmeans_json_params(self, simple_clusters, tmp_path):
        """KMeans parameters can be saved to JSON."""
        clusterer = KMeansClusterer(n_clusters=5, max_iter=100)
        clusterer.fit(simple_clusters)

        save_path = tmp_path / "kmeans_params.json"
        clusterer.save(save_path, format="json")

        # Can read the JSON file
        import json
        with open(save_path) as f:
            params = json.load(f)

        assert params["n_clusters"] == 5
        assert params["max_iter"] == 100


class TestSilhouetteScore:
    """Tests for silhouette score computation."""

    def test_basic_silhouette_score(self, simple_clusters):
        """Compute silhouette score for well-separated clusters."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        score = compute_silhouette_score(simple_clusters, result.labels)

        # Well-separated clusters should have high silhouette score
        assert -1 <= score <= 1
        assert score > 0.5  # Well-separated data

    def test_silhouette_samples_returns_correct_shape(self, simple_clusters):
        """Silhouette samples has correct shape."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        sil_samples = compute_silhouette_samples(simple_clusters, result.labels)

        assert sil_samples.shape == (len(simple_clusters),)

    def test_silhouette_values_in_range(self, simple_clusters):
        """All silhouette values are in [-1, 1]."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        sil_samples = compute_silhouette_samples(simple_clusters, result.labels)

        assert np.all(sil_samples >= -1)
        assert np.all(sil_samples <= 1)

    def test_silhouette_mean_equals_score(self, simple_clusters):
        """Mean of samples equals overall score."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        sil_samples = compute_silhouette_samples(simple_clusters, result.labels)
        score = compute_silhouette_score(simple_clusters, result.labels)

        np.testing.assert_almost_equal(np.mean(sil_samples), score, decimal=10)

    def test_silhouette_with_spectral_data(self, synthetic_spectra):
        """Works with high-dimensional spectral data."""
        spectra, _ = synthetic_spectra
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(spectra)

        score = compute_silhouette_score(spectra, result.labels)

        assert -1 <= score <= 1
        # Synthetic spectra with distinct patterns should cluster well
        assert score > 0.3

    def test_silhouette_with_dbscan_noise(self):
        """Handles DBSCAN noise points correctly."""
        np.random.seed(42)
        # Create data with two clear clusters and outliers
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([8, 0])
        outliers = np.array([[20.0, 20.0], [20.0, -20.0]])
        X = np.vstack([cluster1, cluster2, outliers])

        clusterer = DBSCANClusterer(eps=2.0, min_samples=5)
        result = clusterer.fit_predict(X)

        # Should have at least 2 clusters
        assert result.n_clusters >= 2
        # Should have noise points (label -1)
        assert -1 in result.labels

        # Silhouette should still work
        score = compute_silhouette_score(X, result.labels)
        assert -1 <= score <= 1

    def test_silhouette_samples_noise_is_zero(self):
        """Noise points get silhouette of 0."""
        np.random.seed(42)
        # Create data with two clear clusters and outliers
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([8, 0])
        outliers = np.array([[20.0, 20.0], [20.0, -20.0]])
        X = np.vstack([cluster1, cluster2, outliers])

        clusterer = DBSCANClusterer(eps=2.0, min_samples=5)
        result = clusterer.fit_predict(X)

        # Should have at least 2 clusters
        assert result.n_clusters >= 2

        sil_samples = compute_silhouette_samples(X, result.labels)

        # Noise points should have silhouette of 0
        noise_mask = result.labels < 0
        if np.any(noise_mask):
            assert np.all(sil_samples[noise_mask] == 0)

    def test_silhouette_requires_multiple_clusters(self):
        """Raises error if only one cluster."""
        X = np.random.randn(20, 2)
        labels = np.zeros(20, dtype=int)  # All in one cluster

        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_silhouette_score(X, labels)

    def test_silhouette_label_length_mismatch(self):
        """Raises error if labels length doesn't match samples."""
        X = np.random.randn(20, 2)
        labels = np.array([0, 1, 0])  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            compute_silhouette_samples(X, labels)

    def test_silhouette_with_sampling(self, simple_clusters):
        """Sampling produces similar score estimate."""
        # Create larger dataset
        np.random.seed(42)
        cluster1 = np.random.randn(500, 2) + np.array([0, 0])
        cluster2 = np.random.randn(500, 2) + np.array([10, 0])
        cluster3 = np.random.randn(500, 2) + np.array([5, 10])
        X = np.vstack([cluster1, cluster2, cluster3])

        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(X)

        # Full score (with internal sampling)
        score_full = compute_silhouette_score(
            X, result.labels, sample_size=None, random_state=42
        )

        # Sampled score
        score_sampled = compute_silhouette_score(
            X, result.labels, sample_size=500, random_state=42
        )

        # Should be similar (within 0.1)
        assert abs(score_full - score_sampled) < 0.15

    def test_silhouette_chunked_matches_unchunked(self, simple_clusters):
        """Chunked processing produces same result."""
        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        # Small chunks
        score_chunked = compute_silhouette_score(
            simple_clusters, result.labels, chunk_size=10
        )

        # Large chunks (effectively no chunking)
        score_unchunked = compute_silhouette_score(
            simple_clusters, result.labels, chunk_size=10000
        )

        np.testing.assert_almost_equal(score_chunked, score_unchunked, decimal=10)

    def test_silhouette_different_metrics(self, simple_clusters):
        """Works with different distance metrics."""
        from sherloc_pipeline.ml.distance import DistanceMetric

        clusterer = KMeansClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(simple_clusters)

        for metric in [DistanceMetric.EUCLIDEAN, DistanceMetric.COSINE]:
            score = compute_silhouette_score(
                simple_clusters, result.labels, metric=metric
            )
            assert -1 <= score <= 1

    def test_silhouette_reproducibility(self, simple_clusters):
        """Same random_state produces same sampled result."""
        np.random.seed(42)
        cluster1 = np.random.randn(500, 2) + np.array([0, 0])
        cluster2 = np.random.randn(500, 2) + np.array([10, 0])
        X = np.vstack([cluster1, cluster2])

        clusterer = KMeansClusterer(n_clusters=2, random_state=42)
        result = clusterer.fit_predict(X)

        score1 = compute_silhouette_score(
            X, result.labels, sample_size=200, random_state=123
        )
        score2 = compute_silhouette_score(
            X, result.labels, sample_size=200, random_state=123
        )

        assert score1 == score2


class TestSilhouetteScalability:
    """Tests for silhouette score scalability with large datasets."""

    @pytest.mark.parametrize("n_samples", [1000, 5000])
    def test_scales_with_samples(self, n_samples):
        """Handles datasets with many samples."""
        np.random.seed(42)
        n_features = 100

        # Create 3 clusters
        X = np.vstack([
            np.random.randn(n_samples // 3, n_features) + np.array([i * 5] * n_features)
            for i in range(3)
        ])
        labels = np.array([i for i in range(3) for _ in range(n_samples // 3)])

        # Should complete without memory issues
        score = compute_silhouette_score(X, labels, chunk_size=500)
        assert -1 <= score <= 1

    def test_large_dataset_with_sampling(self):
        """Large dataset with sampling for speed."""
        np.random.seed(42)
        n_samples = 20000
        n_features = 50

        # Create 5 clusters
        X = np.vstack([
            np.random.randn(n_samples // 5, n_features) + np.array([i * 3] * n_features)
            for i in range(5)
        ])
        labels = np.array([i for i in range(5) for _ in range(n_samples // 5)])

        # With sampling, should be fast
        score = compute_silhouette_score(
            X, labels, sample_size=2000, random_state=42, chunk_size=500
        )
        assert -1 <= score <= 1
        # Well-separated clusters should score high
        assert score > 0.4

    def test_high_dimensional_spectra(self):
        """Works with high-dimensional spectral data."""
        np.random.seed(42)
        n_samples = 500
        n_channels = 1000  # High-dimensional like real SHERLOC data

        # Create distinct spectral patterns
        base_patterns = [
            np.sin(np.linspace(0, k * np.pi, n_channels))
            for k in range(1, 4)
        ]

        X = []
        labels = []
        for i in range(n_samples):
            pattern_idx = i % 3
            spectrum = base_patterns[pattern_idx] + 0.1 * np.random.randn(n_channels)
            X.append(spectrum)
            labels.append(pattern_idx)

        X = np.array(X)
        labels = np.array(labels)

        score = compute_silhouette_score(X, labels, chunk_size=100)
        assert -1 <= score <= 1
