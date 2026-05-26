"""
Tests for spectral similarity module.

Tests cover:
- Basic similarity metrics (cosine, euclidean, correlation, spectral angle)
- Batch processing functions
- SpectralSimilarity class with fit/transform/find_similar
- Large-scale performance (>10k spectra)
- Real fixture data integration
"""

import numpy as np
import pytest
from pathlib import Path

from sherloc_pipeline.ml.similarity import (
    SpectralSimilarity,
    SimilarityResult,
    SimilarityConfig,
    SimilarityMetric,
    batch_cosine_similarity,
    batch_euclidean_distance,
    batch_correlation,
    batch_spectral_angle,
    find_similar_spectra,
)


class TestBatchCosine:
    """Tests for batch_cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        sim = batch_cosine_similarity(X, X)

        assert sim.shape == (2, 2)
        np.testing.assert_almost_equal(sim[0, 0], 1.0)
        np.testing.assert_almost_equal(sim[1, 1], 1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        X = np.array([[1, 0, 0]])
        Y = np.array([[0, 1, 0]])
        sim = batch_cosine_similarity(X, Y)

        np.testing.assert_almost_equal(sim[0, 0], 0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        X = np.array([[1, 0, 0]])
        Y = np.array([[-1, 0, 0]])
        sim = batch_cosine_similarity(X, Y)

        np.testing.assert_almost_equal(sim[0, 0], -1.0)

    def test_self_comparison(self):
        """Y=None should compute self-similarity matrix."""
        X = np.random.randn(5, 10)
        sim = batch_cosine_similarity(X)

        assert sim.shape == (5, 5)
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(5))
        # Should be symmetric
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_scaling_invariance(self):
        """Cosine similarity should be invariant to scaling."""
        X = np.array([[1, 2, 3]])
        Y = np.array([[10, 20, 30]])  # Scaled version
        sim = batch_cosine_similarity(X, Y)

        np.testing.assert_almost_equal(sim[0, 0], 1.0)

    def test_output_range(self):
        """All similarities should be in [-1, 1]."""
        X = np.random.randn(100, 50)
        Y = np.random.randn(80, 50)
        sim = batch_cosine_similarity(X, Y)

        assert np.all(sim >= -1.0)
        assert np.all(sim <= 1.0)


class TestBatchEuclidean:
    """Tests for batch_euclidean_distance function."""

    def test_identical_vectors(self):
        """Identical vectors should have distance 0.0."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        dist = batch_euclidean_distance(X, X)

        np.testing.assert_almost_equal(dist[0, 0], 0.0)
        np.testing.assert_almost_equal(dist[1, 1], 0.0)

    def test_known_distance(self):
        """Test 3-4-5 triangle."""
        X = np.array([[0, 0]])
        Y = np.array([[3, 4]])
        dist = batch_euclidean_distance(X, Y)

        np.testing.assert_almost_equal(dist[0, 0], 5.0)

    def test_symmetry(self):
        """Distance matrix should be symmetric for self-comparison."""
        X = np.random.randn(10, 20)
        dist = batch_euclidean_distance(X)

        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_non_negative(self):
        """All distances should be non-negative."""
        X = np.random.randn(100, 50)
        Y = np.random.randn(80, 50)
        dist = batch_euclidean_distance(X, Y)

        assert np.all(dist >= 0)


class TestBatchCorrelation:
    """Tests for batch_correlation function."""

    def test_perfect_correlation(self):
        """Linearly related vectors should have correlation 1.0."""
        X = np.array([[1, 2, 3, 4, 5]])
        Y = np.array([[2, 4, 6, 8, 10]])  # Linear scaling
        corr = batch_correlation(X, Y)

        np.testing.assert_almost_equal(corr[0, 0], 1.0)

    def test_perfect_anticorrelation(self):
        """Inversely related vectors should have correlation -1.0."""
        X = np.array([[1, 2, 3, 4, 5]])
        Y = np.array([[5, 4, 3, 2, 1]])  # Reversed
        corr = batch_correlation(X, Y)

        np.testing.assert_almost_equal(corr[0, 0], -1.0)

    def test_offset_invariance(self):
        """Correlation should be invariant to offset."""
        X = np.array([[1, 2, 3]])
        Y = np.array([[101, 102, 103]])  # Shifted by 100
        corr = batch_correlation(X, Y)

        np.testing.assert_almost_equal(corr[0, 0], 1.0)

    def test_output_range(self):
        """All correlations should be in [-1, 1]."""
        X = np.random.randn(50, 100)
        corr = batch_correlation(X)

        assert np.all(corr >= -1.0)
        assert np.all(corr <= 1.0)


class TestBatchSpectralAngle:
    """Tests for batch_spectral_angle function."""

    def test_identical_vectors(self):
        """Identical vectors should have angle 0.0."""
        X = np.array([[1, 2, 3]])
        angle = batch_spectral_angle(X, X)

        np.testing.assert_almost_equal(angle[0, 0], 0.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have angle pi/2."""
        X = np.array([[1, 0]])
        Y = np.array([[0, 1]])
        angle = batch_spectral_angle(X, Y)

        np.testing.assert_almost_equal(angle[0, 0], np.pi / 2)

    def test_opposite_vectors(self):
        """Opposite vectors should have angle pi."""
        X = np.array([[1, 0]])
        Y = np.array([[-1, 0]])
        angle = batch_spectral_angle(X, Y)

        np.testing.assert_almost_equal(angle[0, 0], np.pi)

    def test_output_range(self):
        """All angles should be in [0, pi]."""
        X = np.random.randn(50, 100)
        angle = batch_spectral_angle(X)

        assert np.all(angle >= 0)
        assert np.all(angle <= np.pi + 1e-10)  # Allow small numerical error


class TestFindSimilarSpectra:
    """Tests for find_similar_spectra function."""

    def test_finds_identical_spectrum(self):
        """Should find exact match when query is in reference."""
        reference = np.random.randn(100, 50)
        query = reference[42]  # Pick one from reference

        result = find_similar_spectra(query, reference, k=5)

        assert isinstance(result, SimilarityResult)
        assert result.indices[0] == 42
        np.testing.assert_almost_equal(result.scores[0], 1.0)

    def test_returns_k_results(self):
        """Should return exactly k results."""
        reference = np.random.randn(100, 50)
        query = np.random.randn(50)

        result = find_similar_spectra(query, reference, k=10)

        assert len(result.indices) == 10
        assert len(result.scores) == 10

    def test_scores_descending(self):
        """Scores should be in descending order."""
        reference = np.random.randn(100, 50)
        query = np.random.randn(50)

        result = find_similar_spectra(query, reference, k=20)

        # Verify descending order
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    def test_multiple_queries(self):
        """Should handle multiple queries."""
        reference = np.random.randn(100, 50)
        queries = np.random.randn(5, 50)

        result = find_similar_spectra(queries, reference, k=10)

        assert result.indices.shape == (5, 10)
        assert result.scores.shape == (5, 10)

    def test_different_metrics(self):
        """Should work with all metric types."""
        reference = np.random.randn(50, 30)
        query = np.random.randn(30)

        for metric in SimilarityMetric:
            result = find_similar_spectra(query, reference, k=5, metric=metric)
            assert len(result.indices) == 5

    def test_return_distances(self):
        """Should include distances when requested for distance metrics."""
        reference = np.random.randn(50, 30)
        query = np.random.randn(30)

        result = find_similar_spectra(
            query, reference, k=5,
            metric=SimilarityMetric.EUCLIDEAN,
            return_distances=True
        )

        assert result.distances is not None
        assert len(result.distances) == 5


class TestSpectralSimilarity:
    """Tests for SpectralSimilarity class."""

    def test_fit_stores_reference(self):
        """Fit should store and preprocess reference data."""
        X = np.random.randn(100, 50)
        similarity = SpectralSimilarity()
        similarity.fit(X)

        assert similarity.reference_ is not None
        assert similarity.n_samples_ == 100
        assert similarity.n_features_ == 50

    def test_transform_returns_similarity_matrix(self):
        """Transform should return similarity matrix."""
        reference = np.random.randn(100, 50)
        queries = np.random.randn(10, 50)

        similarity = SpectralSimilarity()
        similarity.fit(reference)
        sim_matrix = similarity.transform(queries)

        assert sim_matrix.shape == (10, 100)

    def test_find_similar_basic(self):
        """Basic find_similar test."""
        reference = np.random.randn(100, 50)
        similarity = SpectralSimilarity()
        similarity.fit(reference)

        query = reference[25]  # Use one from reference
        result = similarity.find_similar(query, k=5)

        assert result.indices[0] == 25
        np.testing.assert_almost_equal(result.scores[0], 1.0)

    def test_find_similar_with_threshold(self):
        """find_similar with threshold should filter results."""
        reference = np.random.randn(100, 50)
        similarity = SpectralSimilarity()
        similarity.fit(reference)

        query = np.random.randn(50)
        result = similarity.find_similar(query, k=20, threshold=0.5)

        # All returned scores should be >= threshold
        valid_scores = result.scores[result.scores != 0]  # Exclude padding
        if len(valid_scores) > 0:
            assert np.all(valid_scores >= 0.5)

    def test_different_metrics(self):
        """Should work with all metric types."""
        reference = np.random.randn(50, 30)

        for metric in SimilarityMetric:
            similarity = SpectralSimilarity(metric=metric)
            similarity.fit(reference)
            result = similarity.find_similar(reference[0], k=5)
            assert len(result.indices) == 5

    def test_pairwise_similarity(self):
        """Pairwise similarity should return symmetric matrix."""
        reference = np.random.randn(20, 30)
        similarity = SpectralSimilarity()
        similarity.fit(reference)

        sim_matrix = similarity.pairwise_similarity()

        assert sim_matrix.shape == (20, 20)
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(20))

    def test_similarity_histogram(self):
        """Should compute histogram of similarities."""
        reference = np.random.randn(50, 30)
        similarity = SpectralSimilarity()
        similarity.fit(reference)

        counts, edges = similarity.similarity_histogram(bins=20)

        assert len(counts) == 20
        assert len(edges) == 21

    def test_not_fitted_error(self):
        """Should raise error if transform called before fit."""
        similarity = SpectralSimilarity()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not been fitted"):
            similarity.transform(X)

    def test_feature_mismatch_error(self):
        """Should raise error if query has different features."""
        reference = np.random.randn(100, 50)
        query = np.random.randn(30)  # Wrong number of features

        similarity = SpectralSimilarity()
        similarity.fit(reference)

        with pytest.raises(ValueError, match="features"):
            similarity.find_similar(query, k=5)


class TestLargeScalePerformance:
    """Tests for handling large datasets (>10k spectra).

    These tests verify that the implementation handles large-scale
    computations efficiently without memory issues.
    """

    @pytest.mark.slow
    def test_large_batch_cosine(self):
        """Batch cosine should handle 10k+ spectra."""
        X = np.random.randn(10000, 100)
        Y = np.random.randn(100, 100)

        sim = batch_cosine_similarity(Y, X)

        assert sim.shape == (100, 10000)
        assert np.all(np.isfinite(sim))

    @pytest.mark.slow
    def test_large_spectral_similarity(self):
        """SpectralSimilarity should handle 10k+ reference spectra."""
        reference = np.random.randn(10000, 100)
        queries = np.random.randn(100, 100)

        similarity = SpectralSimilarity(chunk_size=2000)
        similarity.fit(reference)
        results = similarity.find_similar(queries, k=10)

        assert results.indices.shape == (100, 10)
        assert results.scores.shape == (100, 10)

    @pytest.mark.slow
    def test_chunked_pairwise(self):
        """Pairwise similarity should use chunking for large sets."""
        reference = np.random.randn(5000, 50)

        similarity = SpectralSimilarity(chunk_size=1000)
        similarity.fit(reference)
        sim_matrix = similarity.pairwise_similarity()

        assert sim_matrix.shape == (5000, 5000)
        # Verify symmetry (proves chunking worked correctly)
        np.testing.assert_array_almost_equal(
            sim_matrix[:100, :100],
            sim_matrix[:100, :100].T
        )


class TestWithFixtureData:
    """Tests using real spectral fixture data."""

    @pytest.fixture
    def spectra_fixture(self, fixtures_path: Path):
        """Load spectra from test fixtures."""
        import pandas as pd

        spectra_path = (
            fixtures_path / "loupe" / "sol_0852" / "detail_1" /
            "SrlcSpecSpecSohRaw_0742604031-08149-1_Loupe_working" /
            "activeSpectra.csv"
        )

        if not spectra_path.exists():
            pytest.skip(f"Fixture not found: {spectra_path}")

        df = pd.read_csv(spectra_path)
        # Get just the R1 channels (first ~500 columns)
        r1_cols = [c for c in df.columns if c.startswith("R1_")]
        return df[r1_cols].values

    def test_cosine_with_real_spectra(self, spectra_fixture):
        """Cosine similarity on real spectral data."""
        spectra = spectra_fixture[:50]  # Use first 50

        sim_matrix = batch_cosine_similarity(spectra)

        assert sim_matrix.shape == (50, 50)
        # Real spectra should have positive similarities (non-negative intensities)
        assert np.all(sim_matrix >= -0.1)  # Allow small negative for normalized data

    def test_correlation_with_real_spectra(self, spectra_fixture):
        """Correlation on real spectral data."""
        spectra = spectra_fixture[:50]

        corr_matrix = batch_correlation(spectra)

        assert corr_matrix.shape == (50, 50)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(50))

    def test_find_similar_with_real_spectra(self, spectra_fixture):
        """Find similar spectra in real data."""
        reference = spectra_fixture[:100]
        query = reference[42]

        result = find_similar_spectra(query, reference, k=10)

        # Should find itself as most similar
        assert result.indices[0] == 42

    def test_spectral_similarity_with_real_spectra(self, spectra_fixture):
        """SpectralSimilarity class on real spectral data."""
        spectra = spectra_fixture[:100]

        similarity = SpectralSimilarity(metric="cosine")
        similarity.fit(spectra)

        # Query with a spectrum from the set
        result = similarity.find_similar(spectra[10], k=5)

        assert result.indices[0] == 10
        assert len(result.indices) == 5

    def test_euclidean_with_real_spectra(self, spectra_fixture):
        """Euclidean distance on real spectral data."""
        spectra = spectra_fixture[:50]

        distances = batch_euclidean_distance(spectra)

        assert distances.shape == (50, 50)
        np.testing.assert_array_almost_equal(np.diag(distances), np.zeros(50))


class TestSimilarityResult:
    """Tests for SimilarityResult dataclass."""

    def test_top_matches_single_query(self):
        """top_matches should work for single query."""
        result = SimilarityResult(
            indices=np.array([5, 3, 8]),
            scores=np.array([0.95, 0.85, 0.75]),
        )

        matches = result.top_matches
        assert matches == [(5, 0.95), (3, 0.85), (8, 0.75)]

    def test_top_matches_multiple_queries(self):
        """top_matches should return first query for multiple queries."""
        result = SimilarityResult(
            indices=np.array([[5, 3], [8, 2]]),
            scores=np.array([[0.95, 0.85], [0.75, 0.65]]),
        )

        matches = result.top_matches
        assert matches == [(5, 0.95), (3, 0.85)]

    def test_len(self):
        """len should return number of queries."""
        result = SimilarityResult(
            indices=np.array([1, 2, 3]),
            scores=np.array([0.9, 0.8, 0.7]),
        )
        assert len(result) == 3


class TestSimilarityConfig:
    """Tests for SimilarityConfig Pydantic model."""

    def test_default_values(self):
        """Default config should have sensible defaults."""
        config = SimilarityConfig()

        assert config.metric == SimilarityMetric.COSINE
        assert config.chunk_size == 2000
        assert config.normalize is True

    def test_custom_values(self):
        """Should accept custom configuration."""
        config = SimilarityConfig(
            metric=SimilarityMetric.CORRELATION,
            chunk_size=5000,
            min_similarity=0.8,
        )

        assert config.metric == SimilarityMetric.CORRELATION
        assert config.chunk_size == 5000
        assert config.min_similarity == 0.8

    def test_from_config(self):
        """SpectralSimilarity should accept config object."""
        config = SimilarityConfig(metric=SimilarityMetric.EUCLIDEAN)
        similarity = SpectralSimilarity(config=config)

        assert similarity.config.metric == SimilarityMetric.EUCLIDEAN
