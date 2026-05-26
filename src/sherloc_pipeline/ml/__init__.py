"""
Machine Learning infrastructure for PHASE spectral analysis.

This package provides ML utilities for spectral similarity analysis,
clustering, and classification of SHERLOC spectroscopy data.

Module Structure:
    distance: Distance/similarity metrics for spectra comparison
    clustering: Clustering algorithms (KMeans, DBSCAN, Hierarchical)
    features: Feature extraction utilities for spectral data
    base: Base classes for ML models and transformers

Usage:
    >>> from sherloc_pipeline.ml import cosine_similarity, euclidean_distance
    >>> from sherloc_pipeline.ml import KMeansClusterer, DBSCANClusterer
    >>>
    >>> # Compare two spectra
    >>> similarity = cosine_similarity(spectrum1, spectrum2)
    >>>
    >>> # Cluster spectra
    >>> clusterer = KMeansClusterer(n_clusters=5)
    >>> labels = clusterer.fit_predict(spectra_matrix)
"""

# Distance metrics
from sherloc_pipeline.ml.distance import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    correlation_distance,
    manhattan_distance,
    spectral_angle_distance,
    compute_pairwise_distances,
    DistanceMetric,
)

# Clustering algorithms
from sherloc_pipeline.ml.clustering import (
    BaseClusterer,
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    ClusteringResult,
    compute_silhouette_score,
    compute_silhouette_samples,
)

# Feature extraction
from sherloc_pipeline.ml.features import (
    FeatureExtractor,
    PeakFeatureExtractor,
    StatisticalFeatureExtractor,
    SpectralFeatureExtractor,
    extract_peak_features,
)

# Similarity
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

# Base classes
from sherloc_pipeline.ml.base import (
    MLBaseModel,
    Transformer,
    Estimator,
    ModelPersistence,
)

__all__ = [
    # Distance metrics
    "cosine_similarity",
    "cosine_distance",
    "euclidean_distance",
    "correlation_distance",
    "manhattan_distance",
    "spectral_angle_distance",
    "compute_pairwise_distances",
    "DistanceMetric",
    # Similarity (batch-optimized)
    "SpectralSimilarity",
    "SimilarityResult",
    "SimilarityConfig",
    "SimilarityMetric",
    "batch_cosine_similarity",
    "batch_euclidean_distance",
    "batch_correlation",
    "batch_spectral_angle",
    "find_similar_spectra",
    # Clustering
    "BaseClusterer",
    "KMeansClusterer",
    "DBSCANClusterer",
    "HierarchicalClusterer",
    "ClusteringResult",
    "compute_silhouette_score",
    "compute_silhouette_samples",
    # Features
    "FeatureExtractor",
    "PeakFeatureExtractor",
    "StatisticalFeatureExtractor",
    "SpectralFeatureExtractor",
    "extract_peak_features",
    # Base classes
    "MLBaseModel",
    "Transformer",
    "Estimator",
    "ModelPersistence",
]
