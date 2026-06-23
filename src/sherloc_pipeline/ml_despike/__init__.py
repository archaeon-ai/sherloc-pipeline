"""ML cosmic-ray despike backend.

Public surface of the ``ml_despike`` package: frozen model identity,
featurization, digest-pinned artifact resolution, and the ONNX CPU
detector. Importing this package does NOT import ``onnxruntime`` — the
runtime loads lazily at detector construction, so everything else
(manifest constants, featurization, provenance) works without the
``[ml-despike]`` extra installed.
"""

from sherloc_pipeline.ml_despike.artifact import (
    ArtifactDigestError,
    ArtifactFetchError,
    default_cache_dir,
    resolve_artifact,
)
from sherloc_pipeline.ml_despike.detector import MLCRDetector
from sherloc_pipeline.ml_despike.featurize import (
    N_CHANNELS,
    REGIONS,
    featurize,
    featurize_batch,
)
from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST, ModelManifest

__all__ = [
    "ArtifactDigestError",
    "ArtifactFetchError",
    "DEFAULT_MANIFEST",
    "MLCRDetector",
    "ModelManifest",
    "N_CHANNELS",
    "REGIONS",
    "default_cache_dir",
    "featurize",
    "featurize_batch",
    "resolve_artifact",
]
