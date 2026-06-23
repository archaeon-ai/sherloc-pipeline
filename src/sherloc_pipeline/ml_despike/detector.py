"""ONNX inference for the certified ML CR detector (v1.1).

Executes the certified numeric path exactly (MLD-DET-004): CPU execution
provider in the certified session configuration, probabilities as the
float64 sigmoid of the model logits, and flags as the channels within the
region's certified window ``[lo, hi)`` whose probability strictly exceeds
the region's frozen tau — returned as absolute channel indices (0–2147).

``onnxruntime`` is imported lazily at detector construction
(MLD-DET-005); importing this module never imports the runtime, so the
package stays importable without the ``[ml-despike]`` extra.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from sherloc_pipeline.ml_despike.artifact import resolve_artifact
from sherloc_pipeline.ml_despike.featurize import featurize_batch
from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST, ModelManifest

logger = logging.getLogger(__name__)

#: Certified inference batch size (G7 benchmark sweet spot). Internal
#: constant, not configuration.
_BATCH_SIZE = 16

_INSTALL_HINT = (
    "ML despike extra not installed. Install with: "
    "pip install 'sherloc-pipeline[ml-despike]'"
)


def _require_onnxruntime() -> Any:
    """Import ``onnxruntime`` lazily, naming the extra if it is missing."""
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return onnxruntime


class MLCRDetector:
    """Certified v1.1 CR detector over raw ACTIVE/DARK plane pairs.

    Construction resolves (and digest-verifies) the model artifact and
    creates the ONNX Runtime session in the certified configuration:
    ``CPUExecutionProvider``, 2 intra-op / 1 inter-op threads, full graph
    optimization. No GPU execution provider is ever requested.

    Args:
        manifest: Frozen artifact identity and operating point.
        artifact_path: Optional explicit artifact location (verified, no
            bypass); default resolves via the fetch-and-cache path.
        intra_op_threads: ORT intra-op thread count. The certified
            configuration and default is 2.
    """

    def __init__(
        self,
        manifest: ModelManifest = DEFAULT_MANIFEST,
        artifact_path: Optional[Path] = None,
        intra_op_threads: int = 2,
    ):
        ort = _require_onnxruntime()
        path = resolve_artifact(manifest, artifact_path=artifact_path)

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = ort.InferenceSession(
            str(path), session_options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        self.manifest = manifest
        self.artifact_path = path
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = 1
        #: Installed runtime version, recorded in run provenance
        #: (MLD-SYS-012/MLD-QUA-009 attribution path).
        self.ort_version: str = ort.__version__

    def detect(
        self,
        actives: Sequence[np.ndarray],
        darks: Sequence[np.ndarray],
        regions: Sequence[str],
    ) -> List[np.ndarray]:
        """Flag CR channels for a batch of raw ACTIVE/DARK frame pairs.

        Frames may belong to any mix of regions; a call carrying only R1
        frames is a first-class path (MLD-SYS-006). Frames are featurized
        with the certified 8-channel representation and processed through
        the session in internal batches.

        Args:
            actives: Raw ACTIVE planes (1-D length-2148, raw DN), one per
                frame.
            darks: Raw DARK planes, parallel to ``actives``.
            regions: Region of each frame (``"R1"``/``"R2"``/``"R3"``).

        Returns:
            One int64 array of sorted absolute channel indices per input
            frame: the channels in the region's certified window whose
            probability strictly exceeds the region's tau.
        """
        features = featurize_batch(actives, darks, regions)
        n_frames = features.shape[0]

        masks: List[np.ndarray] = []
        for start in range(0, n_frames, _BATCH_SIZE):
            batch = features[start : start + _BATCH_SIZE]
            logits = self._session.run(None, {self._input_name: batch})[0]
            probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            for i in range(batch.shape[0]):
                region = regions[start + i]
                lo, hi = self.manifest.region_windows[region]
                tau = self.manifest.tau[region]
                relative = np.where(probabilities[i, lo:hi] > tau)[0]
                masks.append((relative + lo).astype(np.int64))
        return masks
