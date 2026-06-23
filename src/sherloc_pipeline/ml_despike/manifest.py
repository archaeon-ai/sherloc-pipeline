"""Frozen identity of the ML cosmic-ray detector (v1.3, "v13c").

The v1.3 model attaches to one exact artifact, numeric path, and
operating point. This module is the single in-repo source for that
identity: artifact digests, per-region decision thresholds (taus),
detection windows, and the public download location.

Everything here is deliberately a frozen in-code constant — not user
configuration, not CLI-tunable, not environment-readable. Changing the
operating point requires a code change that diffs in review, and any
such change must be re-verified through the parity gate
(``scripts/verify_ml_despike_parity.py``).

This module imports nothing beyond the standard library, so the frozen
identity is available everywhere (e.g. for provenance reporting) without
pulling the optional ``[ml-despike]`` runtime.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple


@dataclass(frozen=True)
class ModelManifest:
    """Frozen identity of an ML CR-detection model artifact.

    Field values default to the v1.3 ("v13c") release; tests pin them at
    full precision. The mapping-typed fields are wrapped in
    ``MappingProxyType`` so the manifest is immutable in depth, not just
    at the dataclass surface.
    """

    name: str = "v1_stageB_v13c"
    artifact_filename: str = "v1_stageB_v13c.onnx"
    #: Provenance identity recorded with every run and persisted mask.
    #: The v1.3 retrain keeps the v1.x family provenance lineage.
    provenance_label: str = "ml_v1.3_tau_matched"
    #: sha256 of the fp32 ONNX deployment artifact (the file that is
    #: fetched, cached, and executed).
    sha256: str = "9668a0b2ca257ce333d57e3f76598dda8cb5c1839e2fde6bd955086d959be0ba"
    #: sha256 of the source torch checkpoint the ONNX artifact was
    #: exported from. Recorded for provenance only — the checkpoint is
    #: never fetched or loaded by the pipeline.
    checkpoint_sha256: str = "a77cd435d65631a8728c9d39c01c31dd30805ac37062b8c48937be6fb3594881"
    #: Frozen per-region decision thresholds (rate-matched).
    #: R2 and R3 share the fluorescence tau.
    tau: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(
            {
                "R1": 0.29882812500038747,
                "R2": 0.2656250000008831,
                "R3": 0.2656250000008831,
            }
        )
    )
    n_channels: int = 2148
    #: Detection windows ``[lo, hi)`` per region. Flags exist only inside
    #: these windows; they are part of the frozen operating point and
    #: cannot be extended. Coverage/applicability rules for combined
    #: representations derive from these values.
    region_windows: Mapping[str, Tuple[int, int]] = field(
        default_factory=lambda: MappingProxyType(
            {
                "R1": (52, 575),
                "R2": (575, 1677),
                "R3": (1677, 2140),
            }
        )
    )
    opset: int = 18
    #: The reference runtime configuration. Later 1.x onnxruntime
    #: versions are operationally acceptable; the installed version is
    #: recorded in run provenance.
    certified_runtime: str = "onnxruntime 1.26.0, CPUExecutionProvider, fp32"
    #: Public, versioned artifact host (GitHub release asset under the
    #: non-v* model-artifact hosting tag).
    download_url: str = (
        "https://github.com/archaeon-ai/sherloc-pipeline/releases/download/"
        "model-cr-despike-1.3/v1_stageB_v13c.onnx"
    )


#: The v1.3 ("v13c") manifest used by the pipeline. Tests may build
#: variant instances (e.g. for stub models) via ``dataclasses.replace``;
#: this constant itself is immutable.
DEFAULT_MANIFEST = ModelManifest()
