"""Shared fixtures: a test-time-built stub ONNX model (never committed).

The stub honors the certified I/O contract — input ``"x"``
``(N, 8, 2148)`` float32, output ``"logits"`` ``(N, 2148)`` float32 — so
it can flow through the real artifact-resolution and detector chain
(no binary in git).

Graph: ``logits = x[:, 0, :] + w`` with deterministic seeded weights
``w ~ N(-4, 0.1)``. The −4 baseline keeps unspiked channels far below
the frozen taus (sigmoid(−4) ≈ 0.018), so tests control flags exactly by
engineering the normalized-active feature channel.
"""

import dataclasses
import hashlib

import numpy as np
import pytest

from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

STUB_SEED = 20260611


def stub_weights() -> np.ndarray:
    """The stub model's additive weight vector (deterministic)."""
    rng = np.random.default_rng(STUB_SEED)
    return rng.normal(-4.0, 0.1, size=DEFAULT_MANIFEST.n_channels).astype(np.float32)


def build_stub_model_bytes() -> bytes:
    """Serialize the stub ONNX graph (requires the dev-only ``onnx``)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    n_ch = DEFAULT_MANIFEST.n_channels
    gather = helper.make_node("Gather", ["x", "chan0"], ["x0"], axis=1)
    add = helper.make_node("Add", ["x0", "w"], ["logits"])
    graph = helper.make_graph(
        [gather, add],
        "stub_cr_detector",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 8, n_ch])
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_ch])
        ],
        initializer=[
            numpy_helper.from_array(np.array(0, dtype=np.int64), name="chan0"),
            numpy_helper.from_array(stub_weights(), name="w"),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", DEFAULT_MANIFEST.opset)]
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture(scope="session")
def stub_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("stub_model") / "stub-cr-model.onnx"
    path.write_bytes(build_stub_model_bytes())
    return path


@pytest.fixture(scope="session")
def stub_manifest(stub_model_path):
    """Manifest variant whose pinned digest is the stub's (computed in-test)."""
    digest = hashlib.sha256(stub_model_path.read_bytes()).hexdigest()
    return dataclasses.replace(
        DEFAULT_MANIFEST,
        name="stub-cr-model",
        artifact_filename="stub-cr-model.onnx",
        sha256=digest,
        download_url="https://example.com/models/stub-cr-model.onnx",
    )
