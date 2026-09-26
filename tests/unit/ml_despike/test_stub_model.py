"""The test-time stub model stays loadable by the certified runtime.

The stub's IR version is pinned rather than left to the installed
``onnx`` release, whose default can outrun what onnxruntime loads.
"""

import pytest

from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

from .conftest import build_stub_model_bytes


def test_stub_ir_version_is_the_minimum_for_the_certified_opset():
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    model = onnx.load_from_string(build_stub_model_bytes())
    expected = helper.find_min_ir_version_for([helper.make_opsetid("", DEFAULT_MANIFEST.opset)])
    assert model.ir_version == expected
