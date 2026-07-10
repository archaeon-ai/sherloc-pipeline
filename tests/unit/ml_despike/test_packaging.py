"""Package-metadata tests for the ML despike dependency surface.

The base install stays lean
(no ML runtime, no torch), the ``[ml-despike]`` extra is exactly the
bounded onnxruntime pin, the torch ``[ml]`` (SAM) extra is unchanged,
and the stub-model builder dependency lives in ``[dev]`` only.
"""

import tomllib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pyproject():
    path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def test_base_dependencies_have_no_ml_runtime(pyproject):
    base = [dep.lower() for dep in pyproject["project"]["dependencies"]]
    assert not any("onnxruntime" in dep for dep in base)
    assert not any("torch" in dep for dep in base)
    assert not any(dep.startswith("onnx") for dep in base)


def test_ml_despike_extra_is_exactly_the_bounded_ort_pin(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    # The exact bounded range is normative: certified
    # at 1.26.0, later 1.x compatible-but-not-certified, no major jump.
    assert extras["ml-despike"] == ["onnxruntime>=1.26.0,<2.0"]


def test_torch_ml_extra_unchanged(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert extras["ml"] == ["torch>=2.0.0", "segment-anything>=1.0"]


def test_onnx_stub_builder_in_dev_only(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    dev_onnx = [dep for dep in extras["dev"] if dep.startswith("onnx")]
    assert dev_onnx == ["onnx>=1.21.0"]
    for name, deps in extras.items():
        if name in ("dev", "ml-despike"):
            continue
        assert not any(dep.lower().startswith("onnx") for dep in deps), name


def test_ci_installs_the_extra():
    """CI must exercise the real detector/fetch code against the stub
    (spec §4.8, key decision 9)."""
    ci = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
    assert "'.[dev,web,pds,ml-despike]'" in ci.read_text()
