"""Import-isolation and lazy-runtime tests (MLD-DET-005, MLD-QUA-002).

The detector package must be importable everywhere without dragging the
optional ML runtime; the runtime import happens lazily at detector
construction and fails with an actionable message naming the
``[ml-despike]`` extra.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import importlib

from sherloc_pipeline import ml_despike
from sherloc_pipeline.ml_despike.detector import MLCRDetector

# The package __init__ re-exports the featurize *function*, shadowing the
# submodule attribute — resolve the actual modules via importlib.
featurize_mod = importlib.import_module("sherloc_pipeline.ml_despike.featurize")
manifest_mod = importlib.import_module("sherloc_pipeline.ml_despike.manifest")


def test_package_import_does_not_import_ml_runtime():
    """Fresh interpreter: importing the package (detector module included)
    must not load onnxruntime or torch (MLD-DET-005 AC1)."""
    code = (
        "import sys\n"
        "import sherloc_pipeline.ml_despike\n"
        "import sherloc_pipeline.ml_despike.detector\n"
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime was imported'\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print('ISOLATION-OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "ISOLATION-OK" in result.stdout


def _third_party_top_level_imports(module) -> set:
    """Top-level imported distributions of a module, stdlib excluded."""
    tree = ast.parse(Path(module.__file__).read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module)
    return {
        name
        for name in names
        if name.split(".")[0] not in sys.stdlib_module_names
        and not name.startswith("sherloc_pipeline.ml_despike")
    }


def test_featurize_module_imports_numpy_only():
    """MLD-DET-002 AC2: numpy-only import surface (no scipy, no pandas,
    no runtime, no experiment paths)."""
    assert _third_party_top_level_imports(featurize_mod) == {"numpy"}


def test_manifest_module_is_stdlib_only():
    assert _third_party_top_level_imports(manifest_mod) == set()


def test_missing_runtime_error_names_extra(monkeypatch, stub_manifest):
    """MLD-DET-005 AC2 / MLD-IFC-008 AC1: the no-runtime failure names the
    extra and the exact install command."""
    # A None entry makes `import onnxruntime` raise ImportError even when
    # the runtime is installed.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError) as excinfo:
        MLCRDetector(manifest=stub_manifest)
    message = str(excinfo.value)
    assert "[ml-despike]" in message
    assert "pip install 'sherloc-pipeline[ml-despike]'" in message


def test_public_api_exports():
    expected = {
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
    }
    assert set(ml_despike.__all__) == expected
    for name in expected:
        assert hasattr(ml_despike, name)
