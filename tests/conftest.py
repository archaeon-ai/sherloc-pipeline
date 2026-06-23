"""
Pytest configuration and fixtures for SHERLOC pipeline tests.

Fixtures provide:
- fixtures_path: Path to tests/fixtures directory
- manifest: Parsed manifest.json data
- test_context: RuntimeContext configured for test fixtures
- tmp_results: Temporary directory for test outputs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# Force a non-color terminal for the test session. typer/rich detect the
# CI=true / GITHUB_ACTIONS=true env vars (set by GitHub Actions runners) and
# enable ANSI escape sequences in --help output even when stdout is not a
# TTY. The CLI help-text tests use plain substring assertions
# (`assert "--single-peak" in result.output`) and break against rich-rendered
# output. TERM=dumb is the env signal rich respects to disable styling. This
# only affects in-process pytest invocations; shell-level `sherloc --help`
# in a normal terminal is unaffected.
os.environ["TERM"] = "dumb"

# Anchor the background-subtraction file at the in-tree fixture so the
# golden baseline pipeline (and any test that exercises preprocessing's
# default background) runs out of the box on a fresh clone. Production
# deployments override this via /etc/sherloc/<deployment>.env. Setting
# the env var before sherloc_pipeline imports below ensures the resolved
# default_file is correct on first config load.
os.environ.setdefault(
    "SHERLOC_BACKGROUND_DIR",
    str((Path(__file__).parent / "fixtures" / "background").resolve()),
)

# Suppress matplotlib's `Software` PNG metadata so test fixture PNGs stay
# byte-identical across matplotlib versions. Without this, every minor
# matplotlib bump (e.g. 3.10.8 → 3.10.9) flips the tEXt 'Software' chunk
# in fixtures/pipeline_outputs/ and pollutes `git status` after pytest
# runs. We only inject for PNG — PDF accepts the kwarg too but uses a
# different metadata schema, and SVG raises on unknown keys.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.figure import Figure as _SherlocFigure  # noqa: E402

_sherloc_orig_savefig = _SherlocFigure.savefig


def _sherloc_deterministic_savefig(self, fname, *args, **kwargs):
    fmt = kwargs.get("format")
    if fmt is None:
        try:
            fmt = str(fname).rsplit(".", 1)[-1].lower()
        except Exception:
            fmt = ""
    if fmt == "png":
        md = dict(kwargs.get("metadata") or {})
        md.setdefault("Software", None)
        kwargs["metadata"] = md
    return _sherloc_orig_savefig(self, fname, *args, **kwargs)


_SherlocFigure.savefig = _sherloc_deterministic_savefig

import pytest

from sherloc_pipeline.services.runtime import RuntimeContext


@pytest.fixture(scope="session")
def fixtures_path() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def manifest(fixtures_path: Path) -> Dict[str, Any]:
    """Load and return parsed manifest.json."""
    manifest_path = fixtures_path / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


@pytest.fixture
def test_context(fixtures_path: Path, tmp_path: Path) -> RuntimeContext:
    """Create RuntimeContext configured for test fixtures.
    
    - data_dir points to fixtures/loupe (for Loupe data loading)
    - results_dir points to a temporary directory
    """
    return RuntimeContext.bootstrap(
        data_dir=fixtures_path / "loupe",
        results_dir=tmp_path / "results",
    )


@pytest.fixture
def tmp_results(tmp_path: Path) -> Path:
    """Provide temporary directory for test outputs.
    
    Creates the directory if it doesn't exist.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@pytest.fixture(scope="session")
def golden_baseline_path(fixtures_path: Path) -> Path:
    """Return path to golden baseline directory for sol 921 detail_1."""
    return fixtures_path.parent / "golden" / "sol_921_detail_1"


@pytest.fixture(scope="session")
def background_paths(fixtures_path: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    """Return paths to background spectra files."""
    return {
        key: fixtures_path / bg["path"]
        for key, bg in manifest["backgrounds"].items()
    }


@pytest.fixture(scope="session")
def reference_paths(fixtures_path: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    """Return paths to reference spectra files."""
    return {
        ref["mineral"]: fixtures_path / ref["path"]
        for ref in manifest["reference"]
    }


@pytest.fixture(scope="session")
def loupe_datasets(fixtures_path: Path, manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return dataset metadata indexed by sol.

    Returns dict like:
        {"0921": {"sol": "0921", "target": "Amherst_Point", "scan": "detail_1", ...}, ...}
    """
    return {ds["sol"]: ds for ds in manifest["datasets"]}


# --- ML despike stub-detector fixtures (MLD-QUA-003 CI proxy) --------------
# The stub ONNX model is built at test time (never committed, MLD-SEC-001
# AC2) and honors the certified I/O contract, so it flows through the REAL
# artifact-resolution and detector chain; only the manifest digest and
# artifact location differ. Builders live in tests/unit/ml_despike/conftest.


@pytest.fixture(scope="session")
def stub_model_path(tmp_path_factory):
    from tests.unit.ml_despike.conftest import build_stub_model_bytes

    path = tmp_path_factory.mktemp("stub_model") / "stub-cr-model.onnx"
    path.write_bytes(build_stub_model_bytes())
    return path


@pytest.fixture(scope="session")
def stub_manifest(stub_model_path):
    import dataclasses
    import hashlib

    from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

    digest = hashlib.sha256(stub_model_path.read_bytes()).hexdigest()
    return dataclasses.replace(
        DEFAULT_MANIFEST,
        name="stub-cr-model",
        artifact_filename="stub-cr-model.onnx",
        sha256=digest,
        download_url="https://example.com/models/stub-cr-model.onnx",
    )


@pytest.fixture
def stub_detector_factory(stub_manifest, stub_model_path):
    """Callable building a real MLCRDetector over the stub artifact."""

    def factory():
        from sherloc_pipeline.ml_despike.detector import MLCRDetector

        return MLCRDetector(stub_manifest, artifact_path=stub_model_path)

    return factory


@pytest.fixture
def preprocessing_service_with_stub(stub_detector_factory, monkeypatch):
    """PreprocessingService whose _build_ml_detector seam yields the stub."""
    from sherloc_pipeline.services.preprocessing import PreprocessingService

    monkeypatch.setattr(
        PreprocessingService,
        "_build_ml_detector",
        lambda self: stub_detector_factory(),
    )
    return PreprocessingService()

