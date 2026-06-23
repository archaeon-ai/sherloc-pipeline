"""Tests for the `backfill-masks` CLI command (masks-only ML-despike backfill).

The command runs preprocessing through the certified ML despike step and
persists the resulting masks to ``cosmic_ray_masks`` WITHOUT re-fitting. These
tests stub the service boundary (PreprocessingService.run_scan and
CRMaskService.persist_masks) and the DB scan iterator, asserting the
masks-only contract:

  * preprocessing is invoked per scan with despike_method="ml"
  * the exact despike provenance produced by run_scan is handed to
    persist_masks (no fitting service is involved at all)
  * the --sol filter narrows to a single sol (canary)
  * --dry-run touches no services
"""

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.config import reset_config
from sherloc_pipeline.services.base import ServiceResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_config():
    yield
    reset_config()


DESPIKE_MD = {
    "method": "ml_v1.3_tau_matched",
    "model_sha256": "deadbeef",
    "tau": {"R1": 0.2773},
    "masks": {"0": {"R1": [10, 11]}, "1": {"R1": []}},
}


def _make_stubs(monkeypatch, scans, despike_md=DESPIKE_MD):
    """Wire stub services + scan iterator; return a `seen` dict of calls."""
    seen = {"run_scan": [], "persist": []}

    class StubPre:
        def __init__(self, *a, **k):
            pass

        def run_scan(self, **kwargs):
            seen["run_scan"].append(kwargs)
            return ServiceResult(
                summary="stub preprocess", artifacts=[], warnings=[],
                metadata={"despike": despike_md},
            )

    class StubCR:
        def __init__(self, *a, **k):
            pass

        def persist_masks(self, **kwargs):
            seen["persist"].append(kwargs)
            return ServiceResult(
                summary="stub persist", artifacts=[], warnings=[],
                metadata={
                    "masks_inserted": 1,
                    "regions_skipped": 0,
                    "method": despike_md["method"],
                },
            )

    monkeypatch.setattr("sherloc_pipeline.cli.app._iter_all_scans",
                        lambda *a, **k: list(scans))
    monkeypatch.setattr(
        "sherloc_pipeline.services.preprocessing.PreprocessingService", StubPre)
    monkeypatch.setattr(
        "sherloc_pipeline.services.cr_masks.CRMaskService", StubCR)
    return seen


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "phase.db"
    p.write_bytes(b"")  # .exists() guard only; the iterator is stubbed
    return p


def test_help_registered_and_describes_masks_only():
    result = runner.invoke(app, ["backfill-masks", "--help"])
    assert result.exit_code == 0
    assert "cosmic_ray_masks" in result.output
    assert "Masks-only" in result.output


def test_missing_db_exits_nonzero(tmp_path):
    result = runner.invoke(
        app, ["backfill-masks", "--database", str(tmp_path / "nope.db")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_persists_masks_only_per_scan(monkeypatch, db, tmp_path):
    scans = [(921, "Amherst_Point", "detail_1"),
             (1000, "Foo_Bar", "line")]
    seen = _make_stubs(monkeypatch, scans)

    result = runner.invoke(app, [
        "backfill-masks",
        "--database", str(db),
        "--data-dir", str(tmp_path / "loupe"),
        "--results-dir", str(tmp_path / "scratch"),
    ])
    assert result.exit_code == 0, result.output

    # Preprocessing called once per scan, always with the certified ml method
    # and plots off (masks-only is lean).
    assert len(seen["run_scan"]) == 2
    for call in seen["run_scan"]:
        assert call["despike_method"] == "ml"
        assert call["generate_plots"] is False

    # The exact despike provenance from run_scan is what gets persisted —
    # and persist is the only DB-writing call (no fitting anywhere).
    assert len(seen["persist"]) == 2
    for call in seen["persist"]:
        assert call["despike_metadata"] is DESPIKE_MD
        assert call["database_path"] == db

    assert "Scans persisted: 2/2" in result.output
    assert "ml_v1.3_tau_matched" in result.output


def test_sol_filter_narrows_to_one_sol(monkeypatch, db, tmp_path):
    scans = [(921, "Amherst_Point", "detail_1"),
             (1000, "Foo_Bar", "line")]
    seen = _make_stubs(monkeypatch, scans)

    result = runner.invoke(app, [
        "backfill-masks",
        "--database", str(db),
        "--data-dir", str(tmp_path / "loupe"),
        "--results-dir", str(tmp_path / "scratch"),
        "--sol", "0921",
    ])
    assert result.exit_code == 0, result.output
    assert len(seen["run_scan"]) == 1
    assert seen["run_scan"][0]["sol"] == "0921"


def test_dry_run_touches_no_services(monkeypatch, db, tmp_path):
    scans = [(921, "Amherst_Point", "detail_1")]
    seen = _make_stubs(monkeypatch, scans)

    result = runner.invoke(app, [
        "backfill-masks", "--database", str(db), "--dry-run",
    ])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert seen["run_scan"] == []
    assert seen["persist"] == []


def test_run_scan_failure_recorded_nonzero_exit(monkeypatch, db, tmp_path):
    scans = [(921, "Amherst_Point", "detail_1")]
    seen = {"persist": []}

    class BoomPre:
        def __init__(self, *a, **k):
            pass

        def run_scan(self, **kwargs):
            raise RuntimeError("workspace missing")

    class StubCR:
        def __init__(self, *a, **k):
            pass

        def persist_masks(self, **kwargs):
            seen["persist"].append(kwargs)
            return ServiceResult(summary="", artifacts=[], warnings=[],
                                 metadata={})

    monkeypatch.setattr("sherloc_pipeline.cli.app._iter_all_scans",
                        lambda *a, **k: list(scans))
    monkeypatch.setattr(
        "sherloc_pipeline.services.preprocessing.PreprocessingService", BoomPre)
    monkeypatch.setattr(
        "sherloc_pipeline.services.cr_masks.CRMaskService", StubCR)

    result = runner.invoke(app, [
        "backfill-masks", "--database", str(db),
        "--results-dir", str(tmp_path / "scratch"),
    ])
    assert result.exit_code == 1
    assert "workspace missing" in result.output
    assert seen["persist"] == []  # a failed preprocess never persists
