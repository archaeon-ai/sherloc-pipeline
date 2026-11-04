"""sherloc refit end to end on the sol_0921 Loupe fixture workspace and a synthetic PDS4 scan.

Preprocessing, despike, Raman persistence, review and the journal are real.
Fitting and spatial rendering are stubbed (see test_despike_methods_e2e).
"""

import copy
import json
import uuid

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app
from sherloc_pipeline.config import get_config
from sherloc_pipeline.database.connection import create_all_tables, get_engine, get_session
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.ingestion import IngestionService

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)
TARGET = "Refit_E2E"


@pytest.fixture
def restore_config():
    cfg = get_config()
    saved = {name: copy.deepcopy(getattr(cfg, name)) for name in ("preprocessing", "fitting")}
    yield
    for name, value in saved.items():
        setattr(cfg, name, value)


def _ingest(db_path, fixtures_path, include_spectra):
    service = IngestionService(
        database_path=db_path, include_spectra=include_spectra, ingestion_mode="all_regions")
    service.ingest_workspace(fixtures_path / WORKSPACE_REL)
    with get_session(service.engine) as session:
        session.query(ScanORM).first().target = TARGET.replace("_", " ")
    service.engine.dispose()
    return db_path


@pytest.fixture
def loupe_db(tmp_path, fixtures_path, monkeypatch):
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
    return _ingest(tmp_path / "phase.db", fixtures_path, include_spectra=True)


@pytest.fixture
def stub_raman_fits(monkeypatch, stub_detector_factory):
    from sherloc_pipeline.services.fitting import FittingService
    from sherloc_pipeline.services.preprocessing import PreprocessingService
    from sherloc_pipeline.services.spatial import SpatialService

    def empty_result(*args, **kwargs):
        return ServiceResult(summary="stubbed", artifacts=[], warnings=[], metadata={})

    for name in ("fit_minerals", "fit_organics", "fit_hydration", "fit_averages"):
        monkeypatch.setattr(FittingService, name, empty_result)
    monkeypatch.setattr(SpatialService, "render_overlay", empty_result)
    monkeypatch.setattr(PreprocessingService, "_build_ml_detector", lambda self: stub_detector_factory())


@pytest.fixture
def stub_heavy_stages(monkeypatch, stub_raman_fits):
    from sherloc_pipeline.services.fitting import FittingService

    def fluorescence(*args, **kwargs):
        return ServiceResult(summary="stubbed", metadata={"points_fitted": 0, "points_missing_regions": 0, "total_scan_points": 1})

    monkeypatch.setattr(FittingService, "fit_fluorescence", fluorescence)


@pytest.fixture
def stub_persistence(monkeypatch):
    import os

    from sherloc_pipeline.services.fitting import FittingService

    seen = []

    def persisted(self, **kwargs):
        seen.append(os.environ.get("SHERLOC_DB_PATH"))
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 0, "csvs_without_spectrum": 0})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", persisted)
    return seen


def _refit(*args):
    result = CliRunner().invoke(app, ["--json", "refit", *args])
    # stdout must be exactly one JSON document.
    payload = json.loads(result.stdout) if result.stdout.strip() else None
    return result, payload


def _journal_scans(path):
    return [json.loads(line) for line in path.read_text().splitlines()[1:]]


def test_records_complete_scans_and_resume_skips_them(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, stub_persistence, restore_config, monkeypatch,
):
    journal = tmp_path / "batch.jsonl"
    common = ["--database", str(loupe_db), "--data-dir", str(fixtures_path / "loupe"),
              "--results-dir", str(tmp_path / "results")]

    result, payload = _refit(*common, "--journal", str(journal))
    assert result.exit_code == 0, result.output
    assert payload["result"] | {"journal": None} == {
        "journal": None, "scans": 1, "processed": 1, "skipped": 0, "failed": 0, "errors": []}
    assert [(e["sol"], e["scan"]) for e in _journal_scans(journal)] == [(921, "detail_1")]
    assert stub_persistence == [str(loupe_db.resolve())] * 3

    from sherloc_pipeline.services.pipeline import PipelineService

    calls = []
    monkeypatch.setattr(PipelineService, "run_full_pipeline", lambda self, **kw: calls.append(kw))
    result, payload = _refit(*common, "--resume", str(journal))
    assert result.exit_code == 0, result.output
    assert (payload["result"]["processed"], payload["result"]["skipped"]) == (0, 1)
    assert calls == []


def test_downgraded_persistence_is_not_recorded(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, restore_config,
):
    # Fitting is stubbed, so real Raman persistence finds no fit output, warns and continues.
    journal = tmp_path / "batch.jsonl"
    result, payload = _refit(
        "--database", str(loupe_db), "--data-dir", str(fixtures_path / "loupe"),
        "--results-dir", str(tmp_path / "results"), "--journal", str(journal),
    )
    assert result.exit_code == 1
    assert payload["result"]["failed"] == 1
    assert "raman_persistence" in payload["result"]["errors"][0]
    assert _journal_scans(journal) == []


def test_resume_refuses_a_different_effective_config(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, stub_persistence, restore_config,
):
    journal = tmp_path / "batch.jsonl"
    common = ["--database", str(loupe_db), "--data-dir", str(fixtures_path / "loupe"),
              "--results-dir", str(tmp_path / "results")]
    assert _refit(*common, "--journal", str(journal))[0].exit_code == 0
    result = CliRunner().invoke(app, ["refit", *common, "--resume", str(journal), "--trim-pct", "8"])
    assert result.exit_code == 1
    assert "config_digest" in result.output
    assert len(_journal_scans(journal)) == 1


def test_scan_without_loupe_inputs_fails_and_is_not_recorded(tmp_path, monkeypatch, restore_config):
    # A PDS4-ingested scan has no Loupe working directory, so full-pipeline stops at Step 1.
    db_path = tmp_path / "pds.db"
    engine = get_engine(db_path)
    create_all_tables(engine)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=4321, data_source="pds4"))
        session.flush()
        session.add(ScanORM(
            id=str(uuid.uuid4()), sol_number=4321, target=None, target_type="mars_target",
            scan_name="detail_1", scan_id="synthetic_pds", sclk_start=0, n_points=1,
            n_channels=2148, data_source="pds4",
        ))
    engine.dispose()
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
    journal = tmp_path / "batch.jsonl"
    result, payload = _refit(
        "--database", str(db_path), "--data-dir", str(tmp_path / "pds"),
        "--results-dir", str(tmp_path / "results"), "--journal", str(journal),
    )
    assert result.exit_code == 1
    assert payload["result"]["failed"] == 1
    assert "Preprocessing failed" in payload["result"]["errors"][0]
    assert _journal_scans(journal) == []


def test_journal_and_resume_are_exclusive(tmp_path, loupe_db):
    result = CliRunner().invoke(app, ["refit", "--database", str(loupe_db),
                                      "--journal", str(tmp_path / "a"), "--resume", str(tmp_path / "b")])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_scan_without_stored_spectra_is_not_recorded(
    tmp_path, fixtures_path, stub_raman_fits, stub_persistence, restore_config, monkeypatch,
):
    # Ingested without spectra: real fluorescence finds no R1-R3 inputs for any point.
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
    db_path = _ingest(tmp_path / "phase.db", fixtures_path, include_spectra=False)
    journal = tmp_path / "batch.jsonl"
    result, payload = _refit(
        "--database", str(db_path), "--data-dir", str(fixtures_path / "loupe"),
        "--results-dir", str(tmp_path / "results"), "--journal", str(journal),
        "--despike-method", "none",
    )
    assert result.exit_code == 1, result.output
    assert "have R1-R3 inputs" in payload["result"]["errors"][0]
    assert _journal_scans(journal) == []


def test_default_roots_come_from_config(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, stub_persistence, restore_config, monkeypatch,
):
    cfg = get_config()
    monkeypatch.setitem(cfg.paths, "data_root", str(fixtures_path / "loupe"))
    monkeypatch.setitem(cfg.paths, "results_root", str(tmp_path / "configured-results"))
    result, payload = _refit("--database", str(loupe_db))
    assert result.exit_code == 0, result.output
    header = json.loads(open(payload["result"]["journal"]).readline())
    assert header["data_root"] == str((fixtures_path / "loupe").resolve())
    assert header["results_root"] == str((tmp_path / "configured-results").resolve())
    assert payload["result"]["journal"].startswith(str((tmp_path / "configured-results").resolve()))


def test_json_output_with_overrides_and_json_errors(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, stub_persistence, restore_config,
):
    common = ["--database", str(loupe_db), "--data-dir", str(fixtures_path / "loupe"),
              "--results-dir", str(tmp_path / "results")]
    result, payload = _refit(*common, "--trim-pct", "8", "--model-selection", "ftest",
                             "--despike-method", "modz", "--allow-missing-background")
    assert result.exit_code == 0, result.output
    assert payload["command"] == "refit"

    for bad in (["--trim-pct", "80"], ["--model-selection", "other"]):
        result, payload = _refit(*common, *bad)
        assert result.exit_code == 1
        assert payload is None
        assert json.loads(result.stderr.strip().splitlines()[-1])["error_type"] == "RefitError"


def test_resume_ignores_ambient_db_path_when_database_is_given(
    tmp_path, fixtures_path, loupe_db, stub_heavy_stages, stub_persistence, restore_config, monkeypatch,
):
    journal = tmp_path / "batch.jsonl"
    common = ["--database", str(loupe_db), "--data-dir", str(fixtures_path / "loupe"),
              "--results-dir", str(tmp_path / "results")]
    monkeypatch.setitem(get_config().database, "path", str(tmp_path / "ambient-a.db"))
    assert _refit(*common, "--journal", str(journal))[0].exit_code == 0
    monkeypatch.setitem(get_config().database, "path", str(tmp_path / "ambient-b.db"))
    result, payload = _refit(*common, "--resume", str(journal))
    assert result.exit_code == 0, result.output
    assert payload["result"]["skipped"] == 1
