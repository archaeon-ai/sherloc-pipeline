"""A refit, with the bundled configuration and no floor option, stores Raman components with
SNR from 2.0 to 3.0 in all three domains.

The refit command builds its context with ``RuntimeContext.bootstrap(data_dir=..., results_dir=...)``
and runs ``PipelineService.run_full_pipeline`` for each scan. Here the fitting stages are
replaced by the fixture's per-point tables, and the persistence stage is the real one.
"""
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_session, open_database
from sherloc_pipeline.database.models import FittedPeakORM
from sherloc_pipeline.services import refit_journal
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.pipeline import PipelineService
from sherloc_pipeline.services.runtime import RuntimeContext
from tests.unit.services.test_raman_persistence import populated_db, results_all  # noqa: F401
from tests.unit.services.test_store_min_snr import TABLES, _add_components


@pytest.fixture
def engine(tmp_path):
    """A database file, since the pipeline opens the database by its path."""
    path = tmp_path / "phase.db"
    eng = open_database(path, create_missing=True)
    eng.path = path
    return eng


def _ok(*args, **kwargs):
    return ServiceResult(summary="ok", metadata={})


def _stub_fitting(stack, results_all):
    """Replace the fitting stages; the per-point tables come from the fixture."""
    scan_ctx = SimpleNamespace(base_data_dir=results_all, results_dir=results_all)
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_all
    fit = "sherloc_pipeline.services.fitting.FittingService."
    for target in ("sherloc_pipeline.services.preprocessing.PreprocessingService.run_scan",
                   fit + "fit_minerals", fit + "fit_organics", fit + "fit_hydration",
                   fit + "fit_fluorescence", fit + "fit_averages",
                   "sherloc_pipeline.services.spatial.SpatialService.render_overlay"):
        stack.enter_context(patch(target, _ok))
    stack.enter_context(patch("sherloc_pipeline.services.fitting.resolve_scan_context", return_value=scan_ctx))
    stack.enter_context(patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=ingestion))


def _assert_stored_below_display(engine, spectrum_ids, detail):
    with get_session(engine) as session:
        stored = {(p.fit_modality, p.spectrum_id, p.center_cm1, p.snr)
                  for p in session.query(FittedPeakORM).all()}
    for domain, (_, _, center, _) in TABLES.items():
        assert (domain, spectrum_ids[0], center - 60.0, 2.5) in stored, (domain, detail)
        # The narrow component stays out: the FWHM gate is unchanged.
        assert not any(row[0] == domain and row[2] == center + 120.0 for row in stored)


def test_refit_pipeline_with_the_bundled_configuration_stores_below_the_display_threshold(
        tmp_path, monkeypatch, engine, populated_db, results_all):
    for domain in TABLES:
        _add_components(results_all, domain)
    _, spectrum_ids, _ = populated_db
    monkeypatch.setenv("SHERLOC_DB_PATH", str(engine.path))

    context = RuntimeContext.bootstrap(data_dir=tmp_path / "data", results_dir=tmp_path / "out")
    with ExitStack() as stack:
        _stub_fitting(stack, results_all)
        result = PipelineService(context=context).run_full_pipeline(
            sol="0851", target="Lake_Haiyaha", scan="detail_1")

    _assert_stored_below_display(engine, spectrum_ids, result.metadata)
    assert context.config.fitting["store_min_snr"] == 2.0


def test_sherloc_refit_without_a_floor_option_stores_below_the_display_threshold(
        tmp_path, monkeypatch, engine, populated_db, results_all):
    """The command itself: ``sherloc refit`` with no floor option, on the fixture database."""
    for domain in TABLES:
        _add_components(results_all, domain)
    _, spectrum_ids, _ = populated_db
    monkeypatch.delenv("SHERLOC_DB_PATH", raising=False)
    # The fitting stages are stubbed, so the run-completeness rule is not under test here.
    monkeypatch.setattr(refit_journal, "run_failures", lambda metadata: [])
    (tmp_path / "data").mkdir()
    (tmp_path / "out").mkdir()

    from sherloc_pipeline.cli.app import app
    with ExitStack() as stack:
        _stub_fitting(stack, results_all)
        result = CliRunner().invoke(app, [
            "--json", "refit", "--database", str(engine.path), "--data-dir", str(tmp_path / "data"),
            "--results-dir", str(tmp_path / "out"), "--target", "Lake_Haiyaha",
        ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result"]["processed"], result.stdout

    _assert_stored_below_display(engine, spectrum_ids, result.stdout)
