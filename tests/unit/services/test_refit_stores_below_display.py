"""A refit, with the bundled configuration and no floor option, stores Raman components with
SNR from 2.0 to 3.0 in all three domains.

The refit command builds its context with ``RuntimeContext.bootstrap(data_dir=..., results_dir=...)``
and runs ``PipelineService.run_full_pipeline`` for each scan. Here the fitting stages are
replaced by the fixture's per-point tables, and the persistence stage is the real one.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sherloc_pipeline.database.connection import create_all_tables, get_engine, get_session
from sherloc_pipeline.database.models import FittedPeakORM
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.pipeline import PipelineService
from sherloc_pipeline.services.runtime import RuntimeContext
from tests.unit.services.test_raman_persistence import populated_db, results_all  # noqa: F401
from tests.unit.services.test_store_min_snr import TABLES, _add_components


@pytest.fixture
def engine(tmp_path):
    """A database file, since the pipeline opens the database by its path."""
    path = tmp_path / "phase.db"
    eng = get_engine(str(path))
    create_all_tables(eng)
    eng.path = path
    return eng


def _ok(*args, **kwargs):
    return ServiceResult(summary="ok", metadata={})


def test_refit_with_the_bundled_configuration_stores_below_the_display_threshold(
        tmp_path, monkeypatch, engine, populated_db, results_all):
    for domain in TABLES:
        _add_components(results_all, domain)
    _, spectrum_ids, _ = populated_db
    monkeypatch.setenv("SHERLOC_DB_PATH", str(engine.path))

    context = RuntimeContext.bootstrap(data_dir=tmp_path / "data", results_dir=tmp_path / "out")

    scan_ctx = SimpleNamespace(base_data_dir=results_all, results_dir=results_all)
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_all
    fit = "sherloc_pipeline.services.fitting.FittingService."
    with patch("sherloc_pipeline.services.preprocessing.PreprocessingService.run_scan", _ok), \
         patch(fit + "fit_minerals", _ok), patch(fit + "fit_organics", _ok), \
         patch(fit + "fit_hydration", _ok), patch(fit + "fit_fluorescence", _ok), \
         patch(fit + "fit_averages", _ok), \
         patch("sherloc_pipeline.services.spatial.SpatialService.render_overlay", _ok), \
         patch("sherloc_pipeline.services.fitting.resolve_scan_context", return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=ingestion):
        result = PipelineService(context=context).run_full_pipeline(
            sol="0851", target="Lake_Haiyaha", scan="detail_1")

    with get_session(engine) as session:
        stored = {(p.fit_modality, p.spectrum_id, p.center_cm1, p.snr)
                  for p in session.query(FittedPeakORM).all()}
    for domain, (_, _, center, _) in TABLES.items():
        assert (domain, spectrum_ids[0], center - 60.0, 2.5) in stored, (domain, result.metadata)
        # The narrow component stays out: the FWHM gate is unchanged.
        assert not any(row[0] == domain and row[2] == center + 120.0 for row in stored)
    assert context.config.fitting["store_min_snr"] == 2.0
