"""The storage SNR floor (``fitting.store_min_snr``) apart from the display threshold
(``fitting.min_snr``).

The shipped floor is 2.0, the fit's own floor, with the display threshold at 3.0. With the
floor at ``min_snr`` the stored rows are those of a configuration without ``store_min_snr``,
where storage used ``min_snr`` (release 7.0.0), in all three Raman domains. A run floor stores
components between it and 3 from the existing per-point tables, through
``persist-peaks --store-min-snr``, without refitting; the FWHM gate is unchanged.
"""
import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_session
from sherloc_pipeline.database.models import FittedPeakORM
from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.fitting import store_snr_floor
from tests.unit.services.test_raman_persistence import (  # noqa: F401  (pytest fixtures)
    _make_service,
    engine,
    populated_db,
    results_all,
)

CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text())["fitting"]

#: Per-point table names, as the fitters write them, for point 0 of the fixture scan.
TABLES = {
    "minerals": ("minerals_fit", "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", 1085.5, 30.0),
    "organics": ("organics_fit", "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", 1600.0, 50.0),
    "hydration": ("hydration_fit", "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", 3500.0, 60.0),
}


def _persist(service, results_base, domain, *, fitting=None, store_min_snr=None):
    """persist_raman_peaks on the fixture scan, with an optional fitting configuration."""
    mock_ctx = MagicMock()
    mock_ctx.base_data_dir = results_base
    mock_ctx.results_dir = results_base
    mock_di = MagicMock()
    mock_di.get_results_path.return_value = results_base
    if fitting is not None:
        service.context.config.fitting = fitting
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context", return_value=mock_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=mock_di):
        return service.persist_raman_peaks(sol="0851", target="Lake_Haiyaha", scan="detail_1",
                                           domain=domain, store_min_snr=store_min_snr)


def _rows(engine):
    with get_session(engine) as session:
        return sorted(
            (p.spectrum_id, p.fit_modality, p.center_cm1, p.fwhm_cm1, p.amplitude, p.snr,
             p.mineral_assignment, p.fit_quality, p.kept_by)
            for p in session.query(FittedPeakORM).all()
        )


def _add_components(results_base, domain):
    """Point 0 gains a component at SNR 2.5 and one at exactly 3.0 (both wide enough), and one
    at SNR 2.5 narrower than the FWHM gate."""
    subdir, name, center, fwhm = TABLES[domain]
    path = results_base / subdir / name
    df = pd.read_csv(path)
    extra = pd.DataFrame([
        {"center_cm1": center - 60.0, "fwhm_cm1": fwhm, "amplitude_a": 250.0, "area": 20.0, "snr": 2.5},
        {"center_cm1": center + 60.0, "fwhm_cm1": fwhm, "amplitude_a": 300.0, "area": 24.0, "snr": 3.0},
        {"center_cm1": center + 120.0, "fwhm_cm1": 20.0, "amplitude_a": 250.0, "area": 8.0, "snr": 2.5},
    ])
    pd.concat([df, extra], ignore_index=True).to_csv(path, index=False)


def test_shipped_settings_store_everything_the_fit_reports():
    """The shipped floor equals the fit's own floor; the display threshold and the hydration
    point gates keep their values."""
    assert CONFIG["store_min_snr"] == 2.0
    assert CONFIG["store_min_snr"] == CONFIG["min_display_snr"]
    assert CONFIG["min_snr"] == 3.0
    assert CONFIG["hydration_min_snr"] == 3.0
    assert CONFIG["hydration_r2_min"] == 0.25
    assert CONFIG["reviewable_fwhm_min_cm1"] == 25.0
    assert store_snr_floor(CONFIG) == 2.0


def test_floor_falls_back_to_min_snr_and_a_run_override_wins():
    assert store_snr_floor({"min_snr": 3.0}) == 3.0
    assert store_snr_floor({"min_snr": 4.0}) == 4.0
    assert store_snr_floor({"min_snr": 3.0, "store_min_snr": 2.0}) == 2.0
    assert store_snr_floor({"min_snr": 3.0, "store_min_snr": 2.0}, 2.5) == 2.5
    assert store_snr_floor({}) == 3.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), "x", None])
def test_an_invalid_floor_is_refused(bad):
    with pytest.raises(FittingError):
        store_snr_floor({"min_snr": 3.0, "store_min_snr": bad})


@pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
def test_a_floor_equal_to_min_snr_stores_what_storage_at_min_snr_stored(tmp_path, domain, populated_db, results_all):
    """store_min_snr 3.0 against a configuration without store_min_snr (storage at min_snr, as in 7.0.0)."""
    _add_components(results_all, domain)
    engine, _, _ = populated_db
    service = _make_service(engine)
    before = copy.deepcopy(service.context.config.fitting)
    legacy = {k: v for k, v in before.items() if k != "store_min_snr"}
    at_min_snr = dict(before, store_min_snr=before["min_snr"])

    _persist(service, results_all, domain, fitting=legacy)
    old = _rows(engine)
    _persist(service, results_all, domain, fitting=at_min_snr)
    new = _rows(engine)

    assert new == old
    snrs = [r[5] for r in new]
    assert 2.5 not in snrs and 3.0 in snrs


@pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
def test_shipped_settings_store_the_component_below_the_display_threshold(domain, populated_db, results_all):
    _add_components(results_all, domain)
    engine, spectrum_ids, _ = populated_db
    service = _make_service(engine)
    _, _, center, _ = TABLES[domain]

    shipped = {k: CONFIG[k] for k in ("min_snr", "store_min_snr", "reviewable_fwhm_min_cm1")}
    result = _persist(service, results_all, domain,
                      fitting=dict(service.context.config.fitting, **shipped))
    rows = _rows(engine)
    assert result.metadata["min_snr_threshold"] == 2.0
    assert (spectrum_ids[0], center - 60.0, 2.5) in [(r[0], r[2], r[5]) for r in rows]
    # The narrow component stays out: the FWHM gate is unchanged.
    assert all(r[2] != center + 120.0 for r in rows)


@pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
def test_a_run_floor_re_stores_from_the_point_tables(domain, populated_db, results_all):
    _add_components(results_all, domain)
    engine, spectrum_ids, _ = populated_db
    service = _make_service(engine)
    _, _, center, _ = TABLES[domain]

    shipped = _persist(service, results_all, domain, store_min_snr=3.0)
    at_shipped = _rows(engine)
    lowered = _persist(service, results_all, domain, store_min_snr=2.0)
    at_two = _rows(engine)

    assert lowered.metadata["peaks_inserted"] == shipped.metadata["peaks_inserted"] + 1
    added = sorted(set(at_two) - set(at_shipped))
    assert [(r[0], r[2], r[5]) for r in added] == [(spectrum_ids[0], center - 60.0, 2.5)]
    # The narrow component stays out: the FWHM gate does not move with the SNR floor.
    assert all(r[2] != center + 120.0 for r in at_two)
    assert lowered.metadata["min_snr_threshold"] == 2.0

    # Back at 3.0 the domain's rows are replaced, not added to.
    _persist(service, results_all, domain, store_min_snr=3.0)
    assert _rows(engine) == at_shipped


def test_the_configured_floor_applies_without_an_override(populated_db, results_all):
    _add_components(results_all, "minerals")
    engine, _, _ = populated_db
    service = _make_service(engine)
    fitting = dict(service.context.config.fitting, store_min_snr=2.0)
    result = _persist(service, results_all, "minerals", fitting=fitting)
    assert 2.5 in [r[5] for r in _rows(engine)]
    assert result.metadata["min_snr_threshold"] == 2.0


def test_persist_peaks_passes_the_floor_and_refuses_a_bad_one(tmp_path):
    from sherloc_pipeline.cli.app import app

    db = tmp_path / "phase.db"
    db.write_bytes(b"")
    runner = CliRunner()
    bad = runner.invoke(app, ["persist-peaks", "--domain", "minerals", "--sol", "851", "--scan", "detail_1",
                              "--database", str(db), "--store-min-snr", "-1"])
    assert bad.exit_code == 1
    assert "--store-min-snr" in bad.output

    seen = {}

    def fake_persist(self, **kwargs):
        seen.update(kwargs)
        from sherloc_pipeline.services.base import ServiceResult
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 0})

    with patch("sherloc_pipeline.services.fitting.FittingService.persist_raman_peaks", fake_persist), \
         patch("sherloc_pipeline.cli.app._resolve_single_scan", return_value=("Lake_Haiyaha", None)):
        runner.invoke(app, ["persist-peaks", "--domain", "minerals", "--sol", "851", "--target", "Lake_Haiyaha",
                            "--scan", "detail_1", "--database", str(db), "--store-min-snr", "2.0"])
    assert seen.get("store_min_snr") == 2.0
