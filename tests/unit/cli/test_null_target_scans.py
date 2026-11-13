"""Scans stored with a NULL target are reachable by fit-fluor and persist-peaks (#175).

PDS ingest stores ``target`` as NULL when no target name resolves. The bulk
loops look such scans up by scan id, and single-scan mode accepts a missing
``--target`` when (sol, scan) names exactly one scan. All data is synthetic.
"""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_session, open_database
from sherloc_pipeline.database.models import (
    FittedPeakORM, ScanORM, ScanPointORM, SolORM, SpectrumORM,
)
from sherloc_pipeline.services import fluor_inputs
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.fitting import (
    FittingService, _fit_run_marker_path, _write_fit_run_marker,
)

cli = importlib.import_module("sherloc_pipeline.cli.app")

SOL = 76544
SCAN = "synthetic_scan"


def _add_scan(session, target, scan_name=SCAN, sol=SOL):
    scan_id = str(uuid4())
    session.add(ScanORM(
        id=scan_id, sol_number=sol, target=target, target_type="mars_target",
        scan_name=scan_name, scan_id=f"synthetic_{uuid4().hex[:8]}",
        sclk_start=0, n_points=2, n_channels=2148,
    ))
    session.flush()
    for point in range(2):
        point_id = str(uuid4())
        session.add(ScanPointORM(id=point_id, scan_id=scan_id, point_index=point))
        session.flush()
        for region in fluor_inputs.REGIONS:
            session.add(SpectrumORM(
                id=str(uuid4()), scan_point_id=point_id, region=region,
                spectrum_type="dark_subtracted", processing_level="raw",
                intensities=fluor_inputs._encode(np.full(2148, 31.0 + point)),
            ))
            session.flush()
    return scan_id


@pytest.fixture
def run(tmp_path, monkeypatch):
    from sherloc_pipeline.config import load_config
    from sherloc_pipeline.core import fluor_fitting
    from sherloc_pipeline.services.runtime import RuntimeContext

    assert not any(tmp_path.resolve().is_relative_to(root) for root in (
        Path("/data"), Path("/nas"),
    ))
    database_path = tmp_path / "synthetic.db"
    engine = open_database(database_path, create_missing=True)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=SOL, data_source="pds"))
        session.flush()
        scan_id = _add_scan(session, None)

    def stub(wavelength, intensity, **kwargs):
        return fluor_fitting.FluorFitResult(
            peaks=[fluor_fitting.FluorPeakFit(
                center_nm=314.0, amplitude=73.0, fwhm_nm=13.0, area=97.0, snr=19.0,
            )],
            r2=0.91, rss=11.0, aicc=7.0, n_peaks=1, is_saturated=False,
            fit_skipped=False, n_saturated_channels=0, n_masked_channels=0,
        )

    monkeypatch.setattr(fluor_fitting, "fit_fluorescence_spectrum", stub)
    cfg = copy.deepcopy(load_config())
    cfg.fluorescence_fitting = dict(cfg.fluorescence_fitting or {})
    cfg.fluorescence_fitting.update(parallel_workers=1, laser_normalize=False)
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg,
    )
    fitting = FittingService(context=context, database_path=database_path)

    def invoke(*args):
        def factory(**kwargs):
            fitting.console = kwargs["console"]
            return fitting

        with patch("sherloc_pipeline.services.fitting.FittingService", side_effect=factory):
            return CliRunner().invoke(cli.app, [
                "--json", *args, "--database", str(database_path),
            ])

    yield SimpleNamespace(
        engine=engine, database_path=database_path, fitting=fitting,
        scan_id=scan_id, invoke=invoke,
    )
    if fitting._engine is not None:
        fitting._engine.dispose()
    engine.dispose()


def _fluor_peak_count(engine, scan_id):
    with get_session(engine) as session:
        return session.query(FittedPeakORM).join(
            SpectrumORM, FittedPeakORM.spectrum_id == SpectrumORM.id,
        ).join(ScanPointORM, SpectrumORM.scan_point_id == ScanPointORM.id).filter(
            ScanPointORM.scan_id == scan_id,
            FittedPeakORM.fit_modality == "fluorescence",
        ).count()


# --- service ---------------------------------------------------------------

def test_fit_fluorescence_finds_a_null_target_scan_by_id(run):
    result = run.fitting.fit_fluorescence(
        f"{SOL:04d}", "", SCAN, generate_plots=False, scan_id=run.scan_id,
    )
    assert result.metadata["peaks_inserted"] == 2
    assert _fluor_peak_count(run.engine, run.scan_id) == 2


def test_fit_fluorescence_finds_a_unique_null_target_scan_without_id(run):
    result = run.fitting.fit_fluorescence(f"{SOL:04d}", "", SCAN, generate_plots=False)
    assert result.metadata["peaks_inserted"] == 2


def test_fit_fluorescence_refuses_an_ambiguous_null_target_scan(run):
    with get_session(run.engine) as session:
        _add_scan(session, None)
    with pytest.raises(FittingError, match="scan id"):
        run.fitting.fit_fluorescence(f"{SOL:04d}", "", SCAN, generate_plots=False)


def test_fit_fluorescence_rejects_a_scan_id_for_another_scan(run):
    with pytest.raises(FittingError, match="Scan not found"):
        run.fitting.fit_fluorescence(
            f"{SOL:04d}", "", "other_scan", generate_plots=False, scan_id=run.scan_id,
        )


def test_persist_raman_peaks_finds_a_null_target_scan_by_id(run, tmp_path):
    results_base = tmp_path / "results" / f"{SOL:04d}_{SCAN}"
    fit_dir = results_base / "minerals_fit"
    fit_dir.mkdir(parents=True)
    sol = f"{SOL:04d}"
    _write_fit_run_marker(
        _fit_run_marker_path(fit_dir, sol, "", SCAN, "R1", "minerals"),
        domain="minerals", points_fitted=2, accepted_peaks=1,
    )
    pd.DataFrame([{
        "center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0,
        "area": 100.0, "snr": 15.0,
    }]).to_csv(fit_dir / f"{sol}__{SCAN}_R1_point0_fit_peaks.csv", index=False)
    with get_session(run.engine) as session:
        session.query(SpectrumORM).filter_by(region="R1").update({"intensities": b"\x00"})

    scan_ctx = MagicMock(base_data_dir=results_base, results_dir=results_base)
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_base
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context", return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=ingestion):
        result = run.fitting.persist_raman_peaks(
            sol=sol, target="", scan=SCAN, domain="minerals", scan_id=run.scan_id,
        )
    assert result.metadata["peaks_inserted"] == 1


# --- CLI ---------------------------------------------------------------------

def test_fit_fluor_all_reaches_a_null_target_scan(run):
    result = run.invoke("fit-fluor", "--all", "--no-plots")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout.strip().splitlines()[-1])["result"]
    assert payload["errors"] == []
    assert payload["total_peaks"] == 2


def test_fit_fluor_single_scan_without_target(run):
    result = run.invoke("fit-fluor", "--sol", str(SOL), "--scan", SCAN, "--no-plots")
    assert result.exit_code == 0, result.output
    assert _fluor_peak_count(run.engine, run.scan_id) == 2


def test_fit_fluor_single_scan_without_target_needs_a_unique_scan(run):
    with get_session(run.engine) as session:
        _add_scan(session, "Synthetic Other")
    result = run.invoke("fit-fluor", "--sol", str(SOL), "--scan", SCAN, "--no-plots")
    assert result.exit_code == 1
    assert "--target" in result.output
    assert _fluor_peak_count(run.engine, run.scan_id) == 0


def test_bulk_loops_pass_the_scan_id(run, monkeypatch):
    calls = []

    def record(self, sol, target, scan, **kw):
        calls.append((sol, target, scan, kw.get("scan_id")))
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 0})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", record)
    monkeypatch.setattr(FittingService, "fit_fluorescence", record)
    monkeypatch.setattr(FittingService, "__init__", lambda self, **kw: None)
    for args in (
        ("persist-peaks", "--domain", "minerals", "--all"),
        ("fit-fluor", "--all", "--no-plots"),
        ("backfill", "--domains", "minerals,fluorescence"),
    ):
        result = CliRunner().invoke(cli.app, [
            "--json", *args, "--database", str(run.database_path),
        ])
        assert result.exit_code == 0, result.output
    assert calls == [(f"{SOL:04d}", "", SCAN, run.scan_id)] * 4


def test_persist_peaks_single_scan_without_target(run, monkeypatch):
    calls = []

    def record(self, sol, target, scan, **kw):
        calls.append((sol, target, scan, kw.get("scan_id")))
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 0})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", record)
    monkeypatch.setattr(FittingService, "__init__", lambda self, **kw: None)
    result = CliRunner().invoke(cli.app, [
        "--json", "persist-peaks", "--domain", "minerals", "--sol", str(SOL),
        "--scan", SCAN, "--database", str(run.database_path),
    ])
    assert result.exit_code == 0, result.output
    assert calls == [(f"{SOL:04d}", "", SCAN, run.scan_id)]


# --- shared output location ---------------------------------------------------

def test_bulk_loops_refuse_scans_that_share_an_output_location(run, monkeypatch):
    """Two NULL-target rows with one (sol, scan) would share one results dir."""
    with get_session(run.engine) as session:
        twin_id = _add_scan(session, None)
        other_id = _add_scan(session, None, scan_name="synthetic_other")
    calls = []

    def record(self, sol, target, scan, **kw):
        calls.append(kw.get("scan_id"))
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 0})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", record)
    monkeypatch.setattr(FittingService, "fit_fluorescence", record)
    monkeypatch.setattr(FittingService, "__init__", lambda self, **kw: None)
    for args in (
        ("persist-peaks", "--domain", "minerals", "--all"),
        ("fit-fluor", "--all", "--no-plots"),
        ("backfill", "--domains", "minerals,fluorescence"),
    ):
        result = CliRunner().invoke(cli.app, [
            "--json", *args, "--database", str(run.database_path),
        ])
        assert result.exit_code == 1, result.output
        errors = json.loads(result.stdout.strip().splitlines()[-1])["result"]["errors"]
        refused = [e for e in errors if "share this output location" in e]
        assert len(refused) == (4 if args[0] == "backfill" else 2), errors
        assert any(run.scan_id in e for e in refused)
        assert any(twin_id in e for e in refused)
    assert calls == [other_id] * 4


def test_fit_fluor_all_writes_nothing_for_scans_that_share_an_output_location(run, tmp_path):
    with get_session(run.engine) as session:
        twin_id = _add_scan(session, None)
    result = run.invoke("fit-fluor", "--all")
    assert result.exit_code == 1, result.output
    assert _fluor_peak_count(run.engine, run.scan_id) == 0
    assert _fluor_peak_count(run.engine, twin_id) == 0
    fluor_dir = tmp_path / "results" / f"{SOL:04d}_{SCAN}" / "fluorescence_fit"
    assert not fluor_dir.exists() or not any(fluor_dir.iterdir())
