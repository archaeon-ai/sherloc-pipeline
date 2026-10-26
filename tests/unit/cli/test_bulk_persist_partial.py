"""Bulk totals count rows committed before an unreadable CSV or spectrum error."""

import importlib
import json

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.fitting import FittingService

cli = importlib.import_module("sherloc_pipeline.cli.app")

SCANS = [(1, "Synthetic_A", "scan_1"), (2, "Synthetic_B", "scan_2")]


@pytest.fixture
def bulk(tmp_path, monkeypatch):
    for root in ("/data", "/nas"):
        assert not str(tmp_path).startswith(root)
    database = tmp_path / "synthetic.db"
    database.write_bytes(b"")
    monkeypatch.setattr(cli, "_iter_all_scans", lambda *a, **kw: list(SCANS))
    monkeypatch.setattr(FittingService, "__init__", lambda self, **kw: None)

    def persist(self, sol, target, scan, domain, **kw):
        if scan == "scan_2":
            raise FittingError(
                "Persisted 2 peaks, but could not read peak CSVs",
                exit_code=1,
                context={"peaks_inserted": 2, "unreadable_peak_csvs": [{"point_index": 1}]},
            )
        return ServiceResult(summary="ok", artifacts=[], warnings=[], metadata={"peaks_inserted": 3})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", persist)

    def run(*args):
        result = CliRunner().invoke(cli.app, ["--json", *args, "--database", str(database)])
        return result.exit_code, json.loads(result.stdout.strip().splitlines()[-1])["result"]

    return run


def test_persist_peaks_all_counts_partially_committed_rows(bulk):
    exit_code, result = bulk("persist-peaks", "--domain", "minerals", "--all")
    assert exit_code == 1
    assert result["total_peaks"] == 5
    assert len(result["errors"]) == 1


def test_backfill_counts_partially_committed_rows(bulk):
    exit_code, result = bulk("backfill", "--domains", "minerals")
    assert result["total_peaks"] == 5
    assert result["domain_stats"]["minerals"] == {"peaks": 5, "errors": 1}
    assert exit_code == 1


def test_other_errors_add_no_peaks(bulk, monkeypatch):
    def fail(self, **kw):
        raise FittingError("never fitted", exit_code=1, context={"peaks_inserted": 9})

    monkeypatch.setattr(FittingService, "persist_raman_peaks", fail)
    _, result = bulk("persist-peaks", "--domain", "minerals", "--all")
    assert result["total_peaks"] == 0


@pytest.fixture
def fluor_bulk(bulk, monkeypatch):
    def fit(self, sol, target, scan, **kw):
        if scan == "scan_2":
            raise FittingError(
                "Persisted 2 fluorescence peaks, but could not read spectra",
                exit_code=1,
                context={"peaks_inserted": 2, "unreadable_spectra": [
                    {"point_index": 1, "exception_type": "error"},
                ]},
            )
        return ServiceResult(summary="ok", metadata={"peaks_inserted": 3})

    monkeypatch.setattr(FittingService, "fit_fluorescence", fit)
    return bulk


def test_fit_fluor_all_counts_partially_committed_rows(fluor_bulk):
    exit_code, result = fluor_bulk("fit-fluor", "--all", "--no-plots")
    assert exit_code == 1
    assert result["total_peaks"] == 5
    assert len(result["errors"]) == 1


def test_fluor_backfill_counts_partially_committed_rows(fluor_bulk):
    exit_code, result = fluor_bulk("backfill", "--domains", "fluorescence")
    assert exit_code == 1
    assert result["total_peaks"] == 5
    assert result["domain_stats"]["fluorescence"] == {"peaks": 5, "errors": 1}
