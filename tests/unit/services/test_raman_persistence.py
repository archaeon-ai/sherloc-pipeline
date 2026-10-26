"""
Tests for FittingService.persist_raman_peaks() — all 3 Raman domains.

Covers: minerals, organics, hydration persistence; idempotency;
domain isolation; stale row cleanup; feature assignment; assignment_confidence NULL.

AC: bd-3cqz.10 (spec step 2.5)
"""

import json
import uuid
import pytest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import (
    SolORM, ScanORM, ScanPointORM, SpectrumORM, FittedPeakORM,
)
from sherloc_pipeline.services.fitting import (
    FittingService,
    _begin_fit_run,
    _fit_run_marker_path,
    _read_fit_run_marker,
    _write_fit_run_marker,
)
from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.base import ServiceResult


def _write_marker(fit_dir: Path, *, points_fitted: int = 3, accepted_peaks: int = 1) -> None:
    """Write the completion marker persistence requires for every fit dir.

    Real fits write this as their last step; tests that hand-build a fit
    directory have to stand in for that.
    """
    domain = fit_dir.name[: -len("_fit")]
    _write_fit_run_marker(
        _fit_run_marker_path(
            fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", domain,
        ),
        domain=domain,
        points_fitted=points_fitted,
        accepted_peaks=accepted_peaks,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """In-memory SQLite with full schema."""
    eng = get_engine(":memory:")
    create_all_tables(eng)
    return eng


@pytest.fixture
def populated_db(engine):
    """Sol 851 / Lake_Haiyaha / detail_1, 3 points with dark_subtracted spectra."""
    spectrum_ids = {}
    scan_point_ids = {}
    with get_session(engine) as session:
        sol = SolORM(sol_number=851, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id, sol_number=851, scan_name="detail_1",
            target="Lake Haiyaha", scan_id="test_scan_id",
            sclk_start=0, n_points=3, n_channels=2148,
        )
        session.add(scan)
        session.flush()

        for i in range(3):
            pt_id = str(uuid.uuid4())
            pt = ScanPointORM(id=pt_id, scan_id=scan_id, point_index=i)
            session.add(pt)
            session.flush()
            scan_point_ids[i] = pt_id

            spec_id = str(uuid.uuid4())
            spec = SpectrumORM(
                id=spec_id, scan_point_id=pt_id, region="R1",
                spectrum_type="dark_subtracted", processing_level="raw",
                intensities=b"\x00",
            )
            session.add(spec)
            session.flush()
            spectrum_ids[i] = spec_id

    return engine, spectrum_ids, scan_point_ids


def _make_minerals_csvs(base: Path) -> Path:
    """Create minerals_fit/ with CSVs for points 0-2."""
    d = base / "minerals_fit"
    d.mkdir(parents=True, exist_ok=True)
    _write_marker(d)

    # Point 0: peak at 1085.5 (carbonate range)
    pd.DataFrame([
        {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0,
         "area": 100.0, "snr": 15.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

    # Point 1: peak at 700 (unidentified for default rules, but still a peak)
    pd.DataFrame([
        {"center_cm1": 700.0, "fwhm_cm1": 35.0, "amplitude_a": 600.0,
         "area": 40.0, "snr": 5.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point1_fit_peaks.csv", index=False)

    # Point 2: empty
    pd.DataFrame(
        columns=["center_cm1", "fwhm_cm1", "amplitude_a", "area", "snr"]
    ).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point2_fit_peaks.csv", index=False)

    # AICc summary
    pd.DataFrame([
        {"point": 0, "r2": 0.95},
        {"point": 1, "r2": 0.88},
        {"point": 2, "r2": 0.72},
    ]).to_csv(base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

    return base


def _make_organics_csvs(base: Path) -> Path:
    """Create organics_fit/ with DG peaks for points 0-2."""
    d = base / "organics_fit"
    d.mkdir(parents=True, exist_ok=True)
    _write_marker(d)

    # Point 0: D band (1350) + G band (1600)
    pd.DataFrame([
        {"center_cm1": 1350.0, "fwhm_cm1": 60.0, "amplitude_a": 300.0,
         "area": 80.0, "snr": 6.0},
        {"center_cm1": 1600.0, "fwhm_cm1": 50.0, "amplitude_a": 500.0,
         "area": 120.0, "snr": 10.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

    # Point 1: G band only
    pd.DataFrame([
        {"center_cm1": 1580.0, "fwhm_cm1": 45.0, "amplitude_a": 400.0,
         "area": 90.0, "snr": 7.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point1_organics_dg_peaks.csv", index=False)

    # Point 2: empty
    pd.DataFrame(
        columns=["center_cm1", "fwhm_cm1", "amplitude_a", "area", "snr"]
    ).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point2_organics_dg_peaks.csv", index=False)

    # Accepted-peaks CSV with R²
    pd.DataFrame([
        {"point": 0, "r2": 0.91},
        {"point": 1, "r2": 0.85},
    ]).to_csv(
        d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False
    )

    return base


def _make_hydration_csvs(base: Path) -> Path:
    """Create hydration_fit/ with peaks for points 0-2."""
    d = base / "hydration_fit"
    d.mkdir(parents=True, exist_ok=True)
    _write_marker(d)

    # Point 0: OH stretch (3500)
    pd.DataFrame([
        {"center_cm1": 3500.0, "fwhm_cm1": 60.0, "amplitude_a": 800.0,
         "area": 200.0, "snr": 12.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", index=False)

    # Point 1: H2O bend (1630)
    pd.DataFrame([
        {"center_cm1": 1630.0, "fwhm_cm1": 35.0, "amplitude_a": 350.0,
         "area": 70.0, "snr": 5.0},
    ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point1_hydration_peaks.csv", index=False)

    # Point 2: empty
    pd.DataFrame(
        columns=["center_cm1", "fwhm_cm1", "amplitude_a", "area", "snr"]
    ).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point2_hydration_peaks.csv", index=False)

    # Accepted-peaks CSV with R²
    pd.DataFrame([
        {"point": 0, "r2": 0.93},
        {"point": 1, "r2": 0.82},
    ]).to_csv(
        d / "0851_Lake_Haiyaha_detail_1_R1_hydration_accepted_peaks.csv", index=False
    )

    return base


@pytest.fixture
def results_all(tmp_path):
    """Create results dir with minerals, organics, hydration CSV subdirs."""
    base = tmp_path / "results"
    base.mkdir()
    _make_minerals_csvs(base)
    _make_organics_csvs(base)
    _make_hydration_csvs(base)
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(engine):
    service = FittingService(database_path=Path(":memory:"))
    service._engine = engine
    return service


def _call_persist_raman(service, results_base, domain):
    """Call persist_raman_peaks with mocked resolve_scan_context and DataIngestion."""
    mock_ctx = MagicMock()
    mock_ctx.base_data_dir = results_base
    mock_ctx.results_dir = results_base

    mock_di = MagicMock()
    mock_di.get_results_path.return_value = results_base

    with patch("sherloc_pipeline.services.fitting.resolve_scan_context") as mock_rsc, \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=mock_di):
        mock_rsc.return_value = mock_ctx
        return service.persist_raman_peaks(
            sol="0851", target="Lake_Haiyaha", scan="detail_1", domain=domain,
        )


# ---------------------------------------------------------------------------
# Unreadable peak CSVs must preserve the affected points' persisted rows.
# ---------------------------------------------------------------------------

def _snapshot_peak_rows(database_path):
    """Read every peak column through a separate engine to check committed rows."""
    engine = get_engine(database_path)
    try:
        with get_session(engine) as session:
            rows = {}
            for peak, point_index in session.query(FittedPeakORM, ScanPointORM.point_index).join(
                SpectrumORM, FittedPeakORM.spectrum_id == SpectrumORM.id
            ).join(ScanPointORM, SpectrumORM.scan_point_id == ScanPointORM.id).order_by(
                FittedPeakORM.id
            ):
                rows.setdefault(point_index, []).append({
                    column.name: getattr(peak, column.name)
                    for column in FittedPeakORM.__table__.columns
                })
            return rows
    finally:
        engine.dispose()


@pytest.fixture
def point_count():
    return 3


@pytest.fixture(params=["minerals", "organics", "hydration"])
def synthetic_peak_run(tmp_path, request, point_count):
    """Generated CSVs and prior peaks for entirely invented scan points."""
    assert not any(tmp_path.resolve().is_relative_to(root) for root in (
        Path("/data"), Path("/nas"),
    ))
    domain = request.param
    database_path = tmp_path / "synthetic.db"
    engine = get_engine(database_path)
    create_all_tables(engine)
    fit_dir = tmp_path / f"{domain}_fit"
    fit_dir.mkdir()
    prefix = "87654_Synthetic_CSV_synthetic_scan_R1"
    suffix = {
        "minerals": "fit_peaks",
        "organics": "organics_dg_peaks",
        "hydration": "hydration_peaks",
    }[domain]
    center = {"minerals": 1143.0, "organics": 1473.0, "hydration": 3437.0}[domain]
    csv_paths = []
    new_centers = [center + 7 * point for point in range(point_count)]

    with get_session(engine) as session:
        session.add(SolORM(sol_number=87654, data_source="loupe"))
        session.flush()
        scan_id = str(uuid.uuid4())
        session.add(ScanORM(
            id=scan_id, sol_number=87654, target="Synthetic CSV",
            target_type="mars_target",
            scan_name="synthetic_scan", scan_id="synthetic_csv_run",
            sclk_start=0, n_points=point_count, n_channels=1,
        ))
        session.flush()
        for point in range(point_count):
            point_id = str(uuid.uuid4())
            session.add(ScanPointORM(id=point_id, scan_id=scan_id, point_index=point))
            session.flush()
            spectrum_id = str(uuid.uuid4())
            session.add(SpectrumORM(
                id=spectrum_id, scan_point_id=point_id, region="R1",
                spectrum_type="dark_subtracted", processing_level="raw", intensities=b"\x00",
            ))
            session.flush()
            for offset in (17, 29):
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()), spectrum_id=spectrum_id,
                    fit_modality=domain, center_cm1=center - offset,
                    amplitude=41.0, fwhm_cm1=67.0, snr=13.0,
                ))
            csv_path = fit_dir / f"{prefix}_point{point}_{suffix}.csv"
            pd.DataFrame([{
                "center_cm1": new_centers[point], "amplitude_a": 73.0,
                "fwhm_cm1": 83.0, "snr": 19.0,
            }]).to_csv(csv_path, index=False)
            csv_paths.append(csv_path)

    _write_fit_run_marker(
        _fit_run_marker_path(
            fit_dir, "87654", "Synthetic_CSV", "synthetic_scan", "R1", domain,
        ),
        domain=domain, points_fitted=point_count, accepted_peaks=point_count,
    )
    yield SimpleNamespace(
        engine=engine, database_path=database_path, base=tmp_path, domain=domain,
        csv_paths=csv_paths, new_centers=new_centers,
        previous=_snapshot_peak_rows(database_path),
    )
    engine.dispose()


def _persist_synthetic_run(run, console=None):
    service = _make_service(run.engine)
    if console is not None:
        service.console = console
    mock_ctx = MagicMock(base_data_dir=run.base, results_dir=run.base)
    mock_di = MagicMock()
    mock_di.get_results_path.return_value = run.base
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context", return_value=mock_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion", return_value=mock_di):
        return service.persist_raman_peaks(
            sol="87654", target="Synthetic_CSV", scan="synthetic_scan", domain=run.domain,
        )


class TestUnreadablePeakCSVs:
    @pytest.mark.parametrize("failure", ["corrupt", "permission"])
    def test_unreadable_point_preserved_valid_points_committed(
        self, synthetic_peak_run, failure, monkeypatch, caplog
    ):
        run = synthetic_peak_run
        unreadable = run.csv_paths[1]
        if failure == "corrupt":
            unreadable.write_bytes(b"\xff\xfe\x00synthetic-private-contents")
            exception_type = "UnicodeDecodeError"
        else:
            read_csv = pd.read_csv

            def deny_one_csv(path, *args, **kwargs):
                if Path(path) == unreadable:
                    raise PermissionError("synthetic-private-contents")
                return read_csv(path, *args, **kwargs)

            monkeypatch.setattr(pd, "read_csv", deny_one_csv)
            exception_type = "PermissionError"

        with pytest.raises(FittingError) as exc_info:
            _persist_synthetic_run(run)

        error = exc_info.value
        assert error.exit_code == 1
        assert error.context["peaks_inserted"] == 2
        assert error.context["unreadable_peak_csvs"] == [{
            "point_index": 1, "file_name": unreadable.name, "exception_type": exception_type,
        }]
        assert unreadable.name in str(error)
        assert "previously persisted peaks were kept" in str(error)
        assert "synthetic-private-contents" not in str(error.context) + str(error) + caplog.text

        rows = _snapshot_peak_rows(run.database_path)
        assert rows[1] == run.previous[1]
        for point in (0, 2):
            assert len(rows[point]) == 1
            assert rows[point][0]["id"] not in {row["id"] for row in run.previous[point]}
            assert rows[point][0]["center_cm1"] == run.new_centers[point]
            assert rows[point][0]["amplitude"] == 73.0

    def test_all_valid_csvs_replace_prior_peaks(self, synthetic_peak_run):
        run = synthetic_peak_run
        result = _persist_synthetic_run(run)
        assert result.metadata["peaks_inserted"] == 3
        rows = _snapshot_peak_rows(run.database_path)
        for point in range(3):
            assert len(rows[point]) == 1
            assert rows[point][0]["id"] not in {row["id"] for row in run.previous[point]}
            assert rows[point][0]["center_cm1"] == run.new_centers[point]

    @pytest.mark.parametrize("synthetic_peak_run", ["minerals", "hydration"], indirect=True)
    def test_unreadable_point_with_readable_duplicate_is_preserved(self, synthetic_peak_run):
        run = synthetic_peak_run
        unreadable = run.csv_paths[1]
        duplicate = unreadable.with_name(f"duplicate_{unreadable.name}")
        duplicate.write_bytes(unreadable.read_bytes())
        unreadable.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(FittingError) as exc_info:
            _persist_synthetic_run(run)
        assert exc_info.value.context["peaks_inserted"] == 2
        assert _snapshot_peak_rows(run.database_path)[1] == run.previous[1]

    @pytest.mark.parametrize("point_count", [1])
    def test_only_csv_unreadable_preserves_all_prior_peaks(self, synthetic_peak_run):
        run = synthetic_peak_run
        run.csv_paths[0].write_bytes(b"\xff\xfe\x00")
        with pytest.raises(FittingError) as exc_info:
            _persist_synthetic_run(run)
        assert exc_info.value.exit_code == 1
        assert exc_info.value.context["peaks_inserted"] == 0
        assert _snapshot_peak_rows(run.database_path) == run.previous

    def test_process_new_exits_one_for_unreadable_peak_csv(self, synthetic_peak_run, monkeypatch):
        from sherloc_pipeline.cli.app import app

        run = synthetic_peak_run
        run.csv_paths[1].write_bytes(b"\xff\xfe\x00")
        sol_dir = run.base / "sol_87654"
        sol_dir.mkdir()
        monkeypatch.setenv("SHERLOC_DB_PATH", str(run.database_path))
        fitted = ServiceResult(summary="Synthetic fitting completed")
        fitting = MagicMock()
        for method in ("fit_minerals", "fit_organics", "fit_hydration", "fit_fluorescence"):
            getattr(fitting, method).return_value = fitted

        def persist(**kwargs):
            if kwargs["domain"] == run.domain:
                return _persist_synthetic_run(run, console=fitting_factory.call_args.kwargs["console"])
            return ServiceResult(summary="Synthetic persistence completed")

        fitting.persist_raman_peaks.side_effect = persist
        with patch("sherloc_pipeline.services.pipeline.FittingService", return_value=fitting) as fitting_factory, \
             patch("sherloc_pipeline.services.pipeline.PreprocessingService") as preprocessing, \
             patch("sherloc_pipeline.services.pipeline.DataIngestion") as ingestion, \
             patch("sherloc_pipeline.services.pipeline.archive_existing_results", return_value=None), \
             patch("sherloc_pipeline.services.fluor_inputs.refresh_despiked_rows_for_scan", return_value={}):
            preprocessing.return_value.run_scan.return_value = fitted
            ingestion.return_value.get_results_path.return_value = run.base
            result = CliRunner().invoke(app, [
                "--json", "process-new", str(sol_dir), "--skip-ingest",
                "--database", str(run.database_path), "--data-dir", str(run.base),
                "--results-dir", str(run.base),
            ])

        assert result.exit_code == 1, result.output
        errors = json.loads(result.stdout)["result"]["errors"]
        assert len(errors) == 1
        assert run.csv_paths[1].name in errors[0]
        assert "previously persisted peaks were kept" in errors[0]
        assert _snapshot_peak_rows(run.database_path)[1] == run.previous[1]


# ---------------------------------------------------------------------------
# AC 1 — Minerals domain (matches old behavior + assignment)
# ---------------------------------------------------------------------------

class TestMineralsDomain:
    def test_minerals_inserts_reviewable_peaks(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        result = _call_persist_raman(service, results_all, "minerals")

        assert result.metadata["peaks_inserted"] == 2
        assert result.metadata["domain"] == "minerals"

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).all()
            assert len(peaks) == 2
            for p in peaks:
                assert p.fit_modality == "minerals"
                assert p.peak_type == "gaussian"

    def test_minerals_r2_from_aicc(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "minerals")

        with get_session(engine) as session:
            p0 = [p for p in session.query(FittedPeakORM).all()
                   if p.spectrum_id == spectrum_ids[0]]
            assert len(p0) == 1
            assert p0[0].fit_quality == 0.95


# ---------------------------------------------------------------------------
# AC 2 — Organics domain (discovers CSVs, classifies bands)
# ---------------------------------------------------------------------------

class TestOrganicsDomain:
    def test_organics_inserts_and_classifies(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        result = _call_persist_raman(service, results_all, "organics")

        assert result.metadata["peaks_inserted"] == 3
        assert result.metadata["domain"] == "organics"

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).all()
            assert len(peaks) == 3
            for p in peaks:
                assert p.fit_modality == "organics"

    def test_organics_r2_from_accepted_peaks(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "organics")

        with get_session(engine) as session:
            p0 = [p for p in session.query(FittedPeakORM).all()
                   if p.spectrum_id == spectrum_ids[0]]
            for p in p0:
                assert p.fit_quality == 0.91

    def test_organics_prefers_dg_over_g(self, populated_db, tmp_path):
        """When both DG and G CSVs exist, only DG CSV is used."""
        engine, spectrum_ids, _ = populated_db
        base = tmp_path / "results_dg"
        base.mkdir()
        d = base / "organics_fit"
        d.mkdir()
        _write_marker(d)

        # DG CSV: peak at 1350
        pd.DataFrame([
            {"center_cm1": 1350.0, "fwhm_cm1": 60.0, "amplitude_a": 300.0,
             "area": 80.0, "snr": 6.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

        # G-only CSV: peak at 1600 (should be ignored)
        pd.DataFrame([
            {"center_cm1": 1600.0, "fwhm_cm1": 50.0, "amplitude_a": 500.0,
             "area": 120.0, "snr": 10.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_g_peaks.csv", index=False)

        # Accepted-peaks CSV
        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False
        )

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "organics")

        assert result.metadata["peaks_inserted"] == 1
        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).first()
            assert abs(peak.center_cm1 - 1350.0) < 0.1  # DG peak, not G-only


# ---------------------------------------------------------------------------
# AC 3 — Hydration domain (discovers CSVs, classifies bands)
# ---------------------------------------------------------------------------

class TestHydrationDomain:
    def test_hydration_inserts_and_classifies(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        result = _call_persist_raman(service, results_all, "hydration")

        # Only 1 peak: OH at 3500 passes; bend at 1630 filtered by center range [3000, 3900]
        assert result.metadata["peaks_inserted"] == 1
        assert result.metadata["domain"] == "hydration"

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).all()
            assert len(peaks) == 1
            for p in peaks:
                assert p.fit_modality == "hydration"

    def test_hydration_r2_from_accepted_peaks(self, populated_db, results_all):
        engine, spectrum_ids, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            p0 = [p for p in session.query(FittedPeakORM).all()
                   if p.spectrum_id == spectrum_ids[0]]
            assert len(p0) == 1
            assert p0[0].fit_quality == 0.93


# ---------------------------------------------------------------------------
# AC 4 — Idempotent write: insert, re-insert -> no duplicates
# ---------------------------------------------------------------------------

class TestIdempotency:
    @pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
    def test_idempotent_no_duplicates(self, populated_db, results_all, domain):
        engine, _, _ = populated_db
        service = _make_service(engine)

        r1 = _call_persist_raman(service, results_all, domain)
        count1 = r1.metadata["peaks_inserted"]

        r2 = _call_persist_raman(service, results_all, domain)
        count2 = r2.metadata["peaks_inserted"]
        assert count1 == count2

        with get_session(engine) as session:
            total = session.query(FittedPeakORM).filter_by(fit_modality=domain).count()
            assert total == count1


# ---------------------------------------------------------------------------
# AC 5 — Domain isolation: mineral re-persist doesn't touch organics
# ---------------------------------------------------------------------------

class TestDomainIsolation:
    def test_mineral_repersist_preserves_organics(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)

        # Persist organics first
        r_org = _call_persist_raman(service, results_all, "organics")
        org_count = r_org.metadata["peaks_inserted"]
        assert org_count > 0

        # Now persist minerals — organics should be untouched
        _call_persist_raman(service, results_all, "minerals")

        with get_session(engine) as session:
            org = session.query(FittedPeakORM).filter_by(fit_modality="organics").count()
            assert org == org_count

    def test_hydration_repersist_preserves_minerals(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)

        _call_persist_raman(service, results_all, "minerals")
        with get_session(engine) as session:
            m_count = session.query(FittedPeakORM).filter_by(fit_modality="minerals").count()

        _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            m_after = session.query(FittedPeakORM).filter_by(fit_modality="minerals").count()
            assert m_after == m_count

    def test_all_three_domains_coexist(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)

        r_m = _call_persist_raman(service, results_all, "minerals")
        r_o = _call_persist_raman(service, results_all, "organics")
        r_h = _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            total = session.query(FittedPeakORM).count()
            expected = (
                r_m.metadata["peaks_inserted"]
                + r_o.metadata["peaks_inserted"]
                + r_h.metadata["peaks_inserted"]
            )
            assert total == expected

            # Each domain's peaks intact
            for dom, r in [("minerals", r_m), ("organics", r_o), ("hydration", r_h)]:
                cnt = session.query(FittedPeakORM).filter_by(fit_modality=dom).count()
                assert cnt == r.metadata["peaks_inserted"]


# ---------------------------------------------------------------------------
# AC 6 — Stale row cleanup: spectrum preference change doesn't leave orphans
# ---------------------------------------------------------------------------

class TestStaleRowCleanup:
    def test_spectrum_variant_change_cleans_old_peaks(self, engine, results_all):
        """If spectrum preference changes (e.g., active -> dark_subtracted),
        the domain delete covers ALL spectrum variants for the scan points."""
        spectrum_ids_v1 = {}
        spectrum_ids_v2 = {}

        with get_session(engine) as session:
            sol = SolORM(sol_number=851, data_source="loupe")
            session.add(sol)
            session.flush()

            scan_id = str(uuid.uuid4())
            scan = ScanORM(
                id=scan_id, sol_number=851, scan_name="detail_1",
                target="Lake Haiyaha", scan_id="stale_test",
                sclk_start=0, n_points=3, n_channels=2148,
            )
            session.add(scan)
            session.flush()

            for i in range(3):
                pt_id = str(uuid.uuid4())
                pt = ScanPointORM(id=pt_id, scan_id=scan_id, point_index=i)
                session.add(pt)
                session.flush()

                # Two spectrum variants per point
                for spec_type, proc_level in [
                    ("active", "raw"),
                    ("dark_subtracted", "raw"),
                ]:
                    spec_id = str(uuid.uuid4())
                    spec = SpectrumORM(
                        id=spec_id, scan_point_id=pt_id, region="R1",
                        spectrum_type=spec_type, processing_level=proc_level,
                        intensities=b"\x00",
                    )
                    session.add(spec)
                    session.flush()
                    if spec_type == "active":
                        spectrum_ids_v1[i] = spec_id
                    else:
                        spectrum_ids_v2[i] = spec_id

            # Manually insert a "stale" peak linked to the 'active' spectrum
            stale = FittedPeakORM(
                id=str(uuid.uuid4()),
                spectrum_id=spectrum_ids_v1[0],
                peak_type="gaussian",
                fit_modality="minerals",
                center_cm1=999.0,
                amplitude=100.0,
                fwhm_cm1=30.0,
            )
            session.add(stale)

        service = _make_service(engine)
        # persist_raman_peaks prefers dark_subtracted over active, so new peaks
        # will link to v2 spectra — but the delete should remove the stale v1 peak
        _call_persist_raman(service, results_all, "minerals")

        with get_session(engine) as session:
            # Stale peak at 999.0 should be gone
            stale_peaks = session.query(FittedPeakORM).filter(
                FittedPeakORM.center_cm1 == 999.0
            ).count()
            assert stale_peaks == 0

            # All remaining peaks should be linked to v2 (dark_subtracted)
            all_peaks = session.query(FittedPeakORM).all()
            for p in all_peaks:
                assert p.spectrum_id in spectrum_ids_v2.values()


# ---------------------------------------------------------------------------
# AC 7 — Feature assignment: minerals get assign_min_id labels
# ---------------------------------------------------------------------------

class TestFeatureAssignmentMinerals:
    def test_minerals_get_mineral_labels(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "minerals")

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).filter_by(fit_modality="minerals").all()
            assert len(peaks) > 0
            for p in peaks:
                # mineral_assignment should be set (may be "unidentified" or a mineral label)
                assert p.mineral_assignment is not None
                assert isinstance(p.mineral_assignment, str)
                assert len(p.mineral_assignment) > 0


# ---------------------------------------------------------------------------
# AC 8 — Feature assignment: organics get D_band/G_band labels
# ---------------------------------------------------------------------------

class TestFeatureAssignmentOrganics:
    def test_organics_get_band_labels(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "organics")

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).filter_by(fit_modality="organics").all()
            labels = {p.mineral_assignment for p in peaks}
            # 1350 -> D_band, 1600/1580 -> G_band
            assert "D_band" in labels
            assert "G_band" in labels

    def test_organics_d_band_at_1350(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "organics")

        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).filter(
                FittedPeakORM.center_cm1.between(1340, 1360)
            ).first()
            assert peak is not None
            assert peak.mineral_assignment == "D_band"

    def test_organics_g_band_at_1600(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "organics")

        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).filter(
                FittedPeakORM.center_cm1.between(1590, 1610)
            ).first()
            assert peak is not None
            assert peak.mineral_assignment == "G_band"


# ---------------------------------------------------------------------------
# AC 9 — Feature assignment: hydration get OH_stretch labels
#         (H2O_bend at 1630 now filtered by center range [3000, 3900])
# ---------------------------------------------------------------------------

class TestFeatureAssignmentHydration:
    def test_hydration_get_oh_stretch_label(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).filter_by(fit_modality="hydration").all()
            labels = {p.mineral_assignment for p in peaks}
            assert "OH_stretch" in labels

    def test_hydration_oh_stretch_at_3500(self, populated_db, results_all):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).filter(
                FittedPeakORM.center_cm1.between(3490, 3510)
            ).first()
            assert peak is not None
            assert peak.mineral_assignment == "OH_stretch"

    def test_hydration_bend_filtered_by_center_range(self, populated_db, results_all):
        """H2O bend at 1630 cm⁻¹ is outside center range [3000, 3900] and should be filtered."""
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, "hydration")

        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).filter(
                FittedPeakORM.center_cm1.between(1620, 1640)
            ).first()
            assert peak is None  # Filtered by center range gate


# ---------------------------------------------------------------------------
# AC 10 — assignment_confidence is NULL (not auto 1.0)
# ---------------------------------------------------------------------------

class TestAssignmentConfidence:
    @pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
    def test_assignment_confidence_is_null(self, populated_db, results_all, domain):
        engine, _, _ = populated_db
        service = _make_service(engine)
        _call_persist_raman(service, results_all, domain)

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).filter_by(fit_modality=domain).all()
            assert len(peaks) > 0
            for p in peaks:
                assert p.assignment_confidence is None


# ---------------------------------------------------------------------------
# AC 11 — Post-hoc R² > 0 filter: negative R² points are skipped
# ---------------------------------------------------------------------------

class TestR2PosthocFilter:
    def test_negative_r2_point_filtered(self, populated_db, tmp_path):
        """A point with R² = -0.5 should have ALL its peaks excluded."""
        engine, spectrum_ids, _ = populated_db
        base = tmp_path / "results_r2"
        base.mkdir()
        d = base / "minerals_fit"
        d.mkdir(parents=True, exist_ok=True)
        _write_marker(d)

        # Point 0: good peak, R² = 0.95
        pd.DataFrame([
            {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0,
             "area": 100.0, "snr": 15.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

        # Point 1: good peak, but R² = -0.5 (bad fit)
        pd.DataFrame([
            {"center_cm1": 700.0, "fwhm_cm1": 35.0, "amplitude_a": 600.0,
             "area": 40.0, "snr": 5.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point1_fit_peaks.csv", index=False)

        # AICc summary with negative R² for point 1
        pd.DataFrame([
            {"point": 0, "r2": 0.95},
            {"point": 1, "r2": -0.5},
        ]).to_csv(base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "minerals")

        # Only point 0 peak should be inserted; point 1 filtered by R² ≤ 0
        assert result.metadata["peaks_inserted"] == 1

        with get_session(engine) as session:
            peaks = session.query(FittedPeakORM).all()
            assert len(peaks) == 1
            assert abs(peaks[0].center_cm1 - 1085.5) < 0.1

    def test_zero_r2_point_filtered(self, populated_db, tmp_path):
        """R² = 0.0 (exactly at threshold) should be filtered (≤ 0 check)."""
        engine, spectrum_ids, _ = populated_db
        base = tmp_path / "results_r2_zero"
        base.mkdir()
        d = base / "minerals_fit"
        d.mkdir(parents=True, exist_ok=True)
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0,
             "area": 100.0, "snr": 15.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

        pd.DataFrame([
            {"point": 0, "r2": 0.0},
        ]).to_csv(base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "minerals")
        assert result.metadata["peaks_inserted"] == 0

    def test_positive_r2_passes(self, populated_db, tmp_path):
        """R² = 0.01 (just above threshold) should pass."""
        engine, spectrum_ids, _ = populated_db
        base = tmp_path / "results_r2_pos"
        base.mkdir()
        d = base / "minerals_fit"
        d.mkdir(parents=True, exist_ok=True)
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0,
             "area": 100.0, "snr": 15.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

        pd.DataFrame([
            {"point": 0, "r2": 0.01},
        ]).to_csv(base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "minerals")
        assert result.metadata["peaks_inserted"] == 1


# ---------------------------------------------------------------------------
# AC 12 — Post-hoc organics FWHM filter: narrow peaks rejected per band
# ---------------------------------------------------------------------------

class TestOrganicsFWHMFilter:
    def test_d_band_narrow_filtered(self, populated_db, tmp_path):
        """D_band peak with FWHM=25 cm⁻¹ should be filtered (requires >= 50)."""
        engine, _, _ = populated_db
        base = tmp_path / "results_org_fwhm"
        base.mkdir()
        d = base / "organics_fit"
        d.mkdir()
        _write_marker(d)

        # D_band at 1350, FWHM=25 (too narrow)
        pd.DataFrame([
            {"center_cm1": 1350.0, "fwhm_cm1": 25.0, "amplitude_a": 300.0,
             "area": 80.0, "snr": 6.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "organics")
        assert result.metadata["peaks_inserted"] == 0

    def test_d_band_wide_passes(self, populated_db, tmp_path):
        """D_band peak with FWHM=60 cm⁻¹ should pass (>= 50)."""
        engine, _, _ = populated_db
        base = tmp_path / "results_org_fwhm2"
        base.mkdir()
        d = base / "organics_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 1350.0, "fwhm_cm1": 60.0, "amplitude_a": 300.0,
             "area": 80.0, "snr": 6.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "organics")
        assert result.metadata["peaks_inserted"] == 1

    def test_g_band_narrow_filtered(self, populated_db, tmp_path):
        """G_band peak with FWHM=25 cm⁻¹ should be filtered (requires >= 40)."""
        engine, _, _ = populated_db
        base = tmp_path / "results_org_g"
        base.mkdir()
        d = base / "organics_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 1600.0, "fwhm_cm1": 25.0, "amplitude_a": 500.0,
             "area": 120.0, "snr": 10.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "organics")
        assert result.metadata["peaks_inserted"] == 0

    def test_unidentified_organics_narrow_filtered(self, populated_db, tmp_path):
        """Unidentified organic peak with FWHM=25 cm⁻¹ filtered (requires >= 40)."""
        engine, _, _ = populated_db
        base = tmp_path / "results_org_unid"
        base.mkdir()
        d = base / "organics_fit"
        d.mkdir()
        _write_marker(d)

        # Peak at 1450 cm⁻¹ — not in D or G range, classified as unidentified
        pd.DataFrame([
            {"center_cm1": 1450.0, "fwhm_cm1": 25.0, "amplitude_a": 200.0,
             "area": 50.0, "snr": 4.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_organics_dg_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.80}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_organics_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "organics")
        assert result.metadata["peaks_inserted"] == 0


# ---------------------------------------------------------------------------
# AC 13 — Post-hoc hydration center range gate
# ---------------------------------------------------------------------------

class TestHydrationCenterRangeGate:
    def test_peak_outside_high_end_filtered(self, populated_db, tmp_path):
        """Hydration peak at 3950 cm⁻¹ outside [3000, 3900] should be filtered."""
        engine, _, _ = populated_db
        base = tmp_path / "results_hyd_hi"
        base.mkdir()
        d = base / "hydration_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 3950.0, "fwhm_cm1": 60.0, "amplitude_a": 800.0,
             "area": 200.0, "snr": 12.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_hydration_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "hydration")
        assert result.metadata["peaks_inserted"] == 0

    def test_peak_at_3500_passes(self, populated_db, tmp_path):
        """Hydration peak at 3500 cm⁻¹ within [3000, 3900] should pass."""
        engine, _, _ = populated_db
        base = tmp_path / "results_hyd_pass"
        base.mkdir()
        d = base / "hydration_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 3500.0, "fwhm_cm1": 60.0, "amplitude_a": 800.0,
             "area": 200.0, "snr": 12.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_hydration_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "hydration")
        assert result.metadata["peaks_inserted"] == 1

    def test_serpentine_oh_at_3700_passes(self, populated_db, tmp_path):
        """Serpentine structural OH at ~3700 cm⁻¹ within [3000, 3900] should pass."""
        engine, _, _ = populated_db
        base = tmp_path / "results_hyd_serp"
        base.mkdir()
        d = base / "hydration_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 3700.0, "fwhm_cm1": 55.0, "amplitude_a": 600.0,
             "area": 150.0, "snr": 10.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.88}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_hydration_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "hydration")
        assert result.metadata["peaks_inserted"] == 1

        with get_session(engine) as session:
            peak = session.query(FittedPeakORM).first()
            assert abs(peak.center_cm1 - 3700.0) < 0.1
            assert peak.mineral_assignment == "OH_stretch"

    def test_peak_below_3000_filtered(self, populated_db, tmp_path):
        """Hydration peak at 2900 cm⁻¹ below center range should be filtered."""
        engine, _, _ = populated_db
        base = tmp_path / "results_hyd_lo"
        base.mkdir()
        d = base / "hydration_fit"
        d.mkdir()
        _write_marker(d)

        pd.DataFrame([
            {"center_cm1": 2900.0, "fwhm_cm1": 60.0, "amplitude_a": 800.0,
             "area": 200.0, "snr": 12.0},
        ]).to_csv(d / "0851_Lake_Haiyaha_detail_1_R1_point0_hydration_peaks.csv", index=False)

        pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
            d / "0851_Lake_Haiyaha_detail_1_R1_hydration_accepted_peaks.csv", index=False)

        service = _make_service(engine)
        result = _call_persist_raman(service, base, "hydration")
        assert result.metadata["peaks_inserted"] == 0


# ---------------------------------------------------------------------------
# Run-marker gating — persistence replaces every row for a domain, so it must
# refuse to run against anything but a completed, non-empty fit.
# ---------------------------------------------------------------------------

class TestRunMarkerGating:
    """An incomplete or empty run must never overwrite valid persisted peaks."""

    @pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
    def test_interrupted_rerun_with_a_surviving_csv_raises(
        self, populated_db, results_all, domain
    ):
        """Leftover CSVs are not evidence that this run finished.

        A rerun that dies partway leaves the marker recording an in-progress
        run (written at fit start) while some of the previous run's per-point
        CSVs are still on disk. Persisting from that mix would replace the
        database with a partial result.
        """
        engine, _, _ = populated_db
        service = _make_service(engine)
        first = _call_persist_raman(service, results_all, domain)
        assert first.metadata["peaks_inserted"] > 0

        fit_dir = results_all / f"{domain}_fit"
        _begin_fit_run(
            _fit_run_marker_path(
                fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", domain,
            ),
            domain=domain,
        )
        # Point 0's CSV is gone, point 1's survives: at least one CSV remains,
        # so discovery still succeeds.
        next(fit_dir.glob("*point0*peaks.csv")).unlink()
        assert list(fit_dir.glob("*point1*peaks.csv"))

        with pytest.raises(FittingError) as exc_info:
            _call_persist_raman(_make_service(engine), results_all, domain)
        assert "was interrupted" in str(exc_info.value)

        with get_session(engine) as session:
            assert session.query(FittedPeakORM).filter_by(
                fit_modality=domain
            ).count() == first.metadata["peaks_inserted"], (
                "an aborted persist must leave persisted rows untouched"
            )

    @pytest.mark.parametrize("domain", ["hydration", "organics"])
    def test_zero_point_marker_is_rejected(self, populated_db, results_all, domain):
        """A run that fitted no points says nothing about the scan.

        Its `accepted_peaks: 0` must not be read as "the fit found nothing" and
        used to clear rows a real earlier run persisted.
        """
        engine, _, _ = populated_db
        service = _make_service(engine)
        first = _call_persist_raman(service, results_all, domain)
        assert first.metadata["peaks_inserted"] > 0

        fit_dir = results_all / f"{domain}_fit"
        for csv in fit_dir.glob("*peaks.csv"):
            csv.unlink()
        _write_fit_run_marker(
            _fit_run_marker_path(
                fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", domain,
            ),
            domain=domain, points_fitted=0, accepted_peaks=0,
        )

        with pytest.raises(FittingError) as exc_info:
            _call_persist_raman(_make_service(engine), results_all, domain)
        assert "fitted 0 points" in str(exc_info.value)

        with get_session(engine) as session:
            assert session.query(FittedPeakORM).filter_by(
                fit_modality=domain
            ).count() == first.metadata["peaks_inserted"]

    def test_zero_point_marker_rejected_before_csvs_are_read(
        self, populated_db, results_all
    ):
        """The point count is checked even when CSVs are present."""
        engine, _, _ = populated_db
        fit_dir = results_all / "hydration_fit"
        _write_fit_run_marker(
            _fit_run_marker_path(
                fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", "hydration",
            ),
            domain="hydration", points_fitted=0, accepted_peaks=1,
        )

        with pytest.raises(FittingError) as exc_info:
            _call_persist_raman(_make_service(engine), results_all, "hydration")
        assert "fitted 0 points" in str(exc_info.value)

    def test_marker_for_another_domain_is_rejected(self, populated_db, results_all):
        """A marker naming a different domain is not this domain's evidence."""
        engine, _, _ = populated_db
        fit_dir = results_all / "hydration_fit"
        _write_fit_run_marker(
            _fit_run_marker_path(
                fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", "hydration",
            ),
            domain="organics", points_fitted=3, accepted_peaks=1,
        )

        with pytest.raises(FittingError) as exc_info:
            _call_persist_raman(_make_service(engine), results_all, "hydration")
        assert "records domain" in str(exc_info.value)

    def test_marker_without_a_point_count_is_rejected(
        self, populated_db, results_all
    ):
        """A marker missing `points_fitted` cannot certify a run.

        It is reported as `invalid` rather than absent: something wrote it, so
        this is not a pre-marker directory and the legacy fallback must not
        apply.
        """
        engine, _, _ = populated_db
        fit_dir = results_all / "hydration_fit"
        marker = _fit_run_marker_path(
            fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", "hydration",
        )
        payload = json.loads(marker.read_text())
        del payload["points_fitted"]
        marker.write_text(json.dumps(payload))

        assert _read_fit_run_marker(marker)["status"] == "invalid"
        with pytest.raises(FittingError) as exc_info:
            _call_persist_raman(_make_service(engine), results_all, "hydration")
        assert "unreadable or malformed" in str(exc_info.value)

    @pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
    def test_pre_marker_fit_directory_still_persists(
        self, populated_db, results_all, domain
    ):
        """Removing the marker reproduces a directory fitted before markers.

        `persist-peaks` must behave exactly as it did then (docs/INVARIANTS.md
        §1): persist the CSVs it discovers, not exit 1.
        """
        engine, _, _ = populated_db
        fit_dir = results_all / f"{domain}_fit"
        _fit_run_marker_path(
            fit_dir, "0851", "Lake_Haiyaha", "detail_1", "R1", domain,
        ).unlink()
        assert list(fit_dir.glob("*peaks.csv"))

        result = _call_persist_raman(_make_service(engine), results_all, domain)

        assert result.metadata["peaks_inserted"] > 0
        with get_session(engine) as session:
            assert session.query(FittedPeakORM).filter_by(
                fit_modality=domain
            ).count() == result.metadata["peaks_inserted"]
