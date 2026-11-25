"""Backfill scan discovery, domain selection, and Raman–fluorescence co-occurrence."""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import (
    SolORM, ScanORM, ScanPointORM, SpectrumORM, FittedPeakORM,
)
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.services.base import ServiceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli_ctx(json_mode=False):
    """Create a mock typer.Context for CLI function testing."""
    return SimpleNamespace(obj={"json": json_mode})


def _make_service(engine):
    """Create FittingService wired to an in-memory engine."""
    service = FittingService(database_path=Path(":memory:"))
    service._engine = engine
    return service


def _seed_peak_db(engine):
    """Seed a DB with peaks for co-occurrence queries.

    Creates:
    - Sol 7293, target T7293, scan HDR_1
      - Point 0: sulfate mineral peak (sulf1_v1) + Ce3+ doublet (group1a, group1b)
      - Point 1: organics (D_band, G_band) only
    - Sol 7257, target T7257, scan detail_1
      - Point 0: hydration (OH_stretch) + group2 fluorescence
      - Point 1: low-SNR mineral peak without fluorescence
    """
    point_ids = {}

    with get_session(engine) as session:
        # Sol 7293
        sol293 = SolORM(sol_number=7293, data_source="loupe")
        session.add(sol293)
        session.flush()

        scan1_id = str(uuid.uuid4())
        scan1 = ScanORM(
            id=scan1_id, sol_number=7293, scan_name="HDR_1",
            target="T7293", scan_id="peak_test_1",
            sclk_start=0, n_points=2, n_channels=2148,
        )
        session.add(scan1)
        session.flush()

        # Point 0: sulfate + Ce3+ doublet
        pt0_id = str(uuid.uuid4())
        pt0 = ScanPointORM(id=pt0_id, scan_id=scan1_id, point_index=0)
        session.add(pt0)
        session.flush()
        point_ids[(7293, "T7293", "HDR_1", 0)] = pt0_id

        spec0_r1 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt0_id, region="R1",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec0_r1)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec0_r1.id,
            fit_modality="minerals", center_cm1=1015.0, fwhm_cm1=12.0,
            amplitude=5000.0, snr=303.0, mineral_assignment="sulf1_v1",
        ))

        spec0_r2 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt0_id, region="R2",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec0_r2)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec0_r2.id,
            fit_modality="fluorescence", center_nm=304.1, fwhm_nm=18.0,
            amplitude=8000.0, snr=82.0, mineral_assignment="group1a",
        ))
        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec0_r2.id,
            fit_modality="fluorescence", center_nm=326.1, fwhm_nm=16.0,
            amplitude=6500.0, snr=65.0, mineral_assignment="group1b",
        ))

        # Point 1: organics only (D_band, G_band)
        pt1_id = str(uuid.uuid4())
        pt1 = ScanPointORM(id=pt1_id, scan_id=scan1_id, point_index=1)
        session.add(pt1)
        session.flush()
        point_ids[(7293, "T7293", "HDR_1", 1)] = pt1_id

        spec1_r1 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt1_id, region="R1",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec1_r1)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec1_r1.id,
            fit_modality="organics", center_cm1=1350.0, fwhm_cm1=50.0,
            amplitude=1200.0, snr=15.0, mineral_assignment="D_band",
        ))
        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec1_r1.id,
            fit_modality="organics", center_cm1=1600.0, fwhm_cm1=40.0,
            amplitude=1800.0, snr=25.0, mineral_assignment="G_band",
        ))

        # Sol 7257
        sol257 = SolORM(sol_number=7257, data_source="loupe")
        session.add(sol257)
        session.flush()

        scan2_id = str(uuid.uuid4())
        scan2 = ScanORM(
            id=scan2_id, sol_number=7257, scan_name="detail_1",
            target="T7257", scan_id="peak_test_2",
            sclk_start=0, n_points=2, n_channels=2148,
        )
        session.add(scan2)
        session.flush()

        # Point 0: hydration + group2 fluor
        pt2_id = str(uuid.uuid4())
        pt2 = ScanPointORM(id=pt2_id, scan_id=scan2_id, point_index=0)
        session.add(pt2)
        session.flush()
        point_ids[(7257, "T7257", "detail_1", 0)] = pt2_id

        spec2_r1 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt2_id, region="R1",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec2_r1)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec2_r1.id,
            fit_modality="hydration", center_cm1=3400.0, fwhm_cm1=100.0,
            amplitude=2000.0, snr=20.0, mineral_assignment="OH_stretch",
        ))

        spec2_r2 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt2_id, region="R2",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec2_r2)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec2_r2.id,
            fit_modality="fluorescence", center_nm=340.0, fwhm_nm=25.0,
            amplitude=3000.0, snr=35.0, mineral_assignment="group2",
        ))

        # Point 1: low-SNR mineral peak without fluorescence
        pt3_id = str(uuid.uuid4())
        pt3 = ScanPointORM(id=pt3_id, scan_id=scan2_id, point_index=1)
        session.add(pt3)
        session.flush()
        point_ids[(7257, "T7257", "detail_1", 1)] = pt3_id

        spec3_r1 = SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=pt3_id, region="R1",
            spectrum_type="dark_subtracted", processing_level="raw",
            intensities=b"\x00",
        )
        session.add(spec3_r1)
        session.flush()

        session.add(FittedPeakORM(
            id=str(uuid.uuid4()), spectrum_id=spec3_r1.id,
            fit_modality="minerals", center_cm1=1000.0, fwhm_cm1=10.0,
            amplitude=50.0, snr=1.5, mineral_assignment="sulf1_v1",
        ))

        session.commit()

    return point_ids


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
def peak_db(engine):
    """DB seeded with Raman and fluorescence peaks."""
    point_ids = _seed_peak_db(engine)
    return engine, point_ids


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_rejects_retired_training_command(tmp_path, monkeypatch):
    from sherloc_pipeline.cli.app import app

    monkeypatch.chdir(tmp_path)
    command = "-".join(("extract", "training"))
    result = CliRunner().invoke(app, [command])

    assert result.exit_code == 2
    assert f"No such command '{command}'" in result.output


class TestBackfillDiscovery:
    """Validate backfill discovers scans and dispatches to correct service methods."""

    def test_iter_all_scans_returns_tuples(self, engine):
        """_iter_all_scans returns (sol_number, target, scan_name) tuples."""
        from sherloc_pipeline.cli.app import _iter_all_scans
        from rich.console import Console

        with get_session(engine) as session:
            sol = SolORM(sol_number=7293, data_source="loupe")
            session.add(sol)
            session.flush()
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=7293, scan_name="HDR_1",
                target="T7293", scan_id="iter_test",
                sclk_start=0, n_points=2, n_channels=2148,
            ))
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=7257, scan_name="detail_1",
                target="T7257", scan_id="iter_test2",
                sclk_start=0, n_points=1, n_channels=2148,
            ))
            sol2 = SolORM(sol_number=7257, data_source="loupe")
            session.add(sol2)
            session.commit()

        # Patch get_engine at the source module (local import resolves here)
        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            scans = _iter_all_scans(Path(":memory:"), Console())

        assert len(scans) == 2
        # Ordered by sol_number, target, scan_name
        assert scans[0] == (7257, "T7257", "detail_1")
        assert scans[1] == (7293, "T7293", "HDR_1")

    def test_iter_all_scans_normalizes_target_spaces(self, engine):
        """_iter_all_scans replaces spaces with underscores in target names."""
        from sherloc_pipeline.cli.app import _iter_all_scans
        from rich.console import Console

        with get_session(engine) as session:
            sol = SolORM(sol_number=59, data_source="loupe")
            session.add(sol)
            session.flush()
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=59, scan_name="meteorite",
                target="external calibration", scan_id="norm_test",
                sclk_start=0, n_points=1, n_channels=2148,
            ))
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=59, scan_name="detail_1",
                target="Guillaumes post abrasion 1", scan_id="norm_test2",
                sclk_start=0, n_points=1, n_channels=2148,
            ))
            session.commit()

        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            scans = _iter_all_scans(Path(":memory:"), Console())

        assert len(scans) == 2
        # Spaces in DB target names should be replaced with underscores
        assert scans[0][1] == "Guillaumes_post_abrasion_1"
        assert scans[1][1] == "external_calibration"

    def test_backfill_calls_persist_raman_peaks_per_domain(self, tmp_path):
        """Backfill calls persist_raman_peaks once per Raman domain per scan."""
        mock_service = MagicMock(spec=FittingService)
        mock_service.persist_raman_peaks.return_value = ServiceResult(
            summary="OK", metadata={"peaks_inserted": 5}
        )
        mock_service.fit_fluorescence.return_value = ServiceResult(
            summary="OK", metadata={"peaks_inserted": 3}
        )

        scans = [(7293, "T7293", "HDR_1", "synthetic-id-1")]

        # Create a real file so db_path.exists() passes
        db_file = tmp_path / "test.db"
        db_file.touch()

        from sherloc_pipeline.cli.app import backfill_cmd

        with patch("sherloc_pipeline.cli.app._iter_all_scans", return_value=scans), \
             patch("sherloc_pipeline.services.fitting.FittingService", return_value=mock_service), \
             pytest.raises(SystemExit) as exc_info:
            backfill_cmd(
                ctx=_make_cli_ctx(),
                database=db_file,
                data_dir=None,
                results_dir=None,
                domains=None,
                dry_run=False,
            )

        assert exc_info.value.code == 0

        # 3 Raman domains called
        raman_calls = mock_service.persist_raman_peaks.call_args_list
        assert len(raman_calls) == 3
        domains_called = [c.kwargs["domain"] for c in raman_calls]
        assert domains_called == ["minerals", "organics", "hydration"]

        # 1 fluorescence call
        fluor_calls = mock_service.fit_fluorescence.call_args_list
        assert len(fluor_calls) == 1

    def test_backfill_domain_filter(self, tmp_path):
        """Backfill --domains flag selects only specified domains."""
        mock_service = MagicMock(spec=FittingService)
        mock_service.persist_raman_peaks.return_value = ServiceResult(
            summary="OK", metadata={"peaks_inserted": 5}
        )

        scans = [(7293, "T7293", "HDR_1", "synthetic-id-1")]
        db_file = tmp_path / "test.db"
        db_file.touch()

        from sherloc_pipeline.cli.app import backfill_cmd

        with patch("sherloc_pipeline.cli.app._iter_all_scans", return_value=scans), \
             patch("sherloc_pipeline.services.fitting.FittingService", return_value=mock_service), \
             pytest.raises(SystemExit) as exc_info:
            backfill_cmd(
                ctx=_make_cli_ctx(),
                database=db_file,
                data_dir=None,
                results_dir=None,
                domains="minerals,organics",
                dry_run=False,
            )

        assert exc_info.value.code == 0

        # Only minerals and organics
        raman_calls = mock_service.persist_raman_peaks.call_args_list
        assert len(raman_calls) == 2
        domains_called = [c.kwargs["domain"] for c in raman_calls]
        assert domains_called == ["minerals", "organics"]

        # No fluorescence
        assert mock_service.fit_fluorescence.call_count == 0

    def test_backfill_dry_run_no_service_calls(self, tmp_path):
        """Backfill --dry-run does not call any service methods."""
        import typer

        scans = [(7293, "T7293", "HDR_1", "synthetic-id-1")]
        db_file = tmp_path / "test.db"
        db_file.touch()

        from sherloc_pipeline.cli.app import backfill_cmd

        with patch("sherloc_pipeline.cli.app._iter_all_scans", return_value=scans), \
             patch("sherloc_pipeline.services.fitting.FittingService") as mock_cls, \
             pytest.raises(typer.Exit) as exc_info:
            backfill_cmd(
                ctx=_make_cli_ctx(),
                database=db_file,
                data_dir=None,
                results_dir=None,
                domains=None,
                dry_run=True,
            )

        # Dry run exits with code 0 via typer.Exit
        assert exc_info.value.exit_code == 0
        # FittingService should not have been instantiated
        mock_cls.assert_not_called()

    def test_backfill_error_handling_continues(self, tmp_path):
        """Backfill continues processing after a scan error."""
        mock_service = MagicMock(spec=FittingService)

        # First scan raises, second succeeds
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Scan failed")
            return ServiceResult(summary="OK", metadata={"peaks_inserted": 5})

        mock_service.persist_raman_peaks.side_effect = side_effect
        mock_service.fit_fluorescence.return_value = ServiceResult(
            summary="OK", metadata={"peaks_inserted": 0}
        )

        scans = [(7293, "T7293", "HDR_1", "synthetic-id-1"), (7257, "T7257", "detail_1", "synthetic-id-2")]
        db_file = tmp_path / "test.db"
        db_file.touch()

        from sherloc_pipeline.cli.app import backfill_cmd

        with patch("sherloc_pipeline.cli.app._iter_all_scans", return_value=scans), \
             patch("sherloc_pipeline.services.fitting.FittingService", return_value=mock_service), \
             pytest.raises(SystemExit) as exc_info:
            backfill_cmd(
                ctx=_make_cli_ctx(),
                database=db_file,
                data_dir=None,
                results_dir=None,
                domains="minerals",
                dry_run=False,
            )

        # Exits with code 1 because there were errors
        assert exc_info.value.code == 1
        # Both scans were attempted
        assert mock_service.persist_raman_peaks.call_count == 2


class TestCoOccurrence:
    def test_co_occurrence_idempotent(self, peak_db):
        """Running co-occurrence query twice returns identical results."""
        engine, _ = peak_db
        service = _make_service(engine)

        results1 = service.query_co_occurrences()
        results2 = service.query_co_occurrences()

        assert results1 == results2

    def test_finds_sulfate_ce3(self, peak_db):
        """Finds Point 0 Sol 7293 (sulfate + Ce3+)."""
        engine, _ = peak_db
        service = _make_service(engine)

        results = service.query_co_occurrences()

        assert len(results) == 1
        assert results[0]["sol_number"] == 7293
        assert results[0]["point_index"] == 0

    def test_excludes_organics_from_co_occurrence(self, peak_db):
        """Organics-only point (Point 1 Sol 7293) not in co-occurrence results."""
        engine, point_ids = peak_db
        service = _make_service(engine)

        results = service.query_co_occurrences()

        result_ids = {r["scan_point_id"] for r in results}
        assert point_ids[(7293, "T7293", "HDR_1", 1)] not in result_ids

    def test_hydration_group2_not_in_default_query(self, peak_db):
        """Hydration + group2 point not matched by default sulfate+Ce3+ query."""
        engine, point_ids = peak_db
        service = _make_service(engine)

        results = service.query_co_occurrences()  # Default: sulf% + group1a/1b

        result_ids = {r["scan_point_id"] for r in results}
        assert point_ids[(7257, "T7257", "detail_1", 0)] not in result_ids

    def test_custom_hydration_group2_query(self, peak_db):
        """Custom query finds hydration + group2 co-occurrence."""
        engine, _ = peak_db
        service = _make_service(engine)

        results = service.query_co_occurrences(
            raman_modality="hydration",
            raman_assignment_pattern="OH%",
            fluor_groups=["group2"],
        )

        assert len(results) == 1
        assert results[0]["sol_number"] == 7257
        assert results[0]["point_index"] == 0
