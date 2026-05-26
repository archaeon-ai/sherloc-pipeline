"""
Tests for backfill CLI and training data extraction (spec §9.3).

Covers:
- Backfill script discovers scans and calls persist_raman_peaks per domain
- Backfill idempotency: running twice produces same result
- JSONL extraction format matches spec §9.3
- Co-occurrence query returns expected results on fixture data (extended)

AC: bd-3cqz.39 (spec step 6.4)
"""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import (
    SolORM, ScanORM, ScanPointORM, SpectrumORM, FittedPeakORM,
)
from sherloc_pipeline.services.fitting import FittingService, _build_phase_label
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


def _seed_training_db(engine):
    """Seed a DB with diverse peaks for training data extraction.

    Creates:
    - Sol 293, target Quartier, scan HDR_1
      - Point 0: sulfate mineral peak (sulf1_v1) + Ce3+ doublet (group1a, group1b)
      - Point 1: organics (D_band, G_band) only
    - Sol 310, target Dourbes, scan detail_1
      - Point 0: hydration (OH_stretch) + group2 fluorescence
      - Point 1: low-SNR peak (below threshold) — should be filtered out
    """
    point_ids = {}

    with get_session(engine) as session:
        # Sol 293
        sol293 = SolORM(sol_number=293, data_source="loupe")
        session.add(sol293)
        session.flush()

        scan1_id = str(uuid.uuid4())
        scan1 = ScanORM(
            id=scan1_id, sol_number=293, scan_name="HDR_1",
            target="Quartier", scan_id="train_test_1",
            sclk_start=0, n_points=2, n_channels=2148,
        )
        session.add(scan1)
        session.flush()

        # Point 0: sulfate + Ce3+ doublet
        pt0_id = str(uuid.uuid4())
        pt0 = ScanPointORM(id=pt0_id, scan_id=scan1_id, point_index=0)
        session.add(pt0)
        session.flush()
        point_ids[(293, "Quartier", "HDR_1", 0)] = pt0_id

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
        point_ids[(293, "Quartier", "HDR_1", 1)] = pt1_id

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

        # Sol 310
        sol310 = SolORM(sol_number=310, data_source="loupe")
        session.add(sol310)
        session.flush()

        scan2_id = str(uuid.uuid4())
        scan2 = ScanORM(
            id=scan2_id, sol_number=310, scan_name="detail_1",
            target="Dourbes", scan_id="train_test_2",
            sclk_start=0, n_points=2, n_channels=2148,
        )
        session.add(scan2)
        session.flush()

        # Point 0: hydration + group2 fluor
        pt2_id = str(uuid.uuid4())
        pt2 = ScanPointORM(id=pt2_id, scan_id=scan2_id, point_index=0)
        session.add(pt2)
        session.flush()
        point_ids[(310, "Dourbes", "detail_1", 0)] = pt2_id

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

        # Point 1: low-SNR peak (should be filtered at threshold 2.0)
        pt3_id = str(uuid.uuid4())
        pt3 = ScanPointORM(id=pt3_id, scan_id=scan2_id, point_index=1)
        session.add(pt3)
        session.flush()
        point_ids[(310, "Dourbes", "detail_1", 1)] = pt3_id

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
def training_db(engine):
    """DB seeded with diverse training data."""
    point_ids = _seed_training_db(engine)
    return engine, point_ids


# ---------------------------------------------------------------------------
# Tests: Backfill CLI
# ---------------------------------------------------------------------------

class TestBackfillDiscovery:
    """Validate backfill discovers scans and dispatches to correct service methods."""

    def test_iter_all_scans_returns_tuples(self, engine):
        """_iter_all_scans returns (sol_number, target, scan_name) tuples."""
        from sherloc_pipeline.cli.app import _iter_all_scans
        from rich.console import Console

        with get_session(engine) as session:
            sol = SolORM(sol_number=293, data_source="loupe")
            session.add(sol)
            session.flush()
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=293, scan_name="HDR_1",
                target="Quartier", scan_id="iter_test",
                sclk_start=0, n_points=2, n_channels=2148,
            ))
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=310, scan_name="detail_1",
                target="Dourbes", scan_id="iter_test2",
                sclk_start=0, n_points=1, n_channels=2148,
            ))
            sol2 = SolORM(sol_number=310, data_source="loupe")
            session.add(sol2)
            session.commit()

        # Patch get_engine at the source module (local import resolves here)
        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            scans = _iter_all_scans(Path(":memory:"), Console())

        assert len(scans) == 2
        # Ordered by sol_number, target, scan_name
        assert scans[0] == (293, "Quartier", "HDR_1")
        assert scans[1] == (310, "Dourbes", "detail_1")

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

        scans = [(293, "Quartier", "HDR_1")]

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

        scans = [(293, "Quartier", "HDR_1")]
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

        scans = [(293, "Quartier", "HDR_1")]
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

        scans = [(293, "Quartier", "HDR_1"), (310, "Dourbes", "detail_1")]
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


class TestBackfillIdempotency:
    """Validate backfill produces same result when run twice."""

    def test_extract_training_twice_same_output(self, training_db, tmp_path):
        """Running extract_training_jsonl twice produces identical output."""
        engine, _ = training_db
        service = _make_service(engine)

        path1 = tmp_path / "run1.jsonl"
        path2 = tmp_path / "run2.jsonl"

        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            result1 = service.extract_training_jsonl(path1)
            result2 = service.extract_training_jsonl(path2)

        assert result1.metadata["total_records"] == result2.metadata["total_records"]
        assert result1.metadata["total_peaks_queried"] == result2.metadata["total_peaks_queried"]
        assert path1.read_text() == path2.read_text()

    def test_co_occurrence_idempotent(self, training_db):
        """Running co-occurrence query twice returns identical results."""
        engine, _ = training_db
        service = _make_service(engine)

        results1 = service.query_co_occurrences()
        results2 = service.query_co_occurrences()

        assert results1 == results2


# ---------------------------------------------------------------------------
# Tests: JSONL extraction format (spec §9.3)
# ---------------------------------------------------------------------------

class TestJSONLExtractionFormat:
    """Validate JSONL output matches spec §9.3 format."""

    def _extract(self, engine, out_path, snr_threshold=2.0):
        """Helper: extract JSONL with get_engine patched to our engine."""
        service = _make_service(engine)
        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            return service.extract_training_jsonl(out_path, snr_threshold=snr_threshold)

    def test_output_is_valid_jsonl(self, training_db, tmp_path):
        """Each line is valid JSON with 'input' and 'output' keys."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            record = json.loads(line)
            assert "input" in record
            assert "output" in record
            assert isinstance(record["input"], str)
            assert isinstance(record["output"], str)

    def test_record_count_matches_metadata(self, training_db, tmp_path):
        """JSONL line count matches metadata total_records."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        result = self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == result.metadata["total_records"]

    def test_snr_filtering(self, training_db, tmp_path):
        """Peaks below SNR threshold are excluded."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        # Default threshold 2.0 — low-SNR peak (1.5) should be excluded
        self._extract(engine, out, snr_threshold=2.0)

        lines = out.read_text().strip().split("\n")
        for line in lines:
            record = json.loads(line)
            # The low-SNR point (Sol 310 point 1) should not appear
            assert "Point 1, Sol 310" not in record["input"]

    def test_high_snr_threshold_reduces_records(self, training_db, tmp_path):
        """Higher SNR threshold produces fewer records."""
        engine, _ = training_db

        out_low = tmp_path / "low.jsonl"
        out_high = tmp_path / "high.jsonl"

        result_low = self._extract(engine, out_low, snr_threshold=2.0)
        result_high = self._extract(engine, out_high, snr_threshold=100.0)

        assert result_high.metadata["total_records"] <= result_low.metadata["total_records"]

    def test_input_contains_point_header(self, training_db, tmp_path):
        """Input text starts with 'Point N, Sol M target scan.'."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        for line in lines:
            record = json.loads(line)
            assert record["input"].startswith("Point ")
            assert "Sol " in record["input"]

    def test_raman_peaks_format(self, training_db, tmp_path):
        """Raman peaks described as 'label center cm-1 (SNR N)'."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        sulfate_record = None
        for line in lines:
            record = json.loads(line)
            if "sulf1_v1" in record["input"]:
                sulfate_record = record
                break

        assert sulfate_record is not None, "Sulfate record not found"
        assert "Raman: sulf1_v1 1015.0 cm-1 (SNR 303)" in sulfate_record["input"]

    def test_fluor_peaks_format(self, training_db, tmp_path):
        """Fluorescence peaks described as 'group center nm (SNR N)'."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        ce3_record = None
        for line in lines:
            record = json.loads(line)
            if "group1a" in record["input"]:
                ce3_record = record
                break

        assert ce3_record is not None, "Ce3+ record not found"
        assert "Fluor: group1a 304.1 nm (SNR 82)" in ce3_record["input"]
        assert "group1b 326.1 nm (SNR 65)" in ce3_record["input"]

    def test_doublet_detection_in_output(self, training_db, tmp_path):
        """Doublets are detected and included in input text."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        ce3_record = None
        for line in lines:
            record = json.loads(line)
            if "group1a" in record["input"] and "group1b" in record["input"]:
                ce3_record = record
                break

        assert ce3_record is not None
        assert "Doublet ratio" in ce3_record["input"]
        assert "sep" in ce3_record["input"]
        assert "nm." in ce3_record["input"]

    def test_organics_record_format(self, training_db, tmp_path):
        """Organic peaks (D_band, G_band) appear in Raman section."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        organic_record = None
        for line in lines:
            record = json.loads(line)
            if "D_band" in record["input"]:
                organic_record = record
                break

        assert organic_record is not None
        assert "Raman:" in organic_record["input"]
        assert "D_band" in organic_record["input"]
        assert "G_band" in organic_record["input"]
        assert "cm-1" in organic_record["input"]

    def test_hydration_with_group2_fluor(self, training_db, tmp_path):
        """Hydration + group2 fluorescence record format."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        self._extract(engine, out)

        lines = out.read_text().strip().split("\n")
        hydration_record = None
        for line in lines:
            record = json.loads(line)
            if "OH_stretch" in record["input"]:
                hydration_record = record
                break

        assert hydration_record is not None
        assert "Raman: OH_stretch" in hydration_record["input"]
        assert "Fluor: group2" in hydration_record["input"]

    def test_metadata_has_required_fields(self, training_db, tmp_path):
        """ServiceResult metadata includes counts and threshold."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        result = self._extract(engine, out)

        assert "total_records" in result.metadata
        assert "total_peaks_queried" in result.metadata
        assert "total_doublets" in result.metadata
        assert "snr_threshold" in result.metadata
        assert result.metadata["snr_threshold"] == 2.0

    def test_artifacts_contain_output_path(self, training_db, tmp_path):
        """ServiceResult artifacts list includes the output path."""
        engine, _ = training_db
        out = tmp_path / "training.jsonl"

        result = self._extract(engine, out)

        assert out in result.artifacts

    def test_output_creates_parent_dirs(self, training_db, tmp_path):
        """Output path parent directories are created if missing."""
        engine, _ = training_db
        out = tmp_path / "nested" / "dir" / "training.jsonl"

        result = self._extract(engine, out)

        assert out.exists()
        assert result.metadata["total_records"] > 0

    def test_no_database_raises(self):
        """extract_training_jsonl raises FittingError without database."""
        from sherloc_pipeline.services.errors import FittingError

        service = FittingService()
        with pytest.raises(FittingError, match="Database path required"):
            service.extract_training_jsonl(Path("/tmp/test.jsonl"))

    def test_empty_db_produces_empty_jsonl(self, engine, tmp_path):
        """Empty database produces empty JSONL file."""
        out = tmp_path / "empty.jsonl"

        with patch("sherloc_pipeline.database.connection.get_engine", return_value=engine):
            service = _make_service(engine)
            result = service.extract_training_jsonl(out)

        assert result.metadata["total_records"] == 0
        assert out.read_text() == ""


# ---------------------------------------------------------------------------
# Tests: _build_phase_label
# ---------------------------------------------------------------------------

class TestBuildPhaseLabel:
    """Validate phase label generation for training data output."""

    def _make_peak(self, assignment, modality="minerals"):
        """Create a minimal mock peak with mineral_assignment."""
        peak = MagicMock()
        peak.mineral_assignment = assignment
        peak.fit_modality = modality
        return peak

    def test_sulfate_plus_doublet(self):
        """Sulfate Raman + Ce3+ doublet -> 'Ce3+-bearing sulfate'."""
        raman = [self._make_peak("sulf1_v1")]
        fluor = [
            self._make_peak("group1a", "fluorescence"),
            self._make_peak("group1b", "fluorescence"),
        ]
        doublets = [MagicMock()]  # Non-empty doublet list

        label = _build_phase_label(raman, fluor, doublets)

        assert "Ce3+-bearing" in label
        assert "sulfate" in label
        assert "Ce3+ fluorescent doublet" in label
        assert "Ca-sulfate sulf1_v1 Raman" in label

    def test_organics_label(self):
        """D_band + G_band -> 'organic carbon'."""
        raman = [
            self._make_peak("D_band", "organics"),
            self._make_peak("G_band", "organics"),
        ]

        label = _build_phase_label(raman, [], [])

        assert "organic carbon" in label
        assert "organic D_band" in label
        assert "organic G_band" in label

    def test_hydration_label(self):
        """OH_stretch -> 'hydrated phase'."""
        raman = [self._make_peak("OH_stretch", "hydration")]

        label = _build_phase_label(raman, [], [])

        assert "hydrated phase" in label
        assert "hydration OH_stretch" in label

    def test_carbonate_label(self):
        """Carbonate assignment -> 'carbonate'."""
        raman = [self._make_peak("carb1_v1")]

        label = _build_phase_label(raman, [], [])

        assert "carbonate" in label
        assert "carbonate carb1_v1 Raman" in label

    def test_group2_fluorescence(self):
        """Group2 -> 'phosphate Ce3+ fluorescence' evidence."""
        fluor = [self._make_peak("group2", "fluorescence")]

        label = _build_phase_label([], fluor, [])

        assert "phosphate Ce3+ fluorescence" in label

    def test_group3_fluorescence(self):
        """Group3 -> 'silicate defect fluorescence' evidence."""
        fluor = [self._make_peak("group3", "fluorescence")]

        label = _build_phase_label([], fluor, [])

        assert "silicate defect fluorescence" in label

    def test_no_peaks_unidentified(self):
        """No peaks -> 'unidentified phase'."""
        label = _build_phase_label([], [], [])

        assert "unidentified phase" in label
        assert "no diagnostic features" in label

    def test_ce3_without_doublet(self):
        """Group1a/1b without doublet still gives Ce3+-bearing."""
        fluor = [self._make_peak("group1a", "fluorescence")]

        label = _build_phase_label([], fluor, [])

        assert "Ce3+-bearing" in label
        assert "Ce3+ fluorescence" in label

    def test_deduplicates_phase_parts(self):
        """Multiple sulfate peaks don't duplicate 'sulfate' in phase."""
        raman = [
            self._make_peak("sulf1_v1"),
            self._make_peak("sulf2_v1"),
        ]

        label = _build_phase_label(raman, [], [])

        # "sulfate" should appear exactly once in the phase part
        phase_part = label.split("Evidence:")[0]
        assert phase_part.count("sulfate") == 1


# ---------------------------------------------------------------------------
# Tests: Co-occurrence on training fixture (extended from 6.3)
# ---------------------------------------------------------------------------

class TestCoOccurrenceOnTrainingData:
    """Validate co-occurrence queries work on the richer training fixture."""

    def test_finds_sulfate_ce3_in_training_db(self, training_db):
        """Finds Point 0 Sol 293 (sulfate + Ce3+)."""
        engine, point_ids = training_db
        service = _make_service(engine)

        results = service.query_co_occurrences()

        assert len(results) == 1
        assert results[0]["sol_number"] == 293
        assert results[0]["point_index"] == 0

    def test_excludes_organics_from_co_occurrence(self, training_db):
        """Organics-only point (Point 1 Sol 293) not in co-occurrence results."""
        engine, point_ids = training_db
        service = _make_service(engine)

        results = service.query_co_occurrences()

        result_ids = {r["scan_point_id"] for r in results}
        assert point_ids[(293, "Quartier", "HDR_1", 1)] not in result_ids

    def test_hydration_group2_not_in_default_query(self, training_db):
        """Hydration + group2 point not matched by default sulfate+Ce3+ query."""
        engine, point_ids = training_db
        service = _make_service(engine)

        results = service.query_co_occurrences()  # Default: sulf% + group1a/1b

        result_ids = {r["scan_point_id"] for r in results}
        assert point_ids[(310, "Dourbes", "detail_1", 0)] not in result_ids

    def test_custom_hydration_group2_query(self, training_db):
        """Custom query finds hydration + group2 co-occurrence."""
        engine, _ = training_db
        service = _make_service(engine)

        results = service.query_co_occurrences(
            raman_modality="hydration",
            raman_assignment_pattern="OH%",
            fluor_groups=["group2"],
        )

        assert len(results) == 1
        assert results[0]["sol_number"] == 310
        assert results[0]["point_index"] == 0
