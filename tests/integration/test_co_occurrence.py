"""
Smoke tests for co-occurrence query validation (spec §9.2).

Covers:
- Co-occurrence query runs successfully with synthetic data
- Finds scan points with sulfate Raman + Ce3+ fluorescence
- Domain isolation: non-matching domains excluded
- Empty results when no co-occurrence
- Custom raman_modality and fluor_groups parameters
- Real DB: EXPLAIN QUERY PLAN shows composite index usage
- Real DB: query performance acceptable

AC: bd-3cqz.27 (spec step 6.3)
"""

import uuid

import numpy as np
import pytest
from pathlib import Path

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import (
    SolORM, ScanORM, ScanPointORM, SpectrumORM, FittedPeakORM,
)
from sherloc_pipeline.services.fitting import FittingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(engine):
    """Create FittingService wired to an in-memory engine."""
    service = FittingService(database_path=Path(":memory:"))
    service._engine = engine
    return service


def _seed_co_occurrence_db(engine):
    """Seed a DB with scan points that have both sulfate Raman and Ce3+ fluorescence peaks.

    Creates:
    - Sol 293, target Quartier, scan HDR_1
    - Point 0: sulfate Raman (sulf1_v1) + Ce3+ fluorescence (group1a, group1b) -> co-occurrence
    - Point 1: sulfate Raman (sulf2_v1) only -> no fluor, no co-occurrence
    - Point 2: Ce3+ fluorescence (group1a) only -> no Raman, no co-occurrence
    - Point 3: olivine Raman + Ce3+ fluorescence (group1b) -> no sulfate, no default co-occurrence
    """
    scan_point_ids = {}

    with get_session(engine) as session:
        sol = SolORM(sol_number=293, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id, sol_number=293, scan_name="HDR_1",
            target="Quartier", scan_id="co_occur_test",
            sclk_start=0, n_points=4, n_channels=2148,
        )
        session.add(scan)
        session.flush()

        for i in range(4):
            pt_id = str(uuid.uuid4())
            pt = ScanPointORM(id=pt_id, scan_id=scan_id, point_index=i)
            session.add(pt)
            session.flush()
            scan_point_ids[i] = pt_id

            # Each point needs at least one spectrum to link peaks
            spec_id = str(uuid.uuid4())
            spec = SpectrumORM(
                id=spec_id, scan_point_id=pt_id, region="R1",
                spectrum_type="dark_subtracted", processing_level="raw",
                intensities=b"\x00",
            )
            session.add(spec)
            session.flush()

            # Point 0: sulfate + Ce3+ doublet
            if i == 0:
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=spec_id, fit_modality="minerals",
                    center_cm1=1015.0, fwhm_cm1=12.0, amplitude=5000.0,
                    snr=303.0, mineral_assignment="sulf1_v1",
                ))
                # Add fluor on a separate spectrum (R2)
                fluor_spec_id = str(uuid.uuid4())
                fluor_spec = SpectrumORM(
                    id=fluor_spec_id, scan_point_id=pt_id, region="R2",
                    spectrum_type="dark_subtracted", processing_level="raw",
                    intensities=b"\x00",
                )
                session.add(fluor_spec)
                session.flush()
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=fluor_spec_id, fit_modality="fluorescence",
                    center_nm=304.1, fwhm_nm=18.0, amplitude=8000.0,
                    snr=82.0, mineral_assignment="group1a",
                ))
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=fluor_spec_id, fit_modality="fluorescence",
                    center_nm=326.1, fwhm_nm=16.0, amplitude=6500.0,
                    snr=65.0, mineral_assignment="group1b",
                ))

            # Point 1: sulfate only (no fluor)
            elif i == 1:
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=spec_id, fit_modality="minerals",
                    center_cm1=1130.0, fwhm_cm1=14.0, amplitude=3200.0,
                    snr=150.0, mineral_assignment="sulf2_v1",
                ))

            # Point 2: Ce3+ fluor only (no Raman)
            elif i == 2:
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=spec_id, fit_modality="fluorescence",
                    center_nm=305.5, fwhm_nm=20.0, amplitude=4000.0,
                    snr=40.0, mineral_assignment="group1a",
                ))

            # Point 3: olivine (not sulfate) + Ce3+ fluor
            elif i == 3:
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=spec_id, fit_modality="minerals",
                    center_cm1=856.0, fwhm_cm1=20.0, amplitude=2500.0,
                    snr=80.0, mineral_assignment="olivine",
                ))
                fluor_spec_id2 = str(uuid.uuid4())
                fluor_spec2 = SpectrumORM(
                    id=fluor_spec_id2, scan_point_id=pt_id, region="R2",
                    spectrum_type="dark_subtracted", processing_level="raw",
                    intensities=b"\x00",
                )
                session.add(fluor_spec2)
                session.flush()
                session.add(FittedPeakORM(
                    id=str(uuid.uuid4()),
                    spectrum_id=fluor_spec_id2, fit_modality="fluorescence",
                    center_nm=326.0, fwhm_nm=15.0, amplitude=3000.0,
                    snr=30.0, mineral_assignment="group1b",
                ))

        session.commit()

    return scan_point_ids


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
def co_occur_db(engine):
    """DB seeded with co-occurrence test data."""
    scan_point_ids = _seed_co_occurrence_db(engine)
    return engine, scan_point_ids


# ---------------------------------------------------------------------------
# Tests: co-occurrence query with synthetic data
# ---------------------------------------------------------------------------

class TestCoOccurrenceQuery:
    """Validate co-occurrence query logic per spec §9.2."""

    def test_query_runs_successfully(self, co_occur_db):
        """Co-occurrence query runs without errors."""
        engine, _ = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()
        assert isinstance(results, list)

    def test_finds_sulfate_ce3_co_occurrence(self, co_occur_db):
        """Finds scan points with sulfate Raman + Ce3+ fluorescence."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()

        assert len(results) == 1
        assert results[0]["scan_point_id"] == scan_point_ids[0]
        assert results[0]["sol_number"] == 293
        assert results[0]["target"] == "Quartier"
        assert results[0]["point_index"] == 0

    def test_excludes_sulfate_only(self, co_occur_db):
        """Point 1 (sulfate, no fluor) is excluded."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()

        result_ids = {r["scan_point_id"] for r in results}
        assert scan_point_ids[1] not in result_ids

    def test_excludes_fluor_only(self, co_occur_db):
        """Point 2 (Ce3+ fluor, no Raman) is excluded."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()

        result_ids = {r["scan_point_id"] for r in results}
        assert scan_point_ids[2] not in result_ids

    def test_excludes_non_sulfate_raman(self, co_occur_db):
        """Point 3 (olivine + Ce3+) excluded from default sulfate query."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()

        result_ids = {r["scan_point_id"] for r in results}
        assert scan_point_ids[3] not in result_ids

    def test_custom_raman_pattern_olivine(self, co_occur_db):
        """Custom raman_assignment_pattern finds olivine + Ce3+ co-occurrence."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences(
            raman_assignment_pattern="oliv%",
        )

        assert len(results) == 1
        assert results[0]["scan_point_id"] == scan_point_ids[3]
        assert results[0]["point_index"] == 3

    def test_custom_fluor_groups(self, co_occur_db):
        """Custom fluor_groups restricts to specific groups."""
        engine, scan_point_ids = co_occur_db
        service = _make_service(engine)

        # Only group1a — should still find point 0 (has group1a)
        results = service.query_co_occurrences(fluor_groups=["group1a"])
        assert len(results) == 1
        assert results[0]["scan_point_id"] == scan_point_ids[0]

    def test_no_match_returns_empty(self, co_occur_db):
        """Non-existent pattern returns empty list."""
        engine, _ = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences(
            raman_assignment_pattern="phosph%",
        )
        assert results == []

    def test_empty_db_returns_empty(self, engine):
        """Query on empty DB returns empty list."""
        service = _make_service(engine)
        results = service.query_co_occurrences()
        assert results == []

    def test_result_structure(self, co_occur_db):
        """Each result dict has expected keys."""
        engine, _ = co_occur_db
        service = _make_service(engine)
        results = service.query_co_occurrences()

        assert len(results) > 0
        for r in results:
            assert "scan_point_id" in r
            assert "sol_number" in r
            assert "target" in r
            assert "point_index" in r


# ---------------------------------------------------------------------------
# Tests: real database validation (index usage, performance)
# ---------------------------------------------------------------------------

REAL_DB = Path("./phase.db")


@pytest.mark.skipif(
    not REAL_DB.exists(),
    reason="Real database not available",
)
class TestCoOccurrenceRealDB:
    """Validate co-occurrence query against production database."""

    def test_explain_uses_composite_index(self):
        """EXPLAIN QUERY PLAN shows ix_fitted_peaks_modality_assignment is used."""
        import sqlite3

        conn = sqlite3.connect(str(REAL_DB))
        cur = conn.cursor()

        sql = """
        EXPLAIN QUERY PLAN
        SELECT DISTINCT sp.id, sc.sol_number, sc.target, sp.point_index
        FROM scan_points sp
        JOIN scans sc ON sc.id = sp.scan_id
        JOIN spectra s ON s.scan_point_id = sp.id
        JOIN fitted_peaks fp ON fp.spectrum_id = s.id
        WHERE sp.id IN (
            SELECT sp2.id FROM scan_points sp2
            JOIN spectra s2 ON s2.scan_point_id = sp2.id
            JOIN fitted_peaks fp2 ON fp2.spectrum_id = s2.id
            WHERE fp2.fit_modality = 'minerals'
              AND fp2.mineral_assignment LIKE 'sulf%'
        )
        AND fp.fit_modality = 'fluorescence'
        AND fp.mineral_assignment IN ('group1a', 'group1b');
        """

        cur.execute(sql)
        plan_rows = cur.fetchall()
        conn.close()

        # Join all plan text into one string for searching
        plan_text = " ".join(str(row) for row in plan_rows)
        assert "ix_fitted_peaks_modality_assignment" in plan_text, (
            f"Composite index not used in query plan:\n{plan_text}"
        )

    def test_query_performance_under_100ms(self):
        """Co-occurrence query completes in <100ms on full dataset."""
        import sqlite3
        import time

        conn = sqlite3.connect(str(REAL_DB))
        cur = conn.cursor()

        sql = """
        SELECT DISTINCT sp.id, sc.sol_number, sc.target, sp.point_index
        FROM scan_points sp
        JOIN scans sc ON sc.id = sp.scan_id
        JOIN spectra s ON s.scan_point_id = sp.id
        JOIN fitted_peaks fp ON fp.spectrum_id = s.id
        WHERE sp.id IN (
            SELECT sp2.id FROM scan_points sp2
            JOIN spectra s2 ON s2.scan_point_id = sp2.id
            JOIN fitted_peaks fp2 ON fp2.spectrum_id = s2.id
            WHERE fp2.fit_modality = 'minerals'
              AND fp2.mineral_assignment LIKE 'sulf%'
        )
        AND fp.fit_modality = 'fluorescence'
        AND fp.mineral_assignment IN ('group1a', 'group1b');
        """

        t0 = time.perf_counter()
        cur.execute(sql)
        _ = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        conn.close()

        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms (limit 100ms)"

    def test_orm_query_runs_on_real_db(self):
        """ORM-based query_co_occurrences() runs without error on real DB."""
        service = FittingService(database_path=REAL_DB)
        results = service.query_co_occurrences()
        # Results may be empty pre-backfill, but query must succeed
        assert isinstance(results, list)
