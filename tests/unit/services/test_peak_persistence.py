"""
Tests for FittingService.persist_fitted_peaks() method.

This module tests the peak persistence functionality that loads fitted peaks
from minerals_fit CSVs and inserts them into the phase.db fitted_peaks table.
"""

import uuid
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import SolORM, ScanORM, ScanPointORM, SpectrumORM, FittedPeakORM
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.services.errors import FittingError


@pytest.fixture
def engine():
    """Create in-memory SQLite database with full schema."""
    eng = get_engine(":memory:")
    create_all_tables(eng)
    return eng


@pytest.fixture
def populated_db(engine):
    """Create Sol 851, scan, 3 points with laser_normalized spectra."""
    spectrum_ids = {}
    with get_session(engine) as session:
        # Create Sol
        sol = SolORM(sol_number=851, data_source="loupe")
        session.add(sol)
        session.flush()

        # Create Scan
        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=851,
            scan_name="detail_1",
            target="Lake_Haiyaha",
            scan_id="test_scan_id",
            sclk_start=0,
            n_points=3,
            n_channels=2148
        )
        session.add(scan)
        session.flush()

        # Create ScanPoints with Spectra
        for i in range(3):
            pt_id = str(uuid.uuid4())
            pt = ScanPointORM(id=pt_id, scan_id=scan_id, point_index=i)
            session.add(pt)
            session.flush()

            spec_id = str(uuid.uuid4())
            spec = SpectrumORM(
                id=spec_id,
                scan_point_id=pt_id,
                region="R1",
                spectrum_type="laser_normalized",
                processing_level="normalized",
                intensities=b"\x00"
            )
            session.add(spec)
            session.flush()
            spectrum_ids[i] = spec_id

    return engine, spectrum_ids


@pytest.fixture
def mock_csvs(tmp_path):
    """Create mock minerals_fit directory with CSVs."""
    results_base = tmp_path / "results"
    minerals_dir = results_base / "minerals_fit"
    minerals_dir.mkdir(parents=True)

    # Point 0: 2 reviewable peaks + 1 non-reviewable (low SNR and FWHM)
    pd.DataFrame([
        {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0, "area": 100.0, "snr": 15.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
        {"center_cm1": 1300.0, "fwhm_cm1": 28.0, "amplitude_a": 800.0, "area": 50.0, "snr": 8.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": False},
        {"center_cm1": 500.0, "fwhm_cm1": 10.0, "amplitude_a": 200.0, "area": 5.0, "snr": 2.0, "pass_snr": False, "pass_fwhm": False, "pass_r2": False},
    ]).to_csv(minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

    # Point 1: 1 reviewable peak
    pd.DataFrame([
        {"center_cm1": 700.0, "fwhm_cm1": 35.0, "amplitude_a": 600.0, "area": 40.0, "snr": 5.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
    ]).to_csv(minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point1_fit_peaks.csv", index=False)

    # Point 2: empty CSV (headers only)
    pd.DataFrame(columns=["center_cm1", "fwhm_cm1", "amplitude_a", "area", "snr", "pass_snr", "pass_fwhm", "pass_r2"]).to_csv(
        minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point2_fit_peaks.csv", index=False
    )

    # AICc summary with R² values (at results_base level, not inside minerals_fit)
    pd.DataFrame([
        {"point": 0, "r2": 0.95},
        {"point": 1, "r2": 0.88},
        {"point": 2, "r2": 0.72},
    ]).to_csv(results_base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

    return results_base


def _make_service(engine):
    """Create FittingService with engine pre-injected."""
    service = FittingService(database_path=Path(":memory:"))
    service._engine = engine  # Use pre-populated in-memory engine
    return service


def _call_persist(service, results_base):
    """Call persist_fitted_peaks with mocked resolve_scan_context and DataIngestion."""
    mock_ctx = MagicMock()
    mock_ctx.base_data_dir = results_base
    mock_ctx.results_dir = results_base

    mock_di = MagicMock()
    mock_di.get_results_path.return_value = results_base

    with patch('sherloc_pipeline.services.fitting.resolve_scan_context') as mock_rsc, \
         patch('sherloc_pipeline.core.data_ingestion.DataIngestion', return_value=mock_di) as mock_di_cls:
        mock_rsc.return_value = mock_ctx
        return service.persist_fitted_peaks(
            sol="0851", target="Lake_Haiyaha", scan="detail_1"
        )


def test_persist_inserts_reviewable_peaks(populated_db, mock_csvs):
    """Verify correct peak count and field mapping from CSVs.

    Only reviewable peaks (SNR >= 3.0 AND FWHM >= 25.0) should be inserted.
    Point 0 has 2 reviewable + 1 non-reviewable.
    Point 1 has 1 reviewable.
    Point 2 has 0 peaks.
    Total: 3 reviewable peaks.
    """
    engine, spectrum_ids = populated_db
    service = _make_service(engine)

    result = _call_persist(service, mock_csvs)

    # Check result
    assert result.metadata["peaks_inserted"] == 3
    assert result.metadata["total_reviewable_peaks"] == 3
    assert "3 minerals peaks" in result.summary

    # Verify peaks in database
    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).all()
        assert len(peaks) == 3

        # Check field mapping for point 0's first peak (1085.5 cm⁻¹)
        peak_1085 = next((p for p in peaks if abs(p.center_cm1 - 1085.5) < 0.1), None)
        assert peak_1085 is not None
        assert peak_1085.spectrum_id == spectrum_ids[0]
        assert peak_1085.peak_type == "gaussian"
        assert peak_1085.amplitude == 1500.0
        assert peak_1085.fwhm_cm1 == 30.0
        assert peak_1085.area == 100.0
        assert peak_1085.snr == 15.0
        assert peak_1085.fit_quality == 0.95  # R² from AICc for point 0


def test_persist_replaces_on_rerun(populated_db, mock_csvs):
    """Verify delete+insert idempotency.

    Run persist twice, verify peak count is same (not doubled).
    """
    engine, spectrum_ids = populated_db
    service = _make_service(engine)

    # First run
    result1 = _call_persist(service, mock_csvs)
    assert result1.metadata["peaks_inserted"] == 3

    # Second run - should delete existing and re-insert
    result2 = _call_persist(service, mock_csvs)
    assert result2.metadata["peaks_inserted"] == 3

    # Verify count in DB is still 3 (not 6)
    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).all()
        assert len(peaks) == 3


def test_persist_no_spectra_warns(populated_db, mock_csvs):
    """Verify missing spectrum for a point generates warning in result."""
    engine, spectrum_ids = populated_db

    # Add CSV for point 5 (which doesn't exist in DB)
    minerals_dir = mock_csvs / "minerals_fit"
    pd.DataFrame([
        {"center_cm1": 900.0, "fwhm_cm1": 30.0, "amplitude_a": 500.0, "area": 30.0, "snr": 10.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
    ]).to_csv(minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point5_fit_peaks.csv", index=False)

    service = _make_service(engine)
    result = _call_persist(service, mock_csvs)

    # Should have warning about point 5
    assert len(result.warnings) > 0
    assert any("Point 5" in w and "not found in database" in w for w in result.warnings)

    # Should still insert the 3 valid peaks
    assert result.metadata["peaks_inserted"] == 3


def test_persist_no_db_raises(mock_csvs):
    """Verify FittingError when database_path not configured (is None)."""
    service = FittingService(database_path=None)

    with pytest.raises(FittingError) as exc_info:
        _call_persist(service, mock_csvs)

    assert "No database_path configured" in str(exc_info.value)
    assert exc_info.value.exit_code == 1


def test_persist_scan_not_found_raises(engine, mock_csvs):
    """Verify FittingError for unknown scan coordinates."""
    # Create service with empty DB (no sol/scan)
    service = _make_service(engine)

    with pytest.raises(FittingError) as exc_info:
        _call_persist(service, mock_csvs)

    assert "Scan not found in database" in str(exc_info.value)
    assert exc_info.value.context["sol"] == "0851"
    assert exc_info.value.context["target"] == "Lake_Haiyaha"


def test_persist_empty_csv_ok(populated_db, tmp_path):
    """Verify zero peaks inserted when CSVs have headers but no data rows.

    Should complete successfully with no errors.
    """
    engine, spectrum_ids = populated_db

    # Create directory with only empty CSVs
    results_base = tmp_path / "results_empty"
    minerals_dir = results_base / "minerals_fit"
    minerals_dir.mkdir(parents=True)

    for i in range(3):
        pd.DataFrame(columns=["center_cm1", "fwhm_cm1", "amplitude_a", "area", "snr", "pass_snr", "pass_fwhm", "pass_r2"]).to_csv(
            minerals_dir / f"0851_Lake_Haiyaha_detail_1_R1_point{i}_fit_peaks.csv", index=False
        )

    # Create AICc summary at results_base level (even though no peaks)
    pd.DataFrame([
        {"point": 0, "r2": 0.95},
        {"point": 1, "r2": 0.88},
        {"point": 2, "r2": 0.72},
    ]).to_csv(results_base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False)

    service = _make_service(engine)
    result = _call_persist(service, results_base)

    assert result.metadata["peaks_inserted"] == 0
    assert "0 minerals peaks" in result.summary

    # Verify no peaks in DB
    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).all()
        assert len(peaks) == 0


def test_persist_r2_mapped(populated_db, mock_csvs):
    """Verify fit_quality populated from AICc summary CSV r2 column, mapped per-point."""
    engine, spectrum_ids = populated_db
    service = _make_service(engine)

    result = _call_persist(service, mock_csvs)
    assert result.metadata["peaks_inserted"] == 3

    # Query peaks and verify R² mapping
    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).join(
            SpectrumORM, SpectrumORM.id == FittedPeakORM.spectrum_id
        ).join(
            ScanPointORM, ScanPointORM.id == SpectrumORM.scan_point_id
        ).all()

        # Point 0 should have 2 peaks, both with R² = 0.95
        point0_peaks = [p for p in peaks if p.spectrum_id == spectrum_ids[0]]
        assert len(point0_peaks) == 2
        for peak in point0_peaks:
            assert peak.fit_quality == 0.95

        # Point 1 should have 1 peak with R² = 0.88
        point1_peaks = [p for p in peaks if p.spectrum_id == spectrum_ids[1]]
        assert len(point1_peaks) == 1
        assert point1_peaks[0].fit_quality == 0.88

        # Point 2 has no peaks (empty CSV)
        point2_peaks = [p for p in peaks if p.spectrum_id == spectrum_ids[2]]
        assert len(point2_peaks) == 0


def test_persist_no_aicc_summary_warns(populated_db, tmp_path):
    """Verify warning when AICc summary CSV is missing.

    fit_quality should be None for all peaks.
    """
    engine, spectrum_ids = populated_db

    # Create CSVs without AICc summary
    results_base = tmp_path / "results_no_aicc"
    minerals_dir = results_base / "minerals_fit"
    minerals_dir.mkdir(parents=True)

    # Only create peak CSV for point 0
    pd.DataFrame([
        {"center_cm1": 1085.5, "fwhm_cm1": 30.0, "amplitude_a": 1500.0, "area": 100.0, "snr": 15.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
    ]).to_csv(minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

    # No AICc summary file

    service = _make_service(engine)
    result = _call_persist(service, results_base)

    # Should insert peak successfully
    assert result.metadata["peaks_inserted"] == 1

    # Verify fit_quality is None
    with get_session(engine) as session:
        peak = session.query(FittedPeakORM).first()
        assert peak.fit_quality is None


def test_persist_minerals_fit_dir_missing_raises(populated_db, tmp_path):
    """Verify FittingError when minerals_fit directory doesn't exist."""
    engine, spectrum_ids = populated_db

    # Create results dir but no minerals_fit subdir
    results_base = tmp_path / "results_no_minerals"
    results_base.mkdir(parents=True)

    service = _make_service(engine)

    with pytest.raises(FittingError) as exc_info:
        _call_persist(service, results_base)

    assert "Minerals fit directory not found" in str(exc_info.value)
    assert "minerals_fit" in str(exc_info.value.context["fit_dir"])


def test_persist_no_csvs_raises(populated_db, tmp_path):
    """Verify FittingError when no peak CSVs found in minerals_fit directory."""
    engine, spectrum_ids = populated_db

    # Create minerals_fit dir but no CSV files
    results_base = tmp_path / "results_no_csvs"
    minerals_dir = results_base / "minerals_fit"
    minerals_dir.mkdir(parents=True)

    service = _make_service(engine)

    with pytest.raises(FittingError) as exc_info:
        _call_persist(service, results_base)

    assert "No fitted peak CSVs found" in str(exc_info.value)


def test_persist_filters_by_min_snr_and_fwhm(populated_db, tmp_path):
    """Verify peaks are filtered by SNR >= min_snr (3.0 default) AND FWHM >= 25.0."""
    engine, spectrum_ids = populated_db

    results_base = tmp_path / "results_filter"
    minerals_dir = results_base / "minerals_fit"
    minerals_dir.mkdir(parents=True)

    # Create CSV with peaks at various SNR and FWHM thresholds
    pd.DataFrame([
        # Reviewable: SNR=3.0, FWHM=25.0 (at threshold)
        {"center_cm1": 1000.0, "fwhm_cm1": 25.0, "amplitude_a": 500.0, "area": 50.0, "snr": 3.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
        # Reviewable: SNR=5.0, FWHM=30.0 (above threshold)
        {"center_cm1": 1100.0, "fwhm_cm1": 30.0, "amplitude_a": 600.0, "area": 60.0, "snr": 5.0, "pass_snr": True, "pass_fwhm": True, "pass_r2": True},
        # Non-reviewable: SNR=2.9 (just below threshold), FWHM=30.0
        {"center_cm1": 1200.0, "fwhm_cm1": 30.0, "amplitude_a": 400.0, "area": 40.0, "snr": 2.9, "pass_snr": False, "pass_fwhm": True, "pass_r2": True},
        # Non-reviewable: SNR=5.0, FWHM=24.9 (just below threshold)
        {"center_cm1": 1300.0, "fwhm_cm1": 24.9, "amplitude_a": 700.0, "area": 70.0, "snr": 5.0, "pass_snr": True, "pass_fwhm": False, "pass_r2": True},
        # Non-reviewable: SNR=2.0, FWHM=20.0 (both below threshold)
        {"center_cm1": 1400.0, "fwhm_cm1": 20.0, "amplitude_a": 300.0, "area": 30.0, "snr": 2.0, "pass_snr": False, "pass_fwhm": False, "pass_r2": True},
    ]).to_csv(minerals_dir / "0851_Lake_Haiyaha_detail_1_R1_point0_fit_peaks.csv", index=False)

    # Add AICc summary at results_base level
    pd.DataFrame([{"point": 0, "r2": 0.90}]).to_csv(
        results_base / "0851_Lake_Haiyaha_detail_1_R1_fit_aicc_summary.csv", index=False
    )

    service = _make_service(engine)
    result = _call_persist(service, results_base)

    # Should only insert 2 reviewable peaks (1000.0 and 1100.0 cm⁻¹)
    assert result.metadata["peaks_inserted"] == 2

    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).all()
        assert len(peaks) == 2
        centers = sorted([p.center_cm1 for p in peaks])
        assert centers == [1000.0, 1100.0]


def test_persist_spectrum_type_fallback(engine, mock_csvs):
    """Verify fallback to 'active' spectrum_type when 'laser_normalized' not found."""
    spectrum_ids = {}

    # Create DB with 'active' spectrum_type instead of 'laser_normalized'
    with get_session(engine) as session:
        sol = SolORM(sol_number=851, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=851,
            scan_name="detail_1",
            target="Lake_Haiyaha",
            scan_id="test_scan_id",
            sclk_start=0,
            n_points=3,
            n_channels=2148
        )
        session.add(scan)
        session.flush()

        for i in range(3):
            pt_id = str(uuid.uuid4())
            pt = ScanPointORM(id=pt_id, scan_id=scan_id, point_index=i)
            session.add(pt)
            session.flush()

            # Use 'active' instead of 'laser_normalized'
            spec_id = str(uuid.uuid4())
            spec = SpectrumORM(
                id=spec_id,
                scan_point_id=pt_id,
                region="R1",
                spectrum_type="active",  # Fallback type
                processing_level="normalized",
                intensities=b"\x00"
            )
            session.add(spec)
            session.flush()
            spectrum_ids[i] = spec_id

    service = _make_service(engine)
    result = _call_persist(service, mock_csvs)

    # Should successfully insert peaks using 'active' spectra
    assert result.metadata["peaks_inserted"] == 3

    # Verify peaks are linked to 'active' spectra
    with get_session(engine) as session:
        peaks = session.query(FittedPeakORM).all()
        for peak in peaks:
            assert peak.spectrum_id in spectrum_ids.values()
