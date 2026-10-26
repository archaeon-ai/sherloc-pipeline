"""ML-despiked fluorescence inputs: building, choosing, freshness, normalization, counts.

Uses the committed sol_0921 fixture workspace ingested into a temporary DB, with
synthetic stored masks. Persistence regressions use entirely synthesized inputs.
"""

import json
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock, patch

import numpy as np
import pytest
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import create_all_tables, get_engine, get_session
from sherloc_pipeline.database.models import (
    CosmicRayMaskORM,
    FittedPeakORM,
    ScanORM,
    ScanPointORM,
    SolORM,
    SpectrumORM,
)
from sherloc_pipeline.services import fluor_inputs
from sherloc_pipeline.services.cr_masks import CRMaskService
from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.ingestion import IngestionService
from sherloc_pipeline.services.pds_ingestion import PDSIngestionService

WORKSPACE_REL = (
    "loupe/sol_0921/detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
)
TARGET = "Fluor_Inputs"
PROVENANCE = {
    "method": "ml_v1.3_tau_matched",
    "model_sha256": "ab" * 32,
    "tau": {"R1": 0.3, "R2": 0.27, "R3": 0.27},
    "ort_version": "1.26.0",
}
#: Masks for points 0-2 only; point 1 R1 is an empty (checked, nothing flagged) mask.
MASKS = {
    "0": {"R1": [60, 100], "R2": [600, 1200], "R3": [1700]},
    "1": {"R1": [], "R2": [800], "R3": []},
    "2": {"R1": [52], "R2": [], "R3": [2139]},
}


def _metadata():
    n_flagged = {}
    for regions in MASKS.values():
        for region, channels in regions.items():
            n_flagged[region] = n_flagged.get(region, 0) + len(channels)
    return {**PROVENANCE, "n_flagged": n_flagged, "masks": MASKS}


class _Config:
    preprocessing = {"despike": {"interpolation_method": "linear"}}
    fluorescence_fitting = {"laser_normalize": True}


@pytest.fixture
def db(tmp_path, fixtures_path):
    db_path = tmp_path / "phase.db"
    service = IngestionService(
        database_path=db_path, include_spectra=True, ingestion_mode="all_regions"
    )
    service.ingest_workspace(fixtures_path / WORKSPACE_REL)
    with get_session(service.engine) as session:
        session.query(ScanORM).first().target = TARGET.replace("_", " ")
    CRMaskService().persist_masks(
        sol="0921", target=TARGET, scan="detail_1",
        despike_metadata=_metadata(), database_path=db_path,
    )
    return db_path, service


def _points(session):
    rows = session.query(ScanPointORM).order_by(ScanPointORM.point_index).all()
    return {p.point_index: p.id for p in rows}


def _decode(blob):
    return np.frombuffer(zlib.decompress(blob), dtype=np.float32)


def _rebuild(db_path):
    return fluor_inputs.refresh_despiked_rows_for_scan(
        db_path, "0921", TARGET, "detail_1", _Config(), rebuild=True
    )


def test_rebuild_writes_one_row_per_masked_region(db):
    db_path, service = db
    result = _rebuild(db_path)

    assert result["despiked_rows_written"] == 9
    with get_session(service.engine) as session:
        points = _points(session)
        rows = session.query(SpectrumORM).filter(
            SpectrumORM.spectrum_type == fluor_inputs.DESPIKED_SPECTRUM_TYPE
        ).all()
        assert {(r.scan_point_id, r.region) for r in rows} == {
            (points[i], reg) for i in range(3) for reg in ("R1", "R2", "R3")
        }
        # An empty mask leaves the spectrum as it was.
        r1 = {r.spectrum_type: r for r in session.query(SpectrumORM).filter(
            SpectrumORM.scan_point_id == points[1], SpectrumORM.region == "R1")}
        np.testing.assert_array_equal(
            _decode(r1[fluor_inputs.DESPIKED_SPECTRUM_TYPE].intensities),
            _decode(r1["dark_subtracted"].intensities),
        )
        # No masks or peaks attach to the derived rows.
        ids = [r.id for r in rows]
        assert session.query(CosmicRayMaskORM).filter(
            CosmicRayMaskORM.spectrum_id.in_(ids)).count() == 0


def test_points_with_current_rows_use_them_and_the_rest_fall_back(db):
    db_path, service = db
    _rebuild(db_path)
    with get_session(service.engine) as session:
        points = _points(session)
        chosen = fluor_inputs.load_fluorescence_inputs(session, list(points.values()))
        assert [chosen[points[i]]["despiked"] for i in range(4)] == [True, True, True, False]
        r2 = session.query(SpectrumORM).filter(
            SpectrumORM.scan_point_id == points[0], SpectrumORM.region == "R2",
            SpectrumORM.spectrum_type == "dark_subtracted").one()
        assert chosen[points[0]]["link_spectrum_id"] == r2.id


def test_a_mask_newer_than_the_rows_makes_them_stale(db):
    db_path, service = db
    _rebuild(db_path)
    with get_session(service.engine) as session:
        points = _points(session)
        parent = session.query(SpectrumORM).filter(
            SpectrumORM.scan_point_id == points[0], SpectrumORM.region == "R3",
            SpectrumORM.spectrum_type == "dark_subtracted").one()
        mask = session.query(CosmicRayMaskORM).filter(
            CosmicRayMaskORM.spectrum_id == parent.id).one()
        mask.created_at = datetime.now(timezone.utc) + timedelta(hours=1)
    with get_session(service.engine) as session:
        chosen = fluor_inputs.load_fluorescence_inputs(session, [points[0], points[1]])
        assert chosen[points[0]]["despiked"] is False
        assert chosen[points[1]]["despiked"] is True


def test_a_refresh_without_rebuild_removes_the_rows(db):
    db_path, service = db
    _rebuild(db_path)
    result = fluor_inputs.refresh_despiked_rows_for_scan(
        db_path, "0921", TARGET, "detail_1", _Config(), rebuild=False
    )
    assert result["despiked_rows_removed"] == 9
    with get_session(service.engine) as session:
        assert session.query(SpectrumORM).filter(
            SpectrumORM.spectrum_type == fluor_inputs.DESPIKED_SPECTRUM_TYPE).count() == 0


def test_an_unknown_scan_is_reported_not_raised(db):
    db_path, _service = db
    result = fluor_inputs.refresh_despiked_rows_for_scan(
        db_path, "0921", TARGET, "detail_9", _Config(), rebuild=True
    )
    assert result == {"scan_found": False}


def test_counts_exclude_the_derived_rows(db):
    db_path, service = db
    pds = PDSIngestionService(pds_db_path=db_path)
    before = (service.get_database_stats()["spectra"], pds.get_database_stats()["spectra"],
              pds.validate_database()["spectra_by_region"])
    _rebuild(db_path)
    after = (service.get_database_stats()["spectra"], pds.get_database_stats()["spectra"],
             pds.validate_database()["spectra_by_region"])
    assert after == before


def test_laser_normalization_stage():
    factors = fluor_inputs.laser_normalization_factors({"a": 50.0, "b": 100.0, "c": None, "d": 0.0})
    assert factors == {"a": 2.0, "b": 1.0, "c": None, "d": None}

    arrays = {"R1": np.ones(3), "R2": np.full(3, 2.0), "R3": np.zeros(3)}
    scaled, applied = fluor_inputs.apply_fluorescence_laser_normalization(arrays, 2.0)
    assert applied is True
    np.testing.assert_array_equal(scaled["R2"], np.full(3, 4.0))
    same, applied = fluor_inputs.apply_fluorescence_laser_normalization(arrays, None)
    assert applied is False and same is arrays


def test_laser_normalize_reads_the_packaged_default_when_absent():
    assert fluor_inputs.laser_normalize_enabled({"fluorescence_fitting": {}}) is False
    assert fluor_inputs.laser_normalize_enabled(
        {"fluorescence_fitting": {"laser_normalize": True}}) is True


@pytest.mark.parametrize("laser_normalize", [True, False])
def test_fit_fluorescence_records_its_input(db, tmp_path, monkeypatch, laser_normalize):
    import copy

    from sherloc_pipeline.config import load_config
    from sherloc_pipeline.core import fluor_fitting
    from sherloc_pipeline.services.fitting import FittingService
    from sherloc_pipeline.services.runtime import RuntimeContext

    db_path, service = db
    _rebuild(db_path)

    def stub(wavelength, intensity, **kwargs):
        return fluor_fitting.FluorFitResult(
            peaks=[], r2=0.0, rss=0.0, aicc=float("inf"), n_peaks=0,
            is_saturated=False, fit_skipped=True, n_saturated_channels=0,
            n_masked_channels=0, warnings=["stub"],
        )

    monkeypatch.setattr(fluor_fitting, "fit_fluorescence_spectrum", stub)
    cfg = copy.deepcopy(load_config())
    cfg.fluorescence_fitting = dict(cfg.fluorescence_fitting or {})
    cfg.fluorescence_fitting.update(parallel_workers=1, laser_normalize=laser_normalize)
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    fitting = FittingService(context=context)
    fitting._database_path = db_path

    result = fitting.fit_fluorescence("0921", TARGET, "detail_1", generate_plots=False)

    meta = result.metadata["input"]
    with get_session(service.engine) as session:
        n_points = session.query(ScanPointORM).count()
        with_pd = session.query(ScanPointORM).filter(ScanPointORM.photodiode_mean > 0).count()
    assert meta["points_despiked"] == 3
    assert meta["points_fallback"] == n_points - 3
    assert meta["input_spectrum_type"] == "mixed"
    assert meta["laser_normalize"] is laser_normalize
    assert meta["points_laser_normalized"] == (with_pd if laser_normalize else 0)
    assert any("non-despiked input" in w for w in result.warnings)


def test_a_row_no_newer_than_its_mask_is_stale(db):
    db_path, service = db
    _rebuild(db_path)
    with get_session(service.engine) as session:
        points = _points(session)
        parent = session.query(SpectrumORM).filter(
            SpectrumORM.scan_point_id == points[0], SpectrumORM.region == "R2",
            SpectrumORM.spectrum_type == "dark_subtracted").one()
        row = session.query(SpectrumORM).filter(
            SpectrumORM.scan_point_id == points[0], SpectrumORM.region == "R2",
            SpectrumORM.spectrum_type == fluor_inputs.DESPIKED_SPECTRUM_TYPE).one()
        mask = session.query(CosmicRayMaskORM).filter(
            CosmicRayMaskORM.spectrum_id == parent.id).one()
        mask.created_at = row.created_at  # equal, not older
    with get_session(service.engine) as session:
        chosen = fluor_inputs.load_fluorescence_inputs(session, [points[0]])
        assert chosen[points[0]]["despiked"] is False


def test_the_pipeline_rebuild_uses_only_this_runs_masks(db):
    """A stored mask from an earlier run never builds a row the run did not produce."""
    db_path, service = db
    result = fluor_inputs.refresh_despiked_rows_for_scan(
        db_path, "0921", TARGET, "detail_1", _Config(), rebuild=True,
        run_masks={"0": {"R1": [60], "R2": [600], "R3": [1700]}},  # point 0 only
    )
    assert result["despiked_rows_written"] == 3
    with get_session(service.engine) as session:
        points = _points(session)
        chosen = fluor_inputs.load_fluorescence_inputs(session, [points[0], points[1]])
        # Points 1 and 2 have stored masks from the fixture's earlier persist,
        # but this run did not produce them: no rows, so they fall back.
        assert chosen[points[0]]["despiked"] is True
        assert chosen[points[1]]["despiked"] is False


def test_allow_despiked_false_forces_the_measured_rows(db):
    db_path, service = db
    _rebuild(db_path)
    with get_session(service.engine) as session:
        points = _points(session)
        chosen = fluor_inputs.load_fluorescence_inputs(
            session, [points[0]], allow_despiked=False)
        assert chosen[points[0]]["despiked"] is False


def test_map_mode_normalizes_a_subset_with_the_scan_maximum(db):
    from sherloc_pipeline.services import map_fitting

    db_path, service = db
    with get_session(service.engine) as session:
        scan_id = session.query(ScanORM).first().id
        all_points = session.query(ScanPointORM).all()
        pds = {sp.id: sp.photodiode_mean for sp in all_points}
        subset = [sp for sp in all_points if sp.point_index in (0, 1)]
        whole = fluor_inputs.laser_normalization_factors(pds)
        raw = fluor_inputs.load_fluorescence_inputs(session, [sp.id for sp in subset])
        parts, _notes, scales = map_fitting._load_fluorescence_inputs(
            session, subset, {"fluorescence_fitting": {"laser_normalize": True}}, scan_id)
        for sp in subset:
            np.testing.assert_allclose(
                parts[sp.id]["R2"], raw[sp.id]["arrays"]["R2"] * whole[sp.id])
            assert scales[sp.id] == pytest.approx(whole[sp.id])


def test_scaled_saturation_threshold():
    assert fluor_inputs.scaled_saturation_threshold(60000.0, None) == 60000.0
    assert fluor_inputs.scaled_saturation_threshold(60000.0, 2.0) == 120000.0


@pytest.mark.parametrize("laser_normalize", [True, False])
def test_point_fit_scales_the_saturation_threshold_with_the_counts(
    db, tmp_path, monkeypatch, laser_normalize
):
    """A band at 35,000 counts scaled by 2 is 70,000: not saturated (P1, #133)."""
    import copy

    from sherloc_pipeline.config import load_config
    from sherloc_pipeline.core import fluor_fitting
    from sherloc_pipeline.services.fitting import FittingService
    from sherloc_pipeline.services.runtime import RuntimeContext

    db_path, service = db
    seen = []

    def stub(wavelength, intensity, **kwargs):
        seen.append(kwargs["saturation_threshold"])
        return fluor_fitting.FluorFitResult(
            peaks=[], r2=0.0, rss=0.0, aicc=float("inf"), n_peaks=0,
            is_saturated=False, fit_skipped=True, n_saturated_channels=0,
            n_masked_channels=0, warnings=["stub"],
        )

    monkeypatch.setattr(fluor_fitting, "fit_fluorescence_spectrum", stub)
    cfg = copy.deepcopy(load_config())
    cfg.fluorescence_fitting = dict(cfg.fluorescence_fitting or {})
    cfg.fluorescence_fitting.update(
        parallel_workers=1, laser_normalize=laser_normalize, saturation_threshold=60000.0)
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    fitting = FittingService(context=context)
    fitting._database_path = db_path
    fitting.fit_fluorescence("0921", TARGET, "detail_1", generate_plots=False)

    with get_session(service.engine) as session:
        pds = dict(session.query(ScanPointORM.id, ScanPointORM.photodiode_mean))
    factors = fluor_inputs.laser_normalization_factors(pds)
    expected = sorted(
        60000.0 * f if (laser_normalize and f is not None) else 60000.0
        for f in factors.values()
    )
    assert sorted(seen) == pytest.approx(expected)
    if laser_normalize:
        assert max(seen) > 60000.0


def test_map_fit_scales_the_saturation_threshold(monkeypatch):
    from sherloc_pipeline.core import fluor_fitting
    from sherloc_pipeline.services import map_fitting

    seen = []

    def stub(**kwargs):
        seen.append(kwargs["saturation_threshold"])
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(fluor_fitting, "fit_fluorescence_spectrum", stub)
    parts = {reg: np.ones(2148, dtype=np.float64) for reg in ("R1", "R2", "R3")}
    cfg = {"fluorescence_fitting": {"saturation_threshold": 60000.0}}
    for scale in (1.0, 2.5):
        map_fitting._fit_fluorescence_domain(
            None, "sp", cfg, preloaded_parts=parts, saturation_scale=scale)
    assert seen == [60000.0, 150000.0]


@pytest.mark.parametrize("allow_despiked", [True, False])
def test_a_database_without_the_mask_table_uses_the_measured_rows(db, allow_despiked):
    """Pre-migration DBs have no cosmic_ray_masks table (P2, #133)."""
    from sqlalchemy import text

    db_path, service = db
    _rebuild(db_path)
    with get_session(service.engine) as session:
        session.execute(text("DROP TABLE cosmic_ray_masks"))
    with get_session(service.engine) as session:
        points = _points(session)
        chosen = fluor_inputs.load_fluorescence_inputs(
            session, list(points.values()), allow_despiked=allow_despiked)
        assert len(chosen) == len(points)
        assert not any(c["despiked"] for c in chosen.values())


def test_only_the_rows_this_run_wrote_are_used_when_ids_are_given(db):
    db_path, service = db
    first = _rebuild(db_path)
    with get_session(service.engine) as session:
        points = _points(session)
        mine = fluor_inputs.load_fluorescence_inputs(
            session, [points[0]], allowed_despiked_ids=first["row_ids"])
        assert mine[points[0]]["despiked"] is True
        # Another run's ids (here: none of this point's rows) are not accepted.
        other = fluor_inputs.load_fluorescence_inputs(
            session, [points[0]], allowed_despiked_ids=["not-a-row"])
        assert other[points[0]]["despiked"] is False


def _snapshot_fluor_peaks(database_path):
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
def synthetic_fluor_run(tmp_path, monkeypatch):
    """Entirely invented R1-R3 spectra and prior fluorescence peaks."""
    import copy

    from sherloc_pipeline.config import load_config
    from sherloc_pipeline.core import fluor_fitting
    from sherloc_pipeline.services.fitting import FittingService
    from sherloc_pipeline.services.runtime import RuntimeContext

    assert not any(tmp_path.resolve().is_relative_to(root) for root in (
        Path("/data"), Path("/nas"),
    ))
    database_path = tmp_path / "synthetic.db"
    engine = get_engine(database_path)
    create_all_tables(engine)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=76543, data_source="loupe"))
        session.flush()
        scan_id = str(uuid4())
        session.add(ScanORM(
            id=scan_id, sol_number=76543, target="Synthetic Fluor",
            target_type="mars_target", scan_name="synthetic_scan", scan_id="synthetic_fluor",
            sclk_start=0, n_points=3, n_channels=2148,
        ))
        session.flush()
        for point in range(3):
            point_id = str(uuid4())
            session.add(ScanPointORM(id=point_id, scan_id=scan_id, point_index=point))
            session.flush()
            for region in fluor_inputs.REGIONS:
                spectrum_id = str(uuid4())
                session.add(SpectrumORM(
                    id=spectrum_id, scan_point_id=point_id, region=region,
                    spectrum_type="dark_subtracted", processing_level="raw",
                    intensities=fluor_inputs._encode(np.full(2148, 23.0 + point)),
                ))
                session.flush()
                if region == "R2":
                    for center in (301.0, 323.0):
                        session.add(FittedPeakORM(
                            id=str(uuid4()), spectrum_id=spectrum_id, fit_modality="fluorescence",
                            center_nm=center, amplitude=41.0, fwhm_nm=17.0, snr=13.0,
                        ))

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
    yield SimpleNamespace(
        engine=engine, database_path=database_path, fitting=fitting,
        previous=_snapshot_fluor_peaks(database_path),
    )
    if fitting._engine is not None:
        fitting._engine.dispose()
    engine.dispose()


def _fit_synthetic_fluor(run):
    return run.fitting.fit_fluorescence(
        "76543", "Synthetic_Fluor", "synthetic_scan", generate_plots=False,
    )


@pytest.mark.parametrize("failure", ["zlib", "value", "private_message"])
def test_unreadable_point_preserved_valid_fluor_points_committed(
    synthetic_fluor_run, failure, monkeypatch, caplog,
):
    run = synthetic_fluor_run
    blob = zlib.compress(b"x") if failure == "value" else b"synthetic-corrupt-intensity"
    with get_session(run.engine) as session:
        spectrum = session.get(SpectrumORM, run.previous[1][0]["spectrum_id"])
        spectrum.intensities = blob
    if failure == "private_message":
        decode = fluor_inputs._decode

        def decode_one_blob(value):
            if value == blob:
                raise ValueError("synthetic-private-contents")
            return decode(value)

        monkeypatch.setattr(fluor_inputs, "_decode", decode_one_blob)

    with pytest.raises(FittingError) as exc_info:
        _fit_synthetic_fluor(run)

    error = exc_info.value
    assert error.exit_code == 1
    assert error.context["peaks_inserted"] == 2
    exception_type = "error" if failure == "zlib" else "ValueError"
    assert error.context["unreadable_spectra"] == [{
        "point_index": 1, "exception_type": exception_type,
    }]
    assert "point 1" in str(error)
    assert "previously persisted fluorescence peaks were kept" in str(error)
    assert "synthetic-private-contents" not in str(error.context) + str(error) + caplog.text
    rows = _snapshot_fluor_peaks(run.database_path)
    assert rows[1] == run.previous[1]
    for point in (0, 2):
        assert len(rows[point]) == 1
        assert rows[point][0]["id"] not in {row["id"] for row in run.previous[point]}
        assert rows[point][0]["center_nm"] == 314.0
        assert rows[point][0]["amplitude"] == 73.0


def test_all_readable_fluor_points_replace_prior_peaks(synthetic_fluor_run):
    run = synthetic_fluor_run
    result = _fit_synthetic_fluor(run)
    assert result.metadata["peaks_inserted"] == 3
    rows = _snapshot_fluor_peaks(run.database_path)
    for point in range(3):
        assert len(rows[point]) == 1
        assert rows[point][0]["id"] not in {row["id"] for row in run.previous[point]}
        assert rows[point][0]["center_nm"] == 314.0


def test_never_measured_fluor_point_stays_in_delete_scope(synthetic_fluor_run, monkeypatch):
    run = synthetic_fluor_run
    missing_id = str(uuid4())
    with get_session(run.engine) as session:
        scan = session.query(ScanORM).one()
        scan.n_points = 4
        session.add(ScanPointORM(id=missing_id, scan_id=scan.id, point_index=3))
    writer = Mock(wraps=run.fitting._write_peaks_to_db)
    monkeypatch.setattr(run.fitting, "_write_peaks_to_db", writer)
    result = _fit_synthetic_fluor(run)
    assert result.metadata["peaks_inserted"] == 3
    assert result.metadata["points_skipped"] == 1
    assert missing_id in writer.call_args.args[2]
    assert len(writer.call_args.args[2]) == 4
    assert len(writer.call_args.args[1]) == 3


def test_all_unreadable_fluor_points_keep_prior_peaks(synthetic_fluor_run):
    run = synthetic_fluor_run
    with get_session(run.engine) as session:
        for spectrum in session.query(SpectrumORM).filter_by(region="R2"):
            spectrum.intensities = b"synthetic-corrupt-intensity"
    with pytest.raises(FittingError) as exc_info:
        _fit_synthetic_fluor(run)
    assert exc_info.value.context["peaks_inserted"] == 0
    assert {item["point_index"] for item in exc_info.value.context["unreadable_spectra"]} == {0, 1, 2}
    assert _snapshot_fluor_peaks(run.database_path) == run.previous


@pytest.mark.parametrize("measured_corrupt", [True, False])
def test_corrupt_despiked_fluor_reports_only_unreadable_fallbacks(
    synthetic_fluor_run, measured_corrupt,
):
    run = synthetic_fluor_run
    now = datetime.now(timezone.utc)
    with get_session(run.engine) as session:
        point = session.query(ScanPointORM).filter_by(point_index=1).one()
        for parent in session.query(SpectrumORM).filter_by(scan_point_id=point.id).all():
            parent.created_at = now - timedelta(hours=2)
            session.add(CosmicRayMaskORM(
                id=str(uuid4()), spectrum_id=parent.id, method="synthetic",
                model_sha256="cd" * 32, tau=0.4, channel_indices=[], n_flagged=0,
                created_at=now - timedelta(hours=1),
            ))
            session.add(SpectrumORM(
                id=str(uuid4()), scan_point_id=point.id, region=parent.region,
                spectrum_type=fluor_inputs.DESPIKED_SPECTRUM_TYPE, processing_level="despiked",
                intensities=b"bad-despiked" if parent.region == "R2" else parent.intensities,
                created_at=now,
            ))
            if measured_corrupt and parent.region == "R2":
                parent.intensities = b"bad-measured"
        point_id = point.id
    unreadable = {}
    with get_session(run.engine) as session:
        chosen = fluor_inputs.load_fluorescence_inputs(
            session, [point_id], unreadable_spectra=unreadable,
        )
    if measured_corrupt:
        assert chosen == {}
        assert unreadable == {point_id: "error"}
    else:
        assert chosen[point_id]["despiked"] is False
        assert unreadable == {}


def test_single_scan_fit_fluor_exits_one_for_unreadable_spectrum(synthetic_fluor_run):
    from sherloc_pipeline.cli.app import app

    run = synthetic_fluor_run
    with get_session(run.engine) as session:
        session.get(SpectrumORM, run.previous[1][0]["spectrum_id"]).intensities = b"bad"

    def factory(**kwargs):
        run.fitting.console = kwargs["console"]
        return run.fitting

    with patch("sherloc_pipeline.services.fitting.FittingService", side_effect=factory):
        result = CliRunner().invoke(app, [
            "--json", "fit-fluor", "--sol", "76543", "--target", "Synthetic_Fluor",
            "--scan", "synthetic_scan", "--no-plots", "--database", str(run.database_path),
        ])
    assert result.exit_code == 1, result.output
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["error_type"] == "FittingError"
    assert "previously persisted fluorescence peaks were kept" in error["message"]
    assert _snapshot_fluor_peaks(run.database_path)[1] == run.previous[1]


def test_process_new_exits_one_for_unreadable_fluor_spectrum(
    synthetic_fluor_run, tmp_path, monkeypatch,
):
    from sherloc_pipeline.cli.app import app
    from sherloc_pipeline.services.base import ServiceResult

    run = synthetic_fluor_run
    with get_session(run.engine) as session:
        session.get(SpectrumORM, run.previous[1][0]["spectrum_id"]).intensities = b"bad"
    sol_dir = tmp_path / "sol_76543"
    sol_dir.mkdir()
    monkeypatch.setenv("SHERLOC_DB_PATH", str(run.database_path))
    fitted = ServiceResult(summary="Synthetic fitting completed")
    for method in ("fit_minerals", "fit_organics", "fit_hydration"):
        monkeypatch.setattr(run.fitting, method, Mock(return_value=fitted))

    def factory(**kwargs):
        run.fitting.console = kwargs["console"]
        return run.fitting

    with patch("sherloc_pipeline.services.pipeline.FittingService", side_effect=factory), \
         patch("sherloc_pipeline.services.pipeline.PreprocessingService") as preprocessing, \
         patch("sherloc_pipeline.services.pipeline.DataIngestion") as ingestion, \
         patch("sherloc_pipeline.services.pipeline.archive_existing_results", return_value=None), \
         patch("sherloc_pipeline.services.fluor_inputs.refresh_despiked_rows_for_scan", return_value={}):
        preprocessing.return_value.run_scan.return_value = fitted
        ingestion.return_value.get_results_path.return_value = tmp_path / "results"
        result = CliRunner().invoke(app, [
            "--json", "process-new", str(sol_dir), "--skip-ingest",
            "--database", str(run.database_path), "--data-dir", str(tmp_path),
            "--results-dir", str(tmp_path / "results"),
        ])
    assert result.exit_code == 1, result.output
    errors = json.loads(result.stdout)["result"]["errors"]
    assert len(errors) == 1
    assert "point 1" in errors[0]
    assert "previously persisted fluorescence peaks were kept" in errors[0]
    assert _snapshot_fluor_peaks(run.database_path)[1] == run.previous[1]
