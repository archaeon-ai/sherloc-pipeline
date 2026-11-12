"""record_scan_processing: which row it writes, and what it writes."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from sherloc_pipeline.database.connection import create_all_tables, get_engine, get_session
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.services.scan_processing import record_scan_processing


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "phase.db"
    engine = get_engine(path)
    create_all_tables(engine)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=921))
        session.flush()
        for target in ("Some Target", "Other Target"):
            session.add(ScanORM(
                id=str(uuid.uuid4()), sol_number=921, target=target, target_type="mars_target",
                scan_name="detail_1", scan_id=f"id-{target}", sclk_start=0, n_points=1, n_channels=2148,
            ))
    engine.dispose()
    return path


def _fields(path):
    engine = get_engine(path)
    try:
        with get_session(engine) as session:
            return {
                s.target: (s.processing_status, s.processed_at, s.processing_config_hash,
                           s.processing_pipeline_version, s.processing_error)
                for s in session.query(ScanORM)
            }
    finally:
        engine.dispose()


def _record(path, **overrides):
    args = dict(sol="0921", target="Some_Target", scan="detail_1", status="completed", error=None,
                config_hash="a" * 64, pipeline_version="5.7.6")
    args.update(overrides)
    return record_scan_processing(path, **args)


def test_writes_only_the_matching_scan_with_utc_time(db):
    when = datetime(2026, 11, 12, 9, 0, tzinfo=timezone(timedelta(hours=-8)))
    assert _record(db, processed_at=when)
    fields = _fields(db)
    assert fields["Some Target"] == ("completed", datetime(2026, 11, 12, 17, 0), "a" * 64, "5.7.6", None)
    assert fields["Other Target"] == (None, None, None, None, None)


def test_completed_clears_an_earlier_error(db):
    assert _record(db, status="failed", error="stage x: broke")
    assert _fields(db)["Some Target"][4] == "stage x: broke"
    assert _record(db)
    assert _fields(db)["Some Target"][0] == "completed"
    assert _fields(db)["Some Target"][4] is None


def test_failed_without_reason_still_has_an_error(db):
    assert _record(db, status="failed", error=None)
    assert _fields(db)["Some Target"][4] == "unknown error"


def test_no_matching_scan_returns_false(db):
    assert _record(db, target="Missing") is False
    assert all(v == (None,) * 5 for v in _fields(db).values())


def test_rejects_other_statuses(db):
    with pytest.raises(ValueError):
        _record(db, status="pending")
