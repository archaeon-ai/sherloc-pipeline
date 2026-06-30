"""DB-level regression tests for fittable scan selection.

Covers the two production scan-selection surfaces that must agree with
is_fittable() (SCAN_CLASSIFICATION_SPEC §4.2.2 + K6):

1. ``process-new`` Step 3 — the candidate query (target_type ∈ {mars,cal})
   filtered through ``is_fittable`` (cli/app.py:process_new_cmd).
2. ``_iter_all_scans(science_only=True)`` — the SQL-level equivalent used by
   ``backfill --science`` / ``backfill-masks --science``.

Both must select exactly the meteorite science-scans and exclude engineered
standards AND engineering scans — including the load-bearing case of a
``power_on`` engineering scan on a *dedicated* meteorite sol whose sol-wide
target is "ext cal meteorite" (target contains "meteorite", but the
cal_target guard must drop it).
"""

import uuid

import pytest
from rich.console import Console
from sqlalchemy.orm import Session

from sherloc_pipeline.cli.app import _iter_all_scans
from sherloc_pipeline.database.connection import get_engine
from sherloc_pipeline.database.models import Base, ScanORM, SolORM
from sherloc_pipeline.models.spectra import is_fittable


def _make_scan(**overrides):
    kwargs = dict(
        id=str(uuid.uuid4()),
        scan_id=f"scan_{uuid.uuid4().hex[:8]}",
        sclk_start=100000,
        n_points=9,
        n_channels=2148,
        laser_wavelength_nm=248.6,
        scan_class="primary",
    )
    kwargs.update(overrides)
    return ScanORM(**kwargs)


# (sol, target, scan_name, target_type, expected_fittable)
SEED = [
    # exploration-area science → fittable
    (900, "Amherst Point", "detail_1", "mars_target", True),
    # mixed cal sol: meteorite named by material → fittable; standards → not
    (1521, "external calibration", "meteorite", "cal_target", True),
    (1521, "external calibration", "AlGaN_1", "cal_target", False),
    (1521, "external calibration", "maze", "cal_target", False),
    (1521, "external calibration", "power_on", "engineering", False),
    # dedicated meteorite sol: meteorite by target → fittable;
    # power_on engineering must be dropped despite the "meteorite" target (F1)
    (1256, "ext cal meteorite", "detail_1", "cal_target", True),
    (1256, "ext cal meteorite", "power_on", "engineering", False),
]


@pytest.fixture
def db_path(tmp_path):
    """A file-backed phase.db seeded with the SEED corpus."""
    path = tmp_path / "phase.db"
    engine = get_engine(str(path))
    Base.metadata.create_all(engine)
    with Session(engine) as sess:
        for sol in {row[0] for row in SEED}:
            sess.add(SolORM(sol_number=sol, data_source="loupe"))
        sess.commit()
        for sol, target, scan_name, target_type, _ in SEED:
            sess.add(_make_scan(
                sol_number=sol, target=target, scan_name=scan_name,
                target_type=target_type,
            ))
        sess.commit()
    return path


EXPECTED_FITTABLE = {(sol, scn) for sol, _, scn, _, fit in SEED if fit}


def test_iter_all_scans_science_only_selects_fittable(db_path):
    """science_only must select exactly the fittable set (real production fn)."""
    rows = _iter_all_scans(db_path, Console(quiet=True), science_only=True)
    selected = {(sol, scn) for sol, _tgt, scn in rows}
    assert selected == EXPECTED_FITTABLE


def test_science_only_excludes_engineering_on_meteorite_sol(db_path):
    """F1: power_on on a dedicated meteorite sol is dropped by the cal guard."""
    rows = _iter_all_scans(db_path, Console(quiet=True), science_only=True)
    selected = {(sol, scn) for sol, _tgt, scn in rows}
    assert (1256, "power_on") not in selected
    assert (1521, "power_on") not in selected


def test_process_new_step3_selection_matches_is_fittable(db_path):
    """Mirror process-new Step 3: candidate query (mars ∪ cal) + is_fittable."""
    engine = get_engine(str(db_path))
    with Session(engine) as session:
        candidates = (
            session.query(
                ScanORM.sol_number,
                ScanORM.target,
                ScanORM.scan_name,
                ScanORM.target_type,
            )
            .filter(ScanORM.target_type.in_(["mars_target", "cal_target"]))
            .all()
        )
    selected = {
        (sol, scn)
        for sol, tgt, scn, ttype in candidates
        if is_fittable(ttype, tgt, scn)
    }
    assert selected == EXPECTED_FITTABLE
    # engineering never even enters the candidate set
    assert all(ttype != "engineering" for _, _, _, ttype in candidates)
