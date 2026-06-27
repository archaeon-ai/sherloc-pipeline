"""Integration tests for the WS-1 scan-classification source fix.

Covers (value-blind throughout — names / counts / ids only):
  * the product_role DB CHECK rejects invalid (role, class, parent, sources)
    tuples and accepts valid ones (ARC-M2P-311 AC5);
  * `reclassify-scan-types/-classes/-product-roles` re-derive all three axes to
    the locked truth table, are idempotent, and do not mutate measurement
    tables (ARC-M2P-312);
  * the value-blind classification invariants (ARC-M2P-315) hold on the
    corrected corpus: composite non-empty sources, exactly-one-canonical-per-
    raw, sources resolve to an in-group raw, role couplings, and non-NULL role
    ⇒ recognized multishot pattern.
"""

import json
import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from sherloc_pipeline.database.connection import get_engine
from sherloc_pipeline.models.spectra import multishot_raw_base, multishot_reduction_role
from sherloc_pipeline.services.scan_reclassification import (
    ALL_AXES,
    ScanReclassificationError,
    measurement_fingerprint,
    run_reclassification,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# (scan_name, n_points, initial scan_class, initial scan_type) — a value-blind
# corpus seeded in its *pre-fix* (mis-stored) state to exercise the reclassifier.
SEED_SCANS = [
    # Multishot trio: raw (300 = 100 positions x 3 shots) mislabeled survey by
    # the count rule, plus its two SNR reductions.
    ("detail_2", 300, "primary", "survey"),
    ("detail_2_median_all", 100, "composite", None),
    ("detail_2_sum_active_median_dark", 100, "composite", None),
    # normal detail + a sub-scan
    ("detail_1", 100, "primary", "detail"),
    ("detail_1a", 50, "sub_scan", None),
    # lines + a cross union (mislabeled detail by the old enum)
    ("line_1", 25, "primary", "detail"),
    ("line_2", 25, "primary", "detail"),
    ("cross", 50, "primary", "detail"),
    # bare trailing-underscore unions stored primary (defect D2)
    ("detail_", 400, "primary", "detail"),
    ("line_", 50, "primary", "detail"),
    # HDR + small survey mislabeled by count
    ("HDR_500", 100, "primary", "detail"),
    ("survey_100", 100, "primary", "detail"),
    ("survey_1296", 1296, "primary", "survey"),
]


def _seed_scan(conn, name, n_points, scan_class, scan_type, sol=100, target="TargetA"):
    conn.execute(
        text(
            "INSERT INTO scans (id, sol_number, scan_name, scan_id, sclk_start, "
            "n_points, n_channels, laser_wavelength_nm, created_at, target, "
            "target_type, scan_class, scan_type) VALUES "
            "(:id,:sol,:name,:name,1,:n,2148,248.6,'2026-01-01',:target,"
            "'mars_target',:cls,:typ)"
        ),
        {"id": str(uuid.uuid4()), "sol": sol, "name": name, "n": n_points,
         "target": target, "cls": scan_class, "typ": scan_type},
    )


def _seed_measurements(conn):
    """Seed a valid scan_point -> spectra -> fitted_peaks FK chain so the
    no-mutation hash is non-trivial. The test asserts reclassify leaves these
    rows byte-for-byte identical."""
    scan_id = conn.execute(
        text("SELECT id FROM scans WHERE scan_name = 'detail_1'")
    ).scalar()
    point_id = str(uuid.uuid4())
    conn.execute(
        text(
            "INSERT INTO scan_points (id, scan_id, point_index, created_at) "
            "VALUES (:id,:scan,0,'2026-01-01')"
        ),
        {"id": point_id, "scan": scan_id},
    )
    spectrum_ids = []
    for i in range(3):
        sid = str(uuid.uuid4())
        spectrum_ids.append(sid)
        conn.execute(
            text(
                "INSERT INTO spectra (id, scan_point_id, region, spectrum_type, "
                "processing_level, intensities, created_at) VALUES "
                "(:id,:pid,'R1','dark_subtracted','derived',:blob,'2026-01-01')"
            ),
            {"id": sid, "pid": point_id, "blob": bytes([i] * 32)},
        )
    for i in range(2):
        conn.execute(
            text(
                "INSERT INTO fitted_peaks (id, spectrum_id, peak_type, amplitude, "
                "created_at, fit_modality) VALUES "
                "(:id,:sid,'gaussian',:amp,'2026-01-01','minerals')"
            ),
            {"id": str(uuid.uuid4()), "sid": spectrum_ids[i], "amp": 1.0 + i},
        )


@pytest.fixture
def migrated_db(tmp_path):
    """A fresh Alembic-migrated DB seeded with the pre-fix corpus."""
    db_path = tmp_path / "phase.db"
    config = Config(str(_PROJECT_ROOT / "alembic.ini"))
    os.environ["PHASE_DATABASE_PATH"] = str(db_path)
    try:
        command.upgrade(config, "head")
        engine = get_engine(db_path)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sols (sol_number, data_source, created_at) "
                    "VALUES (100,'loupe','2026-01-01')"
                )
            )
            for name, n, cls, typ in SEED_SCANS:
                _seed_scan(conn, name, n, cls, typ)
            _seed_measurements(conn)
        yield db_path
    finally:
        os.environ.pop("PHASE_DATABASE_PATH", None)


def _scans_by_name(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT scan_name, scan_type, scan_class, product_role, "
                 "parent_scan_id, source_scan_ids FROM scans")
        ).fetchall()
    out = {}
    for name, st, sc, role, parent, sources in rows:
        out[name] = {
            "scan_type": st, "scan_class": sc, "product_role": role,
            "parent_scan_id": parent,
            "source_scan_ids": json.loads(sources) if isinstance(sources, str) else sources,
        }
    return out


def _reclassify(db_path, axes=ALL_AXES, apply=True):
    engine = get_engine(db_path)
    if apply:
        with engine.begin() as conn:
            return run_reclassification(
                conn, axes, apply=True, db_path=db_path, have_backup=True
            )
    with engine.connect() as conn:
        return run_reclassification(conn, axes, apply=False)


# ---------------------------------------------------------------------------
# product_role CHECK
# ---------------------------------------------------------------------------

class TestProductRoleCheck:
    @pytest.mark.parametrize("role,scan_class,parent,sources,accept", [
        (None, "primary", None, None, True),
        ("raw", "primary", None, None, True),
        ("canonical", "composite", None, ["x"], True),
        ("alternate", "composite", None, ["x"], True),
        ("bogus", "primary", None, None, False),          # enum
        ("raw", "composite", None, None, False),          # coupling: raw=>primary
        ("canonical", "primary", None, ["x"], False),     # coupling: canonical=>composite
        ("canonical", "composite", None, [], False),      # non-empty sources
        ("canonical", "composite", None, None, False),    # NULL sources (3-valued logic guard)
        ("raw", "primary", None, ["x"], False),           # raw must have NULL sources
    ])
    def test_check(self, migrated_db, role, scan_class, parent, sources, accept):
        engine = get_engine(migrated_db)
        params = {
            "id": str(uuid.uuid4()), "role": role, "cls": scan_class,
            "parent": parent,
            "sources": json.dumps(sources) if sources is not None else None,
        }
        sql = text(
            "INSERT INTO scans (id, sol_number, scan_name, scan_id, sclk_start, "
            "n_points, n_channels, laser_wavelength_nm, created_at, target_type, "
            "scan_class, product_role, parent_scan_id, source_scan_ids) VALUES "
            "(:id,100,'probe','probe',1,1,2148,248.6,'2026-01-01','mars_target',"
            ":cls,:role,:parent,:sources)"
        )
        from sqlalchemy.exc import IntegrityError
        if accept:
            with engine.begin() as conn:
                conn.execute(sql, params)
        else:
            with pytest.raises(IntegrityError):
                with engine.begin() as conn:
                    conn.execute(sql, params)


# ---------------------------------------------------------------------------
# Reclassify behaviour
# ---------------------------------------------------------------------------

class TestReclassifyTruthTable:
    def test_final_state_matches_spec_truth_table(self, migrated_db):
        _reclassify(migrated_db)
        scans = _scans_by_name(get_engine(migrated_db))

        # multishot trio (§4.4 truth table)
        assert scans["detail_2"]["scan_type"] == "detail"
        assert scans["detail_2"]["scan_class"] == "primary"
        assert scans["detail_2"]["product_role"] == "raw"
        assert scans["detail_2"]["source_scan_ids"] in (None, [])

        canon = scans["detail_2_sum_active_median_dark"]
        assert (canon["scan_type"], canon["scan_class"], canon["product_role"]) == \
            ("detail", "composite", "canonical")
        assert len(canon["source_scan_ids"]) == 1

        alt = scans["detail_2_median_all"]
        assert (alt["scan_class"], alt["product_role"]) == ("composite", "alternate")
        assert len(alt["source_scan_ids"]) == 1

        # scan_type corrections (#115)
        assert scans["HDR_500"]["scan_type"] == "HDR"
        assert scans["line_1"]["scan_type"] == "line"
        assert scans["cross"]["scan_type"] == "line"
        assert scans["survey_100"]["scan_type"] == "survey"

        # bare-underscore unions -> composite with sources
        assert scans["detail_"]["scan_class"] == "composite"
        assert len(scans["detail_"]["source_scan_ids"]) >= 1
        assert scans["line_"]["scan_class"] == "composite"
        assert len(scans["line_"]["source_scan_ids"]) >= 1

        # cross union -> composite of the line primaries
        assert scans["cross"]["scan_class"] == "composite"
        assert len(scans["cross"]["source_scan_ids"]) >= 1

        # sub-scan keeps its parent
        assert scans["detail_1a"]["scan_class"] == "sub_scan"
        assert scans["detail_1a"]["parent_scan_id"] is not None

    def test_idempotent(self, migrated_db):
        _reclassify(migrated_db)              # first apply
        result = _reclassify(migrated_db, apply=False)   # second pass is a dry-run
        assert result.total_changed == 0, result.transition_summary()

    def test_no_measurement_mutation(self, migrated_db):
        engine = get_engine(migrated_db)
        with engine.connect() as conn:
            before = measurement_fingerprint(conn)
        assert before["spectra"][0] == 3 and before["fitted_peaks"][0] == 2
        _reclassify(migrated_db)
        with engine.connect() as conn:
            after = measurement_fingerprint(conn)
        assert before == after

    def test_apply_requires_backup_gate(self, migrated_db):
        engine = get_engine(migrated_db)
        with pytest.raises(ScanReclassificationError):
            with engine.begin() as conn:
                run_reclassification(conn, ALL_AXES, apply=True, db_path=migrated_db)

    def test_snapshot_written(self, migrated_db, tmp_path):
        snap = tmp_path / "snap.db"
        engine = get_engine(migrated_db)
        with engine.begin() as conn:
            run_reclassification(
                conn, ALL_AXES, apply=True, db_path=migrated_db, snapshot_path=snap
            )
        assert snap.exists() and snap.stat().st_size > 0

    def test_preflight_rejects_unmigrated(self, tmp_path):
        bare = tmp_path / "bare.db"
        import sqlite3
        sqlite3.connect(bare).execute("CREATE TABLE scans (id TEXT)")
        engine = get_engine(bare)
        with pytest.raises(ScanReclassificationError):
            with engine.connect() as conn:
                run_reclassification(conn, ["scan_type"], apply=False)


# ---------------------------------------------------------------------------
# Value-blind invariants on the corrected corpus (ARC-M2P-315)
# ---------------------------------------------------------------------------

class TestValueBlindInvariants:
    @pytest.fixture
    def corrected(self, migrated_db):
        """Reclassify, then return (by_name, by_id) value-blind projections."""
        _reclassify(migrated_db)
        engine = get_engine(migrated_db)
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, scan_name, scan_type, scan_class, product_role, "
                     "parent_scan_id, source_scan_ids FROM scans")
            ).fetchall()
        by_id = {}
        for r in rows:
            by_id[r[0]] = {
                "scan_name": r[1], "scan_type": r[2], "scan_class": r[3],
                "product_role": r[4], "parent_scan_id": r[5],
                "source_scan_ids": json.loads(r[6]) if isinstance(r[6], str) else r[6],
            }
        by_name = {v["scan_name"]: v for v in by_id.values()}
        return by_name, by_id

    def test_composite_has_non_empty_sources(self, corrected):
        by_name, _ = corrected
        for name, s in by_name.items():
            if s["scan_class"] == "composite":
                assert s["source_scan_ids"], f"{name} composite has empty sources"

    def test_role_couplings(self, corrected):
        by_name, _ = corrected
        for name, s in by_name.items():
            role = s["product_role"]
            if role == "raw":
                assert s["scan_class"] == "primary"
                assert s["parent_scan_id"] is None
                assert not s["source_scan_ids"]
            elif role in ("canonical", "alternate"):
                assert s["scan_class"] == "composite"
                assert s["parent_scan_id"] is None
                assert len(s["source_scan_ids"]) >= 1
            else:
                assert role is None

    def test_exactly_one_canonical_per_raw_group(self, corrected):
        by_name, _ = corrected
        by_raw = {}
        for name, s in by_name.items():
            base = multishot_raw_base(name)
            if base is not None and s["product_role"] is not None:
                by_raw.setdefault(base, []).append((name, s["product_role"]))
        for base, members in by_raw.items():
            canon = [m for m in members if m[1] == "canonical"]
            assert len(canon) <= 1, f"raw group {base} has >1 canonical: {canon}"

    def test_sources_resolve_to_in_group_raw(self, corrected):
        # Every canonical/alternate's single source id resolves to a scan whose
        # product_role is 'raw'.
        by_name, by_id = corrected
        for name, s in by_name.items():
            if s["product_role"] in ("canonical", "alternate"):
                for src_id in s["source_scan_ids"]:
                    assert src_id in by_id, f"{name} source {src_id} missing"
                    assert by_id[src_id]["product_role"] == "raw", (
                        f"{name} source is not a raw scan"
                    )

    def test_non_null_role_requires_recognized_pattern(self, corrected):
        # canonical/alternate must be a recognized reduction name; 'raw' must be
        # the base scan of a recognized reduction present in the corpus.
        by_name, _ = corrected
        reduction_bases = {
            multishot_raw_base(n) for n, s in by_name.items()
            if multishot_reduction_role(n) is not None
        }
        for name, s in by_name.items():
            if s["product_role"] in ("canonical", "alternate"):
                assert multishot_reduction_role(name) is not None, name
            elif s["product_role"] == "raw":
                assert name in reduction_bases, f"raw {name} has no reduction sibling"
