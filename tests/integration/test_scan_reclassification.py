"""Integration tests for the scan-classification source fix.

Covers (value-blind throughout — names / counts / ids only):
  * the product_role DB CHECK rejects invalid (role, class, parent, sources)
    tuples and accepts valid ones;
  * `reclassify-scan-types/-classes/-product-roles` re-derive all three axes to
    the locked truth table, are idempotent, and do not mutate measurement
    tables;
  * the value-blind classification invariants hold on the
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
    finalize_sol_scans,
    measurement_fingerprint,
    plan_product_roles,
    run_reclassification,
    _ScanRow,
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
# Value-blind invariants on the corrected corpus
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
                     "parent_scan_id, source_scan_ids, sol_number, target FROM scans")
            ).fetchall()
        by_id = {}
        for r in rows:
            by_id[r[0]] = {
                "scan_name": r[1], "scan_type": r[2], "scan_class": r[3],
                "product_role": r[4], "parent_scan_id": r[5],
                "source_scan_ids": json.loads(r[6]) if isinstance(r[6], str) else r[6],
                "sol_number": r[7], "target": r[8],
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

    def test_exactly_one_canonical_per_tagged_raw_group(self, corrected):
        # Invariant: every TAGGED raw group has EXACTLY one canonical
        # (not merely <= 1 — a zero-canonical group must never be tagged).
        by_name, _ = corrected
        reductions_by_base = {}
        for name, s in by_name.items():
            base = multishot_raw_base(name)
            if base is not None and s["product_role"] in ("canonical", "alternate"):
                reductions_by_base.setdefault(base, []).append((name, s["product_role"]))
        tagged_raws = [n for n, s in by_name.items() if s["product_role"] == "raw"]
        assert tagged_raws  # the fixture has a complete multishot group
        for raw_name in tagged_raws:
            members = reductions_by_base.get(raw_name, [])
            canon = [m for m in members if m[1] == "canonical"]
            assert len(canon) == 1, (
                f"tagged raw {raw_name} must have exactly one canonical, got {canon}"
            )

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

    def test_no_incomplete_multishot_groups(self, corrected):
        # Every multishot group in the corrected corpus has exactly one canonical
        # (the relational invariant the DB CHECK cannot express).
        from sherloc_pipeline.services.scan_reclassification import (
            multishot_groups_missing_canonical,
        )
        _, by_id = corrected
        records = [(s["sol_number"], s["target"], s["scan_name"]) for s in by_id.values()]
        assert multishot_groups_missing_canonical(records) == []


# ---------------------------------------------------------------------------
# Codex Round-1 finding regressions (F1, F3, F4, F5)
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_migrated_db(tmp_path):
    """A migrated DB with a sol but no scans — tests seed their own corpus."""
    db_path = tmp_path / "phase.db"
    config = Config(str(_PROJECT_ROOT / "alembic.ini"))
    os.environ["PHASE_DATABASE_PATH"] = str(db_path)
    try:
        command.upgrade(config, "head")
        engine = get_engine(db_path)
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO sols (sol_number, data_source, created_at) "
                "VALUES (100,'loupe','2026-01-01')"
            ))
        yield db_path
    finally:
        os.environ.pop("PHASE_DATABASE_PATH", None)


class TestCodexFindings:
    def test_f1_pds_missing_count_null_preserved(self, blank_migrated_db):
        """A PDS synthetic-name row left scan_type=NULL by a missing count
        (stored as the n_points=1 placeholder) must NOT be flipped to detail."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            _seed_scan(conn, "pds_0100_123_001", 1, "primary", None)  # NULL type
            _seed_scan(conn, "detail_1", 100, "primary", None)        # recognized
        _reclassify(blank_migrated_db, axes=("scan_type",))
        scans = _scans_by_name(get_engine(blank_migrated_db))
        assert scans["pds_0100_123_001"]["scan_type"] is None  # preserved
        assert scans["detail_1"]["scan_type"] == "detail"       # corrected

    def test_f3_alternate_only_group_not_tagged(self, blank_migrated_db):
        """A multishot group with no canonical reduction is left untagged so
        its raw stays a counted primary (no silent undercount)."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            # Complete group (canonical + alternate) -> tagged.
            _seed_scan(conn, "detail_2", 300, "primary", "detail")
            _seed_scan(conn, "detail_2_median_all", 100, "composite", "detail")
            _seed_scan(conn, "detail_2_sum_active_median_dark", 100, "composite", "detail")
            # Incomplete group (alternate only) -> NOT tagged.
            _seed_scan(conn, "detail_3", 300, "primary", "detail")
            _seed_scan(conn, "detail_3_median_all", 100, "composite", "detail")
        _reclassify(blank_migrated_db)
        scans = _scans_by_name(get_engine(blank_migrated_db))
        # complete group tagged
        assert scans["detail_2"]["product_role"] == "raw"
        assert scans["detail_2_sum_active_median_dark"]["product_role"] == "canonical"
        # incomplete group: raw stays a counted primary, reduction untagged
        assert scans["detail_3"]["product_role"] is None
        assert scans["detail_3"]["scan_class"] == "primary"
        assert scans["detail_3_median_all"]["product_role"] is None

    def test_f4_create_all_schema_has_product_role_check(self, tmp_path):
        """The ORM (create_all) path produces the same product_role CHECK as
        Alembic, and reclassify's preflight accepts it."""
        from sherloc_pipeline.database.connection import create_all_tables
        from sherloc_pipeline.services.scan_reclassification import preflight_schema
        from sqlalchemy.exc import IntegrityError

        db_path = tmp_path / "create_all.db"
        engine = get_engine(db_path)
        create_all_tables(engine)
        with engine.connect() as conn:
            scans_sql = conn.execute(text(
                "SELECT sql FROM sqlite_master WHERE name='scans'")).scalar()
        assert "ck_scans_product_role" in scans_sql
        # CHECK is enforced on a create_all DB:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO sols (sol_number, data_source, created_at) "
                "VALUES (100,'loupe','2026-01-01')"))
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO scans (id, sol_number, scan_name, scan_id, sclk_start, "
                    "n_points, n_channels, laser_wavelength_nm, created_at, target_type, "
                    "scan_class, product_role) VALUES "
                    "('x',100,'s','s',1,1,2148,248.6,'2026-01-01','mars_target',"
                    "'composite','raw')"))  # raw must be primary
        # preflight accepts the create_all schema (no alembic_version needed):
        with engine.connect() as conn:
            preflight_schema(conn)

    def test_f5_spatial_union_excludes_unrelated_base(self, blank_migrated_db):
        """A target-prefixed spatial union links only its own base family, not
        an unrelated different-base same-kind primary in the same group."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            _seed_scan(conn, "meteorite_detail_1", 100, "primary", "detail")
            _seed_scan(conn, "meteorite_detail_2", 100, "primary", "detail")
            _seed_scan(conn, "cal_detail_1", 100, "primary", "detail")  # unrelated base
            _seed_scan(conn, "meteorite_detail_all", 200, "primary", "detail")
        _reclassify(blank_migrated_db, axes=("scan_class",))
        engine = get_engine(blank_migrated_db)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, scan_name, source_scan_ids FROM scans")).fetchall()
        by_id = {r[0]: r[1] for r in rows}
        union = next(r for r in rows if r[1] == "meteorite_detail_all")
        src_names = {by_id[i] for i in json.loads(union[2])}
        assert src_names == {"meteorite_detail_1", "meteorite_detail_2"}
        assert "cal_detail_1" not in src_names


def _row(scan_id, name, n_points=100, scan_class="primary", product_role=None,
         parent=None, sources=None, scan_type=None, sol=100, target="TargetA"):
    return _ScanRow(
        id=scan_id, sol_number=sol, target=target, scan_name=name,
        sequence_id=None, n_points=n_points, scan_type=scan_type,
        scan_class=scan_class, parent_scan_id=parent, source_scan_ids=sources,
        product_role=product_role,
    )


class TestCodexRound2Findings:
    def test_f8_planner_requires_canonical(self):
        """plan_product_roles tags a group only with exactly one canonical;
        an alternate-only group leaves the raw untagged (counted)."""
        # Complete group -> tagged.
        good = plan_product_roles([
            _row("r1", "detail_2", 300),
            _row("a1", "detail_2_median_all", 100, scan_class="composite"),
            _row("c1", "detail_2_sum_active_median_dark", 100, scan_class="composite"),
        ])
        assert good.updates["r1"]["product_role"] == "raw"
        assert good.updates["c1"]["product_role"] == "canonical"
        # Alternate-only group -> NOT tagged (raw absent from updates).
        bad = plan_product_roles([
            _row("r2", "detail_3", 300),
            _row("a2", "detail_3_median_all", 100, scan_class="composite"),
        ])
        assert "r2" not in bad.updates  # raw stays a counted primary
        assert "a2" not in bad.updates
        assert "a2" in bad.quarantined

    def test_f9_pds_real_count_reclassified(self, blank_migrated_db):
        """A PDS row with a real count > 1 but stale/NULL scan_type IS
        reclassified via the permitted count fallback; only the n_points<=1
        missing-count NULL is preserved."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            _seed_scan(conn, "pds_0100_1_001", 300, "primary", None)   # real count
            _seed_scan(conn, "pds_0100_1_002", 100, "primary", None)   # real count
            _seed_scan(conn, "pds_0100_1_003", 1, "primary", None)     # missing-count
        _reclassify(blank_migrated_db, axes=("scan_type",))
        scans = _scans_by_name(get_engine(blank_migrated_db))
        assert scans["pds_0100_1_001"]["scan_type"] == "survey"   # 300 > threshold
        assert scans["pds_0100_1_002"]["scan_type"] == "detail"   # <= threshold
        assert scans["pds_0100_1_003"]["scan_type"] is None       # preserved

    def test_f7_finalize_populates_lineage_and_roles(self, blank_migrated_db):
        """finalize_sol_scans (the write-time hook) fills composite
        source_scan_ids, sub_scan parents, and multishot product_role for a
        sol whose scans were written without them."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            # Written as a fresh single-scan path would: scan_class set, but no
            # lineage / product_role yet.
            _seed_scan(conn, "detail_1", 100, "primary", "detail")
            _seed_scan(conn, "detail_2", 300, "primary", "detail")
            _seed_scan(conn, "detail_1a", 50, "sub_scan", "detail")
            _seed_scan(conn, "detail_", 200, "composite", "detail")
            _seed_scan(conn, "detail_2_median_all", 100, "composite", "detail")
            _seed_scan(conn, "detail_2_sum_active_median_dark", 100, "composite", "detail")
        with engine.begin() as conn:
            finalize_sol_scans(conn, 100)
        scans = _scans_by_name(get_engine(blank_migrated_db))
        # composite gets non-empty sources
        assert len(scans["detail_"]["source_scan_ids"]) >= 1
        # sub_scan gets a parent
        assert scans["detail_1a"]["parent_scan_id"] is not None
        # multishot roles assigned
        assert scans["detail_2"]["product_role"] == "raw"
        assert scans["detail_2_sum_active_median_dark"]["product_role"] == "canonical"
        assert scans["detail_2_median_all"]["product_role"] == "alternate"

    def test_f6_named_union_uses_locked_constituents(self, blank_migrated_db):
        """The locked taxonomy (§4.1) fixes `cross` = line_1,line_2: a cross
        links exactly those, NOT an unrelated extra line in the same group."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            _seed_scan(conn, "line_1", 25, "primary", "line")
            _seed_scan(conn, "line_2", 25, "primary", "line")
            _seed_scan(conn, "line_3", 25, "primary", "line")  # extra, not in the cross
            _seed_scan(conn, "cross", 50, "composite", "line")
        _reclassify(blank_migrated_db, axes=("scan_class",))
        engine = get_engine(blank_migrated_db)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, scan_name, source_scan_ids FROM scans")).fetchall()
        by_id = {r[0]: r[1] for r in rows}
        cross = next(r for r in rows if r[1] == "cross")
        src = {by_id[i] for i in json.loads(cross[2])}
        assert src == {"line_1", "line_2"}  # line_3 excluded

    def test_f6_asterisk_links_four_lines(self, blank_migrated_db):
        """`asterisk` = line_1..line_4 per the locked taxonomy."""
        engine = get_engine(blank_migrated_db)
        with engine.begin() as conn:
            for i in range(1, 5):
                _seed_scan(conn, f"line_{i}", 25, "primary", "line")
            _seed_scan(conn, "asterisk", 100, "composite", "line")
        _reclassify(blank_migrated_db, axes=("scan_class",))
        engine = get_engine(blank_migrated_db)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, scan_name, source_scan_ids FROM scans")).fetchall()
        by_id = {r[0]: r[1] for r in rows}
        ast = next(r for r in rows if r[1] == "asterisk")
        src = {by_id[i] for i in json.loads(ast[2])}
        assert src == {"line_1", "line_2", "line_3", "line_4"}

    def test_f8_invariant_catches_incomplete_multishot_group(self):
        """The value-blind invariant flags a multishot group lacking its
        canonical (raw + alternate, no canonical) — the synthetic invalid case
        the V&V suite must reject."""
        from sherloc_pipeline.services.scan_reclassification import (
            multishot_groups_missing_canonical,
        )
        # Complete group -> holds.
        assert multishot_groups_missing_canonical([
            (100, "T", "detail_2"),
            (100, "T", "detail_2_median_all"),
            (100, "T", "detail_2_sum_active_median_dark"),
        ]) == []
        # Alternate-only group -> flagged.
        assert multishot_groups_missing_canonical([
            (100, "T", "detail_3"),
            (100, "T", "detail_3_median_all"),
        ]) == [(100, "T", "detail_3")]

    def test_f10_invariant_is_sol_target_group_scoped(self):
        """A raw base that recurs across sol/target groups cannot mask an
        incomplete group: an alternate-only group is flagged even when another
        group with the same raw base name has a canonical."""
        from sherloc_pipeline.services.scan_reclassification import (
            multishot_groups_missing_canonical,
        )
        records = [
            # sol 100: incomplete (alternate only)
            (100, "T", "detail_3"),
            (100, "T", "detail_3_median_all"),
            # sol 200: complete (same raw base name, different group)
            (200, "T", "detail_3"),
            (200, "T", "detail_3_sum_active_median_dark"),
        ]
        assert multishot_groups_missing_canonical(records) == [(100, "T", "detail_3")]
