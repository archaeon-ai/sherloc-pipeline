"""In-place reclassification of the three SHERLOC scan-classification axes.

WS-1 scan-classification spec §4.5 (ARC-M2P-312). Re-derives ``scan_type``
(name-authoritative), ``scan_class`` (+ ``parent_scan_id`` / ``source_scan_ids``
lineage), and ``product_role`` (multishot analytical role) for the existing
corpus **in place**, going-forward-safe, with operational safeguards:

* ``--dry-run`` is the default (caller passes ``apply=True`` to write);
* a single transaction per run;
* a value-blind transition-count diff per axis (no science measurement values
  — only names, counts, sequence codes, ids, and error tokens);
* a backup/snapshot gate before applying;
* a preflight schema/migration-version check;
* **no mutation of measurement tables** (``spectra`` / ``fitted_peaks``),
  asserted by a pre/post row-count + content hash;
* idempotence (a second run is a no-op).

This module is value-blind: it reads ``scan_name`` / ``n_points`` /
``sequence_id`` / ids only, never spectra or peak values — so it is safe to run
under pds-guard.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection

from sherloc_pipeline.models.spectra import (
    ScanType,
    classify_scan_class,
    classify_scan_type,
    derive_parent_name,
    multishot_raw_base,
    multishot_reduction_role,
    _CALIBRATION_SEQUENCE_CODES,
    _is_calibration_name,
    _is_uninformative_name,
    _scan_type_from_name,
)
from sherloc_pipeline.services.errors import SherlocServiceError

# Migration that introduced the product_role column + CHECK. The reclassifier
# refuses to run against a DB older than this (the column/CHECK must exist).
PRODUCT_ROLE_REVISION = "9b2e7c4a1f08"

# Measurement tables whose contents must be invariant across a reclassify run.
MEASUREMENT_TABLES: Tuple[str, ...] = ("spectra", "fitted_peaks")

# The three reclassification axes, in dependency-safe order.
AXIS_SCAN_TYPE = "scan_type"
AXIS_SCAN_CLASS = "scan_class"
AXIS_PRODUCT_ROLE = "product_role"
ALL_AXES: Tuple[str, ...] = (AXIS_SCAN_TYPE, AXIS_SCAN_CLASS, AXIS_PRODUCT_ROLE)


class ScanReclassificationError(SherlocServiceError):
    """Raised when reclassification cannot proceed safely."""


# ---------------------------------------------------------------------------
# Value-blind data structures
# ---------------------------------------------------------------------------

@dataclass
class _ScanRow:
    """A value-blind projection of a scan row (no measurement values)."""

    id: str
    sol_number: int
    target: Optional[str]
    scan_name: str
    sequence_id: Optional[str]
    n_points: int
    scan_type: Optional[str]
    scan_class: str
    parent_scan_id: Optional[str]
    source_scan_ids: Optional[list]
    product_role: Optional[str]


@dataclass
class AxisPlan:
    """The planned (and optionally applied) changes for a single axis."""

    axis: str
    # Each update is a value-blind dict of column -> new value, keyed by id.
    updates: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # "<old> -> <new>" transition string -> count (value-blind summary).
    transitions: Dict[str, int] = field(default_factory=dict)
    # Scan ids quarantined this pass (informative-unknown names / unresolved
    # multishot reductions). Value-blind: ids + names only.
    quarantined: List[str] = field(default_factory=list)

    @property
    def n_changed(self) -> int:
        return len(self.updates)


@dataclass
class ReclassificationResult:
    """Outcome of a reclassification run (value-blind)."""

    axes: List[str]
    applied: bool
    total_scans: int
    plans: Dict[str, AxisPlan]
    snapshot_path: Optional[str] = None

    def transition_summary(self) -> Dict[str, Dict[str, int]]:
        return {axis: dict(plan.transitions) for axis, plan in self.plans.items()}

    @property
    def total_changed(self) -> int:
        return sum(plan.n_changed for plan in self.plans.values())


# ---------------------------------------------------------------------------
# Schema preflight + measurement-table fingerprint
# ---------------------------------------------------------------------------

def preflight_schema(conn: Connection) -> str:
    """Verify the scans table carries the product_role column + governance
    CHECK (WS-1 §4.4) before reclassifying.

    Checks the schema artifact directly rather than the Alembic version, so it
    accepts BOTH migration-built and ORM-created (``create_all``) schemas —
    which carry the same ``ck_scans_product_role`` constraint (Codex F4).
    Returns the Alembic head if the DB is migration-managed (informational).
    """
    scans_sql = conn.execute(
        text("SELECT sql FROM sqlite_master WHERE type='table' AND name='scans'")
    ).scalar()
    if not scans_sql:
        raise ScanReclassificationError(
            "No 'scans' table found; initialize the database first "
            "(`sherloc init` / `alembic upgrade head`).",
        )
    if "product_role" not in scans_sql or "ck_scans_product_role" not in scans_sql:
        raise ScanReclassificationError(
            "Database schema is missing the product_role column / CHECK "
            f"(migration {PRODUCT_ROLE_REVISION}); run `alembic upgrade head` "
            "before reclassifying.",
            context={"has_product_role_column": "product_role" in scans_sql},
        )
    try:
        versions = [
            r[0]
            for r in conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
        ]
    except Exception:  # noqa: BLE001 — alembic_version absent on create_all DBs
        versions = []
    return ",".join(sorted(versions))


def measurement_fingerprint(conn: Connection) -> Dict[str, Tuple[int, str]]:
    """Return {table: (row_count, content_sha256)} for the measurement tables.

    The content hash streams rows in primary-key order so it is deterministic
    and order-independent of the query plan. Used to assert that a reclassify
    run does not mutate ``spectra`` / ``fitted_peaks``.
    """
    fingerprint: Dict[str, Tuple[int, str]] = {}
    for table in MEASUREMENT_TABLES:
        exists = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:t"),
            {"t": table},
        ).scalar()
        if not exists:
            continue
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
        hasher = hashlib.sha256()
        # Order by rowid for a stable, index-free traversal that covers every
        # column (including BLOBs) without loading the whole table into memory.
        result = conn.execution_options(stream_results=True).execute(
            text(f"SELECT * FROM {table} ORDER BY rowid")
        )
        for row in result:
            hasher.update(repr(tuple(row)).encode("utf-8", "surrogatepass"))
        fingerprint[table] = (int(count), hasher.hexdigest())
    return fingerprint


# ---------------------------------------------------------------------------
# Per-axis planners (pure: read rows, emit value-blind update plans)
# ---------------------------------------------------------------------------

def _load_rows(conn: Connection, sol_number: Optional[int] = None) -> List[_ScanRow]:
    import json

    sql = (
        "SELECT id, sol_number, target, scan_name, sequence_id, n_points, "
        "scan_type, scan_class, parent_scan_id, source_scan_ids, product_role "
        "FROM scans"
    )
    params: Dict[str, object] = {}
    if sol_number is not None:
        sql += " WHERE sol_number = :sol"
        params["sol"] = sol_number
    rows = conn.execute(text(sql), params).fetchall()
    out: List[_ScanRow] = []
    for r in rows:
        raw_sources = r[9]
        sources = json.loads(raw_sources) if isinstance(raw_sources, str) else raw_sources
        out.append(
            _ScanRow(
                id=r[0],
                sol_number=r[1],
                target=r[2],
                scan_name=r[3],
                sequence_id=r[4],
                n_points=r[5],
                scan_type=r[6],
                scan_class=r[7],
                parent_scan_id=r[8],
                source_scan_ids=sources,
                product_role=r[10],
            )
        )
    return out


def _group_key(row: _ScanRow) -> Tuple[int, Optional[str]]:
    return (row.sol_number, row.target)


def _name_kind(scan_name: str) -> Optional[str]:
    """The inherited ScanType value for a name, or None (calibration/unknown)."""
    resolved = classify_scan_type(scan_name)
    return resolved.value if isinstance(resolved, ScanType) else None


def plan_scan_types(rows: Sequence[_ScanRow]) -> AxisPlan:
    """Re-derive name-authoritative ``scan_type`` in place (the #115 fix).

    Per the §4.2 resolver:

    * calibration (sequence code or AlGaN name) -> calibration;
    * a RECOGNIZED name -> its name-implied type (corrects the mislabels);
    * an UNINFORMATIVE name (empty / synthetic ``pds_*``) -> the spectrum-count
      fallback (ARC-M2P-308 permits it here; ARC-M2P-312 wants in-place
      re-derivation, Codex F9) — **except** a missing-count NULL is preserved:
      ``n_points <= 1`` is the PDS missing-count placeholder, so a row left
      ``scan_type=NULL`` at ingest is never promoted to a guessed type from it
      (Codex F1). A genuine 1-spectrum PDS scan was already typed at ingest, so
      this guard only catches the placeholder;
    * an informative-but-unrecognized (UNKNOWN) name -> quarantined: NULL (no
      guessed type kept), recorded. On the current corpus this class is empty.
    """
    plan = AxisPlan(axis=AXIS_SCAN_TYPE)
    for row in rows:
        resolved = classify_scan_type(row.scan_name, row.sequence_id, row.n_points)
        if isinstance(resolved, ScanType):
            new_type: Optional[str] = resolved.value
        else:  # SCAN_TYPE_QUARANTINE (informative-unknown)
            new_type = None
            plan.quarantined.append(row.id)

        # Preserve a PDS missing-count NULL: never promote an ingest-time NULL
        # to a count-guessed type from the n_points==1 placeholder of an
        # uninformative (synthetic pds_*) name.
        if (
            new_type is not None
            and row.scan_type is None
            and row.n_points <= 1
            and _is_uninformative_name(row.scan_name)
        ):
            continue

        if new_type != row.scan_type:
            plan.updates[row.id] = {"scan_type": new_type}
            key = f"{row.scan_type!r}->{new_type!r}"
            plan.transitions[key] = plan.transitions.get(key, 0) + 1
    return plan


def plan_scan_classes(rows: Sequence[_ScanRow]) -> AxisPlan:
    """Re-derive ``scan_class`` + ``parent_scan_id`` / ``source_scan_ids``.

    Catches the bare trailing-underscore unions the historical backfill
    missed, and (re)derives lineage so the value-blind non-empty-sources
    invariant holds for composites.
    """
    plan = AxisPlan(axis=AXIS_SCAN_CLASS)
    by_group: Dict[Tuple[int, Optional[str]], List[_ScanRow]] = {}
    by_name: Dict[Tuple[int, Optional[str], str], _ScanRow] = {}
    for row in rows:
        by_group.setdefault(_group_key(row), []).append(row)
        by_name[(row.sol_number, row.target, row.scan_name)] = row

    for row in rows:
        new_class = classify_scan_class(row.scan_name)
        new_parent: Optional[str] = None
        new_sources: Optional[list] = None

        if new_class == "sub_scan":
            parent_name = derive_parent_name(row.scan_name)
            if parent_name:
                parent = by_name.get((row.sol_number, row.target, parent_name))
                if parent is not None:
                    new_parent = parent.id
        elif new_class == "composite":
            new_sources = _derive_composite_sources(row, by_group.get(_group_key(row), []), by_name)

        if (
            new_class != row.scan_class
            or new_parent != row.parent_scan_id
            or (new_sources or None) != (row.source_scan_ids or None)
        ):
            plan.updates[row.id] = {
                "scan_class": new_class,
                "parent_scan_id": new_parent,
                "source_scan_ids": new_sources if new_sources else None,
            }
            key = f"{row.scan_class!r}->{new_class!r}"
            plan.transitions[key] = plan.transitions.get(key, 0) + 1
    return plan


def _extends_base(scan_name: Optional[str], base_prefix_lower: str) -> bool:
    """True if scan_name is the base itself or a member extending it by a
    separator/index (e.g. base 'detail' matches 'detail', 'detail_1',
    'detail2' but NOT 'detailx_1')."""
    low = (scan_name or "").strip().lower()
    if low == base_prefix_lower:
        return True
    if not low.startswith(base_prefix_lower):
        return False
    nxt = low[len(base_prefix_lower)]
    return nxt == "_" or nxt.isdigit()


def _derive_composite_sources(
    composite: _ScanRow,
    group: Sequence[_ScanRow],
    by_name: Dict[Tuple[int, Optional[str], str], _ScanRow],
) -> list:
    """Best-effort constituent ids for a composite (value-blind, name-based).

    Mirrors the historical Alembic backfill (suffix-strip -> base prefix), so
    a spatial/bare union attaches only the in-group primaries that extend its
    OWN base (``detail_all``/``detail_`` -> ``detail_1``/``detail_2``, NOT an
    unrelated different-base same-kind scan — Codex F5). Specifically:

    * Multishot reduction (``*_median_all`` / ``*_sum_active_*``): the single
      raw base scan by exact name; if absent, the base-prefix members.
    * Spatial / bare union (``*_all`` / trailing ``_``): the base-prefix
      primaries.
    * Named union (``cross`` / ``asterisk``): constituents are not
      name-derivable, so fall back to the same-kind primaries in the group
      (the documented best-effort — the line scans at this sol/target).
    """
    clean = composite.scan_name.strip()
    low = clean.lower()

    raw_base = multishot_raw_base(clean)
    base_prefix: Optional[str] = None
    if raw_base is not None:
        raw = by_name.get((composite.sol_number, composite.target, raw_base))
        if raw is not None:
            return [raw.id]
        base_prefix = raw_base.lower()
    elif low.endswith("_all"):
        base_prefix = low[: -len("_all")]
    elif low.endswith("_"):
        base_prefix = low[:-1]

    if base_prefix:
        members = [
            member.id
            for member in group
            if member.id != composite.id
            and classify_scan_class(member.scan_name) == "primary"
            and _extends_base(member.scan_name, base_prefix)
        ]
        if members:
            return members

    # Named union (cross / asterisk) or an unresolved reduction: the EXACT
    # constituents are not derivable from the name — `cross` / `asterisk`
    # carry no link to specific `line_N` scans, and only operational metadata
    # (which lines the operator combined) would identify the subset, which is
    # outside SP1's value-blind name/count inputs. Spec §4.3 defines a
    # composite's constituents as "the constituent primaries on the same
    # sol/target", and §4.9 REQUIRES a non-empty source_scan_ids array — so the
    # same-kind primaries at the sol/target are the documented best-effort
    # (exact when the target's lines are precisely the union's constituents,
    # the real corpus). Leaving sources empty is not an option (it would
    # violate §4.9 / ARC-M2P-310 AC2).
    kind = _name_kind(composite.scan_name)
    if kind is None:
        return []
    return [
        member.id
        for member in group
        if member.id != composite.id
        and classify_scan_class(member.scan_name) == "primary"
        and _name_kind(member.scan_name) == kind
    ]


def plan_product_roles(rows: Sequence[_ScanRow]) -> AxisPlan:
    """Assign ``product_role`` to multishot products, at the raw-group level.

    A multishot group is the raw scan plus its recognized reductions
    (``*_sum_active_median_dark`` → canonical; ``*_median_all`` /
    ``*_sum_active_sum_dark`` → alternate), keyed by the reduction's raw base
    name. A group is tagged **only when it has its raw sibling AND exactly one
    canonical reduction** (the ARC-M2P-311 / -315 "exactly one canonical per
    raw group" invariant, Codex F3). An incomplete group — no raw, no
    canonical, or >1 canonical — is left untagged: the raw stays a counted
    primary and its reductions stay composites with ``product_role`` NULL, so
    no spatial positions are silently dropped. Reductions in untagged groups
    are recorded as quarantined.
    """
    plan = AxisPlan(axis=AXIS_PRODUCT_ROLE)
    by_name: Dict[Tuple[int, Optional[str], str], _ScanRow] = {
        (r.sol_number, r.target, r.scan_name): r for r in rows
    }

    # Group reductions by their resolved raw scan.
    groups: Dict[str, Dict[str, object]] = {}
    for row in rows:
        role = multishot_reduction_role(row.scan_name)
        if role is None:
            continue
        base = multishot_raw_base(row.scan_name)
        raw = by_name.get((row.sol_number, row.target, base)) if base else None
        if raw is None:
            plan.quarantined.append(row.id)  # reduction with no raw sibling
            continue
        group = groups.setdefault(raw.id, {"raw": raw, "reductions": []})
        group["reductions"].append((row, role))  # type: ignore[union-attr]

    for group in groups.values():
        raw: _ScanRow = group["raw"]  # type: ignore[assignment]
        reductions = group["reductions"]  # type: ignore[index]
        canonical_count = sum(1 for _r, role in reductions if role == "canonical")
        if canonical_count != 1:
            # Incomplete group: do not tag. The raw remains counted.
            for r, _role in reductions:
                plan.quarantined.append(r.id)
            continue
        for r, role in reductions:
            _stage_role_update(plan, r, {
                "product_role": role,
                "scan_class": "composite",
                "parent_scan_id": None,
                "source_scan_ids": [raw.id],
            })
        _stage_role_update(plan, raw, {
            "product_role": "raw",
            "scan_class": "primary",
            "parent_scan_id": None,
            "source_scan_ids": None,
        })
    return plan


def _stage_role_update(plan: AxisPlan, row: _ScanRow, new: Dict[str, object]) -> None:
    changed = (
        new["product_role"] != row.product_role
        or new["scan_class"] != row.scan_class
        or new["parent_scan_id"] != row.parent_scan_id
        or (new["source_scan_ids"] or None) != (row.source_scan_ids or None)
    )
    if changed:
        plan.updates[row.id] = dict(new)
        key = f"{row.product_role!r}->{new['product_role']!r}"
        plan.transitions[key] = plan.transitions.get(key, 0) + 1


PLANNERS = {
    AXIS_SCAN_TYPE: plan_scan_types,
    AXIS_SCAN_CLASS: plan_scan_classes,
    AXIS_PRODUCT_ROLE: plan_product_roles,
}


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _apply_plan(conn: Connection, plan: AxisPlan) -> None:
    """Write a plan's updates as single coupled UPDATE statements per row."""
    import json

    for scan_id, cols in plan.updates.items():
        assignments = []
        params: Dict[str, object] = {"id": scan_id}
        for col, value in cols.items():
            if col == "source_scan_ids":
                params[col] = json.dumps(value) if value else None
            else:
                params[col] = value
            assignments.append(f"{col} = :{col}")
        conn.execute(
            text(f"UPDATE scans SET {', '.join(assignments)} WHERE id = :id"),
            params,
        )


# ---------------------------------------------------------------------------
# Write-time finalization (going-forward ingests)
# ---------------------------------------------------------------------------

def finalize_sol_scans(conn: Connection, sol_number: int) -> Dict[str, AxisPlan]:
    """Populate corpus-level lineage + product_role for a freshly-ingested sol.

    ``scan_type`` and ``scan_class`` are set name-authoritatively at write time
    (``to_scan()`` / ``ScanORM.__init__``), but ``source_scan_ids`` /
    ``parent_scan_id`` (lineage) and ``product_role`` (multishot role) require
    the sibling scans of the sol/target, which a single-scan write path does
    not have. This runs once, after all of a sol's workspaces are ingested, so
    going-forward ingests satisfy the composite non-empty-sources guarantee
    (ARC-M2P-310) and the multishot product_role model (ARC-M2P-311) without a
    separate operator reclassify pass (Codex F7).

    Operates in place on the caller's connection/transaction; only the
    ``scan_class`` (lineage) and ``product_role`` axes are derived (scan_type
    is already final). Returns the per-axis plans (value-blind).
    """
    plans: Dict[str, AxisPlan] = {}
    rows = _load_rows(conn, sol_number=sol_number)
    for axis in (AXIS_SCAN_CLASS, AXIS_PRODUCT_ROLE):
        plan = PLANNERS[axis](rows)
        plans[axis] = plan
        if plan.updates:
            _apply_plan(conn, plan)
            rows = _load_rows(conn, sol_number=sol_number)
    return plans


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_reclassification(
    conn: Connection,
    axes: Sequence[str],
    *,
    apply: bool = False,
    snapshot_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    have_backup: bool = False,
) -> ReclassificationResult:
    """Plan (and optionally apply) reclassification for the requested axes.

    The caller owns the transaction boundary (``conn`` should be inside a
    ``begin()``). On ``apply=True`` a measurement-table fingerprint is taken
    before and after and asserted unchanged; any mismatch raises (rolling back
    the surrounding transaction).
    """
    bad = [a for a in axes if a not in PLANNERS]
    if bad:
        raise ScanReclassificationError(f"Unknown reclassification axis: {bad}")

    preflight_schema(conn)

    if apply:
        if snapshot_path is None and not have_backup:
            raise ScanReclassificationError(
                "Refusing to --apply without a backup: pass --snapshot <path> "
                "(writes a copy of the DB first) or --i-have-a-backup.",
                exit_code=2,
            )
        if snapshot_path is not None:
            if db_path is None:
                raise ScanReclassificationError(
                    "--snapshot requires a file-backed database path."
                )
            _write_snapshot(db_path, snapshot_path)

    rows = _load_rows(conn)
    pre_fp = measurement_fingerprint(conn) if apply else {}

    plans: Dict[str, AxisPlan] = {}
    for axis in axes:
        plan = PLANNERS[axis](rows)
        plans[axis] = plan
        if apply and plan.updates:
            _apply_plan(conn, plan)
            # Refresh the in-memory rows so a later axis sees this axis's writes
            # (e.g. product_role's CHECK relies on scan_class already correct).
            rows = _load_rows(conn)

    if apply:
        post_fp = measurement_fingerprint(conn)
        _assert_measurements_unchanged(pre_fp, post_fp)

    return ReclassificationResult(
        axes=list(axes),
        applied=apply,
        total_scans=len(rows),
        plans=plans,
        snapshot_path=str(snapshot_path) if snapshot_path else None,
    )


def _assert_measurements_unchanged(
    pre: Dict[str, Tuple[int, str]],
    post: Dict[str, Tuple[int, str]],
) -> None:
    for table in MEASUREMENT_TABLES:
        if pre.get(table) != post.get(table):
            raise ScanReclassificationError(
                f"Reclassification mutated measurement table '{table}' "
                f"(pre={pre.get(table)} post={post.get(table)}) — rolling back.",
                context={"table": table, "pre": pre.get(table), "post": post.get(table)},
            )


def _write_snapshot(db_path: Path, snapshot_path: Path) -> None:
    if not db_path.exists():
        raise ScanReclassificationError(f"Database not found for snapshot: {db_path}")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(db_path, snapshot_path)
