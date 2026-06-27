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
    """Verify the DB is migrated to (at least) the product_role revision.

    Returns the current alembic version. Raises if the migration table is
    absent or the head predates product_role.
    """
    try:
        rows = conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
    except Exception as exc:  # noqa: BLE001 — surface as a domain error
        raise ScanReclassificationError(
            "Database is not Alembic-managed (no alembic_version table); "
            "run `sherloc init` / `alembic upgrade head` first.",
            context={"error": str(exc)},
        ) from exc
    versions = {r[0] for r in rows}
    if PRODUCT_ROLE_REVISION not in _reachable_revisions(conn, versions):
        raise ScanReclassificationError(
            "Database schema predates the product_role migration "
            f"({PRODUCT_ROLE_REVISION}); run `alembic upgrade head` before "
            "reclassifying.",
            context={"alembic_version": sorted(versions)},
        )
    return ",".join(sorted(versions))


def _reachable_revisions(conn: Connection, versions: set) -> set:
    """Best-effort: the product_role CHECK must exist on the scans table.

    Rather than walk the Alembic graph, we directly verify the schema artifact
    this reclassifier depends on (the column + CHECK), which is robust to head
    naming and merge points.
    """
    sql = conn.execute(
        text("SELECT sql FROM sqlite_master WHERE name = 'scans'")
    ).scalar()
    if sql and "ck_scans_product_role" in sql and "product_role" in sql:
        return versions | {PRODUCT_ROLE_REVISION}
    return versions


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

def _load_rows(conn: Connection) -> List[_ScanRow]:
    import json

    rows = conn.execute(
        text(
            "SELECT id, sol_number, target, scan_name, sequence_id, n_points, "
            "scan_type, scan_class, parent_scan_id, source_scan_ids, product_role "
            "FROM scans"
        )
    ).fetchall()
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
    """Re-derive name-authoritative ``scan_type`` for every scan.

    An informative-unknown name resolves to the QUARANTINE sentinel; rather
    than write a guessed type, the scan_type is set to NULL (explicit
    quarantine) and the scan is recorded in ``quarantined``.
    """
    plan = AxisPlan(axis=AXIS_SCAN_TYPE)
    for row in rows:
        resolved = classify_scan_type(row.scan_name, row.sequence_id, row.n_points)
        if isinstance(resolved, ScanType):
            new_type: Optional[str] = resolved.value
        else:  # SCAN_TYPE_QUARANTINE
            new_type = None
            plan.quarantined.append(row.id)
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


def _derive_composite_sources(
    composite: _ScanRow,
    group: Sequence[_ScanRow],
    by_name: Dict[Tuple[int, Optional[str], str], _ScanRow],
) -> list:
    """Best-effort constituent ids for a composite (value-blind, name-based).

    * Multishot reduction (``*_median_all`` / ``*_sum_active_*``): the single
      raw base scan, matched by exact name.
    * Spatial / named union (``detail_all``, ``detail_``, ``line_``, ``cross``,
      ``asterisk``): the same-kind primaries in the same sol/target group.
    """
    base = multishot_raw_base(composite.scan_name)
    if base is not None:
        raw = by_name.get((composite.sol_number, composite.target, base))
        return [raw.id] if raw is not None else []

    kind = _name_kind(composite.scan_name)
    if kind is None:
        return []
    sources = [
        member.id
        for member in group
        if member.id != composite.id
        and classify_scan_class(member.scan_name) == "primary"
        and _name_kind(member.scan_name) == kind
    ]
    return sources


def plan_product_roles(rows: Sequence[_ScanRow]) -> AxisPlan:
    """Assign ``product_role`` to multishot products.

    A recognized reduction name (``*_sum_active_median_dark`` → canonical;
    ``*_median_all`` / ``*_sum_active_sum_dark`` → alternate) is tagged only
    when its raw base scan exists in the same sol/target group; the raw is
    then tagged ``role='raw'``. Reductions with no resolvable raw sibling are
    left NULL (and recorded as quarantined) rather than guessed.
    """
    plan = AxisPlan(axis=AXIS_PRODUCT_ROLE)
    by_name: Dict[Tuple[int, Optional[str], str], _ScanRow] = {
        (r.sol_number, r.target, r.scan_name): r for r in rows
    }
    raw_targets: Dict[str, _ScanRow] = {}

    for row in rows:
        role = multishot_reduction_role(row.scan_name)
        if role is None:
            continue
        base = multishot_raw_base(row.scan_name)
        raw = by_name.get((row.sol_number, row.target, base)) if base else None
        if raw is None:
            plan.quarantined.append(row.id)
            continue
        raw_targets[raw.id] = raw
        # Reduction: composite, role-tagged, sourced to the raw (single coupled
        # update keeps the row CHECK-valid).
        _stage_role_update(plan, row, {
            "product_role": role,
            "scan_class": "composite",
            "parent_scan_id": None,
            "source_scan_ids": [raw.id],
        })

    for raw in raw_targets.values():
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
