"""add_scan_class_parent_source_columns

Add scan_class, parent_scan_id, source_scan_ids columns to scans table.

Phase 1: Add columns (nullable scan_class initially for backfill).
Phase 2: Python-based backfill of scan_class using frozen classification logic.
Phase 3: Python-based backfill of parent_scan_id for sub-scans.
Phase 4: Best-effort backfill of source_scan_ids for composites.
Phase 5: Tighten scan_class to NOT NULL, add CHECK + index.

Revision ID: 1449d3a649cd
Revises: 1dea6b52bb6f
Create Date: 2026-03-16 17:05:51.840538

"""
import json
import re
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '1449d3a649cd'
down_revision: Union[str, Sequence[str], None] = '1dea6b52bb6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# --- Frozen classification logic (do not import from models) ---

_COMPOSITE_PATTERNS = ("_all", "_median", "_sum_active")


def _classify_scan_class(scan_name: str) -> str:
    """Classify a scan as primary, sub_scan, or composite."""
    clean = (scan_name or "").strip()
    lower = clean.lower()
    for pat in _COMPOSITE_PATTERNS:
        if pat in lower:
            return "composite"
    if len(clean) >= 2:
        last = clean[-1]
        prev = clean[-2]
        if last in ("a", "b", "c") and (prev.isdigit() or prev == "_"):
            return "sub_scan"
    return "primary"


def _derive_parent_name(scan_name: str) -> str | None:
    """Derive parent scan name from a sub-scan name."""
    if not scan_name or len(scan_name) < 2:
        return None
    last = scan_name[-1]
    prev = scan_name[-2]
    if last in ("a", "b", "c") and (prev.isdigit() or prev == "_"):
        if prev == "_":
            return scan_name[:-2]
        else:
            return scan_name[:-1]
    return None


def upgrade() -> None:
    """Add scan_class, parent_scan_id, source_scan_ids with backfill."""

    # --- Phase 1: Add nullable columns ---
    with op.batch_alter_table('scans') as batch_op:
        batch_op.add_column(sa.Column('scan_class', sa.String(20), nullable=True))
        batch_op.add_column(sa.Column('parent_scan_id', sa.String(36), nullable=True))
        batch_op.add_column(sa.Column('source_scan_ids', sa.JSON(), nullable=True))

    # --- Phase 2: Backfill scan_class ---
    conn = op.get_bind()
    rows = conn.execute(text("SELECT id, scan_name FROM scans")).fetchall()

    # Group by classification for batch updates
    by_class = {"primary": [], "sub_scan": [], "composite": []}
    for row_id, scan_name in rows:
        cls = _classify_scan_class(scan_name)
        by_class[cls].append(row_id)

    for cls, ids in by_class.items():
        if ids:
            # Batch in chunks of 500 to avoid SQLite variable limits
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                placeholders = ",".join(f"'{id_}'" for id_ in chunk)
                conn.execute(text(
                    f"UPDATE scans SET scan_class = '{cls}' WHERE id IN ({placeholders})"
                ))

    # --- Phase 3: Backfill parent_scan_id for sub-scans ---
    sub_scans = conn.execute(text(
        "SELECT id, scan_name, sol_number, target FROM scans WHERE scan_class = 'sub_scan'"
    )).fetchall()

    for sub_id, scan_name, sol_number, target in sub_scans:
        parent_name = _derive_parent_name(scan_name)
        if parent_name:
            # Find parent at same (sol_number, target) — use parameterized query
            result = conn.execute(text(
                "SELECT id FROM scans "
                "WHERE sol_number = :sol AND scan_name = :pname "
                "AND (target = :target OR (target IS NULL AND :target IS NULL)) "
                "LIMIT 1"
            ), {"sol": sol_number, "pname": parent_name, "target": target}).fetchone()
            if result:
                conn.execute(text(
                    "UPDATE scans SET parent_scan_id = :parent_id WHERE id = :sub_id"
                ), {"parent_id": result[0], "sub_id": sub_id})

    # --- Phase 4: Best-effort backfill source_scan_ids for composites ---
    composites = conn.execute(text(
        "SELECT id, scan_name, sol_number, target FROM scans WHERE scan_class = 'composite'"
    )).fetchall()

    for comp_id, scan_name, sol_number, target in composites:
        # Derive base prefix: strip _all, _median_all, _sum_active_*, etc.
        # e.g., detail_all -> detail, meteorite_median_all -> meteorite
        base = scan_name.lower()
        for suffix in ("_sum_active_sum_dark", "_sum_active_median_dark",
                        "_median_all", "_all"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break

        if not base:
            continue

        # Find primary sources at same (sol_number, target) with matching prefix
        sources = conn.execute(text(
            "SELECT id FROM scans "
            "WHERE sol_number = :sol "
            "AND (target = :target OR (target IS NULL AND :target IS NULL)) "
            "AND scan_class = 'primary' "
            "AND LOWER(scan_name) LIKE :prefix "
            "ORDER BY scan_name"
        ), {"sol": sol_number, "target": target, "prefix": base + "%"}).fetchall()

        if not sources:
            # Try sub_scans as fallback
            sources = conn.execute(text(
                "SELECT id FROM scans "
                "WHERE sol_number = :sol "
                "AND (target = :target OR (target IS NULL AND :target IS NULL)) "
                "AND scan_class = 'sub_scan' "
                "AND LOWER(scan_name) LIKE :prefix "
                "ORDER BY scan_name"
            ), {"sol": sol_number, "target": target, "prefix": base + "%"}).fetchall()

        if sources:
            source_ids = json.dumps([row[0] for row in sources])
            conn.execute(text(
                "UPDATE scans SET source_scan_ids = :source_ids WHERE id = :comp_id"
            ), {"source_ids": source_ids, "comp_id": comp_id})

    # --- Phase 5: Tighten constraints + index ---
    with op.batch_alter_table('scans') as batch_op:
        batch_op.alter_column('scan_class', existing_type=sa.String(20), nullable=False,
                              server_default='primary')
        batch_op.create_index('ix_scans_scan_class', ['scan_class'])
        batch_op.create_check_constraint(
            'ck_scans_scan_class',
            "scan_class IN ('primary', 'sub_scan', 'composite')"
        )
        batch_op.create_check_constraint(
            'ck_scans_class_fields',
            "(scan_class = 'primary' AND parent_scan_id IS NULL AND source_scan_ids IS NULL) OR "
            "(scan_class = 'sub_scan' AND source_scan_ids IS NULL) OR "
            "(scan_class = 'composite' AND parent_scan_id IS NULL)"
        )
        batch_op.create_foreign_key(
            'fk_scans_parent_scan_id', 'scans', ['parent_scan_id'], ['id'],
            ondelete='SET NULL'
        )

    # --- Post-migration assertions (skip on empty DB, e.g. test fixtures) ---
    if rows:
        counts = conn.execute(text(
            "SELECT scan_class, COUNT(*) FROM scans GROUP BY scan_class ORDER BY scan_class"
        )).fetchall()
        count_map = {row[0]: row[1] for row in counts}

        total = sum(count_map.values())
        assert total == len(rows), (
            f"Total scan count changed during migration: {len(rows)} -> {total}"
        )

        # Verify invariants
        violations = conn.execute(text(
            "SELECT COUNT(*) FROM scans WHERE scan_class = 'sub_scan' AND source_scan_ids IS NOT NULL"
        )).scalar()
        assert violations == 0, f"sub_scan rows with source_scan_ids: {violations}"

        violations = conn.execute(text(
            "SELECT COUNT(*) FROM scans WHERE scan_class = 'composite' AND parent_scan_id IS NOT NULL"
        )).scalar()
        assert violations == 0, f"composite rows with parent_scan_id: {violations}"

        violations = conn.execute(text(
            "SELECT COUNT(*) FROM scans WHERE scan_class = 'primary' "
            "AND (parent_scan_id IS NOT NULL OR source_scan_ids IS NOT NULL)"
        )).scalar()
        assert violations == 0, f"primary rows with linkage fields: {violations}"


def downgrade() -> None:
    """Remove scan_class, parent_scan_id, source_scan_ids columns."""
    with op.batch_alter_table('scans') as batch_op:
        batch_op.drop_constraint('ck_scans_class_fields', type_='check')
        batch_op.drop_constraint('ck_scans_scan_class', type_='check')
        batch_op.drop_constraint('fk_scans_parent_scan_id', type_='foreignkey')
        batch_op.drop_index('ix_scans_scan_class')
        batch_op.drop_column('source_scan_ids')
        batch_op.drop_column('parent_scan_id')
        batch_op.drop_column('scan_class')
