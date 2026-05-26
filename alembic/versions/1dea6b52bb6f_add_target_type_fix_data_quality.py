"""add_target_type_fix_data_quality

Add target_type column to scans table with data quality fixes.

Phase 0: Fix leading spaces in target names, typos in scan_names.
Phase 1: Add nullable target_type column.
Phase 2: Backfill via SQL CASE (frozen rules, not calling Python).
Phase 3: Tighten to NOT NULL + CHECK constraint + index.

Revision ID: 1dea6b52bb6f
Revises: 6cff8ea9cb06
Create Date: 2026-02-11 07:54:11.173429

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1dea6b52bb6f'
down_revision: Union[str, Sequence[str], None] = '6cff8ea9cb06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add target_type column with data quality fixes and backfill.

    Phase 0: Data quality fixes (26 affected rows).
    Phase 1: Add nullable target_type column.
    Phase 2: Backfill via SQL CASE with priority cascade.
    Phase 3: Tighten constraint + CHECK + index.
    """
    # --- Phase 0: Data quality fixes ---
    # Fix leading/trailing spaces in target names (5 targets, 22 scans)
    op.execute("UPDATE scans SET target = TRIM(target) WHERE target != TRIM(target)")
    # Fix "meteroite" typo in scan_names (3 scans)
    op.execute(
        "UPDATE scans SET scan_name = REPLACE(scan_name, 'meteroite', 'meteorite') "
        "WHERE scan_name LIKE '%meteroite%'"
    )
    # Fix "Ortthofabric" typo in scan_names (1 scan)
    op.execute(
        "UPDATE scans SET scan_name = REPLACE(scan_name, 'Ortthofabric', 'Orthofabric') "
        "WHERE scan_name LIKE '%Ortthofabric%'"
    )

    # --- Phase 1: Add nullable column ---
    with op.batch_alter_table('scans') as batch_op:
        batch_op.add_column(sa.Column('target_type', sa.String(20), nullable=True))

    # --- Phase 2: Backfill via SQL CASE ---
    # Priority cascade: engineering > cal_target > mars_target.
    # Rules are frozen in SQL, matching classify_target_type() in models/spectra.py.
    op.execute("""
        UPDATE scans SET target_type = CASE
            -- Engineering: NULL/empty target
            WHEN target IS NULL OR TRIM(target) = '' THEN 'engineering'
            -- Engineering: known engineering targets
            WHEN LOWER(TRIM(target)) IN (
                'conjunction', 'b conjunction',
                'arm stowed', 'arm stowed dark', 'arm docked'
            ) THEN 'engineering'
            -- Engineering: power_* or *laser_disabled* scan_names
            WHEN LOWER(scan_name) LIKE 'power_%'
                OR LOWER(scan_name) LIKE '%laser_disabled%'
                THEN 'engineering'
            -- Calibration: known calibration targets
            WHEN LOWER(TRIM(target)) IN (
                'external calibration', 'teflon calibration', 'calibration',
                'algan340 calibration', 'maze calibration',
                'ext cal meteorite', 'passive diffusil'
            ) THEN 'cal_target'
            -- Calibration: AlGaN* scan_names
            WHEN LOWER(scan_name) LIKE 'algan%' THEN 'cal_target'
            -- Mars target: everything else
            ELSE 'mars_target'
        END
    """)

    # --- Phase 3: Tighten constraint + CHECK + index ---
    with op.batch_alter_table('scans') as batch_op:
        batch_op.alter_column('target_type', existing_type=sa.String(20), nullable=False)
        batch_op.create_index('ix_scans_target_type', ['target_type'])
        batch_op.create_check_constraint(
            'ck_scans_target_type',
            "target_type IN ('mars_target', 'cal_target', 'engineering')"
        )


def downgrade() -> None:
    """Remove target_type column. Data quality fixes are NOT reverted."""
    with op.batch_alter_table('scans') as batch_op:
        batch_op.drop_constraint('ck_scans_target_type', type_='check')
        batch_op.drop_index('ix_scans_target_type')
        batch_op.drop_column('target_type')
