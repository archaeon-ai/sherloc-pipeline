"""add_pds_columns_backfill_loupe_defaults

Add 7 new PDS-related columns across 4 tables and backfill existing Loupe data
with appropriate defaults per spec s13. Also changes scans.shots_per_point from
NOT NULL to nullable (PDS processed products lack this field).

New columns:
  scans:          data_source, site_drive, sequence_id, scan_type
  scan_points:    coordinate_frame
  spectra:        wavelength_source
  context_images: pds_lidvid

Backfill rules (existing Loupe data):
  scans.data_source       = 'loupe'
  scans.scan_type         = derived: calibration if AlGaN/algan in scan_name,
                            survey if n_points > 200, detail otherwise
  scan_points.coord_frame = 'scanner_workspace'
  spectra.wavelength_src  = 'loupe_polynomial'
  Others                  = NULL (not available in Loupe metadata)

Revision ID: cd1b1897542e
Revises: 172d8c59b5c9
Create Date: 2026-01-26 22:30:39.397583

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cd1b1897542e'
down_revision: Union[str, Sequence[str], None] = '172d8c59b5c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add PDS columns and backfill Loupe defaults."""

    # --- scans table ---
    # Use batch mode because we also need to change shots_per_point nullability.
    # batch_alter_table recreates the table in SQLite.
    with op.batch_alter_table('scans', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('data_source', sa.String(length=20), nullable=True)
        )
        batch_op.add_column(
            sa.Column('site_drive', sa.String(length=20), nullable=True)
        )
        batch_op.add_column(
            sa.Column('sequence_id', sa.String(length=20), nullable=True)
        )
        batch_op.add_column(
            sa.Column('scan_type', sa.String(length=20), nullable=True)
        )
        # Change shots_per_point from NOT NULL to nullable
        batch_op.alter_column(
            'shots_per_point',
            existing_type=sa.Integer(),
            nullable=True,
        )

    # Backfill scans.data_source for all existing rows
    op.execute("UPDATE scans SET data_source = 'loupe' WHERE data_source IS NULL")

    # Backfill scans.scan_type using deterministic rules from spec s13
    op.execute("""
        UPDATE scans SET scan_type = CASE
            WHEN scan_name LIKE '%AlGaN%' OR scan_name LIKE '%algan%'
                THEN 'calibration'
            WHEN n_points > 200
                THEN 'survey'
            ELSE 'detail'
        END
        WHERE scan_type IS NULL
    """)

    # --- scan_points table ---
    op.execute(
        "ALTER TABLE scan_points ADD COLUMN coordinate_frame VARCHAR(30)"
    )
    op.execute(
        "UPDATE scan_points SET coordinate_frame = 'scanner_workspace' "
        "WHERE coordinate_frame IS NULL"
    )

    # --- spectra table ---
    op.execute(
        "ALTER TABLE spectra ADD COLUMN wavelength_source VARCHAR(30)"
    )
    op.execute(
        "UPDATE spectra SET wavelength_source = 'loupe_polynomial' "
        "WHERE wavelength_source IS NULL"
    )

    # --- context_images table ---
    op.execute(
        "ALTER TABLE context_images ADD COLUMN pds_lidvid VARCHAR(200)"
    )
    # pds_lidvid stays NULL for Loupe-sourced images (no backfill needed)


def downgrade() -> None:
    """Remove PDS columns and revert shots_per_point to NOT NULL."""

    # --- context_images: drop pds_lidvid ---
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        batch_op.drop_column('pds_lidvid')

    # --- spectra: drop wavelength_source ---
    with op.batch_alter_table('spectra', schema=None) as batch_op:
        batch_op.drop_column('wavelength_source')

    # --- scan_points: drop coordinate_frame ---
    with op.batch_alter_table('scan_points', schema=None) as batch_op:
        batch_op.drop_column('coordinate_frame')

    # --- scans: drop PDS columns, revert shots_per_point to NOT NULL ---
    with op.batch_alter_table('scans', schema=None) as batch_op:
        batch_op.drop_column('scan_type')
        batch_op.drop_column('sequence_id')
        batch_op.drop_column('site_drive')
        batch_op.drop_column('data_source')
        batch_op.alter_column(
            'shots_per_point',
            existing_type=sa.Integer(),
            nullable=False,
        )
