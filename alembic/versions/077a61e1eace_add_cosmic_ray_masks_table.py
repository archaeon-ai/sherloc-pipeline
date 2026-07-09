"""add_cosmic_ray_masks_table

Adds the ``cosmic_ray_masks`` satellite table that persists per-spectrum
cosmic-ray detection masks for the ML despike integration.
One row exists per
(DARK_SUBTRACTED spectrum, despike method) pair, carrying the absolute
flagged channel indices (JSON list, 0..2147 on the region's 2148-channel
plane) plus the provenance trio: the method
identity (e.g. ``ml_v1.1_tau_matched``), the model artifact sha256
digest, and the applied per-region threshold ``tau``.

The table is keyed to ``spectra.id`` with ``ON DELETE CASCADE`` so that
re-ingest (which wipes and regenerates a scan's derived data) cascades
through ``scans → scan_points → spectra → cosmic_ray_masks`` and leaves
no orphaned mask rows. A unique constraint on
(``spectrum_id``, ``method``) enforces one mask per method per spectrum
(idempotent delete+insert on re-run); masks are immutable, so the table
has **no** ``updated_at`` column.

Schema mirrors ``CosmicRayMaskORM`` in ``database/models.py``; no columns
are added or removed by this migration.

Revision ID: 077a61e1eace
Revises: 412fc1e3ee92
Create Date: 2026-06-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '077a61e1eace'
down_revision: Union[str, Sequence[str], None] = '412fc1e3ee92'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the ``cosmic_ray_masks`` table + spectrum_id index."""
    op.create_table(
        'cosmic_ray_masks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('spectrum_id', sa.String(length=36), nullable=False),
        sa.Column('method', sa.String(length=40), nullable=False),
        sa.Column('model_sha256', sa.String(length=64), nullable=False),
        sa.Column('tau', sa.Float(), nullable=False),
        sa.Column('channel_indices', sqlite.JSON(), nullable=False),
        sa.Column('n_flagged', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['spectrum_id'], ['spectra.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'spectrum_id', 'method', name='uq_cr_mask_spectrum_method'
        ),
    )
    with op.batch_alter_table('cosmic_ray_masks', schema=None) as batch_op:
        batch_op.create_index(
            'ix_cosmic_ray_masks_spectrum_id',
            ['spectrum_id'],
            unique=False,
        )


def downgrade() -> None:
    """Drop the ``cosmic_ray_masks`` table + spectrum_id index."""
    with op.batch_alter_table('cosmic_ray_masks', schema=None) as batch_op:
        batch_op.drop_index('ix_cosmic_ray_masks_spectrum_id')

    op.drop_table('cosmic_ray_masks')
