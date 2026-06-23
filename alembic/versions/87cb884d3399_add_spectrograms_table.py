"""add_spectrograms_table

Adds the ``spectrograms`` table that was previously declared in the ORM
(``SpectrogramORM`` in ``database/models.py``) but never represented in
the Alembic migration chain. Production databases were bootstrapped via
``create_all_tables()`` so they have the table; Alembic-only test
fixtures (``init_pds_database`` against a fresh tmp DB) do not, which
caused integration tests in ``tests/integration/test_pds_*`` to fail
when ``force=True`` re-ingestion lazy-loaded ``ScanORM.spectrograms``.

This migration brings the Alembic chain back in sync with the ORM by
creating the table and its two indexes (``ix_spectrograms_scan_id`` and
``ix_spectrograms_scan_region``). FK to ``scans.id`` uses
``ON DELETE CASCADE``.

Schema is taken verbatim from ``SpectrogramORM``; no columns are added
or removed by this migration.

Revision ID: 87cb884d3399
Revises: 1ed5cea6c32d
Create Date: 2026-04-28 21:20:50.742949

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '87cb884d3399'
down_revision: Union[str, Sequence[str], None] = '1ed5cea6c32d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the ``spectrograms`` table + two indexes."""
    op.create_table(
        'spectrograms',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('scan_id', sa.String(length=36), nullable=False),
        sa.Column('region', sa.String(length=10), nullable=False),
        sa.Column('processing_level', sa.String(length=20), nullable=False),
        sa.Column('config', sqlite.JSON(), nullable=False),
        sa.Column('intensity_matrix', sa.LargeBinary(), nullable=False),
        sa.Column('n_points', sa.Integer(), nullable=False),
        sa.Column('n_channels', sa.Integer(), nullable=False),
        sa.Column('wavenumber_min', sa.Float(), nullable=False),
        sa.Column('wavenumber_max', sa.Float(), nullable=False),
        sa.Column('wavenumbers', sa.LargeBinary(), nullable=True),
        sa.Column('point_labels', sqlite.JSON(), nullable=True),
        sa.Column('point_indices', sqlite.JSON(), nullable=True),
        sa.Column('intensity_min', sa.Float(), nullable=True),
        sa.Column('intensity_max', sa.Float(), nullable=True),
        sa.Column('title', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['scan_id'], ['scans.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('spectrograms', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_spectrograms_scan_id'),
            ['scan_id'],
            unique=False,
        )
        batch_op.create_index(
            'ix_spectrograms_scan_region',
            ['scan_id', 'region'],
            unique=False,
        )


def downgrade() -> None:
    """Drop the ``spectrograms`` table + two indexes."""
    with op.batch_alter_table('spectrograms', schema=None) as batch_op:
        batch_op.drop_index('ix_spectrograms_scan_region')
        batch_op.drop_index(batch_op.f('ix_spectrograms_scan_id'))

    op.drop_table('spectrograms')
