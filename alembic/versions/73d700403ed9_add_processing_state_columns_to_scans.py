"""add processing state columns to scans

Add processing_status, processed_at, processing_config_hash,
processing_pipeline_version, and processing_error columns to the scans table.

State semantics for processing_status:
  NULL        = unprocessed (never run)
  "completed" = successfully processed
  "failed"    = pipeline errored

Revision ID: 73d700403ed9
Revises: 1449d3a649cd
Create Date: 2026-03-19 19:11:12.344977

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '73d700403ed9'
down_revision: Union[str, Sequence[str], None] = '1449d3a649cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add processing state columns to scans table."""
    with op.batch_alter_table('scans', schema=None) as batch_op:
        batch_op.add_column(sa.Column('processing_status', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('processed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('processing_config_hash', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('processing_pipeline_version', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('processing_error', sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove processing state columns from scans table."""
    with op.batch_alter_table('scans', schema=None) as batch_op:
        batch_op.drop_column('processing_error')
        batch_op.drop_column('processing_pipeline_version')
        batch_op.drop_column('processing_config_hash')
        batch_op.drop_column('processed_at')
        batch_op.drop_column('processing_status')
