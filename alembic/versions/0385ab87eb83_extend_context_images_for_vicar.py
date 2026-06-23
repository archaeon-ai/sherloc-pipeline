"""extend_context_images_for_vicar

Extend context_images table with VICAR metadata columns for raw IMG ingestion.

This migration adds columns to support:
- File format distinction (IMG vs PNG)
- SCLK-based timing for scan linkage
- Camera identification
- Focus and exposure metadata
- Full VICAR header JSON blob

Revision ID: 0385ab87eb83
Revises: c8d2a36d1083
Create Date: 2026-01-24 00:11:16.856170

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite


# revision identifiers, used by Alembic.
revision: str = '0385ab87eb83'
down_revision: Union[str, Sequence[str], None] = 'c8d2a36d1083'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add VICAR metadata columns to context_images table."""
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        # File format distinction
        batch_op.add_column(
            sa.Column('file_format', sa.String(length=10), nullable=True)
        )
        # Camera identification
        batch_op.add_column(
            sa.Column('camera_id', sa.String(length=10), nullable=True)
        )
        # Sol number for direct indexing
        batch_op.add_column(
            sa.Column('sol_number', sa.Integer(), nullable=True)
        )
        # SCLK timing (critical for scan linkage)
        batch_op.add_column(
            sa.Column('sclk_start', sa.BigInteger(), nullable=True)
        )
        batch_op.add_column(
            sa.Column('sclk_stop', sa.BigInteger(), nullable=True)
        )
        # Sequence identification
        batch_op.add_column(
            sa.Column('sequence_id', sa.String(length=50), nullable=True)
        )
        # Image timing
        batch_op.add_column(
            sa.Column('image_time', sa.DateTime(), nullable=True)
        )
        # Focus parameters
        batch_op.add_column(
            sa.Column('focus_mode', sa.String(length=20), nullable=True)
        )
        batch_op.add_column(
            sa.Column('focus_position_count', sa.Integer(), nullable=True)
        )
        # Local Mars time
        batch_op.add_column(
            sa.Column('local_mean_solar_time', sa.String(length=50), nullable=True)
        )
        # Rover motion counter (JSON array)
        batch_op.add_column(
            sa.Column('rover_motion_counter', sa.Text(), nullable=True)
        )
        # Full VICAR metadata blob
        batch_op.add_column(
            sa.Column('vicar_metadata', sqlite.JSON(), nullable=True)
        )
        # Source IMG path (for PNG derived from IMG)
        batch_op.add_column(
            sa.Column('source_img_path', sa.Text(), nullable=True)
        )

        # Create indexes for efficient querying
        batch_op.create_index('ix_context_images_sol', ['sol_number'], unique=False)
        batch_op.create_index('ix_context_images_sclk', ['sclk_start'], unique=False)
        batch_op.create_index('ix_context_images_format', ['file_format'], unique=False)
        batch_op.create_index('ix_context_images_camera', ['camera_id'], unique=False)

    # Update existing records to have file_format='PNG'
    op.execute("UPDATE context_images SET file_format = 'PNG' WHERE file_path LIKE '%.PNG'")

    # Make scan_id nullable (orphan images may not link to a scan)
    # Note: SQLite doesn't support ALTER COLUMN, so we handle this with batch mode
    # The schema already has scan_id as NOT NULL with FK, but we need to keep it
    # for backward compatibility. New IMG files can have NULL scan_id initially.


def downgrade() -> None:
    """Remove VICAR metadata columns from context_images table."""
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        # Drop indexes first
        batch_op.drop_index('ix_context_images_camera')
        batch_op.drop_index('ix_context_images_format')
        batch_op.drop_index('ix_context_images_sclk')
        batch_op.drop_index('ix_context_images_sol')

        # Drop columns
        batch_op.drop_column('source_img_path')
        batch_op.drop_column('vicar_metadata')
        batch_op.drop_column('rover_motion_counter')
        batch_op.drop_column('local_mean_solar_time')
        batch_op.drop_column('focus_position_count')
        batch_op.drop_column('focus_mode')
        batch_op.drop_column('image_time')
        batch_op.drop_column('sequence_id')
        batch_op.drop_column('sclk_stop')
        batch_op.drop_column('sclk_start')
        batch_op.drop_column('sol_number')
        batch_op.drop_column('camera_id')
        batch_op.drop_column('file_format')
