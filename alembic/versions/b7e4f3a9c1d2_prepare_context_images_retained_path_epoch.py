"""prepare the retained-path compatibility epoch

The canonical context-image locator is ``r2_rel_key``. During the bounded
rollout that precedes removal of the legacy ``file_path`` column, current
ingestion code must nevertheless be able to insert rows into a database that
still retains the old values for rollback and verification. The predecessor
schema declares ``file_path`` NOT NULL while current writers intentionally do
not populate it, so that combination is not write-compatible.

This additive compatibility migration preserves every existing value and only
relaxes the legacy column to nullable. The destructive ``17db1a1940d6``
migration remains a separate successor revision.

Revision ID: b7e4f3a9c1d2
Revises: 931df60632cb
Create Date: 2026-07-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7e4f3a9c1d2"
down_revision: Union[str, Sequence[str], None] = "931df60632cb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Retain legacy values while admitting r2_rel_key-only writers."""
    with op.batch_alter_table("context_images", schema=None) as batch_op:
        batch_op.alter_column(
            "file_path",
            existing_type=sa.Text(),
            existing_nullable=False,
            nullable=True,
        )


def downgrade() -> None:
    """Restore NOT NULL only when no post-upgrade NULL values exist."""
    bind = op.get_bind()
    null_rows = bind.execute(
        sa.text("SELECT COUNT(*) FROM context_images WHERE file_path IS NULL")
    ).scalar_one()
    if null_rows:
        raise RuntimeError(
            "cannot downgrade retained-path epoch: context_images.file_path "
            "contains NULL rows; restore the byte-exact predecessor snapshot"
        )
    with op.batch_alter_table("context_images", schema=None) as batch_op:
        batch_op.alter_column(
            "file_path",
            existing_type=sa.Text(),
            existing_nullable=True,
            nullable=False,
        )
