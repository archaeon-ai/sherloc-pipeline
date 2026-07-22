"""drop file_path column from context_images

Drops the transitional ``context_images.file_path`` column (issue #7).

``r2_rel_key`` (added in migration ``931df60632cb``) has been the sole
read path for ACI object identity since that migration landed — the
serve path (``web/routes/images.py``, ``web/routes/pds.py``), the
processing-side disk resolver (``core/r2_keys.resolve_disk_path``), and
ingestion (``services/image_ingestion.py``, ``services/pds_ingestion.py``,
``services/ingestion.py``) all write and read ``r2_rel_key`` exclusively.
``file_path`` had become write-only: still populated on ingestion for
old-code rollback, never read.

Both serving databases are fully backfilled (team 1107/1107, public
1529/1529 including the 5 ``pds:`` sentinel rows — verified prior to
this migration), so the column carries no information ``r2_rel_key``
lacks. Dropping it retires the last of the machine-specific
absolute-path bookkeeping described in ``core/r2_keys.py``'s module
docstring.

SQLite does not support ``DROP COLUMN`` directly (it requires a
table-rebuild), so this uses ``op.batch_alter_table`` — the standard
Alembic SQLite pattern, consistent with the ``downgrade()`` in
``931df60632cb``.

``downgrade()`` re-adds ``file_path`` as **nullable** Text. The absolute
paths cannot be reconstructed from ``r2_rel_key`` alone (the deployment
root that was stripped at ingestion time is not recoverable from the
locator), so a downgrade produces an empty (NULL) column rather than
lying about data that no longer exists — the honest reverse of an
irreversible drop.

Revision ID: 17db1a1940d6
Revises: b7e4f3a9c1d2
Create Date: 2026-07-10 13:14:35.625861

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '17db1a1940d6'
down_revision: Union[str, Sequence[str], None] = 'b7e4f3a9c1d2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop the transitional context_images.file_path column."""
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        batch_op.drop_column('file_path')


def downgrade() -> None:
    """Re-add file_path as nullable Text (data is not recoverable)."""
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        batch_op.add_column(sa.Column('file_path', sa.Text(), nullable=True))
