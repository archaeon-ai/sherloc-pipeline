"""Add cr_mask_review to fitted_peaks.

A nullable boolean review flag on each fitted peak: True when one of the
fit's cosmic-ray mask channels lies within center +/- FWHM/2, False when
masks were checked and none does, NULL when the fit had no masks to check
(older rows, non-mineral domains, runs without ml despike). It is a review
flag, never a filter: no row is added, removed or changed by it.

Upgrade is a single ``ALTER TABLE ... ADD COLUMN`` (no table rebuild on
SQLite) and leaves every existing row NULL. Downgrade drops the column.

Revision ID: 4f638701cf7c
Revises: a13c4e5f6071
Create Date: 2026-11-12 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4f638701cf7c"
down_revision: Union[str, Sequence[str], None] = "a13c4e5f6071"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the nullable cr_mask_review flag to fitted_peaks."""
    with op.batch_alter_table("fitted_peaks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("cr_mask_review", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Remove the cr_mask_review flag from fitted_peaks."""
    with op.batch_alter_table("fitted_peaks", schema=None) as batch_op:
        batch_op.drop_column("cr_mask_review")
