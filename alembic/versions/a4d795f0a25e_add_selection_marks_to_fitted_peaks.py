"""Add the selection marks to fitted_peaks.

Three nullable columns that record how the fitter's model selection treated
each persisted component:

- ``kept_by`` (text): which rule keeps the component, ``aicc`` or ``both``
  (the F-test pick has a component within 2 cm-1 that passes the same persist
  gates); ``ftest`` is reserved for components only the F-test keeps, which
  are not persisted yet.
- ``aicc_gain`` (real): AICc(k-1) - AICc(k) at the component's seed height rank k.
- ``step_p`` (real): the F-test p-value of that step.

They are records, never filters: no row is added, removed or changed by them.
NULL means not computed (older rows, fits with no size search, a needed size
that failed). Upgrade adds the columns (existing rows NULL); downgrade drops them.

Revision ID: a4d795f0a25e
Revises: 4f638701cf7c
Create Date: 2026-11-23 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4d795f0a25e"
down_revision: Union[str, Sequence[str], None] = "4f638701cf7c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the nullable kept_by, aicc_gain and step_p columns to fitted_peaks."""
    with op.batch_alter_table("fitted_peaks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("kept_by", sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column("aicc_gain", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("step_p", sa.Float(), nullable=True))


def downgrade() -> None:
    """Remove the selection marks from fitted_peaks."""
    with op.batch_alter_table("fitted_peaks", schema=None) as batch_op:
        batch_op.drop_column("step_p")
        batch_op.drop_column("aicc_gain")
        batch_op.drop_column("kept_by")

    # SQLite drops the triggers from 6cff8ea9cb06 when the batch rebuilds the table.
    if op.get_bind().dialect.name == "sqlite":
        for name, event in (("insert", "INSERT"), ("update", "UPDATE OF fit_modality")):
            op.execute(f"""CREATE TRIGGER check_fit_modality_{name}
        BEFORE {event} ON fitted_peaks
        BEGIN
            SELECT RAISE(ABORT, 'invalid fit_modality value')
            WHERE NEW.fit_modality NOT IN ('minerals', 'organics', 'hydration', 'fluorescence');
        END""")
