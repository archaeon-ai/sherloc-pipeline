"""Add the nullable hydration width mark to fitted_peaks."""
from alembic import op
import sqlalchemy as sa

revision = "dba3f925e841"
down_revision = "a4d795f0a25e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("fitted_peaks", sa.Column("fwhm_at_cap", sa.Boolean(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("fitted_peaks") as batch_op:
        batch_op.drop_column("fwhm_at_cap")

    # SQLite drops the triggers from 6cff8ea9cb06 when the batch rebuilds the table.
    if op.get_bind().dialect.name == "sqlite":
        for name, event in (("insert", "INSERT"), ("update", "UPDATE OF fit_modality")):
            op.execute(f"""CREATE TRIGGER check_fit_modality_{name}
        BEFORE {event} ON fitted_peaks
        BEGIN
            SELECT RAISE(ABORT, 'invalid fit_modality value')
            WHERE NEW.fit_modality NOT IN ('minerals', 'organics', 'hydration', 'fluorescence');
        END""")
