"""Add colorized dimension to map_display_coordinates cache.

Adds a ``colorized`` boolean column to ``map_display_coordinates`` and
promotes the primary key from ``(scan_point_id)`` to
``(scan_point_id, colorized)`` so the grayscale and colorized ACI variants
of a scan-point coordinate can be cached side by side (issue #8 — the
Map Mode / Workbench overlay must stay registered when the user toggles
the colorized ACI, which is a pure crop of the grayscale image with its
own re-solved per-point coordinates).

``map_display_coordinates`` is a pure cache: every row is recomputable on
demand by ``core.coordinates.resolve_display_coordinates`` from the Loupe
workspace ``spatial.csv`` / ``loupe.csv``. Adding a column to the primary
key on SQLite requires a table rebuild, so this migration drops and
recreates the table (the existing grayscale rows simply repopulate on the
next Map Mode access per scan). Recreating — rather than copy-migrating —
keeps the operation a single, idempotent DDL step with no fragile
batch-mode primary-key surgery.

Revision ID: 0c0107a1bed5
Revises: 077a61e1eace
Create Date: 2026-06-25 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0c0107a1bed5"
down_revision: Union[str, Sequence[str], None] = "077a61e1eace"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Recreate map_display_coordinates with a (scan_point_id, colorized) PK."""
    op.drop_table("map_display_coordinates")
    op.create_table(
        "map_display_coordinates",
        sa.Column(
            "scan_point_id",
            sa.String(36),
            sa.ForeignKey("scan_points.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "colorized",
            sa.Boolean(),
            nullable=False,
            # Dialect-portable false literal (renders 0 on SQLite, false on
            # Postgres) — avoids the integer-vs-boolean ambiguity of text("0").
            server_default=sa.false(),
            primary_key=True,
        ),
        sa.Column("aci_x", sa.Float(), nullable=False),
        sa.Column("aci_y", sa.Float(), nullable=False),
        sa.Column("transform_method", sa.String(30), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    """Restore the single-column (scan_point_id) primary key."""
    op.drop_table("map_display_coordinates")
    op.create_table(
        "map_display_coordinates",
        sa.Column(
            "scan_point_id",
            sa.String(36),
            sa.ForeignKey("scan_points.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("aci_x", sa.Float(), nullable=False),
        sa.Column("aci_y", sa.Float(), nullable=False),
        sa.Column("transform_method", sa.String(30), nullable=False),
        # Match the ORM (DateTime) — the upgrade recreates computed_at as
        # DateTime, so the downgrade does too to avoid schema drift.
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )
