"""add_map_mode_e1_tables

Add Map Mode E1 tables: map_display_coordinates, users, user_preferences,
classification_profiles, map_fit_cache.

Revision ID: 23b44fd37fd9
Revises: 1449d3a649cd
Create Date: 2026-03-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '23b44fd37fd9'
down_revision: Union[str, Sequence[str], None] = '1449d3a649cd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Map Mode E1 tables."""

    op.create_table(
        'map_display_coordinates',
        sa.Column('scan_point_id', sa.String(36),
                  sa.ForeignKey('scan_points.id', ondelete='CASCADE'),
                  primary_key=True),
        sa.Column('aci_x', sa.Float(), nullable=False),
        sa.Column('aci_y', sa.Float(), nullable=False),
        sa.Column('transform_method', sa.String(30), nullable=False),
        sa.Column('computed_at', sa.String(30), nullable=False),
    )

    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('email', sa.Text(), unique=True, nullable=False),
        sa.Column('display_name', sa.Text(), nullable=True),
        sa.Column('created_at', sa.Text(), nullable=False,
                  server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))")),
        sa.Column('last_seen_at', sa.Text(), nullable=False,
                  server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))")),
    )

    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(),
                  sa.ForeignKey('users.id'), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False,
                  server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))")),
        sa.UniqueConstraint('user_id', 'key', name='uq_user_preferences_user_key'),
    )

    op.create_table(
        'classification_profiles',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('user_id', sa.Integer(),
                  sa.ForeignKey('users.id'), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('profile_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
    )

    op.create_table(
        'map_fit_cache',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('scan_id', sa.Text(), nullable=False),
        sa.Column('domains', sa.Text(), nullable=False),
        sa.Column('point_subset', sa.Text(), nullable=True),
        sa.Column('profile_hash', sa.Text(), nullable=False),
        sa.Column('profile_name', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(),
                  sa.ForeignKey('users.id'), nullable=True),
        sa.Column('results_json', sa.Text(), nullable=False),
        sa.Column('n_points', sa.Integer(), nullable=False),
        sa.Column('n_detections_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('expires_at', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop Map Mode E1 tables in reverse FK dependency order."""
    op.drop_table('map_fit_cache')
    op.drop_table('classification_profiles')
    op.drop_table('user_preferences')
    op.drop_table('users')
    op.drop_table('map_display_coordinates')
