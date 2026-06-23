"""merge map_mode and processing_state heads

Revision ID: 1ed5cea6c32d
Revises: 23b44fd37fd9, 73d700403ed9
Create Date: 2026-04-08 14:14:08.681071

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ed5cea6c32d'
down_revision: Union[str, Sequence[str], None] = ('23b44fd37fd9', '73d700403ed9')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
