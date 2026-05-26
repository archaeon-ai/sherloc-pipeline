"""add user.sub column for Auth0 identity (B.12 F4)

Per architecture spec §13.1, ``email`` is an OPTIONAL claim. Auth0 access
tokens commonly include ``sub`` and the namespaced roles claim, but NOT
``email`` (which lives on the ID token, not the access token). The
B.7+B.8 backend keyed user records on ``email``, so a valid role-bearing
access token without ``email`` would fail user routes — flagged by Codex
review of PR #6 as F4 / Major.

This migration adds a ``sub`` column to ``users`` (the stable identity
key from JWT). Existing rows are backfilled with ``sub = email`` so
nothing breaks for the cf-access path (where CFAccessValidator already
sets ``claims.sub = email``). Going forward:

  * Auth0 mode keys on ``claims.sub`` (e.g., ``auth0|abc123``)
  * CF Access mode keys on ``claims.sub`` (which equals the email)
  * Dev mode keys on ``DevValidator.DEV_SUB`` (``localhost-dev``)
  * ``email`` becomes optional metadata; populated when the validator
    surfaces it (CF Access always does; Auth0 only if the namespaced
    email claim is added to the Action — see §13.0.6)

SQLite-specific: ``users.email`` was UNIQUE NOT NULL. The unique
constraint stays for now (operator data is small; collisions would
indicate a real problem) but the not-null constraint is dropped via
batch_alter_table.

Revision ID: 412fc1e3ee92
Revises: 87cb884d3399
Create Date: 2026-05-03 21:09:19.647029
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '412fc1e3ee92'
down_revision: Union[str, Sequence[str], None] = '87cb884d3399'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add users.sub (UNIQUE NOT NULL); backfill sub = email."""
    # Step 1: add sub as nullable so we can populate it.
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("sub", sa.Text(), nullable=True))

    # Step 2: backfill existing rows. CF Access mode set claims.sub to
    # the email (see web/auth.py:CFAccessValidator); preserving that
    # invariant for legacy rows means user lookups continue to resolve
    # the same record.
    op.execute("UPDATE users SET sub = email WHERE sub IS NULL")

    # Step 3: enforce NOT NULL + UNIQUE on the populated column,
    # and drop NOT NULL on email (now optional metadata).
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column("sub", existing_type=sa.Text(), nullable=False)
        batch_op.create_unique_constraint("uq_users_sub", ["sub"])
        batch_op.alter_column("email", existing_type=sa.Text(), nullable=True)


def downgrade() -> None:
    """Remove users.sub; restore email NOT NULL."""
    # Note: rows where email is NULL would block this downgrade. Operator
    # must clear those (or backfill) before downgrading.
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_constraint("uq_users_sub", type_="unique")
        batch_op.drop_column("sub")
        batch_op.alter_column("email", existing_type=sa.Text(), nullable=False)
