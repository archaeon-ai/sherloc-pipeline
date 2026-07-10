"""add r2_rel_key locator to context_images

Adds the relative-locator column and backfills it from existing
``file_path`` values — the one-time, frozen copy of the absolute-path →
R2-key strip translation that ``core/r2_keys.py`` performed at serve
time through v5.3.x. After this migration the serve path derives R2 keys
by concatenation (``sherloc-aci/`` + locator); the strip-prefix table,
``PHASE_*_STRIP_PREFIX`` env fallbacks, legacy-ingestion aliases, and
tier inference are deleted from live code.

Backfill rules (idempotent — every UPDATE is guarded by
``r2_rel_key IS NULL``):

  1. ``pds:<lidvid>`` sentinel rows copy through unchanged (the locator
     round-trips the scheme until the download step resolves them).
  2. Absolute-path rows matching a known ingestion layout have that
     prefix stripped. The prefixes are the same set the retired runtime
     translation accepted: the canonical per-tier roots and the known
     legacy team alias, still overridable/extendable via the (retired)
     ``PHASE_{TEAM,PUBLIC}_STRIP_PREFIX`` / ``_LEGACY_STRIP_ALIASES``
     env vars so a deployment that relied on the override knobs gets an
     identical translation. These literals are FROZEN HISTORICAL
     INGESTION LAYOUTS, not live configuration.
  3. Rows matching nothing stay NULL and are logged; they could not be
     served before this migration (strip-prefix mismatch → 500) and
     fail identically after it (missing locator → 500).

Revision ID: 931df60632cb
Revises: 9b2e7c4a1f08
Create Date: 2026-07-09 22:45:56.672710

"""
import logging
import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '931df60632cb'
down_revision: Union[str, Sequence[str], None] = '9b2e7c4a1f08'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

# Frozen historical ingestion layouts (see module docstring). These are
# not live configuration — live code no longer carries any deployment
# filesystem topology.
_HISTORICAL_TEAM_STRIP = "/data/sherloc/data/"
_HISTORICAL_PUBLIC_STRIP = "/data/sherloc/pds/"
_HISTORICAL_TEAM_LEGACY_ALIASES = ("/nas/000_sherloc/data/",)


def _backfill_prefixes() -> list[str]:
    """The exact prefix set the retired runtime translation accepted."""
    prefixes = [
        os.environ.get("PHASE_TEAM_STRIP_PREFIX") or _HISTORICAL_TEAM_STRIP,
        os.environ.get("PHASE_PUBLIC_STRIP_PREFIX") or _HISTORICAL_PUBLIC_STRIP,
        *_HISTORICAL_TEAM_LEGACY_ALIASES,
    ]
    for env in ("PHASE_TEAM_LEGACY_STRIP_ALIASES",
                "PHASE_PUBLIC_LEGACY_STRIP_ALIASES"):
        prefixes.extend(p for p in os.environ.get(env, "").split(":") if p)
    # De-duplicate, longest first so no shorter prefix shadows a longer one.
    return sorted(set(prefixes), key=len, reverse=True)


def upgrade() -> None:
    """Add r2_rel_key and backfill it from file_path."""
    bind = op.get_bind()

    # Retry-safe DDL: if a prior partially-applied run added the column
    # but did not stamp the revision (the known swap/stamp hazard), skip
    # the ALTER instead of failing with "duplicate column name".
    existing_cols = [
        row[1]
        for row in bind.exec_driver_sql("PRAGMA table_info(context_images)")
    ]
    if "r2_rel_key" not in existing_cols:
        op.execute("ALTER TABLE context_images ADD COLUMN r2_rel_key TEXT")

    # Rule 1: pds: sentinel rows round-trip.
    bind.execute(sa.text(
        "UPDATE context_images SET r2_rel_key = file_path "
        "WHERE r2_rel_key IS NULL AND file_path LIKE 'pds:%'"
    ))

    # Rule 2: strip known ingestion-layout prefixes. The traversal guard
    # mirrors core.r2_keys._validate_key: no '..', no backslash, and the
    # first post-prefix character must not be '/'.
    for prefix in _backfill_prefixes():
        bind.execute(
            sa.text(
                "UPDATE context_images "
                "SET r2_rel_key = substr(file_path, :cut) "
                "WHERE r2_rel_key IS NULL "
                "  AND substr(file_path, 1, :plen) = :prefix "
                "  AND file_path NOT LIKE '%..%' "
                "  AND instr(file_path, :backslash) = 0 "
                "  AND substr(file_path, :cut, 1) != '/'"
            ),
            {
                "prefix": prefix,
                "plen": len(prefix),
                "cut": len(prefix) + 1,
                "backslash": "\\",
            },
        )

    # Rule 3: report what stayed NULL (serve-path behavior unchanged for
    # these rows: they 500'd before and 500 after).
    remaining = bind.execute(sa.text(
        "SELECT COUNT(*) FROM context_images WHERE r2_rel_key IS NULL"
    )).scalar()
    if remaining:
        logger.warning(
            "r2_rel_key backfill left %d context_images row(s) NULL "
            "(file_path matched no known ingestion layout); these rows "
            "were not servable before this migration either.",
            remaining,
        )


def downgrade() -> None:
    """Drop r2_rel_key."""
    with op.batch_alter_table('context_images', schema=None) as batch_op:
        batch_op.drop_column('r2_rel_key')
