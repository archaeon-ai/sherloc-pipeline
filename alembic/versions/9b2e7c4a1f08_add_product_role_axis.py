"""add_product_role_axis

Add the analytical ``product_role`` axis to the ``scans`` table (WS-1 spec
§4.4 — SHERLOC scan classification).

``product_role ∈ {raw, canonical, alternate}`` (NULL for every non-multishot
scan ≈ the entire corpus) separates the *analytical role* of a multishot
acquisition's co-spatial products (raw N×k shots, ``*_median_all`` reduction,
``*_sum_active_median_dark`` canonical reduction) from the structural
``scan_class`` topology. The live source CHECK forbids a ``primary`` from
carrying ``source_scan_ids``, so a counted-standalone-and-lineage-linked
reduction cannot be encoded in ``scan_class`` alone — hence a dedicated axis
(Key Decision K3).

A single nullable column + a CHECK; **no new table**; forward-safe per
ARC-SHR-SYS-031. Values are assigned by ``sherloc reclassify-product-roles``
(corpus-level — the raw role and source linkage require the sibling raw
scan), NOT at this migration; every existing row is left NULL, so the new
CHECK is trivially satisfied on upgrade.

Revision ID: 9b2e7c4a1f08
Revises: 0c0107a1bed5
Create Date: 2026-06-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "9b2e7c4a1f08"
down_revision: Union[str, Sequence[str], None] = "0c0107a1bed5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# product_role governance CHECK (WS-1 spec §4.4). A single constraint enforces
# enum membership AND the role ⇒ class/parent/sources couplings:
#   - NULL                    : any non-multishot scan (≈ the whole corpus)
#   - raw                     : scan_class='primary',  parent NULL, sources NULL
#   - canonical / alternate   : scan_class='composite', parent NULL,
#                               source_scan_ids a non-empty JSON array
#
# The explicit ``source_scan_ids IS NOT NULL`` guard is load-bearing:
# ``json_array_length(NULL)`` is NULL, and ``NULL >= 1`` is NULL, which a
# CHECK treats as "pass" (three-valued logic) — so without the guard a
# canonical/alternate row with NULL sources would slip through.
_PRODUCT_ROLE_CHECK = (
    "product_role IS NULL OR "
    "(product_role = 'raw' AND scan_class = 'primary' "
    "AND parent_scan_id IS NULL AND source_scan_ids IS NULL) OR "
    "(product_role IN ('canonical', 'alternate') AND scan_class = 'composite' "
    "AND parent_scan_id IS NULL AND source_scan_ids IS NOT NULL "
    "AND json_array_length(source_scan_ids) >= 1)"
)


def upgrade() -> None:
    """Add the nullable product_role column + its governance CHECK."""
    # Adding a nullable column on SQLite is a native ALTER (no table rebuild),
    # so existing constraints/indexes/FKs are untouched here.
    with op.batch_alter_table("scans") as batch_op:
        batch_op.add_column(sa.Column("product_role", sa.String(20), nullable=True))

    # Adding a CHECK requires a batch table rebuild on SQLite. Batch mode
    # reflects and re-emits the existing constraints (ck_scans_scan_class,
    # ck_scans_class_fields, the parent-scan FK, indexes); the migration test
    # suite asserts all of them survive the rebuild.
    with op.batch_alter_table("scans") as batch_op:
        batch_op.create_check_constraint("ck_scans_product_role", _PRODUCT_ROLE_CHECK)

    # Post-migration assertion (skipped on an empty DB, e.g. test fixtures):
    # every existing row is NULL product_role, so the new CHECK holds.
    conn = op.get_bind()
    bad = conn.execute(
        text("SELECT COUNT(*) FROM scans WHERE product_role IS NOT NULL")
    ).scalar()
    assert bad == 0, f"product_role unexpectedly populated pre-reclassify: {bad}"


def downgrade() -> None:
    """Drop the product_role CHECK + column."""
    with op.batch_alter_table("scans") as batch_op:
        batch_op.drop_constraint("ck_scans_product_role", type_="check")
        batch_op.drop_column("product_role")
