"""Add immutable reference release catalog tables.

Revision ID: a13c4e5f6071
Revises: 17db1a1940d6
"""

from alembic import op
import sqlalchemy as sa

revision = "a13c4e5f6071"
down_revision = "17db1a1940d6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "reference_releases",
        sa.Column("release_id", sa.String(64), primary_key=True),
        sa.Column("source_snapshot_sha256", sa.String(64), nullable=False),
        sa.Column("metadata_sha256", sa.String(64), nullable=False),
        sa.Column("array_index_sha256", sa.String(64), nullable=False),
        sa.Column("serving_policy_sha256", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(40), nullable=False),
        sa.Column("source_name", sa.String(), nullable=False),
        sa.Column("source_base_url", sa.String(), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("importer_version", sa.String(), nullable=False),
        sa.Column("parser_version", sa.String(), nullable=False),
        sa.Column("qa_version", sa.String(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.CheckConstraint("status IN ('staged', 'ready', 'retired')"),
    )
    op.create_index("ix_reference_releases_status", "reference_releases", ["status"])
    op.create_table(
        "reference_files",
        sa.Column("file_revision_id", sa.String(64), primary_key=True),
        sa.Column("release_id", sa.String(64), sa.ForeignKey("reference_releases.release_id"), nullable=False),
        sa.Column("datatype_id", sa.String(), nullable=False),
        sa.Column("record_id", sa.String(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("raw_locator", sa.String(), nullable=False),
        sa.Column("raw_sha256", sa.String(64), nullable=False),
        sa.Column("raw_size", sa.Integer(), nullable=False),
        sa.Column("array_locator", sa.String(), nullable=False),
        sa.Column("array_sha256", sa.String(64), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.UniqueConstraint("release_id", "record_id", "file_id"),
        sa.UniqueConstraint("release_id", "file_revision_id"),
    )
    op.create_index("ix_reference_files_release_id", "reference_files", ["release_id"])
    op.create_index("ix_reference_files_record_id", "reference_files", ["record_id"])
    op.create_table(
        "reference_traces",
        sa.Column("trace_revision_id", sa.String(64), primary_key=True),
        sa.Column("release_id", sa.String(64), sa.ForeignKey("reference_releases.release_id"), nullable=False),
        sa.Column("file_revision_id", sa.String(64), nullable=False),
        sa.Column("trace_index", sa.Integer(), nullable=False),
        sa.Column("source_key", sa.String(), nullable=False),
        sa.Column("point_count", sa.Integer(), nullable=False),
        sa.Column("axis_sha256", sa.String(64), nullable=False),
        sa.Column("intensity_sha256", sa.String(64), nullable=False),
        sa.Column("rights_known", sa.Boolean(), nullable=False),
        sa.Column("serving_status", sa.String(20), nullable=False),
        sa.Column("permitted_tiers", sa.JSON(), nullable=False),
        sa.Column("exclusions", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["release_id", "file_revision_id"],
                                ["reference_files.release_id", "reference_files.file_revision_id"]),
        sa.UniqueConstraint("file_revision_id", "trace_index"),
        sa.CheckConstraint("trace_index >= 0 AND point_count > 0"),
        sa.CheckConstraint("serving_status IN ('off', 'approved', 'excluded')"),
    )
    for column in ("release_id", "file_revision_id", "source_key"):
        op.create_index(f"ix_reference_traces_{column}", "reference_traces", [column])
    op.create_table(
        "reference_templates",
        sa.Column("template_id", sa.String(64), primary_key=True),
        sa.Column("release_id", sa.String(64), sa.ForeignKey("reference_releases.release_id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("recipe_sha256", sa.String(64), nullable=False),
        sa.Column("member_trace_ids", sa.JSON(), nullable=False),
        sa.Column("permitted_tiers", sa.JSON(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.UniqueConstraint("release_id", "name"),
    )
    op.create_index("ix_reference_templates_release_id", "reference_templates", ["release_id"])


def downgrade():
    for table in ("reference_templates", "reference_traces", "reference_files", "reference_releases"):
        op.drop_table(table)
