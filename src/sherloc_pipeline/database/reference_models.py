"""Reference catalogs are separate from mission scans and spectra."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, ForeignKeyConstraint, JSON, String, UniqueConstraint, event, inspect
from sqlalchemy.orm import Mapped, Session, mapped_column

from sherloc_pipeline.database.models import Base


class ReferenceReleaseORM(Base):
    __tablename__ = "reference_releases"
    __table_args__ = (CheckConstraint("status IN ('staged', 'ready', 'retired')"),)

    release_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_snapshot_sha256: Mapped[str] = mapped_column(String(64))
    metadata_sha256: Mapped[str] = mapped_column(String(64))
    array_index_sha256: Mapped[str] = mapped_column(String(64))
    serving_policy_sha256: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(40))
    source_name: Mapped[str] = mapped_column(String)
    source_base_url: Mapped[str | None] = mapped_column(String)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    importer_version: Mapped[str] = mapped_column(String)
    parser_version: Mapped[str] = mapped_column(String)
    qa_version: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String(20), default="staged", index=True, active_history=True)


class ReferenceFileORM(Base):
    __tablename__ = "reference_files"
    __table_args__ = (
        UniqueConstraint("release_id", "record_id", "file_id"),
        UniqueConstraint("release_id", "file_revision_id"),
    )

    file_revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("reference_releases.release_id"), index=True)
    datatype_id: Mapped[str] = mapped_column(String)
    record_id: Mapped[str] = mapped_column(String, index=True)
    file_id: Mapped[str] = mapped_column(String)
    raw_locator: Mapped[str] = mapped_column(String)
    raw_sha256: Mapped[str] = mapped_column(String(64))
    raw_size: Mapped[int]
    array_locator: Mapped[str] = mapped_column(String)
    array_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON)


class ReferenceTraceORM(Base):
    __tablename__ = "reference_traces"
    __table_args__ = (
        ForeignKeyConstraint(
            ["release_id", "file_revision_id"],
            ["reference_files.release_id", "reference_files.file_revision_id"],
        ),
        UniqueConstraint("file_revision_id", "trace_index"),
        CheckConstraint("trace_index >= 0 AND point_count > 0"),
        CheckConstraint("serving_status IN ('off', 'approved', 'excluded')"),
    )

    trace_revision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("reference_releases.release_id"), index=True)
    file_revision_id: Mapped[str] = mapped_column(String(64), index=True)
    trace_index: Mapped[int]
    source_key: Mapped[str] = mapped_column(String, index=True)
    point_count: Mapped[int]
    axis_sha256: Mapped[str] = mapped_column(String(64))
    intensity_sha256: Mapped[str] = mapped_column(String(64))
    rights_known: Mapped[bool]
    serving_status: Mapped[str] = mapped_column(String(20))
    permitted_tiers: Mapped[list] = mapped_column(JSON)
    exclusions: Mapped[list] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON)


class ReferenceTemplateORM(Base):
    __tablename__ = "reference_templates"
    __table_args__ = (UniqueConstraint("release_id", "name"),)

    template_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("reference_releases.release_id"), index=True)
    name: Mapped[str] = mapped_column(String)
    recipe_sha256: Mapped[str] = mapped_column(String(64))
    member_trace_ids: Mapped[list] = mapped_column(JSON)
    permitted_tiers: Mapped[list] = mapped_column(JSON)
    metadata_json: Mapped[dict] = mapped_column(JSON)


@event.listens_for(ReferenceFileORM, "before_update")
@event.listens_for(ReferenceTraceORM, "before_update")
@event.listens_for(ReferenceTemplateORM, "before_update")
def _immutable_revision(mapper, connection, target):
    if inspect(target).modified:
        raise ValueError("reference revisions are immutable")


@event.listens_for(ReferenceReleaseORM, "before_update")
def _release_transition(mapper, connection, target):
    state = inspect(target)
    if any(attr.history.has_changes() for attr in state.attrs if attr.key != "status"):
        raise ValueError("reference release metadata is immutable")
    history = state.attrs.status.history
    if history.has_changes() and (history.deleted[0], history.added[0]) not in {
        ("staged", "ready"), ("ready", "retired"),
    }:
        raise ValueError("invalid reference release transition")


@event.listens_for(ReferenceReleaseORM, "before_delete")
@event.listens_for(ReferenceFileORM, "before_delete")
@event.listens_for(ReferenceTraceORM, "before_delete")
@event.listens_for(ReferenceTemplateORM, "before_delete")
def _prevent_revision_delete(mapper, connection, target):
    raise ValueError("reference catalog rows cannot be deleted")


@event.listens_for(Session, "do_orm_execute")
def _prevent_bulk_revision_change(state):
    if (state.is_delete or state.is_update) and state.bind_mapper is not None and state.bind_mapper.class_ in {
        ReferenceReleaseORM, ReferenceFileORM, ReferenceTraceORM, ReferenceTemplateORM,
    }:
        if state.is_update:
            raise ValueError("reference catalog rows cannot be bulk updated")
        raise ValueError("reference catalog rows cannot be deleted")
