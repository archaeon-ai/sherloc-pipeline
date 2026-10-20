"""Tier-aware reference serving checks; shared by web adapters and services."""

from collections.abc import Callable

from sherloc_pipeline.database.reference_models import ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.feature_policy import FeaturePolicy
from sherloc_pipeline.models.references import ReferenceTrace, known_rights


def can_serve_trace(
    trace: ReferenceTraceORM, release: ReferenceReleaseORM, policy: FeaturePolicy, tier: str,
) -> bool:
    refs = policy.references
    if (
        tier != "team" or refs.mode == "off" or release.status != "ready"
        or refs.release_id != release.release_id or trace.release_id != release.release_id
        or refs.serving_policy_version != release.serving_policy_sha256
        or not trace.rights_known or trace.serving_status != "approved"
        or "team" not in trace.permitted_tiers
        or refs.mode == "allowlist" and trace.trace_revision_id not in refs.allowed_trace_ids
    ):
        return False
    try:
        return known_rights(ReferenceTrace.model_validate(trace.metadata_json))
    except ValueError:
        return False


def can_serve_template(
    template: ReferenceTemplateORM, members: list[ReferenceTraceORM],
    release: ReferenceReleaseORM, policy: FeaturePolicy, tier: str,
    trace_check: Callable[..., bool] = can_serve_trace,
) -> bool:
    return (
        template.release_id == release.release_id and "team" in template.permitted_tiers
        and bool(members)
        and [member.trace_revision_id for member in members] == template.member_trace_ids
        and all(trace_check(member, release, policy, tier) for member in members)
    )
