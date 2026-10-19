"""Tier-aware reference read adapter; no HTTP reference routes."""

from collections.abc import Callable

from sherloc_pipeline.database.reference_models import ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.references import ReferenceTrace, known_rights
from sherloc_pipeline.services.reference_store import object_locator, verify_bytes
from sherloc_pipeline.web.feature_policy import FeaturePolicy


class R2ReferenceStore:
    def __init__(self, tier: str, reader: Callable[[str], bytes] | None = None):
        if tier != "team":
            raise ValueError("public instances cannot read reference objects")
        self.tier = tier
        self.reader = reader

    def read_verified(self, locator: str, sha256: str) -> bytes:
        if locator != object_locator(sha256):
            raise ValueError("invalid reference locator")
        if self.reader is not None:
            data = self.reader(locator)
        else:
            from sherloc_pipeline.web.r2_reader import get_r2_client_and_config, r2_get_bytes

            _, config = get_r2_client_and_config()
            if config["tier"] != self.tier:
                raise ValueError("reference storage tier mismatch")
            data = r2_get_bytes(locator)
        return verify_bytes(data, sha256)


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
) -> bool:
    return (
        template.release_id == release.release_id and "team" in template.permitted_tiers
        and bool(members)
        and [member.trace_revision_id for member in members] == template.member_trace_ids
        and all(can_serve_trace(member, release, policy, tier) for member in members)
    )
