"""Strict, default-off reference deployment policy models."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PublicReleasePolicy(PolicyModel):
    index_path: str = Field(min_length=1)
    index_sha256: Digest


class ReferencePolicy(PolicyModel):
    mode: Literal["off", "allowlist", "all_approved"] = "off"
    release_id: Digest | None = None
    subset_id: Digest | None = None
    allowed_trace_ids: list[Digest] = Field(default_factory=list)
    serving_policy_version: Digest | None = None
    storage_backend: Literal["local", "r2"] = "local"
    max_concurrency: Annotated[int, Field(ge=1, le=8)] = 1
    timeout_seconds: Annotated[int, Field(ge=1, le=120)] = 10


class FeaturePolicy(PolicyModel):
    public_release: PublicReleasePolicy | None = None
    references: ReferencePolicy = Field(default_factory=ReferencePolicy)

    @model_validator(mode="after")
    def validate_dependencies(self):
        refs = self.references
        if refs.mode != "off":
            if not refs.release_id or not refs.serving_policy_version or not refs.subset_id:
                raise ValueError("references require pinned release, subset and serving policy")
            if refs.mode == "allowlist" and not refs.allowed_trace_ids:
                raise ValueError("references.allowlist requires trace IDs")
        if len(refs.allowed_trace_ids) != len(set(refs.allowed_trace_ids)):
            raise ValueError("duplicate reference trace IDs")
        return self
