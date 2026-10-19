"""Strict, default-off reference and identification deployment policy models."""

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


class MethodBinding(PolicyModel):
    method_release_id: Digest
    reference_release_id: Digest
    reference_subset_id: Digest


class SourceDenylist(PolicyModel):
    sols: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)
    product_ids: list[Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,200}$")]] = Field(default_factory=list)
    target_names: list[Annotated[str, Field(min_length=1, pattern=r"[A-Za-z0-9]")]] = Field(default_factory=list)


class IdentificationPolicy(PolicyModel):
    source_denylist: SourceDenylist = Field(default_factory=SourceDenylist)
    enabled: bool = False
    experimental_candidates: bool = False
    allowed_method_releases: list[MethodBinding] = Field(default_factory=list)
    max_templates: Annotated[int, Field(ge=1, le=10000)] = 100
    max_selection_points: Annotated[int, Field(ge=1, le=1000)] = 1
    max_concurrency: Annotated[int, Field(ge=1, le=8)] = 1
    timeout_seconds: Annotated[int, Field(ge=1, le=120)] = 10


class FeaturePolicy(PolicyModel):
    evidence_desk: bool = False
    public_release: PublicReleasePolicy | None = None
    references: ReferencePolicy = Field(default_factory=ReferencePolicy)
    identification: IdentificationPolicy = Field(default_factory=IdentificationPolicy)

    @model_validator(mode="after")
    def validate_dependencies(self):
        refs, identify = self.references, self.identification
        if refs.mode != "off":
            if not refs.release_id or not refs.serving_policy_version or not refs.subset_id:
                raise ValueError("references require pinned release, subset and serving policy")
            if refs.mode == "allowlist" and not refs.allowed_trace_ids:
                raise ValueError("references.allowlist requires trace IDs")
        if len(refs.allowed_trace_ids) != len(set(refs.allowed_trace_ids)):
            raise ValueError("duplicate reference trace IDs")
        if identify.enabled:
            if refs.mode == "off":
                raise ValueError("identification requires references")
            if not identify.allowed_method_releases or any(
                binding.reference_release_id != refs.release_id
                or binding.reference_subset_id != refs.subset_id
                for binding in identify.allowed_method_releases
            ):
                raise ValueError("identification requires matching method/reference subset bindings")
        if identify.experimental_candidates and not identify.enabled:
            raise ValueError("experimental candidates require identification")
        return self
