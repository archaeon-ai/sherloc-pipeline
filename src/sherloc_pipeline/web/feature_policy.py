"""Strict, default-off reference and identification deployment policy."""

import json
import os
from pathlib import Path

# Policy models live in the models layer; re-exported here for web callers.
from sherloc_pipeline.models.feature_policy import (
    Digest,
    PolicyModel,
    PublicReleasePolicy,
    ReferencePolicy,
    MethodBinding,
    IdentificationPolicy,
    FeaturePolicy,
)

__all__ = [
    "Digest",
    "PolicyModel",
    "PublicReleasePolicy",
    "ReferencePolicy",
    "MethodBinding",
    "IdentificationPolicy",
    "FeaturePolicy",
    "load_feature_policy",
    "load_public_release_index",
]


def load_feature_policy(access_mode: str) -> FeaturePolicy:
    """Read one JSON setting; reject alternate/misspelled feature env keys."""
    for key in os.environ:
        if key.startswith(("SHERLOC_FEATURE_", "SHERLOC_REFERENCES", "SHERLOC_IDENTIFICATION")):
            if key not in {"SHERLOC_FEATURE_POLICY", "SHERLOC_FEATURE_PDS_BROWSER"}:
                raise ValueError(f"unknown feature setting: {key}")
    values = json.loads(os.environ.get("SHERLOC_FEATURE_POLICY", "{}"))
    if (access_mode == "public" and isinstance(values, dict)
            and "public_release" in values and values["public_release"] is None):
        raise ValueError("public_release requires index_path and index_sha256")
    policy = FeaturePolicy.model_validate(values)
    if access_mode == "public" and policy.references.mode != "off":
        raise ValueError("public references.mode must be off")
    return policy


def load_public_release_index(access_mode: str, policy: FeaturePolicy):
    """Load a pinned local index only when public startup opts in."""
    if access_mode != "public" or policy.public_release is None:
        return None
    from sherloc_pipeline.services.public_release import PublicReleaseIndex

    try:
        return PublicReleaseIndex(
            Path(policy.public_release.index_path), policy.public_release.index_sha256,
        )
    except OSError as exc:
        raise ValueError("public release index unavailable") from exc
