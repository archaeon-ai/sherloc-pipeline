"""Relocated feature-policy and serving modules keep the web boundary unchanged."""

from types import SimpleNamespace

import pytest

from sherloc_pipeline.models import feature_policy as policy_models
from sherloc_pipeline.services import reference_serving
from sherloc_pipeline.web import feature_policy as web_policy
from sherloc_pipeline.web import reference_storage

DIGEST = "a" * 64


@pytest.mark.parametrize("name", [
    "Digest", "PolicyModel", "PublicReleasePolicy", "ReferencePolicy", "MethodBinding",
    "IdentificationPolicy", "FeaturePolicy",
])
def test_web_policy_reexports_are_the_model_objects(name):
    assert getattr(web_policy, name) is getattr(policy_models, name)


@pytest.mark.parametrize("name", ["can_serve_trace", "can_serve_template"])
def test_web_serving_reexports_are_the_service_objects(name):
    assert getattr(reference_storage, name) is getattr(reference_serving, name)


def test_default_policy_stays_off(monkeypatch):
    for key in ("SHERLOC_FEATURE_POLICY", "SHERLOC_FEATURE_PDS_BROWSER"):
        monkeypatch.delenv(key, raising=False)
    for policy in (web_policy.FeaturePolicy(), web_policy.load_feature_policy("internal")):
        assert policy.references.mode == "off"
        assert not policy.identification.enabled and not policy.identification.experimental_candidates
        assert policy.public_release is None


def test_public_startup_still_refuses_references(monkeypatch):
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", (
        '{"references": {"mode": "all_approved", "release_id": "%s", "subset_id": "%s",'
        ' "serving_policy_version": "%s"}}' % (DIGEST, DIGEST, DIGEST)))
    with pytest.raises(ValueError, match="public references.mode must be off"):
        web_policy.load_feature_policy("public")


def synthetic_trace(**changes):
    values = dict(release_id=DIGEST, rights_known=True, serving_status="approved", permitted_tiers=["team"],
                  trace_revision_id="b" * 64, metadata_json={})
    values.update(changes)
    return SimpleNamespace(**values)


def test_serving_denies_by_default_and_outside_team():
    release = SimpleNamespace(release_id=DIGEST, status="ready", serving_policy_sha256=DIGEST)
    trace = synthetic_trace()
    template = SimpleNamespace(release_id=DIGEST, permitted_tiers=["team"], member_trace_ids=[trace.trace_revision_id])
    default = web_policy.FeaturePolicy()
    enabled = web_policy.FeaturePolicy.model_validate({"references": {
        "mode": "all_approved", "release_id": DIGEST, "subset_id": DIGEST, "serving_policy_version": DIGEST}})
    assert not reference_storage.can_serve_trace(trace, release, default, "team")
    assert not reference_storage.can_serve_template(template, [trace], release, default, "team")
    assert not reference_storage.can_serve_trace(trace, release, enabled, "public")
    assert not reference_storage.can_serve_template(template, [trace], release, enabled, "public")
    assert not reference_storage.can_serve_trace(synthetic_trace(rights_known=False), release, enabled, "team")
    # Invalid trace metadata fails closed through the unchanged known-rights check.
    assert not reference_storage.can_serve_trace(trace, release, enabled, "team")
