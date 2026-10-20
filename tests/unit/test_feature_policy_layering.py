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


KNOWN_RIGHTS = {
    "trace_index": 0, "source_label": "synthetic measurement", "x_key": "x", "y_key": "y",
    "modality": "raman", "x_quantity": "raman_shift", "x_unit": "cm-1", "specimen_key": "synthetic-specimen",
    "license_identifier": "synthetic-license", "license_source": "synthetic rights record", "rights_status": "known",
}


def synthetic_trace(**changes):
    values = dict(release_id=DIGEST, rights_known=True, serving_status="approved", permitted_tiers=["team"],
                  trace_revision_id="b" * 64, metadata_json=dict(KNOWN_RIGHTS))
    values.update(changes)
    return SimpleNamespace(**values)


def serving_policy(mode="all_approved", allowed=()):
    return web_policy.FeaturePolicy.model_validate({"references": {
        "mode": mode, "release_id": DIGEST, "subset_id": DIGEST, "serving_policy_version": DIGEST,
        "allowed_trace_ids": list(allowed)}})


RELEASE = SimpleNamespace(release_id=DIGEST, status="ready", serving_policy_sha256=DIGEST)


@pytest.mark.parametrize("policy", [serving_policy(), serving_policy("allowlist", ["b" * 64])])
def test_team_serves_known_rights_trace_and_template(policy):
    trace = synthetic_trace()
    template = SimpleNamespace(release_id=DIGEST, permitted_tiers=["team"], member_trace_ids=[trace.trace_revision_id])
    assert reference_storage.can_serve_trace(trace, RELEASE, policy, "team")
    assert reference_storage.can_serve_template(template, [trace], RELEASE, policy, "team")


def test_serving_denials_unchanged():
    trace = synthetic_trace()
    template = SimpleNamespace(release_id=DIGEST, permitted_tiers=["team"], member_trace_ids=[trace.trace_revision_id])
    enabled = serving_policy()
    assert not reference_storage.can_serve_trace(trace, RELEASE, web_policy.FeaturePolicy(), "team")
    assert not reference_storage.can_serve_template(template, [trace], RELEASE, web_policy.FeaturePolicy(), "team")
    assert not reference_storage.can_serve_trace(trace, RELEASE, enabled, "public")
    assert not reference_storage.can_serve_template(template, [trace], RELEASE, enabled, "public")
    assert not reference_storage.can_serve_trace(synthetic_trace(rights_known=False), RELEASE, enabled, "team")
    unknown = dict(KNOWN_RIGHTS, rights_status="unknown")
    assert not reference_storage.can_serve_trace(synthetic_trace(metadata_json=unknown), RELEASE, enabled, "team")
    assert not reference_storage.can_serve_trace(synthetic_trace(metadata_json={}), RELEASE, enabled, "team")
    assert not reference_storage.can_serve_trace(trace, RELEASE, serving_policy("allowlist", ["c" * 64]), "team")
    other = synthetic_trace(trace_revision_id="d" * 64)
    assert not reference_storage.can_serve_template(template, [other], RELEASE, enabled, "team")
    assert not reference_storage.can_serve_template(template, [], RELEASE, enabled, "team")
