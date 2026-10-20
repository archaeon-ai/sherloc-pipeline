"""Analytic synthetic traces and identities only; no scientific data fixtures."""

import hashlib
import io
import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from typer.testing import CliRunner

from sherloc_pipeline.cli.identification import app
from sherloc_pipeline.core.identification.compatibility import query_exclusion, reference_exclusion
from sherloc_pipeline.core.identification.decision import decide
from sherloc_pipeline.core.identification.inspection import inspection_fit
from sherloc_pipeline.core.identification.pearson import compare, signed_pearson
from sherloc_pipeline.core.identification.prep import Prepared, grid_for, prepare
from sherloc_pipeline.core.identification.recipe import load_recipe
from sherloc_pipeline.database.reference_models import ReferenceFileORM, ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.models.identification import Candidate, IdentificationRequest, PointSelection
from sherloc_pipeline.models.references import ReferenceTemplate, ReferenceTrace, TemplateMember
from sherloc_pipeline.services.identification import IdentificationService, LocalPointResolver, deny_source
from sherloc_pipeline.services.reference_import import array_digest, digest
from sherloc_pipeline.services.reference_store import object_locator, verify_bytes
from sherloc_pipeline.web.feature_policy import FeaturePolicy, SourceDenylist


class SyntheticResolver:
    def __init__(self, x, y):
        self.resolve = Mock(return_value=(x, y))

    def resolve_bounded(self, selection, timeout):
        return self.resolve(selection)


class SyntheticStore:
    def __init__(self, data):
        self.data = data
        self.read_verified = Mock(side_effect=lambda locator, expected: verify_bytes(self.data, expected))

    def __reduce__(self):
        checksum = hashlib.sha256(self.data).hexdigest()
        data = self.read_verified(object_locator(checksum), checksum)
        return SyntheticStore, (data,)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))


@pytest.fixture
def recipe():
    return load_recipe()[0]


@pytest.fixture
def team_deployment(monkeypatch):
    monkeypatch.setenv("PHASE_TIER", "team")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")


def selection(**changes):
    return PointSelection(**{
        "product_id": "ss__9999_0000000000_000rrs__synthetic", "sol": 9999,
        "target": "Synthetic Target", "point_index": 0, "source_sha256": "1" * 64,
        "modality": "raman", "x_quantity": "raman_shift", "x_unit": "cm-1",
        "excitation_nm": 250.0, "excitation_evidence": "synthetic stated excitation",
        **changes,
    })


def trace_metadata(**changes):
    return ReferenceTrace(**{
        "trace_index": 0, "source_label": "synthetic", "x_key": "x", "y_key": "y",
        "modality": "raman", "x_quantity": "raman_shift", "x_unit": "cm-1",
        "x_evidence": "synthetic physical axis", "y_unit": "synthetic units",
        "excitation_value": 250.0, "excitation_unit": "nm",
        "excitation_evidence": "synthetic stated excitation", "instrument": "Synthetic Instrument",
        "species_name": "Syntheticite", "specimen_key": "synthetic-specimen-1", "sample_form": "pure",
        "rights_status": "known", "license_identifier": "Synthetic-Test-License",
        "license_source": "synthetic test grant", **changes,
    })


def bands():
    x = np.linspace(760., 1840., 541)
    y = np.exp(-((x-1123.)/17.)**2) + .6 / (1+((x-1517.)/23.)**2) + .0003*x
    return x, y


@pytest.fixture
def system():
    _, recipe_id = load_recipe()
    release_id, template_id, trace_id, file_id, serving_id = [c * 64 for c in "abcde"]
    subset_id = digest([template_id])
    request = IdentificationRequest(selection=selection(), reference_release_id=release_id,
                                    reference_subset_id=subset_id, method_release_id=recipe_id,
                                    template_ids=[template_id])
    policy = FeaturePolicy.model_validate({
        "references": {"mode": "all_approved", "release_id": release_id,
                       "subset_id": subset_id, "serving_policy_version": serving_id},
        "identification": {"enabled": True, "experimental_candidates": True,
                           "allowed_method_releases": [{"method_release_id": recipe_id,
                                                       "reference_release_id": release_id,
                                                       "reference_subset_id": subset_id}]},
    })
    x, y = bands()
    buffer = io.BytesIO()
    np.savez(buffer, x=x, y=y)
    data = buffer.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    meta = trace_metadata()
    release = ReferenceReleaseORM(release_id=release_id, status="ready", serving_policy_sha256=serving_id)
    trace = ReferenceTraceORM(trace_revision_id=trace_id, release_id=release_id, file_revision_id=file_id,
                              trace_index=0, source_key="synthetic-0001:synthetic-file-1:0",
                              point_count=len(x), axis_sha256=array_digest(x),
                              intensity_sha256=array_digest(y), rights_known=True,
                              serving_status="approved", permitted_tiers=["team"],
                              exclusions=[], metadata_json=meta.model_dump(mode="json"))
    template_meta = ReferenceTemplate(name="synthetic-template", species_name="Syntheticite",
                                      recipe_sha256=recipe_id, grouping_evidence="synthetic single specimen",
                                      members=[TemplateMember(record_id="synthetic-0001", file_id="synthetic-file-1",
                                                              trace_index=0)])
    template = ReferenceTemplateORM(template_id=template_id, release_id=release_id,
                                     name="synthetic-template", recipe_sha256=recipe_id,
                                     member_trace_ids=[trace_id], permitted_tiers=["team"],
                                     metadata_json=template_meta.model_dump(mode="json"))
    file = ReferenceFileORM(file_revision_id=file_id, release_id=release_id,
                            array_locator=object_locator(sha), array_sha256=sha, metadata_json={"arrays_size": len(data)})
    rows = {(ReferenceReleaseORM, release_id): release, (ReferenceTraceORM, trace_id): trace,
            (ReferenceTemplateORM, template_id): template, (ReferenceFileORM, file_id): file}
    session = Mock()
    session.get.side_effect = lambda cls, identity: rows.get((cls, identity))
    store = SyntheticStore(data)
    resolver = SyntheticResolver(x, y)
    service = IdentificationService(session, store, resolver, policy, "team")
    return SimpleNamespace(**locals())


def test_frozen_recipe(recipe):
    assert (recipe.method_id, recipe.status, recipe.support_min, recipe.threshold_source,
            recipe.evaluation_unit) == ("raman-pearson", "provisional", .80, "fixed_provisional", "point")
    assert np.array_equal(grid_for(recipe), np.arange(800., 1800.1, 5.))
    assert len(load_recipe()[1]) == 64
    assert load_recipe() == load_recipe()
    assert recipe.extra_fwhm_cm1 == [0.0]
    assert recipe.version == "0.2.0"


@pytest.mark.parametrize("widths", [None, [], [1.0], [0.0, 1.0], [0.0, 0.0]])
def test_recipe_rejects_alternate_broadening(recipe, monkeypatch, widths):
    values = recipe.model_dump()
    if widths is None:
        del values["extra_fwhm_cm1"]
    else:
        values["extra_fwhm_cm1"] = widths
    data = json.dumps(values).encode()
    monkeypatch.setattr("sherloc_pipeline.core.identification.recipe.files", lambda package: SimpleNamespace(
        joinpath=lambda path: SimpleNamespace(read_bytes=lambda: data)))
    with pytest.raises(ValueError):
        load_recipe()


def test_recipe_hash_includes_zero_broadening(recipe, monkeypatch):
    data = json.dumps(recipe.model_dump()).encode()
    monkeypatch.setattr("sherloc_pipeline.core.identification.recipe.files", lambda package: SimpleNamespace(
        joinpath=lambda path: SimpleNamespace(read_bytes=lambda: data)))
    loaded, identity = load_recipe()
    assert loaded.extra_fwhm_cm1 == [0.0]
    assert identity == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("a,b,expected", [([1., 2., 3.], [7., 9., 11.], 1.),
                                         ([1., 2., 3.], [11., 9., 7.], -1.),
                                         ([-1., 0., 1.], [1., -2., 1.], 0.)])
def test_signed_analytic_pearson(a, b, expected):
    assert signed_pearson(a, b) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("a,b", [([1., 1., 1.], [2., 3., 4.]), ([1.], [1.]),
                               ([1., 2.], [1.]), ([np.nan, 2.], [1., 2.])])
def test_undefined_pearson(a, b):
    assert signed_pearson(a, b) is None


def test_preparation_sort_duplicates_negative_and_immutable(recipe):
    x, y = bands()
    original_x, original_y = x.copy(), y.copy()
    prepared = prepare(x, y-7., recipe)
    duplicated = prepare(np.repeat(x[::-1], 2), np.repeat(y[::-1]-7., 2), recipe)
    assert prepared.vector is not None
    np.testing.assert_allclose(prepared.vector, duplicated.vector, atol=1e-10)
    assert np.linalg.norm(prepared.vector[prepared.mask]) == pytest.approx(1.)
    assert not prepared.mask[(grid_for(recipe) >= 965) & (grid_for(recipe) <= 985)].any()
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


def test_preparation_triple_duplicates_do_not_create_saturation(recipe):
    x, y = bands()
    expected = prepare(x, y, recipe)
    order = np.random.default_rng(42).permutation(len(x) * 3)
    actual = prepare(np.repeat(x, 3)[order], np.repeat(y, 3)[order], recipe)
    assert actual.vector is not None
    np.testing.assert_allclose(actual.vector, expected.vector, atol=1e-10)
    np.testing.assert_array_equal(actual.mask, expected.mask)


@pytest.mark.parametrize("saturated", [False, True])
def test_preparation_saturation_is_invariant_to_permutation(recipe, saturated):
    x, y = bands()
    if saturated:
        middle = len(y) // 2
        y[middle:middle+4] = y.max() + 1.
    order = np.random.default_rng(42).permutation(len(x))
    expected, actual = prepare(x, y, recipe), prepare(x[order], y[order], recipe)
    assert actual.diagnostics["reason"] == expected.diagnostics["reason"]
    if saturated:
        assert actual.diagnostics["reason"] == "flat_top_possible_saturation"
    else:
        np.testing.assert_array_equal(actual.vector, expected.vector)
    np.testing.assert_array_equal(actual.mask, expected.mask)


@pytest.mark.parametrize("kind,expected", [("flat", "constant_signal"), ("nan", "nonfinite_intensity"),
                                         ("top", "flat_top_possible_saturation"),
                                         ("short", "invalid_axis"), ("gap", "insufficient_valid_overlap"),
                                         ("emission", "insufficient_native_samples")])
def test_preparation_rejects_synthetic_bad_inputs(recipe, kind, expected):
    x, y = bands()
    if kind == "flat":
        y[:] = -3.
    elif kind == "nan":
        y[0] = np.nan
    elif kind == "top":
        middle = len(y) // 2
        y[middle:middle+4] = 10.
    elif kind == "short":
        x, y = x[:10], y[:10]
    elif kind == "gap":
        keep = (x < 1200) | (x > 1350)
        x, y = x[keep], y[keep]
    elif kind == "emission":
        x = np.linspace(230, 400, len(y))
    assert prepare(x, y, recipe).diagnostics["reason"] == expected


@pytest.mark.parametrize("region", ["outside_coverage", "gap", "excluded"])
def test_preparation_invalid_regions_do_not_affect_vectors_or_support(recipe, region):
    recipe = recipe.model_copy(update={
        "grid": [0., 200., 1.], "shifts": [-1., 0., 1.],
        "excluded_intervals": [[70., 75.]], "max_gap": 5.,
        "min_coverage": .85, "min_overlap": .80,
    })
    x = np.arange(-10.25, 211., .5)
    if region == "gap":
        x = x[(x <= 39.) | (x >= 51.) | (x == 44.75)]
    y = np.exp(-((x-110.)/8.)**2) + .6 / (1+((x-160.)/7.)**2) + .002*x
    changed = y.copy()
    invalid = {"outside_coverage": (x < 0.) | (x > 200.),
               "gap": x == 44.75, "excluded": (x >= 70.) & (x <= 75.)}[region]
    changed[invalid] = 1e6
    expected, actual = prepare(x, y, recipe), prepare(x, changed, recipe)
    assert expected.vector is not None and actual.vector is not None
    np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(actual.vector[actual.mask], expected.vector[expected.mask])
    assert actual.diagnostics == expected.diagnostics
    assert not actual.mask[[0, -1]].any()
    grid = grid_for(recipe)
    assert not actual.mask[(grid >= 70.) & (grid <= 75.)].any()
    if region == "gap":
        assert not actual.mask[(grid > 39.) & (grid < 51.)].any()
    reference = prepare(x, y + .1*np.cos(x/7.), recipe)
    before, after = compare(expected, reference, recipe), compare(actual, reference, recipe)
    assert before is not None and after is not None
    assert after["support"] == before["support"]
    assert after["shift_cm_1"] == before["shift_cm_1"]


def test_shift_recovery_and_signed_negative_support(recipe):
    grid = grid_for(recipe)
    y = np.exp(-((grid-1200)/23)**2)
    reference = Prepared(y, np.ones(len(grid), dtype=bool), {})
    query = Prepared(np.interp(grid-5, grid, y), reference.mask, {})
    result = compare(query, reference, recipe)
    assert result["shift_cm_1"] == 5.
    assert result["extra_fwhm_cm1"] == 0.0
    assert result["support"] == pytest.approx(1.)
    negative = compare(Prepared(-y, reference.mask, {}), reference, recipe)
    assert negative["support"] < 0


def test_inspection_analytic():
    b = np.array([-2., 0., 1., 4.])
    result = inspection_fit(3.*b+7., b, np.ones(4, dtype=bool))
    assert result["interpretation"] == "mathematical_inspection"
    assert result["inverted"] is False
    assert result["scale"] == pytest.approx(3.)
    assert result["offset"] == pytest.approx(7.)
    assert result["sse"] < 1e-25


@pytest.mark.parametrize("support,status", [(.799999, "abstained"), (.80, "candidate"), (-.9, "abstained")])
def test_fixed_threshold(recipe, support, status):
    row = Candidate(template_id="a"*64, species_name="Syntheticite", supported_class=None,
                    trace_revision_ids=["b"*64], support=support, shift_cm_1=0., extra_fwhm_cm1=0.)
    assert decide([row], {"snr_proxy": 10.}, recipe)[0] == status
    assert decide([row], {"snr_proxy": 2.}, recipe)[1] == ["insufficient_signal"]


@pytest.mark.parametrize("changes,reason", [({"modality": "fluorescence"}, "unsupported_reference_axis"),
    ({"excitation_evidence": None}, "missing_excitation_evidence"),
    ({"excitation_value": 251.}, "incompatible_excitation"),
    ({"sample_form": "mixed"}, "not_pure_species_reference"),
    ({"source_processing": ["unknown"]}, "unsupported_reference_processing"),
    ({"x_evidence": None}, "missing_axis_evidence"),
    ({"qa_flags": ["synthetic-flag"]}, "reference_qa_exclusion")])
def test_compatibility(changes, reason):
    assert reference_exclusion(trace_metadata(**changes), selection()) == reason


def test_query_compatibility():
    assert query_exclusion(selection(processing=["manual"])) == "unsupported_query_processing"
    assert query_exclusion(selection(modality="fluorescence")) == "unsupported_query_axis"


def test_service_deterministic_candidate(system):
    first = system.service.identify(system.request)
    second = system.service.identify(system.request)
    assert first == second
    assert first.status == "candidate"
    assert first.candidate == "Syntheticite"
    assert first.ranking[0].support == pytest.approx(1.)
    assert all(item.extra_fwhm_cm1 == 0.0 for item in first.ranking)
    assert first.method["threshold_source"] == "fixed_provisional"
    assert first.selection.point_index == 0
    assert first.evidence["inspection"]["sse"] < 1e-20
    assert len(first.result_digest) == 64


@pytest.mark.parametrize("denial", [
    {"sols": [9999]},
    {"product_ids": ["ss__9999_0000000000_000rrs__synthetic"]},
    {"target_names": ["Synthetic Target"]},
])
def test_configured_denial_before_all_reads(system, denial, monkeypatch):
    values = system.policy.model_dump()
    values["identification"]["source_denylist"] = denial
    system.service.policy = FeaturePolicy.model_validate(values)
    recipe_reader = Mock(side_effect=AssertionError("recipe read"))
    code_reader = Mock(side_effect=AssertionError("implementation read"))
    monkeypatch.setattr("sherloc_pipeline.services.identification.load_recipe", recipe_reader)
    monkeypatch.setattr("sherloc_pipeline.services.identification.runtime_provenance", code_reader)
    with pytest.raises(ValueError, match="source denied"):
        system.service.identify(system.request)
    recipe_reader.assert_not_called()
    code_reader.assert_not_called()
    system.session.get.assert_not_called()
    system.resolver.resolve.assert_not_called()
    system.store.read_verified.assert_not_called()


def test_product_identity_cannot_override_sol():
    point = selection(product_id="ss__9998_0000000000_000rrs__synthetic")
    with pytest.raises(ValueError, match="source denied"):
        deny_source(point, SourceDenylist(sols=[9998]))
    with pytest.raises(ValueError, match="sol identity mismatch"):
        deny_source(point)


@pytest.mark.parametrize("mode", ["default", "experimental_off", "public"])
def test_default_off_before_all_reads(system, mode, monkeypatch):
    if mode == "default":
        system.service.policy = FeaturePolicy()
    elif mode == "experimental_off":
        values = system.policy.model_dump()
        values["identification"]["experimental_candidates"] = False
        system.service.policy = FeaturePolicy.model_validate(values)
    else:
        system.service.tier = "public"
    monkeypatch.setattr("sherloc_pipeline.services.identification.load_recipe", Mock(side_effect=AssertionError("recipe read")))
    with pytest.raises(ValueError, match="disabled"):
        system.service.identify(system.request)
    system.session.get.assert_not_called()
    system.resolver.resolve.assert_not_called()
    system.store.read_verified.assert_not_called()


@pytest.mark.parametrize("change", ["retired", "unknown_rights", "unapproved", "wrong_subset", "wrong_method", "wrong_hash"])
def test_service_rejects_invalid_pins_permissions_or_hashes(system, change):
    request = system.request
    if change == "retired":
        system.release.status = "retired"
    elif change == "unknown_rights":
        system.trace.metadata_json = trace_metadata(rights_status="unknown").model_dump(mode="json")
    elif change == "unapproved":
        system.trace.serving_status = "off"
    elif change == "wrong_subset":
        request = request.model_copy(update={"reference_subset_id": "f"*64})
    elif change == "wrong_method":
        request = request.model_copy(update={"method_release_id": "f"*64})
    else:
        system.trace.axis_sha256 = "f"*64
    with pytest.raises(ValueError):
        system.service.identify(request)
    if change != "wrong_hash":
        system.resolver.resolve.assert_not_called()
        system.store.read_verified.assert_not_called()


def test_incompatible_reference_excluded_before_array_read(system):
    system.trace.metadata_json = trace_metadata(sample_form="mixed").model_dump(mode="json")
    result = system.service.identify(system.request)
    assert result.status == "abstained"
    assert result.diagnostics["excluded_references"][0]["reason"] == "not_pure_species_reference"
    system.store.read_verified.assert_not_called()


def test_flat_source_abstains_without_reference_read(system):
    x, y = bands()
    system.resolver.resolve.return_value = (x, np.zeros(len(y)))
    result = system.service.identify(system.request)
    assert result.reasons == ["constant_signal"]
    system.store.read_verified.assert_not_called()


def test_concurrency_limit(system):
    system.service._slots.acquire()
    try:
        with pytest.raises(ValueError, match="concurrency"):
            system.service.identify(system.request)
    finally:
        system.service._slots.release()
    system.session.get.assert_not_called()


def test_timeout(system, monkeypatch):
    ticks = iter([0., 100.])
    monkeypatch.setattr("sherloc_pipeline.services.identification.time.monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="time limit"):
        system.service.identify(system.request)
    system.resolver.resolve.assert_not_called()


def test_local_point_source_hash_and_metadata(tmp_path):
    point = selection()
    x, y = bands()
    buffer = io.BytesIO()
    np.savez(buffer, x=x, y=y, metadata=json.dumps(point.model_dump(mode="json", exclude={"source_sha256"})))
    data = buffer.getvalue()
    point = point.model_copy(update={"source_sha256": hashlib.sha256(data).hexdigest()})
    path = tmp_path / f"{point.product_id}-point-0.npz"
    path.write_bytes(data)
    resolver = LocalPointResolver(tmp_path)
    actual_x, actual_y = resolver.resolve(point)
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    with pytest.raises(ValueError, match="SHA-256"):
        resolver.resolve(point.model_copy(update={"source_sha256": "0"*64}))
    with pytest.raises(ValueError, match="metadata"):
        resolver.resolve(point.model_copy(update={"target": "Other Synthetic Target"}))


@pytest.mark.parametrize("denial", [
    {"sols": [9999]},
    {"product_ids": ["ss__9999_0000000000_000rrs__synthetic"]},
    {"target_names": ["Synthetic Target"]},
])
def test_local_denial_before_open(tmp_path, monkeypatch, denial):
    resolver = LocalPointResolver(tmp_path, SourceDenylist.model_validate(denial))
    monkeypatch.setattr("pathlib.Path.open", Mock(side_effect=AssertionError("file read")))
    with pytest.raises(ValueError, match="source denied"):
        resolver.resolve(selection())


def test_cli_default_off_and_configured_denial(system, tmp_path, monkeypatch, team_deployment):
    monkeypatch.delenv("SHERLOC_FEATURE_POLICY", raising=False)
    monkeypatch.setattr("sherloc_pipeline.cli.identification.create_engine", Mock(side_effect=AssertionError("database read")))
    runner = CliRunner()
    options = ["--database", str(tmp_path / "unopened.db"), "--objects", str(tmp_path), "--sources", str(tmp_path)]
    result = runner.invoke(app, ["--request-json", system.request.model_dump_json(), *options])
    assert result.exit_code == 1
    assert "disabled" in result.output
    result = runner.invoke(app, ["--request-json", system.request.model_dump_json(), *options,
                                 "--source-denylist-json", '{"sols":[9999]}'])
    assert result.exit_code == 1
    assert "source denied" in result.output


@pytest.mark.parametrize("extra", [None, "{}", '{"target_names":["Other Synthetic Target"]}'])
def test_cli_denylist_adds_to_run_configuration_before_reads(system, tmp_path, monkeypatch, extra, team_deployment):
    values = system.policy.model_dump()
    values["identification"]["source_denylist"] = {"sols": [9999]}
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps(values))
    monkeypatch.setattr("sherloc_pipeline.cli.identification.create_engine", Mock(side_effect=AssertionError("database read")))
    monkeypatch.setattr("sherloc_pipeline.services.identification.load_recipe", Mock(side_effect=AssertionError("recipe read")))
    options = ["--database", str(tmp_path / "unopened.db"), "--objects", str(tmp_path), "--sources", str(tmp_path)]
    if extra is not None:
        options += ["--source-denylist-json", extra]
    result = CliRunner().invoke(app, ["--request-json", system.request.model_dump_json(), *options])
    assert result.exit_code == 1
    assert "source denied" in result.output


def test_seeded_noise_abstains(system):
    x, _ = bands()
    y = np.random.default_rng(2037).normal(size=len(x))
    system.resolver.resolve.return_value = (x, y)
    result = system.service.identify(system.request)
    assert result.status == "abstained"
    assert result.candidate is None


def test_template_tie_order_and_effective_set(system):
    first_id, second_id = system.template_id, "f"*64
    metadata = dict(system.template.metadata_json)
    metadata["name"] = "synthetic-template-2"
    second = ReferenceTemplateORM(template_id=second_id, release_id=system.release_id,
                                  name=metadata["name"], recipe_sha256=system.recipe_id,
                                  member_trace_ids=[system.trace_id], permitted_tiers=["team"],
                                  metadata_json=metadata)
    system.rows[(ReferenceTemplateORM, second_id)] = second
    ids = sorted([first_id, second_id])
    values = system.policy.model_dump()
    values["references"]["subset_id"] = digest(ids)
    values["identification"]["allowed_method_releases"][0]["reference_subset_id"] = digest(ids)
    system.service.policy = FeaturePolicy.model_validate(values)
    request = system.request.model_copy(update={"template_ids": ids, "reference_subset_id": digest(ids)})
    result = system.service.identify(request)
    assert len(result.ranking) == 1
    assert result.ranking[0].template_id == first_id
    assert result.effective_set_sha256 == digest(ids)
    assert system.service.identify(request) == result


def test_template_bounds_before_reads(system):
    values = system.policy.model_dump()
    values["identification"]["max_templates"] = 1
    system.service.policy = FeaturePolicy.model_validate(values)
    request = system.request.model_copy(update={"template_ids": ["a"*64, "f"*64]})
    with pytest.raises(ValueError, match="bound mismatch"):
        system.service.identify(request)
    system.session.get.assert_not_called()
    system.resolver.resolve.assert_not_called()


def test_local_source_symlink_escape(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    other = tmp_path / "synthetic-outside.npz"
    other.write_bytes(b"synthetic not an archive")
    point = selection()
    (root / f"{point.product_id}-point-0.npz").symlink_to(other)
    with pytest.raises(ValueError, match="escapes root"):
        LocalPointResolver(root).resolve(point)


@pytest.mark.parametrize("tier", ["team", " TEAM "])
def test_cli_enabled_synthetic(system, tmp_path, monkeypatch, team_deployment, tier):
    from sherloc_pipeline.cli import identification as cli
    from sqlalchemy import create_engine
    from sherloc_pipeline.database.models import Base
    from sherloc_pipeline.services.reference_store import LocalReferenceStore

    database = tmp_path / "synthetic.db"
    engine = create_engine(f"sqlite:///{database}")
    Base.metadata.create_all(engine)
    # Use actual catalog persistence and verified local object/source adapters.
    from sqlalchemy.orm import Session
    system.release.source_snapshot_sha256 = "1"*64
    system.release.metadata_sha256 = "2"*64
    system.release.array_index_sha256 = "3"*64
    system.release.schema_version = "reference-snapshot-1"
    system.release.source_name = "Synthetic Library"
    from datetime import datetime, timezone
    system.release.captured_at = datetime(2001, 1, 1, tzinfo=timezone.utc)
    system.release.importer_version = "synthetic-importer"
    system.release.parser_version = "synthetic-parser"
    system.release.qa_version = "synthetic-qa"
    system.file.datatype_id = "synthetic-type"
    system.file.record_id = "synthetic-0001"
    system.file.file_id = "synthetic-file-1"
    system.file.raw_locator = object_locator(system.sha)
    system.file.raw_sha256 = system.sha
    system.file.raw_size = len(system.data)
    system.file.metadata_json = {"arrays_size": len(system.data)}
    system.trace.source_key = "synthetic-0001:synthetic-file-1:0"
    system.release.status = "staged"
    with Session(engine) as session:
        session.add(system.release)
        session.flush()
        session.add(system.file)
        session.flush()
        session.add_all([system.trace, system.template])
        session.flush()
        system.release.status = "ready"
        session.commit()
    engine.dispose()
    objects = tmp_path / "objects"
    LocalReferenceStore(objects).put_verified(system.data, system.sha)
    source = io.BytesIO()
    np.savez(source, x=system.x, y=system.y,
             metadata=json.dumps(system.request.selection.model_dump(mode="json", exclude={"source_sha256"})))
    data = source.getvalue()
    point = system.request.selection.model_copy(update={"source_sha256": hashlib.sha256(data).hexdigest()})
    (tmp_path / f"{point.product_id}-point-0.npz").write_bytes(data)
    request = system.request.model_copy(update={"selection": point})
    monkeypatch.setenv("PHASE_TIER", tier)
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", system.policy.model_dump_json())
    load_policy = Mock(wraps=cli.load_feature_policy)
    service = Mock(wraps=cli.IdentificationService)
    monkeypatch.setattr(cli, "load_feature_policy", load_policy)
    monkeypatch.setattr(cli, "IdentificationService", service)
    result = CliRunner().invoke(app, ["--request-json", request.model_dump_json(),
                                    "--database", str(database), "--objects", str(objects),
                                    "--sources", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["candidate"] == "Syntheticite"
    load_policy.assert_called_once_with("team")
    assert service.call_args.args[-1] == "team"


@pytest.mark.parametrize("tier,access_mode", [
    ("public", "public"), ("public", "internal"), (None, "internal"),
    ("", "internal"), ("unknown", "internal"), ("team", "public"), ("team", "unknown"),
])
def test_cli_rejects_deployment_before_reads(system, tmp_path, monkeypatch, tier, access_mode):
    from sherloc_pipeline.cli import identification as cli

    if tier is None:
        monkeypatch.delenv("PHASE_TIER", raising=False)
    else:
        monkeypatch.setenv("PHASE_TIER", tier)
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", access_mode)
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", system.policy.model_dump_json())
    readers = [Mock(side_effect=AssertionError("read before tier validation")) for _ in range(4)]
    monkeypatch.setattr(cli, "load_feature_policy", readers[0])
    monkeypatch.setattr(cli, "create_engine", readers[1])
    monkeypatch.setattr("pathlib.Path.open", readers[2])
    monkeypatch.setattr("sherloc_pipeline.services.identification.load_recipe", readers[3])
    result = CliRunner().invoke(app, ["--request-json", system.request.model_dump_json(),
                                    "--database", str(tmp_path / "unopened.db"),
                                    "--objects", str(tmp_path), "--sources", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "Identification failed" in result.output
    for reader in readers:
        reader.assert_not_called()


@pytest.mark.parametrize("field,value", [("record_id", "synthetic-0002"), ("trace_index", 1)])
def test_ordered_template_membership_checked_before_arrays(system, field, value):
    metadata = json.loads(json.dumps(system.template.metadata_json))
    metadata["members"][0][field] = value
    system.template.metadata_json = metadata
    with pytest.raises(ValueError, match="membership metadata mismatch"):
        system.service.identify(system.request)
    system.store.read_verified.assert_not_called()


def test_source_and_environment_provenance_affect_result_digest(system, monkeypatch):
    from sherloc_pipeline.core.identification import provenance

    original = system.service.identify(system.request)
    method = original.method
    assert method["code_identity_basis"] == "installed_source_bytes"
    assert len(method["code_sha256"]) == 64
    assert len(method["environment_sha256"]) == 64
    assert set(method["environment"]) == {"python", "numpy", "scipy"}
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "synthetic-runtime-version")
    changed_runtime = system.service.identify(system.request)
    assert changed_runtime.method["environment_sha256"] != method["environment_sha256"]
    assert changed_runtime.method["code_sha256"] == method["code_sha256"]
    assert changed_runtime.result_digest != original.result_digest
    monkeypatch.setattr(provenance, "files", lambda package: SimpleNamespace(
        joinpath=lambda path: SimpleNamespace(read_bytes=lambda: b"synthetic changed implementation")))
    changed_source = system.service.identify(system.request)
    assert changed_source.method["code_sha256"] != method["code_sha256"]
    assert changed_source.result_digest != changed_runtime.result_digest



@pytest.mark.parametrize("shared", [
    "database/reference_models.py", "models/feature_policy.py", "models/references.py",
    "services/reference_import.py", "services/reference_serving.py", "services/reference_store.py",
])
def test_shared_engine_modules_affect_code_fingerprint(monkeypatch, shared):
    from sherloc_pipeline.core.identification import provenance

    root = provenance.files("sherloc_pipeline")
    original = provenance.implementation_fingerprint()

    def changed(path):
        data = b"synthetic changed shared module" if path == shared else root.joinpath(path).read_bytes()
        return SimpleNamespace(read_bytes=lambda: data)

    monkeypatch.setattr(provenance, "files", lambda package: SimpleNamespace(joinpath=changed))
    assert shared in provenance.IMPLEMENTATION_FILES
    assert provenance.implementation_fingerprint() != original

@pytest.mark.parametrize("target", ["Syntheticite Ridge", "SYNTHETICITE_RIDGE", "Syntheticite-Ridge"])
def test_named_denial_normalizes_only_configured_names(target):
    point = selection(target=target)
    deny_source(point)
    with pytest.raises(ValueError, match="source denied"):
        deny_source(point, SourceDenylist(target_names=["Syntheticite Ridge"]))
    deny_source(point, SourceDenylist(target_names=["Other Synthetic Target"]))


def test_source_denylist_defaults_empty(system):
    assert FeaturePolicy().identification.source_denylist.model_dump() == {
        "sols": [], "product_ids": [], "target_names": [],
    }
    deny_source(selection())
    assert system.service.identify(system.request).status == "candidate"


@pytest.mark.parametrize("values", [
    {"sols": ["9999"]}, {"sols": [-1]}, {"product_ids": [123]},
    {"target_names": [""]}, {"target_names": ["---"]}, {"unexpected": []},
])
def test_source_denylist_is_strict(values):
    with pytest.raises(ValueError):
        SourceDenylist.model_validate(values)


def test_negative_inspection_disclosed_without_changing_signed_support():
    reference = np.array([-2., 0., 1., 4.])
    query = -3.*reference+7.
    before = signed_pearson(query, reference)
    result = inspection_fit(query, reference, np.ones(4, dtype=bool))
    assert result["interpretation"] == "mathematical_inspection"
    assert result["inverted"] is True
    assert result["scale"] == pytest.approx(-3.)
    assert result["offset"] == pytest.approx(7.)
    assert result["sse"] < 1e-25
    assert signed_pearson(query, reference) == before == pytest.approx(-1.)


@pytest.mark.parametrize("bound", ["points", "bytes", "members"])
def test_admission_before_source_or_reference_arrays(system, bound):
    if bound == "points":
        system.trace.point_count = 1000000
    elif bound == "bytes":
        system.file.metadata_json = {"arrays_size": 1000000000}
    else:
        system.template.member_trace_ids = [system.trace_id] * 101
    with pytest.raises(ValueError):
        system.service.identify(system.request)
    system.resolver.resolve.assert_not_called()
    system.store.read_verified.assert_not_called()


def pearson_golden_case(system):
    from sherloc_pipeline.services.identification import compute_identification

    recipe, recipe_id = load_recipe()
    x, y = system.x, system.y
    query = prepare(x, y, recipe)
    template = {
        "template_id": system.template_id, "recipe_sha256": recipe_id,
        "member_trace_ids": [system.trace_id],
        "metadata_json": {**system.template.metadata_json, "recipe_sha256": recipe_id},
    }
    member = {
        "source_key": system.trace.source_key, "trace_index": 0,
        "metadata_json": system.trace.metadata_json,
        "axis_sha256": array_digest(x), "intensity_sha256": array_digest(y),
        "arrays": {"x": x, "y": y},
    }
    # Hold provenance fixed: installed source fingerprints intentionally change.
    provenance = {"code_sha256": "f" * 64, "code_identity_basis": "synthetic",
                  "environment": {"runtime": "synthetic"}, "environment_sha256": "e" * 64}
    return compute_identification(
        system.request.model_copy(update={"method_release_id": recipe_id}),
        recipe, recipe_id, provenance, query, [(template, [member])],
    )


def test_pearson_complete_payload_matches_pr7_golden(system):
    # Generated at pr7-integration f793393 from this same synthetic case.
    expected = {'schema_version': 'identification-1',
     'status': 'candidate',
     'reasons': [],
     'warnings': ['provisional_method', 'experimental_candidates'],
     'method': {'method_id': 'raman-pearson',
                'version': '0.2.0',
                'status': 'provisional',
                'support_min': 0.8,
                'threshold_source': 'fixed_provisional',
                'evaluation_unit': 'point',
                'grid': [800.0, 1800.0, 5.0],
                'shifts': [-10.0, -5.0, 0.0, 5.0, 10.0],
                'extra_fwhm_cm1': [0.0],
                'excluded_intervals': [[790.0, 810.0], [965.0, 985.0]],
                'min_coverage': 0.95,
                'min_overlap': 0.9,
                'max_gap': 30.0,
                'min_samples': 30,
                'baseline_lambda': 100000.0,
                'baseline_iterations': 10,
                'baseline_asymmetry': 0.01,
                'snr_min': 3.0,
                'recipe_sha256': '3a2d27da061756c69ef9c2473a59fbd2def13fe2df1ae3be429dcab37f010bce',
                'release_id': '3a2d27da061756c69ef9c2473a59fbd2def13fe2df1ae3be429dcab37f010bce',
                'code_sha256': 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
                'code_identity_basis': 'synthetic',
                'environment': {'runtime': 'synthetic'},
                'environment_sha256': 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'},
     'selection': {'product_id': 'ss__9999_0000000000_000rrs__synthetic',
                   'sol': 9999,
                   'target': 'Synthetic Target',
                   'point_index': 0,
                   'source_sha256': '1111111111111111111111111111111111111111111111111111111111111111',
                   'modality': 'raman',
                   'x_quantity': 'raman_shift',
                   'x_unit': 'cm-1',
                   'excitation_nm': 250.0,
                   'excitation_evidence': 'synthetic stated excitation',
                   'processing': []},
     'reference_release_id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
     'reference_subset_id': 'bd8aff2bc7e7d9450ce0f4b4acc9982d5fd2abaecaf1700ddce7c8b8e3222661',
     'effective_set_sha256': 'bd8aff2bc7e7d9450ce0f4b4acc9982d5fd2abaecaf1700ddce7c8b8e3222661',
     'ranking': [{'template_id': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                  'species_name': 'Syntheticite',
                  'supported_class': None,
                  'trace_revision_ids': ['cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'],
                  'support': 1.0,
                  'shift_cm_1': 0.0,
                  'extra_fwhm_cm1': 0.0,
                  'citations': [{'trace_revision_id': 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                                 'citation_text': None,
                                 'citation_doi': None,
                                 'citation_source': None,
                                 'license_identifier': 'Synthetic-Test-License',
                                 'license_text': None,
                                 'license_source': 'synthetic test grant'}]}],
     'candidate': 'Syntheticite',
     'diagnostics': {'reason': None,
                     'noise_native': 0.00017522242646774016,
                     'snr_proxy': 2743.2218101495473,
                     'masked_bins': 8,
                     'native_points': 541,
                     'norm': 2.5669100127514315,
                     'cosmic_mask_available': False,
                     'excluded_references': []},
     'evidence': {'grid': [800.0, 805.0, 810.0, 815.0, 820.0, 825.0, 830.0, 835.0, 840.0, 845.0, 850.0, 855.0,
                           860.0, 865.0, 870.0, 875.0, 880.0, 885.0, 890.0, 895.0, 900.0, 905.0, 910.0, 915.0,
                           920.0, 925.0, 930.0, 935.0, 940.0, 945.0, 950.0, 955.0, 960.0, 965.0, 970.0, 975.0,
                           980.0, 985.0, 990.0, 995.0, 1000.0, 1005.0, 1010.0, 1015.0, 1020.0, 1025.0, 1030.0,
                           1035.0, 1040.0, 1045.0, 1050.0, 1055.0, 1060.0, 1065.0, 1070.0, 1075.0, 1080.0, 1085.0,
                           1090.0, 1095.0, 1100.0, 1105.0, 1110.0, 1115.0, 1120.0, 1125.0, 1130.0, 1135.0, 1140.0,
                           1145.0, 1150.0, 1155.0, 1160.0, 1165.0, 1170.0, 1175.0, 1180.0, 1185.0, 1190.0, 1195.0,
                           1200.0, 1205.0, 1210.0, 1215.0, 1220.0, 1225.0, 1230.0, 1235.0, 1240.0, 1245.0, 1250.0,
                           1255.0, 1260.0, 1265.0, 1270.0, 1275.0, 1280.0, 1285.0, 1290.0, 1295.0, 1300.0, 1305.0,
                           1310.0, 1315.0, 1320.0, 1325.0, 1330.0, 1335.0, 1340.0, 1345.0, 1350.0, 1355.0, 1360.0,
                           1365.0, 1370.0, 1375.0, 1380.0, 1385.0, 1390.0, 1395.0, 1400.0, 1405.0, 1410.0, 1415.0,
                           1420.0, 1425.0, 1430.0, 1435.0, 1440.0, 1445.0, 1450.0, 1455.0, 1460.0, 1465.0, 1470.0,
                           1475.0, 1480.0, 1485.0, 1490.0, 1495.0, 1500.0, 1505.0, 1510.0, 1515.0, 1520.0, 1525.0,
                           1530.0, 1535.0, 1540.0, 1545.0, 1550.0, 1555.0, 1560.0, 1565.0, 1570.0, 1575.0, 1580.0,
                           1585.0, 1590.0, 1595.0, 1600.0, 1605.0, 1610.0, 1615.0, 1620.0, 1625.0, 1630.0, 1635.0,
                           1640.0, 1645.0, 1650.0, 1655.0, 1660.0, 1665.0, 1670.0, 1675.0, 1680.0, 1685.0, 1690.0,
                           1695.0, 1700.0, 1705.0, 1710.0, 1715.0, 1720.0, 1725.0, 1730.0, 1735.0, 1740.0, 1745.0,
                           1750.0, 1755.0, 1760.0, 1765.0, 1770.0, 1775.0, 1780.0, 1785.0, 1790.0, 1795.0,
                           1800.0],
                  'mask': [False, False, False, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, False, False, False, False, False, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True],
                  'prepared_trace': [0.0, 0.0, 0.0, 0.0012706589901055408, 0.001230172378268285,
                                     0.001189767083244026, 0.0011494387815142977, 0.001109196015083663,
                                     0.0010690340188311302, 0.0010289616810141501, 0.00098897379799933,
                                     0.0009490796668017073, 0.0009092736473511736, 0.0008695655124339392,
                                     0.0008299491865265968, 0.0007904349902383879, 0.000751016412398819,
                                     0.0007117043991326564, 0.0006724920021610085, 0.0006333908771420608,
                                     0.000594393635861854, 0.0005555127346355197, 0.000516740341017897,
                                     0.00047808981110202366, 0.00043955286233393897, 0.00040114385872304066,
                                     0.00036285406002863063, 0.00032469895650312004, 0.00028666934081417693,
                                     0.0002487819593173038, 0.00021102712622329348, 0.00017342298693488894,
                                     0.00013595936374065406, 0.0, 0.0, 0.0, 0.0, 0.0, -8.54124084318264e-05,
                                     -0.00012168400851239192, -0.00015777015269102812, -0.000193640913548898,
                                     -0.0002293056261398487, -0.00026472826440528574, -0.0002999146435427042,
                                     -0.00033482093635025297, -0.00036944808169568736, -0.00040374272752020175,
                                     -0.00043769958918977645, -0.0004712537731412492, -0.0005043895832762491,
                                     -0.0005369912628120424, -0.000568749096933821, -0.000596999457840576,
                                     -0.0006081755450153796, -0.0005203760457394867, -4.24275570960909e-05,
                                     0.0019967106450119267, 0.00824998328645213, 0.025470902230998533,
                                     0.061666184760801566, 0.1266911285536651, 0.21623730020863124,
                                     0.3107181658948715, 0.37674223039977695, 0.38202369272497594,
                                     0.3278961221228664, 0.23575713759555478, 0.14236557712467027,
                                     0.07261857726550934, 0.03029217827540741, 0.010518387030549026,
                                     0.002419653410273973, -9.922296779128569e-05, -0.0008249583541146958,
                                     -0.0009832287469567053, -0.0010207438354721604, -0.0010314902769180412,
                                     -0.0010381201593466871, -0.0010437246830709, -0.0010489750910268913,
                                     -0.0010537812065489907, -0.0010582788148848696, -0.001062269844328857,
                                     -0.0010658634934049382, -0.0010687957574359913, -0.0010711540127012085,
                                     -0.0010725996721946934, -0.0010731970876588029, -0.001072521933413136,
                                     -0.0010706138791479235, -0.0010669489460085435, -0.00106153986668923,
                                     -0.0010537454992608306, -0.0010435485373978233, -0.001030168427536924,
                                     -0.0010135535086122599, -0.0009927552492532314, -0.0009676815591610173,
                                     -0.0009371788357173001, -0.0009011059064246127, -0.0008580553418634589,
                                     -0.0008078244092405044, -0.0007486869227242108, -0.0006803603673603599,
                                     -0.0006007121027020298, -0.0005093528746045106, -0.00040362332205057174,
                                     -0.00028298702768856385, -0.00014409025172107618, 1.3812084293644595e-05,
                                     0.0001950060186625695, 0.00040054045044063137, 0.0006359799847417328,
                                     0.0009028323942820015, 0.0012084576276675861, 0.0015550707968721338,
                                     0.0019526070636612894, 0.002404392237260435, 0.0029241441865923215,
                                     0.003516970653909564, 0.0042022958465598986, 0.004988157845440204,
                                     0.005902850059289131, 0.006959367040314155, 0.008200256118773543,
                                     0.009647164971262077, 0.011366415120673766, 0.013395294083275529,
                                     0.015841405363316262, 0.01877133580642354, 0.02236737424087435,
                                     0.026752743787590856, 0.032249562262972334, 0.03909158940818573,
                                     0.047863654004468195, 0.05899999835082356, 0.07352308100183504,
                                     0.09208055758064161, 0.11593718099568388, 0.14490467341860894,
                                     0.1774164827525707, 0.20764161252712904, 0.22527364752231535,
                                     0.2235355322524052, 0.20201983443165664, 0.1708580586436235,
                                     0.13876587670222681, 0.11060466205428494, 0.08802514092347258,
                                     0.07019652294850721, 0.056534668265860406, 0.04583138888895296,
                                     0.03755058497227906, 0.030952333322868363, 0.02573492529055668,
                                     0.021490088610775337, 0.018058939274412596, 0.015213688241907141,
                                     0.012869546479007682, 0.01089439428220178, 0.00924146042112246,
                                     0.007830704240198492, 0.006635373404531025, 0.005604926741913875,
                                     0.004723529678215823, 0.003958050380397417, 0.0032988236708363333,
                                     0.002723395283068994, 0.00222570307635247, 0.0017900743299200871,
                                     0.0014126142857449466, 0.0010820941128451911, 0.0007959572602897272,
                                     0.000545969272349673, 0.0003304188508665844, 0.00014314001147977429,
                                     -1.7034057700372004e-05, -0.0001548083470428695, -0.00027099281385974187,
                                     -0.0003692380571320093, -0.0004501128684254549, -0.000516489313641697,
                                     -0.0005687688507934372, -0.0006092382852280791, -0.0006381803767822993,
                                     -0.0006574347231055133, -0.0006671980742530648, -0.0006689632481769812,
                                     -0.0006628634423315383, -0.0006501190420638013, -0.0006308155108599472,
                                     -0.0006059567835561782, -0.0005755920625089258, -0.0005405516951725381,
                                     -0.0005008572328771453, -0.0004571988032077546, -0.0004095770246682429,
                                     -0.00035856821617240476, -0.0003041575319146607, -0.0002468287408311329],
                  'inspection': {'space': 'prepared',
                                 'interpretation': 'mathematical_inspection',
                                 'inverted': False,
                                 'scale': 1.0,
                                 'offset': -1.1475638309121266e-18,
                                 'fit': [-1.1475638309121266e-18, -1.1475638309121266e-18,
                                         -1.1475638309121266e-18, 0.0012706589901055397, 0.001230172378268284,
                                         0.001189767083244025, 0.0011494387815142967, 0.001109196015083662,
                                         0.001069034018831129, 0.001028961681014149, 0.000988973797999329,
                                         0.0009490796668017061, 0.0009092736473511724, 0.000869565512433938,
                                         0.0008299491865265956, 0.0007904349902383867, 0.0007510164123988179,
                                         0.0007117043991326552, 0.0006724920021610073, 0.0006333908771420596,
                                         0.0005943936358618528, 0.0005555127346355185, 0.0005167403410178958,
                                         0.0004780898111020225, 0.0004395528623339378, 0.0004011438587230395,
                                         0.0003628540600286295, 0.0003246989565031189, 0.0002866693408141758,
                                         0.0002487819593173027, 0.00021102712622329234, 0.0001734229869348878,
                                         0.00013595936374065292, -1.1475638309121266e-18, -1.1475638309121266e-18,
                                         -1.1475638309121266e-18, -1.1475638309121266e-18,
                                         -1.1475638309121266e-18, -8.541240843182756e-05, -0.00012168400851239307,
                                         -0.00015777015269102926, -0.00019364091354889915,
                                         -0.00022930562613984984, -0.0002647282644052869, -0.0002999146435427053,
                                         -0.0003348209363502541, -0.0003694480816956885, -0.0004037427275202029,
                                         -0.0004376995891897776, -0.00047125377314125033, -0.0005043895832762503,
                                         -0.0005369912628120436, -0.0005687490969338222, -0.0005969994578405772,
                                         -0.0006081755450153808, -0.0005203760457394878, -4.242755709609205e-05,
                                         0.0019967106450119254, 0.008249983286452128, 0.025470902230998533,
                                         0.061666184760801566, 0.1266911285536651, 0.21623730020863124,
                                         0.3107181658948715, 0.37674223039977695, 0.38202369272497594,
                                         0.3278961221228664, 0.23575713759555478, 0.14236557712467027,
                                         0.07261857726550934, 0.03029217827540741, 0.010518387030549024,
                                         0.002419653410273972, -9.922296779128684e-05, -0.000824958354114697,
                                         -0.0009832287469567064, -0.0010207438354721615, -0.0010314902769180422,
                                         -0.0010381201593466882, -0.001043724683070901, -0.0010489750910268924,
                                         -0.0010537812065489918, -0.0010582788148848707, -0.0010622698443288581,
                                         -0.0010658634934049393, -0.0010687957574359924, -0.0010711540127012096,
                                         -0.0010725996721946945, -0.001073197087658804, -0.001072521933413137,
                                         -0.0010706138791479246, -0.0010669489460085446, -0.001061539866689231,
                                         -0.0010537454992608317, -0.0010435485373978244, -0.001030168427536925,
                                         -0.001013553508612261, -0.0009927552492532325, -0.0009676815591610185,
                                         -0.0009371788357173013, -0.0009011059064246139, -0.0008580553418634601,
                                         -0.0008078244092405056, -0.000748686922724212, -0.0006803603673603611,
                                         -0.000600712102702031, -0.0005093528746045118, -0.0004036233220505729,
                                         -0.000282987027688565, -0.00014409025172107732, 1.3812084293643449e-05,
                                         0.00019500601866256835, 0.0004005404504406302, 0.0006359799847417316,
                                         0.0009028323942820003, 0.001208457627667585, 0.0015550707968721327,
                                         0.0019526070636612883, 0.0024043922372604335, 0.00292414418659232,
                                         0.0035169706539095627, 0.004202295846559898, 0.004988157845440203,
                                         0.00590285005928913, 0.006959367040314154, 0.00820025611877354,
                                         0.009647164971262076, 0.011366415120673764, 0.013395294083275527,
                                         0.015841405363316262, 0.01877133580642354, 0.02236737424087435,
                                         0.026752743787590856, 0.032249562262972334, 0.03909158940818573,
                                         0.047863654004468195, 0.05899999835082356, 0.07352308100183504,
                                         0.09208055758064161, 0.11593718099568388, 0.14490467341860894,
                                         0.1774164827525707, 0.20764161252712904, 0.22527364752231535,
                                         0.2235355322524052, 0.20201983443165664, 0.1708580586436235,
                                         0.13876587670222681, 0.11060466205428494, 0.08802514092347258,
                                         0.07019652294850721, 0.056534668265860406, 0.04583138888895296,
                                         0.03755058497227906, 0.030952333322868363, 0.02573492529055668,
                                         0.021490088610775337, 0.018058939274412596, 0.01521368824190714,
                                         0.01286954647900768, 0.010894394282201778, 0.009241460421122458,
                                         0.00783070424019849, 0.006635373404531024, 0.005604926741913874,
                                         0.004723529678215822, 0.003958050380397416, 0.003298823670836332,
                                         0.0027233952830689927, 0.0022257030763524685, 0.001790074329920086,
                                         0.0014126142857449455, 0.00108209411284519, 0.000795957260289726,
                                         0.0005459692723496718, 0.00033041885086658327, 0.00014314001147977315,
                                         -1.7034057700373152e-05, -0.00015480834704287064, -0.000270992813859743,
                                         -0.0003692380571320104, -0.000450112868425456, -0.0005164893136416982,
                                         -0.0005687688507934384, -0.0006092382852280803, -0.0006381803767823005,
                                         -0.0006574347231055145, -0.000667198074253066, -0.0006689632481769824,
                                         -0.0006628634423315395, -0.0006501190420638025, -0.0006308155108599484,
                                         -0.0006059567835561794, -0.000575592062508927, -0.0005405516951725393,
                                         -0.0005008572328771464, -0.00045719880320775575, -0.00040957702466824406,
                                         -0.0003585682161724059, -0.00030415753191466183,
                                         -0.00024682874083113406],
                                 'residual': [1.1475638309121266e-18, 1.1475638309121266e-18,
                                              1.1475638309121266e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1475638309121266e-18,
                                              1.1475638309121266e-18, 1.1475638309121266e-18,
                                              1.1475638309121266e-18, 1.1475638309121266e-18,
                                              1.1519648082658485e-18, 1.1519648082658485e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.145188544687814e-18, 1.3010426069826053e-18,
                                              1.734723475976807e-18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 1.734723475976807e-18, 1.3010426069826053e-18,
                                              1.1519648082658485e-18, 1.1926223897340549e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1468826105823227e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.3010426069826053e-18,
                                              1.3010426069826053e-18, 1.3010426069826053e-18,
                                              8.673617379884035e-19, 8.673617379884035e-19, 8.673617379884035e-19,
                                              8.673617379884035e-19, 1.734723475976807e-18, 1.734723475976807e-18,
                                              1.734723475976807e-18, 1.734723475976807e-18, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              1.734723475976807e-18, 1.734723475976807e-18, 1.734723475976807e-18,
                                              1.734723475976807e-18, 1.734723475976807e-18, 8.673617379884035e-19,
                                              8.673617379884035e-19, 8.673617379884035e-19, 8.673617379884035e-19,
                                              1.3010426069826053e-18, 1.3010426069826053e-18,
                                              1.3010426069826053e-18, 1.0842021724855044e-18,
                                              1.0842021724855044e-18, 1.0842021724855044e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1485766764768313e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1926223897340549e-18,
                                              1.1926223897340549e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18,
                                              1.1384122811097797e-18, 1.1384122811097797e-18],
                                 'mask': [False, False, False, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, False,
                                          False, False, False, False, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True, True,
                                          True, True, True, True, True, True, True, True, True, True, True,
                                          True],
                                 'sse': 2.1616591791370403e-34}},
     'result_digest': '96bb27d1e3e162e412ab6cf73f150ed3f862a5f7a4033058fd8d5fda1ae8bbc4'}
    result = pearson_golden_case(system)
    assert result.model_dump(mode="json") == expected
    assert result.model_dump() == expected
    assert json.loads(result.model_dump_json()) == expected
    # The response must retain its shape after the service cache round trip.
    assert type(result).model_validate(expected).model_dump(mode="json") == expected
