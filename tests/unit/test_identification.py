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
                            array_locator=object_locator(sha), array_sha256=sha)
    rows = {(ReferenceReleaseORM, release_id): release, (ReferenceTraceORM, trace_id): trace,
            (ReferenceTemplateORM, template_id): template, (ReferenceFileORM, file_id): file}
    session = Mock()
    session.get.side_effect = lambda cls, identity: rows.get((cls, identity))
    store = Mock()
    store.read_verified.side_effect = lambda locator, expected: verify_bytes(data, expected)
    resolver = Mock()
    resolver.resolve.return_value = (x, y)
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
    system.file.metadata_json = {}
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
