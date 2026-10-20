"""Analytic synthetic arrays and invented records; no measurement fixtures."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from tests.unit.test_evidence_api import evidence_api  # noqa: F401
from tests.unit.test_reference_release import snapshot  # noqa: F401
from sherloc_pipeline.core.identification.mixture import background_basis, fit_mixture, retrieve
from sherloc_pipeline.core.identification.prep import Prepared, grid_for
from sherloc_pipeline.core.identification.recipe import load_mixture_recipe, load_recipe, recipe_for_release
from sherloc_pipeline.web.feature_policy import FeaturePolicy


@pytest.fixture
def recipe():
    return load_mixture_recipe()[0]


def prepared(vector):
    return Prepared(np.asarray(vector, float), np.ones(len(vector), dtype=bool), {"snr_proxy": 100.})


def analytic_vectors(recipe):
    """Orthonormal invented bands, exactly orthogonal to the broad background."""
    size = len(grid_for(recipe))
    bg = background_basis(size, np.ones(size, dtype=bool))
    x = np.linspace(0., 1., size)
    bands = np.column_stack([np.exp(-((x-c)/.017)**2) for c in (.23, .51, .79)])
    bands -= bg @ np.linalg.lstsq(bg, bands, rcond=None)[0]
    vectors, _ = np.linalg.qr(bands)
    return vectors, bg


def candidates(vectors):
    return [{"template_id": str(i+1)*64, "species_name": f"Syntheticite {i}",
             "vector": v, "mask": np.ones(len(v), dtype=bool), "shift_cm_1": 0.,
             "retrieval_score": 1.-i*.1} for i, v in enumerate(vectors.T)]


@pytest.mark.parametrize("weights", [[1., 0., 0.], [.8, .6, 0.]])
def test_analytic_one_and_two_components(recipe, weights):
    vectors, bg = analytic_vectors(recipe)
    target = vectors @ weights + bg @ np.array([.04, .07, .02])
    result = fit_mixture(prepared(target), candidates(vectors), recipe)
    assert result["reason"] is None
    assert result["diagnostics"]["converged"]
    expected = np.maximum(np.array(weights)-.03, 0)
    np.testing.assert_allclose(result["diagnostics"]["coefficients"], expected, atol=2e-6)
    np.testing.assert_allclose(result["background_coefficients"], [.04, .07, .02], atol=2e-6)
    np.testing.assert_allclose(result["background_only_coefficients"], [.04, .07, .02], atol=2e-6)
    assert result["background_only_sse"] == pytest.approx(np.dot(weights, weights), abs=1e-10)
    assert result["sse"] == pytest.approx(np.sum((np.array(weights)-expected)**2), abs=1e-10)
    assert result["support"] == pytest.approx(1-result["sse"]/result["background_only_sse"])
    assert sum(c["weight"] for c in result["components"]) == pytest.approx(1.)
    assert len(result["components"]) == np.count_nonzero(weights)
    np.testing.assert_allclose(result["evidence"]["residual"], target-np.array(result["evidence"]["fit"]))


def test_background_is_bounded_and_null_is_independent(recipe):
    vectors, bg = analytic_vectors(recipe)
    target = .8*vectors[:, 0] + bg @ np.array([1., .9, .8])
    result = fit_mixture(prepared(target), candidates(vectors), recipe)
    assert all(0 <= c <= .10 for c in result["background_coefficients"])
    assert all(0 <= c <= .10 for c in result["background_only_coefficients"])
    np.testing.assert_allclose(result["background_coefficients"], [.1, .1, .1])
    expected_null = target-bg @ np.array([.1, .1, .1])
    assert result["background_only_sse"] == pytest.approx(expected_null @ expected_null)
    assert result["support"] == pytest.approx(max(0, 1-result["sse"]/max(result["background_only_sse"], 1e-12)))


def test_nonnegative_at_most_two_and_cutoff(recipe):
    vectors, _ = analytic_vectors(recipe)
    result = fit_mixture(prepared(vectors @ [.8, .5, .3]), candidates(vectors), recipe)
    assert np.count_nonzero(result["diagnostics"]["coefficients"]) == 2
    assert all(c >= 0 for c in result["diagnostics"]["coefficients"])
    assert len(result["components"]) == 2
    negative = fit_mixture(prepared(-vectors[:, 0]), candidates(vectors), recipe)
    assert negative["components"] == []
    assert negative["reason"] == "no_component_above_cutoff"
    tiny = fit_mixture(prepared(.07*vectors[:, 0]), candidates(vectors), recipe)
    assert tiny["diagnostics"]["coefficients"][0] == pytest.approx(.04)
    assert tiny["components"] == []


def test_null_floor_and_low_support_reject(recipe):
    vectors, bg = analytic_vectors(recipe)
    only_background = fit_mixture(prepared(bg @ np.array([.03, .02, .04])), candidates(vectors), recipe)
    assert only_background["background_only_sse"] < 1e-12
    assert only_background["support"] == pytest.approx(max(0, 1-only_background["sse"]/1e-12))
    assert only_background["reason"] == "no_component_above_cutoff"
    for amount, reason in [(.2, "unexplained_structure"), (.6, "below_support_threshold")]:
        result = fit_mixture(prepared(amount*vectors[:, 0] + vectors[:, 2]), candidates(vectors[:, :1]), recipe)
        assert result["reason"] == reason
        assert result["support"] < .60


def test_collinear_alternatives_are_deterministic(recipe):
    vectors, _ = analytic_vectors(recipe)
    rows = candidates(np.column_stack([vectors[:, 0]]*3))
    first = fit_mixture(prepared(vectors[:, 0]), rows, recipe)
    assert first == fit_mixture(prepared(vectors[:, 0]), rows, recipe)
    assert first["components"][0]["template_id"] == rows[0]["template_id"]
    assert len(first["components"]) == 1


@pytest.mark.parametrize("bad", [0., np.nan, np.inf])
def test_degenerate_fit(recipe, bad):
    vectors, _ = analytic_vectors(recipe)
    result = fit_mixture(prepared(np.full(len(vectors), bad)), candidates(vectors), recipe)
    assert result["reason"] == "degenerate_fit"
    assert result["support"] is None
    assert not result["diagnostics"]["finite_solution"]


def test_missing_and_degenerate_references(recipe):
    vectors, _ = analytic_vectors(recipe)
    query = prepared(vectors[:, 0])
    assert fit_mixture(query, [], recipe)["reason"] == "no_compatible_species_references"
    assert fit_mixture(query, candidates(np.zeros_like(vectors)), recipe)["reason"] == "degenerate_fit"
    assert retrieve(query, prepared(np.zeros(len(vectors))), recipe) is None


@pytest.mark.parametrize("stage", range(3))
def test_iteration_limit_is_never_validated(recipe, stage):
    vectors, _ = analytic_vectors(recipe)
    iterations = list(recipe.solver_iterations)
    iterations[stage] = 1
    limited = recipe.model_copy(update={"solver_iterations": iterations})
    result = fit_mixture(prepared(vectors[:, 0]), candidates(vectors), limited)
    assert result["reason"] == "fit_not_converged"
    assert result["diagnostics"]["finite_solution"]
    assert not result["diagnostics"]["converged"]
    info = result["diagnostics"][("initial_fit", "active_refit", "background_only")[stage]]
    assert info["iteration_limit"]
    assert not result["evidence"]["converged"]


def test_broadened_cosine_retrieval(recipe):
    x = grid_for(recipe)
    vector = np.exp(-((x-1237.)/7.)**2) + .6*np.exp(-((x-1537.)/9.)**2)
    query = gaussian_filter1d(vector, 20./2.35482/5., mode="nearest")
    query = np.interp(x-5., x, query, left=0., right=0.)
    match = retrieve(prepared(query), prepared(vector), recipe)
    assert match["extra_fwhm_cm1"] == 20.
    assert match["shift_cm_1"] == 5.
    assert match["retrieval_score"] == pytest.approx(1.)
    assert "support" not in match


def test_joint_mask_and_residual(recipe):
    vectors, _ = analytic_vectors(recipe)
    rows = candidates(vectors)
    rows[0]["mask"][:5] = False
    result = fit_mixture(prepared(vectors[:, 0]), rows, recipe)
    evidence = result["evidence"]
    assert evidence["mask"][:5] == [False]*5
    assert evidence["fit"][:5] == [None]*5
    mask = np.array(evidence["mask"])
    np.testing.assert_allclose(np.array(evidence["prepared_trace"])[mask].astype(float),
                               np.array(evidence["fit"])[mask].astype(float) + np.array(evidence["residual"])[mask].astype(float),
                               atol=1e-14)
    rows[1]["mask"][:30] = False
    assert fit_mixture(prepared(vectors[:, 0]), rows, recipe)["reason"] == "insufficient_joint_overlap"


def test_frozen_method_hashes_and_pearson_unchanged(recipe):
    pearson, pearson_id = load_recipe()
    assert pearson_id == "3a2d27da061756c69ef9c2473a59fbd2def13fe2df1ae3be429dcab37f010bce"
    assert load_mixture_recipe() == load_mixture_recipe()
    assert load_mixture_recipe()[1] != pearson_id
    assert recipe_for_release(load_mixture_recipe()[1])[0] == recipe
    assert pearson.method_id == "raman-pearson" and pearson.support_min == .8
    assert (recipe.support_min, recipe.status, recipe.threshold_source) == (.6, "provisional", "fixed_provisional")
    with pytest.raises(ValueError):
        recipe_for_release("f"*64)
    assert not FeaturePolicy().identification.enabled
    assert FeaturePolicy().identification.allowed_method_releases == []


@pytest.fixture
def mixture_api(monkeypatch, request):
    # The existing API fixture builds an entirely invented local catalog.
    monkeypatch.setattr("tests.unit.test_evidence_api.load_recipe", load_mixture_recipe)
    return request.getfixturevalue("evidence_api")


def test_synthetic_mixture_through_api_and_exact_binding(mixture_api):
    s = mixture_api
    listed = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert listed.json()['binding']['method_release_id'] == load_mixture_recipe()[1]
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['status'] == 'candidate'
    assert payload['mixture']['support'] > .99
    assert payload['mixture']['diagnostics']['converged']
    assert payload['ranking'][0]['support'] is None
    assert payload['ranking'][0]['retrieval_score'] > .99
    assert payload['method']['method_id'] == 'raman-mixture'
    assert payload['method']['release_id'] == load_mixture_recipe()[1]
    assert payload['evidence']['inspection'] is None
    assert payload['evidence']['mixture_fit']['label'] == 'mixture fit'
    assert payload['mixture']['weight_basis'] == 'normalized_spectral_contributions'
    assert s.client.post('/api/identify', json=s.body).json() == payload
    assert not any(word in json.dumps(payload).lower() for word in ('probability', 'confidence'))
    assert s.client.post('/api/identify', json={**s.body, 'method_release_id': load_recipe()[1]}).status_code == 422
    original = s.app.state.feature_policy
    # Leave identification on, but remove permission for this exact method release.
    s.app.state.feature_policy = original.model_copy(update={'identification': original.identification.model_copy(
        update={'allowed_method_releases': []})})
    s.app.state.reference_store = Mock(read_verified=Mock(side_effect=AssertionError('array read after denial')))
    assert s.client.post('/api/identify', json=s.body).status_code == 422
    s.app.state.reference_store.read_verified.assert_not_called()


def test_top_five_species_and_nonconvergence_in_result(recipe):
    """Compute adapter preserves retrieval order, caps it, and abstains on limits."""
    from sherloc_pipeline.core.identification.prep import prepare
    from sherloc_pipeline.models.identification import IdentificationRequest
    from sherloc_pipeline.services.identification import compute_identification
    from sherloc_pipeline.services.reference_import import array_digest, digest
    from tests.unit.test_identification import bands, selection, trace_metadata

    x, y = bands()
    recipe_id = load_mixture_recipe()[1]
    templates = []
    identities = [digest(f"synthetic-template-{i}") for i in range(7)]
    for index, identity in enumerate(identities):
        name = f"Syntheticite {index}"
        metadata = {"name": f"synthetic-template-{index}", "species_name": name,
                    "recipe_sha256": recipe_id, "grouping_evidence": "invented specimen",
                    "members": [{"record_id": "synthetic-record", "file_id": "synthetic-file", "trace_index": 0}]}
        templates.append(({"template_id": identity, "metadata_json": metadata,
                           "recipe_sha256": recipe_id, "member_trace_ids": [digest(f"synthetic-trace-{index}")]},
                          [{"metadata_json": trace_metadata(species_name=name).model_dump(mode="json"),
                            "source_key": "synthetic-record:synthetic-file:0", "trace_index": 0,
                            "arrays": {"x": x, "y": y}, "axis_sha256": array_digest(x),
                            "intensity_sha256": array_digest(y)}]))
    request = IdentificationRequest(selection=selection(), reference_release_id="a"*64,
                                    reference_subset_id=digest(sorted(identities)), method_release_id=recipe_id,
                                    template_ids=sorted(identities))
    query = prepare(x, y, recipe)
    result = compute_identification(request, recipe, recipe_id, {}, query, templates)
    assert result.status == "candidate"
    assert len(result.ranking) == 5
    assert [item.species_name for item in result.ranking] == [f"Syntheticite {i}" for i in range(5)]
    assert all(item.support is None and item.retrieval_score > .99 for item in result.ranking)
    assert result.mixture.support > .99
    assert len(result.mixture.components) == 1  # Collinear alternatives share no claim of unique identity.
    limited = recipe.model_copy(update={"solver_iterations": [1, 200, 100]})
    result = compute_identification(request, limited, recipe_id, {}, query, templates)
    assert result.status == "abstained"
    assert result.candidate is None
    assert result.reasons == ["fit_not_converged"]
    assert not result.mixture.diagnostics["converged"]
