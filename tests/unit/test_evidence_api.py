"""Synthetic route-to-engine evidence, policy and resource-bound regressions."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock
import zlib

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy import create_engine, event, update
from sqlalchemy.orm import Session

from tests.unit.test_reference_release import snapshot  # noqa: F401
from tests.unit._spawn_helpers import HangingStore, _crash, _hang, ready_path
from sherloc_pipeline.core.identification.recipe import load_recipe
from sherloc_pipeline.database.models import Base, ScanORM, ScanPointORM, SolORM, SpectrumORM
from sherloc_pipeline.database.reference_models import ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.runtime import worker as runtime_worker
from sherloc_pipeline.services.identification_runtime import ExecutionFailed, IdentificationRuntime, ResponseCache, WorkRejected, run_process
from sherloc_pipeline.services.reference_import import digest, import_release
from sherloc_pipeline.services.reference_store import LocalReferenceStore
from sherloc_pipeline.web.app import create_app
from sherloc_pipeline.web.auth import require_authenticated_request
from sherloc_pipeline.web.feature_policy import FeaturePolicy


@pytest.fixture
def evidence_api(tmp_path, snapshot, monkeypatch, request):
    root, manifest, serving = snapshot
    x = np.linspace(720., 1880., 581)
    y = np.exp(-((x - 1137.) / 19.) ** 2) + .7 * np.exp(-((x - 1573.) / 29.) ** 2) + .0002 * x
    buffer = io.BytesIO()
    np.savez(buffer, x=x, y=y)
    data = buffer.getvalue()
    (root / 'arrays.npz').write_bytes(data)
    file = manifest['files'][0]
    file.update(arrays_sha256=hashlib.sha256(data).hexdigest(), arrays_size=len(data))
    trace = file['traces'][0]
    trace.update(excitation_value=250., excitation_unit='nm', excitation_evidence='synthetic stated excitation',
                 instrument='Synthetic instrument', sample_form='pure')
    for index in (1, 2):
        extra = copy.deepcopy(trace)
        extra.update(trace_index=index, source_label=f'Synthetic trace {index}')
        if index == 2:
            extra.update(rights_status='unknown', license_identifier=None, license_source=None)
        file['traces'].append(extra)
    manifest['templates'][0]['recipe_sha256'] = load_recipe()[1]
    manifest['templates'][0]['members'].append({**manifest['templates'][0]['members'][0], 'trace_index': 1})
    options = getattr(request, 'param', {})
    for index in range(1, options.get('templates', 1)):
        extra = copy.deepcopy(manifest['templates'][0])
        extra['name'] = f'synthetic-template-{index}'
        manifest['templates'].append(extra)
    if options.get('unpermitted_template'):
        extra = copy.deepcopy(manifest['templates'][0])
        extra['name'] = 'synthetic-unpermitted-template'
        extra['members'] = [{**extra['members'][0], 'trace_index': 2}]
        manifest['templates'].append(extra)
    (root / 'snapshot.json').write_text(json.dumps(manifest))
    serving.write_text(json.dumps({'approved_trace_keys': [f'synthetic-0001:synthetic-file-1:{i}' for i in range(3)]}))
    database = tmp_path / 'synthetic.db'
    engine = create_engine(f'sqlite:///{database}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    objects = tmp_path / 'objects'
    imported = import_release(root, database, objects, serving)
    with Session(engine) as session:
        release = session.get(ReferenceReleaseORM, imported.release_id)
        template = session.query(ReferenceTemplateORM).filter_by(name='synthetic-template').one()
        template_ids = sorted(row.template_id for row in session.query(ReferenceTemplateORM).all()
                              if row.name != 'synthetic-unpermitted-template')
        trace_ids = list(template.member_trace_ids)
        unknown = session.query(ReferenceTraceORM).filter_by(rights_known=False).one().trace_revision_id
        values = {'evidence_desk': True, 'references': {'mode': 'all_approved', 'release_id': release.release_id,
                  'serving_policy_version': release.serving_policy_sha256, 'subset_id': digest(template_ids)},
                  'identification': {'enabled': True, 'experimental_candidates': True, 'allowed_method_releases': [
                      {'method_release_id': load_recipe()[1], 'reference_release_id': release.release_id,
                       'reference_subset_id': digest(template_ids)}]}}
        binding = {**values['identification']['allowed_method_releases'][0], 'template_ids': template_ids}
        session.add(SolORM(sol_number=9999))
        scan = ScanORM(id='synthetic-scan', sol_number=9999, scan_name='Synthetic scan', target='Synthetic Target',
                       scan_id='ss__9999_0000000000_000rrs__synthetic', sclk_start=0, n_points=1,
                       n_channels=len(x), laser_wavelength_nm=250., target_type='rock', data_source='pds4',
                       pds4_metadata={'product_id': 'ss__9999_0000000000_000rrs__synthetic', 'intensity_unit': 'synthetic units'})
        session.add(scan)
        session.add(ScanPointORM(id='synthetic-point', scan_id=scan.id, point_index=0))
        wavelength = (1. / (1. / 250. - x / 1e7)).astype(np.float32)
        session.add(SpectrumORM(id='synthetic-spectrum', scan_point_id='synthetic-point', region='R1',
                                spectrum_type='laser_normalized', processing_level='raw',
                                intensities=zlib.compress(y.astype(np.float32).tobytes()),
                                wavelengths=zlib.compress(wavelength.tobytes()), wavelength_source='pds_embedded'))
        session.commit()
    monkeypatch.setenv('PHASE_TIER', 'team')
    monkeypatch.setenv('SHERLOC_AUTH_MODE', 'dev')
    monkeypatch.setenv('SHERLOC_FEATURE_POLICY', json.dumps(values))
    config = SimpleNamespace(fitting={}, fluorescence_fitting={}, preprocessing={}, paths={},
                             wavelength=SimpleNamespace(laser_wavelength=250., cutoff_channel=1, n_channels=len(x),
                                                        raman_coefficients=[0., 1., 200.], fluorescence_coefficients=[0., 1., 200.]))
    app = create_app(engine=engine, config=config)
    app.dependency_overrides[require_authenticated_request] = lambda: None
    app.state.reference_store = LocalReferenceStore(objects)
    client = TestClient(app)
    source = client.get('/api/spectra/synthetic-scan/point/0', params={'despike_method': 'none'}).json()
    body = {**binding, 'scan_id': 'synthetic-scan', 'selection': {'mode': 'point', 'index': 0}, 'region': 'R1',
            'spectrum_type': 'laser_normalized', 'expected_source_digest': source['provenance']['source_digest']}
    yield SimpleNamespace(**locals())
    client.close()
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)
    engine.dispose()


def test_route_to_engine_candidate_native_provenance_and_cache(evidence_api):
    s = evidence_api
    listed = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert listed.status_code == 200
    assert len(listed.json()['items']) == 2
    assert listed.json()['binding'] == s.binding
    assert listed.headers['cache-control'] == 'private, no-store'
    response = s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id})
    assert response.status_code == 200
    assert response.json()['x'] == s.x.tolist()
    assert response.json()['provenance']['citation_text'] == 'Synthetic citation'
    assert 'locator' not in response.text and 'x_key' not in response.text
    result = s.client.post('/api/identify', json=s.body)
    assert result.status_code == 200, result.text
    assert result.headers['cache-control'] == 'private, no-store'
    payload = result.json()
    assert payload['status'] == 'candidate'
    assert payload['ranking'][0]['support'] > .99
    assert len(payload['ranking'][0]['citations']) == 2
    assert payload['method']['support_min'] == .80
    assert payload['method']['status'] == 'provisional'
    assert payload['source_digest'] == s.body['expected_source_digest']
    assert payload['input']['selection']['contributor_indices'] == [0]
    assert s.client.post('/api/identify', json=s.body).json() == payload


@pytest.mark.parametrize('tier', ['team', 'public'])
@pytest.mark.parametrize('refs,enabled,experimental', [(a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)])
def test_all_feature_and_tier_guards(evidence_api, tier, refs, enabled, experimental):
    s = evidence_api
    s.app.state.access_mode = 'internal' if tier == 'team' else 'public'
    original = s.app.state.feature_policy
    s.app.state.feature_policy = original.model_copy(update={
        'references': original.references.model_copy(update={'mode': 'all_approved' if refs else 'off'}),
        'identification': original.identification.model_copy(update={'enabled': enabled, 'experimental_candidates': experimental})})
    permitted = tier == 'team' and refs
    for path in ('/api/references', f'/api/references/{s.trace_ids[0]}'):
        response = s.client.get(path, params={'release_id': s.imported.release_id})
        assert response.status_code == (200 if permitted else 404)
        assert response.headers['cache-control'] == 'private, no-store'
    response = s.client.post('/api/identify', json={**s.body, 'expected_source_digest': '0' * 64})
    assert response.status_code == (409 if permitted and enabled and experimental else 404)


def test_authentication_and_unknown_rights(evidence_api):
    s = evidence_api
    assert s.client.get(f'/api/references/{s.unknown}', params={'release_id': s.imported.release_id}).status_code == 404
    def denied():
        raise HTTPException(401, 'Synthetic authentication refusal')
    s.app.dependency_overrides[require_authenticated_request] = denied
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id}).status_code == 401
    assert s.client.post('/api/identify', json=s.body).status_code == 401


def test_member_permission_rechecked_before_cache_or_source(evidence_api):
    s = evidence_api
    assert s.client.post('/api/identify', json=s.body).status_code == 200
    original = s.app.state.feature_policy
    s.app.state.feature_policy = original.model_copy(update={'references': original.references.model_copy(
        update={'mode': 'allowlist', 'allowed_trace_ids': [s.trace_ids[0]]})})
    s.app.state.reference_store = Mock(read_verified=Mock(side_effect=AssertionError('array read after denial')))
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 422
    assert s.client.get(f'/api/references/{s.trace_ids[1]}', params={'release_id': s.imported.release_id}).status_code == 404
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id}).json()['binding'] is None


@pytest.mark.parametrize('key', ['url', 'source_url', 'object_key', 'locator', 'arrays'])
def test_no_source_or_reference_parameter_bypass(evidence_api, key):
    s = evidence_api
    assert s.client.post('/api/identify', json={**s.body, key: 'synthetic-invalid'}).status_code == 422
    assert s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id, key: 'synthetic-invalid'}).status_code == 422


def test_cursor_is_bounded_and_policy_bound(evidence_api):
    s = evidence_api
    first = s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 1}).json()
    second = s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 1, 'cursor': first['next_cursor']}).json()
    assert len(first['items']) == len(second['items']) == 1
    assert first['items'][0]['trace_revision_id'] != second['items'][0]['trace_revision_id']
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 101}).status_code == 422
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'cursor': 'invalid'}).status_code == 422
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'modality': 'fluorescence', 'cursor': first['next_cursor']}).status_code == 422


def test_source_change_conflict_and_scientific_abstention(evidence_api):
    s = evidence_api
    assert s.client.post('/api/identify', json={**s.body, 'expected_source_digest': '0' * 64}).status_code == 409
    with Session(s.engine) as session:
        session.execute(update(SpectrumORM).values(intensities=zlib.compress(np.zeros(len(s.x), dtype=np.float32).tobytes())))
        session.commit()
    assert s.client.post('/api/identify', json=s.body).status_code == 409
    fresh = s.client.get('/api/spectra/synthetic-scan/point/0', params={'despike_method': 'none'}).json()
    response = s.client.post('/api/identify', json={**s.body, 'expected_source_digest': fresh['provenance']['source_digest']})
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'abstained'
    assert response.json()['reasons'] == ['constant_signal']


def test_execution_failure_is_http_error(evidence_api, monkeypatch):
    monkeypatch.setattr('sherloc_pipeline.services.identification.run_process', Mock(side_effect=ExecutionFailed('synthetic failure')))
    response = evidence_api.client.post('/api/identify', json=evidence_api.body)
    assert response.status_code == 503
    assert response.json()['detail']['code'] == 'execution_failed'
    assert 'status' not in response.json()


def test_service_concurrency_and_separate_rate_limit(evidence_api):
    s = evidence_api
    runtime = s.app.state.identification_runtime
    runtime.slots.acquire()
    try:
        assert s.client.post('/api/identify', json=s.body).status_code == 429
    finally:
        runtime.slots.release()
    for _ in range(9):
        assert s.client.post('/api/identify', json={**s.body, 'expected_source_digest': '0' * 64}).status_code == 409
    assert s.client.post('/api/identify', json=s.body).status_code == 429
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id}).status_code == 200
    runtime.slots.acquire()
    try:
        assert s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id}).status_code == 429
    finally:
        runtime.slots.release()


@pytest.mark.parametrize('selection', [{'mode': 'subset', 'indices': [0, 1]}, {'mode': 'average'}, {'mode': 'point', 'index': 1}])
def test_selection_rejected_before_array_reads(evidence_api, selection):
    s = evidence_api
    s.app.state.reference_store = Mock(read_verified=Mock(side_effect=AssertionError('array read')))
    assert s.client.post('/api/identify', json={**s.body, 'selection': selection}).status_code == 422


def test_source_metadata_and_decompression_limits(evidence_api):
    s = evidence_api
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(n_channels=1000000))
        session.commit()
    assert s.client.post('/api/identify', json=s.body).status_code == 429
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(n_channels=len(s.x)))
        session.execute(update(SpectrumORM).values(intensities=zlib.compress(b'\0' * 1000000)))
        session.commit()
    assert s.client.post('/api/identify', json=s.body).status_code == 429


def test_deadline_kills_and_reaps_worker(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic-worker.txt'
    # Record the real child's pid at the class run_process spawns, even if it never reaches _hang.
    process_class = runtime_worker.multiprocessing.get_context('spawn').Process
    original_start = process_class.start
    spawned = []

    def recording_start(process):
        original_start(process)
        spawned.append(process.pid)

    monkeypatch.setattr(process_class, 'start', recording_start)
    started = time.monotonic()
    with pytest.raises(ExecutionFailed, match='time limit'):
        run_process(_hang, (str(path),), 3.)
    returned = time.monotonic()
    monkeypatch.undo()
    assert len(spawned) == 1 and spawned[0], 'spawn wrapper did not capture the worker pid'
    assert returned - started >= 3. - .05
    with pytest.raises(ProcessLookupError):
        os.kill(spawned[0], 0)
    # Worker start-up (spawn and imports) counts against the deadline but is not what this test measures.
    assert path.exists(), (f'worker did not start within the 3 s deadline ({returned - started:.2f} s elapsed; '
                           'host load); SIGTERM-to-kill escalation not exercised')
    assert int(path.read_text()) == spawned[0]
    ready = float(Path(ready_path(path)).read_text())
    assert returned - max(started + 3., ready) < 2.
    with pytest.raises(ExecutionFailed, match='worker failed'):
        run_process(_crash, (), 3.)


def test_cache_is_bounded_detached_and_scoped():
    cache = ResponseCache(capacity=2)
    one = ('team', 'synthetic-policy-1', 'release', 'source', 'method')
    cache.put(one, {'x': [1]})
    retrieved = cache.get(one)
    retrieved['x'][0] = 5
    assert cache.get(one) == {'x': [1]}
    for index in (0, 1, 2, 3, 4):
        changed = list(one)
        changed[index] += '-changed'
        assert cache.get(tuple(changed)) is None
    cache.put(('second',), {})
    cache.put(('third',), {})
    assert cache.get(one) is None
    expired = ResponseCache(ttl=0)
    expired.put(one, {})
    assert expired.get(one) is None
    # A fresh app has its own slots and result cache.
    assert IdentificationRuntime().cache.get(one) is None


def test_store_deadline_is_execution_failure_and_releases_slot(evidence_api, tmp_path):
    s = evidence_api
    policy = s.app.state.feature_policy
    s.app.state.feature_policy = policy.model_copy(update={
        'identification': policy.identification.model_copy(update={'timeout_seconds': 8})})
    path = tmp_path / 'synthetic-store-worker.txt'
    s.app.state.reference_store = HangingStore(str(path))
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 503, response.text
    assert response.json()['detail']['code'] == 'execution_failed'
    pid = int(path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert s.app.state.identification_runtime.slots.acquire(blocking=False)
    s.app.state.identification_runtime.slots.release()


@pytest.mark.parametrize('source_kind', ['stored', 'legacy'])
def test_detached_source_matches_canonical_provenance(evidence_api, source_kind):
    s = evidence_api
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(data_source='loupe', loupe_metadata={'intensity_unit': 'synthetic units'}))
        session.execute(update(ScanPointORM).values(photodiode_mean=3.))
        fields = {'spectrum_type': 'dark_subtracted'}
        if source_kind == 'legacy':
            fields['wavelengths'] = None
            fields['wavelength_source'] = None
        session.execute(update(SpectrumORM).values(**fields))
        session.commit()
    source = s.client.get('/api/spectra/synthetic-scan/point/0', params={'despike_method': 'none'}).json()
    response = s.client.post('/api/identify', json={**s.body, 'spectrum_type': 'dark_subtracted',
                             'expected_source_digest': source['provenance']['source_digest']})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['input'] == source['provenance']
    assert result['selection']['excitation_nm'] == source['provenance']['excitation_nm']
    assert result['selection']['excitation_evidence'] == source['provenance']['excitation_basis']


@pytest.mark.parametrize('evidence_api', [{'templates': count} for count in (1, 10, 100)], indirect=True)
def test_synthetic_budgets(evidence_api):
    """Measure admitted work sizes; stage the complete synthetic catalog first."""
    import resource

    s = evidence_api
    count = len(s.body['template_ids'])
    started = time.monotonic()
    response = s.client.post('/api/identify', json=s.body)
    cold = time.monotonic() - started
    assert response.status_code == 200, response.text
    started = time.monotonic()
    cached = s.client.post('/api/identify', json=s.body)
    warm = time.monotonic() - started
    assert cached.json() == response.json()
    measurement = {'templates': count, 'member_evaluations': count * 2, 'native_points_per_trace': len(s.x),
                   'cold_seconds': round(cold, 4), 'warm_seconds': round(warm, 4),
                   'result_bytes': len(response.content), 'parent_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'worker_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    print('SYNTHETIC_BUDGETS=' + json.dumps(measurement, sort_keys=True))


def test_stored_product_denial_without_metadata_alias(evidence_api, monkeypatch):
    s = evidence_api
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(pds4_metadata={"intensity_unit": "synthetic units"}))
        session.commit()
    policy = s.app.state.feature_policy.model_dump()
    policy['identification']['source_denylist']['product_ids'] = ['ss__9999_0000000000_000rrs__synthetic']
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    worker = Mock(side_effect=AssertionError('source worker after denial'))
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.run_process', worker)
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 422
    worker.assert_not_called()



@pytest.mark.parametrize('evidence_api', [{'unpermitted_template': True}], indirect=True)
def test_pinned_subset_binding_ignores_unrelated_unpermitted_templates(evidence_api):
    s = evidence_api
    policy = s.app.state.feature_policy.model_dump()
    policy['references']['template_ids'] = s.binding['template_ids']
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    response = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert response.status_code == 200
    assert response.json()['binding'] == s.binding
    # An explicit subset never relaxes permissions on any selected member.
    policy['references'].update(mode='allowlist', allowed_trace_ids=[s.trace_ids[0]])
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id}).json()['binding'] is None


@pytest.mark.parametrize('ids', [['b' * 64, 'a' * 64], ['a' * 64, 'a' * 64], ['b' * 64]])
def test_explicit_subset_policy_requires_sorted_unique_matching_ids(ids):
    with pytest.raises(ValueError, match='exact pinned subset'):
        FeaturePolicy.model_validate({'references': {'template_ids': ids, 'subset_id': digest(['a' * 64])}})


@pytest.mark.parametrize('evidence_api', [{'templates': 100}], indirect=True)
def test_binding_bulk_authorization_has_bounded_query_count(evidence_api):
    s = evidence_api
    queries = []

    def count_query(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            queries.append(statement)

    event.listen(s.engine, 'before_cursor_execute', count_query)
    try:
        response = s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 1})
    finally:
        event.remove(s.engine, 'before_cursor_execute', count_query)
    assert response.status_code == 200
    assert response.json()['binding'] == s.binding
    assert len(response.json()['binding']['template_ids']) == 100
    assert len(queries) <= 4


def test_binding_rejects_large_aggregate_before_loading_members(evidence_api):
    from sherloc_pipeline.web.routes.references import identification_binding

    s = evidence_api
    ids = sorted(digest(['synthetic-template', index]) for index in range(10_000))
    rows = [SimpleNamespace(template_id=identity, member_trace_ids=s.trace_ids * 50) for identity in ids]
    db = Mock()
    db.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = rows
    policy = s.app.state.feature_policy
    policy = policy.model_copy(update={
        'references': policy.references.model_copy(update={'subset_id': digest(ids)}),
        'identification': policy.identification.model_copy(update={'max_templates': 10_000}),
    })
    with Session(s.engine) as session:
        release = session.get(ReferenceReleaseORM, s.imported.release_id)
        request = SimpleNamespace(state=SimpleNamespace(db=db))
        assert identification_binding(request, policy, release) is None
    assert db.query.call_count == 1
    db.get.assert_not_called()


@pytest.mark.parametrize('route_path,method', [
    ('/api/identify', 'POST'), ('/api/references', 'GET'),
    ('/api/references/{trace_revision_id}', 'GET'),
])
def test_unhandled_evidence_errors_are_private_no_store(evidence_api, monkeypatch, route_path, method):
    s = evidence_api

    def fail():
        raise RuntimeError('synthetic unhandled route exception')

    monkeypatch.setitem(s.app.dependency_overrides, require_authenticated_request, fail)
    url = route_path.replace('{trace_revision_id}', s.trace_ids[0])
    client = TestClient(s.app, raise_server_exceptions=False)
    try:
        response = client.request(method, url, params={'release_id': s.imported.release_id} if method == 'GET' else {},
                                  json=s.body if method == 'POST' else None)
    finally:
        client.close()
    assert response.status_code == 500
    assert response.json()['error'] == 'internal_error'
    assert response.headers['cache-control'] == 'private, no-store'


class SyntheticFailedNativeStore:
    def __init__(self, failure, marker):
        self.failure, self.marker = failure, marker

    def read_verified(self, *_args, **_kwargs):
        self.marker.write_text(str(os.getpid()))
        if self.failure == 'crash':
            os._exit(7)
        if self.failure == 'deadline':
            time.sleep(60)
        if self.failure == 'rejected':
            raise WorkRejected('synthetic native admission refusal')
        raise OSError('synthetic missing object')


@pytest.mark.parametrize('failure', ['crash', 'deadline', 'rejected', 'missing'])
def test_native_reference_worker_errors_are_classified_and_reaped(evidence_api, tmp_path, failure):
    s = evidence_api
    marker = tmp_path / 'synthetic-native-pid.txt'
    s.app.state.reference_store = SyntheticFailedNativeStore(failure, marker)
    policy = s.app.state.feature_policy
    s.app.state.feature_policy = policy.model_copy(update={
        'identification': policy.identification.model_copy(update={'timeout_seconds': 5})})
    response = s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id})
    assert response.status_code == (429 if failure == 'rejected' else 503)
    assert response.headers['cache-control'] == 'private, no-store'
    if failure != 'rejected':
        assert response.json()['detail']['code'] == 'execution_failed'
    else:
        assert response.json()['detail'] == 'synthetic native admission refusal'
    assert marker.exists(), 'Native reader must have started'
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    # Failures release admission and never fill the success cache.
    s.app.state.reference_store = LocalReferenceStore(s.objects)
    recovered = s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id})
    assert recovered.status_code == 200
    assert recovered.json()['x'] == s.x.tolist()


@pytest.mark.parametrize('denied_identity', ['canonical', 'alias', 'internal'])
def test_metadata_alias_cannot_hide_any_denied_persisted_identity(evidence_api, monkeypatch, denied_identity):
    s = evidence_api
    alias = 'ss__9999_0000000000_001rrs__synthetic_alias'
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(pds4_metadata={'product_id': alias, 'intensity_unit': 'synthetic units'}))
        session.commit()
    identities = {'canonical': 'ss__9999_0000000000_000rrs__synthetic',
                  'alias': alias, 'internal': 'synthetic-scan'}
    policy = s.app.state.feature_policy.model_dump()
    policy['identification']['source_denylist']['product_ids'] = [identities[denied_identity]]
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    worker = Mock(side_effect=AssertionError('source worker after identity denial'))
    store = Mock(side_effect=AssertionError('reference store after identity denial'))
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.run_process', worker)
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.reference_store', store)
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 422
    assert response.headers['cache-control'] == 'private, no-store'
    worker.assert_not_called()
    store.assert_not_called()


@pytest.mark.parametrize('lidvid_suffix, persisted', [('', 'ss__9999_0000000000_002rrs__synthetic'),
                                                      ('::1.0', 'ss__9999_0000000000_002rrs__synthetic'),
                                                      ('::1.0', 'SS__9999_0000000000_002RRS__SYNTHETIC')])
def test_denylist_matches_product_id_of_persisted_pds4_logical_identifier(evidence_api, monkeypatch, lidvid_suffix,
                                                                          persisted):
    s = evidence_api
    product = 'ss__9999_0000000000_002rrs__synthetic'
    lid = f'urn:nasa:pds:synthetic_bundle:data_processed:{persisted}'
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(
            scan_id=lid, pds4_metadata={'lidvid': lid + lidvid_suffix, 'intensity_unit': 'synthetic units'}))
        session.commit()
    policy = s.app.state.feature_policy.model_dump()
    policy['identification']['source_denylist']['product_ids'] = [product]
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    worker = Mock(side_effect=AssertionError('source worker after identity denial'))
    store = Mock(side_effect=AssertionError('reference store after identity denial'))
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.run_process', worker)
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.reference_store', store)
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 422
    assert response.headers['cache-control'] == 'private, no-store'
    worker.assert_not_called()
    store.assert_not_called()


@pytest.mark.parametrize('lid', ['urn:nasa:pds:synthetic_bundle:data_processed:',
                                 'urn:nasa:pds:synthetic_bundle:data_processed:synthetic product'])
def test_unparsable_logical_identifier_is_refused_before_reads(evidence_api, monkeypatch, lid):
    s = evidence_api
    with Session(s.engine) as session:
        session.execute(update(ScanORM).values(scan_id=lid, pds4_metadata={'lidvid': lid,
                                                                             'intensity_unit': 'synthetic units'}))
        session.commit()
    worker = Mock(side_effect=AssertionError('source worker without a product identity'))
    store = Mock(side_effect=AssertionError('reference store without a product identity'))
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.run_process', worker)
    monkeypatch.setattr('sherloc_pipeline.web.routes.identification.reference_store', store)
    response = s.client.post('/api/identify', json=s.body)
    assert response.status_code == 422
    assert response.json()['detail'] == 'Source product identity unavailable'
    worker.assert_not_called()
    store.assert_not_called()


@pytest.mark.parametrize('matches', [0, 1, 2])
def test_binding_requires_one_bundled_method_match(evidence_api, matches):
    s = evidence_api
    policy = s.app.state.feature_policy.model_dump()
    # A stale method before the usable binding must never hide the usable one.
    stale = {**s.binding, 'method_release_id': 'f' * 64}
    stale.pop('template_ids')
    usable = policy['identification']['allowed_method_releases'][0]
    policy['identification']['allowed_method_releases'] = [stale] + [usable] * matches
    s.app.state.feature_policy = FeaturePolicy.model_validate(policy)
    response = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert response.status_code == 200
    if matches == 1:
        assert response.json()['binding'] == s.binding
        body = {**s.body, **response.json()['binding']}
        identified = s.client.post('/api/identify', json=body)
        assert identified.status_code == 200, identified.text
        assert identified.json()['status'] == 'candidate'
    else:
        assert response.json()['binding'] is None
