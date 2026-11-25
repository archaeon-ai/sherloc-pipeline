"""Native reference routes, permissions and resource limits."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tests.unit.test_reference_release import PROCESSING_STATE_KEYS, importer_before_processing_state, snapshot  # noqa: F401
from tests.unit._spawn_helpers import _crash, _hang, ready_path
from sherloc_pipeline.database.models import Base
from sherloc_pipeline.database.reference_models import ReferenceReleaseORM, ReferenceTemplateORM, ReferenceTraceORM
from sherloc_pipeline.runtime import worker as runtime_worker
from sherloc_pipeline.runtime.worker import ExecutionFailed, WorkRejected, run_process
from sherloc_pipeline.services.reference_import import digest, import_release
from sherloc_pipeline.services.reference_store import LocalReferenceStore
from sherloc_pipeline.web.app import create_app
from sherloc_pipeline.web.auth import require_authenticated_request


@pytest.fixture
def reference_api(tmp_path, snapshot, monkeypatch, request):
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
    options = getattr(request, 'param', {})
    for index in (1, 2):
        extra = copy.deepcopy(trace)
        extra.update(trace_index=index, source_label=f'Synthetic trace {index}')
        if index == 2:
            extra.update(rights_status='unknown', license_identifier=None, license_source=None)
        if index == 1 and 'processing_state' in options:
            extra.update(options['processing_state'])
        file['traces'].append(extra)
    manifest['templates'][0]['members'].append({**manifest['templates'][0]['members'][0], 'trace_index': 1})
    (root / 'snapshot.json').write_text(json.dumps(manifest))
    serving.write_text(json.dumps({'approved_trace_keys': [f'synthetic-0001:synthetic-file-1:{i}' for i in range(3)]}))
    database = tmp_path / 'synthetic.db'
    engine = create_engine(f'sqlite:///{database}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    objects = tmp_path / 'objects'
    if options.get('before_processing_state'):
        with importer_before_processing_state():
            imported = import_release(root, database, objects, serving)
    else:
        imported = import_release(root, database, objects, serving)
    with Session(engine) as session:
        release = session.get(ReferenceReleaseORM, imported.release_id)
        template = session.query(ReferenceTemplateORM).filter_by(name='synthetic-template').one()
        trace_ids = list(template.member_trace_ids)
        unknown = session.query(ReferenceTraceORM).filter_by(rights_known=False).one().trace_revision_id
        values = {'references': {'mode': 'all_approved', 'release_id': release.release_id,
                  'serving_policy_version': release.serving_policy_sha256, 'subset_id': digest([template.template_id]),
                  **options.get('reference_limits', {})}}
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
    yield SimpleNamespace(**locals())
    client.close()
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)
    engine.dispose()


def test_reference_list_omits_binding(reference_api):
    response = reference_api.client.get('/api/references', params={'release_id': reference_api.imported.release_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload['items']
    assert 'binding' not in payload


@pytest.mark.parametrize('reference_api', [{'reference_limits': {'max_concurrency': 2}}], indirect=True)
def test_native_reference_provenance(reference_api):
    s = reference_api
    listed = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert listed.status_code == 200
    assert len(listed.json()['items']) == 2
    assert listed.headers['cache-control'] == 'private, no-store'
    url = f'/api/references/{s.trace_ids[0]}'
    params = {'release_id': s.imported.release_id}
    slots = s.app.state.reference_slots
    with slots:
        assert slots.acquire(blocking=False)
        try:
            assert s.client.get(url, params=params).status_code == 429
        finally:
            slots.release()
    response = s.client.get(url, params=params)
    assert response.status_code == 200
    assert response.json()['x'] == s.x.tolist()
    assert response.json()['provenance']['citation_text'] == 'Synthetic citation'
    assert response.json()['provenance']['source_processing_state'] == 'unknown'
    assert response.json()['provenance']['source_processing_evidence'] is None
    assert 'locator' not in response.text and 'x_key' not in response.text
    s.app.state.reference_store = Mock(read_verified=Mock(side_effect=AssertionError('unexpected cache miss')))
    assert s.client.get(url, params=params).json() == response.json()


@pytest.mark.parametrize('reference_api', [{'processing_state': {
    'source_processing_state': 'baseline_corrected', 'source_processing_evidence': 'synthetic baseline record'}}],
    indirect=True)
def test_route_serves_recorded_processing_state(reference_api):
    s = reference_api
    provenance = [s.client.get(f'/api/references/{trace_id}', params={'release_id': s.imported.release_id}).json()['provenance']
                  for trace_id in s.trace_ids]
    assert [(p['source_processing_state'], p['source_processing_evidence']) for p in provenance] == [
        ('unknown', None), ('baseline_corrected', 'synthetic baseline record')]


@pytest.mark.parametrize('reference_api', [{'before_processing_state': True}], indirect=True)
def test_catalog_imported_before_processing_state_serves_native_traces(reference_api):
    s = reference_api
    with Session(s.engine) as session:
        assert all(not set(PROCESSING_STATE_KEYS) & set(row.metadata_json) for row in session.query(ReferenceTraceORM))
    for trace_id in s.trace_ids:
        response = s.client.get(f'/api/references/{trace_id}', params={'release_id': s.imported.release_id})
        assert response.status_code == 200
        provenance = response.json()['provenance']
        assert provenance['source_processing_state'] == 'unknown' and provenance['source_processing_evidence'] is None
    listed = s.client.get('/api/references', params={'release_id': s.imported.release_id})
    assert listed.status_code == 200 and len(listed.json()['items']) == 2


@pytest.mark.parametrize('tier', ['team', 'public'])
@pytest.mark.parametrize('refs', [False, True])
def test_reference_tier_guards(reference_api, tier, refs):
    s = reference_api
    s.app.state.access_mode = 'internal' if tier == 'team' else 'public'
    original = s.app.state.feature_policy
    s.app.state.feature_policy = original.model_copy(update={
        'references': original.references.model_copy(update={'mode': 'all_approved' if refs else 'off'})})
    permitted = tier == 'team' and refs
    for path in ('/api/references', f'/api/references/{s.trace_ids[0]}'):
        response = s.client.get(path, params={'release_id': s.imported.release_id})
        assert response.status_code == (200 if permitted else 404)
        assert response.headers['cache-control'] == 'private, no-store'


def test_authentication_and_unknown_rights(reference_api):
    s = reference_api
    assert s.client.get(f'/api/references/{s.unknown}', params={'release_id': s.imported.release_id}).status_code == 404
    def denied():
        raise HTTPException(401, 'Synthetic authentication refusal')
    s.app.dependency_overrides[require_authenticated_request] = denied
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id}).status_code == 401


def test_allowlist_denies_trace_before_storage(reference_api):
    s = reference_api
    original = s.app.state.feature_policy
    s.app.state.feature_policy = original.model_copy(update={'references': original.references.model_copy(
        update={'mode': 'allowlist', 'allowed_trace_ids': [s.trace_ids[0]]})})
    s.app.state.reference_store = Mock(read_verified=Mock(side_effect=AssertionError('array read after denial')))
    assert s.client.get(f'/api/references/{s.trace_ids[1]}', params={'release_id': s.imported.release_id}).status_code == 404


@pytest.mark.parametrize('key', ['url', 'source_url', 'object_key', 'locator', 'arrays'])
def test_no_reference_parameter_bypass(reference_api, key):
    s = reference_api
    assert s.client.get(f'/api/references/{s.trace_ids[0]}', params={'release_id': s.imported.release_id, key: 'synthetic-invalid'}).status_code == 422


def test_cursor_is_bounded_and_policy_bound(reference_api):
    s = reference_api
    first = s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 1}).json()
    second = s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 1, 'cursor': first['next_cursor']}).json()
    assert len(first['items']) == len(second['items']) == 1
    assert first['items'][0]['trace_revision_id'] != second['items'][0]['trace_revision_id']
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'limit': 101}).status_code == 422
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'cursor': 'invalid'}).status_code == 422
    assert s.client.get('/api/references', params={'release_id': s.imported.release_id, 'modality': 'fluorescence', 'cursor': first['next_cursor']}).status_code == 422


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
    # Record the exact deadline run_process computes and hands to _receive.
    original_receive = runtime_worker._receive
    deadlines = []

    def recording_receive(reader, deadline):
        deadlines.append(deadline)
        return original_receive(reader, deadline)

    monkeypatch.setattr(runtime_worker, '_receive', recording_receive)
    started = time.monotonic()
    with pytest.raises(ExecutionFailed, match='time limit'):
        run_process(_hang, (str(path),), 3.)
    returned = time.monotonic()
    monkeypatch.undo()
    assert len(spawned) == 1 and spawned[0], 'spawn wrapper did not capture the worker pid'
    assert len(deadlines) == 1, 'receive wrapper did not capture the deadline'
    deadline = deadlines[0]
    assert returned >= deadline
    assert returned - started >= 3. - .05
    with pytest.raises(ProcessLookupError):
        os.kill(spawned[0], 0)
    # Worker start-up (spawn and imports) counts against the deadline but is not what this test measures.
    assert path.exists(), (f'worker did not start within the 3 s deadline ({returned - started:.2f} s elapsed; '
                           'host load); SIGTERM-to-kill escalation not exercised')
    assert int(path.read_text()) == spawned[0]
    ready = float(Path(ready_path(path)).read_text())
    # A worker that becomes ready after the deadline, before a delayed parent kills it, is still a start-up overrun.
    assert ready < deadline, (f'worker did not start within the 3 s deadline (ready {ready - deadline:.2f} s after it; '
                              'host load); SIGTERM-to-kill escalation not exercised')
    assert returned - deadline < 2.
    with pytest.raises(ExecutionFailed, match='worker failed'):
        run_process(_crash, (), 3.)


def test_cache_is_bounded_detached_and_scoped():
    from sherloc_pipeline.web.reference_storage import ResponseCache

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


@pytest.mark.parametrize('route_path', ['/api/references', '/api/references/{trace_revision_id}'])
def test_unhandled_reference_errors_are_private_no_store(reference_api, monkeypatch, route_path):
    s = reference_api

    def fail():
        raise RuntimeError('synthetic unhandled route exception')

    monkeypatch.setitem(s.app.dependency_overrides, require_authenticated_request, fail)
    url = route_path.replace('{trace_revision_id}', s.trace_ids[0])
    client = TestClient(s.app, raise_server_exceptions=False)
    try:
        response = client.get(url, params={'release_id': s.imported.release_id})
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
@pytest.mark.parametrize('reference_api', [{'reference_limits': {'timeout_seconds': 5}}], indirect=True)
def test_native_reference_worker_errors_are_classified_and_reaped(reference_api, tmp_path, failure):
    s = reference_api
    marker = tmp_path / 'synthetic-native-pid.txt'
    s.app.state.reference_store = SyntheticFailedNativeStore(failure, marker)
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
