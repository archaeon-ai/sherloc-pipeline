import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import { AuthRequiredError, fetchAciImage, getConfig, getScan, getMapData, getMapJobResults, getMapJobStatus, getMapLayers, getJobStatus, getPointSpectrum, postFit, postPdsDownload, postSpectraSubset, startMapFit } from './api';
import { clearSessionState, lastFitResult, scienceAuthorized, sessionRequests } from './stores';

const { getToken, isAuthenticated } = vi.hoisted(() => ({ getToken: vi.fn(), isAuthenticated: vi.fn() }));

vi.mock('./auth', () => ({
  bootstrapAuthReady: Promise.resolve(null),
  getSession: () => ({ mode: 'auth0', isAuthenticated, getToken }),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => {
  getToken.mockReset().mockResolvedValue('synthetic-token');
  isAuthenticated.mockReset().mockResolvedValue(true);
  clearSessionState();
  scienceAuthorized.set(true);
});
afterEach(() => { vi.unstubAllGlobals(); });

it('discards a pending fit reply so it cannot refill stores after logout', async () => {
  const response = deferred<Response>();
  vi.stubGlobal('fetch', vi.fn().mockReturnValue(response.promise));
  const fit = postFit({ synthetic: true } as never).then((value) => lastFitResult.set(value));
  const rejected = expect(fit).rejects.toBeInstanceOf(AuthRequiredError);
  clearSessionState();
  response.resolve(new Response('{"synthetic":true}', { status: 200 }));
  await rejected;
  expect(get(lastFitResult)).toBeNull();
});

it('also discards a reply whose JSON body completes after logout', async () => {
  const body = deferred<unknown>();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: () => body.promise }));
  const config = getConfig();
  const rejected = expect(config).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  clearSessionState();
  body.resolve({ synthetic: true });
  await rejected;
});

it.each([401, 403])('clears session state on denied JSON response %s', async (status) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', {
    status, headers: status === 403 ? { 'X-Sherloc-Auth-Failure': 'authorization' } : {},
  })));
  await expect(getConfig()).rejects.toThrow();
  expect(get(scienceAuthorized)).toBe(false);
});

it.each([401, 403])('clears session state on denied image response %s', async (status) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', {
    status, headers: status === 403 ? { 'X-Sherloc-Auth-Failure': 'authorization' } : {},
  })));
  await expect(fetchAciImage('synthetic-scan')).rejects.toThrow();
  expect(get(scienceAuthorized)).toBe(false);
});

it.each([
  ['scan', () => getScan('synthetic-scan')],
  ['spectrum', () => getPointSpectrum('synthetic-scan', 0)],
  ['PDS download', () => postPdsDownload({ sol: 9999 })],
  ['image', () => fetchAciImage('synthetic-scan')],
])('preserves the session for a resource-level %s denial', async (_kind, request) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"detail":"Synthetic resource denied"}', { status: 403 })));
  await expect(request())
    .rejects.toMatchObject({ status: 403, name: 'ApiError' });
  expect(get(scienceAuthorized)).toBe(true);
});

it('discards an image whose bytes arrive after logout', async () => {
  const body = deferred<Blob>();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, blob: () => body.promise }));
  const image = fetchAciImage('synthetic-scan');
  const rejected = expect(image).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  clearSessionState();
  body.resolve(new Blob(['synthetic image']));
  await rejected;
});

it('discards a decoded image if logout happens during decoding', async () => {
  let decoded!: { onload: () => void };
  const revoke = vi.fn();
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL() { return 'blob:synthetic'; }
    static revokeObjectURL = revoke;
  });
  vi.stubGlobal('Image', class {
    constructor() { decoded = this as never; }
    onload = () => {};
    onerror = () => {};
    src = '';
  });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('synthetic image')));
  const image = fetchAciImage('synthetic-scan');
  const rejected = expect(image).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(decoded).toBeDefined());
  clearSessionState();
  decoded.onload();
  await rejected;
  expect(revoke).toHaveBeenCalledWith('blob:synthetic');
});


it.each(['json', 'image'])('never sends a %s request when the token resolves after sign-out', async (kind) => {
  const token = deferred<string>();
  getToken.mockReturnValue(token.promise);
  vi.stubGlobal('fetch', vi.fn());
  const pending = kind === 'json' ? getConfig() : fetchAciImage('synthetic-scan');
  const rejected = expect(pending).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(getToken).toHaveBeenCalled());
  clearSessionState();
  token.resolve('synthetic-old-token');
  await rejected;
  expect(fetch).not.toHaveBeenCalled();
  expect(sessionRequests.size).toBe(0);
});

it.each(['json', 'image'])('aborts an outstanding %s fetch on sign-out', async (kind) => {
  let signal!: AbortSignal;
  vi.stubGlobal('fetch', vi.fn((_url, init) => {
    signal = init.signal;
    return new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('synthetic abort', 'AbortError')));
    });
  }));
  const pending = kind === 'json' ? getConfig() : fetchAciImage('synthetic-scan');
  const rejected = expect(pending).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(fetch).toHaveBeenCalled());
  expect(signal.aborted).toBe(false);
  clearSessionState();
  expect(signal.aborted).toBe(true);
  await rejected;
  expect(sessionRequests.size).toBe(0);
});

it('removes completed requests from the session abort set', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"synthetic":true}')));
  await getConfig();
  expect(sessionRequests.size).toBe(0);
});


it.each([
  ['map layers', () => getMapLayers('synthetic-scan')],
  ['map data', () => getMapData('synthetic-scan', 'synthetic-domain', 'synthetic-value')],
  ['point spectrum', () => getPointSpectrum('synthetic-scan', 0)],
  ['map fit', () => startMapFit('synthetic-scan', ['synthetic-domain'])],
  ['map status', () => getMapJobStatus('synthetic-job')],
  ['map results', () => getMapJobResults('synthetic-job')],
  ['job status', () => getJobStatus('synthetic-job')],
  ['subset spectrum', () => postSpectraSubset('synthetic-scan', {})],
  ['image', () => fetchAciImage('synthetic-scan')],
])('never sends %s when authentication resolves after sign-out', async (_label, request) => {
  const authenticated = deferred<boolean>();
  isAuthenticated.mockReturnValue(authenticated.promise);
  vi.stubGlobal('fetch', vi.fn());
  const rejected = expect(request()).rejects.toBeInstanceOf(AuthRequiredError);
  await vi.waitFor(() => expect(isAuthenticated).toHaveBeenCalled());
  clearSessionState();
  authenticated.resolve(true);
  await rejected;
  expect(fetch).not.toHaveBeenCalled();
  expect(getToken).not.toHaveBeenCalled();
});
