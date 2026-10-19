import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import { AuthRequiredError, fetchAciImage, getConfig, postFit } from './api';
import { clearSessionState, lastFitResult, scienceAuthorized } from './stores';

vi.mock('./auth', () => ({
  bootstrapAuthReady: Promise.resolve(null),
  getSession: () => ({ mode: 'cf-access', isAuthenticated: async () => true }),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

beforeEach(() => { clearSessionState(); scienceAuthorized.set(true); });
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
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status })));
  await expect(getConfig()).rejects.toThrow();
  expect(get(scienceAuthorized)).toBe(false);
});

it.each([401, 403])('clears session state on denied image response %s', async (status) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status })));
  await expect(fetchAciImage('synthetic-scan')).rejects.toThrow();
  expect(get(scienceAuthorized)).toBe(false);
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
