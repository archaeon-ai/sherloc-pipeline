// Helper-boundary tests for `lib/api.ts`.
//
// Issue #17 R1 F2 (Codex): the panel-level tests in
// `components/map/MapSpectrumPanel.test.ts` mock `getPointSpectrum` and
// therefore do not exercise the new `await ensureAuthenticated()` line
// added to the helper itself. This file covers the helper boundary: with
// the auth singleton in an unauthenticated state, `getPointSpectrum`
// must throw `AuthRequiredError` BEFORE any network request goes out.
//
// The test mocks the `./auth` module so `bootstrapAuthReady` resolves to
// `null` synchronously and `getSession()` returns `null` — the same
// shape `ensureAuthenticated()` checks before throwing.

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('./auth', () => ({
  bootstrapAuthReady: Promise.resolve(null),
  getSession: () => null,
}));

import { AuthRequiredError, getPointSpectrum } from './api';

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('getPointSpectrum — ensureAuthenticated gate (issue #17 R1 F2)', () => {
  it('throws AuthRequiredError before any network request when no session', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } }),
    );

    await expect(
      getPointSpectrum('ae5578c9-5a91-41c9-8431-190117be23b4', 91, { region: 'R1' }),
    ).rejects.toBeInstanceOf(AuthRequiredError);

    // The auth gate must fire BEFORE the network call — guarantees we
    // do not leak unauthenticated point-spectrum requests under Auth0
    // mode (the durable defect issue #17 captured).
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
