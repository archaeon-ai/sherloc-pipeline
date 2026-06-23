// Helper-boundary tests for `lib/api.ts`.
//
// Regression guard for issue #17: the panel-level tests in
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

import { AuthRequiredError, getBadpixChannels, getPointSpectrum } from './api';

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('getPointSpectrum — ensureAuthenticated gate (issue #17)', () => {
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

describe('getBadpixChannels — static per-region client cache (issue #9)', () => {
  function badpixResponse(region: string) {
    return new Response(
      JSON.stringify({ schema_version: '1.0.0', region, n_channels: 523, badpix: [] }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  }

  it('fetches once per region and serves later calls from cache', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockImplementation(() => Promise.resolve(badpixResponse('R1')));

    const a = await getBadpixChannels('R1');
    const b = await getBadpixChannels('R1');
    expect(a).toBe(b); // same cached promise result
    expect(a.region).toBe('R1');
    // A different region triggers a fresh request; R1 stays cached.
    fetchSpy.mockImplementation(() => Promise.resolve(badpixResponse('R2')));
    await getBadpixChannels('R2');
    // R1 fetched once + R2 fetched once = two network calls total.
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(fetchSpy.mock.calls[0][0]).toContain('/spectra/badpix?region=R1');
    expect(fetchSpy.mock.calls[1][0]).toContain('/spectra/badpix?region=R2');
  });
});
