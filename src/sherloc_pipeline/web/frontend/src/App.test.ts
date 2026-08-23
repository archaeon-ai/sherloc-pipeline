// App-level routing tests for the PDS Browser feature gate
// (regression guard for issue #21). The previous "soft-404 via
// empty state" fall-through didn't satisfy the requirement that the
// route "returns 404 OR redirects to root when flag is disabled". This
// file pins down the fix: `navigate('#/')` fires the moment the route
// sees `pds` while features.pds_browser is false.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import { tick } from 'svelte';
import App from './App.svelte';
import { accessMode, accessModeResolved, features, currentHash } from './lib/stores';

beforeEach(() => {
  accessMode.set('internal');
  features.set({ pds_browser: true });
  accessModeResolved.set(false);
  currentHash.set('#/');
  window.location.hash = '#/';
  sessionStorage.clear();
  // Block App's onMount network calls (getAccessMode) from polluting
  // the assertions — leave them rejecting so the public-mode default
  // redirect path doesn't fire on the synchronous render.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('blocked in test')));
});

describe('App routing', () => {
  it('preserves the public-mode default redirect when a saved scan filter exists', async () => {
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Amherst');
    vi.stubGlobal('fetch', vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const body = String(input).includes('/config/access-mode')
        ? { schema_version: '1.0.0', access_mode: 'public' }
        : { status: 'ok' };
      return Promise.resolve(new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    }));

    render(App);

    await waitFor(() => expect(window.location.hash).toBe('#/pds'));
  });

  it('redirects #/pds → #/ synchronously when pds_browser=false', async () => {
    features.set({ pds_browser: false });
    currentHash.set('#/pds');
    window.location.hash = '#/pds';

    render(App);
    await tick();

    expect(window.location.hash).toBe('#/');
  });

  it('does NOT redirect when pds_browser=true (PDS stays renderable)', async () => {
    features.set({ pds_browser: true });
    currentHash.set('#/pds');
    window.location.hash = '#/pds';

    render(App);
    await tick();

    expect(window.location.hash).toBe('#/pds');
  });

  it('redirects after a feature-store flip mid-session', async () => {
    features.set({ pds_browser: true });
    currentHash.set('#/pds');
    window.location.hash = '#/pds';

    render(App);
    await tick();
    expect(window.location.hash).toBe('#/pds');

    // Operator hot-flips the env behind a reload-less reconnect: the
    // reactive block in App.svelte must catch the next features-store
    // update and bounce the route.
    features.set({ pds_browser: false });
    await tick();

    expect(window.location.hash).toBe('#/');
  });
});
