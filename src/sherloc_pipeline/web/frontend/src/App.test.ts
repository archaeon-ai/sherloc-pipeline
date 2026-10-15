// App-level routing tests for the PDS Browser feature gate
// (regression guard for issue #21). The previous "soft-404 via
// empty state" fall-through didn't satisfy the requirement that the
// route "returns 404 OR redirects to root when flag is disabled". This
// file pins down the fix: `navigate('#/')` fires the moment the route
// sees `pds` while features.pds_browser is false.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/svelte';
import { tick } from 'svelte';
import App from './App.svelte';
import { accessMode, accessModeResolved, features, currentHash } from './lib/stores';
import * as api from './lib/api';

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

afterEach(() => { vi.restoreAllMocks(); });

describe('App routing', () => {
  it('preserves a resolver link during public-mode startup and reopens changed links', async () => {
    vi.spyOn(api, 'getAccessMode').mockResolvedValue({ schema_version: '1.0.0', access_mode: 'public' });
    const lookup = vi.spyOn(api, 'getScans').mockResolvedValue({
      schema_version: '1.0.0', scans: [], total: 0, offset: 0, limit: 200,
    });
    const hash = '#/open?scan_name=Testrock_detail&sol=42424&point=0';
    window.history.replaceState(null, '', hash);
    currentHash.set(hash);
    render(App);
    expect(await screen.findByRole('alert')).toHaveTextContent('No scan found');
    expect(window.location.hash).toBe(hash);

    window.location.hash = '#/open?scan_name=Testrock_detail&sol=42425&points=0,2';
    await waitFor(() => expect(lookup).toHaveBeenLastCalledWith(expect.objectContaining({ sol: 42425 })));
  });

  it('remounts the Workbench when only the subset changes', async () => {
    const load = vi.spyOn(api, 'getScan').mockRejectedValue(new Error('Synthetic scan unavailable'));
    vi.spyOn(api, 'getBadpixChannels').mockRejectedValue(new Error('No synthetic channels'));
    const hash = '#/scan/test-id/workbench?points=0,2';
    window.history.replaceState(null, '', hash);
    currentHash.set(hash);
    render(App);
    await waitFor(() => expect(load).toHaveBeenCalledTimes(1));
    window.location.hash = '#/scan/test-id/workbench?points=0,7';
    await waitFor(() => expect(load).toHaveBeenCalledTimes(2));
  });

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
