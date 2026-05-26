// App-level routing tests for the PDS Browser feature gate
// (issue #21, R1 F1). Codex flagged that the previous "soft-404 via
// empty state" fall-through didn't satisfy the AC's "returns 404 OR
// redirects to root when flag is disabled". This file pins down the
// Round 2 fix: `navigate('#/')` fires the moment the route sees
// `pds` while features.pds_browser is false.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import App from './App.svelte';
import { features, currentHash } from './lib/stores';

beforeEach(() => {
  features.set({ pds_browser: true });
  currentHash.set('#/');
  window.location.hash = '#/';
  // Block App's onMount network calls (getAccessMode) from polluting
  // the assertions — leave them rejecting so the public-mode default
  // redirect path doesn't fire on the synchronous render.
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('blocked in test')));
});

describe('App — disabled-feature `#/pds` redirect (issue #21 R2 F1)', () => {
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
