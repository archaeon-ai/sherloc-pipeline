// Frontend component tests for the PDS Browser nav-button gating
// (issue #21, AC trace in `.ralph/spec.md` §1.1). Codex R1 F3 flagged
// the absence of nav-level behavioral coverage — the existing pytest
// suite proves `/api/config` returns the right `features.pds_browser`
// boolean but not that the SPA actually hides the button on it.

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import Navigation from './Navigation.svelte';
import { features, healthStatus, currentHash } from '../lib/stores';

beforeEach(() => {
  // Each test starts on the default route with no health status.
  // Health probe + auth wiring in onMount fire async network/promise
  // work; we let them no-op (fetch is unmocked → rejects silently
  // inside Navigation's try/catch) and assert against the synchronous
  // initial render.
  currentHash.set('#/');
  healthStatus.set(null);
});

describe('Navigation — PDS Browser tab gating (issue #21)', () => {
  it('renders the PDS Browser button when pds_browser=true (default)', () => {
    features.set({ pds_browser: true });
    render(Navigation);
    expect(screen.getByRole('button', { name: 'PDS Browser' })).toBeInTheDocument();
  });

  it('hides the PDS Browser button when pds_browser=false', () => {
    features.set({ pds_browser: false });
    render(Navigation);
    expect(screen.queryByRole('button', { name: 'PDS Browser' })).toBeNull();
  });

  it('shows the PDS Browser button after a feature-store flip back to true', async () => {
    // Simulates a defensive backend that returns features later than
    // the SPA's first render (rare but possible if /api/config retries).
    features.set({ pds_browser: false });
    render(Navigation);
    expect(screen.queryByRole('button', { name: 'PDS Browser' })).toBeNull();

    features.set({ pds_browser: true });
    await tick();
    expect(screen.getByRole('button', { name: 'PDS Browser' })).toBeInTheDocument();
  });

  it('always renders the Scans button regardless of feature state', () => {
    features.set({ pds_browser: false });
    render(Navigation);
    expect(screen.getByRole('button', { name: 'Scans' })).toBeInTheDocument();
  });
});
