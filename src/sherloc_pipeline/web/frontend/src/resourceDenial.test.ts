import { afterEach, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('./lib/auth', () => ({
  bootstrapAuthReady: Promise.resolve(null),
  bootstrapAuth: async () => ({ isAuthenticated: async () => true }),
  getSession: () => null,
  redirectingToLogin: () => false,
  authCallbackError: () => null,
}));

let app: { $destroy(): void } | null;
afterEach(() => {
  window.dispatchEvent(new Event('pagehide'));
  app?.$destroy();
  app = null;
  vi.unstubAllGlobals();
  window.location.hash = '';
});

it.each(['resource', 'tier', 'authentication'])('handles a bookmarked scan %s denial after mounting', async (kind) => {
  vi.resetModules();
  document.body.innerHTML = '<div id="app"></div>';
  window.location.hash = '#/scan/synthetic-scan/workbench';
  let refuse!: () => void;
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.startsWith('/api/spectra/badpix')) {
      return new Response(JSON.stringify({ schema_version: '1.0.0', region: 'R1', n_channels: 0, badpix: [] }));
    }
    if (url === '/api/scans/synthetic-scan') {
      await new Promise<void>((resolve) => { refuse = resolve; });
      return new Response(JSON.stringify({ detail: 'Synthetic resource denied' }), {
        status: kind === 'authentication' ? 401 : 403,
        headers: kind === 'tier' ? { 'X-Sherloc-Auth-Failure': 'authorization' } : {},
      });
    }
    return new Response(JSON.stringify(url === '/api/config/bootstrap'
      ? { auth: { auth_mode: 'cf-access' } }
      : { access_mode: 'public', tier: 'public', capabilities: {
        references: false, identification: false, experimental_candidates: false,
      } }));
  }));
  app = await (await import('./main')).default;
  const stores = await import('./lib/stores');
  await vi.waitFor(() => expect(refuse).toBeDefined());
  expect(document.querySelector('.nav-bar')).not.toBeNull();
  refuse();
  if (kind === 'resource') {
    await vi.waitFor(() => expect(document.querySelector('.error-message')?.textContent)
      .toContain('Synthetic resource denied'));
    expect(document.querySelector('.nav-bar')).not.toBeNull();
    expect(get(stores.scienceAuthorized)).toBe(true);
  } else {
    await vi.waitFor(() => expect(get(stores.scienceAuthorized)).toBe(false));
    expect(document.getElementById('app')?.innerHTML).toBe('');
    app = null;
  }
}, 15000);
