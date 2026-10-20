// Authentication and tier authorization must finish before science UI mounts.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

const state = vi.hoisted(() => ({
  error: null as null | { kind: string; returnTo: string },
  authenticated: true, redirecting: false, bootstrapFails: false, scienceMounts: 0,
}));

vi.mock('./lib/auth', () => ({
  bootstrapAuth: vi.fn(async () => {
    if (state.bootstrapFails) throw new Error('synthetic auth failure');
    return { isAuthenticated: async () => state.authenticated };
  }),
  redirectingToLogin: () => state.redirecting,
  authCallbackError: () => state.error,
  getSession: () => null,
}));
vi.mock('./lib/api', () => ({ getConfig: vi.fn() }));
vi.mock('./App.svelte', () => ({
  default: class {
    node = document.createElement('div');
    constructor({ target }: { target: HTMLElement }) {
      state.scienceMounts += 1;
      this.node.dataset.testid = 'app-stub';
      target.appendChild(this.node);
    }
    $destroy() { this.node.remove(); }
  },
}));

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  document.body.innerHTML = '<div id="app"></div>';
  state.error = null;
  state.authenticated = true;
  state.redirecting = false;
  state.bootstrapFails = false;
  state.scienceMounts = 0;
  sessionStorage.clear();
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ auth: { auth_mode: 'cf-access' } }) }));
  const { getConfig } = await import('./lib/api');
  vi.mocked(getConfig).mockResolvedValue({
    access_mode: 'internal', tier: 'team',
    capabilities: { references: false, identification: false, experimental_candidates: false },
  } as never);
});

afterEach(() => { vi.unstubAllGlobals(); });

async function mount() { return (await import('./main')).default; }
function expectClosed(heading: string | null = 'Sign-in did not complete') {
  const root = document.getElementById('app')!;
  expect(root.querySelector('[data-testid="app-stub"]')).toBeNull();
  if (heading === null) {
    expect(root.innerHTML).toBe('');
  } else {
    expect(root.children).toHaveLength(1);
    expect(root.querySelector('main h1')?.textContent).toBe(heading);
    expect(root.textContent).toContain('Sign in');
  }
  expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual(['/api/config/bootstrap']);
}

describe('main.ts bootstrap', () => {
  it('preserves the saved Scan Browser filters through a successful team reload', async () => {
    const saved = '#/?target=Synthetic&search=synthetic';
    sessionStorage.setItem('sherloc.scanBrowserHash', saved);
    await mount();
    expect(state.scienceMounts).toBe(1);
    expect(sessionStorage.getItem('sherloc.scanBrowserHash')).toBe(saved);
  });

  it('mounts the access page when Auth0 refused the sign-in', async () => {
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Synthetic');
    state.error = { kind: 'team_required', returnTo: '/' };
    await mount();
    expect(document.querySelector('h1')?.textContent).toBe('Team membership required');
    expectClosed('Team membership required');
    expect(state.scienceMounts).toBe(0);
    expect((await import('./lib/api')).getConfig).not.toHaveBeenCalled();
    expect(sessionStorage.getItem('sherloc.scanBrowserHash')).toBeNull();
  });

  it('mounts only after the authenticated config check succeeds', async () => {
    const { getConfig } = await import('./lib/api');
    let resolve!: (value: never) => void;
    vi.mocked(getConfig).mockReturnValue(new Promise((r) => { resolve = r; }));
    const pending = mount();
    await vi.waitFor(() => expect(getConfig).toHaveBeenCalled());
    expectClosed(null);
    expect(state.scienceMounts).toBe(0);
    resolve({ access_mode: 'public', tier: 'public', capabilities: {
      references: false, identification: false, experimental_candidates: false,
    } } as never);
    await pending;
    expect(document.querySelector('[data-testid="app-stub"]')).not.toBeNull();
  });

  it.each([401, 403, 503])('fails closed on authenticated check failure %s', async (status) => {
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Synthetic');
    const { getConfig } = await import('./lib/api');
    vi.mocked(getConfig).mockRejectedValue(new Error(`synthetic ${status}`));
    await mount();
    expectClosed();
    expect(state.scienceMounts).toBe(0);
    expect(sessionStorage.getItem('sherloc.scanBrowserHash')).toBeNull();
  });

  it.each(['http', 'network', 'invalid-json'])('fails closed on bootstrap %s failure', async (kind) => {
    vi.stubGlobal('fetch', kind === 'network'
      ? vi.fn().mockRejectedValue(new Error('synthetic network failure'))
      : vi.fn().mockResolvedValue({ ok: kind !== 'http', json: async () => { throw new Error('synthetic JSON failure'); } }));
    await mount();
    expectClosed();
    expect(state.scienceMounts).toBe(0);
  });

  it('fails closed when auth initialization throws', async () => {
    state.bootstrapFails = true;
    await mount();
    expectClosed();
    expect(state.scienceMounts).toBe(0);
  });

  it.each([false, true])('does not mount a signed-out visitor (redirect=%s)', async (redirect) => {
    state.authenticated = false;
    state.redirecting = redirect;
    await mount();
    expectClosed(redirect ? null : 'Sign-in did not complete');
    expect(state.scienceMounts).toBe(0);
    expect((await import('./lib/api')).getConfig).not.toHaveBeenCalled();
  });

  it.each([
    { access_mode: 'internal', tier: 'public', capabilities: {} },
    { access_mode: 'public', tier: 'public', capabilities: { references: true, identification: false, experimental_candidates: false } },
    { access_mode: 'internal', tier: 'team', capabilities: { references: false, identification: true, experimental_candidates: false } },
    { access_mode: 'internal', tier: 'team', capabilities: { references: 'false', identification: false, experimental_candidates: false } },
    { access_mode: 'internal', tier: 'team' },
  ])('rejects malformed or incompatible effective capabilities', async (config) => {
    vi.mocked((await import('./lib/api')).getConfig).mockResolvedValue(config as never);
    await mount();
    expectClosed();
    expect(state.scienceMounts).toBe(0);
  });

  it('destroys mounted science UI and clears stores on sign-out', async () => {
    await mount();
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Synthetic');
    const stores = await import('./lib/stores');
    stores.clearSessionState();
    expectClosed(null);
    expect(state.scienceMounts).toBe(1);
    expect(get(stores.appConfig)).toBeNull();
    expect(get(stores.scienceAuthorized)).toBe(false);
    expect(sessionStorage.getItem('sherloc.scanBrowserHash')).toBeNull();
  });
});
