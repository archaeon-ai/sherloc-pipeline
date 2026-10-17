// main.ts bootstrap: an Auth0 error return mounts the access page, not the app (#308).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const authState = vi.hoisted(() => ({ error: null as null | { kind: string; returnTo: string } }));

vi.mock('./lib/auth', () => ({
  bootstrapAuth: vi.fn().mockResolvedValue({}),
  redirectingToLogin: () => false,
  authCallbackError: () => authState.error,
  getSession: () => null,
}));
vi.mock('./App.svelte', () => ({
  default: class {
    constructor({ target }: { target: HTMLElement }) {
      const stub = document.createElement('div');
      stub.dataset.testid = 'app-stub';
      target.appendChild(stub);
    }
  },
}));

beforeEach(() => {
  vi.resetModules();
  document.body.innerHTML = '<div id="app"></div>';
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ auth: {}, features: null }) }));
});

afterEach(() => {
  vi.unstubAllGlobals();
  authState.error = null;
});

describe('main.ts bootstrap', () => {
  it('mounts the access page when Auth0 refused the sign-in', async () => {
    authState.error = { kind: 'team_required', returnTo: '/' };
    await (await import('./main')).default;
    expect(document.querySelector('h1')?.textContent).toBe('Team membership required');
    expect(document.querySelector('[data-testid="app-stub"]')).toBeNull();
  });

  it('mounts the app otherwise', async () => {
    await (await import('./main')).default;
    expect(document.querySelector('[data-testid="app-stub"]')).not.toBeNull();
    expect(document.querySelector('h1')).toBeNull();
  });
});
