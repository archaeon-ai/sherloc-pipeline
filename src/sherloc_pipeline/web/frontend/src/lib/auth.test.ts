import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createAuth0Client } from '@auth0/auth0-spa-js';
import { _resetForTests, bootstrapAuth, safeReturnTo } from './auth';
import type { AuthConfig } from './types';

vi.mock('@auth0/auth0-spa-js', () => ({ createAuth0Client: vi.fn() }));

const cfg: AuthConfig = {
  auth_mode: 'auth0',
  auth0_domain: 'tenant.example.test',
  auth0_client_id: 'client-id',
  auth0_audience: 'https://api.example.test',
  role_claim_uri: null,
};

const resolverPath = '/#/open?scan_name=Testrock_detail&sol=42424&point=0';

function fakeClient(appState?: unknown) {
  return {
    handleRedirectCallback: vi.fn().mockResolvedValue({ appState }),
    getTokenSilently: vi.fn().mockRejectedValue({ error: 'login_required' }),
    loginWithRedirect: vi.fn().mockResolvedValue(undefined),
    logout: vi.fn(),
    isAuthenticated: vi.fn().mockResolvedValue(false),
  };
}

beforeEach(() => {
  _resetForTests();
  vi.resetAllMocks();
});

afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('Auth0 login round trip', () => {
  it('carries the resolver URL into login and restores it after the callback', async () => {
    window.history.replaceState(null, '', resolverPath);
    const before = fakeClient();
    vi.mocked(createAuth0Client).mockResolvedValue(before as never);
    const session = await bootstrapAuth(cfg);
    await session.login();
    expect(before.loginWithRedirect).toHaveBeenCalledWith({ appState: { returnTo: resolverPath } });

    // Auth0 redirects back to the callback carrying the stored appState.
    _resetForTests();
    window.history.replaceState(null, '', '/auth/callback?code=c&state=s');
    const after = fakeClient({ returnTo: resolverPath });
    vi.mocked(createAuth0Client).mockResolvedValue(after as never);
    await bootstrapAuth(cfg);
    expect(after.handleRedirectCallback).toHaveBeenCalled();
    expect(`${window.location.pathname}${window.location.search}${window.location.hash}`).toBe(resolverPath);
  });

  it('falls back to the app root when the callback has no return URL', async () => {
    window.history.replaceState(null, '', '/auth/callback?code=c&state=s');
    vi.mocked(createAuth0Client).mockResolvedValue(fakeClient() as never);
    await bootstrapAuth(cfg);
    expect(`${window.location.pathname}${window.location.search}${window.location.hash}`).toBe('/');
  });
});

describe('safeReturnTo', () => {
  it.each([
    [undefined, '/'],
    [42, '/'],
    ['https://evil.example/', '/'],
    ['//evil.example/#/open', '/'],
    ['/auth/callback?code=c&state=s', '/'],
    ['/#/scan/abc?points=1,2', '/#/scan/abc?points=1,2'],
  ])('%s → %s', (input, expected) => {
    expect(safeReturnTo(input)).toBe(expected);
  });
});
