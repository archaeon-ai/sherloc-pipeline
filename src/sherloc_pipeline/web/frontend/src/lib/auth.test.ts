import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createAuth0Client } from '@auth0/auth0-spa-js';
import {
  _resetForTests,
  authCallbackError,
  bootstrapAuth,
  classifyAuthError,
  openWithWsAuth,
  redirectingToLogin,
  safeReturnTo,
} from './auth';
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

describe('Sign in first (auth0 mode)', () => {
  it('sends a signed-out visitor to the login page, returning to the same link', async () => {
    window.history.replaceState(null, '', resolverPath);
    const client = fakeClient();
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    expect(client.loginWithRedirect).toHaveBeenCalledWith({ appState: { returnTo: resolverPath } });
    expect(redirectingToLogin()).toBe(true);
  });

  it('does not redirect a signed-in visitor', async () => {
    const client = fakeClient();
    client.getTokenSilently.mockResolvedValue('token');
    client.isAuthenticated.mockResolvedValue(true);
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    expect(client.loginWithRedirect).not.toHaveBeenCalled();
    expect(redirectingToLogin()).toBe(false);
  });

  it('does not loop when Auth0 returns an error', async () => {
    window.history.replaceState(null, '', '/auth/callback?error=access_denied&state=s');
    const client = fakeClient();
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    expect(client.loginWithRedirect).not.toHaveBeenCalled();
    expect(redirectingToLogin()).toBe(false);
  });

  it('mounts the app with its Log in button when the redirect cannot start', async () => {
    const client = fakeClient();
    client.loginWithRedirect.mockRejectedValue(new Error('popup blocked'));
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const session = await bootstrapAuth(cfg);
    expect(client.loginWithRedirect).toHaveBeenCalled();
    expect(redirectingToLogin()).toBe(false);
    expect(session.showsLoginUi).toBe(true);
    warn.mockRestore();
  });

  it('never redirects in dev or cf-access mode', async () => {
    await bootstrapAuth({ ...cfg, auth_mode: 'dev' });
    expect(createAuth0Client).not.toHaveBeenCalled();
    expect(redirectingToLogin()).toBe(false);
  });
});

describe('Auth0 error return (#308)', () => {
  const teamDenied =
    '/auth/callback?error=access_denied&error_description=Team%20membership%20required%20for%20this%20application.&state=s';

  it('records team_required, keeps the start URL and leaves the callback URL', async () => {
    window.history.replaceState(null, '', teamDenied);
    const client = fakeClient();
    const err = Object.assign(new Error('Team membership required'), { appState: { returnTo: resolverPath } });
    client.handleRedirectCallback.mockRejectedValue(err);
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    expect(authCallbackError()).toEqual({ kind: 'team_required', returnTo: resolverPath });
    expect(`${window.location.pathname}${window.location.search}${window.location.hash}`).toBe(resolverPath);
    expect(client.loginWithRedirect).not.toHaveBeenCalled();
    expect(client.getTokenSilently).not.toHaveBeenCalled();
    expect(redirectingToLogin()).toBe(false);
  });

  it('falls back to the app root when the transaction is gone', async () => {
    window.history.replaceState(null, '', teamDenied);
    const client = fakeClient();
    client.handleRedirectCallback.mockRejectedValue(new Error('missing_transaction'));
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    expect(authCallbackError()?.returnTo).toBe('/');
    expect(window.location.pathname).toBe('/');
  });

  it('any other error is generic', async () => {
    window.history.replaceState(null, '', '/auth/callback?error=server_error&state=s');
    vi.mocked(createAuth0Client).mockResolvedValue(fakeClient() as never);
    await bootstrapAuth(cfg);
    expect(authCallbackError()?.kind).toBe('generic');
  });

  it('a successful callback records no error', async () => {
    window.history.replaceState(null, '', '/auth/callback?code=c&state=s');
    vi.mocked(createAuth0Client).mockResolvedValue(fakeClient() as never);
    await bootstrapAuth(cfg);
    expect(authCallbackError()).toBeNull();
  });

  it('classifies like the Worker', () => {
    expect(classifyAuthError('access_denied', 'Team membership required for this application.')).toBe('team_required');
    expect(classifyAuthError('access_denied', 'Please verify your email before logging in.')).toBe('verify_email');
    expect(classifyAuthError('access_denied', 'Your team request is under review.')).toBe('team_pending');
    expect(classifyAuthError('access_denied', 'something else')).toBe('denied');
    expect(classifyAuthError('login_required', 'Team membership required')).toBe('generic');
  });

  it('login can force the Auth0 login screen and return to a given URL', async () => {
    const client = fakeClient();
    client.getTokenSilently.mockResolvedValue('token');
    client.isAuthenticated.mockResolvedValue(true);
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    const session = await bootstrapAuth(cfg);
    await session.login({ prompt: 'login', returnTo: resolverPath });
    expect(client.loginWithRedirect).toHaveBeenCalledWith({
      appState: { returnTo: resolverPath },
      authorizationParams: { prompt: 'login' },
    });
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

describe('openWithWsAuth', () => {
  it('offers the access token as the bearer subprotocol in auth0 mode', async () => {
    const client = fakeClient();
    client.getTokenSilently.mockResolvedValue('tok.en');
    client.isAuthenticated.mockResolvedValue(true);
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    await bootstrapAuth(cfg);
    const open = vi.fn();
    openWithWsAuth(open);
    await vi.waitFor(() => expect(open).toHaveBeenCalledWith(['bearer', 'tok.en']));
  });

  it('opens at once with no subprotocol outside auth0 mode', async () => {
    await bootstrapAuth({ ...cfg, auth_mode: 'dev' });
    const open = vi.fn();
    openWithWsAuth(open);
    expect(open).toHaveBeenCalledWith();
  });
});

describe('Session cleanup', () => {
  it('rejects an unknown bootstrap auth mode', async () => {
    await expect(bootstrapAuth({ auth_mode: 'typo' } as never)).rejects.toThrow('Unknown authentication mode');
  });

  it('clears science, processing, undo, map and navigation stores on logout', async () => {
    const stores = await import('./stores');
    const processing = await import('./processingStore');
    const map = await import('./stores/mapStore');
    const client = fakeClient();
    client.isAuthenticated.mockResolvedValue(true);
    vi.mocked(createAuth0Client).mockResolvedValue(client as never);
    const session = await bootstrapAuth(cfg);
    stores.scienceAuthorized.set(true);
    stores.lastFitResult.set({ synthetic: true } as never);
    processing.processingState.set({ synthetic: true } as never);
    processing.undoStack.set([{ synthetic: true }] as never);
    processing.pointSelection.set({ mode: 'point', pointIdx: 2 } as never);
    map.mapScanId.set('synthetic-scan');
    map.mapFittedResultsCache.set([{ synthetic: true }] as never);
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Synthetic');
    await session.logout();
    const { get } = await import('svelte/store');
    expect(get(stores.scienceAuthorized)).toBe(false);
    expect(get(stores.lastFitResult)).toBeNull();
    expect(get(processing.processingState)).toBeNull();
    expect(get(processing.undoStack)).toEqual([]);
    expect(get(processing.pointSelection)).toEqual({ mode: 'average' });
    expect(get(map.mapScanId)).toBeNull();
    expect(get(map.mapFittedResultsCache)).toEqual([]);
    expect(sessionStorage.getItem('sherloc.scanBrowserHash')).toBeNull();
  });
});
