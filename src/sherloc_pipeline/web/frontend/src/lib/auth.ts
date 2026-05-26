// ============================================================
// Frontend auth bootstrap (§13.4)
//
// The same SPA build runs against any auth backend without rebuild.
// At startup the SPA fetches /api/config and reads the `auth` block
// to decide which auth provider — if any — to wire up:
//
//   auth_mode === "auth0"     → bootstrap @auth0/auth0-spa-js with the
//                                runtime config; expose login/logout
//                                and a getToken() that returns a fresh
//                                access token.
//   auth_mode === "cf-access" → no client-side wiring needed; CF Access
//                                cookies are managed at the edge and
//                                are transparent to the SPA.
//   auth_mode === "dev"       → no real auth; getToken() returns "" and
//                                the backend DevValidator ignores the
//                                token entirely (§13.5; localhost-only).
//
// The auth0 path is dead code in any cf-access deployment, so this
// module ships safely to the frozen legacy SPA build (§13.6).
// ============================================================

import { Auth0Client, createAuth0Client } from '@auth0/auth0-spa-js';
import type { AuthConfig } from './types';

export interface AuthSession {
  /** Returns an access token suitable for `Authorization: Bearer ...`,
   *  or "" when no token is needed (cf-access mode). */
  getToken(): Promise<string>;
  /** Triggers an interactive login flow (auth0 mode only). */
  login(): Promise<void>;
  /** Logs the user out (auth0 mode only). */
  logout(): Promise<void>;
  /** True when the user has a valid session (auth0 mode); always
   *  true for cf-access and dev (the backend handles identity). */
  isAuthenticated(): Promise<boolean>;
  /** True when the auth backend exposes login/logout UX. */
  readonly showsLoginUi: boolean;
  /** Mirrors the AuthConfig.auth_mode for diagnostic surfaces. */
  readonly mode: AuthConfig['auth_mode'];
}

/** Module-level singleton; initialized on first bootstrapAuth() call. */
let session: AuthSession | null = null;

// Auth-readiness gate: protected-resource helpers (fetchAciImage, getMapLayers
// in api.ts) await this promise before checking session state, so the first
// component mount cannot race the auth bootstrap into anonymous protected
// requests. Resolves on the first bootstrapAuth() call, success OR failure.
// Exported as `let` so the test-only reset can swap the promise reference
// (TS module exports of let-bindings are live-binding for consumers).
let resolveBootstrap!: (s: AuthSession | null) => void;
export let bootstrapAuthReady: Promise<AuthSession | null> = new Promise((r) => {
  resolveBootstrap = r;
});

/**
 * Initialize the SPA's auth backend from the AuthConfig returned by
 * /api/config. Idempotent; the first call wins.
 *
 * NOTE: callers should treat the returned session as the canonical
 * auth handle for the lifetime of the page. Re-bootstrapping after
 * mode changes requires a hard reload.
 */
export async function bootstrapAuth(cfg: AuthConfig): Promise<AuthSession> {
  if (session !== null) return session;
  try {
    if (cfg.auth_mode === 'auth0') {
      session = await buildAuth0Session(cfg);
    } else if (cfg.auth_mode === 'cf-access') {
      session = buildPassThroughSession('cf-access');
    } else {
      session = buildPassThroughSession('dev');
    }
  } finally {
    resolveBootstrap(session);
  }
  return session;
}

/** Synchronous accessor; returns null until bootstrapAuth() resolves. */
export function getSession(): AuthSession | null {
  return session;
}

/** Test-only seam to drop the singleton between scenarios. */
export function _resetForTests(): void {
  session = null;
  // Reset the bootstrap-readiness gate so the next bootstrapAuth() call
  // settles a fresh promise (consumers see the new reference via TS
  // live-binding for let exports).
  bootstrapAuthReady = new Promise<AuthSession | null>((r) => {
    resolveBootstrap = r;
  });
}

// ---------------------------------------------------------------
// Auth0 backend
// ---------------------------------------------------------------

async function buildAuth0Session(cfg: AuthConfig): Promise<AuthSession> {
  if (!cfg.auth0_domain || !cfg.auth0_client_id || !cfg.auth0_audience) {
    throw new Error(
      'AuthConfig.auth_mode is "auth0" but auth0_domain / ' +
        'auth0_client_id / auth0_audience are not all populated. ' +
        'Backend config_check should have caught this; check ' +
        '/api/config response.',
    );
  }

  // Per spec §13.0.2 the Auth0 Allowed Callback URLs are
  // <origin>/auth/callback for each deployment hostname. The redirect URI
  // here MUST match one of those entries exactly or Auth0 rejects the
  // login redirect before the user reaches the app. Using the bare origin
  // (the prior B.7c behavior) breaks the production login path entirely.
  const callbackUrl = `${window.location.origin}/auth/callback`;

  const client: Auth0Client = await createAuth0Client({
    domain: cfg.auth0_domain,
    clientId: cfg.auth0_client_id,
    authorizationParams: {
      audience: cfg.auth0_audience,
      redirect_uri: callbackUrl,
    },
    // In-memory only: a stale localStorage token would survive across
    // an auth-mode change and confuse the frozen legacy cf-access SPA
    // (§13.9.2). Memory storage avoids that attack surface entirely;
    // silent-auth handles refresh per spec §13.4.
    cacheLocation: 'memory',
    useRefreshTokens: false,
  });

  // Handle the redirect callback if Auth0 added the standard query params.
  // The backend SPA-fallback (app.py) serves index.html for /auth/callback
  // so the bootstrap runs whenever Auth0 lands the user there.
  const params = new URLSearchParams(window.location.search);
  const isCallback = params.has('code') && params.has('state');
  if (isCallback) {
    try {
      await client.handleRedirectCallback();
      // Strip auth params AND the /auth/callback path so the address bar
      // shows the app root after login. (Future: preserve the user's
      // original requested URL via Auth0 appState.)
      window.history.replaceState({}, document.title, '/');
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Auth0 redirect callback failed:', err);
    }
  } else {
    // Silent SSO on mount (Session 93 design): if the user has an active
    // Auth0 SSO cookie at auth.m2020-phase.net (e.g., from a prior login
    // at apex/dashboard or viewer via the same tenant), getTokenSilently()
    // completes the OAuth handshake via hidden iframe and the user lands
    // authenticated without clicking Log in. Failure paths fall through
    // to the existing "Log in button" UX — no regression.
    //
    // Skipped on the callback path because handleRedirectCallback above
    // already populates the SDK cache; an immediate getTokenSilently()
    // would be redundant.
    try {
      await client.getTokenSilently({ cacheMode: 'on' });
    } catch (err) {
      const code = (err as { error?: string })?.error;
      if (
        code === 'login_required' ||
        code === 'consent_required' ||
        code === 'interaction_required'
      ) {
        // Expected: user has no Auth0 session yet. Quiet fallback.
      } else {
        // Unexpected: misconfig, CORS, tenant-domain mismatch, etc.
        // Log sanitized — never log tokens or full authorization URLs.
        // eslint-disable-next-line no-console
        console.warn(
          'Silent SSO unexpected error:',
          code ?? 'unknown',
          (err as Error)?.message?.slice(0, 200) ?? '',
        );
      }
    }
  }

  return {
    mode: 'auth0',
    showsLoginUi: true,
    async getToken(): Promise<string> {
      try {
        return await client.getTokenSilently();
      } catch {
        // Silent auth fails when the user isn't logged in; return empty
        // so the request goes anonymous and the API decides what to do.
        return '';
      }
    },
    async login(): Promise<void> {
      await client.loginWithRedirect();
    },
    async logout(): Promise<void> {
      await client.logout({
        logoutParams: { returnTo: window.location.origin },
      });
    },
    async isAuthenticated(): Promise<boolean> {
      return client.isAuthenticated();
    },
  };
}

// ---------------------------------------------------------------
// CF Access / dev backends — no SDK, no UI affordance
// ---------------------------------------------------------------

function buildPassThroughSession(mode: 'cf-access' | 'dev'): AuthSession {
  return {
    mode,
    showsLoginUi: false,
    async getToken(): Promise<string> {
      // CF Access: cookies are at the edge — Authorization not needed.
      // Dev:       DevValidator ignores the token; empty string is fine.
      return '';
    },
    async login(): Promise<void> {
      // No-op: cf-access uses Cloudflare's own login flow at the edge;
      // dev mode has no login.
    },
    async logout(): Promise<void> {
      // No-op for the same reason.
    },
    async isAuthenticated(): Promise<boolean> {
      return true;
    },
  };
}
