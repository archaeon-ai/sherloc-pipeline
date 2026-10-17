import './styles/global.css';
import App from './App.svelte';
import AccessError from './components/AccessError.svelte';
import { authCallbackError, bootstrapAuth, redirectingToLogin } from './lib/auth';
import { currentHash, features } from './lib/stores';
import type { BootstrapConfigResponse } from './lib/types';

// Resolve auth backend BEFORE mounting the app so the first wave of
// component-mount fetches can pick up the right Authorization header.
//
// Per §13.4 the SPA reads /api/config/bootstrap (the anonymous auth and
// feature blocks; the full /api/config needs a token) to learn its auth
// mode. If that fetch fails we fall through to a passthrough cf-access
// session — the legacy code path — so a backend in trouble degrades
// to "anonymous" rather than blocking page load.
async function bootstrap(): Promise<App | AccessError | null> {
  try {
    const res = await fetch('/api/config/bootstrap', { headers: { Accept: 'application/json' } });
    if (res.ok) {
      const cfg = (await res.json()) as BootstrapConfigResponse;
      await bootstrapAuth(cfg.auth);
      if (cfg.features) features.set(cfg.features);
    } else {
      await bootstrapAuth({
        auth_mode: 'cf-access',
        auth0_domain: null,
        auth0_client_id: null,
        auth0_audience: null,
        role_claim_uri: null,
      });
    }
  } catch {
    await bootstrapAuth({
      auth_mode: 'cf-access',
      auth0_domain: null,
      auth0_client_id: null,
      auth0_audience: null,
      role_claim_uri: null,
    });
  }
  // The Auth0 callback may have restored a pre-login URL (e.g. #/open...);
  // replaceState fires no hashchange, so resync the route store before mount.
  currentHash.set(window.location.hash || '#/');
  // Signed out in auth0 mode: the browser is on its way to the login page.
  if (redirectingToLogin()) return null;
  // Auth0 refused the sign-in: say why instead of mounting an empty app.
  const refused = authCallbackError();
  if (refused) return new AccessError({ target: document.getElementById('app')!, props: { error: refused } });
  return new App({ target: document.getElementById('app')! });
}

const appPromise = bootstrap();
export default appPromise;
