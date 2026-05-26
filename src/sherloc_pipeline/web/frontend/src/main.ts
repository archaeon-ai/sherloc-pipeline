import './styles/global.css';
import App from './App.svelte';
import { bootstrapAuth } from './lib/auth';
import { features } from './lib/stores';
import type { ConfigResponse } from './lib/types';

// Resolve auth backend BEFORE mounting the app so the first wave of
// component-mount fetches can pick up the right Authorization header.
//
// Per §13.4 the SPA reads /api/config at startup to learn its auth
// mode. If that fetch fails we fall through to a passthrough cf-access
// session — the legacy code path — so a backend in trouble degrades
// to "anonymous" rather than blocking page load.
async function bootstrap(): Promise<App> {
  try {
    const res = await fetch('/api/config', { headers: { Accept: 'application/json' } });
    if (res.ok) {
      const cfg = (await res.json()) as ConfigResponse;
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
  return new App({ target: document.getElementById('app')! });
}

const appPromise = bootstrap();
export default appPromise;
