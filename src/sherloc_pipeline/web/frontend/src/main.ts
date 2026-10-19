import './styles/global.css';
import App from './App.svelte';
import AccessError from './components/AccessError.svelte';
import { authCallbackError, bootstrapAuth, redirectingToLogin } from './lib/auth';
import { getConfig } from './lib/api';
import {
  accessMode, appConfig, clearSessionState, currentHash, features, scienceAuthorized,
} from './lib/stores';
import type { BootstrapConfigResponse } from './lib/types';

async function bootstrap(): Promise<App | AccessError | null> {
  clearSessionState();
  try {
    const res = await fetch('/api/config/bootstrap', { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error('Authentication bootstrap unavailable');
    const cfg = (await res.json()) as BootstrapConfigResponse;
    const session = await bootstrapAuth(cfg.auth);
    if (redirectingToLogin()) return null;
    const refused = authCallbackError();
    if (refused) {
      return new AccessError({ target: document.getElementById('app')!, props: { error: refused } });
    }
    if (!(await session.isAuthenticated())) throw new Error('Sign-in required');
    // The backend validates identity, audience and tier authorization here,
    // including CF Access sessions whose cookies the SPA cannot inspect.
    const config = await getConfig();
    if (
      !['internal', 'public'].includes(config.access_mode)
      || config.tier !== (config.access_mode === 'public' ? 'public' : 'team')
      || !config.capabilities
      || Object.values(config.capabilities).some((value) => typeof value !== 'boolean')
      || typeof config.capabilities.references !== 'boolean'
      || typeof config.capabilities.identification !== 'boolean'
      || typeof config.capabilities.experimental_candidates !== 'boolean'
      || (config.tier === 'public' && config.capabilities.references)
      || (config.capabilities.identification && !config.capabilities.references)
    ) throw new Error('Invalid authorized configuration');
    appConfig.set(config);
    accessMode.set(config.access_mode);
    features.set(config.features ?? { pds_browser: false });
    currentHash.set(window.location.hash || '#/');
    scienceAuthorized.set(true);
    const app = new App({ target: document.getElementById('app')! });
    const unsubscribe = scienceAuthorized.subscribe((authorized) => {
      if (!authorized) app.$destroy();
    });
    // This module owns the one application instance for the page lifetime.
    window.addEventListener('pagehide', unsubscribe, { once: true });
    return app;
  } catch {
    clearSessionState();
    return new AccessError({
      target: document.getElementById('app')!,
      props: { error: { kind: 'generic', returnTo: '/' } },
    });
  }
}

export default bootstrap();
