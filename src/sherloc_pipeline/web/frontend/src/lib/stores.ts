// ============================================================
// Svelte stores for shared application state
// ============================================================

import { writable, derived, get } from 'svelte/store';
import { processingState, undoStack, pointSelection } from './processingStore';
import { clearEvidence } from './evidenceStore';
import { resetMapState } from './stores/mapStore';
import type {
  BaselineParams,
  FitResponse,
  HealthResponse,
  ConfigResponse,
  FeaturesConfig,
} from './types';

// --- Access mode ---
export const accessMode = writable<'internal' | 'public'>('internal');
// Scan Browser session restoration waits for this signal so it cannot replace
// a bare startup route before App applies the public-mode default redirect.
export const accessModeResolved = writable(false);
export const scienceAuthorized = writable(false);
export let sessionGeneration = 0;
export const sessionRequests = new Set<AbortController>();

// --- Route ---
export const currentHash = writable(window.location.hash || '#/');

// --- Baseline params (carried from BaselineWorkspace to FittingWorkspace) ---
export const baselineParams = writable<BaselineParams>({
  method: 'aspls',
  lam: 1000000.0,
  max_iter: 10,
});

// --- Fit result ---
export const lastFitResult = writable<FitResponse | null>(null);

// --- System health ---
export const healthStatus = writable<HealthResponse | null>(null);

// --- Config ---
export const appConfig = writable<ConfigResponse | null>(null);

// --- Feature flags (issue #21) ---
// Source: /api/config `features` block. Default `pds_browser=true` so a
// legacy backend that omits the block keeps the tab visible — production
// templates opt out explicitly via SHERLOC_FEATURE_PDS_BROWSER=disabled.
export const features = writable<FeaturesConfig>({ pds_browser: true });

export function clearSessionState(): void {
  sessionGeneration += 1;
  for (const controller of sessionRequests) controller.abort();
  sessionRequests.clear();
  clearEvidence();
  scienceAuthorized.set(false);
  appConfig.set(null);
  features.set({ pds_browser: false });
  accessMode.set('public');
  accessModeResolved.set(false);
  currentHash.set('#/');
  lastFitResult.set(null);
  healthStatus.set(null);
  baselineParams.set({ method: 'aspls', lam: 1000000.0, max_iter: 10 });
  processingState.set(null);
  undoStack.set([]);
  pointSelection.set({ mode: 'average' });
  resetMapState();
  sessionStorage.removeItem('sherloc.scanBrowserHash');
}

let evidenceTier: string | null = null;
appConfig.subscribe(config => {
  const tier = config?.tier ?? null;
  if (tier === evidenceTier) return;
  evidenceTier = tier;
  if (get(scienceAuthorized)) clearSessionState();
  else clearEvidence();
});
let evidenceAccess = 'internal';
accessMode.subscribe(mode => {
  if (mode === evidenceAccess) return;
  evidenceAccess = mode;
  if (get(scienceAuthorized)) clearSessionState();
  else clearEvidence();
});

// --- Derived ---
export const healthIndicator = derived(healthStatus, ($h) => {
  if (!$h) return 'unknown';
  return $h.status;
});

// --- Route parsing ---
export interface ParsedRoute {
  page: 'browser' | 'open' | 'fit' | 'baseline' | 'workbench' | 'pds' | 'map';
  scanId?: string;
  queryParams?: Record<string, string>;
}

export const parsedRoute = derived(currentHash, ($hash): ParsedRoute => {
  const h = $hash.replace(/^#/, '');

  // Split path from query string
  const qIdx = h.indexOf('?');
  const path = qIdx >= 0 ? h.slice(0, qIdx) : h;
  const queryParams: Record<string, string> = {};
  if (qIdx >= 0) {
    const sp = new URLSearchParams(h.slice(qIdx + 1));
    sp.forEach((value, key) => { queryParams[key] = value; });
  }

  const mapMatch = path.match(/^\/scan\/([^/]+)\/map$/);
  if (mapMatch) return { page: 'map', scanId: mapMatch[1], queryParams };

  const workbenchMatch = path.match(/^\/scan\/([^/]+)\/workbench$/);
  if (workbenchMatch) return { page: 'workbench', scanId: workbenchMatch[1], queryParams };

  const scanFitMatch = path.match(/^\/scan\/([^/]+)\/fit$/);
  if (scanFitMatch) return { page: 'fit', scanId: scanFitMatch[1], queryParams };

  const scanBaselineMatch = path.match(/^\/scan\/([^/]+)\/baseline$/);
  if (scanBaselineMatch) return { page: 'baseline', scanId: scanBaselineMatch[1], queryParams };

  if (path === '/pds') return { page: 'pds', queryParams };
  if (path === '/open') return { page: 'open', queryParams };

  return { page: 'browser', queryParams };
});

// --- Navigation helper ---
export function navigate(hash: string): void {
  window.location.hash = hash;
  currentHash.set(window.location.hash);
}
