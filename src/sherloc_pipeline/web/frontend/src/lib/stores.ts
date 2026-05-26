// ============================================================
// Svelte stores for shared application state
// ============================================================

import { writable, derived } from 'svelte/store';
import type {
  ScanDetail,
  SpectrumResponse,
  BaselineParams,
  FitResponse,
  HealthResponse,
  ConfigResponse,
  FeaturesConfig,
} from './types';

// --- Access mode ---
export const accessMode = writable<'internal' | 'public'>('internal');

// --- Route ---
export const currentHash = writable(window.location.hash || '#/');

// --- Selected scan ---
export const selectedScan = writable<ScanDetail | null>(null);
export const selectedSpectrum = writable<SpectrumResponse | null>(null);

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

// --- Derived ---
export const healthIndicator = derived(healthStatus, ($h) => {
  if (!$h) return 'unknown';
  return $h.status;
});

// --- Route parsing ---
export interface ParsedRoute {
  page: 'browser' | 'scan' | 'fit' | 'baseline' | 'workbench' | 'pds' | 'map';
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

  const scanMatch = path.match(/^\/scan\/([^/]+)$/);
  if (scanMatch) return { page: 'scan', scanId: scanMatch[1], queryParams };

  if (path === '/pds') return { page: 'pds', queryParams };

  return { page: 'browser', queryParams };
});

// --- Navigation helper ---
export function navigate(hash: string): void {
  window.location.hash = hash;
  currentHash.set(window.location.hash);
}
