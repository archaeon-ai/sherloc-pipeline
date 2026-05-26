// ============================================================
// Typed API client for SHERLOC Web API v1.0.0
// ============================================================

import type {
  ScanListResponse,
  ScanDetailResponse,
  ScanPointsResponse,
  SpectrumResponse,
  BaselineRequest,
  BaselineResponse,
  FitRequest,
  FitResponse,
  ConfigResponse,
  HealthResponse,
  PdsAvailableResponse,
  PdsDownloadRequest,
  PdsDownloadResponse,
  ScanFilterParams,
  SpectrumParams,
  SubsetRequest,
  DespikeRequest,
  DespikeResponse,
  BackgroundRequest,
  BackgroundResponse,
  AccessModeResponse,
} from './types';

import { bootstrapAuthReady, getSession } from './auth';
import type { DisplayPoint, VoronoiGeometry } from './types/map';

const API_BASE = '/api';

export class ApiError extends Error {
  constructor(
    public status: number,
    public errorClass: string,
    public detail?: string,
  ) {
    super(`${errorClass}${detail ? ': ' + detail : ''}`);
    this.name = 'ApiError';
  }
}

/**
 * Thrown by protected helpers (fetchAciImage, getMapLayers) when no
 * authenticated session exists at request time. Distinct from ApiError
 * so call sites can render an "auth required" UI state instead of an
 * "operation failed" banner (Session 93 design memo §2.2).
 */
export class AuthRequiredError extends Error {
  constructor() {
    super('Authenticated session required');
    this.name = 'AuthRequiredError';
  }
}

/**
 * Gate protected requests. Awaits the auth-bootstrap settlement (so the
 * initial silent-SSO has a chance to mint a token), then verifies the
 * session is authenticated for auth0 mode. cf-access + dev modes pass
 * through (isAuthenticated() returns true in those modes).
 *
 * Throws AuthRequiredError BEFORE any network request when:
 *   - bootstrap settled with no session at all, OR
 *   - auth0 mode but isAuthenticated() === false.
 */
async function ensureAuthenticated(): Promise<void> {
  await bootstrapAuthReady;
  const session = getSession();
  if (session === null) throw new AuthRequiredError();
  if (session.mode === 'auth0' && !(await session.isAuthenticated())) {
    throw new AuthRequiredError();
  }
}

/**
 * Resolve an Authorization header for the active auth backend.
 * Returns an empty object in cf-access mode (no header needed),
 * before bootstrapAuth() resolves, or whenever getToken() returns "".
 */
async function authHeaders(): Promise<Record<string, string>> {
  const session = getSession();
  if (session === null || session.mode !== 'auth0') return {};
  const token = await session.getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(await authHeaders()),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let errorClass = `HTTP ${res.status}`;
    let detail: string | undefined;
    try {
      const body = await res.json();
      errorClass = body.error || errorClass;
      detail = body.detail;
    } catch {
      // response body was not JSON
    }
    throw new ApiError(res.status, errorClass, detail);
  }
  return res.json();
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildQuery(params: any): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
    }
  }
  return parts.length > 0 ? `?${parts.join('&')}` : '';
}

// --- Scans ---

export async function getScans(params: ScanFilterParams = {}): Promise<ScanListResponse> {
  const query = buildQuery(params);
  return fetchJson<ScanListResponse>(`/scans${query}`);
}

export async function getScan(scanId: string): Promise<ScanDetailResponse> {
  return fetchJson<ScanDetailResponse>(`/scans/${scanId}`);
}

export async function getScanPoints(scanId: string): Promise<ScanPointsResponse> {
  return fetchJson<ScanPointsResponse>(`/scans/${scanId}/points`);
}

// --- Spectra ---

export async function getAverageSpectrum(
  scanId: string,
  params: SpectrumParams = {},
): Promise<SpectrumResponse> {
  const query = buildQuery(params);
  return fetchJson<SpectrumResponse>(`/spectra/${scanId}/average${query}`);
}

/**
 * Authenticated fetch of a single point's spectrum. Calls
 * `ensureAuthenticated()` so callers can branch on AuthRequiredError vs
 * ApiError (same contract as `getMapLayers` / `getMapData`); without the
 * gate, raw-cookie fetches collapsed every auth failure into a generic
 * "Failed to load point spectrum" red banner (issue #17).
 *
 * Caller-facing failure modes:
 *   - AuthRequiredError → render "Log in to view spectrum" placeholder.
 *   - ApiError          → render "Failed to load spectrum" placeholder.
 */
export async function getPointSpectrum(
  scanId: string,
  idx: number,
  params: SpectrumParams = {},
): Promise<SpectrumResponse> {
  await ensureAuthenticated();
  const query = buildQuery(params);
  return fetchJson<SpectrumResponse>(`/spectra/${scanId}/point/${idx}${query}`);
}

export async function postSubsetAverage(
  scanId: string,
  body: SubsetRequest,
): Promise<SpectrumResponse> {
  return fetchJson<SpectrumResponse>(`/spectra/${scanId}/subset`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// --- Processing ---

export async function postBaseline(body: BaselineRequest): Promise<BaselineResponse> {
  return fetchJson<BaselineResponse>('/process/baseline', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function postFit(body: FitRequest): Promise<FitResponse> {
  return fetchJson<FitResponse>('/process/fit', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// --- Config ---

export async function getConfig(): Promise<ConfigResponse> {
  return fetchJson<ConfigResponse>('/config');
}

// --- Health ---

export async function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/health');
}

// --- PDS ---

export async function getPdsAvailableSols(): Promise<PdsAvailableResponse> {
  // Catalog endpoint returns {available_sols: [{sol}], total_available, already_ingested}
  // Map to the frontend PdsAvailableResponse shape
  const raw = await fetchJson<Record<string, unknown>>('/pds/catalog');
  const availSols = (raw.available_sols as Array<{ sol: number }>) ?? [];
  return {
    schema_version: (raw.schema_version as string) ?? '1.0.0',
    sols: availSols.map((s) => ({ sol: s.sol, n_scans: 0, data_volume_mb: null, pds_url: '' })),
    total: (raw.total_available as number) ?? availSols.length,
    already_ingested: (raw.already_ingested as number[]) ?? [],
  };
}

export async function postPdsDownload(body: PdsDownloadRequest): Promise<PdsDownloadResponse> {
  return fetchJson<PdsDownloadResponse>('/pds/download', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// --- Processing: Despike ---

export async function postDespike(body: DespikeRequest): Promise<DespikeResponse> {
  return fetchJson<DespikeResponse>('/process/despike', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// --- Processing: Background ---

export async function postBackground(body: BackgroundRequest): Promise<BackgroundResponse> {
  return fetchJson<BackgroundResponse>('/process/background', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// --- Config: Access Mode ---

export async function getAccessMode(): Promise<AccessModeResponse> {
  return fetchJson<AccessModeResponse>('/config/access-mode');
}

// --- Images: ACI ---

/**
 * @deprecated Use `fetchAciImage()` — `getAciImageUrl()` returns a raw URL
 * intended for `<img src>` consumption, but browser-native image fetches do
 * NOT attach `Authorization: Bearer` headers, so the request lands at the
 * SHERLOC backend anonymous and `phase-platform-auth` returns 401 under
 * Auth0 Bearer-token mode (Session 93 design memo). Kept for any external
 * caller not migrated yet; remove once all call sites use `fetchAciImage`.
 */
export function getAciImageUrl(scanId: string, colorized = false): string {
  const params = new URLSearchParams();
  if (colorized) params.set('colorized', 'true');
  params.set('_v', '4'); // cache-bust: serve base ACI not angle-range overlay
  return `${API_BASE}/images/${scanId}/aci?${params.toString()}`;
}

export interface AciFetchOpts {
  colorized?: boolean;
  enhanced?: boolean;
  upscale?: number;
}

/**
 * Authenticated fetch of an ACI image. Returns a decoded HTMLImageElement
 * suitable for canvas rendering. The blob URL backing the image is revoked
 * after onload; the HTMLImageElement retains its decoded bitmap.
 *
 * Throws:
 *   - AuthRequiredError if no authenticated session (no network request made).
 *   - ApiError on backend non-2xx response.
 *
 * Caller-facing failure modes (Session 93 design §2.3):
 *   - AuthRequiredError → render "Log in to view ACI" placeholder.
 *   - ApiError → render "Failed to load ACI image" placeholder.
 */
export async function fetchAciImage(
  scanId: string,
  opts: AciFetchOpts = {},
): Promise<HTMLImageElement> {
  await ensureAuthenticated();
  const params = new URLSearchParams();
  if (opts.colorized) params.set('colorized', 'true');
  if (opts.enhanced) params.set('enhanced', 'true');
  if (opts.upscale && opts.upscale > 1) params.set('upscale', String(opts.upscale));
  params.set('_v', '4');
  const url = `${API_BASE}/images/${encodeURIComponent(scanId)}/aci?${params.toString()}`;

  const res = await fetch(url, { headers: await authHeaders() });
  if (!res.ok) {
    throw new ApiError(res.status, `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('Image decode failed'));
      img.src = blobUrl;
    });
    return img;
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

// --- Map: layers ---

/**
 * Map-layers response from `/api/map/layers/{scan_id}`. The backend route
 * lives in `routes/map.py`; this interface captures the fields consumed
 * by `MapMode.svelte` (point_set, base_images, available_layers,
 * coordinate_source). Unknown fields pass through as the raw shape; the
 * call site narrows as needed.
 */
export interface MapLayersResponse {
  coordinate_source?: string;
  point_set?: {
    points: DisplayPoint[];
    voronoi: VoronoiGeometry | null;
  };
  base_images?: Array<{ type: string; url: string }>;
  available_layers?: Record<string, Record<string, { n_detections: number; classes: string[] }>>;
  // Future fields tolerated.
  [key: string]: unknown;
}

/**
 * Authenticated fetch of map-layer metadata for a scan. Replaces the raw
 * `fetch('/api/map/layers/...')` pattern in `MapMode.svelte` which bypassed
 * the auth-attaching wrapper and produced 401s under Auth0 Bearer-token mode.
 *
 * Throws AuthRequiredError or ApiError per the same contract as `fetchAciImage`.
 */
export async function getMapLayers(scanId: string): Promise<MapLayersResponse> {
  await ensureAuthenticated();
  return fetchJson<MapLayersResponse>(`/map/layers/${encodeURIComponent(scanId)}`);
}

/**
 * Single point in a map-data response (per-domain or per-class SNR / center / etc.).
 * Loose enough to absorb future fields without breaking callers.
 */
export interface MapDataPoint {
  index: number;
  value: number | null;
  status: string;
  assignment?: string | null;
  center_cm1?: number | null;
  center_nm?: number | null;
}

export interface MapDataResponse {
  points: MapDataPoint[];
  [key: string]: unknown;
}

/**
 * Authenticated fetch of per-domain or per-class map layer data. Used by
 * `MapMode.svelte loadLayerData()` to populate the canvas overlay values
 * after the user picks a display mode. Replaces a raw `fetch()` that
 * skipped Bearer auth (Codex PR9 R3 F5).
 */
export async function getMapData(
  scanId: string,
  domain: string,
  valueType: string,
  classId?: string,
): Promise<MapDataResponse> {
  await ensureAuthenticated();
  const params = new URLSearchParams();
  params.set('domain', domain);
  params.set('value_type', valueType);
  if (classId) params.set('class_id', classId);
  return fetchJson<MapDataResponse>(
    `/map/data/${encodeURIComponent(scanId)}?${params.toString()}`,
  );
}

export interface MapFitResponse {
  job_id: string;
  n_points: number;
  ws_url: string;
  [key: string]: unknown;
}

/**
 * Authenticated POST to start a SHERLOC fit job for one or more domains
 * on a scan. Returns the job descriptor that the caller uses to open the
 * fit-progress WebSocket. Replaces a raw `fetch()` POST that skipped
 * Bearer auth (Codex PR9 R3 F5).
 */
export async function startMapFit(
  scanId: string,
  domains: string[],
): Promise<MapFitResponse> {
  await ensureAuthenticated();
  return fetchJson<MapFitResponse>(`/map/fit`, {
    method: 'POST',
    body: JSON.stringify({ scan_id: scanId, domains }),
  });
}

export interface JobStatus {
  job_id: string;
  status: string;
  [key: string]: unknown;
}

/**
 * Authenticated fetch of job status. Replaces the raw `fetch()` in
 * `lib/websocket.ts` which produced 401s under Auth0 Bearer-token mode
 * (Codex PR9 R3 F5).
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
  await ensureAuthenticated();
  return fetchJson<JobStatus>(`/jobs/${encodeURIComponent(jobId)}`);
}

/**
 * Authenticated POST to fetch the average spectrum for a subset of points.
 * Replaces the raw `fetch()` in `MapSpectrumPanel.svelte` (Codex PR9 R3 F5).
 *
 * Caller passes the request body shape as-is; backend Pydantic model
 * validates. Loose typing here lets the caller migrate without forcing
 * a TS-level body schema change.
 */
export async function postSpectraSubset<T = unknown>(
  scanId: string,
  body: unknown,
): Promise<T> {
  await ensureAuthenticated();
  return fetchJson<T>(`/spectra/${encodeURIComponent(scanId)}/subset`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}
