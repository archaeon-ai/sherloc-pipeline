// ============================================================
// TypeScript interfaces matching WEB_API_SPEC.md v1.0.0
// ============================================================

// --- Error ---

export interface ApiErrorResponse {
  schema_version: string;
  error: string;
  detail?: string;
}

// --- Scans ---

export interface ScanListItem {
  id: string;
  sol_number: number;
  target: string | null;
  scan_name: string;
  scan_id: string;
  n_points: number;
  scan_class: 'primary' | 'sub_scan' | 'composite';
  scan_type: 'detail' | 'line' | 'hdr' | 'survey' | null;
  target_type: 'mars_target' | 'cal_target' | 'engineering';
  processing_status: 'completed' | 'failed' | null;
  processed_at: string | null;
  processing_pipeline_version: string | null;
}

export interface ScanListResponse {
  schema_version: string;
  scans: ScanListItem[];
  total: number;
  offset: number;
  limit: number;
  // Populated only on the unfiltered empty-DB response per spec §12.2;
  // SPA renders the onboarding panel when this is non-null.
  message?: string | null;
}

export interface ScanDetail {
  id: string;
  sol_number: number;
  target: string | null;
  scan_name: string;
  scan_id: string;
  n_points: number;
  n_channels: number;
  shots_per_point: number;
  laser_wavelength_nm: number;
  scan_class: 'primary' | 'sub_scan' | 'composite';
  scan_type: 'detail' | 'line' | 'hdr' | 'survey' | null;
  target_type: 'mars_target' | 'cal_target' | 'engineering';
  data_source: string;
  site_drive: string | null;
  sequence_id: string | null;
  parent_scan_id: string | null;
  source_scan_ids: string[] | null;
  processing_status: 'completed' | 'failed' | null;
  processed_at: string | null;
  processing_pipeline_version: string | null;
  processing_config_hash: string | null;
  processing_error: string | null;
  sclk_start: number | null;
  sclk_stop: number | null;
  created_at: string;
  updated_at: string;
  // True iff R2 has a sol_NNNN_colorized/ sibling for this scan's ACI.
  // Drives Workbench AciViewer's "Colorized" button enabled state.
  // Optional + defaults to false on the SPA side so the older backends
  // (no `colorized_aci_available` field in the JSON) keep their pre-fix
  // behavior of treating the variant as absent — i.e. button disabled
  // rather than a TypeScript runtime crash.
  colorized_aci_available?: boolean;
}

export interface ScanDetailResponse {
  schema_version: string;
  scan: ScanDetail;
}

// --- Points ---

export interface ScanPoint {
  id: string;
  point_index: number;
  x_pixel: number | null;
  y_pixel: number | null;
  x_aci_pixel: number | null;
  y_aci_pixel: number | null;
  azimuth_dn: number | null;
  elevation_dn: number | null;
  azimuth_error: number | null;
  elevation_error: number | null;
  photodiode_mean: number | null;
  photodiode_std: number | null;
  coordinate_frame: string | null;
}

export interface ScanPointsResponse {
  schema_version: string;
  scan_id: string;
  points: ScanPoint[];
  n_points: number;
}

// --- Spectra ---

export interface SpectrumProvenance {
  calibration_version: string;
  wavenumber_unit: string;
  intensity_unit: string;
  averaging_method?: string;
  wavelength_filter?: { min_nm: number; max_nm: number };
}

export interface SpectrumResponse {
  schema_version: string;
  scan_id: string;
  region: string;
  n_points_averaged?: number;
  point_index?: number;
  point_indices?: number[];
  effective_trim_pct_per_tail?: number;
  m_trimmed_per_tail?: number;
  baseline_corrected?: boolean;
  laser_normalized?: boolean;
  spectrum_type?: string;
  wavenumber: number[];
  wavelength?: number[];
  intensity: number[];
  n_channels: number;
  photodiode_mean?: number;
  provenance: SpectrumProvenance;
}

// --- Processing Params ---

export interface BaselineParams {
  method?: string;
  lam?: number;
  max_iter?: number;
}

export interface FitParams {
  domain?: 'minerals' | 'organics' | 'hydration' | 'fluorescence';
  wavenumber_range?: [number, number];
  max_peaks?: number;
  min_snr?: number;
  fwhm_bounds?: [number, number];
  model_selection?: 'f-test' | 'aicc';
}

export interface ProcessingParams {
  baseline?: BaselineParams;
  fitting?: FitParams;
}

// --- Baseline ---

export interface BaselineRequest {
  wavenumber: number[];
  intensity: number[];
  params?: BaselineParams;
}

export interface BaselineResponse {
  schema_version: string;
  raw: number[];
  baseline: number[];
  corrected: number[];
  wavenumber: number[];
  params_used: BaselineParams;
}

// --- Fit ---

export interface FitRequest {
  wavenumber: number[];
  intensity: number[];
  params?: ProcessingParams;
  // Forwarded to the backend quality classifier. "cal_target" /
  // "engineering" cap each peak's quality at "review" regardless of fit
  // numerics (no Mars-target ground truth). Workbench passes the active
  // scan's target_type from the ScanDetail response.
  target_type?: string;
}

export interface Peak {
  center_cm1: number | null;
  center_uncertainty: number | null;
  amplitude: number;
  amplitude_uncertainty: number | null;
  fwhm_cm1: number | null;
  fwhm_uncertainty: number | null;
  area: number | null;
  snr: number | null;
  fit_quality: number | null;
  mineral_assignment: string | null;
  assignment_confidence: number | null;
  fit_modality: string;
  sharpness_ratio: number | null;
  pass_sharpness: boolean | null;
  // Tri-state Workbench Quality badge: "pass" | "review" | "fail".
  // Null only on responses from legacy backends that predate v4.1.12; the
  // current backend always populates it via services.quality.
  quality: 'pass' | 'review' | 'fail' | null;
}

export interface FitProvenance {
  params_used: ProcessingParams;
  calibration_version: string;
  config_hash: string;
  wavenumber_unit: string;
  intensity_unit: string;
}

export interface FitResponse {
  schema_version: string;
  peaks: Peak[];
  n_peaks: number;
  r_squared: number;
  model_selection_method: string;
  residual: number[];
  baseline: number[];
  corrected: number[];
  wavenumber: number[];
  provenance: FitProvenance;
}

// --- Config ---

export interface MineralRule {
  label: string;
  lo: number;
  hi: number;
}

export interface FeaturesConfig {
  // Runtime feature flags (see backend `FeaturesConfig`). Default each
  // to `true` so a missing field on legacy backends (pre-issue-#21)
  // doesn't accidentally hide nav.
  pds_browser: boolean;
}

export interface ConfigResponse {
  schema_version: string;
  config_hash: string;
  fitting: {
    r1_fit_range: [number, number];
    max_peaks: number;
    min_snr: number;
    fwhm_bounds_minerals: [number, number];
    fwhm_bounds_organics: [number, number];
    fwhm_bounds_hydration: [number, number];
    model_selection: string;
    ftest_alpha: number;
    hydration_fit_range: [number, number];
    organics_fit_range: [number, number];
    posthoc_filters: {
      r2_min: number;
      sharpness_max: number;
      organics_fwhm: Record<string, number>;
      hydration_center_range: [number, number];
    };
    mineral_rules: MineralRule[];
    parallel_workers: number;
  };
  fluorescence_fitting: {
    fit_range_nm: [number, number];
    fwhm_range_nm: [number, number];
    max_peaks: number;
    snr_threshold: number;
    strategy: string;
    fluorescence_rules: MineralRule[];
  };
  preprocessing: {
    baseline_method: string;
    lam: number;
    iters: number;
    trim_mean_baseline_pct: number;
  };
  calibration: {
    version: string;
    laser_wavelength_nm: number;
    cutoff_channel: number;
    n_channels: number;
    raman_coefficients: number[];
    fluorescence_coefficients: number[];
  };
  auth: AuthConfig;
  features?: FeaturesConfig;
}

// --- Auth (§13.4) ---

export type AuthMode = 'auth0' | 'cf-access' | 'dev';

export interface AuthConfig {
  auth_mode: AuthMode;
  auth0_domain: string | null;
  auth0_client_id: string | null;
  auth0_audience: string | null;
  role_claim_uri: string | null;
}

// --- Health ---

export interface HealthCheck {
  status: 'ok' | 'error';
  [key: string]: unknown;
}

export interface HealthResponse {
  schema_version: string;
  status: 'ok' | 'degraded' | 'error';
  timestamp: string;
  pipeline_version: string;
  checks: {
    database: HealthCheck;
    config: HealthCheck;
    pds_reachable: HealthCheck;
    job_queue: HealthCheck;
    unprocessed_scans: HealthCheck;
  };
}

// --- PDS ---

export interface PdsSol {
  sol: number;
  n_scans: number;
  data_volume_mb: number | null;
  pds_url: string;
}

export interface PdsAvailableResponse {
  schema_version: string;
  sols: PdsSol[];
  total: number;
  already_ingested: number[];
}

export interface PdsDownloadRequest {
  sol: number;
  force_reingest?: boolean;
}

export interface PdsDownloadResponse {
  schema_version: string;
  job_id: string;
  status: string;
  queue_position: number;
  sol: number;
  submitter_token: string;
  created_at: string;
}

// --- WebSocket Messages (spec protocol v2) ---

export interface WsProgressMessage {
  type: 'progress';
  phase: 'downloading' | 'ingesting' | 'aci_download' | string;
  progress: number;  // 0.0 to 1.0
  message: string;
}

export interface WsCompleteMessage {
  type: 'complete';
  result: {
    n_scans: number;
    n_spectra: number;
    n_aci: number;
    n_downloaded: number;
    n_skipped: number;
    warnings: string[];
  };
}

export interface WsErrorMessage {
  type: 'error';
  error: string;
  partial_result?: { n_scans: number; n_spectra: number };
}

export interface WsHeartbeatMessage {
  type: 'heartbeat';
}

export interface WsCancelledMessage {
  type: 'cancelled';
  job_id: string;
}

export type WsMessage =
  | WsProgressMessage
  | WsCompleteMessage
  | WsErrorMessage
  | WsHeartbeatMessage
  | WsCancelledMessage;

// --- Filter params (client-side) ---

export interface ScanFilterParams {
  sol?: number;
  target?: string;
  scan_class?: string;
  scan_type?: string;
  processing_status?: string;
  offset?: number;
  limit?: number;
}

export interface SpectrumParams {
  region?: string;
  baseline_corrected?: boolean;
  spectrum_type?: string;
  averaging_method?: 'mean' | 'trim_mean' | 'median';
  trim_pct?: number;
}

export interface SubsetRequest {
  point_indices: number[];
  region?: string;
  averaging_method?: 'mean' | 'trim_mean' | 'median';
  trim_pct?: number;
}

// --- Despike ---

export interface DespikeParams {
  window_size?: number;
  zscore_threshold?: number;
  max_iterations?: number;
  sulfate_guard?: boolean;
}

export interface DespikeRequest {
  wavenumber: number[];
  intensity: number[];
  params?: DespikeParams;
}

export interface DespikeResponse {
  schema_version: string;
  despiked: number[];
  spike_mask: boolean[];
  n_spikes: number;
  params_used: DespikeParams;
}

// --- Background Subtraction ---

export interface BackgroundRequest {
  wavenumber: number[];
  intensity: number[];
  bg_type: 'as' | 'fs';
  scale: 'auto' | number;
  scan_ppp: number;
}

export interface BackgroundResponse {
  schema_version: string;
  subtracted: number[];
  background_scaled: number[];
  scale_used: number;
  bg_type: 'as' | 'fs';
}

// --- Access Mode ---

export interface AccessModeResponse {
  schema_version: string;
  access_mode: 'internal' | 'public';
}

// --- Processing Chain ---

export type ProcessingStage = 'raw' | 'despiked' | 'bg_subtracted' | 'baseline_corrected' | 'raman_fitted';

export interface ProcessingArtifacts {
  spikeMask?: boolean[];
  baseline?: number[];
  background?: number[];
  backgroundScaled?: number[];
  fitCurve?: number[];
  peaks?: Peak[];
  residual?: number[];
  rSquared?: number;
  modelSelectionMethod?: string;
  fitRange?: [number, number];
}

export interface ProcessingSnapshot {
  stage: ProcessingStage;
  raman: { wavenumber: number[]; intensity: number[] };
  params: Record<string, unknown>;
  artifacts?: ProcessingArtifacts;
}

export interface PointSelectionState {
  mode: 'average' | 'subset' | 'point';
  indices?: number[];
  pointIdx?: number;
}
