// ============================================================
// Map Mode TypeScript interfaces
// Matches MAP_MODE_SPEC.md wire format
// ============================================================

// --- Point Geometry ---

export interface PointSet {
  scan_id: string;
  source: 'sherloc' | 'pixl' | string;
  coordinate_source: 'aci_pixel' | 'scanner_workspace_transformed' | string;
  points: DisplayPoint[];
  voronoi: VoronoiGeometry | null;
}

export interface DisplayPoint {
  index: number;
  x: number; // ACI pixel
  y: number; // ACI pixel
}

export interface VoronoiGeometry {
  vertices: [number, number][];
  regions: number[][];
  boundary: [number, number][];
  edge_mask: boolean[];
}

// --- Scalar Layers ---

export interface ScalarLayer {
  id: string;
  point_set_id: string;
  label: string;
  domain: string;
  class_id: string | null;
  value_type: string;
  values: LayerValue[];
  colormap: ColormapConfig;
  opacity: number;
  visible: boolean;
}

export interface LayerValue {
  value: number | null;
  status: 'measured' | 'below_threshold' | 'missing' | 'masked';
  uncertainty: number | null;
  metadata: Record<string, unknown>;
}

export interface ColormapConfig {
  type: 'sequential' | 'mono_channel';
  name: string; // 'viridis', 'red', 'green', 'blue'
  range: [number, number];
}

// --- Display Modes ---

export type GeometryMode = 'voronoi' | 'ring' | 'combined';

export type DisplayMode =
  | { type: 'all_domains' }
  | { type: 'domain'; domain: string }
  | { type: 'class'; domain: string; class_id: string }
  | { type: 'rgb_mix'; channels: RGBChannelAssignment };

export interface RGBChannelAssignment {
  red: LayerRef | null;
  green: LayerRef | null;
  blue: LayerRef | null;
}

export interface LayerRef {
  domain: string;
  class_id: string | null;
  value_type: string;
}

// --- WebSocket Messages ---

export interface WSJobStarted {
  type: 'job_started';
  seq: number;
  job_id: string;
  n_points: number;
  domains: string[];
  voronoi: VoronoiGeometry | null;
}

export interface WSPointFitted {
  type: 'point_fitted';
  seq: number;
  point_index: number;
  x: number;
  y: number;
  results: Record<string, { status: string; peaks: WSPeakResult[] }>;
}

export interface WSPeakResult {
  center_cm1?: number;
  center_nm?: number;
  snr: number;
  assignment: string;
  fwhm_cm1?: number;
  fwhm_nm?: number;
}

export interface WSProgress {
  type: 'progress';
  seq: number;
  fitted: number;
  total: number;
  pct: number;
  elapsed_s: number;
  eta_s: number;
}

export interface WSLog {
  type: 'log';
  seq: number;
  point_index: number;
  message: string;
}

export interface WSJobComplete {
  type: 'job_complete';
  seq: number;
  summary: {
    total_points: number;
    detections: Record<string, number>;
    elapsed_s: number;
  };
}

export interface WSJobFailed {
  type: 'job_failed';
  seq: number;
  error: string;
  partial_results: number;
}

export type WSServerMessage =
  | WSJobStarted
  | WSPointFitted
  | WSProgress
  | WSLog
  | WSJobComplete
  | WSJobFailed
  | { type: 'job_queued'; seq: number }
  | { type: 'job_cancelled'; seq: number }
  | { type: 'point_fitted_batch'; seq: number; points: WSPointFitted[] }
  | { type: 'ping' };

// --- Classification ---

export interface ClassificationRule {
  domain: string;
  class_id: string;
  label: string;
  center: number;        // cm-1 for Raman, nm for fluorescence
  range: number;         // half-width for peak matching
  snr_threshold: number; // default: 3.0
  disabled: boolean;
}

export interface ClassificationProfile {
  name: string;
  rules: ClassificationRule[];
}

// --- Transform State ---

export interface Transform {
  x: number;
  y: number;
  scale: number;
}
