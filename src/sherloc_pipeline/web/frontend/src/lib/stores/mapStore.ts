// ============================================================
// Map Mode Svelte stores
// ============================================================

import { writable } from 'svelte/store';
import type {
  PointSet,
  ScalarLayer,
  DisplayMode,
  GeometryMode,
  Transform,
  WSPointFitted,
  ClassificationProfile,
} from '../types/map';

export const mapPointSet = writable<PointSet | null>(null);
export const mapLayers = writable<ScalarLayer[]>([]);
export const mapDisplayMode = writable<DisplayMode>({ type: 'all_domains' });
export const mapGeometryMode = writable<GeometryMode>('ring');
export const mapOverlayOpacity = writable<number>(0.7);
export const mapShowPointPositions = writable<boolean>(true);

export interface MapFitJobState {
  jobId: string;
  status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled';
  fitted: number;
  total: number;
  etaSeconds: number;
}

export const mapFitJob = writable<MapFitJobState | null>(null);
export const mapLogEntries = writable<string[]>([]);

/** Active classification profile for dynamic re-classification. */
export const mapClassificationProfile = writable<ClassificationProfile | null>(null);

/** Saved canvas transform for restoring zoom/pan on remount. */
export const mapSavedTransform = writable<Transform | null>(null);

/** Cached raw WS fitting results (immutable, for re-classification). */
export const mapFittedResultsCache = writable<WSPointFitted[]>([]);

/** Active profile stored for persistence across remounts. */
export const mapActiveProfile = writable<ClassificationProfile | null>(null);

/** The scan ID these stores belong to (used to detect scan changes). */
export const mapScanId = writable<string | null>(null);

export function resetMapState(): void {
  mapPointSet.set(null);
  mapLayers.set([]);
  mapDisplayMode.set({ type: 'all_domains' });
  mapGeometryMode.set('ring');
  mapOverlayOpacity.set(0.7);
  mapShowPointPositions.set(true);
  mapFitJob.set(null);
  mapLogEntries.set([]);
  mapClassificationProfile.set(null);
  mapSavedTransform.set(null);
  mapFittedResultsCache.set([]);
  mapActiveProfile.set(null);
  mapScanId.set(null);
}
