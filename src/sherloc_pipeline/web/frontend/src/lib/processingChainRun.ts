import type { ProcessingStage } from './types';

// Step order and input routing of the Workbench processing chain
// (despike -> background -> baseline -> fit), kept free of component state.

export interface ChainState {
  rawIntensity: number[];
  // Client-side modz result (only ever set when despikeMethod === 'modz').
  afterDespikeIntensity: number[] | null;
  afterBgIntensity: number[] | null;
  afterBaselineIntensity: number[] | null;
  modzActive: boolean;
  bgEnabled: boolean;
  baselineEnabled: boolean;
}

// Compute what each step receives as input. The despike step only diverts
// the stream when modz is the active method AND it has produced a result.
// Each takes only the fields it reads, so a Svelte reactive call names exactly
// its own dependencies.
export function backgroundInput(s: Pick<ChainState, 'rawIntensity' | 'afterDespikeIntensity' | 'modzActive'>): number[] {
  return s.modzActive && s.afterDespikeIntensity ? s.afterDespikeIntensity : s.rawIntensity;
}

export function baselineInput(
  s: Pick<ChainState, 'rawIntensity' | 'afterDespikeIntensity' | 'afterBgIntensity' | 'modzActive' | 'bgEnabled'>,
): number[] {
  return s.bgEnabled && s.afterBgIntensity
    ? s.afterBgIntensity
    : s.modzActive && s.afterDespikeIntensity
      ? s.afterDespikeIntensity
      : s.rawIntensity;
}

export function fitInput(s: ChainState): number[] {
  return s.baselineEnabled && s.afterBaselineIntensity
    ? s.afterBaselineIntensity
    : s.bgEnabled && s.afterBgIntensity
      ? s.afterBgIntensity
      : s.modzActive && s.afterDespikeIntensity
        ? s.afterDespikeIntensity
        : s.rawIntensity;
}

export function currentStage(s: ChainState): ProcessingStage {
  // The "latest completed" stage
  if (s.afterBaselineIntensity && s.baselineEnabled) return 'baseline_corrected';
  if (s.afterBgIntensity && s.bgEnabled) return 'bg_subtracted';
  if (s.afterDespikeIntensity && s.modzActive) return 'despiked';
  return 'raw';
}

export function chainRecords(
  s: ChainState,
  despikeRecord: Record<string, unknown> | null,
  backgroundRecord: Record<string, unknown> | null,
  baselineRecord: Record<string, unknown> | null,
  fitRecord: Record<string, unknown> | null,
): Record<string, unknown>[] {
  return [
    s.modzActive && s.afterDespikeIntensity ? despikeRecord : null,
    s.bgEnabled && s.afterBgIntensity ? backgroundRecord : null,
    s.baselineEnabled && s.afterBaselineIntensity ? baselineRecord : null,
    fitRecord,
  ].filter((record): record is Record<string, unknown> => record != null);
}

export function despikeStepRecord(nSpikes: number, params: Record<string, unknown>): Record<string, unknown> {
  return {
    step: 'despike',
    method: 'modz',
    n_spikes: nSpikes,
    ...params,
  };
}

export function backgroundStepRecord(bgType: string, scaleUsed: number): Record<string, unknown> {
  return {
    step: 'bg_subtract',
    bg_type: bgType,
    scale_used: scaleUsed,
  };
}

export function baselineStepRecord(params: Record<string, unknown>): Record<string, unknown> {
  return {
    step: 'baseline',
    ...params,
  };
}
