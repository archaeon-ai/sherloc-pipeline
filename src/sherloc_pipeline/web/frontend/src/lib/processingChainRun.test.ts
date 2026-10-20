import { describe, expect, it } from 'vitest';
import {
  backgroundInput,
  backgroundStepRecord,
  baselineInput,
  baselineStepRecord,
  chainRecords,
  currentStage,
  despikeStepRecord,
  fitInput,
} from './processingChainRun';
import type { ChainState } from './processingChainRun';

const raw = [1, 2, 3];
const despiked = [4, 5, 6];
const subtracted = [7, 8, 9];
const corrected = [10, 11, 12];

function state(overrides: Partial<ChainState> = {}): ChainState {
  return {
    rawIntensity: raw, afterDespikeIntensity: null, afterBgIntensity: null, afterBaselineIntensity: null,
    modzActive: false, bgEnabled: false, baselineEnabled: false, ...overrides,
  };
}

describe('processingChainRun — step input routing', () => {
  it('feeds the raw input everywhere when no step has produced a result', () => {
    const s = state({ modzActive: true, bgEnabled: true, baselineEnabled: true });
    expect(backgroundInput(s)).toBe(raw);
    expect(baselineInput(s)).toBe(raw);
    expect(fitInput(s)).toBe(raw);
    expect(currentStage(s)).toBe('raw');
  });

  it('diverts through despike only when modz is active', () => {
    expect(backgroundInput(state({ afterDespikeIntensity: despiked }))).toBe(raw);
    const s = state({ modzActive: true, afterDespikeIntensity: despiked });
    expect(backgroundInput(s)).toBe(despiked);
    expect(baselineInput(s)).toBe(despiked);
    expect(fitInput(s)).toBe(despiked);
    expect(currentStage(s)).toBe('despiked');
  });

  it('uses each enabled downstream result in order', () => {
    const all = { modzActive: true, afterDespikeIntensity: despiked, afterBgIntensity: subtracted,
      afterBaselineIntensity: corrected };
    const s = state({ ...all, bgEnabled: true, baselineEnabled: true });
    expect(baselineInput(s)).toBe(subtracted);
    expect(fitInput(s)).toBe(corrected);
    expect(currentStage(s)).toBe('baseline_corrected');
    // A disabled step is skipped even when its last result is still held.
    const off = state({ ...all, bgEnabled: false, baselineEnabled: false });
    expect(baselineInput(off)).toBe(despiked);
    expect(fitInput(off)).toBe(despiked);
    expect(currentStage(off)).toBe('despiked');
    expect(currentStage(state({ ...all, bgEnabled: true }))).toBe('bg_subtracted');
  });
});

describe('processingChainRun — chain records', () => {
  it('keeps the existing record shapes', () => {
    expect(despikeStepRecord(2, { window_size: 5 })).toEqual({ step: 'despike', method: 'modz', n_spikes: 2, window_size: 5 });
    expect(backgroundStepRecord('as', 1.5)).toEqual({ step: 'bg_subtract', bg_type: 'as', scale_used: 1.5 });
    expect(baselineStepRecord({ method: 'aspls', lam: 100 })).toEqual({ step: 'baseline', method: 'aspls', lam: 100 });
  });

  it('lists only the records of steps that are active with a result, then the fit record', () => {
    const d = { step: 'despike' }, b = { step: 'bg_subtract' }, l = { step: 'baseline' }, f = { step: 'raman_fit' };
    const all = state({ modzActive: true, afterDespikeIntensity: despiked, bgEnabled: true, afterBgIntensity: subtracted,
      baselineEnabled: true, afterBaselineIntensity: corrected });
    expect(chainRecords(all, d, b, l, f)).toEqual([d, b, l, f]);
    expect(chainRecords(state({ bgEnabled: true, afterBgIntensity: subtracted }), d, b, l, null)).toEqual([b]);
    expect(chainRecords(all, null, b, null, null)).toEqual([b]);
  });
});
