import { describe, it, expect } from 'vitest';
import { get } from 'svelte/store';
import { processingState, undoStack, pushUndo, undo, resetProcessing } from './processingStore';
import type { ProcessingSnapshot } from './types';

const raw: ProcessingSnapshot = {
  stage: 'raw', raman: { wavenumber: [11, 22], intensity: [2, 3] }, params: {}, processingChain: [],
  trace: {
    source: { scan_id: 'synthetic' }, selection: { mode: 'point', indices: [2] },
    provenance: { data_origin: 'pds4', calibration_version: null, wavenumber_unit: 'cm-1', intensity_unit: null },
    wavenumber: [11, 22], wavelength: [251, 252], channel_indices: [3, 4], intensity: [2, 3],
  },
};

describe('committed export trace history', () => {
  it('restores source and active processing together through undo and reset', () => {
    processingState.set(raw);
    undoStack.set([]);
    pushUndo(raw);
    processingState.set({ ...raw, stage: 'despiked', raman: { wavenumber: [11, 22], intensity: [1, 2] },
      processingChain: [{ step: 'despike', method: 'synthetic' }], params: { method: 'synthetic' } });
    expect(undo()).toEqual(raw);
    expect(get(processingState)?.processingChain).toEqual([]);
    processingState.set({ ...raw, stage: 'baseline_corrected', raman: { wavenumber: [11, 22], intensity: [0, 1] },
      processingChain: [{ step: 'baseline', method: 'synthetic' }] });
    resetProcessing();
    expect(get(processingState)?.raman.intensity).toEqual([2, 3]);
    expect(get(processingState)?.trace).toEqual(raw.trace);
    expect(get(processingState)?.processingChain).toEqual([]);
  });
});
