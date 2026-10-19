import { beforeEach, describe, expect, it } from 'vitest';
import { get } from 'svelte/store';
import { accessMode, appConfig, clearSessionState } from './stores';
import { clearEvidence, setEvidenceContext, evidenceTrace, pinTrace, pinnedTraces,
  displayTrace, comparisonDisplays, evidenceCsv, evidenceExport, toggleTrace, removeTrace } from './evidenceStore';
import type { ConfigResponse, ProcessingSnapshot } from './types';

function snapshot(index = 0): ProcessingSnapshot {
  return { stage: 'raw', raman: { wavenumber: [800, 900, 1000], intensity: [-2, 2, 4 + index] }, params: {}, processingChain: [],
    trace: { wavenumber: [800, 900, 1000], wavelength: [280, 300, 320], channel_indices: [0, 1, 2], intensity: [-2, 2, 4 + index],
      source: { scan_id: `synthetic-${index}`, scan_name: `Synthetic ${index}`, target: 'Invented', sol: 9999 }, selection: { mode: 'point', indices: [index] },
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'invented', source_digest: null } } };
}

beforeEach(() => { clearEvidence(); setEvidenceContext('session:team:invented'); });

describe('pipeline comparison snapshots', () => {
  it('owns immutable arrays, provenance, annotations and processing across remounts', () => {
    const input = snapshot();
    const pin = evidenceTrace(input, { spike_mask: [false, true, false] })!;
    expect(pinTrace(pin, 'session:team:invented')).toBe(true);
    input.raman.intensity[0] = 500;
    input.trace!.source.target = 'Changed';
    setEvidenceContext('session:team:invented');
    const stored = get(pinnedTraces)[0];
    expect(stored.snapshot.raman.intensity).toEqual([-2, 2, 4]);
    expect(stored.snapshot.trace.source.target).toBe('Invented');
    expect(stored.annotations.spike_mask).toEqual([false, true, false]);
    expect(Object.isFrozen(stored.snapshot.trace.provenance)).toBe(true);
  });

  it('caps the current selection plus pins at four and rejects duplicate pins', () => {
    for (let i = 0; i < 3; i++) expect(pinTrace(evidenceTrace(snapshot(i))!, 'session:team:invented')).toBe(true);
    expect(pinTrace(evidenceTrace(snapshot(3))!, 'session:team:invented')).toBe(false);
    expect(pinTrace(evidenceTrace(snapshot(0))!, 'session:team:invented')).toBe(false);
    expect(comparisonDisplays(evidenceTrace(snapshot(3)), get(pinnedTraces), 'raman_shift', false)).toHaveLength(4);
  });

  it('clears on target, tier and sign-out and denies a stale pin context', () => {
    const pin = evidenceTrace(snapshot())!;
    pinTrace(pin, 'session:team:invented');
    setEvidenceContext('session:team:other');
    expect(get(pinnedTraces)).toEqual([]);
    expect(pinTrace(pin, 'session:team:invented')).toBe(false);
    setEvidenceContext('session:team:invented'); pinTrace(pin, 'session:team:invented');
    accessMode.set('public'); expect(get(pinnedTraces)).toEqual([]);
    setEvidenceContext('session:team:invented'); pinTrace(pin, 'session:team:invented');
    appConfig.set({ tier: 'team' } as ConfigResponse); expect(get(pinnedTraces)).toEqual([]);
    setEvidenceContext('session:team:invented'); pinTrace(pin, 'session:team:invented');
    clearSessionState(); expect(get(pinnedTraces)).toEqual([]);
    expect(pinTrace(pin, 'session:team:invented')).toBe(false);
  });

  it('keeps two scans native, supports visibility/removal and avoids a duplicate current trace', () => {
    const a = evidenceTrace(snapshot())!;
    const other = snapshot(1); other.raman.wavenumber = [801, 950, 999];
    const b = evidenceTrace(other)!;
    pinTrace(a, 'session:team:invented'); pinTrace(b, 'session:team:invented');
    const displays = comparisonDisplays(a, get(pinnedTraces), 'raman_shift', true);
    expect(displays.map(d => d.x)).toEqual([[800, 900, 1000], [801, 950, 999]]);
    expect(displays[1].offset).toBe(1.25);
    toggleTrace(b.id); expect(comparisonDisplays(a, get(pinnedTraces), 'raman_shift', false)).toHaveLength(1);
    removeTrace(b.id); expect(get(pinnedTraces)).toHaveLength(1);
  });

  it('retains negative values and marks nonpositive or missing fixed-window divisors unscalable', () => {
    const a = evidenceTrace(snapshot())!;
    expect(displayTrace(a, 'raman_shift')!.displayed).toEqual([-0.5, 0.5, 1]);
    for (const y of [[0, 0, 0], [-4, -3, -2]]) {
      const input = snapshot(); input.raman.intensity = y;
      const d = displayTrace(evidenceTrace(input)!, 'raman_shift')!;
      expect(d.scalable).toBe(false); expect(d.displayed).toEqual(y);
    }
    const input = snapshot(); input.raman.wavenumber = [10, 20, 30];
    expect(displayTrace(evidenceTrace(input)!, 'raman_shift')!.divisor).toBeNull();
  });

  it('supports emission-only traces, masks channel/axis gaps and retains repeated coordinates', () => {
    const input = snapshot(); input.raman.wavenumber = []; input.trace!.wavenumber = [];
    input.trace!.wavelength = [280, 280, 330]; input.trace!.channel_indices = [0, 1, 4];
    const trace = evidenceTrace(input)!;
    expect(displayTrace(trace, 'raman_shift')).toBeNull();
    expect(displayTrace(trace, 'emission_wavelength')!.x).toEqual([280, 280, 330]);
    expect(displayTrace(trace, 'emission_wavelength')!.breaks).toEqual([false, false, true]);
  });

  it('exports per-trace original and displayed values, units, processing and missing provenance explicitly', () => {
    const input = snapshot(1); input.processingChain = [{ operation: 'synthetic-offset', value: 2 }];
    const displays = comparisonDisplays(evidenceTrace(snapshot()), [evidenceTrace(input)!], 'raman_shift', true);
    const output = evidenceExport(displays);
    expect(output.traces[1].snapshot.processingChain).toEqual(input.processingChain);
    expect(output.traces[1].snapshot.trace.provenance.source_digest).toBeNull();
    const csv = evidenceCsv(displays);
    expect(csv).toContain('1,raman_shift,800,-2,0.85,0');
    expect(JSON.parse(csv.split('\n')[0].slice('# provenance: '.length))).toEqual(output);
  });

  it('rejects mismatched arrays and incompatible units without conversion', () => {
    const input = snapshot(); input.raman.wavenumber = [800]; input.trace!.wavelength = null;
    expect(evidenceTrace(input)).toBeNull();
    const other = snapshot(); other.trace!.provenance.wavenumber_unit = 'unknown';
    expect(displayTrace(evidenceTrace(other)!, 'raman_shift')).toBeNull();
  });
});

it('keeps pin colors distinct after removing a non-last trace and pinning another', () => {
  for (let i = 0; i < 3; i++) pinTrace(evidenceTrace(snapshot(i))!, 'session:team:invented');
  removeTrace(get(pinnedTraces)[0].id);
  pinTrace(evidenceTrace(snapshot(3))!, 'session:team:invented');
  expect(new Set(get(pinnedTraces).map(trace => trace.color)).size).toBe(3);
});
