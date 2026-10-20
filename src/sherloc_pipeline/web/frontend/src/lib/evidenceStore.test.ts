import { beforeEach, describe, expect, it } from 'vitest';
import { get } from 'svelte/store';
import { accessMode, appConfig, clearSessionState } from './stores';
import { clearEvidence, setEvidenceContext, evidenceTrace, pinTrace, pinnedTraces,
  displayTrace, comparisonDisplays, evidenceCsv, evidenceExport, toggleTrace, removeTrace,
  cacheSharedOutcome, cachedSharedOutcome, derivedTrace, dropFailedSharedOutcomes, pinShotsPerPoint, sharedOutcomeEpoch } from './evidenceStore';
import type { SharedEvidence } from './evidenceStore';
import { sharedChain } from './sharedChain';
import type { SharedOutcome } from './sharedChain';
import type { ConfigResponse, ProcessingSnapshot } from './types';

function snapshot(index = 0): ProcessingSnapshot {
  return { stage: 'raw', raman: { wavenumber: [800, 900, 1000], intensity: [-2, 2, 4 + index] }, params: {}, processingChain: [],
    trace: { wavenumber: [800, 900, 1000], wavelength: [280, 300, 320], channel_indices: [0, 1, 2], intensity: [-2, 2, 4 + index],
      source: { scan_id: `synthetic-${index}`, scan_name: `Synthetic ${index}`, target: 'Invented', sol: 9999 }, selection: { mode: 'point', indices: [index] },
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'invented', source_digest: null } } };
}

beforeEach(() => { appConfig.set({ tier: 'team', capabilities: { evidence_desk: true } } as ConfigResponse); clearEvidence(); setEvidenceContext('session:team:invented'); });

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
    appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as ConfigResponse); expect(get(pinnedTraces)).toEqual([]);
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


it.each(['R1', 'R2', 'R3'])('keeps both-axis %s traces in their committed domain for display and export', region => {
  const input = snapshot(); input.trace!.region = region;
  const trace = evidenceTrace(input)!;
  input.trace!.region = region === 'R1' ? 'R2' : 'R1';
  const domain = region === 'R1' ? 'raman_shift' : 'emission_wavelength';
  const opposite = region === 'R1' ? 'emission_wavelength' : 'raman_shift';
  expect(trace.nativeDomain).toBe(domain);
  expect(displayTrace(trace, opposite)).toBeNull();
  const excluded = comparisonDisplays(trace, [trace], opposite, false);
  expect(excluded).toEqual([]);
  expect(evidenceExport(excluded).traces).toEqual([]);
  expect(evidenceCsv(excluded)).not.toContain(`0,${opposite},`);
  const included = comparisonDisplays(trace, [], domain, false);
  expect(evidenceExport(included).traces[0].domain).toBe(domain);
  expect(evidenceCsv(included)).toContain(`0,${domain},`);
});

it('uses axes only for unknown regions and rejects unsupported declared regions', () => {
  const input = snapshot();
  for (const region of [undefined, '', 'unknown']) {
    input.trace!.region = region;
    const trace = evidenceTrace(input)!;
    expect(displayTrace(trace, 'raman_shift')).not.toBeNull();
    expect(displayTrace(trace, 'emission_wavelength')).not.toBeNull();
  }
  input.trace!.region = 'synthetic-unsupported';
  const trace = evidenceTrace(input)!;
  expect(displayTrace(trace, 'raman_shift')).toBeNull();
  expect(displayTrace(trace, 'emission_wavelength')).toBeNull();
});


describe('shared processing chain', () => {
  async function shared() {
    const primary = evidenceTrace(snapshot())!;
    const chain = await sharedChain(primary, [{ step: 'baseline', params: { method: 'aspls', lam: 1e6, max_iter: 10 } }]);
    const pin = evidenceTrace(snapshot(1))!;
    const outcome: SharedOutcome = { derived_from: pin.id, label: pin.label, refusal: null, stage: 'baseline_corrected', scale_used: null,
      intensity: [1, 2, 3], steps: [{ step: 'baseline', status: 'applied', params_used: { step: 'baseline', method: 'aspls', lam: 1e6, max_iter: 10 } }] };
    return { primary, chain, pin, outcome };
  }

  it('keeps the pin-time shots per point beside the pin without changing the pin', () => {
    const plain = evidenceTrace(snapshot())!;
    expect(pinTrace(plain, 'session:team:invented', 1800)).toBe(true);
    expect(get(pinnedTraces)[0].id).toBe(evidenceTrace(snapshot())!.id);
    expect(pinShotsPerPoint(plain.id)).toBe(1800);
    pinTrace(evidenceTrace(snapshot(1))!, 'session:team:invented', Number.NaN);
    pinTrace(evidenceTrace(snapshot(2))!, 'session:team:invented');
    expect([1, 2].map(i => pinShotsPerPoint(evidenceTrace(snapshot(i))!.id))).toEqual([null, null]);
    removeTrace(plain.id); expect(pinShotsPerPoint(plain.id)).toBeNull();
    pinTrace(plain, 'session:team:invented', 900); clearEvidence(); expect(pinShotsPerPoint(plain.id)).toBeNull();
  });

  it('derives a read-only trace and leaves the pin untouched', async () => {
    const { chain, pin, outcome } = await shared();
    const derived = derivedTrace(pin, chain, outcome)!;
    expect(derived.id).toBe(JSON.stringify([pin.id, chain.shared_chain_id]));
    expect(derived.snapshot.raman.intensity).toEqual([1, 2, 3]);
    expect(derived.snapshot.stage).toBe('baseline_corrected');
    expect(derived.snapshot.processingChain).toEqual([{ step: 'baseline', method: 'aspls', lam: 1e6, max_iter: 10,
      shared_chain_id: chain.shared_chain_id, status: 'applied' }]);
    expect(derived.derived).toMatchObject({ derived_from: pin.id, original_processing_chain: [] });
    expect(pin.snapshot.raman.intensity).toEqual([-2, 2, 5]);
    expect(Object.isFrozen(derived.snapshot)).toBe(true);
    expect(Object.keys(derived.snapshot).sort()).toEqual(['params', 'processingChain', 'raman', 'stage', 'trace']);
    expect(derived.snapshot.params).toEqual({});
    expect(derivedTrace(pin, chain, { ...outcome, refusal: 'synthetic refusal', intensity: null })).toBeNull();
  });

  it('drops the pin own processing artifacts and own despike mask, and keeps a source ML mask', async () => {
    const { chain, outcome } = await shared();
    const processed = snapshot(1);
    processed.stage = 'raman_fitted'; processed.params = { synthetic_fit: true };
    processed.processingChain = [{ step: 'despike', method: 'modz' }, { step: 'raman_fit', n_peaks: 1 }];
    processed.artifacts = { baseline: [0, 0, 0], fitCurve: [1, 1, 1] } as ProcessingSnapshot['artifacts'];
    const own = evidenceTrace(processed, { spike_mask: [true, false, false], spike_method: 'modz', badpix: [1] })!;
    const derived = derivedTrace(own, chain, { ...outcome, derived_from: own.id })!;
    expect(derived.snapshot).not.toHaveProperty('artifacts');
    expect(derived.snapshot.params).toEqual({});
    expect(derived.annotations).toEqual({ spike_mask: null, spike_method: null, badpix: [1] });
    expect(derived.derived!.original_processing_chain).toEqual(processed.processingChain);
    const despiked = derivedTrace(own, chain, { ...outcome, derived_from: own.id, spike_mask: [false, true, false] })!;
    expect(despiked.annotations).toEqual({ spike_mask: [false, true, false], spike_method: 'modz', badpix: [1] });
    const ml = evidenceTrace(snapshot(2), { spike_mask: [false, false, true], spike_method: 'ml', badpix: null })!;
    expect(derivedTrace(ml, chain, { ...outcome, derived_from: ml.id })!.annotations).toEqual(ml.annotations);
  });

  it('exports schema 1 unchanged unless a shared chain produced or refused a trace', async () => {
    const { primary, chain, pin, outcome } = await shared();
    const displays = comparisonDisplays(primary, [pin], 'raman_shift', false);
    const idle: SharedEvidence = { chain, primary_label: primary.label, refused: [] };
    for (const output of [evidenceExport(displays, null, null), evidenceExport(displays, null, idle)]) {
      expect(JSON.stringify(output)).toBe(JSON.stringify(evidenceExport(displays)));
    }
    expect(evidenceCsv(displays, null, idle)).toBe(evidenceCsv(displays));
    const derived = comparisonDisplays(primary, [derivedTrace(pin, chain, outcome)!], 'raman_shift', false);
    const output = evidenceExport(derived, null, idle) as ReturnType<typeof evidenceExport> & Record<string, unknown>;
    expect(output.schema_version).toBe('pipeline-evidence-2');
    expect(output.shared_chain).toMatchObject({ shared_chain_id: chain.shared_chain_id, primary_trace_id: primary.id, primary_label: primary.label });
    expect(JSON.stringify(output.traces[0])).toBe(JSON.stringify(evidenceExport(displays).traces[0]));
    expect((output.traces[1] as Record<string, unknown>).processing).toMatchObject({ mode: 'shared', derived_from: pin.id,
      shared_chain_id: chain.shared_chain_id, steps: outcome.steps });
    const refused = evidenceExport(displays, null, { ...idle, refused: [{ label: pin.label, derived_from: pin.id, reason: 'synthetic refusal',
      steps: [{ step: 'baseline', status: 'refused', reason: 'synthetic refusal' }] }] });
    expect((refused as Record<string, unknown> & { shared_chain: { refused: unknown[] } }).shared_chain.refused[0]).toMatchObject({ steps: [{ status: 'refused' }] });
    expect(refused.schema_version).toBe('pipeline-evidence-2');
    expect(JSON.stringify(refused.traces)).toBe(JSON.stringify(evidenceExport(displays).traces));
    expect(JSON.parse(evidenceCsv(derived, null, idle).split('\n')[0].slice('# provenance: '.length))).toEqual(output);
  });

  it('caches outcomes until the pins clear, ignores results from before a clear and retries failures', async () => {
    const { chain, pin, outcome } = await shared();
    const epoch = sharedOutcomeEpoch();
    cacheSharedOutcome(chain.shared_chain_id, outcome, epoch);
    expect(cachedSharedOutcome(pin.id, chain.shared_chain_id)).toBe(outcome);
    clearEvidence();
    expect(cachedSharedOutcome(pin.id, chain.shared_chain_id)).toBeUndefined();
    cacheSharedOutcome(chain.shared_chain_id, outcome, epoch);
    expect(cachedSharedOutcome(pin.id, chain.shared_chain_id)).toBeUndefined();
    const failed = { ...outcome, refusal: 'shared chain failed (rate limited)', intensity: null };
    cacheSharedOutcome(chain.shared_chain_id, failed, sharedOutcomeEpoch());
    cacheSharedOutcome('synthetic-other-chain', { ...outcome, refusal: 'synthetic refusal', intensity: null }, sharedOutcomeEpoch());
    dropFailedSharedOutcomes();
    expect(cachedSharedOutcome(pin.id, chain.shared_chain_id)).toBeUndefined();
    expect(cachedSharedOutcome(pin.id, 'synthetic-other-chain')).toBeDefined();
  });
});
