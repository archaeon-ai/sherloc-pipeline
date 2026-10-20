import { describe, expect, it, vi } from 'vitest';
import { ApiError } from './api';
import { evidenceTrace } from './evidenceStore';
import { referenceEvidence } from './identification';
import { planSharedChain, runSharedChain, sharedChain, sharedSteps } from './sharedChain';
import type { ChainApi, SharedStep } from './sharedChain';
import type { EvidenceTrace } from './evidenceTypes';
import type { ProcessingSnapshot } from './types';

// Entirely synthetic arrays and identifiers.
const X = [700, 800, 900, 1000, 1100];

function mars(chain: Record<string, unknown>[] = [], options: { region?: string; x?: number[]; mlDespiked?: boolean; tag?: number } = {}): EvidenceTrace {
  const x = options.x ?? X;
  const snapshot: ProcessingSnapshot = { stage: chain.length ? 'baseline_corrected' : 'raw', params: {}, processingChain: chain,
    raman: { wavenumber: x, intensity: x.map((_, i) => 100 + i) },
    trace: { region: options.region ?? 'R1', wavenumber: x, wavelength: x.map((_, i) => 250 + i), intensity: x.map((_, i) => 10 + i),
      channel_indices: x.map((_, i) => i), source: { scan_id: `synthetic-${options.tag ?? 0}` }, selection: { mode: 'point', indices: [0] },
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'synthetic',
        ...(options.mlDespiked ? { processing_chain: [{ step: 'despike', method: 'synthetic-ml' }] } : {}) } } };
  return evidenceTrace(snapshot)!;
}

function reference(provenance: Record<string, unknown> = {}, quantity = 'raman_shift'): EvidenceTrace {
  return referenceEvidence({ trace_revision_id: 'f'.repeat(64), release_id: 'e'.repeat(64), axis_sha256: 'a'.repeat(64),
    intensity_sha256: 'b'.repeat(64), point_count: X.length, x: X, y: X.map((_, i) => 5 + i),
    provenance: { source_label: 'Synthetic reference', species_name: 'Syntheticite', x_quantity: quantity,
      x_unit: quantity === 'raman_shift' ? 'cm-1' : 'nm', y_unit: 'synthetic', citation_text: 'Synthetic citation', ...provenance } })!;
}

const despike = { step: 'despike', method: 'modz', n_spikes: 1, window_size: 5, zscore_threshold: 3.5, max_iterations: 1, sulfate_guard: false };
const background = { step: 'bg_subtract', bg_type: 'as', scale_used: 2 };
const baseline = { step: 'baseline', method: 'aspls', lam: 1e6, max_iter: 10 };
const fit = { step: 'raman_fit', n_peaks: 2 };

function api(): ChainApi & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    postDespike: vi.fn(async body => { calls.push('despike'); return { schema_version: '1', despiked: body.intensity.map(v => v + 1),
      spike_mask: body.intensity.map(() => false), n_spikes: 0, params_used: { ...body.params } }; }),
    postBackground: vi.fn(async body => { calls.push('bg'); const scale = body.scale === 'auto' ? body.scan_ppp / 900 : body.scale;
      return { schema_version: '1', subtracted: body.intensity.map(v => v - scale), background_scaled: body.intensity.map(() => scale),
        scale_used: scale, bg_type: body.bg_type }; }),
    postBaseline: vi.fn(async body => { calls.push('baseline'); return { schema_version: '1', raw: body.intensity,
      baseline: body.intensity.map(() => 1), corrected: body.intensity.map(v => v * 2), wavenumber: body.wavenumber, params_used: { ...body.params } }; }),
  };
}

async function chainOf(steps: SharedStep[], primary = mars([baseline])) {
  return sharedChain(primary, steps);
}

describe('shared chain source', () => {
  it('takes the committed despike, background and baseline steps and never the fit', () => {
    const steps = sharedSteps(mars([despike, background, baseline, fit]), { mode: 'auto', value: 1 }, 1800);
    expect(steps).toEqual([
      { step: 'despike', params: { window_size: 5, zscore_threshold: 3.5, max_iterations: 1, sulfate_guard: false } },
      { step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto' },
      { step: 'baseline', params: { method: 'aspls', lam: 1e6, max_iter: 10 } },
    ]);
    expect(sharedSteps(mars([background]), { mode: 'manual', value: 2 }, 900)).toEqual([
      { step: 'bg_subtract', bg_type: 'as', scale_mode: 'manual', scale: 2 }]);
  });

  it('shares nothing it cannot reproduce', () => {
    const scale = { mode: 'auto' as const, value: 1 };
    expect(sharedSteps(null, scale, 900)).toBeNull();
    expect(sharedSteps(mars([]), scale, 900)).toBeNull();
    expect(sharedSteps(mars([fit]), scale, 900)).toBeNull();
    expect(sharedSteps(mars([baseline], { region: 'R2' }), scale, 900)).toBeNull();
    expect(sharedSteps(mars([{ step: 'synthetic-step' }, baseline]), scale, 900)).toBeNull();
    expect(sharedSteps(reference(), scale, 900)).toBeNull();
    // The background record must agree with the Workbench's scale mode.
    expect(sharedSteps(mars([background]), null, 1800)).toBeNull();
    expect(sharedSteps(mars([background]), scale, 900)).toBeNull();
    expect(sharedSteps(mars([background]), scale, null)).toBeNull();
    expect(sharedSteps(mars([background]), { mode: 'manual', value: 3 }, 1800)).toBeNull();
  });

  it('identifies the chain by its steps and its primary', async () => {
    const steps = sharedSteps(mars([baseline]), null, null)!;
    const first = await chainOf(steps);
    expect(first.shared_chain_id).toMatch(/^[0-9a-f]{64}$/);
    expect((await chainOf(steps)).shared_chain_id).toBe(first.shared_chain_id);
    expect((await chainOf([{ step: 'baseline', params: { max_iter: 10, lam: 1e6, method: 'aspls' } }])).shared_chain_id).toBe(first.shared_chain_id);
    expect((await chainOf(steps, mars([baseline], { tag: 1 }))).shared_chain_id).not.toBe(first.shared_chain_id);
    expect((await chainOf([{ step: 'baseline', params: { method: 'aspls', lam: 1e5, max_iter: 10 } }])).shared_chain_id).not.toBe(first.shared_chain_id);
  });
});

describe('applying the shared chain', () => {
  const all: SharedStep[] = [
    { step: 'despike', params: { window_size: 5 } },
    { step: 'bg_subtract', bg_type: 'fs', scale_mode: 'auto' },
    { step: 'baseline', params: { method: 'aspls', lam: 1e6, max_iter: 10 } },
  ];

  it('runs every step in order from the pin source array, with the pin own shots per point', async () => {
    const calls = api();
    const pin = mars([despike], { tag: 2 });
    const outcome = (await runSharedChain(pin, await chainOf(all), 1800, undefined, calls))!;
    expect(calls.calls).toEqual(['despike', 'bg', 'baseline']);
    expect(vi.mocked(calls.postDespike).mock.calls[0][0].intensity).toEqual([10, 11, 12, 13, 14]);
    expect(vi.mocked(calls.postBackground).mock.calls[0][0]).toMatchObject({ intensity: [11, 12, 13, 14, 15], scale: 'auto', scan_ppp: 1800 });
    expect(outcome.intensity).toEqual([18, 20, 22, 24, 26]);
    expect(outcome).toMatchObject({ refusal: null, stage: 'baseline_corrected', scale_used: 2, derived_from: pin.id });
    expect(outcome.steps.map(step => [step.step, step.status])).toEqual([['despike', 'applied'], ['bg_subtract', 'applied'], ['baseline', 'applied']]);
    expect(outcome.steps[1].params_used).toEqual({ step: 'bg_subtract', bg_type: 'fs', scale_used: 2 });
  });

  it('applies one manual scale to every pin', async () => {
    const calls = api();
    const outcome = (await runSharedChain(mars(), await chainOf([{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'manual', scale: 0.5 }]), null, undefined, calls))!;
    expect(vi.mocked(calls.postBackground).mock.calls[0][0].scale).toBe(0.5);
    expect(outcome).toMatchObject({ refusal: null, scale_used: 0.5, stage: 'bg_subtracted' });
  });

  it.each([
    ['an ML-despiked source', mars([], { mlDespiked: true }), 1800, all, 'source already despiked (ML); no double despike'],
    ['a pin without shots per point', mars(), null, all, 'shots per point not recorded for this pin; auto background scale unavailable'],
    ['a luminescence-domain trace', mars([], { region: 'R2' }), 1800, all, 'luminescence-domain trace: the shared Raman chain does not apply'],
    ['a decreasing axis', mars([], { x: [...X].reverse() }), 1800, all.slice(2), 'baseline needs at least 3 strictly increasing axis values'],
    ['an uncovered baseline range', mars(), 1800, [{ step: 'baseline', params: { method: 'aspls', wavenumber_range: [600, 1000] } }],
      'baseline range 600–1000 cm⁻¹ is not covered by this trace\'s axis; never extrapolated'],
  ] as [string, EvidenceTrace, number | null, SharedStep[], string][])('refuses %s before any request', async (_name, trace, ppp, steps, reason) => {
    const calls = api();
    const outcome = (await runSharedChain(trace, await chainOf(steps), ppp, undefined, calls))!;
    expect(outcome).toMatchObject({ refusal: reason, intensity: null, steps: [] });
    expect(calls.calls).toEqual([]);
  });

  it('runs a covered baseline range and despikes a source that was not ML-despiked', async () => {
    const plan = planSharedChain(mars(), await chainOf([{ step: 'baseline', params: { wavenumber_range: [800, 1000] } }, ...all.slice(0, 1)]), null);
    expect(plan.refusal).toBeNull();
  });

  it('reports a rate-limited or failed request and drops a superseded one', async () => {
    const limited = api();
    vi.mocked(limited.postBaseline).mockRejectedValueOnce(new ApiError(429, 'rate_limited'));
    expect(await runSharedChain(mars(), await chainOf(all.slice(2)), 1800, undefined, limited))
      .toMatchObject({ refusal: 'shared chain failed (rate limited)', intensity: null });
    const failed = api();
    vi.mocked(failed.postBaseline).mockRejectedValueOnce(new ApiError(422, 'invalid_input', 'synthetic detail'));
    expect((await runSharedChain(mars(), await chainOf(all.slice(2)), 1800, undefined, failed))!.refusal)
      .toBe('shared chain failed: invalid_input: synthetic detail');
    expect(await runSharedChain(mars(), await chainOf(all), 1800, () => true, api())).toBeNull();
  });
});

describe('references under the shared chain', () => {
  const all: SharedStep[] = [
    { step: 'despike', params: { window_size: 5 } },
    { step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto' },
    { step: 'baseline', params: { method: 'aspls', lam: 1e6, max_iter: 10 } },
  ];

  it('skips instrument corrections and baselines an unprocessed reference', async () => {
    const calls = api();
    const outcome = (await runSharedChain(reference({ source_processing_state: 'unprocessed', source_processing_evidence: 'synthetic record' }),
      await chainOf(all), null, undefined, calls))!;
    expect(calls.calls).toEqual(['baseline']);
    expect(vi.mocked(calls.postBaseline).mock.calls[0][0].intensity).toEqual([5, 6, 7, 8, 9]);
    expect(outcome.steps).toEqual([
      { step: 'despike', status: 'not_applicable', reason: 'not_applicable: instrument_cosmic_ray_correction' },
      { step: 'bg_subtract', status: 'not_applicable', reason: 'not_applicable: instrument_background' },
      { step: 'baseline', status: 'applied', params_used: { step: 'baseline', method: 'aspls', lam: 1e6, max_iter: 10 } },
    ]);
    expect(outcome.reference_processing_state).toEqual({ consulted: true, release_id: 'e'.repeat(64),
      source_processing_state: 'unprocessed', source_processing_evidence: 'synthetic record' });
    expect(outcome.stage).toBe('baseline_corrected');
  });

  it('skips the baseline of an already baseline-corrected reference and records why', async () => {
    const calls = api();
    const outcome = (await runSharedChain(reference({ source_processing_state: 'baseline_corrected', source_processing_evidence: 'synthetic record' }),
      await chainOf(all), null, undefined, calls))!;
    expect(calls.calls).toEqual([]);
    expect(outcome.steps[2]).toEqual({ step: 'baseline', status: 'not_applicable', reason: 'not_applicable: reference_already_baseline_corrected' });
    expect(outcome).toMatchObject({ refusal: null, stage: 'raw', intensity: [5, 6, 7, 8, 9] });
    expect(outcome.reference_processing_state).toMatchObject({ consulted: true, source_processing_state: 'baseline_corrected' });
  });

  it.each([
    [{}, `reference processing state not recorded in release ${'e'.repeat(64)}`],
    [{ source_processing_state: 'unknown', source_processing_evidence: 'synthetic record' }, `reference processing state not recorded in release ${'e'.repeat(64)}`],
    [{ source_processing_state: 'unprocessed' }, `reference processing state recorded without evidence in release ${'e'.repeat(64)}`],
    [{ source_processing_state: 'baseline_corrected', source_processing_evidence: ' ' }, `reference processing state recorded without evidence in release ${'e'.repeat(64)}`],
  ])('refuses a reference whose state is %j when the chain has a baseline', async (provenance, reason) => {
    const calls = api();
    const outcome = (await runSharedChain(reference(provenance), await chainOf(all), null, undefined, calls))!;
    expect(outcome).toMatchObject({ refusal: reason, intensity: null });
    expect(outcome.reference_processing_state).toMatchObject({ consulted: true, release_id: 'e'.repeat(64) });
    expect(calls.calls).toEqual([]);
  });

  it('does not consult the state without a baseline step and refuses a wavelength reference', async () => {
    const calls = api();
    const outcome = (await runSharedChain(reference(), await chainOf(all.slice(0, 2)), null, undefined, calls))!;
    expect(outcome).toMatchObject({ refusal: null, stage: 'raw', reference_processing_state: { consulted: false, note: 'state not consulted: no baseline step' } });
    expect(outcome.steps.map(step => step.status)).toEqual(['not_applicable', 'not_applicable']);
    expect((await runSharedChain(reference({}, 'wavelength'), await chainOf(all), null, undefined, calls))!.refusal)
      .toBe('wavelength-domain reference: the shared Raman chain does not apply');
    expect(calls.calls).toEqual([]);
  });
});
