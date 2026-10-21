import { ApiError, postBackground, postBaseline, postDespike } from './api';
import { backgroundStepRecord, baselineStepRecord, despikeStepRecord } from './processingChainRun';
import type { BaselineParams, DespikeParams, ProcessingStage } from './types';
import type { EvidenceTrace } from './evidenceTypes';

// One processing chain applied to every Evidence desk trace at plot time.
// The chain is the primary trace's committed transform steps (the fit is not
// shared). Each trace starts from its own source array, so the shared chain
// replaces a pin's own processing instead of stacking on it.

export type SharedStep =
  | { step: 'despike'; params: DespikeParams }
  | { step: 'bg_subtract'; bg_type: 'as' | 'fs'; scale_mode: 'auto' }
  | { step: 'bg_subtract'; bg_type: 'as' | 'fs'; scale_mode: 'manual'; scale: number }
  | { step: 'baseline'; params: BaselineParams };

export interface SharedChain {
  shared_chain_id: string;
  primary_trace_id: string;
  steps: SharedStep[];
}

// Every chain step gets one outcome, also on a refused or failed trace:
// not_run marks the steps after the one that refused or failed.
export interface StepOutcome {
  step: SharedStep['step'];
  status: 'applied' | 'not_applicable' | 'refused' | 'failed' | 'not_run';
  reason?: string;
  params_used?: Record<string, unknown>;
}

export interface ReferenceStateRecord {
  consulted: boolean;
  note?: string;
  release_id?: string | null;
  source_processing_state?: unknown;
  source_processing_evidence?: unknown;
}

export interface SharedOutcome {
  derived_from: string;
  label: string;
  refusal: string | null;
  steps: StepOutcome[];
  scale_used: number | null;
  reference_processing_state?: ReferenceStateRecord;
  intensity: number[] | null;
  stage: ProcessingStage;
  // The shared despike step's mask, when that step ran.
  spike_mask?: boolean[];
}

export interface ChainApi {
  postDespike: typeof postDespike;
  postBackground: typeof postBackground;
  postBaseline: typeof postBaseline;
}

const DESPIKE_KEYS = ['window_size', 'zscore_threshold', 'max_iterations', 'sulfate_guard'] as const;
const BASELINE_KEYS = ['method', 'lam', 'max_iter', 'wavenumber_range'] as const;

function pick(record: Record<string, unknown>, keys: readonly string[]): Record<string, unknown> {
  return Object.fromEntries(keys.filter(key => record[key] !== undefined).map(key => [key, record[key]]));
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical((value as Record<string, unknown>)[key])]));
  }
  return value;
}

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

function isReference(trace: EvidenceTrace): boolean {
  return trace.snapshot.trace.provenance.data_origin === 'reference';
}

export interface SharedSource {
  steps: SharedStep[] | null;
  // Why the primary's chain cannot be shared, when it has steps but one of
  // them cannot be reproduced.
  refusal: string | null;
}

const NOTHING: SharedSource = { steps: null, refusal: null };

// The primary's committed transform steps. The background scale mode is read
// only from the committed record; a record without one (stored before the
// mode was recorded) is refused instead of guessed.
export function sharedSource(primary: EvidenceTrace | null): SharedSource {
  if (!primary || isReference(primary) || primary.snapshot.trace.region !== 'R1' || primary.nativeDomain !== 'raman_shift') return NOTHING;
  const steps: SharedStep[] = [];
  for (const record of primary.snapshot.processingChain ?? []) {
    if (record.step === 'despike' && record.method === 'modz') {
      steps.push({ step: 'despike', params: pick(record, DESPIKE_KEYS) as DespikeParams });
    } else if (record.step === 'bg_subtract' && (record.bg_type === 'as' || record.bg_type === 'fs')) {
      if (record.scale_mode === 'auto') {
        steps.push({ step: 'bg_subtract', bg_type: record.bg_type, scale_mode: 'auto' });
      } else if (record.scale_mode === 'manual' && typeof record.scale_used === 'number' && Number.isFinite(record.scale_used)) {
        steps.push({ step: 'bg_subtract', bg_type: record.bg_type, scale_mode: 'manual', scale: record.scale_used });
      } else {
        return { steps: null, refusal: 'background scale mode not recorded for this trace\'s background step; re-apply it to share the chain' };
      }
    } else if (record.step === 'baseline') {
      steps.push({ step: 'baseline', params: pick(record, BASELINE_KEYS) as BaselineParams });
    } else if (record.step !== 'raman_fit') {
      return NOTHING;
    }
  }
  return steps.length ? { steps, refusal: null } : NOTHING;
}

export function sharedSteps(primary: EvidenceTrace | null): SharedStep[] | null {
  return sharedSource(primary).steps;
}

// Changing the primary changes the chain id even when the steps are equal.
export async function sharedChain(primary: EvidenceTrace, steps: SharedStep[]): Promise<SharedChain> {
  const id = await sha256(JSON.stringify(canonical({ primary_trace_id: primary.id, steps })));
  return { shared_chain_id: id, primary_trace_id: primary.id, steps };
}

function strictlyIncreasing(x: number[]): boolean {
  return x.every((v, i) => Number.isFinite(v) && (i === 0 || v > x[i - 1]));
}

function baselineRefusal(x: number[], params: BaselineParams): string | null {
  if (x.length < 3 || !strictlyIncreasing(x)) return 'baseline needs at least 3 strictly increasing axis values';
  const range = params.wavenumber_range;
  if (range && (x[0] > range[0] || x[x.length - 1] < range[1])) {
    return `baseline range ${range[0]}–${range[1]} cm⁻¹ is not covered by this trace's axis; never extrapolated`;
  }
  return null;
}

interface ReferenceRecordLike {
  release_id?: string;
  provenance?: Record<string, unknown>;
}

function referenceState(trace: EvidenceTrace) {
  const record = (trace.snapshot.trace.provenance.source as Record<string, unknown> | undefined)?.reference as ReferenceRecordLike | undefined;
  const provenance = record?.provenance ?? {};
  return { release_id: record?.release_id ?? null, source_processing_state: provenance.source_processing_state,
    source_processing_evidence: provenance.source_processing_evidence };
}

type Decision = { status: 'run' } | { status: 'not_applicable' | 'refused'; reason: string };

interface Plan {
  // A trace-level refusal (domain or axis) applies before any step is considered.
  refusal: string | null;
  decisions: Decision[];
  reference?: ReferenceStateRecord;
}

// Decide every step before any request.
export function planSharedChain(trace: EvidenceTrace, chain: SharedChain, ppp: number | null): Plan {
  const x = trace.snapshot.trace.wavenumber;
  const y = trace.snapshot.trace.intensity;
  if (trace.nativeDomain !== 'raman_shift' || trace.snapshot.trace.region !== 'R1') {
    return { decisions: [], refusal: isReference(trace) ? 'wavelength-domain reference: the shared Raman chain does not apply'
      : 'luminescence-domain trace: the shared Raman chain does not apply' };
  }
  if (!x?.length || x.length !== y?.length) return { decisions: [], refusal: 'trace has no source Raman axis of matching length' };
  if (isReference(trace)) {
    const hasBaseline = chain.steps.some(step => step.step === 'baseline');
    const state = referenceState(trace);
    const reference: ReferenceStateRecord = hasBaseline ? { consulted: true, ...state }
      : { consulted: false, note: 'state not consulted: no baseline step' };
    const decisions = chain.steps.map((step): Decision => {
      if (step.step === 'despike') return { status: 'not_applicable', reason: 'not_applicable: instrument_cosmic_ray_correction' };
      if (step.step === 'bg_subtract') return { status: 'not_applicable', reason: 'not_applicable: instrument_background' };
      const release = state.release_id ?? 'unknown';
      if (state.source_processing_state !== 'unprocessed' && state.source_processing_state !== 'baseline_corrected') {
        return { status: 'refused', reason: `reference processing state not recorded in release ${release}` };
      }
      if (typeof state.source_processing_evidence !== 'string' || state.source_processing_evidence.trim() === '') {
        return { status: 'refused', reason: `reference processing state recorded without evidence in release ${release}` };
      }
      if (state.source_processing_state === 'baseline_corrected') {
        return { status: 'not_applicable', reason: 'not_applicable: reference_already_baseline_corrected' };
      }
      const refusal = baselineRefusal(x, step.params);
      return refusal ? { status: 'refused', reason: refusal } : { status: 'run' };
    });
    return { refusal: null, decisions, reference };
  }
  const mlDespiked = trace.snapshot.trace.provenance.processing_chain?.some(item => item.step === 'despike');
  const decisions = chain.steps.map((step): Decision => {
    if (step.step === 'despike' && mlDespiked) return { status: 'refused', reason: 'source already despiked (ML); no double despike' };
    if (step.step === 'bg_subtract' && step.scale_mode === 'auto' && ppp === null) {
      return { status: 'refused', reason: 'shots per point not recorded for this pin; auto background scale unavailable' };
    }
    if (step.step === 'baseline') {
      const refusal = baselineRefusal(x, step.params);
      if (refusal) return { status: 'refused', reason: refusal };
    }
    return { status: 'run' };
  });
  return { refusal: null, decisions };
}

const STAGES: Record<SharedStep['step'], ProcessingStage> = {
  despike: 'despiked', bg_subtract: 'bg_subtracted', baseline: 'baseline_corrected',
};

// Runs the planned steps in chain order through the same api.ts calls the
// Workbench steps make. `stale` is checked after every request.
export async function runSharedChain(
  trace: EvidenceTrace, chain: SharedChain, ppp: number | null, stale: () => boolean = () => false,
  api: ChainApi = { postDespike, postBackground, postBaseline },
): Promise<SharedOutcome | null> {
  const plan = planSharedChain(trace, chain, ppp);
  const base = { derived_from: trace.id, label: trace.label, ...(plan.reference ? { reference_processing_state: plan.reference } : {}) };
  const refused = (reason: string, steps: StepOutcome[]): SharedOutcome =>
    ({ ...base, refusal: reason, steps, scale_used: null, intensity: null, stage: 'raw' });
  const notRun = (from: number) => chain.steps.slice(from).map((step): StepOutcome => ({ step: step.step, status: 'not_run' }));
  if (plan.refusal !== null) return refused(plan.refusal, notRun(0));
  // A refusing step stops the trace before any request; earlier recorded skips are kept.
  const refusing = plan.decisions.findIndex(decision => decision.status === 'refused');
  if (refusing >= 0) {
    const recorded = plan.decisions.slice(0, refusing).map((decision, index): StepOutcome =>
      decision.status === 'not_applicable' ? { step: chain.steps[index].step, status: 'not_applicable', reason: decision.reason }
        : { step: chain.steps[index].step, status: 'not_run' });
    const reason = (plan.decisions[refusing] as { reason: string }).reason;
    return refused(reason, [...recorded, { step: chain.steps[refusing].step, status: 'refused', reason }, ...notRun(refusing + 1)]);
  }
  const x = trace.snapshot.trace.wavenumber;
  let y = trace.snapshot.trace.intensity;
  let stage: ProcessingStage = 'raw';
  let scaleUsed: number | null = null;
  let spikeMask: boolean[] | undefined;
  const steps: StepOutcome[] = [];
  for (const [index, step] of chain.steps.entries()) {
    const decision = plan.decisions[index];
    if (decision.status === 'not_applicable') { steps.push({ step: step.step, status: 'not_applicable', reason: decision.reason }); continue; }
    try {
      if (step.step === 'despike') {
        const result = await api.postDespike({ wavenumber: x, intensity: y, params: step.params });
        if (stale()) return null;
        y = result.despiked;
        spikeMask = result.spike_mask;
        steps.push({ step: 'despike', status: 'applied', params_used: despikeStepRecord(result.n_spikes, { ...result.params_used }) });
      } else if (step.step === 'bg_subtract') {
        const result = await api.postBackground({ wavenumber: x, intensity: y, bg_type: step.bg_type,
          scale: step.scale_mode === 'auto' ? 'auto' : step.scale, scan_ppp: ppp ?? 900 });
        if (stale()) return null;
        y = result.subtracted;
        scaleUsed = result.scale_used;
        steps.push({ step: 'bg_subtract', status: 'applied', params_used: backgroundStepRecord(result.bg_type, result.scale_used, step.scale_mode) });
      } else {
        const result = await api.postBaseline({ wavenumber: x, intensity: y, params: step.params });
        if (stale()) return null;
        y = result.corrected;
        steps.push({ step: 'baseline', status: 'applied', params_used: baselineStepRecord({ ...result.params_used }) });
      }
    } catch (error) {
      if (stale()) return null;
      const reason = error instanceof ApiError && error.status === 429 ? 'shared chain failed (rate limited)'
        : `shared chain failed: ${error instanceof Error ? error.message : 'request error'}`;
      return refused(reason, [...steps, { step: step.step, status: 'failed', reason }, ...notRun(index + 1)]);
    }
    stage = STAGES[step.step];
  }
  return { ...base, refusal: null, steps, scale_used: scaleUsed, intensity: y, stage, ...(spikeMask ? { spike_mask: spikeMask } : {}) };
}
