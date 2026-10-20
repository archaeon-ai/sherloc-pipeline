import { get, writable } from 'svelte/store';
import type { IdentificationEvidence } from './identification';
import type { ExportTrace, ProcessingSnapshot } from './types';
import type { EvidenceDisplay, EvidenceDomain, EvidenceTrace, EvidenceTheme } from './evidenceTypes';
import type { SharedChain, SharedOutcome } from './sharedChain';

// The current selection occupies one of the four comparison slots.
export const MAX_PINNED_TRACES = 3;
const pins = writable<EvidenceTrace[]>([]);
export const pinnedTraces = { subscribe: pins.subscribe };
let context: string | null = null;
// Pin-time metadata kept beside the pins, so pin ids and pin snapshots stay unchanged.
const pinShots = new Map<string, number | null>();
// Shared-chain results by [trace id, shared chain id]; an epoch keeps a result
// computed before a clear from landing after it.
const sharedOutcomes = new Map<string, SharedOutcome>();
let sharedEpoch = 0;
export let evidenceGeneration = 0;
const boundary = writable(evidenceGeneration);
export const evidenceBoundary = { subscribe: boundary.subscribe };

export function invalidateEvidence(): void {
  evidenceGeneration += 1;
  clearEvidence();
  boundary.set(evidenceGeneration);
}
const colors = ['#2563eb', '#c026d3', '#ea580c'];

export function evidenceColor(trace: EvidenceTrace, theme: EvidenceTheme): EvidenceTrace {
  const dark = theme === 'dark' || (theme === 'system' && typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  return { ...trace, color: trace.color === '#1e293b' || trace.color === '#e2e8f0' ? (dark ? '#e2e8f0' : '#1e293b') : trace.color };
}

export function clearEvidence(): void {
  context = null;
  pinShots.clear();
  sharedOutcomes.clear();
  sharedEpoch += 1;
  pins.set([]);
}

export function setEvidenceContext(key: string): void {
  if (context !== key) {
    clearEvidence();
    context = key;
  }
}

function copy<T>(value: T): T {
  if (Array.isArray(value)) return value.map(copy) as T;
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, copy(v)])) as T;
  }
  return value;
}

function freeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

export function evidenceTrace(snapshot: ProcessingSnapshot | null, annotations: Record<string, unknown> = {}): EvidenceTrace | null {
  if (!snapshot?.trace) return null;
  const trace = snapshot.trace;
  const y = snapshot.raman.intensity;
  if (!y.length) return null;
  const validAxis = (x: number[] | null) => x?.length === y.length;
  if (!validAxis(snapshot.raman.wavenumber) && !validAxis(trace.wavelength)) return null;
  const captured = copy(snapshot) as EvidenceTrace['snapshot'];
  const id = JSON.stringify([trace.source, trace.selection, snapshot.stage,
    trace.region, trace.provenance, snapshot.processingChain, snapshot.raman, trace.wavelength, annotations]);
  return freeze({ id, label: `${trace.source.scan_name ?? trace.source.scan_id ?? 'Pipeline'} · ${JSON.stringify(trace.selection)} · ${snapshot.stage}`,
    nativeDomain: trace.region === 'R1' ? 'raman_shift' : trace.region === 'R2' || trace.region === 'R3' ? 'emission_wavelength' : null,
    snapshot: captured, annotations: copy(annotations), visible: true, color: '#1e293b',
    breaks: sourceBreaks(trace, y),
  });
}

function sourceBreaks(trace: ExportTrace, y: number[]): boolean[] {
  return y.map((v, i) => !Number.isFinite(v) ||
    (i > 0 && trace.channel_indices.length === y.length && trace.channel_indices[i] !== trace.channel_indices[i - 1] + 1));
}

export function pinTrace(trace: EvidenceTrace, expectedContext: string, shotsPerPoint: number | null = null): boolean {
  if (expectedContext !== context || trace.snapshot.trace.region === 'R123') return false;
  const current = get(pins);
  if (current.some(item => item.id === trace.id) || current.length >= MAX_PINNED_TRACES) return false;
  pinShots.set(trace.id, Number.isFinite(shotsPerPoint) && shotsPerPoint! > 0 ? shotsPerPoint : null);
  pins.set([...current, freeze({ ...copy(trace), color: colors.find(color => !current.some(item => item.color === color))! })]);
  return true;
}

export function pinShotsPerPoint(id: string): number | null {
  return pinShots.get(id) ?? null;
}

export function removeTrace(id: string): void {
  pinShots.delete(id);
  pins.update(items => items.filter(item => item.id !== id));
}

export function sharedOutcomeEpoch(): number {
  return sharedEpoch;
}

export function cachedSharedOutcome(traceId: string, chainId: string): SharedOutcome | undefined {
  return sharedOutcomes.get(JSON.stringify([traceId, chainId]));
}

// Failed requests (e.g. rate limited) are retried when sharing is turned on again.
export function dropFailedSharedOutcomes(): void {
  for (const [key, outcome] of sharedOutcomes) if (outcome.refusal?.startsWith('shared chain failed')) sharedOutcomes.delete(key);
}

export function cacheSharedOutcome(chainId: string, outcome: SharedOutcome, epoch: number): void {
  if (epoch === sharedEpoch) sharedOutcomes.set(JSON.stringify([outcome.derived_from, chainId]), outcome);
}

// A read-only trace for the shared chain's output. The pin itself is never changed.
// It carries only the shared chain's results: none of the pin's own params or
// artifacts (fit, baseline, background), and no spike mask from its own despike.
// An ML mask describes the source itself and is kept.
export function derivedTrace(trace: EvidenceTrace, chain: SharedChain, outcome: SharedOutcome): EvidenceTrace | null {
  if (outcome.refusal !== null || !outcome.intensity) return null;
  const { snapshot } = trace;
  const records = outcome.steps.map(step => ({ ...(step.params_used ?? { step: step.step }),
    shared_chain_id: chain.shared_chain_id, status: step.status, ...(step.reason ? { reason: step.reason } : {}) }));
  const annotations = outcome.spike_mask ? { ...trace.annotations, spike_mask: outcome.spike_mask, spike_method: 'modz' }
    : trace.annotations.spike_method === 'modz' ? { ...trace.annotations, spike_mask: null, spike_method: null } : trace.annotations;
  return freeze({ ...copy(trace), id: JSON.stringify([trace.id, chain.shared_chain_id]), annotations: copy(annotations),
    snapshot: copy({ stage: outcome.stage, processingChain: records, params: {}, trace: snapshot.trace,
      raman: { wavenumber: snapshot.trace.wavenumber, intensity: outcome.intensity } }),
    breaks: sourceBreaks(snapshot.trace, outcome.intensity),
    derived: copy({ derived_from: trace.id, shared_chain_id: chain.shared_chain_id,
      original_processing_chain: snapshot.processingChain ?? [], steps: outcome.steps, scale_used: outcome.scale_used,
      ...(outcome.reference_processing_state ? { reference_processing_state: outcome.reference_processing_state } : {}) }) });
}

export function toggleTrace(id: string): void {
  pins.update(items => items.map(item => item.id === id ? freeze({ ...item, visible: !item.visible }) : item));
}

function axisBreaks(trace: EvidenceTrace, x: number[]): boolean[] {
  const spacings = x.slice(1).map((v, i) => Math.abs(v - x[i])).filter(v => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
  const spacing = spacings[Math.floor((spacings.length - 1) / 2)];
  return x.map((v, i) => trace.breaks[i] || !Number.isFinite(v) ||
    (i > 0 && Number.isFinite(spacing) && Math.abs(v - x[i - 1]) > 3 * spacing));
}

export function nativeRenderTrace(trace: EvidenceTrace, domain: EvidenceDomain): EvidenceDisplay | null {
  const x = domain === 'raman_shift' ? trace.snapshot.raman.wavenumber : trace.snapshot.trace.wavelength;
  const original = trace.snapshot.raman.intensity;
  if (!x?.length || x.length !== original.length) return null;
  return { trace, domain, x, original, displayed: original, breaks: axisBreaks(trace, x),
    normalization: 'native', divisor: null, scalable: false, offset: 0 };
}

export function nativeFigureEvidence(trace: EvidenceTrace, domain: EvidenceDomain, identification: IdentificationEvidence | null = null) {
  const native = nativeRenderTrace(trace, domain);
  return evidenceExport(native ? [native] : [], identification);
}

export function displayTrace(trace: EvidenceTrace, domain: EvidenceDomain, offset = 0): EvidenceDisplay | null {
  const { snapshot } = trace;
  if (trace.nativeDomain ? trace.nativeDomain !== domain : snapshot.trace.region && snapshot.trace.region !== 'unknown') return null;
  const x = domain === 'raman_shift' ? snapshot.raman.wavenumber : snapshot.trace.wavelength;
  const original = snapshot.raman.intensity;
  if (!x?.length || x.length !== original.length) return null;
  const unit = domain === 'raman_shift' ? snapshot.trace.provenance.wavenumber_unit : snapshot.trace.provenance.wavelength_unit;
  if (unit !== (domain === 'raman_shift' ? 'cm-1' : 'nm')) return null;
  const breaks = axisBreaks(trace, x);
  const window = domain === 'raman_shift' ? [800, 1800] : [280, 350];
  let maximum = -Infinity;
  x.forEach((v, i) => {
    if (Number.isFinite(v) && v >= window[0] && v <= window[1] && Number.isFinite(original[i])) maximum = Math.max(maximum, original[i]);
  });
  const divisor = Number.isFinite(maximum) && maximum > 0 ? maximum : null;
  return { trace, domain, x, original, breaks, divisor, scalable: divisor !== null, normalization: 'fixed_window_maximum', offset,
    displayed: original.map(y => y / (divisor ?? 1) + offset) };
}

export function comparisonDisplays(primary: EvidenceTrace | null, pinned: EvidenceTrace[], domain: EvidenceDomain, stacked: boolean): EvidenceDisplay[] {
  const traces = [...(primary ? [primary] : []), ...pinned.filter(item => item.id !== primary?.id && item.visible)];
  return traces.map(item => displayTrace(item, domain)).filter((item): item is EvidenceDisplay => item !== null)
    .map((item, i) => displayTrace(item.trace, domain, stacked ? i * 1.25 : 0)!);
}

export interface SharedEvidence {
  chain: SharedChain;
  primary_label: string;
  refused: { label: string; derived_from: string; reason: string; steps: unknown[]; reference_processing_state?: unknown }[];
}

// Schema 2 only when a shared chain produced or refused a trace; otherwise today's schema 1, unchanged.
export function evidenceExport(displays: EvidenceDisplay[], identification: IdentificationEvidence | null = null,
  shared: SharedEvidence | null = null) {
  const base = nativeEvidenceExport(displays, identification);
  if (!shared || !(shared.refused.length || displays.some(item => item.trace.derived))) return base;
  return { ...base, schema_version: 'pipeline-evidence-2',
    shared_chain: { ...shared.chain, primary_label: shared.primary_label, refused: shared.refused },
    traces: base.traces.map((entry, i) => {
      const derived = displays[i].trace.derived;
      return derived ? { ...entry, processing: { mode: 'shared', ...derived } } : entry;
    }) };
}

function nativeEvidenceExport(displays: EvidenceDisplay[], identification: IdentificationEvidence | null) {
  return { schema_version: 'pipeline-evidence-1', ...(identification ? { identification } : {}), traces: displays.map(item => ({
    label: item.trace.label, domain: item.domain, color: item.trace.color,
    x_unit: (item.domain === 'raman_shift' ? item.trace.snapshot.trace.provenance.wavenumber_unit : item.trace.snapshot.trace.provenance.wavelength_unit) ?? null,
    intensity_unit: item.trace.snapshot.trace.provenance.intensity_unit ?? null,
    snapshot: item.trace.snapshot, annotations: item.trace.annotations, breaks: item.breaks,
    display: { divisor: item.divisor, scalable: item.scalable, offset: item.offset, normalization: item.normalization },
    x: item.x, original: item.original, displayed: item.displayed,
  })) };
}

export function evidenceCsv(displays: EvidenceDisplay[], identification: IdentificationEvidence | null = null,
  shared: SharedEvidence | null = null): string {
  const lines = [`# provenance: ${JSON.stringify(evidenceExport(displays, identification, shared))}`,
    'trace_index,domain,x,original,displayed,break_before'];
  displays.forEach((item, index) => item.x.forEach((x, i) => {
    const number = (v: number) => Number.isFinite(v) ? String(v) : '';
    lines.push([index, item.domain, number(x), number(item.original[i]), number(item.displayed[i]), item.breaks[i] ? 1 : 0].join(','));
  }));
  return lines.join('\n');
}
