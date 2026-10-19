import { get, writable } from 'svelte/store';
import type { ProcessingSnapshot } from './types';
import type { EvidenceDisplay, EvidenceDomain, EvidenceTrace, EvidenceTheme } from './evidenceTypes';

// The current selection occupies one of the four comparison slots.
export const MAX_PINNED_TRACES = 3;
const pins = writable<EvidenceTrace[]>([]);
export const pinnedTraces = { subscribe: pins.subscribe };
let context: string | null = null;
export let evidenceGeneration = 0;

export function invalidateEvidence(): void {
  evidenceGeneration += 1;
  clearEvidence();
}
const colors = ['#2563eb', '#c026d3', '#ea580c'];

export function evidenceColor(trace: EvidenceTrace, theme: EvidenceTheme): EvidenceTrace {
  const dark = theme === 'dark' || (theme === 'system' && typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  return { ...trace, color: trace.color === '#1e293b' || trace.color === '#e2e8f0' ? (dark ? '#e2e8f0' : '#1e293b') : trace.color };
}

export function clearEvidence(): void {
  context = null;
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
    breaks: y.map((v, i) => !Number.isFinite(v) ||
      (i > 0 && trace.channel_indices.length === y.length && trace.channel_indices[i] !== trace.channel_indices[i - 1] + 1)),
  });
}

export function pinTrace(trace: EvidenceTrace, expectedContext: string): boolean {
  if (expectedContext !== context) return false;
  const current = get(pins);
  if (current.some(item => item.id === trace.id) || current.length >= MAX_PINNED_TRACES) return false;
  pins.set([...current, freeze({ ...copy(trace), color: colors.find(color => !current.some(item => item.color === color))! })]);
  return true;
}

export function removeTrace(id: string): void {
  pins.update(items => items.filter(item => item.id !== id));
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

export function nativeFigureEvidence(trace: EvidenceTrace, domain: EvidenceDomain) {
  const x = (domain === 'raman_shift' ? trace.snapshot.raman.wavenumber : trace.snapshot.trace.wavelength) ?? [];
  const original = trace.snapshot.raman.intensity;
  return evidenceExport([{ trace, domain, x, original, displayed: original, breaks: axisBreaks(trace, x),
    normalization: 'native', divisor: null, scalable: false, offset: 0 }]);
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

export function evidenceExport(displays: EvidenceDisplay[]) {
  return { schema_version: 'pipeline-evidence-1', traces: displays.map(item => ({
    label: item.trace.label, domain: item.domain, color: item.trace.color,
    x_unit: (item.domain === 'raman_shift' ? item.trace.snapshot.trace.provenance.wavenumber_unit : item.trace.snapshot.trace.provenance.wavelength_unit) ?? null,
    intensity_unit: item.trace.snapshot.trace.provenance.intensity_unit ?? null,
    snapshot: item.trace.snapshot, annotations: item.trace.annotations, breaks: item.breaks,
    display: { divisor: item.divisor, scalable: item.scalable, offset: item.offset, normalization: item.normalization },
    x: item.x, original: item.original, displayed: item.displayed,
  })) };
}

export function evidenceCsv(displays: EvidenceDisplay[]): string {
  const lines = [`# provenance: ${JSON.stringify(evidenceExport(displays))}`,
    'trace_index,domain,x,original,displayed,break_before'];
  displays.forEach((item, index) => item.x.forEach((x, i) => {
    const number = (v: number) => Number.isFinite(v) ? String(v) : '';
    lines.push([index, item.domain, number(x), number(item.original[i]), number(item.displayed[i]), item.breaks[i] ? 1 : 0].join(','));
  }));
  return lines.join('\n');
}
