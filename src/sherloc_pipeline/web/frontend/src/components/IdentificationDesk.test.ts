import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import EvidenceDesk from './EvidenceDesk.svelte';
import { accessMode, appConfig } from '../lib/stores';
import { clearEvidence, setEvidenceContext } from '../lib/evidenceStore';
import type { ConfigResponse, ProcessingSnapshot } from '../lib/types';
import type { IdentificationEvidence, ReferenceRecord } from '../lib/identification';

const api = vi.hoisted(() => ({ getReferences: vi.fn(), getReference: vi.fn(), identifyStoredPoint: vi.fn() }));
const plot = vi.hoisted(() => ({ react: vi.fn((div, _data, layout) => { div.layout = layout; }), purge: vi.fn(), downloadImage: vi.fn() }));
vi.mock('../lib/api', () => ({ ...api, ApiError: class extends Error { constructor(public status: number) { super('Synthetic error'); } } }));
vi.mock('plotly.js-basic-dist-min', () => plot);

const release = 'a'.repeat(64), subset = 'b'.repeat(64), method = 'c'.repeat(64), trace = 'd'.repeat(64);
const binding = { reference_release_id: release, reference_subset_id: subset, method_release_id: method, template_ids: ['e'.repeat(64)] };
const reference: ReferenceRecord & { x: number[]; y: number[] } = {
  trace_revision_id: trace, release_id: release, axis_sha256: 'f'.repeat(64), intensity_sha256: '1'.repeat(64), point_count: 3,
  x: [820, 940, 1080], y: [-1, 2, 4], provenance: { source_label: 'Synthetic trace', species_name: 'Syntheticite',
    x_quantity: 'raman_shift', x_unit: 'cm-1', y_unit: 'synthetic units', citation_text: 'Synthetic citation' },
};
function config(flags = [true, true, true], tier: 'team' | 'public' = 'team'): ConfigResponse {
  return { tier, evidence_policy_digest: '2'.repeat(64), reference_release_id: release,
    capabilities: { evidence_desk: true, references: flags[0], identification: flags[1], experimental_candidates: flags[2] } } as ConfigResponse;
}
function input(digest = '3'.repeat(64)): ProcessingSnapshot {
  return { stage: 'raw', params: {}, processingChain: [], raman: { wavenumber: [810, 930, 1050], intensity: [-2, 3, 5] },
    trace: { region: 'R1', source: { scan_id: 'synthetic-scan' }, selection: { mode: 'point', indices: [0] },
      wavenumber: [810, 930, 1050], intensity: [-2, 3, 5], wavelength: null, channel_indices: [0, 1, 2],
      provenance: { source_digest: digest, calibration_version: null, wavenumber_unit: 'cm-1', intensity_unit: 'synthetic units',
        source_processing: { spectrum_type: 'laser_normalized' } } } };
}
function result(changes: Partial<IdentificationEvidence> = {}): IdentificationEvidence {
  return { result_digest: '4'.repeat(64), source_digest: '3'.repeat(64), status: 'candidate', reasons: [],
    reference_release_id: release, reference_subset_id: subset,
    method: { release_id: method, support_min: .80, status: 'provisional' }, ranking: [
      { template_id: 'e'.repeat(64), species_name: 'Syntheticite', support: .91, trace_revision_ids: [trace],
        citations: [{ trace_revision_id: trace, citation_text: 'Synthetic citation' }] },
      { template_id: '5'.repeat(64), species_name: 'Alternatite', support: .72, trace_revision_ids: [], citations: [] }], ...changes };
}
const flush = async () => { for (let i = 0; i < 5; i++) { await tick(); await new Promise(r => setTimeout(r, 0)); } };
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(r => { resolve = r; }); return { promise, resolve }; };

beforeEach(() => {
  vi.clearAllMocks(); clearEvidence(); setEvidenceContext('synthetic-context');
  accessMode.set('internal'); appConfig.set(config());
  api.getReferences.mockResolvedValue({ items: [reference], next_cursor: null, binding });
  api.getReference.mockResolvedValue(reference); api.identifyStoredPoint.mockResolvedValue(result());
  URL.createObjectURL = vi.fn(() => 'blob:synthetic'); URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

describe('Identification evidence', () => {
  it.each(Array.from({ length: 16 }, (_, n) => n))('enforces the client feature and tier combination %i', async n => {
    const flags = [Boolean(n & 1), Boolean(n & 2), Boolean(n & 4)];
    const tier = n & 8 ? 'public' : 'team';
    appConfig.set(config(flags, tier)); accessMode.set(tier === 'public' ? 'public' : 'internal');
    render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' }); await flush();
    expect(!!screen.queryByRole('button', { name: 'Identify' })).toBe(tier === 'team' && flags.every(Boolean));
    expect(!!screen.queryByRole('button', { name: 'Browse references' })).toBe(tier === 'team' && flags[0]);
    expect(api.identifyStoredPoint).not.toHaveBeenCalled(); expect(api.getReferences).not.toHaveBeenCalled();
  });

  it('hides Identify in luminescence and disables manual or aggregate selections', async () => {
    const { rerender } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'emission' } }); await flush();
    expect(screen.queryByRole('button', { name: 'Identify' })).toBeNull();
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'overlay' } });
    const manual = input(); manual.stage = 'baseline_corrected';
    await rerender({ snapshot: manual, contextKey: 'synthetic-context' }); await flush();
    expect(screen.getByRole('button', { name: 'Identify' })).toBeDisabled();
    const aggregate = input(); aggregate.trace!.selection = { mode: 'average' };
    await rerender({ snapshot: aggregate, contextKey: 'synthetic-context' }); await flush();
    expect(screen.getByRole('button', { name: 'Identify' })).toBeDisabled();
  });

  it('runs only the committed source with pinned releases and shows support, alternatives and citations', async () => {
    const { container } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    expect(api.identifyStoredPoint).toHaveBeenCalledWith({ ...binding, scan_id: 'synthetic-scan', selection: { mode: 'point', index: 0 },
      region: 'R1', spectrum_type: 'laser_normalized', expected_source_digest: '3'.repeat(64) });
    expect(container.textContent).toContain('support 0.910'); expect(container.textContent).toContain('provisional threshold 0.80');
    expect(container.textContent).toContain('Alternative: Alternatite'); expect(container.textContent).toContain('Synthetic citation');
    expect(container.textContent).not.toMatch(/probab|confiden|percent|\b91%/i);
    await fireEvent.click(screen.getByRole('button', { name: 'Add Syntheticite trace' })); await flush();
    expect(api.getReference).toHaveBeenCalledWith(release, trace);
    expect(plot.react.mock.calls.at(-1)![1]).toHaveLength(2);
    expect(plot.react.mock.calls.at(-1)![2].meta.identification).toEqual(result());
  });

  it('shows a per-trace DOI when citation text is missing', async () => {
    const evidence = result();
    evidence.ranking[0].citations[0] = { trace_revision_id: trace, citation_text: null, citation_doi: 'synthetic:reference-0001' };
    api.identifyStoredPoint.mockResolvedValue(evidence);
    api.getReference.mockResolvedValue({ ...reference, provenance: { ...reference.provenance, citation_text: null, citation_doi: 'synthetic:reference-0001' } });
    const { container } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    expect(container.textContent).toContain('Citation: synthetic:reference-0001');
    await fireEvent.click(screen.getByRole('button', { name: 'Add Syntheticite trace' })); await flush();
    expect(plot.react.mock.calls.at(-1)![2].meta.traces[1].snapshot.trace.source.citation).toBe('synthetic:reference-0001');
  });

  it('browses a permitted reference independently of identification', async () => {
    appConfig.set(config([true, false, false]));
    render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Browse references' })); await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Add reference' })); await flush();
    expect(plot.react.mock.calls.at(-1)![1]).toHaveLength(2);
    expect(api.identifyStoredPoint).not.toHaveBeenCalled();
  });

  it.each(['source', 'method', 'release', 'subset'])('rejects a stale %s digest', async stale => {
    const changed = result();
    if (stale === 'source') changed.source_digest = '6'.repeat(64);
    if (stale === 'method') changed.method.release_id = '6'.repeat(64);
    if (stale === 'release') changed.reference_release_id = '6'.repeat(64);
    if (stale === 'subset') changed.reference_subset_id = '6'.repeat(64);
    api.identifyStoredPoint.mockResolvedValue(changed);
    render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    expect(screen.getByRole('alert').textContent).toContain('Stale identification rejected');
    expect(screen.queryByText(/support 0.910/)).toBeNull();
  });

  it.each(['selection', 'policy', 'tier'])('discards pending results across a %s change', async change => {
    const pending = deferred<IdentificationEvidence>(); api.identifyStoredPoint.mockReturnValue(pending.promise);
    const { rerender } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    if (change === 'selection') await rerender({ snapshot: input('7'.repeat(64)), contextKey: 'synthetic-context' });
    if (change === 'policy') appConfig.set({ ...config(), evidence_policy_digest: '7'.repeat(64) });
    if (change === 'tier') accessMode.set('public');
    await flush(); pending.resolve(result()); await flush();
    expect(screen.queryByText(/support 0.910/)).toBeNull();
  });

  it('distinguishes scientific abstention from execution failure', async () => {
    api.identifyStoredPoint.mockResolvedValue(result({ status: 'abstained', ranking: [], reasons: ['constant_signal'] }));
    const { container } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    expect(container.textContent).toContain('Scientific abstention'); expect(container.textContent).toContain('constant signal');
    expect(screen.queryByRole('alert')).toBeNull();
    api.identifyStoredPoint.mockRejectedValue(new Error('Synthetic worker failure'));
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    expect(screen.getByRole('alert').textContent).toContain('execution failed');
    expect(container.textContent).not.toContain('Scientific abstention');
  });

  it('exports the same displayed support, citations, arrays and normalization', async () => {
    const { container } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context' });
    await fireEvent.click(screen.getByRole('button', { name: 'Identify' })); await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Add Syntheticite trace' })); await flush();
    await fireEvent.click(container.querySelector('.export-trigger')!);
    await fireEvent.click(screen.getByText('Full provenance JSON')); await flush();
    const exported = JSON.parse(container.querySelector('.provenance-json')!.textContent!);
    expect(exported).toEqual(plot.react.mock.calls.at(-1)![2].meta);
    expect(exported.identification).toEqual(result());
    expect(exported.traces[1].original).toEqual(reference.y);
    expect(exported.traces[1].displayed).toEqual([-.25, .5, 1]);
    expect(exported.traces[1].snapshot.trace.source.reference.provenance.citation_text).toBe('Synthetic citation');
    await fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await fireEvent.click(container.querySelector('.export-trigger')!);
    await fireEvent.click(screen.getByText('Comparison CSV')); await flush();
    const blob = vi.mocked(URL.createObjectURL).mock.calls.at(-1)![0] as Blob;
    const csv = await new Promise<string>(resolve => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });
    expect(JSON.parse(csv.split('\n')[0].slice('# provenance: '.length))).toEqual(exported);
  });
});
