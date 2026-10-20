import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import type { ProcessingSnapshot } from '../lib/types';
import { postBackground, postBaseline } from '../lib/api';
import { clearEvidence, evidenceTrace, pinTrace, setEvidenceContext } from '../lib/evidenceStore';
import EvidenceDesk from './EvidenceDesk.svelte';
import EvidenceRail from './EvidenceRail.svelte';
import type { IdentificationEvidence } from '../lib/identification';

const react = vi.fn((div, _traces, layout) => { div.layout = layout; });
vi.mock('plotly.js-basic-dist-min', () => ({ react, purge: vi.fn(), downloadImage: vi.fn() }));
vi.mock('../lib/api', async importOriginal => ({
  ...await importOriginal<typeof import('../lib/api')>(),
  postDespike: vi.fn(),
  postBackground: vi.fn(async body => ({ schema_version: '1', subtracted: body.intensity.map((v: number) => v - 1),
    background_scaled: body.intensity.map(() => 1), scale_used: body.scale === 'auto' ? body.scan_ppp / 900 : body.scale, bg_type: body.bg_type })),
  postBaseline: vi.fn(async body => ({ schema_version: '1', raw: body.intensity, baseline: body.intensity.map(() => 0),
    corrected: body.intensity.map((v: number) => v * 2), wavenumber: body.wavenumber, params_used: { ...body.params } })),
}));

// Entirely synthetic arrays and identifiers.
function input(index = 0, chain: Record<string, unknown>[] = []): ProcessingSnapshot {
  const x = index ? [810, 910, 1010] : [800, 900, 1000];
  return { stage: chain.length ? 'baseline_corrected' : 'raw', params: {}, processingChain: chain,
    raman: { wavenumber: x, intensity: chain.length ? [1, 3, 5] : [-1, 2, 4] },
    trace: { region: 'R1', wavenumber: x, wavelength: [285, 305, 325], intensity: [-1, 2, 4], channel_indices: [0, 1, 2],
      source: { scan_id: `synthetic-${index}`, scan_name: `Synthetic ${index}` }, selection: { mode: 'point', indices: [index] },
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'synthetic' } } };
}
const baseline = { step: 'baseline', method: 'aspls', lam: 1e6, max_iter: 10 };
const flush = async () => { for (let i = 0; i < 4; i++) { await tick(); await new Promise(r => setTimeout(r, 0)); } };
const settle = async () => { await flush(); await new Promise(r => setTimeout(r, 350)); await flush(); };
const meta = () => react.mock.calls.at(-1)![2].meta;
const plotted = () => react.mock.calls.at(-1)![1].map((trace: { y: (number | null)[] }) => trace.y);

beforeEach(() => {
  clearEvidence(); setEvidenceContext('synthetic-context'); react.mockClear();
  vi.mocked(postBaseline).mockClear(); vi.mocked(postBackground).mockClear();
});

async function desk(primary: ProcessingSnapshot, props: Record<string, unknown> = {}) {
  const view = render(EvidenceDesk, { snapshot: primary, contextKey: 'synthetic-context', ...props });
  await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'overlay' } });
  await flush();
  return view;
}

describe('Evidence desk shared processing chain', () => {
  it('is off by default, applies the primary chain to a pin from its source, and opting out restores today\'s export', async () => {
    const pin = evidenceTrace(input(1))!;
    pinTrace(pin, 'synthetic-context', 900);
    await desk(input(0, [baseline]));
    const today = JSON.stringify(meta());
    expect(meta().schema_version).toBe('pipeline-evidence-1');
    expect(postBaseline).not.toHaveBeenCalled();

    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(postBaseline).toHaveBeenCalledTimes(1);
    expect(vi.mocked(postBaseline).mock.calls[0][0]).toEqual({ wavenumber: [810, 910, 1010], intensity: [-1, 2, 4],
      params: { method: 'aspls', lam: 1e6, max_iter: 10 } });
    expect(meta().schema_version).toBe('pipeline-evidence-2');
    expect(meta().shared_chain.primary_label).toBe(evidenceTrace(input(0, [baseline]))!.label);
    expect(meta().traces[1].processing).toMatchObject({ mode: 'shared', derived_from: pin.id,
      steps: [{ step: 'baseline', status: 'applied' }] });
    expect(meta().traces[1].original).toEqual([-2, 4, 8]);
    expect(screen.getByText(/^Shared chain [0-9a-f]{12} from /)).toBeTruthy();

    await fireEvent.click(screen.getByLabelText(`Own processing: ${pin.label}`));
    await settle();
    expect(JSON.stringify(meta())).toBe(today);
    expect(postBaseline).toHaveBeenCalledTimes(1);
  });

  it('shares nothing when the primary has no committed transform step', async () => {
    pinTrace(evidenceTrace(input(1))!, 'synthetic-context', 900);
    await desk(input(0));
    const today = JSON.stringify(meta());
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(screen.getByText('This trace has no committed despike, background or baseline step to share.')).toBeTruthy();
    expect(JSON.stringify(meta())).toBe(today);
    expect(postBaseline).not.toHaveBeenCalled();
  });

  it('leaves a refused trace out of the plot and lists the reason in the desk and the export', async () => {
    const pin = evidenceTrace(input(1))!;
    pinTrace(pin, 'synthetic-context', 900);
    await desk(input(0, [{ ...baseline, wavenumber_range: [800, 1000] }]));
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    const reason = 'baseline range 800–1000 cm⁻¹ is not covered by this trace\'s axis; never extrapolated';
    expect(screen.getByRole('status').textContent).toBe(`Not shown: ${pin.label}. ${reason}`);
    expect(plotted()).toHaveLength(1);
    expect(meta().shared_chain.refused).toEqual([{ label: pin.label, derived_from: pin.id, reason,
      steps: [{ step: 'baseline', status: 'refused', reason }] }]);
    expect(postBaseline).not.toHaveBeenCalled();
  });

  it('scales an auto background from each pin\'s own shots per point and refuses a pin without one', async () => {
    const withPpp = evidenceTrace(input(1))!;
    const without = evidenceTrace(input(2))!;
    pinTrace(withPpp, 'synthetic-context', 450);
    pinTrace(without, 'synthetic-context');
    await desk(input(0, [{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto', scale_used: 2 }]), { shotsPerPoint: 1800 });
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(postBackground).toHaveBeenCalledTimes(1);
    expect(vi.mocked(postBackground).mock.calls[0][0]).toMatchObject({ scale: 'auto', scan_ppp: 450 });
    expect(meta().shared_chain.steps).toEqual([{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto' }]);
    expect(meta().traces[1].processing.scale_used).toBe(0.5);
    const reason = 'shots per point not recorded for this pin; auto background scale unavailable';
    expect(meta().shared_chain.refused).toEqual([{ label: without.label, derived_from: without.id, reason,
      steps: [{ step: 'bg_subtract', status: 'refused', reason }] }]);
  });

  it('keeps an auto background automatic through undo, even when a manual value equals its scale', async () => {
    pinTrace(evidenceTrace(input(1))!, 'synthetic-context', 450);
    const auto = input(0, [{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto', scale_used: 2 }]);
    const view = await desk(auto, { shotsPerPoint: 1800 });
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(vi.mocked(postBackground).mock.calls.at(-1)![0]).toMatchObject({ scale: 'auto', scan_ppp: 450 });
    // A manual re-apply at the same scale value, then undo back to the auto record.
    await view.component.$set({ snapshot: input(0, [{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'manual', scale_used: 2 }]) });
    await settle();
    expect(vi.mocked(postBackground).mock.calls.at(-1)![0]).toMatchObject({ scale: 2 });
    await view.component.$set({ snapshot: auto });
    await settle();
    expect(meta().shared_chain.steps).toEqual([{ step: 'bg_subtract', bg_type: 'as', scale_mode: 'auto' }]);
    expect(meta().traces[1].processing.scale_used).toBe(0.5);
  });

  it('refuses, without throwing or requesting, a background step stored before its scale mode was recorded', async () => {
    const pin = evidenceTrace(input(1))!;
    pinTrace(pin, 'synthetic-context', 900);
    await desk(input(0, [{ step: 'bg_subtract', bg_type: 'as', scale_used: 2 }, baseline]));
    const today = JSON.stringify(meta());
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(screen.getByText(/Chain not shared: background scale mode not recorded/)).toBeTruthy();
    expect(postBackground).not.toHaveBeenCalled();
    expect(postBaseline).not.toHaveBeenCalled();
    expect(JSON.stringify(meta())).toBe(today);
  });

  it('drops a result from a chain that changed while its request was in flight', async () => {
    let release = () => {};
    vi.mocked(postBaseline).mockImplementationOnce(body => new Promise(resolve => {
      release = () => resolve({ schema_version: '1', raw: body.intensity, baseline: [0, 0, 0], corrected: [100, 100, 100],
        wavenumber: body.wavenumber, params_used: { ...body.params } });
    }));
    pinTrace(evidenceTrace(input(1))!, 'synthetic-context', 900);
    const { component } = await desk(input(0, [baseline]));
    await fireEvent.click(screen.getByLabelText('Apply this trace\'s chain to all'));
    await settle();
    expect(postBaseline).toHaveBeenCalledTimes(1);
    // Hold the new chain's hash open so the old request finishes inside that window.
    const digest = crypto.subtle.digest.bind(crypto.subtle);
    let hashed = () => {};
    const held = vi.spyOn(crypto.subtle, 'digest').mockImplementationOnce((...args) => new Promise(resolve => {
      hashed = () => resolve(digest(...args));
    }));
    component.$set({ snapshot: input(0, [{ ...baseline, lam: 1e5 }]) });
    await flush();
    release();
    await settle();
    expect(JSON.stringify(react.mock.calls.at(-1)![2].meta)).not.toContain('100,100,100');
    expect(plotted()).toHaveLength(1);
    hashed();
    await settle();
    held.mockRestore();
    expect(postBaseline).toHaveBeenCalledTimes(2);
    expect(vi.mocked(postBaseline).mock.calls[1][0].params).toEqual({ method: 'aspls', lam: 1e5, max_iter: 10 });
    expect(meta().traces[1].original).toEqual([-2, 4, 8]);
    expect(meta().traces[1].processing.shared_chain_id).toBe(meta().shared_chain.shared_chain_id);
    expect(meta().shared_chain.steps).toEqual([{ step: 'baseline', params: { method: 'aspls', lam: 1e5, max_iter: 10 } }]);
    expect(JSON.stringify(meta())).not.toContain('100,100,100');
  });
});

it('labels identification scores as unchanged while a shared chain is active', () => {
  const result: IdentificationEvidence = { result_digest: 'a'.repeat(64), source_digest: 'b'.repeat(64), status: 'candidate', reasons: [],
    method: { release_id: 'c'.repeat(64), support_min: .8, status: 'provisional' },
    reference_release_id: 'd'.repeat(64), reference_subset_id: 'e'.repeat(64),
    ranking: [{ template_id: '1'.repeat(64), species_name: 'Syntheticite', support: .91, trace_revision_ids: [], citations: [] }] };
  const { unmount } = render(EvidenceRail, { result, contextKey: 'synthetic-context', sharedActive: true });
  expect(screen.getByText(/support 0\.910 · score from identification prep \(unchanged\)/)).toBeTruthy();
  unmount();
  render(EvidenceRail, { result, contextKey: 'synthetic-context' });
  expect(screen.queryByText(/score from identification prep/)).toBeNull();
});
