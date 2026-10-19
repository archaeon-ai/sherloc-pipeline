import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import type { ProcessingSnapshot } from '../lib/types';
import { clearEvidence, evidenceTrace, pinTrace, setEvidenceContext } from '../lib/evidenceStore';
import EvidenceDesk from './EvidenceDesk.svelte';
import RamanView from './RamanView.svelte';

const react = vi.fn((div, _traces, layout) => { div.layout = layout; });
const downloadImage = vi.fn();
vi.mock('plotly.js-basic-dist-min', () => ({ react, purge: vi.fn(), downloadImage }));

function input(index = 0): ProcessingSnapshot {
  const x = index ? [810, 910, 1010] : [800, 900, 1000];
  return { stage: 'raw', params: {}, processingChain: [], raman: { wavenumber: x, intensity: [-1, 2, 4] },
    trace: { wavenumber: x, wavelength: [285, 305, 325], intensity: [-1, 2, 4], channel_indices: [0, 1, 3],
      source: { scan_id: `synthetic-${index}`, scan_name: `Synthetic ${index}` }, selection: { mode: 'point', indices: [index] },
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'synthetic', region: 'R1' } } };
}
const flush = async () => { for (let i = 0; i < 4; i++) { await tick(); await new Promise(r => setTimeout(r, 0)); } };
const last = () => react.mock.calls.at(-1)!;

beforeEach(() => {
  clearEvidence(); setEvidenceContext('synthetic-context'); react.mockClear(); downloadImage.mockClear();
  URL.createObjectURL = vi.fn(() => 'blob:synthetic'); URL.revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
});

describe('Evidence desk', () => {
  it('provides the four labeled native controls without Identify and pins the committed trace', async () => {
    const { container } = render(EvidenceDesk, { snapshot: input(), contextKey: 'synthetic-context', wavenumber: [800, 900, 1000], intensity: [-1, 2, 4] });
    await flush();
    expect(screen.getByLabelText('View')).toBeTruthy(); expect(screen.getByLabelText('Theme')).toBeTruthy();
    expect(container.querySelector('summary')?.textContent).toBe('Traces');
    expect(screen.queryByText('Identify')).toBeNull();
    const pin = screen.getByRole('button', { name: 'Pin current trace' });
    pin.focus(); expect(document.activeElement).toBe(pin);
    await fireEvent.click(pin); await flush();
    expect(screen.getByLabelText('View')).toHaveValue('overlay');
    expect(last()[1]).toHaveLength(1);
    expect(last()[1][0].y).toEqual([-0.25, 0.5, null, 1]);
  });

  it('keeps native grids across remounts and exports the plotted comparison with per-trace provenance', async () => {
    pinTrace(evidenceTrace(input())!, 'synthetic-context');
    const { container, unmount } = render(EvidenceDesk, { snapshot: input(1), contextKey: 'synthetic-context' });
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'overlay' } }); await flush();
    expect(last()[1].map((trace: { x: number[] }) => trace.x)).toEqual([[810, 910, null, 1010], [800, 900, null, 1000]]);
    expect(last()[2].meta.traces.map((trace: { snapshot: ProcessingSnapshot }) => trace.snapshot.trace!.source.scan_id)).toEqual(['synthetic-1', 'synthetic-0']);
    await fireEvent.click(container.querySelector('.export-trigger')!);
    await fireEvent.click(screen.getByText('SVG + provenance')); await flush();
    expect(downloadImage).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ format: 'svg' }));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
    const text = await new Promise<string>(resolve => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });
    expect(JSON.parse(text)).toEqual(last()[2].meta);
    unmount();
    render(EvidenceDesk, { snapshot: input(1), contextKey: 'synthetic-context' });
    expect(screen.getByRole('button', { name: /Remove Synthetic 0/ })).toBeTruthy();
  });

  it('renders emission-only x, disables Raman artifacts and exposes incompatible domains', async () => {
    const snapshot = input(); snapshot.raman.wavenumber = []; snapshot.trace!.wavenumber = [];
    const { container } = render(EvidenceDesk, { snapshot, contextKey: 'synthetic-context', wavelength: [285, 305, 325], intensity: [-1, 2, 4],
      peaks: [{ center_cm1: 900 }], spikeMask: [true, false, false] });
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'emission' } }); await flush();
    expect(last()[2].xaxis.title.text).toBe('Wavelength (nm)');
    expect(last()[1]).toHaveLength(1);
    expect(last()[1][0].x).toEqual([285, 305, null, 325]);
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'overlay' } }); await flush();
    expect(last()[1]).toEqual([]);
    expect(container.textContent).toContain('No compatible native axis');
    expect(container.querySelector('.export-trigger')).toBeDisabled();
  });

  it('resets same-length zoom across domains and trace identities and applies theme', async () => {
    const snapshot = input();
    const { component, container } = render(RamanView, { primaryTrace: evidenceTrace(snapshot), traceIdentity: 'first', xDomain: 'raman_shift' });
    await flush();
    const fields = container.querySelectorAll('.axis-input');
    await fireEvent.input(fields[0], { target: { value: '820' } });
    await fireEvent.change(fields[1], { target: { value: '980' } }); await flush();
    const firstKey = last()[2].uirevision;
    await component.$set({ xDomain: 'emission_wavelength', theme: 'dark' }); await flush();
    expect(last()[2].uirevision).not.toBe(firstKey);
    expect(last()[2].xaxis.autorange).toBe(true);
    expect(last()[2].paper_bgcolor).toBe('#0f172a');
    const emissionKey = last()[2].uirevision;
    await component.$set({ traceIdentity: 'second' }); await flush();
    expect(last()[2].uirevision).not.toBe(emissionKey);
  });
});


it('exports native processing values and committed labels with a provenance sidecar', async () => {
  const snapshot = input();
  const { container } = render(EvidenceDesk, { snapshot, contextKey: 'synthetic-context',
    wavenumber: [800, 900, 1000], intensity: [-1, 2, 4], selectionMode: 'subset', selectedIndices: [99] });
  await flush();
  expect(container.textContent).toContain('Point 0');
  expect(container.textContent).not.toContain('Points 99');
  expect(last()[1][0].x).toEqual([800, 900, null, 1000]);
  expect(last()[1][0].y).toEqual([-1, 2, null, 4]);
  expect(last()[2].meta.traces[0].display.normalization).toBe('native');
  expect(last()[2].meta.traces[0].displayed).toEqual([-1, 2, 4]);
  await fireEvent.click(container.querySelector('.export-trigger')!);
  await fireEvent.click(screen.getByText('PNG + provenance')); await flush();
  expect(downloadImage).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ format: 'png' }));
  const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
  const text = await new Promise<string>(resolve => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });
  expect(JSON.parse(text)).toEqual(last()[2].meta);
});


it('resets fallback zoom when the native axis changes length', async () => {
  const { component, container } = render(RamanView, { wavenumber: [11, 12, 13], intensity: [2, 3, 4] }); await flush();
  const fields = container.querySelectorAll('.axis-input');
  await fireEvent.input(fields[0], { target: { value: '11' } });
  await fireEvent.change(fields[1], { target: { value: '12' } }); await flush();
  const key = last()[2]._viewKey;
  await component.$set({ wavenumber: [11, 12, 13, 14], intensity: [2, 3, 4, 5] }); await flush();
  expect(last()[2]._viewKey).not.toBe(key); expect(last()[2].xaxis.autorange).toBe(true);
});

it.each(['processing', 'overlay'])('records the plotted primary color for %s in each theme', async view => {
  const snapshot = input();
  render(EvidenceDesk, { snapshot, contextKey: 'synthetic-context', wavenumber: snapshot.raman.wavenumber, intensity: snapshot.raman.intensity });
  await fireEvent.change(screen.getByLabelText('View'), { target: { value: view } });
  for (const [theme, color] of [['dark', '#e2e8f0'], ['light', '#1e293b']]) {
    await fireEvent.change(screen.getByLabelText('Theme'), { target: { value: theme } }); await flush();
    expect(last()[1][0].line.color).toBe(color); expect(last()[2].meta.traces[0].color).toBe(color);
  }
});

it.each([
  ['bg_subtracted', 'raw'], ['bg_subtracted', 'background'], ['bg_subtracted', 'bg (scaled)'],
  ['raman_fitted', 'baseline'], ['raman_fitted', 'fit'], ['raman_fitted', 'residual'],
])('breaks native gaps in %s %s overlays including restricted ranges', async (stage, name) => {
  const snapshot = input();
  render(RamanView, { processingTrace: evidenceTrace(snapshot), wavenumber: snapshot.raman.wavenumber,
    intensity: [2, 3, 4], rawIntensity: [5, 6, 7], background: [1, 2, 3], backgroundScaled: [1, 2, 3],
    baseline: [1, 2, 3], fitCurve: [1, 2, 3], residual: [1, 2, 3], stage,
    baselineRange: [850, 1050], fitRange: [850, 1050], showRaw: true, showResidual: true }); await flush();
  const trace = last()[1].find((item: { name: string }) => item.name === name);
  expect(trace).toBeTruthy(); const gap = trace.x.indexOf(null); expect(gap).toBeGreaterThan(0);
  expect(trace.x.slice(gap - 1, gap + 2)).toEqual([900, null, 1000]); expect(trace.y[gap]).toBeNull();
});


it.each(['R1', 'R2', 'R3'])('omits both-axis %s from the opposite view and figure export', async region => {
  const snapshot = input(); snapshot.trace!.region = region;
  const other = input(1); other.trace!.region = region === 'R1' ? 'R2' : 'R1';
  pinTrace(evidenceTrace(other)!, 'synthetic-context');
  const { container } = render(EvidenceDesk, { snapshot, contextKey: 'synthetic-context' });
  await fireEvent.change(screen.getByLabelText('View'), { target: { value: region === 'R1' ? 'emission' : 'overlay' } }); await flush();
  expect(last()[1]).toHaveLength(1);
  expect(last()[2].meta.traces.map((trace: { snapshot: ProcessingSnapshot }) => trace.snapshot.trace!.region)).toEqual([other.trace!.region]);
  expect(last()[2].meta.traces[0].domain).toBe(region === 'R1' ? 'emission_wavelength' : 'raman_shift');
  await fireEvent.click(container.querySelector('.export-trigger')!);
  await fireEvent.click(screen.getByText('SVG + provenance')); await flush();
  const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
  const text = await new Promise<string>(resolve => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.readAsText(blob); });
  expect(JSON.parse(text)).toEqual(last()[2].meta);
});
