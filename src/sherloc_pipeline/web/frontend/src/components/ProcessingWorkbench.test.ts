// Workbench-level despike fetch semantics — PR #7 review F2 + F3. Locks:
//   [F2] with modz active, every spectrum reload fetches RAW
//        (despike_method 'none'): the backend's server-side modz path
//        live-computes on the served array and exists for API consumers;
//        the Workbench computes modz client-side in the chain step with
//        its tunable params. Sending 'modz' on the fetch would despike
//        twice (server then client) — single application site required.
//   [F3] stale-response guard: a slower older spectrum response must not
//        overwrite a newer one (quick none→ml→none toggling).
//
// jsdom has no Plotly; mock the dynamic import RamanView performs. The api
// module is spied (not vi.mock'd) per the ProcessingChain.test.ts idiom so
// untouched exports (ApiError, AuthRequiredError classes) stay real.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { MockInstance } from 'vitest';
import { render, fireEvent, screen, within } from '@testing-library/svelte';
import { tick } from 'svelte';
import { get } from 'svelte/store';
import * as api from '../lib/api';
import { pointSelection, processingState } from '../lib/processingStore';
import { accessMode, appConfig, scienceAuthorized, currentHash, sessionGeneration, clearSessionState, processingNotice } from '../lib/stores';
import type { FitResponse, ProcessingSnapshot, SpectrumParams, SpectrumResponse } from '../lib/types';
import ProcessingWorkbench from './ProcessingWorkbench.svelte';

// Module-level react spy so the issue-#8 tests can inspect the traces the
// embedded RamanView hands to Plotly (the spike-marker trace in particular).
const reactSpy = vi.fn();
vi.mock('plotly.js-basic-dist-min', () => {
  const stub = { react: reactSpy, purge: vi.fn(), newPlot: vi.fn(), downloadImage: vi.fn() };
  return { default: stub, ...stub };
});

/** The spike-marker trace from the most-recent RamanView Plotly.react call. */
function lastSpikeTrace(): { name?: string; x?: number[] } | undefined {
  for (let i = reactSpy.mock.calls.length - 1; i >= 0; i--) {
    const traces = reactSpy.mock.calls[i][1] as Array<{ name?: string; mode?: string; x?: number[] }>;
    if (!Array.isArray(traces)) continue;
    const t = traces.find((tr) => tr.mode === 'markers' && /Spikes|ML-masked/.test(tr.name ?? ''));
    if (t) return t;
  }
  return undefined;
}

const SCAN_ID = 'scan-uuid-1';
const INTENSITY_A = [10, 12, 15];
const INTENSITY_NEW = [4, 5, 6];
const INTENSITY_STALE = [99, 98, 97];

function fakeScan() {
  return {
    id: SCAN_ID,
    sol_number: 9999,
    target: 'Invented',
    scan_name: 'synthetic',
    scan_id: 'ss__9999_0000000000_000rrs__synthetic',
    n_points: 1,
    n_channels: 3,
    shots_per_point: 50,
    laser_wavelength_nm: 247,
    scan_class: 'primary' as const,
    scan_type: 'detail' as const,
    target_type: 'mars_target' as const,
    data_source: 'loupe',
    site_drive: null,
    sequence_id: null,
    parent_scan_id: null,
    source_scan_ids: null,
    processing_status: 'completed' as const,
    processed_at: null,
    processing_pipeline_version: null,
    has_fitted_peaks: false,
    processing_config_hash: null,
    processing_error: null,
    sclk_start: null,
    sclk_stop: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    colorized_aci_available: false,
    // ML masks stored → the ML selector option is enabled.
    ml_mask_count: 5,
  };
}

function fakePoint(point_index = 0) {
  return {
    id: `p${point_index}`,
    point_index,
    x_pixel: null,
    y_pixel: null,
    x_aci_pixel: null,
    y_aci_pixel: null,
    azimuth_dn: null,
    elevation_dn: null,
    azimuth_error: null,
    elevation_error: null,
    photodiode_mean: null,
    photodiode_std: null,
    coordinate_frame: null,
  };
}

function fakeSpectrum(intensity: number[], overrides: Partial<SpectrumResponse> = {}): SpectrumResponse {
  return {
    schema_version: '1.0.0',
    scan_id: SCAN_ID,
    region: 'R1',
    n_points_averaged: 1,
    effective_trim_pct_per_tail: 0,
    m_trimmed_per_tail: 0,
    baseline_corrected: false,
    wavenumber: [800, 900, 1000],
    intensity,
    n_channels: 3,
    channel_indices: [52, 600, 54],
    provenance: {
      calibration_version: 'loupe_v5.1.5a',
      wavenumber_unit: 'cm-1',
      intensity_unit: 'counts',
    },
    despike_applied: false,
    despike_method: null,
    n_masked_channels: 0,
    despike_missing_regions: [],
    n_uncovered_contributor_channels: 0,
    ...overrides,
  };
}

function fakeFit(): FitResponse {
  return {
    schema_version: '1', peaks: [], n_peaks: 0, r_squared: 0.8,
    model_selection_method: 'aicc', residual: [1, 1, 1], baseline: [0, 0, 0],
    corrected: [7, 8, 9], wavenumber: [800, 900, 1000],
    provenance: { params_used: {}, config_hash: 'synthetic-fit-hash',
      calibration_version: 'synthetic-calibration', wavenumber_unit: 'cm-1', intensity_unit: 'synthetic-unit' },
  };
}

describe('ProcessingWorkbench processing snapshot identity', () => {
  it.each([false, true])('carries fit-response provenance through the store and both exports (unknown=%s)', async (unknown) => {
    const fit = fakeFit();
    if (unknown) {
      fit.provenance.calibration_version = null;
      fit.provenance.intensity_unit = null;
    }
    vi.spyOn(api, 'postFit').mockResolvedValue(fit);
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    const card = Array.from(container.querySelectorAll('.step-title'))
      .find(el => el.textContent?.trim() === '4. Peak Fit')!.closest('.step-card')!;
    await fireEvent.click(card.querySelector('.step-header')!);
    await fireEvent.click(card.querySelector('.btn-primary')!);
    await flush();
    const record = get(processingState)?.processingChain?.at(-1);
    expect(record?.provenance).toEqual(fit.provenance);
    const metadata = csvProvenance(await exportCsv(container));
    expect(metadata.user_processing_chain.at(-1).provenance).toEqual(fit.provenance);
    await fireEvent.click(container.querySelector('.export-trigger')!);
    await fireEvent.click(screen.getByText('Full provenance JSON'));
    expect(JSON.parse(container.querySelector('.provenance-json')!.textContent!)).toEqual(metadata);
  });

  it.each(['baseline', 'fit'])('drops %s A when spectrum B commits before the chain reset flushes', async (step) => {
    vi.spyOn(api, 'getScan').mockResolvedValue({
      schema_version: '1.0.0', scan: { ...fakeScan(), ml_mask_count: 0 },
    });
    let resolveProcessing = () => {};
    if (step === 'baseline') {
      vi.spyOn(api, 'postBaseline').mockReturnValue(new Promise((resolve) => {
        resolveProcessing = () => resolve({ schema_version: '1', raw: INTENSITY_A,
          wavenumber: [800, 900, 1000], corrected: [7, 8, 9], baseline: [3, 4, 6],
          params_used: { method: 'aspls', lam: 123, max_iter: 2 } });
      }));
    } else {
      vi.spyOn(api, 'postFit').mockReturnValue(new Promise((resolve) => {
        resolveProcessing = () => resolve(fakeFit());
      }));
    }
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    const title = step === 'baseline' ? '3. Baseline' : '4. Peak Fit';
    const card = Array.from(container.querySelectorAll('.step-title'))
      .find(el => el.textContent?.trim() === title)!.closest('.step-card')!;
    if (step === 'baseline') {
      await fireEvent.click(card.querySelector('input[type="checkbox"]')!);
    } else {
      await fireEvent.click(card.querySelector('.step-header')!);
      await fireEvent.click(card.querySelector('.btn-primary')!);
    }
    expect(step === 'baseline' ? api.postBaseline : api.postFit).toHaveBeenCalledTimes(1);
    let resolveSpectrum: (value: SpectrumResponse) => void = () => {};
    getAvg.mockReturnValueOnce(new Promise(resolve => { resolveSpectrum = resolve; }));
    await fireEvent.change(container.querySelector('.averaging-controls select')!, { target: { value: 'mean' } });
    const committed: Array<ProcessingSnapshot | null> = [];
    const unsubscribe = processingState.subscribe(state => committed.push(state));
    // Both continuations queue before Svelte's flush: B commits, then A emits.
    resolveSpectrum(fakeSpectrum(INTENSITY_NEW, { provenance: {
      source_digest: 'synthetic-B', calibration_version: null, wavenumber_unit: 'cm-1', intensity_unit: null,
    } }));
    resolveProcessing();
    await flush();
    unsubscribe();
    expect(committed.some(state => state?.trace?.provenance.source_digest === 'synthetic-B'
      && state.stage !== 'raw')).toBe(false);
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
    expect(get(processingState)?.processingChain).toEqual([]);
    expect(await exportCsv(container)).toContain('synthetic-B');
  });
});

function flush(): Promise<void> {
  return new Promise<void>((r) => setTimeout(r, 0)).then(() => tick());
}

// Locate the despike step card by its title and expand its body.
async function expandDespikePanel(container: HTMLElement): Promise<HTMLElement> {
  const titles = Array.from(container.querySelectorAll<HTMLElement>('.step-title'));
  const match = titles.find((t) => t.textContent?.trim() === '1. Despike');
  expect(match, 'no step card titled "1. Despike"').toBeDefined();
  const card = match!.closest('.step-card') as HTMLElement;
  await fireEvent.click(card.querySelector<HTMLButtonElement>('.step-header')!);
  await tick();
  return card;
}

function getMethodRadio(card: HTMLElement, value: 'none' | 'ml' | 'modz'): HTMLInputElement {
  const radio = card.querySelector<HTMLInputElement>(
    `input[name="despike-method"][value="${value}"]`,
  );
  expect(radio, `no despike-method radio for value=${value}`).not.toBeNull();
  return radio as HTMLInputElement;
}

// Concrete spy signature — the loose `ReturnType<typeof vi.spyOn>` resolves
// to MockInstance<unknown[], unknown>, which the concrete spy is not
// assignable to under strict TS (same trap noted in AciViewer.test.ts).
let getAvg: MockInstance<[scanId: string, params?: SpectrumParams], Promise<SpectrumResponse>>;
let originalGetContext: typeof HTMLCanvasElement.prototype.getContext;

beforeEach(() => {
  processingNotice.set(null);
  appConfig.set({ tier: 'team', capabilities: { references: false } } as import('../lib/types').ConfigResponse);
  vi.restoreAllMocks();
  reactSpy.mockClear();  // vi.mock factory spy survives restoreAllMocks
  // AciViewer calls canvas.getContext on mount; jsdom logs a noisy
  // "Not implemented" error for it. Return null — AciViewer's render path
  // early-returns on a null context (same precedent as AciViewer.test.ts).
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => null,
  ) as typeof HTMLCanvasElement.prototype.getContext;
  vi.spyOn(api, 'getScan').mockResolvedValue({ schema_version: '1.0.0', scan: fakeScan() });
  vi.spyOn(api, 'getScanPoints').mockResolvedValue({
    schema_version: '1.0.0',
    scan_id: SCAN_ID,
    points: [fakePoint()],
    n_points: 1,
  });
  getAvg = vi.spyOn(api, 'getAverageSpectrum').mockResolvedValue(fakeSpectrum(INTENSITY_A));
  // AciViewer mounts inside the workbench; reject its image fetch with the
  // auth placeholder path so no real network/auth machinery runs.
  vi.spyOn(api, 'fetchAciImage').mockRejectedValue(new api.AuthRequiredError());
  vi.spyOn(api, 'postDespike').mockResolvedValue({
    schema_version: '1',
    despiked: [9, 11, 14],
    spike_mask: [false, true, false],
    n_spikes: 1,
    params_used: { window_size: 7, zscore_threshold: 6, max_iterations: 1, sulfate_guard: true },
  });
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = originalGetContext;
  processingNotice.set(null);
});

it('shows a temporary retry notice as a status beside the error area', async () => {
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await flush();
  processingNotice.set({ message: 'Server busy, retrying in 5 s' });
  await tick();
  expect(screen.getByRole('status')).toHaveTextContent('Server busy, retrying in 5 s');
  expect(container.querySelector('.error-message')).toBeNull();
  processingNotice.set(null);
  await tick();
  expect(screen.queryByRole('status')).toBeNull();
});

describe('ProcessingWorkbench references panel', () => {
  it('names the background step and cites its sources as published', async () => {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'References' }));
    await tick();
    const sections = Array.from(container.querySelectorAll('.ref-section')).map((s) => s.textContent ?? '');
    const background = sections.find((t) => t.includes('2. Background subtraction'));
    expect(background).toBeDefined();
    // The instrument-background step is not dark frame subtraction (active minus dark).
    expect(background).not.toMatch(/dark frame/i);
    // The step names this pipeline's correction and the paper's separately.
    expect(background).toContain('This pipeline’s stowed-arm correction subtracts a trimmed-mean reference.');
    expect(background).toContain('the number of shots per spectrum divided by 900');
    expect(background).toContain('the median of all stowed-arm spectra with a scale factor for each pulses-per-point setting');
    expect(background).toContain('Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals (SHERLOC) Spectrometer Onboard the Perseverance Rover');
    const cites = Array.from(container.querySelectorAll('.ref-cite')).map((c) => c.textContent ?? '');
    const loupe = cites.find((t) => t.includes('nasa/Loupe'));
    expect(loupe).toContain('Uckert, K. (2022). nasa/Loupe: LoupeV5.1.5 (version LoupeV5.1.5a). Zenodo.');
    expect(loupe).not.toContain('et al.');
    const baseline = sections.find((t) => t.includes('pybaselines'));
    expect(baseline).not.toMatch(/Erb, 20\d\d/);
  });
});

describe('ProcessingWorkbench deep links', () => {
  it('loads the requested subset before any average and updates selection state', async () => {
    vi.mocked(api.getScanPoints).mockResolvedValue({
      schema_version: '1.0.0', scan_id: SCAN_ID,
      points: [fakePoint(0), fakePoint(7)], n_points: 2,
    });
    const subset = vi.spyOn(api, 'postSubsetAverage').mockResolvedValue(fakeSpectrum(INTENSITY_A));
    render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { points: '7,0,7' } });
    await flush();
    expect(subset).toHaveBeenCalledWith(SCAN_ID, expect.objectContaining({ point_indices: [7, 0] }));
    expect(getAvg).not.toHaveBeenCalled();
    expect(get(pointSelection)).toEqual({ mode: 'subset', indices: [7, 0], pointIdx: undefined });
  });

  it('loads point zero directly', async () => {
    const point = vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(fakeSpectrum(INTENSITY_A));
    render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } });
    await flush();
    expect(point).toHaveBeenCalledWith(SCAN_ID, 0, expect.anything());
    expect(getAvg).not.toHaveBeenCalled();
  });

  it.each<Record<string, string>>([{ point: '1' }, { points: '0,7' }, { point: '0tail' }, { points: '' }])(
    'shows an error without fetching a spectrum for %j', async (queryParams) => {
      const point = vi.spyOn(api, 'getPointSpectrum');
      const subset = vi.spyOn(api, 'postSubsetAverage');
      render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams });
      await flush();
      expect(screen.getByText(/Point index/)).toBeInTheDocument();
      expect(getAvg).not.toHaveBeenCalled();
      expect(point).not.toHaveBeenCalled();
      expect(subset).not.toHaveBeenCalled();
    },
  );
});

describe('ProcessingWorkbench despike fetch semantics (PR #7 F2/F3)', () => {
  it('with modz active, a spectrum reload fetches despike_method "none" (single application site)', async () => {
    // modz semantics are orthogonal to the ML-default change. Use a scan with
    // no ML masks so the Workbench initialises to 'none' by default (the
    // applyDespikeDefault fallback path), keeping the assertion on mount-fetch
    // despike_method unchanged.
    vi.spyOn(api, 'getScan').mockResolvedValue({
      schema_version: '1.0.0',
      scan: { ...fakeScan(), ml_mask_count: 0 },
    });
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Mount settles with 2 identical raw fetches: loadScan's loadSpectrum
    // plus RamanFitStep's mount-time domain-defaults onRegionSwitch('R1')
    // (pre-existing behavior; the F3 generation guard makes the overlap
    // harmless). Both must carry despike_method 'none'.
    const mountCalls = getAvg.mock.calls.length;
    expect(mountCalls).toBeGreaterThanOrEqual(1);
    for (const call of getAvg.mock.calls) {
      expect(call[1]).toEqual(expect.objectContaining({ despike_method: 'none' }));
    }

    // Select modz: no refetch (raw is already what modz needs).
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'modz'));
    await flush();
    expect(getAvg).toHaveBeenCalledTimes(mountCalls);
    // The client compute ran instead (F1 kick).
    expect(api.postDespike).toHaveBeenCalledTimes(1);

    // Reload with modz active (averaging method change) → the fetch must be
    // RAW: despike_method 'none', never 'modz' (the server would otherwise
    // live-compute modz and the chain would despike a second time).
    const select = container.querySelector<HTMLSelectElement>('.averaging-controls select')!;
    await fireEvent.change(select, { target: { value: 'mean' } });
    await flush();
    expect(getAvg).toHaveBeenCalledTimes(mountCalls + 1);
    expect(getAvg).toHaveBeenLastCalledWith(
      SCAN_ID,
      expect.objectContaining({ despike_method: 'none', averaging_method: 'mean' }),
    );
  });

  it('drops a stale slower spectrum response (ml→none→ml race, F3 generation guard)', async () => {
    // Default is 'ml' (fakeScan ml_mask_count=5). The race is driven via a
    // ml→none→ml sequence; the stale-drop assertion is identical to the
    // pre-A7 none→ml→none version — the generation guard is symmetric.
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Let the mount fetches (loadScan + RamanFitStep domain-defaults region
    // switch) fully settle before queueing the hand-controlled promises.
    const mountCalls = getAvg.mock.calls.length;

    // Hand-controlled promises: the none fetch resolves LAST (slow), the
    // follow-up ml fetch resolves FIRST (fast).
    let resolveNone: (v: SpectrumResponse) => void = () => {};
    const pendingNone = new Promise<SpectrumResponse>((r) => { resolveNone = r; });
    let resolveMl: (v: SpectrumResponse) => void = () => {};
    const pendingMl = new Promise<SpectrumResponse>((r) => { resolveMl = r; });
    getAvg.mockReturnValueOnce(pendingNone).mockReturnValueOnce(pendingMl);

    const card = await expandDespikePanel(container);
    // ml→none: crosses the ML boundary → refetch (suspends on pendingNone).
    await fireEvent.change(getMethodRadio(card, 'none'));
    await tick();
    expect(getAvg).toHaveBeenCalledTimes(mountCalls + 1);
    expect(getAvg).toHaveBeenNthCalledWith(
      mountCalls + 1, SCAN_ID, expect.objectContaining({ despike_method: 'none' }),
    );
    // none→ml: crosses again → second refetch (suspends on pendingMl).
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await tick();
    expect(getAvg).toHaveBeenCalledTimes(mountCalls + 2);
    expect(getAvg).toHaveBeenNthCalledWith(
      mountCalls + 2, SCAN_ID, expect.objectContaining({ despike_method: 'ml' }),
    );

    // Newer (ml) response lands first and commits.
    resolveMl(fakeSpectrum(INTENSITY_NEW));
    await flush();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);

    // Older (none) response lands last — the generation guard must drop it.
    resolveNone(fakeSpectrum(INTENSITY_STALE));
    await flush();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  });

  // PR #7 review F5: a superseded request that FAILS after a newer one
  // succeeded must not surface an error UI (or disturb the newer spectrum).
  it('does not surface an error when a stale superseded request rejects (F5)', async () => {
    // Default is 'ml' (fakeScan ml_mask_count=5). Drive the ml→none→ml
    // sequence: the ml→none fetch is the one that will REJECT (stale);
    // the none→ml fetch succeeds. Symmetric to the pre-A7 none→ml→none case.
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    // Older (ml→none) request will REJECT; newer (none→ml) request succeeds.
    let rejectNone: (e: unknown) => void = () => {};
    const pendingNone = new Promise<SpectrumResponse>((_resolve, reject) => {
      rejectNone = reject;
    });
    let resolveMl: (v: SpectrumResponse) => void = () => {};
    const pendingMl = new Promise<SpectrumResponse>((r) => { resolveMl = r; });
    getAvg.mockReturnValueOnce(pendingNone).mockReturnValueOnce(pendingMl);

    const card = await expandDespikePanel(container);
    // ml→none: refetch suspends on the (doomed) pendingNone.
    await fireEvent.change(getMethodRadio(card, 'none'));
    await tick();
    // none→ml: newer refetch suspends on pendingMl.
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await tick();

    // Newer request succeeds and commits.
    resolveMl(fakeSpectrum(INTENSITY_NEW));
    await flush();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);

    // Stale request now fails with a non-404 — without the F5 guard this
    // rethrows into reloadAndReset and renders the error banner for a
    // request the user has already superseded.
    rejectNone(new api.ApiError(500, 'InternalServerError', 'backend exploded'));
    await flush();
    expect(container.querySelector('.error-message')).toBeNull();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  });
});

// --- Shared CSV-export observables (issue #12 / PR #13 F1) ------------------

/** Export the CSV and return its captured text (Blob/URL/anchor stubbed). */
async function exportCsv(container: HTMLElement): Promise<string> {
  let capturedCsv = '';
  vi.stubGlobal(
    'Blob',
    class {
      constructor(parts: string[]) {
        capturedCsv = parts.join('');
      }
    },
  );
  (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => 'blob:mock');
  (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  try {
    const trigger = container.querySelector<HTMLButtonElement>('.export-trigger');
    expect(trigger, 'no Export trigger').not.toBeNull();
    await fireEvent.click(trigger!);
    await tick();
    await fireEvent.click(screen.getByText('CSV at current stage'));
    await tick();
    expect(capturedCsv).not.toBe('');
    return capturedCsv;
  } finally {
    vi.unstubAllGlobals();
  }
}

function csvHasSpikeMaskColumn(csv: string): boolean {
  const header = csv.split('\n').find((l) => l.startsWith('wavenumber'));
  return !!header && header.split(',').includes('spike_mask');
}

// Issue #8 + #12: ML despike markers. In ML mode the fetched array is already
// server-despiked (stage 'raw') and the server returns masked_positions
// (served-array indices). The Workbench builds the full-length boolean mask
// and hands RamanView the "ML-masked" markers — but ONLY on a single-point
// view (issue #12). On aggregate (average/subset) views the union mask spans
// many points and the per-channel markers are misleading, so they are
// suppressed (see the issue-#12 describe block below). The existing Spikes
// overlay toggle governs the point-view markers, with no new toggle and no
// client-side modz double-run.
describe('ProcessingWorkbench ML despike markers (issue #8)', () => {
  it('renders "ML-masked" markers from masked_positions in ML mode (point view)', async () => {
    // Point view: issue #12 keeps ML markers exact on a single point (the
    // served masked_positions ARE the channels repaired on this trace). Drive
    // point mode via the `point` query param so getPointSpectrum is the fetch.
    // mockResolvedValue (not Once): every point fetch — mount, region-switch,
    // ML refetch — must commit deterministically (the gate is commit-bound
    // per PR #13 F1, so the assertion of record is the post-refetch commit).
    vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 2,
        masked_positions: [0, 2],
      }),
    );
    const { container } = render(ProcessingWorkbench, {
      scanId: SCAN_ID,
      queryParams: { point: '0' },
    });
    await screen.findByText('1. Despike');
    await flush();

    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await flush();

    const trace = lastSpikeTrace();
    expect(trace, 'expected an ML spike-marker trace').toBeDefined();
    expect(trace?.name).toBe('ML-masked (2)');
    // Markers land at the served wavenumbers of positions 0 and 2.
    expect(trace?.x).toEqual([800, 1000]);
    // modz client compute never ran (no double-despike).
    expect(api.postDespike).not.toHaveBeenCalled();
  });

  it('renders no marker trace in ML mode when masked_positions is empty (point view)', async () => {
    vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 0,
        masked_positions: [],
      }),
    );
    const { container } = render(ProcessingWorkbench, {
      scanId: SCAN_ID,
      queryParams: { point: '0' },
    });
    await screen.findByText('1. Despike');
    await flush();
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await flush();
    expect(lastSpikeTrace()).toBeUndefined();
  });

  it('does not render ML markers in none mode (markers gated on method)', async () => {
    // No ML masks stored → applyDespikeDefault falls back to 'none'.
    vi.spyOn(api, 'getScan').mockResolvedValue({
      schema_version: '1.0.0',
      scan: { ...fakeScan(), ml_mask_count: 0 },
    });
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Default is 'none' (no masks) with no masked_positions → no marker trace.
    expect(lastSpikeTrace()).toBeUndefined();
    // The card exists; the method is none by default — sanity that the panel
    // mounted (the marker absence is the assertion of record).
    await expandDespikePanel(container);
  });

  // PR #10 review F1 (Major) regression: a modz artifact can OUTLIVE the
  // method. Apply modz → switch to 'none' (the method change emits a raw
  // state, pushing the despiked snapshot — artifacts.spikeMask included —
  // onto the undo stack) → undo restores that snapshot while the active
  // method is 'none'. The strict three-way gate must yield null/null: no
  // marker trace reaches RamanView, and the CSV export carries neither the
  // spike_mask column nor the provenance header.
  it('undo-restored modz artifact renders no markers and exports no spike_mask while method is none (F1)', async () => {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    // Apply modz (client compute; postDespike mock returns mask [f,t,f]).
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'modz'));
    await flush();
    // Sanity: modz markers render while the method is modz.
    expect(lastSpikeTrace()?.name).toBe('Spikes (1)');
    expect(get(processingState)?.stage).toBe('despiked');

    // Switch to 'none': chain emits a raw state; the workbench pushes the
    // despiked snapshot (with artifacts.spikeMask) onto the undo stack.
    await fireEvent.change(getMethodRadio(card, 'none'));
    await flush();

    // Undo restores the despiked snapshot — stale modz artifact, method none.
    reactSpy.mockClear(); // only post-undo renders count below
    const undoBtn = Array.from(container.querySelectorAll<HTMLButtonElement>('.undo-controls button'))
      .find((b) => b.textContent?.trim() === 'Undo');
    expect(undoBtn, 'no Undo button').toBeDefined();
    expect(undoBtn!.disabled).toBe(false);
    await fireEvent.click(undoBtn!);
    await flush();

    // The stale artifact is back in state...
    expect(get(processingState)?.stage).toBe('despiked');
    expect(get(processingState)?.artifacts?.spikeMask).toBeDefined();
    // ...but the strict gate keeps it off the plot: RamanView re-rendered
    // with NO spike-marker trace.
    expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
    expect(lastSpikeTrace()).toBeUndefined();

    // CSV export: no spike_mask column, no provenance header.
    let capturedCsv = '';
    vi.stubGlobal(
      'Blob',
      class {
        constructor(parts: string[]) {
          capturedCsv = parts.join('');
        }
      },
    );
    (URL as unknown as { createObjectURL: unknown }).createObjectURL = vi.fn(() => 'blob:mock');
    (URL as unknown as { revokeObjectURL: unknown }).revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    try {
      const trigger = container.querySelector<HTMLButtonElement>('.export-trigger');
      expect(trigger, 'no Export trigger').not.toBeNull();
      await fireEvent.click(trigger!);
      await tick();
      await fireEvent.click(screen.getByText('CSV at current stage'));
      await tick();
      expect(capturedCsv).not.toBe('');
      expect(capturedCsv).not.toContain('spike_mask');
      expect(capturedCsv).not.toContain('spike_mask_method');
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

// Issue #12: aggregate-view ML marker semantics. The ML mask is a UNION of
// per-point masked_positions; on a single point it is exactly the channels
// repaired on the displayed trace (markers true), but on an average/subset
// view it spans ~N contributing points — each on-trace marker would falsely
// assert "removal happened here" on a mean/trim-mean trace where any one
// point's repair contributes ~1/N (and trim-mean often already dropped the
// outlier). So on aggregate ML views: NO marker trace, NO CSV spike_mask
// column / method header, and DespikeStep shows the aggregate-count copy.
// modz is unaffected (it computes on the displayed aggregate trace, so its
// markers are exact for every view — re-asserted here as a control).
describe('ProcessingWorkbench aggregate ML marker semantics (issue #12)', () => {
  // FLIPPED FROM PR #10 (issue #12). The original PR #10 expectation —
  // "renders ML-masked markers from masked_positions in ML mode" — ran in the
  // DEFAULT (average) view and asserted markers DID render. That is the exact
  // defect #12 fixes: on an aggregate view the union markers are misleading.
  // The marker-rendering assertion now lives in the issue-#8 block under a
  // POINT view; here we assert the opposite for the aggregate (average) case.
  it('ML + average view: no marker trace, no CSV spike_mask column, aggregate provenance copy', async () => {
    // With the ML-default change the mount fetch already uses despike_method=
    // 'ml'. Pre-configure getAvg so the very first committed spectrum is an
    // aggregate ML response (despike_applied=true, n_masked_channels=247).
    // No explicit ML switch is needed — the default is already 'ml'.
    getAvg.mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 247,
        masked_positions: [0, 2],
      }),
    );
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    const card = await expandDespikePanel(container);
    await flush();

    // No on-trace markers despite masked_positions being non-empty.
    expect(lastSpikeTrace()).toBeUndefined();

    // DespikeStep shows the aggregate-count copy, NOT per-position provenance.
    const note = card.querySelector('[data-testid="ml-aggregate-note"]');
    expect(note, 'expected the aggregate provenance note').not.toBeNull();
    expect(note!.textContent).toMatch(
      /Masks applied on 247 channels across the contributing points — select a single point to see positions\./,
    );

    // CSV: no spike_mask column and no method header on the aggregate view.
    const csv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(csv)).toBe(false);
    expect(csv).not.toContain('spike_mask_method');
  });

  it('ML + point view: markers render and CSV carries the spike_mask column (unchanged)', async () => {
    vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 2,
        masked_positions: [0, 2],
      }),
    );
    const { container } = render(ProcessingWorkbench, {
      scanId: SCAN_ID,
      queryParams: { point: '0' },
    });
    await screen.findByText('1. Despike');
    await flush();
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await flush();

    // Markers present (point view is exact).
    const trace = lastSpikeTrace();
    expect(trace?.name).toBe('ML-masked (2)');
    expect(trace?.x).toEqual([800, 1000]);
    // The point view shows per-position provenance, not the aggregate copy.
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).toBeNull();

    // CSV: spike_mask column present with the method header.
    const csv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(csv)).toBe(true);
    expect(csv).toContain('# spike_mask_method: ml');
  });

  it('ML + subset view: no marker trace and no CSV spike_mask column (same as average)', async () => {
    // 3-point scan so "Select All" lands in subset mode.
    vi.spyOn(api, 'getScan').mockResolvedValue({
      schema_version: '1.0.0',
      scan: { ...fakeScan(), n_points: 3 },
    });
    vi.spyOn(api, 'getScanPoints').mockResolvedValue({
      schema_version: '1.0.0',
      scan_id: SCAN_ID,
      points: [fakePoint(0), fakePoint(1), fakePoint(2)],
      n_points: 3,
    });
    const subsetSpy = vi.spyOn(api, 'postSubsetAverage').mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 19,
        masked_positions: [0, 2],
      }),
    );

    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    // Enter subset mode: "Select All" with a multi-point scan → subset.
    await fireEvent.click(screen.getByLabelText(/Select All/i));
    await flush();
    // Select ML.
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'ml'));
    await flush();
    expect(subsetSpy).toHaveBeenCalled();

    // Aggregate semantics: no markers, aggregate copy, no CSV column.
    expect(lastSpikeTrace()).toBeUndefined();
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).not.toBeNull();
    const csv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(csv)).toBe(false);
    expect(csv).not.toContain('spike_mask_method');
  });

  it('modz + average view: markers and CSV spike_mask column unchanged (control)', async () => {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    // Default average view; select modz (client compute on the displayed
    // aggregate trace — mask [false,true,false] from the beforeEach mock).
    const card = await expandDespikePanel(container);
    await fireEvent.change(getMethodRadio(card, 'modz'));
    await flush();

    // modz markers render on the aggregate trace (exact — computed on what is
    // shown), and there is no ML aggregate note (that copy is ML-only).
    expect(lastSpikeTrace()?.name).toBe('Spikes (1)');
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).toBeNull();

    // CSV: spike_mask column present with modz method header.
    const csv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(csv)).toBe(true);
    expect(csv).toContain('# spike_mask_method: modz');
  });
});

// PR #13 review F1 (Major): the ML marker/export gate must key on the
// selection context OF THE COMMITTED spectrum, not the live selectionMode
// flag. onPointSelect flips selectionMode to 'point' BEFORE the point refetch
// commits (reloadAndReset is fire-and-forget), so a live-flag gate would let
// the still-committed AGGREGATE spectrum's union masked_positions render and
// export as a point-view mask during the pending window — or indefinitely if
// the refetch rejects. These tests drive both windows with hand-controlled
// getPointSpectrum responses from a committed aggregate ML view whose union
// mask is non-empty.
describe('ProcessingWorkbench commit-bound ML gating (PR #13 F1)', () => {
  /** Mount in average view with ML as the default, committing an aggregate ML
   *  spectrum with a NON-EMPTY union mask. Returns the container and despike card. */
  async function mountCommittedAggregateMl(): Promise<{
    container: HTMLElement;
    card: HTMLElement;
  }> {
    // With the ML-default change the mount fetch already uses 'ml'. Pre-configure
    // getAvg so the initial committed spectrum is an aggregate ML response
    // (despike_applied=true, n_masked_channels=247) — no explicit ML switch needed.
    getAvg.mockResolvedValue(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 247,
        masked_positions: [0, 2],
      }),
    );
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Default is 'ml'; mount fetch committed the aggregate ML spectrum.
    const card = await expandDespikePanel(container);
    await flush();
    // Sanity: aggregate ML committed — union mask present in state, markers
    // suppressed (issue #12), aggregate note showing.
    expect(lastSpikeTrace()).toBeUndefined();
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).not.toBeNull();
    return { container, card };
  }

  /** Click the point-grid chip for point index 0 (live mode flips to 'point'
   *  synchronously; the point refetch is whatever getPointSpectrum returns). */
  async function clickPointChip(container: HTMLElement): Promise<void> {
    const chip = Array.from(
      container.querySelectorAll<HTMLButtonElement>('.point-chip'),
    ).find((b) => b.textContent?.trim() === '0');
    expect(chip, 'no point chip for index 0').toBeDefined();
    await fireEvent.click(chip!);
    await flush();
  }

  it('pending aggregate→point switch: union mask stays suppressed until the point response commits', async () => {
    const { container, card } = await mountCommittedAggregateMl();

    // Point fetch suspends: the live flag flips to 'point' while the
    // committed spectrum is still the aggregate one.
    let resolvePoint!: (v: SpectrumResponse) => void;
    const pendingPoint = new Promise<SpectrumResponse>((r) => {
      resolvePoint = r;
    });
    const getPoint = vi.spyOn(api, 'getPointSpectrum').mockReturnValue(pendingPoint);

    reactSpy.mockClear(); // only renders from the transition onward count
    await clickPointChip(container);
    expect(getPoint).toHaveBeenCalledWith(
      SCAN_ID,
      0,
      expect.objectContaining({ despike_method: 'ml' }),
    );

    // PENDING WINDOW: the committed spectrum is aggregate, so no marker trace
    // may render from its union masked_positions despite the live 'point'
    // flag (a live-flag gate would emit 'ML-masked (2)' here). RamanView DID
    // re-render in the window (selection props changed), so the absence is a
    // real observation, not a vacuous one.
    expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
    expect(lastSpikeTrace()).toBeUndefined();
    // ...the provenance area still shows the aggregate copy (commit-bound)...
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).not.toBeNull();
    // ...and the CSV (which exports the committed aggregate trace) carries
    // neither the spike_mask column nor the method header.
    const pendingCsv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(pendingCsv)).toBe(false);
    expect(pendingCsv).not.toContain('spike_mask_method');

    // The matching point response COMMITS → markers become eligible.
    resolvePoint(
      fakeSpectrum([5, 6, 7], {
        despike_applied: true,
        despike_method: 'ml_v1.3_tau_matched',
        n_masked_channels: 2,
        masked_positions: [0, 2],
      }),
    );
    await flush();
    const trace = lastSpikeTrace();
    expect(trace, 'expected ML markers after the point commit').toBeDefined();
    expect(trace?.name).toBe('ML-masked (2)');
    expect(trace?.x).toEqual([800, 1000]);
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).toBeNull();
    const committedCsv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(committedCsv)).toBe(true);
    expect(committedCsv).toContain('# spike_mask_method: ml');
  });

  it('rejected aggregate→point switch: union mask never leaks as a point-view mask', async () => {
    const { container, card } = await mountCommittedAggregateMl();

    // Point fetch REJECTS (non-404): the live flag stays 'point' indefinitely
    // while the committed spectrum remains the aggregate one.
    vi.spyOn(api, 'getPointSpectrum').mockRejectedValue(
      new api.ApiError(500, 'InternalServerError', 'backend exploded'),
    );

    reactSpy.mockClear();
    await clickPointChip(container);

    // The failure IS surfaced (current request, not a superseded one — the
    // PR #7 F5 stale-failure guard does not apply here)...
    expect(container.querySelector('.error-message')).not.toBeNull();
    // ...but marker/export eligibility stays bound to the committed aggregate
    // spectrum: no markers (re-render confirmed non-vacuous), aggregate
    // provenance copy, no CSV column/header.
    expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
    expect(lastSpikeTrace()).toBeUndefined();
    expect(card.querySelector('[data-testid="ml-aggregate-note"]')).not.toBeNull();
    const csv = await exportCsv(container);
    expect(csvHasSpikeMaskColumn(csv)).toBe(false);
    expect(csv).not.toContain('spike_mask_method');
  });
});

describe('Workbench anomalous pixel removal', () => {
  async function openSpectrum() {
    // Channel 160 is in the published table. Supply its entry if the view requests it.
    getAvg.mockResolvedValue(fakeSpectrum(INTENSITY_A, { channel_indices: [159, 160, 161] }));
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({
      schema_version: '2.0.0', region: 'R123', n_channels: 2148,
      anomalous_pixels: [{ channel: 160, position: 160, table: 'S3' }],
    }));
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    const grid = await screen.findByRole('checkbox', { name: 'Grid' });
    await flush();
    return { container, toolbar: within(grid.closest<HTMLElement>('.overlay-toggles')!) };
  }

  it('omits the anomalous pixels checkbox from the spectrum toolbar', async () => {
    const { toolbar } = await openSpectrum();
    expect(toolbar.queryByRole('checkbox', { name: /anomalous pixels/i })).toBeNull();
  });

  it('omits the anomalous pixel column and source line when exporting with all overlays enabled', async () => {
    const { container, toolbar } = await openSpectrum();
    for (const checkbox of toolbar.queryAllByRole('checkbox', { checked: false })) {
      await fireEvent.click(checkbox);
    }
    const csv = await exportCsv(container);
    const columns = csv.split('\n').find(line => line.startsWith('wavenumber,'))!.split(',');
    expect(columns).toContain('intensity');
    expect.soft(columns).not.toContain('anomalous_pixel');
    expect(csv).not.toMatch(/^# anomalous_pixel_source\b/m);
  });
});

function syntheticPdsResponse(mode: string, values: number[]): SpectrumResponse {
  return fakeSpectrum(values, {
    wavelength: [251, 252, 253], channel_indices: [3, 4, 5],
    provenance: {
      calibration_version: null, axis_source: 'pds_embedded', excitation_nm: 247,
      wavenumber_unit: 'cm-1', intensity_unit: null, data_origin: 'pds4',
      source_digest: `synthetic-${mode}`, application: { version: '99.8.7' },
      source: { scan_id: SCAN_ID, scan_name: 'synthetic', target: 'Invented', sol: 9999, data_origin: 'pds4' },
      selection: { mode, n_points: 1, indices: mode === 'point' ? [0] : 'all' },
      processing_chain: mode === 'point' ? [] : [{ step: 'aggregate', method: 'mean' }],
    },
  });
}

function csvProvenance(csv: string) {
  return JSON.parse(csv.split('\n').find(line => line.startsWith('# provenance: '))!.slice(14));
}

it.each(['success', 'failure'])('keeps complete export identity during a pending selection and %s', async outcome => {
  getAvg.mockResolvedValue(syntheticPdsResponse('average', [2, 3, 4]));
  vi.spyOn(api, 'getScan').mockResolvedValue({ schema_version: '1.0.0', scan: { ...fakeScan(), ml_mask_count: 0 } });
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await screen.findByText('1. Despike');
  await flush();
  let resolvePoint!: (value: SpectrumResponse) => void;
  let rejectPoint!: (reason: Error) => void;
  vi.spyOn(api, 'getPointSpectrum').mockReturnValue(new Promise((resolve, reject) => {
    resolvePoint = resolve;
    rejectPoint = reject;
  }));
  await fireEvent.click(container.querySelector<HTMLButtonElement>('.point-chip')!);
  await flush();
  const pending = csvProvenance(await exportCsv(container));
  expect(pending.selection.mode).toBe('average');
  expect(pending.source_digest).toBe('synthetic-average');
  expect(pending.data.intensity).toEqual([2, 3, 4]);
  expect(pending.data.wavelength).toEqual([251, 252, 253]);
  if (outcome === 'success') resolvePoint(syntheticPdsResponse('point', [7, 8, 9]));
  else rejectPoint(new api.ApiError(500, 'synthetic', 'Invented failure'));
  await flush();
  const after = csvProvenance(await exportCsv(container));
  expect(after.selection.mode).toBe(outcome === 'success' ? 'point' : 'average');
  expect(after.source_digest).toBe(`synthetic-${outcome === 'success' ? 'point' : 'average'}`);
  expect(after.data.intensity).toEqual(outcome === 'success' ? [7, 8, 9] : [2, 3, 4]);
});

it('clears export state when the committed selection has no spectrum', async () => {
  getAvg.mockResolvedValue(syntheticPdsResponse('average', [2, 3, 4]));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await screen.findByText('1. Despike');
  await flush();
  vi.spyOn(api, 'getPointSpectrum').mockRejectedValue(new api.ApiError(404, 'synthetic', 'No invented spectrum'));
  await fireEvent.click(container.querySelector<HTMLButtonElement>('.point-chip')!);
  await flush();
  expect(get(processingState)).toBeNull();
  expect(container.querySelector<HTMLButtonElement>('.export-trigger')!.disabled).toBe(true);
});

it('ignores an older Workbench response after remount', async () => {
  getAvg.mockResolvedValue(syntheticPdsResponse('average', [2, 3, 4]));
  const first = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await screen.findByText('1. Despike');
  await flush();
  let resolvePoint!: (value: SpectrumResponse) => void;
  vi.spyOn(api, 'getPointSpectrum').mockReturnValue(new Promise(resolve => { resolvePoint = resolve; }));
  await fireEvent.click(first.container.querySelector<HTMLButtonElement>('.point-chip')!);
  await flush();
  first.unmount();
  getAvg.mockResolvedValue(syntheticPdsResponse('average', [7, 8, 9]));
  const second = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await screen.findByText('1. Despike');
  await flush();
  resolvePoint(syntheticPdsResponse('point', [12, 13, 14]));
  await flush();
  const metadata = csvProvenance(await exportCsv(second.container));
  expect(metadata.data.intensity).toEqual([7, 8, 9]);
  expect(metadata.selection.mode).toBe('average');
});

it('public-tier JSON and CSV headers exclude private fields planted in source rows', async () => {
  const privateFields = {
    source_path: '/synthetic/private/source.dat', hostname: 'synthetic-host.invalid',
    team_id: 'team-tier-synthetic-42', internal_url: 'https://synthetic-host.invalid/private',
  };
  const sourceRow = { ...fakeScan(), data_source: 'pds4', source_path: privateFields.source_path,
    source_scan_ids: [privateFields.team_id], pds4_metadata: { ...privateFields }, ml_mask_count: 0 };
  const response = syntheticPdsResponse('average', [2, 3, 4]);
  response.provenance.source = { ...response.provenance.source,
    product_id: sourceRow.scan_id, source_sha256: 'a'.repeat(64), config_hash: 'b'.repeat(64),
    pipeline_version: 'synthetic-8' };
  response.provenance.source_spectra = [{ spectrum_id: 'synthetic-pds-spectrum' }];
  vi.spyOn(api, 'getScan').mockResolvedValue({ schema_version: '1.0.0', scan: sourceRow });
  getAvg.mockResolvedValue(response);
  accessMode.set('public');
  try {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    const csv = await exportCsv(container);
    const headers = csv.split('\n').filter(line => line.startsWith('#')).join('\n');
    await fireEvent.click(container.querySelector('.export-trigger')!);
    await fireEvent.click(screen.getByText('Full provenance JSON'));
    const json = container.querySelector('.provenance-json')!.textContent!;
    for (const value of Object.values(privateFields)) {
      expect(headers).not.toContain(value);
      expect(json).not.toContain(value);
    }
    expect(JSON.parse(json)).toEqual(csvProvenance(csv));
    for (const allowed of [sourceRow.scan_id, 'synthetic-pds-spectrum', 'a'.repeat(64),
      'b'.repeat(64), 'synthetic-8', '99.8.7']) {
      expect(headers).toContain(allowed);
      expect(json).toContain(allowed);
    }
  } finally {
    accessMode.set('internal');
  }
});


function syntheticSpectrum(intensity: number[], overrides: Partial<SpectrumResponse> = {}): SpectrumResponse {
  return fakeSpectrum(intensity, { ...overrides, provenance: {
    calibration_version: 'synthetic-calibration', wavenumber_unit: 'cm-1',
    intensity_unit: 'synthetic-unit', ...overrides.provenance,
  } });
}

it('rejects a late processing result after sign-out', async () => {
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
  let finish = () => {};
  vi.spyOn(api, 'postFit').mockReturnValue(new Promise(resolve => { finish = () => resolve(fakeFit()); }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  const card = Array.from(container.querySelectorAll('.step-title')).find(el => el.textContent?.trim() === '4. Peak Fit')!.closest('.step-card')!;
  await fireEvent.click(card.querySelector('.step-header')!);
  await fireEvent.click(card.querySelector('.btn-primary')!);
  expect(api.postFit).toHaveBeenCalledOnce();
  clearSessionState();
  finish(); await flush();
  expect(get(processingState)).toBeNull();
});

it('preserves tier transitions, R2 spike markers and a light plot under dark system theme', async () => {
  const matchMedia = vi.fn(() => ({ matches: true }));
  vi.stubGlobal('matchMedia', matchMedia);
  try {
    const config = { tier: 'team', capabilities: { references: false } } as import('../lib/types').ConfigResponse;
    appConfig.set(config);
    scienceAuthorized.set(true);
    accessMode.set('internal');
    currentHash.set('#/scan/synthetic/workbench');
    const generation = sessionGeneration;
    let finish = (_value: SpectrumResponse) => {};
    const point = vi.spyOn(api, 'getPointSpectrum').mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } }); await flush();
    appConfig.set({ ...config, tier: 'public' }); accessMode.set('public'); await flush();
    finish(syntheticSpectrum(INTENSITY_A)); await flush();
    expect(get(scienceAuthorized)).toBe(true); expect(sessionGeneration).toBe(generation);
    expect(get(currentHash)).toBe('#/scan/synthetic/workbench'); expect(get(appConfig)?.tier).toBe('public');
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_A);
    point.mockResolvedValue(syntheticSpectrum(INTENSITY_NEW, { region: 'R2', wavelength: [285, 305, 325],
      despike_applied: true, despike_method: 'ml', n_masked_channels: 1, masked_positions: [1] }));
    const region = Array.from(container.querySelectorAll('.region-btn')).find(button => button.textContent?.trim() === 'r2')!;
    await fireEvent.click(region); await flush();
    const [, traces, layout] = reactSpy.mock.calls.at(-1)!;
    expect(traces[0].x).toEqual([285, 305, 325]); expect(traces[0].y).toEqual(INTENSITY_NEW);
    expect(traces.find((trace: { name: string }) => trace.name === 'ML-masked (1)').x).toEqual([305]);
    expect(layout.xaxis.title.text).toBe('Wavelength (nm)');
    expect(layout.plot_bgcolor).toBe('#ffffff'); expect(layout.paper_bgcolor).toBe('#ffffff');
    expect(traces[0].line.color).toBe('#1e293b'); expect(layout.font.color).toBeUndefined();
    expect(layout.legend.font.color).toBeUndefined(); expect(layout._viewKey).toBeUndefined();
    expect(container.querySelector('.axis-input')).toHaveAttribute('placeholder', '640');
    expect(matchMedia).not.toHaveBeenCalled();
    point.mockResolvedValue(syntheticSpectrum(INTENSITY_NEW, { region: 'R2', wavenumber: [], wavelength: [285, 305, 325] }));
    await fireEvent.click(Array.from(container.querySelectorAll('.region-btn')).find(button => button.textContent?.trim() === 'r3')!);
    await flush();
    expect(screen.getByText('No spectrum data available')).toBeTruthy();
    expect(container.querySelector('.export-trigger')).toBeDisabled();
  } finally { vi.unstubAllGlobals(); }
});

it('keeps legacy wavelength-only ML masks absent', async () => {
  appConfig.set({ tier: 'team', capabilities: { references: false } } as import('../lib/types').ConfigResponse);
  vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(syntheticSpectrum(INTENSITY_A, {
    region: 'R2', wavenumber: [], wavelength: [285, 305, 325], masked_positions: [1],
    despike_applied: true, despike_method: 'ml', n_masked_channels: 1,
    provenance: { calibration_version: null, wavelength_unit: 'nm', intensity_unit: null } as unknown as import('../lib/types').SpectrumProvenance,
  }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } }); await flush();
  expect(screen.getByText('No spectrum data available')).toBeTruthy();
  expect(lastSpikeTrace()).toBeUndefined();
  expect(screen.queryByLabelText('Spike markers')).toBeNull();
  expect(container.querySelector('.export-trigger')).toBeDisabled();
  const csv = await exportCsv(container);
  expect(csvHasSpikeMaskColumn(csv)).toBe(false);
  expect(csv).not.toContain('spike_mask_method');
  expect(csvProvenance(csv).annotations).not.toHaveProperty('spike_mask');
});

it('preserves legacy metadata timing and boundary behavior', async () => {
  appConfig.set({ tier: 'team', capabilities: { references: false } } as import('../lib/types').ConfigResponse);
  let failPoints = () => {};
  vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise((_resolve, reject) => {
    failPoints = () => reject(new api.ApiError(403, 'Synthetic forbidden'));
  }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  // Legacy metadata appears before points resolve and remains after a points error.
  expect(container.textContent).toContain('Invented');
  appConfig.set({ tier: 'public', capabilities: { references: false } } as import('../lib/types').ConfigResponse);
  accessMode.set('public');
  failPoints(); await flush();
  expect(api.getScan).toHaveBeenCalledOnce(); expect(api.getScanPoints).toHaveBeenCalledOnce();
  expect(container.textContent).toContain('Invented');
  expect(container.querySelector('.workbench-layout')).toBeTruthy();
  expect(container.querySelector('.aci-viewer')).toBeTruthy();
  expect(container.querySelector('.error-message')).toBeTruthy();
});

describe('ProcessingWorkbench fit status', () => {
  it.each([
    [true, 'peaks fitted'],
    [false, 'no fitted peaks'],
  ])('has_fitted_peaks=%s renders "%s" (stored column ignored)', async (hasFittedPeaks, label) => {
    vi.mocked(api.getScan).mockResolvedValue({
      schema_version: '1.0.0',
      scan: { ...fakeScan(), processing_status: 'completed' as const, has_fitted_peaks: hasFittedPeaks },
    });
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    const dt = [...container.querySelectorAll('dt')].find((el) => el.textContent === 'Status')!;
    expect(dt.nextElementSibling).toHaveTextContent(new RegExp(`^${label}$`));
    expect(screen.queryByText('completed')).toBeNull();
    expect(screen.queryByText('unprocessed')).toBeNull();
  });
});

it('renders the spectrum and export controls without the retired desk', async () => {
  const legacyCapabilities = { references: false, evidence_desk: true };
  appConfig.update(config => config && { ...config, capabilities: legacyCapabilities });
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
  await flush();
  expect(container.querySelector('.raman-view')).toBeTruthy();
  expect(container.querySelectorAll('.export-trigger')).toHaveLength(1);
  expect(screen.queryByLabelText('Evidence desk')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Pin current trace' })).toBeNull();
});
