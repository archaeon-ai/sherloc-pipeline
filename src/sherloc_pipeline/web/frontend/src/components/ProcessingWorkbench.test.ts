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
import { render, fireEvent, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import { get } from 'svelte/store';
import * as api from '../lib/api';
import { pointSelection, processingState } from '../lib/processingStore';
import { clearEvidence, pinnedTraces } from '../lib/evidenceStore';
import { accessMode, appConfig, scienceAuthorized, currentHash, sessionGeneration, clearSessionState, processingNotice } from '../lib/stores';
import type { BadpixResponse, FitResponse, ProcessingSnapshot, SpectrumParams, SpectrumResponse } from '../lib/types';
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
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  clearEvidence();
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
  // Badpix annotation fetch (issue #9): default to an empty list so the
  // despike-focused tests never hit the real fetch path (and the overlay
  // stays absent). The PR #11 F1 tests override this with deferred promises.
  vi.spyOn(api, 'getBadpixChannels').mockResolvedValue({
    schema_version: '1.0.0',
    region: 'R1',
    n_channels: 3,
    badpix: [],
  });
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

// PR #11 review F1 (Major): badpix positions are REGION-SPECIFIC served-array
// indices. loadBadpix previously committed `badpix = res.badpix`
// unconditionally, so (a) on a fast region switch an older region's slower
// response could resolve last and overwrite the new region's annotations, and
// (b) with the overlay ON, exportBadpix kept the previous region's list while
// the new region loaded. The fix mirrors the spectrum loader's generation
// guard and clears the list at request start. These tests drive the race with
// hand-controlled (deferred) getBadpixChannels responses.
describe('ProcessingWorkbench badpix staleness guard (PR #11 F1)', () => {
  function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
    let resolve!: (v: T) => void;
    const promise = new Promise<T>((r) => {
      resolve = r;
    });
    return { promise, resolve };
  }

  function badpixResponse(
    region: string,
    items: Array<{ position: number; channel: number; tier: number; source: string }>,
  ): BadpixResponse {
    return { schema_version: '1.0.0', region, n_channels: 3, badpix: items };
  }

  // The R1 (mount-region) list and the R2 list are deliberately
  // distinguishable in every observable: count (2 vs 1), positions (0,2 vs 1),
  // and source (jb25 vs g5_eps) — so a stale overwrite cannot pass unnoticed.
  const R1_ITEMS = [
    { position: 0, channel: 52, tier: 1, source: 'jb25' },
    { position: 2, channel: 54, tier: 2, source: 'jb25' },
  ];
  const R2_ITEMS = [{ position: 1, channel: 600, tier: 1, source: 'g5_eps' }];

  /** The badpix marker trace in the MOST RECENT RamanView render (only the
   *  final Plotly.react call counts — absence there means the overlay is not
   *  currently rendered, unlike the history-searching lastSpikeTrace). */
  function badpixTraceInLastRender(): { name?: string; x?: number[] } | undefined {
    expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
    const traces = reactSpy.mock.calls[reactSpy.mock.calls.length - 1][1] as Array<{
      name?: string;
      mode?: string;
      x?: number[];
    }>;
    return traces.find((t) => t.mode === 'markers' && /Known-noisy/.test(t.name ?? ''));
  }

  /** Click the region-selector button labeled `label` ('r2', 'full', ...). */
  async function clickRegion(container: HTMLElement, label: string): Promise<void> {
    const btn = Array.from(container.querySelectorAll<HTMLButtonElement>('.region-btn')).find(
      (b) => b.textContent?.trim() === label,
    );
    expect(btn, `no region button labeled "${label}"`).toBeDefined();
    await fireEvent.click(btn!);
    await flush();
  }

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

  it('drops an obsolete region\'s slower badpix response (render AND export)', async () => {
    const r1 = deferred<BadpixResponse>();
    const r2 = deferred<BadpixResponse>();
    const loadCatalog = vi.spyOn(api, 'getBadpixChannels').mockReturnValue(r1.promise);

    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Mount's R1 badpix fetch is still pending; switch to R2 before it lands.
    loadCatalog.mockReturnValue(r2.promise);
    await clickRegion(container, 'r2');

    // The CURRENT region's (R2) response lands and commits.
    r2.resolve(badpixResponse('R2', R2_ITEMS));
    await flush();

    // Turn the overlay on (the toggle exists because the R2 list is non-empty)
    // so both the plot and the export gate are live observables.
    await fireEvent.click(screen.getByLabelText(/Known-noisy channels/i));
    await flush();
    expect(badpixTraceInLastRender()?.name).toBe('Known-noisy channels (1)');
    expect(badpixTraceInLastRender()?.x).toEqual([900]); // wavenumber[1]

    // The OBSOLETE region's (R1, mount-time) response lands last — the
    // generation guard must drop it: the plot keeps R2's single marker
    // (a stale commit would re-render with R1's two markers at 800/1000).
    r1.resolve(badpixResponse('R1', R1_ITEMS));
    await flush();
    expect(badpixTraceInLastRender()?.name).toBe('Known-noisy channels (1)');
    expect(badpixTraceInLastRender()?.x).toEqual([900]);

    // Export gate: the CSV carries R2's annotation (row pattern 0,1,0 and
    // g5_eps provenance), not the stale R1 list (1,0,1 / jb25).
    const csv = await exportCsv(container);
    expect(csv).toContain('# badpix_source: g5_eps');
    expect(csv).not.toContain('jb25');
    const header = csv.split('\n').find((l) => l.startsWith('wavenumber'))!.split(',');
    const bIdx = header.indexOf('badpix');
    expect(bIdx).toBeGreaterThan(-1);
    const rows = csv.split('\n').filter((l) => /^\d/.test(l)).map((r) => r.split(','));
    expect(rows.map((r) => r[bIdx])).toEqual(['0', '1', '0']);
  });

  it('clears the previous region\'s annotations during the transition until the matching response lands', async () => {
    const r1 = deferred<BadpixResponse>();
    const r2 = deferred<BadpixResponse>();
    const loadCatalog = vi.spyOn(api, 'getBadpixChannels').mockReturnValue(r1.promise);

    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();

    // Mount region (R1) resolves; overlay on → two markers render.
    r1.resolve(badpixResponse('R1', R1_ITEMS));
    await flush();
    await fireEvent.click(screen.getByLabelText(/Known-noisy channels/i));
    await flush();
    expect(badpixTraceInLastRender()?.name).toBe('Known-noisy channels (2)');

    // Region switch with R2's response still in flight: the R1 list must be
    // cleared IMMEDIATELY — no markers, no toggle (it renders only with a
    // non-empty list), and nothing reaches the export gate.
    loadCatalog.mockReturnValue(r2.promise);
    await clickRegion(container, 'r2');
    expect(badpixTraceInLastRender()).toBeUndefined();
    expect(screen.queryByLabelText(/Known-noisy channels/i)).toBeNull();
    const csvDuringTransition = await exportCsv(container);
    expect(csvDuringTransition).not.toContain('badpix_source');
    expect(
      csvDuringTransition.split('\n').find((l) => l.startsWith('wavenumber')),
    ).not.toContain('badpix');

    // The matching (R2) response lands → the new region's annotations appear.
    // The overlay toggle state is owned by the Workbench and survived the
    // transition, so the marker renders without re-toggling.
    r2.resolve(badpixResponse('R2', R2_ITEMS));
    await flush();
    expect(badpixTraceInLastRender()?.name).toBe('Known-noisy channels (1)');
    expect(badpixTraceInLastRender()?.x).toEqual([900]);
  });
});


it('maps full-catalog channels through the embedded axis channel map', async () => {
  getAvg.mockResolvedValue(fakeSpectrum(INTENSITY_A, { channel_indices: [21, 24, 30] }));
  vi.spyOn(api, 'getBadpixChannels').mockResolvedValue({
    schema_version: '1.0.0', region: 'R123', n_channels: 2148,
    badpix: [{ channel: 24, position: 24, tier: 1, source: 'both' }],
  });
  render(ProcessingWorkbench, { scanId: SCAN_ID });
  await screen.findByText('1. Despike');
  await flush();
  expect(api.getBadpixChannels).toHaveBeenCalledWith('R123');
  await fireEvent.click(screen.getByLabelText(/Known-noisy channels/i));
  await flush();
  const traces = reactSpy.mock.calls.at(-1)![1] as Array<{ name?: string; x?: number[] }>;
  expect(traces.find(trace => trace.name?.startsWith('Known-noisy'))?.x).toEqual([900]);
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

describe('ProcessingWorkbench evidence session boundaries', () => {
  beforeEach(() => { getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_A)); });
  it('renders retained pins after an initial 404 on remount and then commits a valid point', async () => {
    const first = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    first.unmount();
    getAvg.mockRejectedValueOnce(new api.ApiError(404, 'Synthetic selection unavailable'));
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    expect(get(processingState)).toBeNull();
    expect(get(pinnedTraces)).toHaveLength(1);
    expect(screen.getByLabelText('Evidence desk')).toBeTruthy();
    await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'overlay' } });
    await flush();
    expect(container.querySelector('.export-trigger')).not.toBeDisabled();
    vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(syntheticSpectrum(INTENSITY_NEW));
    await fireEvent.click(container.querySelector('.point-chip')!);
    await flush();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
    expect(screen.getByLabelText('Evidence desk')).toBeTruthy();
    expect(get(pinnedTraces)).toHaveLength(1);
  });

  it.each([403, 'network'] as const)('withholds retained pins when the remount spectrum fails with %s', async failure => {
    const first = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    first.unmount();
    getAvg.mockRejectedValueOnce(failure === 403 ? new api.ApiError(403, 'Synthetic forbidden') : new Error('Synthetic network failure'));
    render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    expect(get(pinnedTraces)).toHaveLength(1);
    expect(screen.queryByLabelText('Evidence desk')).toBeNull();
    expect(get(processingState)).toBeNull();
  });

  it.each([200, 404])('does not validate the desk with a stale %s spectrum', async status => {
    let finish = () => {};
    getAvg.mockReturnValueOnce(new Promise((resolve, reject) => {
      finish = () => status === 200 ? resolve(syntheticSpectrum(INTENSITY_STALE)) : reject(new api.ApiError(404, 'Synthetic unavailable'));
    }));
    render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    // Keep the current boundary unvalidated while the old boundary completes.
    vi.mocked(api.getScan).mockReturnValueOnce(new Promise(() => {}));
    appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
    await flush();
    finish(); await flush();
    expect(screen.queryByLabelText('Evidence desk')).toBeNull();
    expect(get(processingState)).toBeNull();
  });

  it('validates a later successful selection after a denied initial spectrum', async () => {
    getAvg.mockRejectedValueOnce(new api.ApiError(403, 'Synthetic forbidden'));
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    expect(screen.queryByLabelText('Evidence desk')).toBeNull();
    vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(syntheticSpectrum(INTENSITY_NEW));
    await fireEvent.click(container.querySelector('.point-chip')!);
    await flush();
    expect(screen.getByLabelText('Evidence desk')).toBeTruthy();
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  });

  it('displays same-target pins with axis controls after the current selection returns 404', async () => {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    await flush();
    getAvg.mockRejectedValue(new api.ApiError(404, 'Synthetic selection unavailable'));
    await fireEvent.change(container.querySelector('.averaging-controls select')!, { target: { value: 'mean' } });
    await flush();
    expect(get(processingState)).toBeNull();
    expect(get(pinnedTraces)).toHaveLength(1);
    expect(screen.getByLabelText('View')).toHaveValue('overlay');
    expect(screen.queryByText('No spectrum data loaded')).toBeNull();
    expect(container.querySelectorAll('.axis-controls .axis-input')).toHaveLength(4);
    const [, traces, layout] = reactSpy.mock.calls.at(-1)!;
    expect(traces).toHaveLength(1);
    expect(layout.meta.traces[0].snapshot.trace.source.scan_id).toBe(SCAN_ID);
    expect(container.querySelector('.export-trigger')).not.toBeDisabled();
    await fireEvent.click(container.querySelector('.pin input[type="checkbox"]')!);
    await flush();
    expect(screen.getByText('No spectrum data loaded')).toBeTruthy();
    expect(container.querySelector('.axis-controls')).toBeNull();
    expect(reactSpy.mock.calls.at(-1)![1]).toEqual([]);
  });

  it('retains a pin across scans of the same target and clears it for a new target', async () => {
    const first = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    expect(get(pinnedTraces)).toHaveLength(1);
    first.unmount();
    vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), id: 'synthetic-second', scan_name: 'Synthetic second' } });
    const second = render(ProcessingWorkbench, { scanId: 'synthetic-second' });
    await flush(); expect(get(pinnedTraces)).toHaveLength(1); second.unmount();
    vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), target: 'Other invented' } });
    render(ProcessingWorkbench, { scanId: 'synthetic-third' });
    await flush(); expect(get(pinnedTraces)).toEqual([]);
  });

  it('clears target A pins when target B metadata succeeds but its points fail', async () => {
    const first = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    expect(get(pinnedTraces)).toHaveLength(1);
    first.unmount();
    vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Other invented' } });
    let failPoints = () => {};
    vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise((_resolve, reject) => {
      failPoints = () => reject(new Error('Synthetic points failure'));
    }));
    const second = render(ProcessingWorkbench, { scanId: 'synthetic-second' });
    await flush();
    expect(get(pinnedTraces)).toEqual([]);
    expect(second.container.textContent).not.toContain('Other invented');
    expect(second.container.querySelector('.workbench-layout')).toBeNull();
    failPoints(); await flush();
    expect(second.container.querySelector('.workbench-layout')).toBeNull();
    second.unmount();
    render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    expect(screen.getByLabelText('Evidence desk')).toBeTruthy();
    expect(get(pinnedTraces)).toEqual([]);
  });

  it('denies a spectrum completing after sign-out before it can populate or pin state', async () => {
    let resolve = (_value: SpectrumResponse) => {};
    getAvg.mockReturnValue(new Promise(done => { resolve = done; }));
    render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush(); clearSessionState();
    resolve(syntheticSpectrum(INTENSITY_STALE)); await flush();
    expect(get(processingState)).toBeNull(); expect(get(pinnedTraces)).toEqual([]);
  });

  it('stops sequential requests when scan metadata completes after sign-out', async () => {
    let resolve = (_value: Awaited<ReturnType<typeof api.getScan>>) => {};
    vi.mocked(api.getScan).mockReturnValue(new Promise(done => { resolve = done; }));
    render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush(); clearSessionState();
    resolve({ schema_version: '1', scan: fakeScan() }); await flush();
    expect(api.getScanPoints).not.toHaveBeenCalled(); expect(getAvg).not.toHaveBeenCalled();
    expect(get(pinnedTraces)).toEqual([]);
  });

  it('pins the committed new request when an older reply finishes later', async () => {
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    let resolve = (_value: SpectrumResponse) => {};
    getAvg.mockReturnValueOnce(new Promise(done => { resolve = done; }));
    await fireEvent.change(container.querySelector('.averaging-controls select')!, { target: { value: 'mean' } });
    getAvg.mockResolvedValueOnce(syntheticSpectrum(INTENSITY_NEW));
    await fireEvent.change(container.querySelector('.averaging-controls select')!, { target: { value: 'median' } });
    await flush();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
    resolve(syntheticSpectrum(INTENSITY_STALE)); await flush();
    expect(get(pinnedTraces)[0].snapshot.raman.intensity).toEqual(INTENSITY_NEW);
    expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  });

  it('mounts and exports a wavelength-only committed spectrum', async () => {
    getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_A, { region: 'R2', wavenumber: [], wavelength: [280, 300, 320], channel_indices: [0, 1, 3],
      provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'synthetic', region: 'R2' } }));
    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await flush();
    expect(container.querySelector('.raman-view')).toBeTruthy();
    expect(container.querySelector('.export-trigger')).not.toBeDisabled();
    const call = reactSpy.mock.calls.at(-1)!;
    expect(call[1][0].x).toEqual([280, 300, null, 320]);
    expect(call[2].xaxis.title.text).toBe('Wavelength (nm)');
  });
});


it.each(['tier', 'access'])('invalidates an in-flight spectrum on an authorized %s transition', async boundary => {
  accessMode.set('internal');
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  scienceAuthorized.set(true);
  currentHash.set('#/scan/synthetic/workbench');
  const generation = sessionGeneration;
  let resolve = (_value: SpectrumResponse) => {};
  getAvg.mockReturnValue(new Promise(done => { resolve = done; }));
  render(ProcessingWorkbench, { scanId: SCAN_ID });
  await flush();
  // Hold revalidation while the spectrum from the old boundary completes.
  vi.mocked(api.getScan).mockReturnValueOnce(new Promise(() => {}));
  if (boundary === 'tier') appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  else accessMode.set('public');
  resolve(syntheticSpectrum(INTENSITY_STALE)); await flush();
  expect(get(scienceAuthorized)).toBe(true);
  expect(get(currentHash)).toBe('#/scan/synthetic/workbench');
  expect(sessionGeneration).toBe(generation);
  expect(get(appConfig)?.capabilities?.evidence_desk).toBe(true);
  expect(get(processingState)).toBeNull(); expect(get(pinnedTraces)).toEqual([]);
});


it.each([undefined, false, true])('gates the evidence desk with capability %s', async enabled => {
  appConfig.set({ tier: 'team', capabilities: enabled === undefined ? undefined : { evidence_desk: enabled } } as import('../lib/types').ConfigResponse);
  getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_A));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  expect(Boolean(screen.queryByLabelText('Evidence desk'))).toBe(enabled === true);
  expect(container.querySelector('.raman-view')).toBeTruthy();
  expect(container.querySelectorAll('.export-trigger')).toHaveLength(1);
  expect(csvProvenance(await exportCsv(container)).data.intensity).toEqual(INTENSITY_A);
});

it.each(['R2', 'R3'])('commits authoritative %s without provenance.region', async region => {
  getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_A, { region, wavelength: [285, 305, 325],
    provenance: { calibration_version: null, wavenumber_unit: 'cm-1', wavelength_unit: 'nm', intensity_unit: 'synthetic' } }));
  render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  expect(get(processingState)?.trace?.region).toBe(region);
  expect(reactSpy.mock.calls.at(-1)![2].xaxis.title.text).toBe('Wavelength (nm)');
  await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' })); await flush();
  expect(get(pinnedTraces)[0].snapshot.trace.region).toBe(region);
  expect(reactSpy.mock.calls.at(-1)![2].xaxis.title.text).toBe('Wavelength (nm)');
});

it.each(['sign-out', 'tier'])('rejects a late processing result after %s', async boundary => {
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
  let finish = () => {};
  vi.spyOn(api, 'postFit').mockReturnValue(new Promise(resolve => { finish = () => resolve(fakeFit()); }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  const card = Array.from(container.querySelectorAll('.step-title')).find(el => el.textContent?.trim() === '4. Peak Fit')!.closest('.step-card')!;
  await fireEvent.click(card.querySelector('.step-header')!);
  await fireEvent.click(card.querySelector('.btn-primary')!);
  expect(api.postFit).toHaveBeenCalledOnce();
  if (boundary === 'sign-out') clearSessionState();
  else { scienceAuthorized.set(true); appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse); }
  finish(); await flush();
  if (boundary === 'sign-out') expect(get(processingState)).toBeNull();
  else expect(get(processingState)?.stage).not.toBe('raman_fitted');
  expect(get(pinnedTraces)).toEqual([]);
});


describe('capability-off legacy Workbench regression', () => {
  it.each([undefined, false])('preserves tier transitions, R2 annotations and a light plot under dark system theme (capability=%s)', async enabled => {
    const matchMedia = vi.fn(() => ({ matches: true }));
    vi.stubGlobal('matchMedia', matchMedia);
    try {
      const config = { tier: 'team', capabilities: enabled === undefined ? undefined : { evidence_desk: enabled } } as import('../lib/types').ConfigResponse;
      appConfig.set(config);
      scienceAuthorized.set(true);
      accessMode.set('internal');
      currentHash.set('#/scan/synthetic/workbench');
      const generation = sessionGeneration;
      const context = vi.spyOn(await import('../lib/evidenceStore'), 'setEvidenceContext');
      const invalidate = vi.spyOn(await import('../lib/evidenceStore'), 'invalidateEvidence');
      let finish = (_value: SpectrumResponse) => {};
      const point = vi.spyOn(api, 'getPointSpectrum').mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
      const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } }); await flush();
      appConfig.set({ ...config, tier: 'public' }); accessMode.set('public'); await flush();
      finish(syntheticSpectrum(INTENSITY_A)); await flush();
      expect(get(scienceAuthorized)).toBe(true); expect(sessionGeneration).toBe(generation);
      expect(get(currentHash)).toBe('#/scan/synthetic/workbench'); expect(get(appConfig)?.tier).toBe('public');
      expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_A);
      vi.mocked(api.getBadpixChannels).mockResolvedValue({ schema_version: '1', region: 'R123', n_channels: 3,
        badpix: [{ position: 1, channel: 600, tier: 1, source: 'synthetic-noisy' }] });
      point.mockResolvedValue(syntheticSpectrum(INTENSITY_NEW, { region: 'R2', wavelength: [285, 305, 325],
        despike_applied: true, despike_method: 'ml', n_masked_channels: 1, masked_positions: [1] }));
      const region = Array.from(container.querySelectorAll('.region-btn')).find(button => button.textContent?.trim() === 'r2')!;
      await fireEvent.click(region); await flush();
      await fireEvent.click(screen.getByLabelText('Known-noisy channels')); await flush();
      expect(screen.queryByLabelText('Evidence desk')).toBeNull();
      const [, traces, layout] = reactSpy.mock.calls.at(-1)!;
      expect(traces[0].x).toEqual([285, 305, 325]); expect(traces[0].y).toEqual(INTENSITY_NEW);
      expect(traces.find((trace: { name: string }) => trace.name === 'ML-masked (1)').x).toEqual([305]);
      expect(traces.find((trace: { name: string }) => trace.name === 'Known-noisy channels (1)').x).toEqual([305]);
      expect(layout.xaxis.title.text).toBe('Wavelength (nm)');
      expect(layout.plot_bgcolor).toBe('#ffffff'); expect(layout.paper_bgcolor).toBe('#ffffff');
      expect(traces[0].line.color).toBe('#1e293b'); expect(layout.font.color).toBeUndefined();
      expect(layout.legend.font.color).toBeUndefined(); expect(layout._viewKey).toBeUndefined();
      expect(container.querySelector('.axis-input')).toHaveAttribute('placeholder', '640');
      expect(matchMedia).not.toHaveBeenCalled(); expect(context).not.toHaveBeenCalled(); expect(invalidate).not.toHaveBeenCalled();
      point.mockResolvedValue(syntheticSpectrum(INTENSITY_NEW, { region: 'R2', wavenumber: [], wavelength: [285, 305, 325] }));
      await fireEvent.click(Array.from(container.querySelectorAll('.region-btn')).find(button => button.textContent?.trim() === 'r3')!);
      await flush();
      expect(screen.getByText('No spectrum data available')).toBeTruthy();
      expect(container.querySelector('.export-trigger')).toBeDisabled();
    } finally { vi.unstubAllGlobals(); }
  });
});


it.each(['', '   '])('isolates pins by scan identity for a blank target (%j)', async target => {
  getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_A));
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), target } });
  const first = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
  expect(get(pinnedTraces)).toHaveLength(1); first.unmount();
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), id: 'synthetic-other', target } });
  render(ProcessingWorkbench, { scanId: 'synthetic-other' }); await flush();
  expect(get(pinnedTraces)).toEqual([]);
});


it.each(['tier', 'access'])('accepts fresh spectrum, annotations and processing after an authorized %s boundary', async boundary => {
  accessMode.set('internal');
  scienceAuthorized.set(true);
  currentHash.set('#/scan/synthetic/workbench');
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
  const generation = sessionGeneration;
  let revalidate = () => {};
  vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
    revalidate = () => resolve({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
  }));
  if (boundary === 'tier') appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  else accessMode.set('public');
  await flush();
  expect(get(processingState)).toBeNull(); expect(get(pinnedTraces)).toEqual([]);
  expect(container.querySelector('.export-trigger')).toBeNull();
  expect(container.querySelector('.aci-viewer')).toBeNull();
  getAvg.mockResolvedValue(syntheticSpectrum(INTENSITY_NEW));
  vi.mocked(api.getBadpixChannels).mockResolvedValue({ schema_version: '1', region: 'R123', n_channels: 3,
    badpix: [{ position: 1, channel: 600, tier: 1, source: 'synthetic-noisy' }] });
  revalidate(); await flush();
  expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  await fireEvent.change(screen.getByLabelText('View'), { target: { value: 'processing' } }); await flush();
  await fireEvent.click(screen.getByLabelText('Known-noisy channels')); await flush();
  expect(csvProvenance(await exportCsv(container)).annotations.badpix[0].source).toBe('synthetic-noisy');
  vi.spyOn(api, 'postFit').mockResolvedValue(fakeFit());
  const card = Array.from(container.querySelectorAll('.step-title')).find(el => el.textContent?.trim() === '4. Peak Fit')!.closest('.step-card')!;
  await fireEvent.click(card.querySelector('.step-header')!);
  await fireEvent.click(card.querySelector('.btn-primary')!); await flush();
  expect(get(processingState)?.stage).toBe('raman_fitted');
  expect(get(scienceAuthorized)).toBe(true); expect(sessionGeneration).toBe(generation);
  expect(get(currentHash)).toBe('#/scan/synthetic/workbench');
});


it.each(['R2', 'R3'])('keeps wavelength-only %s ML annotations in plot, pin and exports', async region => {
  vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(syntheticSpectrum(INTENSITY_A, {
    region, wavenumber: [], wavelength: [285, 305, 325], masked_positions: [1],
    despike_applied: true, despike_method: 'ml', n_masked_channels: 1,
    provenance: { calibration_version: null, wavelength_unit: 'nm', intensity_unit: null } as unknown as import('../lib/types').SpectrumProvenance,
  }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } }); await flush();
  expect(lastSpikeTrace()?.x).toEqual([305]);
  const metadata = csvProvenance(await exportCsv(container));
  expect(metadata.annotations.spike_mask).toEqual([false, true, false]);
  expect(metadata.annotations.spike_method).toBe('ml');
  await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' })); await flush();
  expect(get(pinnedTraces)[0].annotations.spike_mask).toEqual([false, true, false]);
});


it.each([undefined, false])('keeps legacy wavelength-only ML masks absent with capability %s', async enabled => {
  appConfig.set({ tier: 'team', capabilities: enabled === undefined ? undefined : { evidence_desk: enabled } } as import('../lib/types').ConfigResponse);
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


it('keeps distinct nonblank target values separate without trimming identity', async () => {
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), target: 'Synthetic Alpha' } });
  const first = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
  expect(get(pinnedTraces)).toHaveLength(1); first.unmount();
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), id: 'synthetic-space', target: 'Synthetic Alpha ' } });
  render(ProcessingWorkbench, { scanId: 'synthetic-space' }); await flush();
  expect(get(pinnedTraces)).toEqual([]);
});


it.each([['getScan', 'tier'], ['getScan', 'access'], ['getScanPoints', 'tier'], ['getScanPoints', 'access']] as const)('retries initial %s across an authorized %s boundary', async (pendingMethod, boundary) => {
  accessMode.set('internal'); scienceAuthorized.set(true);
  currentHash.set('#/scan/synthetic/workbench');
  const generation = sessionGeneration;
  vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(syntheticSpectrum(INTENSITY_NEW));
  let finish = () => {};
  if (pendingMethod === 'getScan') {
    vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
      finish = () => resolve({ schema_version: '1', scan: { ...fakeScan(), target: 'Stale synthetic' } });
    }));
  } else {
    vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise(resolve => {
      finish = () => resolve({ schema_version: '1', scan_id: SCAN_ID, points: [], n_points: 0 });
    }));
  }
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID, queryParams: { point: '0' } });
  await flush();
  expect(api.getPointSpectrum).not.toHaveBeenCalled();
  if (boundary === 'tier') appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  else accessMode.set('public');
  await flush();
  finish(); await flush(); await flush();
  expect(api.getScan).toHaveBeenCalledTimes(2);
  expect(api.getScanPoints).toHaveBeenCalledTimes(pendingMethod === 'getScan' ? 1 : 2);
  expect(api.getPointSpectrum).toHaveBeenCalled();
  expect(get(processingState)?.trace?.source.target).toBe('Invented');
  expect(container.querySelector('.raman-view')).toBeTruthy();
  expect(container.querySelector('.export-trigger')).not.toBeDisabled();
  expect(sessionGeneration).toBe(generation); expect(get(scienceAuthorized)).toBe(true);
  expect(get(currentHash)).toBe('#/scan/synthetic/workbench');
});

it.each(['destroy', 'session'] as const)('does not retry an invalidated initial scan after %s ends eligibility', async ended => {
  let finish = () => {};
  vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
    finish = () => resolve({ schema_version: '1', scan: fakeScan() });
  }));
  const view = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  if (ended === 'destroy') view.unmount();
  else clearSessionState();
  finish(); await flush();
  expect(api.getScan).toHaveBeenCalledOnce(); expect(api.getScanPoints).not.toHaveBeenCalled();
  expect(get(processingState)).toBeNull();
});

it.each([['loaded', 'tier'], ['loaded', 'access'], ['points-pending', 'tier'], ['points-pending', 'access']] as const)('clears scan, points and ACI across a denied boundary from %s via %s', async (initial, boundary) => {
  accessMode.set('internal'); scienceAuthorized.set(true);
  vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Private synthetic' } });
  let finishPoints = () => {};
  if (initial === 'points-pending') {
    vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise(resolve => {
      finishPoints = () => resolve({ schema_version: '1', scan_id: SCAN_ID, points: [fakePoint()], n_points: 1 });
    }));
  }
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  if (initial === 'loaded') {
    expect(container.textContent).toContain('Private synthetic');
    expect(container.querySelector('.aci-viewer')).toBeTruthy();
    await fireEvent.click(screen.getByRole('button', { name: 'Pin current trace' }));
  } else {
    expect(container.textContent).not.toContain('Private synthetic');
  }
  vi.mocked(api.getScan).mockRejectedValueOnce(new api.ApiError(403, 'Synthetic forbidden'));
  if (boundary === 'tier') appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  else accessMode.set('public');
  expect(get(processingState)).toBeNull(); expect(get(pinnedTraces)).toEqual([]);
  expect(get(pointSelection)).toEqual({ mode: 'average' });
  await flush(); finishPoints(); await flush();
  expect(api.getScan).toHaveBeenCalledTimes(2);
  expect(api.getScanPoints).toHaveBeenCalledOnce();
  expect(container.textContent).not.toContain('Private synthetic');
  expect(container.querySelector('.workbench-layout')).toBeNull();
  expect(container.querySelector('.aci-viewer')).toBeNull();
  expect(container.querySelector('.error-message')).toBeTruthy();
  expect(container.textContent).not.toContain('Loading scan');
  expect(get(processingState)).toBeNull();
});

it('commits fresh scan and points together and remounts ACI after revalidation', async () => {
  accessMode.set('internal'); scienceAuthorized.set(true);
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  const oldAci = container.querySelector('.aci-viewer');
  const aciCalls = vi.mocked(api.fetchAciImage).mock.calls.length;
  let finishPoints = () => {};
  vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Public synthetic', n_points: 2 } });
  vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise(resolve => {
    finishPoints = () => resolve({ schema_version: '1', scan_id: SCAN_ID, points: [fakePoint(), fakePoint(1)], n_points: 2 });
  }));
  appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  await flush();
  expect(container.textContent).not.toContain('Invented');
  expect(container.textContent).not.toContain('Public synthetic');
  expect(container.querySelector('.aci-viewer')).toBeNull();
  expect(get(processingState)).toBeNull();
  finishPoints(); await flush();
  expect(container.textContent).toContain('Public synthetic');
  expect(container.querySelector('.aci-viewer')).toBeTruthy();
  expect(container.querySelector('.aci-viewer')).not.toBe(oldAci);
  expect(api.fetchAciImage).toHaveBeenCalledTimes(aciCalls + 1);
  expect(api.getScan).toHaveBeenCalledTimes(2); expect(api.getScanPoints).toHaveBeenCalledTimes(2);
  expect(get(processingState)?.trace?.source.target).toBe('Public synthetic');
});


it('leaves metadata cleared when point revalidation fails', async () => {
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Uncommitted synthetic' } });
  vi.mocked(api.getScanPoints).mockRejectedValueOnce(new api.ApiError(403, 'Synthetic forbidden'));
  appConfig.set({ tier: 'public', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  await flush();
  expect(container.textContent).not.toContain('Uncommitted synthetic');
  expect(container.textContent).not.toContain('Invented');
  expect(container.querySelector('.aci-viewer')).toBeNull();
  expect(container.querySelector('.workbench-layout')).toBeNull();
  expect(container.querySelector('.error-message')).toBeTruthy();
});

it.each([undefined, false])('preserves legacy metadata timing and boundary behavior with capability %s', async enabled => {
  appConfig.set({ tier: 'team', capabilities: enabled === undefined ? undefined : { evidence_desk: enabled } } as import('../lib/types').ConfigResponse);
  let failPoints = () => {};
  vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise((_resolve, reject) => {
    failPoints = () => reject(new api.ApiError(403, 'Synthetic forbidden'));
  }));
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  // Legacy metadata appears before points resolve and remains after a points error.
  expect(container.textContent).toContain('Invented');
  appConfig.set({ tier: 'public', capabilities: enabled === undefined ? undefined : { evidence_desk: enabled } } as import('../lib/types').ConfigResponse);
  accessMode.set('public');
  failPoints(); await flush();
  expect(api.getScan).toHaveBeenCalledOnce(); expect(api.getScanPoints).toHaveBeenCalledOnce();
  expect(container.textContent).toContain('Invented');
  expect(container.querySelector('.workbench-layout')).toBeTruthy();
  expect(container.querySelector('.aci-viewer')).toBeTruthy();
  expect(container.querySelector('.error-message')).toBeTruthy();
});


it('keeps the desk unmounted when enablement spectrum revalidation is denied', async () => {
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: false } } as import('../lib/types').ConfigResponse);
  accessMode.set('internal');
  vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Private synthetic', ml_mask_count: 0 } });
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), target: 'Public synthetic', ml_mask_count: 0 } });
  getAvg.mockRejectedValue(new api.ApiError(403, 'Synthetic spectrum forbidden'));
  accessMode.set('public');
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  await flush();
  expect(container.textContent).not.toContain('Private synthetic');
  expect(container.textContent).not.toContain('Loading scan');
  expect(container.textContent).toContain('Synthetic spectrum forbidden');
  expect(container.querySelector('[aria-label="Evidence desk"]')).toBeNull();
  expect(get(processingState)).toBeNull();
});

it.each(['access-first', 'config-first'] as const)('revalidates a legacy TEAM scan before enabling evidence (%s)', async order => {
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: false } } as import('../lib/types').ConfigResponse);
  accessMode.set('internal');
  vi.mocked(api.getScan).mockResolvedValueOnce({ schema_version: '1', scan: { ...fakeScan(), target: 'Private synthetic', ml_mask_count: 0 } });
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  expect(get(processingState)?.trace?.source.target).toBe('Private synthetic');
  const pending: Array<() => void> = [];
  vi.mocked(api.getScan).mockImplementation(() => new Promise(resolve => {
    pending.push(() => resolve({ schema_version: '1', scan: { ...fakeScan(), target: 'Public synthetic', ml_mask_count: 0 } }));
  }));
  const enable = () => appConfig.set({ tier: 'team', capabilities: { evidence_desk: true } } as import('../lib/types').ConfigResponse);
  if (order === 'access-first') accessMode.set('public');
  else enable();
  await flush();
  if (order === 'access-first') enable();
  else accessMode.set('public');
  expect(get(processingState)).toBeNull();
  await flush();
  expect(container.textContent).not.toContain('Private synthetic');
  expect(container.querySelector('.workbench-layout')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Pin current trace' })).toBeNull();
  let finishSpectrum = () => {};
  getAvg.mockResolvedValue(fakeSpectrum(INTENSITY_NEW));
  getAvg.mockReturnValueOnce(new Promise(resolve => {
    finishSpectrum = () => resolve(fakeSpectrum(INTENSITY_NEW));
  }));
  pending.at(-1)!(); await flush();
  expect(container.querySelector('.workbench-layout')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Pin current trace' })).toBeNull();
  expect(get(processingState)).toBeNull();
  finishSpectrum(); await flush();
  expect(container.textContent).toContain('Public synthetic');
  expect(screen.getByRole('button', { name: 'Pin current trace' })).toBeTruthy();
  expect(get(processingState)?.trace?.source.target).toBe('Public synthetic');
  expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  // An earlier enablement load cannot overwrite the settled public load.
  if (pending.length > 1) { pending[0](); await flush(); }
  expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
  expect(api.getScanPoints).toHaveBeenCalledTimes(2);
});


it.each(['loaded', 'scan-pending'] as const)('reloads through the legacy path after capability removal (%s)', async initial => {
  let finishInitial = () => {};
  if (initial === 'scan-pending') {
    vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
      finishInitial = () => resolve({ schema_version: '1', scan: { ...fakeScan(), target: 'Stale synthetic' } });
    }));
  }
  const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  let finishScan = () => {};
  let finishPoints = () => {};
  vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
    finishScan = () => resolve({ schema_version: '1', scan: { ...fakeScan(), target: 'Legacy synthetic', ml_mask_count: 0 } });
  }));
  vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise(resolve => {
    finishPoints = () => resolve({ schema_version: '1', scan_id: SCAN_ID, points: [fakePoint()], n_points: 1 });
  }));
  getAvg.mockResolvedValue(fakeSpectrum(INTENSITY_NEW));
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: false } } as import('../lib/types').ConfigResponse);
  accessMode.set('public');
  expect(get(processingState)).toBeNull();
  await flush();
  expect(api.getScan).toHaveBeenCalledTimes(2);
  finishScan(); await flush();
  // Capability removal uses the legacy metadata assignment before points land.
  expect(container.textContent).toContain('Legacy synthetic');
  expect(container.textContent).toContain('Loading scan');
  finishPoints(); await flush(); finishInitial(); await flush();
  expect(container.textContent).not.toContain('Loading scan');
  expect(container.textContent).not.toContain('Stale synthetic');
  expect(container.querySelector('.workbench-layout')).toBeTruthy();
  expect(container.querySelector('.aci-viewer')).toBeTruthy();
  expect(container.querySelector('.raman-view')).toBeTruthy();
  expect(container.querySelector('[aria-label="Evidence desk"]')).toBeNull();
  expect(container.querySelector('.export-trigger')).not.toBeDisabled();
  expect(get(processingState)?.trace?.source.target).toBe('Legacy synthetic');
  expect(get(processingState)?.raman.intensity).toEqual(INTENSITY_NEW);
});


it.each([
  ['getScan', 'session'], ['getScanPoints', 'session'], ['getAverageSpectrum', 'session'],
  ['getScan', 'destroy'], ['getScanPoints', 'destroy'], ['getAverageSpectrum', 'destroy'],
] as const)('abandons capability-removal recovery with %s pending on %s', async (pendingMethod, ended) => {
  vi.mocked(api.getScan).mockResolvedValue({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
  const view = render(ProcessingWorkbench, { scanId: SCAN_ID }); await flush();
  let finish = () => {};
  if (pendingMethod === 'getScan') {
    vi.mocked(api.getScan).mockReturnValueOnce(new Promise(resolve => {
      finish = () => resolve({ schema_version: '1', scan: { ...fakeScan(), ml_mask_count: 0 } });
    }));
  } else if (pendingMethod === 'getScanPoints') {
    vi.mocked(api.getScanPoints).mockReturnValueOnce(new Promise(resolve => {
      finish = () => resolve({ schema_version: '1', scan_id: SCAN_ID, points: [fakePoint()], n_points: 1 });
    }));
  } else {
    getAvg.mockReturnValueOnce(new Promise(resolve => { finish = () => resolve(fakeSpectrum(INTENSITY_STALE)); }));
  }
  appConfig.set({ tier: 'team', capabilities: { evidence_desk: false } } as import('../lib/types').ConfigResponse);
  await flush();
  const pointCalls = vi.mocked(api.getScanPoints).mock.calls.length;
  const spectrumCalls = getAvg.mock.calls.length;
  const annotationCalls = vi.mocked(api.getBadpixChannels).mock.calls.length;
  if (ended === 'session') clearSessionState();
  else { view.unmount(); processingState.set(null); }
  finish(); await flush();
  expect(get(processingState)).toBeNull();
  expect(api.getScanPoints).toHaveBeenCalledTimes(pointCalls);
  expect(getAvg).toHaveBeenCalledTimes(spectrumCalls);
  expect(api.getBadpixChannels).toHaveBeenCalledTimes(annotationCalls);
});
