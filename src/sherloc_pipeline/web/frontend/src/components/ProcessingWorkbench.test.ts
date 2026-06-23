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
import { processingState } from '../lib/processingStore';
import type { BadpixResponse, SpectrumParams, SpectrumResponse } from '../lib/types';
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
    sol_number: 921,
    target: 'Amherst_Point',
    scan_name: 'detail_1',
    scan_id: '0921_Amherst_Point_detail_1',
    n_points: 1,
    n_channels: 3,
    shots_per_point: 50,
    laser_wavelength_nm: 248.5794,
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
    vi.spyOn(api, 'getBadpixChannels').mockImplementation((region: string) =>
      region === 'R1' ? r1.promise : r2.promise,
    );

    const { container } = render(ProcessingWorkbench, { scanId: SCAN_ID });
    await screen.findByText('1. Despike');
    await flush();
    // Mount's R1 badpix fetch is still pending; switch to R2 before it lands.
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
    vi.spyOn(api, 'getBadpixChannels').mockImplementation((region: string) =>
      region === 'R1' ? r1.promise : r2.promise,
    );

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
