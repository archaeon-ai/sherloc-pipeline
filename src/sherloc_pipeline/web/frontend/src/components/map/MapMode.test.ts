// Frontend component tests for Map Mode fit-result recovery on terminal
// WebSocket statuses.
//
// Issue #6 — map-mode fitting stalls on some scans. Fitting streams its
// per-point results and never persists them, so a frame the socket does
// not deliver exists only in the server's retention store. Completion
// already pulled those back; failure and cancellation did not, which lost
// valid measurements in two real cases:
//
//   * cancel — the WS handler acknowledges the cancel and closes the
//     socket immediately, so every point frame still queued behind the
//     acknowledgement is dropped; and
//   * failure after a reconnect that outran the bounded replay buffer,
//     which leaves a hole nothing else fills.
//
// Both leave map points blank as though they were never measured. These
// tests drive the WS handlers MapMode installs and assert the retained
// results are fetched and ingested for every terminal status.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, screen, waitFor } from '@testing-library/svelte';
import { get } from 'svelte/store';
import MapMode from './MapMode.svelte';
import * as api from '../../lib/api';
import {
  mapFitJob,
  mapLayers,
  mapLogEntries,
  resetMapState,
} from '../../lib/stores/mapStore';
import type { MapWSHandlers } from '../../lib/mapWebSocket';
import type { MapJobFitPoint, MapJobResults, WSPointFitted } from '../../lib/types/map';
import type { ScanDetailResponse } from '../../lib/types';

const SCAN_ID = 'ae5578c9-5a91-41c9-8431-190117be23b4';
const JOB_ID = 'job-6-terminal-recovery';

// Hoisted so the (hoisted) vi.mock factory can reach it — the factory runs
// while MapMode's own imports are resolving, before this module's body.
const wsMock = vi.hoisted(() => ({
  handlers: null as MapWSHandlers | null,
  closed: false,
}));

vi.mock('../../lib/mapWebSocket', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../lib/mapWebSocket')>();
  return {
    ...actual,
    // Stands in for the real socket: jsdom has no WebSocket server, and
    // the behaviour under test is what MapMode does with the frames, not
    // how they arrive.
    MapWebSocket: class {
      constructor(_url: string, handlers: MapWSHandlers) {
        wsMock.handlers = handlers;
      }
      sendCancel(): boolean {
        return true;
      }
      close(): void {
        wsMock.closed = true;
      }
      get isReconnecting(): boolean {
        return false;
      }
    },
  };
});

/** One fitted point carrying a single identified mineral peak. */
function fitPoint(pointIndex: number, snr: number): MapJobFitPoint {
  return {
    point_index: pointIndex,
    x: pointIndex * 10,
    y: 0,
    results: {
      minerals: {
        status: 'measured',
        peaks: [{ center_cm1: 1088, snr, assignment: 'hi-carb' }],
      },
    },
  };
}

function pointFrame(pointIndex: number, snr: number, seq: number): WSPointFitted {
  return { type: 'point_fitted', seq, ...fitPoint(pointIndex, snr) } as WSPointFitted;
}

function retained(points: MapJobFitPoint[], truncated = false): MapJobResults {
  return {
    job_id: JOB_ID,
    status: 'cancelled',
    fitted: points.length,
    total: 3,
    truncated,
    points,
  };
}

/** The `hi-carb` class layer MapMode builds for a minerals fit. */
function hiCarbValues() {
  const layer = get(mapLayers).find(
    (l) => l.domain === 'minerals' && l.class_id === 'hi-carb',
  );
  expect(layer, 'minerals/hi-carb layer').toBeDefined();
  return layer!.values;
}

/**
 * Render Map Mode, start a minerals fit, and hand back the WS handlers
 * MapMode installed.
 */
async function startFit(): Promise<MapWSHandlers> {
  render(MapMode, { props: { scanId: SCAN_ID } });
  // onMount resolves the ACI fetch, the scan label and the layer metadata
  // before the point set exists — the fit needs the point set.
  await waitFor(() => expect(screen.getByText('3 points')).toBeInTheDocument());

  await fireEvent.click(screen.getByText('Start Fitting'));
  await waitFor(() => expect(wsMock.handlers).not.toBeNull());
  return wsMock.handlers!;
}

beforeEach(() => {
  vi.restoreAllMocks();
  wsMock.handlers = null;
  wsMock.closed = false;
  resetMapState();

  // jsdom has no 2D context; MapCanvas already no-ops on a null one, but
  // the stub keeps the "not implemented" noise out of the test output.
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  // Nothing here decodes an image; the failure path leaves aciImage null
  // and MapCanvas renders its placeholder.
  vi.spyOn(api, 'fetchAciImage').mockRejectedValue(new Error('no ACI in jsdom'));
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(api, 'getScan').mockResolvedValue({
    scan: { target: 'Test', sol_number: 1000, scan_name: 'test_scan' },
  } as unknown as ScanDetailResponse);
  vi.spyOn(api, 'getMapLayers').mockResolvedValue({
    coordinate_source: 'aci_pixel',
    point_set: {
      points: [
        { index: 0, x: 0, y: 0 },
        { index: 1, x: 10, y: 0 },
        { index: 2, x: 20, y: 0 },
      ],
      voronoi: null,
    },
    base_images: [],
  });
  vi.spyOn(api, 'startMapFit').mockResolvedValue({
    job_id: JOB_ID,
    n_points: 3,
    ws_url: `/api/map/ws/${JOB_ID}`,
  });
});

afterEach(() => {
  resetMapState();
});

describe('MapMode — recovering results on terminal fit statuses (issue #6)', () => {
  it('recovers retained results when the job is cancelled', async () => {
    const results = vi
      .spyOn(api, 'getMapJobResults')
      .mockResolvedValue(retained([fitPoint(0, 12), fitPoint(1, 8), fitPoint(2, 9)]));

    const handlers = await startFit();

    // One point streamed; the other two are still queued server-side when
    // the cancel acknowledgement overtakes them.
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onCancelled();

    await waitFor(() => expect(results).toHaveBeenCalledWith(JOB_ID));
    await waitFor(() => expect(hiCarbValues()[2].status).toBe('measured'));

    const values = hiCarbValues();
    expect(values.map((v) => v.value)).toEqual([12, 8, 9]);
    expect(get(mapFitJob)?.status).toBe('cancelled');
    expect(get(mapFitJob)?.fitted).toBe(3);
    expect(get(mapLogEntries)).toContain(
      'Recovered 2 results the fit stream did not deliver.',
    );
  });

  it('recovers retained results when the job fails', async () => {
    const results = vi
      .spyOn(api, 'getMapJobResults')
      .mockResolvedValue(retained([fitPoint(0, 12), fitPoint(1, 8)]));

    const handlers = await startFit();

    // Point 1 was lost: the reconnect resumed past the replay buffer's
    // oldest retained frame, and then the job failed.
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onFailed({ type: 'job_failed', seq: 9, error: 'fit worker died' });

    await waitFor(() => expect(results).toHaveBeenCalledWith(JOB_ID));
    await waitFor(() => expect(hiCarbValues()[1].status).toBe('measured'));

    expect(hiCarbValues()[1].value).toBe(8);
    expect(get(mapFitJob)?.status).toBe('failed');
    expect(get(mapLogEntries)).toContain('ERROR: fit worker died');
    expect(get(mapLogEntries)).toContain(
      'Recovered 1 result the fit stream did not deliver.',
    );
  });

  it('says so when the server no longer holds the results of a cancelled job', async () => {
    const results = vi
      .spyOn(api, 'getMapJobResults')
      .mockRejectedValue(Object.assign(new Error('Job not found'), { status: 404 }));

    const handlers = await startFit();
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onCancelled();

    await waitFor(() => expect(results).toHaveBeenCalledWith(JOB_ID));
    await waitFor(() =>
      expect(get(mapLogEntries)).toContain(
        'Could not recover the results the fit stream did not deliver — the server no longer holds them. Re-run the fit to see the full map.',
      ),
    );
    // The point that did arrive is kept — recovery failing is not a reason
    // to discard a measurement this client already holds.
    expect(hiCarbValues()[0].value).toBe(12);
    expect(get(mapFitJob)?.status).toBe('cancelled');
  });

  it('reports a truncated retention store on cancellation', async () => {
    vi.spyOn(api, 'getMapJobResults').mockResolvedValue(
      retained([fitPoint(0, 12), fitPoint(1, 8)], true),
    );

    const handlers = await startFit();
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onCancelled();

    await waitFor(() =>
      expect(get(mapLogEntries)).toContain(
        'The server could not retain every point of this fit — the map may be incomplete. Re-run the fit to fill it in.',
      ),
    );
  });

  it('skips the request when completion delivered every point', async () => {
    const results = vi.spyOn(api, 'getMapJobResults').mockResolvedValue(retained([]));

    const handlers = await startFit();
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onPointFitted(pointFrame(1, 8, 2));
    handlers.onPointFitted(pointFrame(2, 9, 3));
    handlers.onComplete({
      type: 'job_complete',
      seq: 4,
      summary: { total_points: 3, detections: {}, elapsed_s: 1 },
    });

    await waitFor(() => expect(get(mapFitJob)?.status).toBe('complete'));
    expect(results).not.toHaveBeenCalled();
    expect(hiCarbValues().map((v) => v.value)).toEqual([12, 8, 9]);
  });

  it('recovers when completion reports more points than arrived', async () => {
    const results = vi
      .spyOn(api, 'getMapJobResults')
      .mockResolvedValue(retained([fitPoint(0, 12), fitPoint(1, 8), fitPoint(2, 9)]));

    const handlers = await startFit();
    handlers.onPointFitted(pointFrame(0, 12, 1));
    handlers.onComplete({
      type: 'job_complete',
      seq: 4,
      summary: { total_points: 3, detections: {}, elapsed_s: 1 },
    });

    await waitFor(() => expect(results).toHaveBeenCalledWith(JOB_ID));
    await waitFor(() => expect(hiCarbValues()[2].status).toBe('measured'));
    expect(get(mapFitJob)?.status).toBe('complete');
  });
});
