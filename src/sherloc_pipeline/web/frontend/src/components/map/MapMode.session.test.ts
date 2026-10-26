import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { render, waitFor, fireEvent, screen } from '@testing-library/svelte';
import { get } from 'svelte/store';
import MapMode from './MapMode.svelte';
import * as api from '../../lib/api';
import { accessMode, clearSessionState } from '../../lib/stores';
import { mapPointSet, mapLogEntries, resetMapState } from '../../lib/stores/mapStore';

const socket = vi.hoisted(() => ({ handlers: null as import('../../lib/mapWebSocket').MapWSHandlers | null }));
vi.mock('../../lib/mapWebSocket', () => ({
  MapWebSocket: class {
    constructor(_url: string, handlers: import('../../lib/mapWebSocket').MapWSHandlers) { socket.handlers = handlers; }
    close() {}
  },
}));

beforeEach(() => {
  accessMode.set('internal');
  socket.handlers = null;
  resetMapState();
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  vi.spyOn(api, 'fetchAciImage').mockResolvedValue(new Image());
  vi.spyOn(api, 'getMapLayers').mockResolvedValue({
    point_set: { points: [], voronoi: null }, available_layers: {}, base_images: [],
  } as never);
});
afterEach(() => { vi.restoreAllMocks(); });

it('does not start another map request after teardown during scan lookup', async () => {
  let reject!: (reason: Error) => void;
  vi.spyOn(api, 'getScan').mockReturnValue(new Promise((_, r) => { reject = r; }));
  const component = render(MapMode, { scanId: 'synthetic-scan' });
  await waitFor(() => expect(api.getScan).toHaveBeenCalled());
  component.unmount();
  clearSessionState();
  reject(new api.AuthRequiredError());
  await new Promise((r) => setTimeout(r, 0));
  expect(api.getMapLayers).not.toHaveBeenCalled();
  expect(get(mapPointSet)).toBeNull();
});

it('ignores layer metadata received after teardown and logout', async () => {
  vi.spyOn(api, 'getScan').mockResolvedValue({ scan: {
    target: 'Synthetic', sol_number: 9999, scan_name: 'synthetic-scan',
  } } as never);
  let resolve!: (value: never) => void;
  vi.mocked(api.getMapLayers).mockReturnValue(new Promise((r) => { resolve = r; }));
  const component = render(MapMode, { scanId: 'synthetic-scan' });
  await waitFor(() => expect(api.getMapLayers).toHaveBeenCalled());
  component.unmount();
  clearSessionState();
  resolve({ point_set: { points: [], voronoi: null }, available_layers: {} } as never);
  await new Promise((r) => setTimeout(r, 0));
  expect(get(mapPointSet)).toBeNull();
});


it('does not refill logs after result recovery rejects during logout', async () => {
  vi.spyOn(api, 'getScan').mockResolvedValue({ scan: {
    target: 'Synthetic', sol_number: 9999, scan_name: 'synthetic-scan',
  } } as never);
  vi.mocked(api.getMapLayers).mockResolvedValue({
    point_set: { points: [{ index: 0, x: 1, y: 1 }], voronoi: null },
    available_layers: { sherloc: { minerals: { n_detections: 0, classes: [] } } }, base_images: [],
  } as never);
  vi.spyOn(api, 'startMapFit').mockResolvedValue({ job_id: 'synthetic-job', n_points: 1, ws_url: '/synthetic' });
  let reject!: (reason: Error) => void;
  vi.spyOn(api, 'getMapJobResults').mockReturnValue(new Promise((_, r) => { reject = r; }));
  const component = render(MapMode, { scanId: 'synthetic-scan' });
  await waitFor(() => expect(screen.getByText('1 points')).toBeInTheDocument());
  await fireEvent.click(screen.getByText('Start Fitting'));
  await waitFor(() => expect(socket.handlers).not.toBeNull());
  socket.handlers!.onComplete({ summary: { total_points: 1 } } as never);
  await waitFor(() => expect(api.getMapJobResults).toHaveBeenCalled());
  component.unmount();
  clearSessionState();
  reject(new api.AuthRequiredError());
  socket.handlers!.onLog({ message: 'synthetic late callback' } as never);
  await new Promise((r) => setTimeout(r, 0));
  expect(get(mapLogEntries)).toEqual([]);
  expect(get(mapPointSet)).toBeNull();
});
