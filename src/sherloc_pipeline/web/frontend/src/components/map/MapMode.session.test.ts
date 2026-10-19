import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import { get } from 'svelte/store';
import MapMode from './MapMode.svelte';
import * as api from '../../lib/api';
import { clearSessionState } from '../../lib/stores';
import { mapPointSet, resetMapState } from '../../lib/stores/mapStore';

beforeEach(() => {
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
