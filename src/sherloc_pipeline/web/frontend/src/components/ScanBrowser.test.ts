import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import ScanBrowser from './ScanBrowser.svelte';
import { getScans } from '../lib/api';
import { currentHash } from '../lib/stores';
import type { ScanListResponse } from '../lib/types';

vi.mock('../lib/api', () => ({
  getScans: vi.fn(),
}));

const emptyResponse: ScanListResponse = {
  schema_version: '1.0.0',
  scans: [],
  total: 0,
  offset: 0,
  limit: 50,
};

const responseWithScan: ScanListResponse = {
  ...emptyResponse,
  scans: [
    {
      id: 'scan-1',
      sol_number: 921,
      target: 'Amherst_Point',
      scan_name: 'detail_1',
      scan_id: '0921_Amherst_Point_detail_1',
      n_points: 10,
      scan_class: 'primary',
      scan_type: 'detail',
      target_type: 'mars_target',
      processing_status: null,
      processed_at: null,
      processing_pipeline_version: null,
    },
  ],
  total: 1,
};

const getScansMock = vi.mocked(getScans);

beforeEach(() => {
  getScansMock.mockReset();
  getScansMock.mockResolvedValue(emptyResponse);
  sessionStorage.clear();
  window.history.replaceState(null, '', '#/');
  currentHash.set('#/');
});

afterEach(() => {
  cleanup();
});

describe('ScanBrowser filters and sorting', () => {
  it('submits sol range and target filters through the Enter-enabled form', async () => {
    render(ScanBrowser);
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(1));

    await fireEvent.input(screen.getByLabelText('Sol From'), { target: { value: '900' } });
    await fireEvent.input(screen.getByLabelText('Sol To'), { target: { value: '950' } });
    await fireEvent.input(screen.getByLabelText('Target'), { target: { value: 'Amherst' } });
    await fireEvent.submit(screen.getByRole('button', { name: 'Filter' }).closest('form')!);

    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(2));
    expect(getScansMock).toHaveBeenLastCalledWith({
      offset: 0,
      limit: 50,
      sol_from: 900,
      sol_to: 950,
      target: 'Amherst',
    });
    expect(window.location.hash).toContain('sol_from=900');
    expect(window.location.hash).toContain('sol_to=950');
  });

  it('toggles Sol and Target header sort direction', async () => {
    render(ScanBrowser);
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(1));

    await fireEvent.click(screen.getByRole('button', { name: /^Sol/ }));
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(2));
    expect(getScansMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ sort_by: 'sol', sort_order: 'asc' }),
    );

    await fireEvent.click(screen.getByRole('button', { name: /^Sol/ }));
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(3));
    expect(getScansMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ sort_by: 'sol', sort_order: 'desc' }),
    );

    await fireEvent.click(screen.getByRole('button', { name: /^Target/ }));
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(4));
    expect(getScansMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ sort_by: 'target', sort_order: 'asc' }),
    );
  });

  it('restores the applied filter after opening a scan and returning via #/', async () => {
    getScansMock.mockResolvedValue(responseWithScan);
    const first = render(ScanBrowser);
    await waitFor(() => expect(screen.getByText('detail_1')).toBeInTheDocument());

    await fireEvent.input(screen.getByLabelText('Target'), { target: { value: 'Amherst' } });
    await fireEvent.submit(screen.getByRole('button', { name: 'Filter' }).closest('form')!);
    await waitFor(() => expect(getScansMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ target: 'Amherst' }),
    ));
    await fireEvent.click(screen.getByText('detail_1').closest('tr')!);
    expect(window.location.hash).toBe('#/scan/scan-1/workbench');

    first.unmount();
    window.history.replaceState(null, '', '#/');
    currentHash.set('#/');
    getScansMock.mockClear();
    render(ScanBrowser);

    expect(screen.getByLabelText('Target')).toHaveValue('Amherst');
    await waitFor(() => expect(getScansMock).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'Amherst' }),
    ));
  });
});
