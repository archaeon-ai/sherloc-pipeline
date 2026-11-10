import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import ScanBrowser from './ScanBrowser.svelte';
import { getScans } from '../lib/api';
import { accessModeResolved, currentHash } from '../lib/stores';
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
      has_fitted_peaks: false,
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
  accessModeResolved.set(true);
});

afterEach(() => {
  cleanup();
});

describe('ScanBrowser filters and sorting', () => {
  it('waits for access-mode routing before restoring a saved hash', async () => {
    accessModeResolved.set(false);
    sessionStorage.setItem('sherloc.scanBrowserHash', '#/?target=Amherst');

    render(ScanBrowser);
    expect(window.location.hash).toBe('#/');
    expect(getScansMock).not.toHaveBeenCalled();

    accessModeResolved.set(true);

    await waitFor(() => expect(screen.getByLabelText('Target')).toHaveValue('Amherst'));
    await waitFor(() => expect(getScansMock).toHaveBeenCalledWith(
      expect.objectContaining({ target: 'Amherst' }),
    ));
  });

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

describe('ScanBrowser fit status', () => {
  const twoScans: ScanListResponse = {
    ...emptyResponse,
    scans: [
      { ...responseWithScan.scans[0], id: 'fitted', scan_name: 'detail_1', has_fitted_peaks: true },
      { ...responseWithScan.scans[0], id: 'unfitted', scan_name: 'detail_2', has_fitted_peaks: false },
    ],
    total: 2,
  };

  it('labels each scan from has_fitted_peaks, not the stored status column', async () => {
    getScansMock.mockResolvedValue({
      ...twoScans,
      // The stored column must not drive the label.
      scans: twoScans.scans.map((s) => ({ ...s, processing_status: 'completed' as const })),
    });
    render(ScanBrowser);
    await waitFor(() => expect(screen.getByText('detail_2')).toBeInTheDocument());

    // Status is the seventh column (Sol, Target, Scan Name, Points, Class, Type, Status).
    const status = (name: string) => screen.getByText(name).closest('tr')!.querySelectorAll('td')[6];
    expect(status('detail_1')).toHaveTextContent(/^peaks fitted$/);
    expect(status('detail_2')).toHaveTextContent(/^no fitted peaks$/);
    expect(screen.queryByText('unprocessed')).toBeNull();
    expect(screen.queryByText('completed')).toBeNull();
  });

  it('offers exactly All / peaks fitted / no fitted peaks and filters on has_fitted_peaks', async () => {
    render(ScanBrowser);
    await waitFor(() => expect(getScansMock).toHaveBeenCalledTimes(1));

    const select = screen.getByLabelText('Status') as HTMLSelectElement;
    expect([...select.options].map((o) => o.textContent?.trim())).toEqual([
      'All',
      'peaks fitted',
      'no fitted peaks',
    ]);

    for (const [label, value] of [['peaks fitted', true], ['no fitted peaks', false]] as const) {
      const option = [...select.options].find((o) => o.textContent?.trim() === label)!;
      await fireEvent.change(select, { target: { value: option.value } });
      await fireEvent.submit(screen.getByRole('button', { name: 'Filter' }).closest('form')!);
      await waitFor(() => expect(getScansMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ has_fitted_peaks: value }),
      ));
      expect(getScansMock.mock.lastCall![0]).not.toHaveProperty('processing_status');
      expect(window.location.hash).toContain(`has_fitted_peaks=${value}`);
    }
  });

  it('restores the fit-status filter from the hash', async () => {
    window.history.replaceState(null, '', '#/?has_fitted_peaks=false');
    currentHash.set('#/?has_fitted_peaks=false');
    render(ScanBrowser);
    await waitFor(() => expect(getScansMock).toHaveBeenCalledWith(
      expect.objectContaining({ has_fitted_peaks: false }),
    ));
    const select = screen.getByLabelText('Status') as HTMLSelectElement;
    expect(select.selectedOptions[0].textContent?.trim()).toBe('no fitted peaks');
  });
});
