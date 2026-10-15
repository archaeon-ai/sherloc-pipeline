import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { get } from 'svelte/store';
import { getScans, getScanPoints } from '../lib/api';
import { currentHash } from '../lib/stores';
import type { ScanListItem, ScanListResponse, ScanPointsResponse } from '../lib/types';
import ScanResolver from './ScanResolver.svelte';

vi.mock('../lib/api', () => ({ getScans: vi.fn(), getScanPoints: vi.fn() }));

function scan(id: string): ScanListItem {
  return {
    id, scan_id: `test-${id}`, scan_name: 'Testrock_detail', sol_number: 42424,
    target: 'Testrock', n_points: 3, scan_class: 'primary', scan_type: 'detail',
    target_type: 'mars_target', processing_status: null, processed_at: null,
    processing_pipeline_version: null,
  };
}

function response(scans: ScanListItem[], total = scans.length): ScanListResponse {
  return { schema_version: '1.0.0', scans, total, offset: 0, limit: 200 };
}

const params = { scan_name: 'Testrock_detail', sol: '42424', point: '0' };

beforeEach(() => {
  vi.resetAllMocks();
  window.history.replaceState(null, '', '#/open?scan_name=Testrock_detail&sol=42424&point=0');
  currentHash.set(window.location.hash);
  vi.mocked(getScans).mockResolvedValue(response([scan('scan-a')]));
  vi.mocked(getScanPoints).mockResolvedValue({
    points: [0, 2, 7].map((point_index) => ({ point_index })),
  } as ScanPointsResponse);
});

afterEach(cleanup);

describe('ScanResolver', () => {
  it('resolves a single match, validates point zero, and replaces the route', async () => {
    const replace = vi.spyOn(window.history, 'replaceState');
    render(ScanResolver, { queryParams: params });
    await waitFor(() => expect(get(currentHash)).toBe('#/scan/scan-a/workbench?point=0'));
    expect(getScans).toHaveBeenCalledWith({ scan_name: 'Testrock_detail', sol: 42424, offset: 0, limit: 200 });
    expect(getScanPoints).toHaveBeenCalledWith('scan-a');
    expect(window.location.hash).toBe('#/scan/scan-a/workbench?point=0');
    expect(replace).toHaveBeenCalledWith(null, '', '#/scan/scan-a/workbench?point=0');
    replace.mockRestore();
  });

  it('passes a subset through to the Workbench', async () => {
    render(ScanResolver, { queryParams: { scan_name: 'Testrock_detail', sol: '42424', points: '7,0,7' } });
    await waitFor(() => expect(get(currentHash)).toBe('#/scan/scan-a/workbench?points=7,0'));
  });

  it('shows a tier-specific no-match message', async () => {
    vi.mocked(getScans).mockResolvedValue(response([]));
    render(ScanResolver, { queryParams: params });
    expect(await screen.findByRole('alert')).toHaveTextContent('No scan found for "Testrock_detail" on sol 42424 in this tier.');
    expect(getScanPoints).not.toHaveBeenCalled();
    expect(get(currentHash)).toContain('#/open?');
  });

  it('lists matches across pages and waits for an explicit choice', async () => {
    vi.mocked(getScans)
      .mockResolvedValueOnce(response([scan('scan-a')], 2))
      .mockResolvedValueOnce(response([scan('scan-b')], 2));
    render(ScanResolver, { queryParams: params });
    const choice = await screen.findByRole('button', { name: /ID: scan-b/ });
    expect(screen.getByText(/More than one scan matches/)).toBeInTheDocument();
    expect(getScans).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 1 }));
    expect(getScanPoints).not.toHaveBeenCalled();
    expect(get(currentHash)).toContain('#/open?');
    await fireEvent.click(choice);
    await waitFor(() => expect(get(currentHash)).toBe('#/scan/scan-b/workbench?point=0'));
  });

  it('rejects an unavailable point without navigating', async () => {
    render(ScanResolver, { queryParams: { ...params, point: '1' } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Point index out of range for this scan: 1');
    expect(get(currentHash)).toContain('#/open?');
  });

  it('rejects malformed links before requesting scans', async () => {
    render(ScanResolver, { queryParams: { ...params, point: '0tail' } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Point index must be a nonnegative whole number.');
    expect(getScans).not.toHaveBeenCalled();
  });

  it('shows lookup failures', async () => {
    vi.mocked(getScans).mockRejectedValue(new Error('Unable to connect to the scan service.'));
    render(ScanResolver, { queryParams: params });
    expect(await screen.findByRole('alert')).toHaveTextContent('Unable to connect');
  });

  it('does not redirect after the user leaves while point validation is pending', async () => {
    let finish!: (value: ScanPointsResponse) => void;
    vi.mocked(getScanPoints).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const { unmount } = render(ScanResolver, { queryParams: params });
    await waitFor(() => expect(getScanPoints).toHaveBeenCalled());
    unmount();
    currentHash.set('#/');
    finish({ points: [{ point_index: 0 }] } as ScanPointsResponse);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(get(currentHash)).toBe('#/');
  });
});
