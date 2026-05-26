// Frontend component tests for MapSpectrumPanel single-point fetch.
//
// Issue #17 — "Failed to load point spectrum" on Fit Hydration point click:
// Map mode's single-point fetch was using raw `fetch()` instead of the
// authed `getPointSpectrum()` helper. Under Auth0 Bearer-token mode any
// session-cookie shortfall (transient renewal, expiry, fresh-tab race)
// collapsed every failure into the generic "Failed to load point spectrum"
// red banner. The fix wires the panel to `getPointSpectrum()` (which
// gates on `ensureAuthenticated()` → AuthRequiredError) so an auth
// shortfall renders the same "Log in to view spectrum" placeholder the
// class-average path already uses.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import { tick } from 'svelte';
import MapSpectrumPanel from './MapSpectrumPanel.svelte';
import { AuthRequiredError, ApiError } from '../../lib/api';
import * as api from '../../lib/api';

const SCAN_ID = 'ae5578c9-5a91-41c9-8431-190117be23b4';

const baseResponse = {
  schema_version: '1',
  scan_id: SCAN_ID,
  region: 'R1',
  wavenumber: [100, 200, 300],
  intensity: [1.1, 1.2, 1.3],
  n_channels: 3,
  provenance: {
    calibration_version: 'test',
    wavenumber_unit: 'cm-1',
    intensity_unit: 'counts',
  },
};

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('MapSpectrumPanel — single-point fetch (issue #17)', () => {
  it('renders the spectrum plot on a successful point fetch', async () => {
    const spy = vi.spyOn(api, 'getPointSpectrum').mockResolvedValue(baseResponse);

    render(MapSpectrumPanel, {
      props: { scanId: SCAN_ID, mode: 'single_point', pointIndex: 91 },
    });

    // Let the reactive `$:` block fire + the awaited helper resolve.
    await tick();
    await Promise.resolve();
    await tick();

    expect(spy).toHaveBeenCalledWith(SCAN_ID, 91, { region: 'R1' });
    expect(await screen.findByText('Point 91')).toBeInTheDocument();
    expect(screen.queryByText('Log in to view spectrum')).toBeNull();
    expect(screen.queryByText(/Failed to load/)).toBeNull();
  });

  it('renders the auth placeholder when the helper throws AuthRequiredError', async () => {
    vi.spyOn(api, 'getPointSpectrum').mockRejectedValue(new AuthRequiredError());

    render(MapSpectrumPanel, {
      props: { scanId: SCAN_ID, mode: 'single_point', pointIndex: 91 },
    });

    expect(await screen.findByText('Log in to view spectrum')).toBeInTheDocument();
    expect(screen.queryByText('Failed to load point spectrum')).toBeNull();
  });

  it('renders the ApiError message when the backend returns a non-2xx', async () => {
    vi.spyOn(api, 'getPointSpectrum').mockRejectedValue(
      new ApiError(404, 'HTTP 404', 'Spectrum not found for this point'),
    );

    render(MapSpectrumPanel, {
      props: { scanId: SCAN_ID, mode: 'single_point', pointIndex: 91 },
    });

    expect(
      await screen.findByText('HTTP 404: Spectrum not found for this point'),
    ).toBeInTheDocument();
    expect(screen.queryByText('Log in to view spectrum')).toBeNull();
  });
});
