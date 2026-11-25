import { fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as api from '../lib/api';
import FittingWorkspace from './FittingWorkspace.svelte';

describe('Fitting workspace model selection', () => {
  beforeEach(() => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({ scan: { target: 'Sample' } })))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        wavenumber: [800, 900, 1000], intensity: [10, 15, 11],
      })));
    render(FittingWorkspace, { props: { scanId: 'sample' } });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('starts on AICc and offers both methods', async () => {
    const select = await screen.findByRole<HTMLSelectElement>('combobox', { name: 'Model Selection' });
    expect(Array.from(select.options, option => [option.value, option.label])).toEqual([
      ['f-test', 'F-test'],
      ['aicc', 'AICc'],
    ]);
    expect(select).toHaveValue('aicc');
  });

  it.each(['aicc', 'f-test'])('sends %s in the mineral fit request', async method => {
    const postFit = vi.spyOn(api, 'postFit').mockReturnValue(new Promise(() => {}));
    const fitButton = await screen.findByRole('button', { name: 'Fit Spectrum' });
    if (method === 'f-test') {
      await fireEvent.change(screen.getByRole('combobox', { name: 'Model Selection' }), {
        target: { value: method },
      });
    }
    await fireEvent.click(fitButton);
    expect(postFit.mock.calls[0]?.[0].params?.fitting).toMatchObject({
      domain: 'minerals', model_selection: method,
    });
  });
});
