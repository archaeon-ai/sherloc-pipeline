import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render } from '@testing-library/svelte';
import { tick } from 'svelte';
import BaselineStep from './BaselineStep.svelte';
import * as api from '../lib/api';
import type { BaselineResponse } from '../lib/types';

const WAVENUMBER = [700, 800, 900, 1000, 1100];
const INTENSITY = [10, 20, 30, 40, 50];
const VISIBLE_RANGE: [number, number] = [750, 1050];

beforeEach(() => {
  vi.restoreAllMocks();
});

describe('BaselineStep request freshness', () => {
  it('ignores a superseded range response that resolves last', async () => {
    let resolveVisible: (response: BaselineResponse) => void = () => {};
    const visibleRequest = new Promise<BaselineResponse>((resolve) => {
      resolveVisible = resolve;
    });
    let resolveFull: (response: BaselineResponse) => void = () => {};
    const fullRequest = new Promise<BaselineResponse>((resolve) => {
      resolveFull = resolve;
    });
    const postBaselineSpy = vi
      .spyOn(api, 'postBaseline')
      .mockReturnValueOnce(visibleRequest)
      .mockReturnValueOnce(fullRequest);

    const apply = vi.fn();
    const { container, component } = render(BaselineStep, {
      props: {
        wavenumber: WAVENUMBER,
        intensity: INTENSITY,
        enabled: true,
        collapsed: false,
        visibleRange: VISIBLE_RANGE,
      },
    });
    component.$on('apply', (event) => apply(event.detail));

    const visibleRadio = container.querySelector<HTMLInputElement>(
      'input[name="baseline-range"][value="visible"]',
    )!;
    const fullRadio = container.querySelector<HTMLInputElement>(
      'input[name="baseline-range"][value="full"]',
    )!;

    await fireEvent.change(visibleRadio);
    await fireEvent.change(fullRadio);
    expect(postBaselineSpy).toHaveBeenCalledTimes(2);
    expect(postBaselineSpy.mock.calls[0][0].params?.wavenumber_range).toEqual(
      VISIBLE_RANGE,
    );
    expect(postBaselineSpy.mock.calls[1][0].params?.wavenumber_range).toBeUndefined();

    resolveFull({
      schema_version: '1',
      raw: INTENSITY,
      wavenumber: WAVENUMBER,
      corrected: [1, 2, 3, 4, 5],
      baseline: [9, 18, 27, 36, 45],
      params_used: { method: 'aspls', lam: 1e6, max_iter: 10 },
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tick();

    expect(apply).toHaveBeenCalledTimes(1);
    expect(apply.mock.calls[0][0].params.wavenumber_range).toBeUndefined();

    resolveVisible({
      schema_version: '1',
      raw: INTENSITY,
      wavenumber: WAVENUMBER,
      corrected: [10, 2, 3, 4, 50],
      baseline: [0, 18, 27, 36, 0],
      params_used: {
        method: 'aspls',
        lam: 1e6,
        max_iter: 10,
        wavenumber_range: VISIBLE_RANGE,
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    await tick();

    expect(apply).toHaveBeenCalledTimes(1);
    expect(container.querySelector('.step-badge')?.textContent).toContain('full');
  });
});
