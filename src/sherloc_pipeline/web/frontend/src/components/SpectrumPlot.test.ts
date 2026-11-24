// SpectrumPlot loads Plotly with a dynamic import. A component destroyed
// before that import settles must not raise: neither from a removed plot
// element nor from an import that rejects after teardown.

import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/svelte';

const plotlyLoad = vi.hoisted(() => {
  let reject: (err: Error) => void = () => {};
  const promise = new Promise<never>((_, r) => {
    reject = r;
  });
  return { promise, reject: (err: Error) => reject(err) };
});

vi.mock('plotly.js-basic-dist-min', () => plotlyLoad.promise);

import SpectrumPlot from './SpectrumPlot.svelte';

describe('SpectrumPlot — destroyed before Plotly loads', () => {
  it('ignores a Plotly import that rejects after the component is destroyed', async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      const { unmount } = render(SpectrumPlot, {
        props: { wavenumber: [100, 200, 300], intensity: [1, 2, 3] },
      });
      unmount();
      plotlyLoad.reject(new ReferenceError('self is not defined'));
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});
