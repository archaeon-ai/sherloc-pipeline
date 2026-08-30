// Component-level coverage for the Workbench-routed spectrum panel.
// Helper-boundary tests in `lib/spectrumLabels.test.ts` lock the
// shape/annotation
// construction; this file locks that RamanView actually wires the
// helper into the Plotly layout AND emits the new axis-spine config
// AND renders two independent toggle inputs at the route the issue's
// reproduction exercises (#/scan/:id/workbench).
//
// jsdom does not implement Plotly internals; we mock the dynamic
// `import('plotly.js-basic-dist-min')` surface so we can assert on the
// `react(div, traces, layout, config)` call args.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte';
import { tick } from 'svelte';

// Spy that captures every Plotly.react call (the component re-runs
// renderPlot on toggle changes via afterUpdate).
const reactSpy = vi.fn();
const purgeSpy = vi.fn();
const downloadImageSpy = vi.fn();

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    react: reactSpy,
    purge: purgeSpy,
    downloadImage: downloadImageSpy,
  },
  // Some bundlers expose named exports too; cover both shapes.
  react: reactSpy,
  purge: purgeSpy,
  downloadImage: downloadImageSpy,
}));

import RamanView from './RamanView.svelte';
import type { Peak } from '../lib/types';

function fakePeak(overrides: Partial<Peak> = {}): Peak {
  return {
    center_cm1: 1016,
    center_uncertainty: null,
    amplitude: 100,
    amplitude_uncertainty: null,
    fwhm_cm1: 25,
    fwhm_uncertainty: null,
    area: null,
    snr: 5,
    fit_quality: null,
    mineral_assignment: 'sulf1',
    assignment_confidence: null,
    fit_modality: 'lorentz',
    sharpness_ratio: null,
    pass_sharpness: null,
    quality: null,
    ...overrides,
  };
}

beforeEach(() => {
  reactSpy.mockClear();
  purgeSpy.mockClear();
  downloadImageSpy.mockClear();
});

/** Drain microtasks + tick to let RamanView's onMount→render fire. */
async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) {
    await tick();
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 5));
  }
}

function lastLayout(): {
  xaxis?: { showline?: boolean; linecolor?: string; ticks?: string };
  yaxis?: { showline?: boolean; linecolor?: string; ticks?: string };
  shapes?: unknown[];
  annotations?: unknown[];
} {
  expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
  const args = reactSpy.mock.calls[reactSpy.mock.calls.length - 1];
  return args[2];
}

/** Most-recent traces array (react(div, traces, layout, config) → args[1]). */
function lastTraces(): Array<{ name?: string; mode?: string; x?: number[] }> {
  expect(reactSpy.mock.calls.length).toBeGreaterThan(0);
  const args = reactSpy.mock.calls[reactSpy.mock.calls.length - 1];
  return args[1] as Array<{ name?: string; mode?: string; x?: number[] }>;
}

/** The spike-marker trace (mode 'markers' with a triangle name), if rendered. */
function spikeTrace(): { name?: string; x?: number[] } | undefined {
  return lastTraces().find(
    (t) => t.mode === 'markers' && /Spikes|ML-masked/.test(t.name ?? ''),
  );
}

/** The known-noisy (badpix) marker trace, if rendered (issue #9). */
function badpixTrace():
  | { name?: string; x?: number[]; mode?: string; marker?: { symbol?: string; color?: string } }
  | undefined {
  return (
    lastTraces() as Array<{
      name?: string;
      x?: number[];
      mode?: string;
      marker?: { symbol?: string; color?: string };
    }>
  ).find((t) => t.mode === 'markers' && /Known-noisy/.test(t.name ?? ''));
}

describe('RamanView — issue #18 polish on the Workbench-routed component', () => {
  const wavenumber = Array.from({ length: 100 }, (_, i) => 640 + i * 10);
  const intensity = wavenumber.map((wn) => Math.exp(-((wn - 1016) ** 2) / 5000));

  it('Y-axis layout carries showline=true + outside ticks + slate-300 linecolor (AC #1)', async () => {
    render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raman_fitted',
        peaks: [],
        title: 'Amherst Point Sol 921 detail_1 — Point 91',
      },
    });
    await flush();

    const layout = lastLayout();
    expect(layout.yaxis?.showline).toBe(true);
    expect(layout.yaxis?.linecolor).toBe('#cbd5e1');
    expect(layout.yaxis?.ticks).toBe('outside');
    // X-axis gets the same treatment for visual parity (issue calls out
    // a missing Y-axis line; the X-axis didn't render a spine either in
    // pre-fix screenshots — silent symptom but same root cause).
    expect(layout.xaxis?.showline).toBe(true);
    expect(layout.xaxis?.linecolor).toBe('#cbd5e1');
    expect(layout.xaxis?.ticks).toBe('outside');
  });

  it('renders two independent toggles: "Fitted peaks" + "Peak labels" (AC #4)', async () => {
    const { getByLabelText } = render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raman_fitted',
        peaks: [fakePeak()],
      },
    });
    await flush();

    expect(getByLabelText(/Fitted peaks/i)).toBeDefined();
    expect(getByLabelText(/Peak labels/i)).toBeDefined();
  });

  it('toggling "Peak labels" off removes annotations but keeps shapes (independent — AC #5)', async () => {
    const { getByLabelText } = render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raman_fitted',
        peaks: [fakePeak()],
      },
    });
    await flush();

    // Default: both toggles on → shape + annotation both present.
    let layout = lastLayout();
    expect((layout.shapes ?? []).length).toBe(1);
    expect((layout.annotations ?? []).length).toBe(1);

    // Click "Peak labels" off.
    const labelsCheckbox = getByLabelText(/Peak labels/i) as HTMLInputElement;
    await fireEvent.click(labelsCheckbox);
    await flush();

    layout = lastLayout();
    expect((layout.shapes ?? []).length).toBe(1); // shapes preserved
    expect((layout.annotations ?? []).length).toBe(0); // annotations gone
  });

  it('toggling "Fitted peaks" off removes shapes but keeps labels (independent — AC #5)', async () => {
    const { getByLabelText } = render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raman_fitted',
        peaks: [fakePeak()],
      },
    });
    await flush();

    const peaksCheckbox = getByLabelText(/Fitted peaks/i) as HTMLInputElement;
    await fireEvent.click(peaksCheckbox);
    await flush();

    const layout = lastLayout();
    expect((layout.shapes ?? []).length).toBe(0); // shapes gone
    expect((layout.annotations ?? []).length).toBe(1); // annotations preserved
  });

  it('annotation yanchor is "top" (defect 2: prior "middle"/"bottom" overlapped title — AC #3)', async () => {
    render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raman_fitted',
        peaks: [fakePeak()],
        title: 'Amherst Point Sol 921 detail_1 — Point 91',
      },
    });
    await flush();

    const layout = lastLayout();
    const ann = (layout.annotations ?? [])[0] as { yanchor?: string; y?: number };
    expect(ann?.yanchor).toBe('top');
    expect(ann?.y).toBe(1);
  });
});

describe('RamanView — baseline viewport selection (issue #5)', () => {
  it('emits the normalized Plotly x-axis zoom and emits null on autorange reset', async () => {
    let relayoutHandler: ((update: Record<string, unknown>) => void) | undefined;
    const originalOn = (HTMLElement.prototype as unknown as { on?: unknown }).on;
    Object.defineProperty(HTMLElement.prototype, 'on', {
      configurable: true,
      value: (event: string, handler: (update: Record<string, unknown>) => void) => {
        if (event === 'plotly_relayout') relayoutHandler = handler;
      },
    });
    try {
      const { component } = render(RamanView, {
        props: { wavenumber: [700, 800, 900], intensity: [1, 2, 3] },
      });
      const ranges: Array<[number, number] | null> = [];
      component.$on('viewRangeChange', (e) => ranges.push(e.detail.range));
      await flush();

      expect(relayoutHandler).toBeDefined();
      relayoutHandler!({ 'xaxis.range[0]': 875, 'xaxis.range[1]': 725 });
      await tick();
      relayoutHandler!({ 'xaxis.autorange': true });
      await tick();
      expect(ranges).toEqual([[725, 875], null]);
    } finally {
      if (originalOn === undefined) {
        delete (HTMLElement.prototype as unknown as { on?: unknown }).on;
      } else {
        Object.defineProperty(HTMLElement.prototype, 'on', {
          configurable: true,
          value: originalOn,
        });
      }
    }
  });

  it('draws a range-limited baseline overlay only inside its fitted interval', async () => {
    render(RamanView, {
      props: {
        wavenumber: [700, 800, 900, 1000],
        intensity: [10, 20, 30, 40],
        baseline: [0, 18, 27, 0],
        baselineRange: [750, 950],
        stage: 'baseline_corrected',
      },
    });
    await flush();

    const trace = lastTraces().find((t) => t.name === 'baseline');
    expect(trace?.x).toEqual([800, 900]);
  });
});

describe('RamanView — ML/modz spike-marker provenance label (issue #8)', () => {
  const wavenumber = Array.from({ length: 100 }, (_, i) => 640 + i * 10);
  const intensity = wavenumber.map((wn) => Math.exp(-((wn - 1016) ** 2) / 5000));
  // Mark a couple of channels; same boolean mask drives both methods.
  const spikeMask = wavenumber.map((_, i) => i === 10 || i === 42);

  it('labels markers "ML-masked (N)" when spikeMethod is "ml"', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', spikeMask, spikeMethod: 'ml' },
    });
    await flush();
    const trace = spikeTrace();
    expect(trace, 'expected a spike-marker trace in ML mode').toBeDefined();
    expect(trace?.name).toBe('ML-masked (2)');
    // Markers sit at the two flagged served positions.
    expect(trace?.x).toEqual([wavenumber[10], wavenumber[42]]);
  });

  it('labels markers "Spikes (N)" when spikeMethod is "modz" (unchanged)', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'despiked', spikeMask, spikeMethod: 'modz' },
    });
    await flush();
    const trace = spikeTrace();
    expect(trace?.name).toBe('Spikes (2)');
  });

  it('falls back to "Spikes (N)" when spikeMethod is null (legacy callers)', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'despiked', spikeMask },
    });
    await flush();
    expect(spikeTrace()?.name).toBe('Spikes (2)');
  });

  it('toggling the Spikes overlay off hides ML markers', async () => {
    const { getByLabelText } = render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', spikeMask, spikeMethod: 'ml' },
    });
    await flush();
    expect(spikeTrace()).toBeDefined();

    // The single "Spikes" overlay checkbox governs both methods (no new toggle).
    const cb = getByLabelText(/^\s*Spikes\s*$/i) as HTMLInputElement;
    await fireEvent.click(cb);
    await flush();
    expect(spikeTrace()).toBeUndefined();
  });

  it('renders no marker trace when the mask is null (no ML positions)', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', spikeMask: null, spikeMethod: 'ml' },
    });
    await flush();
    expect(spikeTrace()).toBeUndefined();
  });
});

describe('RamanView — known-noisy channel annotation overlay (issue #9)', () => {
  const wavenumber = Array.from({ length: 100 }, (_, i) => 640 + i * 10);
  const intensity = wavenumber.map((wn) => Math.exp(-((wn - 1016) ** 2) / 5000));
  const badpix = [
    { position: 12, tier: 1, source: 'g5_eps' },
    { position: 50, tier: 2, source: 'jb25' },
  ];

  it('default-OFF: no badpix trace until the toggle is enabled', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', badpix },
    });
    await flush();
    // showBadpix defaults to false → overlay not rendered.
    expect(badpixTrace()).toBeUndefined();
    // The dedicated toggle is present (the data is available).
    // (asserted in the toggle test below)
  });

  it('renders hollow blue diamonds at badpix positions when showBadpix is on', async () => {
    render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', badpix, showBadpix: true },
    });
    await flush();
    const trace = badpixTrace();
    expect(trace, 'expected a known-noisy marker trace').toBeDefined();
    expect(trace?.name).toBe('Known-noisy channels (2)');
    // Markers sit at the annotated served positions.
    expect(trace?.x).toEqual([wavenumber[12], wavenumber[50]]);
    // Distinct glyph + color from the filled red triangle spike markers.
    expect(trace?.marker?.symbol).toBe('diamond-open');
    expect(trace?.marker?.color).toBe('#0369a1');
  });

  it('glyph + color are distinct from the spike markers (colorblind-safe pairing)', async () => {
    // Both overlays active: spike markers (filled red triangles) and badpix
    // (hollow blue diamonds) must never read as the same marker.
    const spikeMask = wavenumber.map((_, i) => i === 5);
    render(RamanView, {
      props: {
        wavenumber,
        intensity,
        rawIntensity: intensity,
        stage: 'raw',
        spikeMask,
        spikeMethod: 'ml',
        badpix,
        showBadpix: true,
      },
    });
    await flush();
    const spike = lastTraces().find(
      (t) => (t as { marker?: { symbol?: string } }).marker?.symbol === 'triangle-up',
    ) as { marker?: { symbol?: string; color?: string } } | undefined;
    const bp = badpixTrace();
    expect(spike?.marker?.symbol).toBe('triangle-up');
    expect(bp?.marker?.symbol).toBe('diamond-open');
    expect(spike?.marker?.symbol).not.toBe(bp?.marker?.symbol);
    expect(spike?.marker?.color).not.toBe(bp?.marker?.color);
  });

  it('the "Known-noisy channels" toggle gates the overlay', async () => {
    const { getByLabelText } = render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', badpix },
    });
    await flush();
    expect(badpixTrace()).toBeUndefined();

    const cb = getByLabelText(/Known-noisy channels/i) as HTMLInputElement;
    expect(cb.checked).toBe(false); // default OFF
    await fireEvent.click(cb);
    await flush();
    expect(badpixTrace()).toBeDefined();
  });

  it('renders no toggle and no trace when there are no badpix channels', async () => {
    const { queryByLabelText } = render(RamanView, {
      props: { wavenumber, intensity, rawIntensity: intensity, stage: 'raw', badpix: [], showBadpix: true },
    });
    await flush();
    expect(badpixTrace()).toBeUndefined();
    expect(queryByLabelText(/Known-noisy channels/i)).toBeNull();
  });
});
