// Component-level coverage for the Workbench-routed spectrum panel —
// Codex /code-review PR #30 R1 F2 (Major) follow-up. Helper-boundary
// tests in `lib/spectrumLabels.test.ts` lock the shape/annotation
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

describe('RamanView — issue #18 polish on the Workbench-routed component (PR #30 R1 F1/F2)', () => {
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
