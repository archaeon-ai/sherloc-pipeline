// Helper-boundary coverage for SpectrumPlot's peak-overlay path. Issue #18 AC:
// "Two distinct toggles: 'Show fitted peaks' (curves) + 'Show peak labels' (text).
//  Each toggle is independent."
// The two switches inside buildPeakElements ARE the independence contract —
// if a future refactor collapses them back into a single gate, the unit test
// fires before the polish regression reaches a Workbench user.

import { describe, it, expect } from 'vitest';
import { buildPeakElements } from './spectrumLabels';
import type { Peak } from './types';

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

describe('buildPeakElements — issue #18 independent toggles', () => {
  it('returns no shapes and no annotations for empty peaks', () => {
    const result = buildPeakElements([], { showFit: true, showPeakLabels: true });
    expect(result.shapes).toEqual([]);
    expect(result.annotations).toEqual([]);
  });

  it('skips peaks whose center_cm1 is null', () => {
    const result = buildPeakElements([fakePeak({ center_cm1: null })], {
      showFit: true,
      showPeakLabels: true,
    });
    expect(result.shapes).toEqual([]);
    expect(result.annotations).toEqual([]);
  });

  it('with showFit=true, showPeakLabels=true → both shape and annotation', () => {
    const result = buildPeakElements([fakePeak()], { showFit: true, showPeakLabels: true });
    expect(result.shapes).toHaveLength(1);
    expect(result.annotations).toHaveLength(1);
    expect(result.shapes[0].x0).toBe(1016);
    expect(result.annotations[0].text).toBe('sulf1');
  });

  it('with showFit=true, showPeakLabels=false → shape only (peak markers, no text)', () => {
    const result = buildPeakElements([fakePeak()], { showFit: true, showPeakLabels: false });
    expect(result.shapes).toHaveLength(1);
    expect(result.annotations).toHaveLength(0);
  });

  it('with showFit=false, showPeakLabels=true → annotation only (text without markers)', () => {
    // This is the genuinely-independent third state the issue's AC unlocks
    // vs. the prior `peaks={showFit ? displayPeaks : []}` coupling — labels
    // can mark peak positions in data space without the visual clutter of
    // the vertical-line shapes.
    const result = buildPeakElements([fakePeak()], { showFit: false, showPeakLabels: true });
    expect(result.shapes).toHaveLength(0);
    expect(result.annotations).toHaveLength(1);
    expect(result.annotations[0].text).toBe('sulf1');
  });

  it('with showFit=false, showPeakLabels=false → nothing renders', () => {
    const result = buildPeakElements([fakePeak()], { showFit: false, showPeakLabels: false });
    expect(result.shapes).toEqual([]);
    expect(result.annotations).toEqual([]);
  });

  it('falls back to numeric center label when mineral_assignment is empty', () => {
    const result = buildPeakElements([fakePeak({ mineral_assignment: null, center_cm1: 1432.7 })], {
      showFit: false,
      showPeakLabels: true,
    });
    expect(result.annotations[0].text).toBe('1433');
  });

  it('annotation yanchor is "top" so rotated text drops into the plot area (not the title)', () => {
    // Issue #18 defect 2: prior RamanView used yanchor: 'middle' and
    // SpectrumPlot used yanchor: 'bottom'; both anchored the text on or
    // above y=1 paper-coord so the rotated -45° label intruded into the
    // Plotly title region (e.g., "Amherst Point Sol 921 ... — Point 91").
    // Flipping to 'top' clamps text inside the plot region. Asserted here
    // so a future "give labels more breathing room" change can't silently
    // re-introduce the overlap on either consumer.
    const result = buildPeakElements([fakePeak()], { showFit: true, showPeakLabels: true });
    expect(result.annotations[0].yanchor).toBe('top');
    expect(result.annotations[0].y).toBe(1);
    expect(result.annotations[0].yref).toBe('paper');
  });

  // --- RamanView consumer (Workbench route) variant ---

  it('shapeY1 option clips shapes for RamanView (0.95 paper-coord, not 1)', () => {
    // RamanView leaves a 5% sliver at the top of the plot for overlay
    // legends; SpectrumPlot uses the full 0..1. The helper must preserve
    // the consumer-specific clip — a default of 1 would push the vertical
    // line into the legend on the Workbench panel.
    const result = buildPeakElements([fakePeak()], {
      showFit: true,
      showPeakLabels: false,
      shapeY1: 0.95,
    });
    expect(result.shapes).toHaveLength(1);
    expect(result.shapes[0].y0).toBe(0);
    expect(result.shapes[0].y1).toBe(0.95);
  });

  it('labelFormat="paren" renders RamanView-style "assignment (center)"', () => {
    // The issue #18 reproduction screenshot shows labels in the form
    // "sulf1 ν₁~1016 cm⁻¹" — which corresponds to RamanView's
    // `${mineral_assignment} (${center_cm1.toFixed(0)})` template, NOT
    // SpectrumPlot's bare assignment. The paren variant must reach the
    // Workbench panel intact through the helper.
    const result = buildPeakElements([fakePeak({ mineral_assignment: 'sulf1', center_cm1: 1016 })], {
      showFit: false,
      showPeakLabels: true,
      labelFormat: 'paren',
    });
    expect(result.annotations[0].text).toBe('sulf1 (1016)');
  });

  it('labelFormat="paren" still falls back to numeric center when no assignment', () => {
    const result = buildPeakElements([fakePeak({ mineral_assignment: null, center_cm1: 1016 })], {
      showFit: false,
      showPeakLabels: true,
      labelFormat: 'paren',
    });
    expect(result.annotations[0].text).toBe('1016');
  });

  it('labelFormat defaults to "plain" for backwards compat with SpectrumPlot', () => {
    const result = buildPeakElements([fakePeak({ mineral_assignment: 'sulf1', center_cm1: 1016 })], {
      showFit: false,
      showPeakLabels: true,
    });
    expect(result.annotations[0].text).toBe('sulf1');
  });
});
