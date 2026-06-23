import type { Peak } from './types';

export interface PeakElement {
  type: 'line';
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  yref: 'paper';
  line: { color: string; width: number; dash: string };
}

export interface PeakAnnotation {
  x: number;
  y: number;
  yref: 'paper';
  text: string;
  showarrow: false;
  font: { size: number; color: string };
  yanchor: 'top';
  textangle: number;
}

export type PeakLabelFormat = 'plain' | 'paren';

export interface BuildPeakElementsOptions {
  // Whether to emit the per-peak vertical-line shapes. Mapped to:
  // - SpectrumPlot's `showFit`
  // - RamanView's `showOverlayPeaks`
  showFit: boolean;
  // Whether to emit the text annotations. New in #18 — gates labels
  // independently of the line markers.
  showPeakLabels: boolean;
  // Vertical extent (paper coords) of the line markers. SpectrumPlot
  // uses the full 0..1; RamanView clips at 0..0.95 to leave a sliver
  // for residual / overlay legends.
  shapeY1?: number;
  // Label format. SpectrumPlot uses bare `mineral_assignment`
  // falling back to the numeric center; RamanView prefers
  // `mineral_assignment (center)` (matches the issue #18 reproduction
  // screenshot's "sulf1 ν₁~1016 cm⁻¹" style).
  labelFormat?: PeakLabelFormat;
}

export interface BuildPeakElementsResult {
  shapes: PeakElement[];
  annotations: PeakAnnotation[];
}

function formatLabel(peak: Peak, format: PeakLabelFormat): string {
  const numeric = `${(peak.center_cm1 as number).toFixed(0)}`;
  if (!peak.mineral_assignment) return numeric;
  return format === 'paren' ? `${peak.mineral_assignment} (${numeric})` : peak.mineral_assignment;
}

export function buildPeakElements(
  peaks: Peak[],
  options: BuildPeakElementsOptions,
): BuildPeakElementsResult {
  const shapes: PeakElement[] = [];
  const annotations: PeakAnnotation[] = [];
  const shapeY1 = options.shapeY1 ?? 1;
  const labelFormat: PeakLabelFormat = options.labelFormat ?? 'plain';

  for (const peak of peaks) {
    if (peak.center_cm1 === null) continue;

    if (options.showFit) {
      shapes.push({
        type: 'line',
        x0: peak.center_cm1,
        x1: peak.center_cm1,
        y0: 0,
        y1: shapeY1,
        yref: 'paper',
        line: { color: '#2563eb', width: 1, dash: 'dot' },
      });
    }

    if (options.showPeakLabels) {
      annotations.push({
        x: peak.center_cm1,
        y: 1,
        yref: 'paper',
        text: formatLabel(peak, labelFormat),
        showarrow: false,
        font: { size: 10, color: '#2563eb' },
        yanchor: 'top',
        textangle: -45,
      });
    }
  }

  return { shapes, annotations };
}
