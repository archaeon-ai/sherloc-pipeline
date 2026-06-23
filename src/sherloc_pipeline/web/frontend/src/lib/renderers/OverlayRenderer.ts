// ============================================================
// Phase 3 renderer: UI overlays on canvas-ui (screen space).
// Color scale bars, legends, scale bars, lasso outline.
// No world transform — everything in CSS pixel coordinates.
// ============================================================

import type { ColormapFn } from '../colormaps';
import { colormapToRGBA } from '../colormaps';
import type { Transform } from '../types/map';

export interface OverlayOptions {
  colorScale?: {
    colormap: ColormapFn;
    range: [number, number];
    label: string;
    unit: string;
  };
  legend?: {
    entries: { label: string; color: string }[];
  };
  scaleBar?: {
    pixelScale: number; // um per pixel
    transform: Transform;
  };
  selectionPolygon?: [number, number][];
  extrapolationWarning?: boolean;
}

const FONT_FAMILY = 'system-ui, -apple-system, sans-serif';
const FONT_MONO = 'ui-monospace, monospace';

export class OverlayRenderer {
  /**
   * Draw all UI overlays on the canvas-ui layer (screen space coordinates).
   */
  draw(
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    canvasHeight: number,
    options: OverlayOptions,
  ): void {
    if (options.colorScale) {
      this.drawColorScale(ctx, canvasWidth, canvasHeight, options.colorScale);
    }
    if (options.legend) {
      this.drawLegend(ctx, canvasWidth, options.legend);
    }
    if (options.scaleBar) {
      this.drawScaleBar(ctx, canvasWidth, canvasHeight, options.scaleBar);
    }
    if (options.selectionPolygon && options.selectionPolygon.length > 1) {
      this.drawLassoOutline(ctx, options.selectionPolygon);
    }
    if (options.extrapolationWarning) {
      this.drawWarningBadge(ctx, canvasWidth);
    }
  }

  /**
   * Vertical color scale bar in the top-right corner.
   */
  private drawColorScale(
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    _canvasHeight: number,
    config: NonNullable<OverlayOptions['colorScale']>,
  ): void {
    const barWidth = 16;
    const barHeight = 160;
    const margin = 20;
    const labelPad = 6;

    const x = canvasWidth - margin - barWidth - 50; // space for labels
    const y = margin; // top-aligned — avoids bottom-right HTML button overlay

    // Label + unit above bar
    ctx.font = `12px ${FONT_FAMILY}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const centerX = x + barWidth / 2;
    const labelStr = config.unit
      ? `${config.label} (${config.unit})`
      : config.label;
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(labelStr, centerX, y);

    const barY = y + 6; // small gap below the title text

    // Draw gradient bar
    for (let i = 0; i < barHeight; i++) {
      const t = 1 - i / (barHeight - 1); // top = max, bottom = min
      const rgb = config.colormap(t);
      ctx.fillStyle = colormapToRGBA(rgb, 1);
      ctx.fillRect(x, barY + i, barWidth, 1);
    }

    // Border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, barY, barWidth, barHeight);

    // Tick labels
    ctx.fillStyle = '#e2e8f0';
    ctx.font = `11px ${FONT_MONO}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const [min, max] = config.range;
    const labelX = x + barWidth + labelPad;

    // Max at top
    ctx.fillText(this.formatValue(max), labelX, barY);
    // Mid
    ctx.fillText(this.formatValue((min + max) / 2), labelX, barY + barHeight / 2);
    // Min at bottom
    ctx.fillText(this.formatValue(min), labelX, barY + barHeight);
  }

  /**
   * Legend panel in the top-right corner.
   */
  private drawLegend(
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
    config: NonNullable<OverlayOptions['legend']>,
  ): void {
    if (config.entries.length === 0) return;

    const margin = 16;
    const pad = 8;
    const lineHeight = 18;
    const swatchSize = 10;
    const gap = 6;

    ctx.font = `12px ${FONT_FAMILY}`;

    // Measure widths
    let maxTextWidth = 0;
    for (const entry of config.entries) {
      const w = ctx.measureText(entry.label).width;
      if (w > maxTextWidth) maxTextWidth = w;
    }

    const boxWidth = pad * 2 + swatchSize + gap + maxTextWidth;
    const boxHeight = pad * 2 + config.entries.length * lineHeight;
    const x = canvasWidth - margin - boxWidth;
    const y = margin;

    // Background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.beginPath();
    ctx.roundRect(x, y, boxWidth, boxHeight, 4);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, boxWidth, boxHeight, 4);
    ctx.stroke();

    // Entries
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < config.entries.length; i++) {
      const entry = config.entries[i];
      const ey = y + pad + i * lineHeight + lineHeight / 2;

      // Swatch
      ctx.fillStyle = entry.color;
      ctx.fillRect(x + pad, ey - swatchSize / 2, swatchSize, swatchSize);

      // Label
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(entry.label, x + pad + swatchSize + gap, ey);
    }
  }

  /**
   * Physical scale bar in the bottom-left corner.
   * Shows a bar corresponding to a round number of micrometers.
   */
  private drawScaleBar(
    ctx: CanvasRenderingContext2D,
    _canvasWidth: number,
    canvasHeight: number,
    config: NonNullable<OverlayOptions['scaleBar']>,
  ): void {
    const margin = 16;
    const barHeight = 4;

    // Target bar length in screen pixels: ~80-120px
    const targetScreenPx = 100;
    // How many um does that represent?
    const umPerScreenPx = config.pixelScale / config.transform.scale;
    const targetUm = targetScreenPx * umPerScreenPx;

    // Round to a nice number
    const niceUm = this.niceRound(targetUm);
    const barPx = niceUm / umPerScreenPx;

    const x = margin;
    const y = canvasHeight - margin - barHeight;

    // Measure label text width for correctly-sized background
    ctx.font = `11px ${FONT_MONO}`;
    const label =
      niceUm >= 1000 ? `${(niceUm / 1000).toFixed(niceUm >= 10000 ? 0 : 1)} mm` : `${niceUm} um`;
    const textWidth = ctx.measureText(label).width;

    // Bar background for contrast — padded to fit both bar and label
    const bgPad = 8;
    const bgWidth = Math.max(barPx, textWidth) + bgPad * 2;
    const bgHeight = barHeight + 14 + bgPad * 2; // text line(14) + spacing
    const bgX = x - bgPad;
    const bgY = y - 14 - 4 - bgPad; // text height + gap above bar + padding
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

    // Bar
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(x, y, barPx, barHeight);

    // Label
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(label, x + barPx / 2, y - 6);
  }

  /**
   * Lasso selection polygon outline.
   */
  private drawLassoOutline(
    ctx: CanvasRenderingContext2D,
    polygon: [number, number][],
  ): void {
    if (polygon.length < 2) return;

    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(255, 200, 50, 0.8)';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1]);
    for (let i = 1; i < polygon.length; i++) {
      ctx.lineTo(polygon[i][0], polygon[i][1]);
    }
    ctx.closePath();
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = 'rgba(255, 200, 50, 0.08)';
    ctx.fill();

    ctx.restore();
  }

  /**
   * Warning badge when spatial extrapolation is active.
   */
  private drawWarningBadge(
    ctx: CanvasRenderingContext2D,
    canvasWidth: number,
  ): void {
    const text = 'Extrapolation Warning';
    ctx.font = `bold 12px ${FONT_FAMILY}`;
    const w = ctx.measureText(text).width;
    const pad = 8;
    const x = canvasWidth / 2 - w / 2 - pad;
    const y = 8;

    ctx.fillStyle = 'rgba(180, 80, 20, 0.9)';
    ctx.beginPath();
    ctx.roundRect(x, y, w + pad * 2, 28, 4);
    ctx.fill();

    ctx.fillStyle = '#fcd34d';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvasWidth / 2, y + 14);
  }

  /** Format a numeric value for scale labels. */
  private formatValue(v: number): string {
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    if (Math.abs(v) >= 1) return v.toFixed(1);
    if (Math.abs(v) >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  }

  /** Round to a "nice" number for scale bars: 1, 2, 5, 10, 20, 50, ... */
  private niceRound(v: number): number {
    const exponent = Math.floor(Math.log10(v));
    const base = Math.pow(10, exponent);
    const frac = v / base;
    if (frac < 1.5) return base;
    if (frac < 3.5) return 2 * base;
    if (frac < 7.5) return 5 * base;
    return 10 * base;
  }
}
