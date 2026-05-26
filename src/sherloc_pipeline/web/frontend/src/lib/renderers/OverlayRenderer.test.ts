// Renderer-level coverage for OverlayRenderer.drawScaleBar — the function
// Workbench's AciViewer now lifts from Map mode per issue #14. The unit
// + scale-derivation contract IS the AC (same OverlayRenderer + same
// pixelScale default in both viewers), so regression-locking the
// micrometer → millimeter label switch and the nice-round ladder protects
// both call sites against a future renderer rewrite.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OverlayRenderer, type OverlayOptions } from './OverlayRenderer';

type FillRectCall = [number, number, number, number];
type FillTextCall = [string, number, number];

type AnyMock = ReturnType<typeof vi.fn<any[], any>>;

interface MockCtx {
  fillRect: AnyMock;
  fillText: AnyMock;
  strokeRect: AnyMock;
  measureText: AnyMock;
  save: AnyMock;
  restore: AnyMock;
  beginPath: AnyMock;
  closePath: AnyMock;
  moveTo: AnyMock;
  lineTo: AnyMock;
  arc: AnyMock;
  stroke: AnyMock;
  fill: AnyMock;
  setLineDash: AnyMock;
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
  font: string;
  textAlign: string;
  textBaseline: string;
}

function makeCtx(): MockCtx {
  return {
    fillRect: vi.fn(),
    fillText: vi.fn(),
    strokeRect: vi.fn(),
    measureText: vi.fn((): { width: number } => ({ width: 40 })),
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    setLineDash: vi.fn(),
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    font: '',
    textAlign: '',
    textBaseline: '',
  };
}

function drawWith(
  pixelScale: number,
  transformScale: number,
  canvasW = 800,
  canvasH = 600,
): MockCtx {
  const ctx = makeCtx();
  const renderer = new OverlayRenderer();
  const options: OverlayOptions = {
    scaleBar: {
      pixelScale,
      transform: { x: 0, y: 0, scale: transformScale },
    },
  };
  renderer.draw(
    ctx as unknown as CanvasRenderingContext2D,
    canvasW,
    canvasH,
    options,
  );
  return ctx;
}

function scaleBarLabel(ctx: MockCtx): string {
  // The label is the only fillText call when only scaleBar is set.
  const calls = ctx.fillText.mock.calls as FillTextCall[];
  expect(calls.length).toBeGreaterThan(0);
  return calls[0][0];
}

describe('OverlayRenderer.drawScaleBar — unit selection (issue #14 parity)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders bar + label when scaleBar option provided', () => {
    const ctx = drawWith(10, 1);
    // bar background + bar = 2 fillRect calls; label = 1 fillText call
    const fillRectCalls = ctx.fillRect.mock.calls as FillRectCall[];
    expect(fillRectCalls.length).toBe(2);
    expect(ctx.fillText.mock.calls.length).toBe(1);
  });

  it('renders no overlay output when scaleBar option absent', () => {
    const ctx = makeCtx();
    const renderer = new OverlayRenderer();
    renderer.draw(
      ctx as unknown as CanvasRenderingContext2D,
      800,
      600,
      {},
    );
    expect(ctx.fillRect.mock.calls.length).toBe(0);
    expect(ctx.fillText.mock.calls.length).toBe(0);
  });

  it('labels in µm when nice-rounded length is <1000 µm', () => {
    // 5 µm/px × 100px / scale=1 = 500 µm target → niceRound → 500 µm
    const ctx = drawWith(5, 1);
    expect(scaleBarLabel(ctx)).toBe('500 um');
  });

  it('labels in mm with 1-decimal precision when 1000 ≤ length < 10 000 µm', () => {
    // 20 µm/px × 100px / scale=1 = 2000 µm target → niceRound → 2000 µm
    // 2000 µm is in the mm branch (≥1000) but below the 0-decimal boundary
    // (<10 000) → toFixed(1) → "2.0 mm".
    const ctx = drawWith(20, 1);
    expect(scaleBarLabel(ctx)).toBe('2.0 mm');
  });

  it('labels in mm with 0-decimal precision at ≥10 000 µm (boundary)', () => {
    // 10 µm/px × 100px / scale=0.1 = 10 000 µm target → niceRound → 10 000.
    // The renderer's `niceUm >= 10000 ? 0 : 1`-decimal branch hits the
    // boundary at exactly 10 000 → toFixed(0) → "10 mm".
    const ctx = drawWith(10, 0.1);
    expect(scaleBarLabel(ctx)).toBe('10 mm');
  });

  it('honors zoom: chosen nice-µm shrinks as transform.scale grows', () => {
    // The bar's screen-px width = niceRound(targetUm) / (pixelScale / transform.scale).
    // Doubling transform.scale halves the µm/screen-px ratio, so the chosen
    // nice-µm shrinks; the rendered bar should report a smaller label.
    const labelAt1x = scaleBarLabel(drawWith(10, 1));
    const labelAt4x = scaleBarLabel(drawWith(10, 4));
    expect(labelAt1x).not.toBe(labelAt4x);
    // At 1× zoom: target 1000 µm → niceRound → 1000 µm → "1.0 mm".
    expect(labelAt1x).toBe('1.0 mm');
    // At 4× zoom: 10/4 = 2.5 µm/screen-px × 100 = 250 µm → niceRound → 200 µm.
    expect(labelAt4x).toBe('200 um');
  });
});
