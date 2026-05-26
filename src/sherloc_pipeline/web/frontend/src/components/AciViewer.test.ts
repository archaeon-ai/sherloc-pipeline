// Integration coverage for AciViewer's overlay-rendering call site —
// Codex /code-review PR #27 R2 F2 Partially Resolved (Major) follow-up.
// The helper-boundary tests in `lib/aciOverlay.test.ts` lock the option
// shape; this file locks that AciViewer actually invokes
// OverlayRenderer.draw(...) after the ACI image loads. A regression
// that drops the draw call (or the buildAciOverlayOptions wiring) trips
// this test, even if the helper-level tests still pass.
//
// jsdom does not implement HTMLCanvasElement.getContext('2d') or
// HTMLImageElement.decode(); we stub the minimum surface AciViewer
// touches.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/svelte';
import { tick } from 'svelte';
import AciViewer from './AciViewer.svelte';
import * as api from '../lib/api';
import * as renderers from '../lib/renderers/OverlayRenderer';

const SCAN_ID = 'test-scan-0001';

// Use a permissive Mock type — the spy's specific signature trips
// svelte-check (Mock with concrete params not assignable to
// Mock<unknown[], unknown>). The test only consumes drawSpy.mock.calls,
// not the inferred function type, so the looser annotation is harmless.
let drawSpy: ReturnType<typeof vi.fn<unknown[], unknown>>;
let originalGetContext: typeof HTMLCanvasElement.prototype.getContext;

beforeEach(() => {
  vi.restoreAllMocks();

  // Note: AciViewer uses scheduleRender's `animFrameId !== null` guard
  // to coalesce multiple scheduled renders into one rAF tick. A
  // synchronous rAF stub traps animFrameId at the returned handle
  // value, blocking ALL subsequent scheduleRender calls — including
  // the post-loadImage render that exercises the `if (aciImage)`
  // branch. So we leave jsdom's real rAF in place (which fires on the
  // next macrotask) and drain enough setTimeout(0) cycles in flush()
  // to let each rAF callback run.

  // Stub canvas 2D context. AciViewer's render() returns early if
  // getContext returns null, which is jsdom's default. We give it a
  // permissive mock so the render path proceeds far enough to reach the
  // overlayRenderer.draw call.
  originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 0 })),
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
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;

  // Spy on the renderer's draw — captures whatever options the
  // production code path passes in. Spying on the prototype catches all
  // instances regardless of when AciViewer instantiates one.
  drawSpy = vi.spyOn(
    renderers.OverlayRenderer.prototype,
    'draw',
  ) as unknown as typeof drawSpy;
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = originalGetContext;
});

/** Drain queued microtasks + several requestAnimationFrame ticks. */
async function flush(): Promise<void> {
  // The component's flow: onMount -> loadImage (async) -> aciImage set ->
  // afterUpdate -> scheduleRender -> requestAnimationFrame -> render.
  // jsdom's rAF fires on the next macrotask (~16ms in current jsdom);
  // 20 cycles of a small setTimeout drains the rAF queue and any
  // chained promises a few times over.
  for (let i = 0; i < 20; i++) {
    await tick(); // Svelte reactivity flush
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 20));
  }
}

describe('AciViewer — overlay-renderer integration (issue #14 / PR #27 R2 F2)', () => {
  it('invokes OverlayRenderer.draw with scaleBar options once an ACI image loads', async () => {
    // Stub the network: return a real HTMLImageElement so the component's
    // truthy guard fires. We don't need it to actually decode — the
    // stub canvas's drawImage is a no-op.
    const img = new Image();
    vi.spyOn(api, 'fetchAciImage').mockResolvedValue(img);

    render(AciViewer, { props: { scanId: SCAN_ID, points: [] } });

    await flush();

    // Among all draw() calls, at least ONE must carry scaleBar options
    // — the contract this test gates on. (Earlier render passes before
    //  the image promise resolves may fire with empty options; that's
    //  fine.)
    const callsWithScaleBar = drawSpy.mock.calls.filter((args) => {
      const options = args[3] as { scaleBar?: unknown };
      return options && options.scaleBar !== undefined;
    });
    expect(callsWithScaleBar.length).toBeGreaterThan(0);

    // The most-recent scaleBar call must include the prop pixelScale
    // default + a transform object (caller's current viewport state).
    const last = callsWithScaleBar[callsWithScaleBar.length - 1];
    const opts = last[3] as { scaleBar: { pixelScale: number; transform: unknown } };
    expect(opts.scaleBar.pixelScale).toBe(10);
    expect(opts.scaleBar.transform).toBeDefined();
  });

  it('disables the Colorized button when colorizedAvailable=false', async () => {
    // Backend reports no sol_NNNN_colorized/ sibling for this scan
    // (the case 170 of 205 historical sols hit). Button must stay
    // visible (so users can see the feature exists) but be disabled
    // with an explanatory tooltip, rather than the prior silent-
    // fallback UX where clicking re-served grayscale.
    const img = new Image();
    vi.spyOn(api, 'fetchAciImage').mockResolvedValue(img);

    const { container } = render(AciViewer, {
      props: { scanId: SCAN_ID, points: [], colorizedAvailable: false },
    });

    await flush();

    const buttons = Array.from(container.querySelectorAll('button')) as HTMLButtonElement[];
    const colorizedBtn = buttons.find((b) => b.textContent?.trim() === 'Colorized');
    expect(colorizedBtn).toBeDefined();
    expect(colorizedBtn!.disabled).toBe(true);
    expect(colorizedBtn!.title).toMatch(/no colorized/i);
  });

  it('enables the Colorized button when colorizedAvailable=true', async () => {
    const img = new Image();
    vi.spyOn(api, 'fetchAciImage').mockResolvedValue(img);

    const { container } = render(AciViewer, {
      props: { scanId: SCAN_ID, points: [], colorizedAvailable: true },
    });

    await flush();

    const buttons = Array.from(container.querySelectorAll('button')) as HTMLButtonElement[];
    const colorizedBtn = buttons.find((b) => b.textContent?.trim() === 'Colorized');
    expect(colorizedBtn).toBeDefined();
    expect(colorizedBtn!.disabled).toBe(false);
  });

  it('does NOT emit a scaleBar option when ACI image fetch fails (no image visible)', async () => {
    // ApiError simulates the imageError path. fetchPointSpectrum and
    // similar helpers re-throw; AciViewer sets imageError = true and
    // leaves aciImage = null.
    vi.spyOn(api, 'fetchAciImage').mockRejectedValue(new api.ApiError(404, 'not found'));

    render(AciViewer, { props: { scanId: SCAN_ID, points: [] } });

    await flush();

    const callsWithScaleBar = drawSpy.mock.calls.filter((args) => {
      const options = args[3] as { scaleBar?: unknown };
      return options && options.scaleBar !== undefined;
    });
    // Render may still fire (e.g., from afterUpdate or initial mount) but
    // every call must carry empty options because aciImage stayed null.
    expect(callsWithScaleBar.length).toBe(0);
  });
});
