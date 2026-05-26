// Helper-boundary coverage for AciViewer's option-building path. Codex
// /code-review PR #27 R1 F2: pure-renderer tests can't catch a regression
// where AciViewer drops the scale-bar wiring (the `if (aciImage)` gate
// or the OverlayRenderer.draw call). Asserting that buildAciOverlayOptions
// produces the right shape is the cheapest gate that protects the
// Workbench scale-bar contract from a future refactor.

import { describe, it, expect } from 'vitest';
import { buildAciOverlayOptions } from './aciOverlay';
import type { Transform } from './types/map';

const identityTransform: Transform = { x: 0, y: 0, scale: 1 };

// HTMLImageElement is a DOM type; vitest+jsdom provides it, but the
// concrete element matters less than the truthy/null branch.
function fakeImage(): HTMLImageElement {
  return new Image();
}

describe('buildAciOverlayOptions — Workbench scale-bar wiring (issue #14)', () => {
  it('returns an empty options object when aciImage is null', () => {
    const opts = buildAciOverlayOptions({
      aciImage: null,
      pixelScale: 10,
      transform: identityTransform,
    });
    expect(opts).toEqual({});
    expect(opts.scaleBar).toBeUndefined();
  });

  it('returns scaleBar with the current pixelScale + transform when an image is present', () => {
    const transform: Transform = { x: 100, y: 50, scale: 2.5 };
    const opts = buildAciOverlayOptions({
      aciImage: fakeImage(),
      pixelScale: 10,
      transform,
    });
    expect(opts.scaleBar).toEqual({ pixelScale: 10, transform });
  });

  it('forwards a non-default pixelScale verbatim (issue #28 plumbing target)', () => {
    // When per-scan pixel_scale_um plumbing lands (issue #28), the
    // helper must propagate whatever value the caller passed — not
    // silently fall back to the MapCanvas default of 10. Test the
    // contract pre-emptively so the issue-#28 PR doesn't have to
    // re-introduce this assertion.
    const opts = buildAciOverlayOptions({
      aciImage: fakeImage(),
      pixelScale: 7.3,
      transform: identityTransform,
    });
    expect(opts.scaleBar?.pixelScale).toBe(7.3);
  });

  it('emits no overlay output when aciImage is null, even with custom pixelScale', () => {
    const opts = buildAciOverlayOptions({
      aciImage: null,
      pixelScale: 5,
      transform: { x: 10, y: 10, scale: 4 },
    });
    expect(opts).toEqual({});
  });
});
