// Pure-logic helper for the Workbench ACI viewer's screen-space overlay.
// Extracted from AciViewer.svelte's render path so the option-building
// contract (gate on aciImage; ship scaleBar with the current pixelScale
// + transform when an image is loaded) is unit-testable without mounting
// the Svelte component or stubbing a canvas/RAF/devicePixelRatio stack.
// The Workbench scale-bar contract from issue #14 (close PR #27) lives
// here so a future refactor of AciViewer can't accidentally drop the
// overlay without tripping a test.

import type { OverlayOptions } from './renderers/OverlayRenderer';
import type { Transform } from './types/map';

export interface AciOverlayInputs {
  aciImage: HTMLImageElement | null;
  pixelScale: number;
  transform: Transform;
}

/**
 * Build the OverlayOptions blob the AciViewer renderer passes to
 * `OverlayRenderer.draw()` after restoring to screen space.
 *
 * Contract:
 *  - aciImage absent (null): return `{}` — the overlay layer is a no-op
 *    until an image is loaded. The "Log in to view" / "No ACI" placeholders
 *    own the unloaded states; the scale bar has no meaningful units
 *    without a visible raster.
 *  - aciImage present: return `{ scaleBar: { pixelScale, transform } }` —
 *    identical to MapCanvas's options.scaleBar shape, so OverlayRenderer
 *    treats both viewers the same.
 *
 * Issue #28 (follow-up): per-scan pixel_scale_um plumbing for BOTH viewers.
 * Until that lands, pixelScale arrives as MapCanvas's default `10` (true
 * Map↔Workbench parity).
 */
export function buildAciOverlayOptions(
  inputs: AciOverlayInputs,
): OverlayOptions {
  const { aciImage, pixelScale, transform } = inputs;
  if (!aciImage) {
    return {};
  }
  return {
    scaleBar: { pixelScale, transform },
  };
}
