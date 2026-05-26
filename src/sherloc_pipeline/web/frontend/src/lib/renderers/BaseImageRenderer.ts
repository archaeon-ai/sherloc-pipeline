// ============================================================
// Phase 1 renderer: ACI base image on canvas-base.
// Handles world transform and ctx.filter for brightness/contrast.
// ============================================================

import type { Transform } from '../types/map';

export class BaseImageRenderer {
  private image: HTMLImageElement | null = null;
  private imageLoaded = false;

  /**
   * Set the ACI image. The caller is responsible for decoding (e.g., via
   * `fetchAciImage()` in `lib/api.ts`, which attaches Bearer auth and
   * returns a decoded HTMLImageElement). The renderer simply holds the
   * reference for draw operations.
   *
   * Auth note: pre-v4.1.8 this loaded from a URL via `new Image() + img.src`,
   * which does not attach Authorization headers and produced 401s under
   * Auth0 Bearer-token mode. Session 93 design memo §2.3 moved the fetch
   * into the call site (`MapMode.svelte`) using `fetchAciImage()`.
   */
  setImage(img: HTMLImageElement): void {
    this.image = img;
    this.imageLoaded = true;
  }

  /**
   * Draw the ACI base image with world transform and brightness/contrast filters.
   *
   * @param ctx        Target 2D rendering context (canvas-base)
   * @param transform  Pan/zoom transform in CSS pixels
   * @param brightness 0.5 to 2.0, default 1.0 (maps to CSS filter brightness)
   * @param contrast   0.5 to 2.0, default 1.0 (maps to CSS filter contrast)
   */
  draw(
    ctx: CanvasRenderingContext2D,
    transform: Transform,
    brightness: number = 1.0,
    contrast: number = 1.0,
  ): void {
    if (!this.image || !this.imageLoaded) return;

    ctx.save();

    // Apply brightness/contrast via CSS filter on canvas context
    ctx.filter = `brightness(${brightness}) contrast(${contrast})`;

    // Apply world transform
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    ctx.drawImage(this.image, 0, 0);

    ctx.restore();
  }

  /** Natural width of the loaded image, or 0 if not loaded. */
  get naturalWidth(): number {
    return this.image?.naturalWidth ?? 0;
  }

  /** Natural height of the loaded image, or 0 if not loaded. */
  get naturalHeight(): number {
    return this.image?.naturalHeight ?? 0;
  }

  /** Whether an image has been successfully loaded. */
  get loaded(): boolean {
    return this.imageLoaded;
  }

  /** Discard the loaded image and reset state. */
  dispose(): void {
    this.image = null;
    this.imageLoaded = false;
  }
}
