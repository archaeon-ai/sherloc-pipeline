// ============================================================
// Phase 2 renderer: data overlays on OffscreenCanvas, blitted to canvas-data.
// Supports Voronoi polygon fill, ring markers, and incremental updates.
// ============================================================

import type { ColormapFn } from '../colormaps';
import { getColormap, normalizeValue, colormapToRGBA } from '../colormaps';
import type {
  DisplayPoint,
  ScalarLayer,
  VoronoiGeometry,
  GeometryMode,
  Transform,
} from '../types/map';

export class DataLayerRenderer {
  private offscreen: OffscreenCanvas | null = null;
  private offCtx: OffscreenCanvasRenderingContext2D | null = null;
  private renderedPoints: Set<number> = new Set();
  private needsFullRebuild = true;
  private width = 0;
  private height = 0;

  // Bounding box around scan points (image-space), with padding.
  // Used to clip Voronoi cells so infinite/unbounded edges don't shoot off-screen.
  private clipBounds: { x: number; y: number; w: number; h: number } | null = null;

  /**
   * Initialize the offscreen canvas at the given dimensions (image-space pixels).
   * Call this when the ACI image dimensions are known.
   */
  init(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.offscreen = new OffscreenCanvas(width, height);
    this.offCtx = this.offscreen.getContext('2d');
    this.renderedPoints.clear();
    this.needsFullRebuild = true;
  }

  /**
   * Render a single point incrementally (during progressive WebSocket fitting).
   * Draws only one point's geometry onto the offscreen canvas without clearing.
   */
  renderPoint(
    pointIndex: number,
    point: DisplayPoint,
    value: number | null,
    status: string,
    voronoiRegion: number[] | null,
    voronoiVertices: [number, number][] | null,
    isEdge: boolean,
    colormap: ColormapFn,
    range: [number, number],
    geometryMode: GeometryMode,
    opacity: number,
    pixelScale: number,
  ): void {
    if (!this.offCtx) return;

    const ctx = this.offCtx;
    const hasValue = value !== null && status === 'measured';

    // Compute color
    let fillColor: string;
    if (hasValue) {
      const t = normalizeValue(value, range[0], range[1]);
      const rgb = colormap(t);
      fillColor = colormapToRGBA(rgb, opacity);
    } else {
      // Non-detection: dim grey
      fillColor = `rgba(128, 128, 128, ${opacity * 0.15})`;
    }

    if (geometryMode === 'voronoi' || geometryMode === 'combined') {
      this.drawVoronoiCell(ctx, point, voronoiRegion, voronoiVertices, isEdge, fillColor, hasValue, opacity);
    }

    if (geometryMode === 'ring' || geometryMode === 'combined') {
      this.drawRing(ctx, point, hasValue, fillColor, opacity, pixelScale);
    }

    this.renderedPoints.add(pointIndex);
  }

  /**
   * Full render of all layers (on mode change, colormap change, etc.).
   * Clears the offscreen canvas and redraws everything.
   */
  renderFull(
    layers: ScalarLayer[],
    points: DisplayPoint[],
    voronoi: VoronoiGeometry | null,
    geometryMode: GeometryMode,
    compositeMode: 'single' | 'rgb' | 'alpha_stack',
  ): void {
    if (!this.offCtx || !this.offscreen) return;

    const ctx = this.offCtx;
    ctx.clearRect(0, 0, this.width, this.height);
    this.renderedPoints.clear();

    // Layers are already filtered by getVisibleLayers — use as-is
    if (layers.length === 0) return;
    const visibleLayers = layers;

    // Compute clip bounds from scan points to contain Voronoi cells
    this.computeClipBounds(points);

    // Apply clip region for Voronoi/combined modes to prevent unbounded cells
    const needsClip =
      (geometryMode === 'voronoi' || geometryMode === 'combined') &&
      this.clipBounds !== null;

    if (needsClip) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(this.clipBounds!.x, this.clipBounds!.y, this.clipBounds!.w, this.clipBounds!.h);
      ctx.clip();
    }

    if (compositeMode === 'rgb') {
      this.renderRGB(ctx, visibleLayers, points, voronoi, geometryMode);
    } else if (compositeMode === 'alpha_stack') {
      this.renderAlphaStack(ctx, visibleLayers, points, voronoi, geometryMode);
    } else {
      // Single layer (or first visible)
      const layer = visibleLayers[0];
      this.renderSingleLayer(ctx, layer, points, voronoi, geometryMode);
    }

    if (needsClip) {
      ctx.restore();
    }

    this.needsFullRebuild = false;
  }

  /**
   * Blit the offscreen canvas to the visible canvas-data with world transform.
   */
  blitTo(ctx: CanvasRenderingContext2D, transform: Transform): void {
    if (!this.offscreen) return;

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);
    ctx.drawImage(this.offscreen, 0, 0);
    ctx.restore();
  }

  /**
   * Draw rings directly onto a visible canvas context in world-space coordinates.
   * Bypasses the OffscreenCanvas so rings render as vector graphics at screen
   * resolution, staying crisp at any zoom level.
   *
   * When rgbMode is true and multiple layers are provided, each layer maps to an
   * RGB channel (layer 0=red, 1=green, 2=blue) and the per-point colours are
   * additively composited into a single circle per point.
   */
  drawRingsDirect(
    ctx: CanvasRenderingContext2D,
    layers: ScalarLayer[],
    points: DisplayPoint[],
    transform: Transform,
    rgbMode: boolean = false,
  ): void {
    if (layers.length === 0 || points.length === 0) return;

    if (rgbMode && layers.length >= 1) {
      this.drawRingsRGB(ctx, layers, points, transform);
      return;
    }

    const layer = layers[0];
    const colormap = getColormap(layer.colormap.name);
    const [rangeMin, rangeMax] = layer.colormap.range;
    const pixelScale = 10;
    const radiusPx = 50 / pixelScale; // 100 um diameter laser spot

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    // Line width that stays ~2 screen-px regardless of zoom
    const strokeWidth = Math.max(0.5, 2 / transform.scale);

    for (let i = 0; i < points.length && i < layer.values.length; i++) {
      const pt = points[i];
      const lv = layer.values[i];

      // Skip missing/masked — no data, no rendering
      if (lv.status === 'missing' || lv.status === 'masked') continue;

      const hasValue = lv.value !== null && lv.status === 'measured';

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusPx, 0, Math.PI * 2);

      if (hasValue) {
        const t = normalizeValue(lv.value!, rangeMin, rangeMax);
        const rgb = colormap(t);
        ctx.fillStyle = colormapToRGBA(rgb, layer.opacity);
        ctx.fill();
        ctx.strokeStyle = `rgba(255, 255, 255, ${layer.opacity * 0.6})`;
        ctx.lineWidth = strokeWidth;
        ctx.stroke();
      } else {
        // below_threshold: non-detection outline
        ctx.strokeStyle = `rgba(255, 255, 255, ${layer.opacity * 0.3})`;
        ctx.lineWidth = Math.max(0.5, 1 / transform.scale);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  /**
   * Draw rings with additive RGB compositing.
   * Each layer maps to a colour channel: layer 0 → red, 1 → green, 2 → blue.
   * For each point a single composite circle is drawn with the blended colour.
   */
  private drawRingsRGB(
    ctx: CanvasRenderingContext2D,
    layers: ScalarLayer[],
    points: DisplayPoint[],
    transform: Transform,
  ): void {
    const pixelScale = 10;
    const radiusPx = 50 / pixelScale;

    // Pre-compute normalisation ranges per layer
    const layerInfo = layers.map((layer) => ({
      values: layer.values,
      rangeMin: layer.colormap.range[0],
      rangeMax: layer.colormap.range[1],
      opacity: layer.opacity,
    }));

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    const strokeWidth = Math.max(0.5, 2 / transform.scale);
    // Average opacity across provided layers for stroke/non-detection rendering
    const avgOpacity =
      layerInfo.reduce((sum, l) => sum + l.opacity, 0) / layerInfo.length;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      let r = 0;
      let g = 0;
      let b = 0;
      let hasAnyMeasured = false;
      let hasAnyBelowThreshold = false;

      for (let ch = 0; ch < layerInfo.length && ch < 3; ch++) {
        const info = layerInfo[ch];
        if (i >= info.values.length) continue;
        const lv = info.values[i];
        if (lv.status === 'below_threshold') hasAnyBelowThreshold = true;
        if (lv.value === null || lv.status !== 'measured') continue;

        hasAnyMeasured = true;
        const t = normalizeValue(lv.value, info.rangeMin, info.rangeMax);
        const intensity = Math.round(t * 255);
        if (ch === 0) r = Math.min(255, r + intensity);
        else if (ch === 1) g = Math.min(255, g + intensity);
        else b = Math.min(255, b + intensity);
      }

      // Skip if all channels are missing/masked
      if (!hasAnyMeasured && !hasAnyBelowThreshold) continue;

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusPx, 0, Math.PI * 2);

      if (hasAnyMeasured) {
        ctx.fillStyle = `rgba(${r},${g},${b},${avgOpacity})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(255, 255, 255, ${avgOpacity * 0.6})`;
        ctx.lineWidth = strokeWidth;
        ctx.stroke();
      } else {
        // All channels below_threshold: non-detection outline
        ctx.strokeStyle = `rgba(255, 255, 255, ${avgOpacity * 0.3})`;
        ctx.lineWidth = Math.max(0.5, 1 / transform.scale);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  /** Force a full rebuild on the next render. */
  invalidate(): void {
    this.needsFullRebuild = true;
    this.renderedPoints.clear();
  }

  /** Clear the offscreen canvas. */
  clear(): void {
    if (this.offCtx) {
      this.offCtx.clearRect(0, 0, this.width, this.height);
    }
    this.renderedPoints.clear();
  }

  /** Whether a full rebuild is needed. */
  get needsRebuild(): boolean {
    return this.needsFullRebuild;
  }

  /** Whether the offscreen canvas has been initialized. */
  get isInitialized(): boolean {
    return this.offscreen !== null && this.offCtx !== null;
  }

  /** Number of points already rendered. */
  get renderedCount(): number {
    return this.renderedPoints.size;
  }

  // --- Private rendering methods ---

  /**
   * Compute a bounding box around the scan points with generous padding.
   * Used as a clip region for Voronoi rendering to prevent truly infinite
   * cells from shooting across the image, while preserving the natural
   * extent of outer Voronoi cells (which extend ~50-100% beyond the point
   * bounding box for edge cells).
   */
  private computeClipBounds(points: DisplayPoint[]): void {
    if (points.length === 0) {
      this.clipBounds = null;
      return;
    }

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (const pt of points) {
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }

    if (!isFinite(minX)) {
      this.clipBounds = null;
      return;
    }

    const spanX = maxX - minX;
    const spanY = maxY - minY;
    // 100% padding — outer Voronoi cells naturally extend ~50% beyond
    // the point bounding box. This only catches truly infinite cells.
    const padX = Math.max(spanX * 1.0, 50);
    const padY = Math.max(spanY * 1.0, 50);

    this.clipBounds = {
      x: minX - padX,
      y: minY - padY,
      w: spanX + 2 * padX,
      h: spanY + 2 * padY,
    };
  }

  /**
   * Render a single layer with its own colormap.
   */
  private renderSingleLayer(
    ctx: OffscreenCanvasRenderingContext2D,
    layer: ScalarLayer,
    points: DisplayPoint[],
    voronoi: VoronoiGeometry | null,
    geometryMode: GeometryMode,
  ): void {
    const colormap = getColormap(layer.colormap.name);
    const [rangeMin, rangeMax] = layer.colormap.range;
    // Default pixel scale: ~10 um/px for ACI
    const pixelScale = 10;

    for (let i = 0; i < points.length && i < layer.values.length; i++) {
      const pt = points[i];
      const lv = layer.values[i];
      const isEdge = voronoi ? voronoi.edge_mask[i] ?? false : false;
      const region = voronoi && i < voronoi.regions.length ? voronoi.regions[i] : null;

      this.renderPoint(
        pt.index,
        pt,
        lv.value,
        lv.status,
        region,
        voronoi ? voronoi.vertices : null,
        isEdge,
        colormap,
        [rangeMin, rangeMax],
        geometryMode,
        layer.opacity,
        pixelScale,
      );
    }
  }

  /**
   * RGB composite: each layer renders into its designated channel.
   * Uses ImageData for per-pixel channel assignment.
   */
  private renderRGB(
    ctx: OffscreenCanvasRenderingContext2D,
    layers: ScalarLayer[],
    points: DisplayPoint[],
    voronoi: VoronoiGeometry | null,
    geometryMode: GeometryMode,
  ): void {
    // For RGB mode, we render each layer separately with mono-channel colormaps,
    // then composite with 'lighter' blend mode (additive).
    const prevComposite = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = 'lighter';

    for (const layer of layers) {
      const colormap = getColormap(layer.colormap.name);
      const [rangeMin, rangeMax] = layer.colormap.range;
      const pixelScale = 10;

      for (let i = 0; i < points.length && i < layer.values.length; i++) {
        const pt = points[i];
        const lv = layer.values[i];
        const isEdge = voronoi ? voronoi.edge_mask[i] ?? false : false;
        const region = voronoi && i < voronoi.regions.length ? voronoi.regions[i] : null;
        const hasValue = lv.value !== null && lv.status === 'measured';

        let fillColor: string;
        if (hasValue) {
          const t = normalizeValue(lv.value!, rangeMin, rangeMax);
          const rgb = colormap(t);
          fillColor = colormapToRGBA(rgb, layer.opacity);
        } else {
          fillColor = 'rgba(0, 0, 0, 0)'; // transparent for non-detections in RGB mode
        }

        if (geometryMode === 'voronoi' || geometryMode === 'combined') {
          this.drawVoronoiCell(ctx, pt, region, voronoi?.vertices ?? null, isEdge, fillColor, hasValue, layer.opacity);
        }
        if (geometryMode === 'ring' || geometryMode === 'combined') {
          this.drawRing(ctx, pt, hasValue, fillColor, layer.opacity, pixelScale);
        }
      }
    }

    ctx.globalCompositeOperation = prevComposite;
  }

  /**
   * Alpha stack: each layer rendered with its own opacity, composited via 'lighter'.
   */
  private renderAlphaStack(
    ctx: OffscreenCanvasRenderingContext2D,
    layers: ScalarLayer[],
    points: DisplayPoint[],
    voronoi: VoronoiGeometry | null,
    geometryMode: GeometryMode,
  ): void {
    const prevComposite = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = 'lighter';

    for (const layer of layers) {
      this.renderSingleLayer(ctx, layer, points, voronoi, geometryMode);
    }

    ctx.globalCompositeOperation = prevComposite;
  }

  /**
   * Draw a single Voronoi cell polygon.
   */
  private drawVoronoiCell(
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
    _point: DisplayPoint,
    region: number[] | null,
    vertices: [number, number][] | null,
    isEdge: boolean,
    fillColor: string,
    hasValue: boolean,
    opacity: number,
  ): void {
    if (!region || !vertices || region.length < 3) return;

    // Build polygon path from vertex indices
    ctx.beginPath();
    let first = true;
    for (const vi of region) {
      if (vi < 0 || vi >= vertices.length) return; // invalid index
      const [vx, vy] = vertices[vi];
      if (first) {
        ctx.moveTo(vx, vy);
        first = false;
      } else {
        ctx.lineTo(vx, vy);
      }
    }
    ctx.closePath();

    // Fill
    if (isEdge) {
      // Edge cells: reduced opacity
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Dashed outline for edge cells
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = hasValue
        ? `rgba(255, 255, 255, ${opacity * 0.3})`
        : `rgba(128, 128, 128, ${opacity * 0.2})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
  }

  /**
   * Draw a ring marker at a point.
   * 100 um diameter in image space = ~10px radius at 10 um/px ACI scale.
   */
  private drawRing(
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
    point: DisplayPoint,
    hasValue: boolean,
    fillColor: string,
    opacity: number,
    pixelScale: number,
  ): void {
    // 100 um diameter laser spot, convert to image-space pixels
    const radiusPx = 50 / pixelScale; // 50 um radius / (um/px)

    ctx.beginPath();
    ctx.arc(point.x, point.y, radiusPx, 0, Math.PI * 2);

    if (hasValue) {
      // Detection: filled circle + 2px stroke
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.6})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      // Non-detection: 1px white outline, low alpha
      ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.3})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}
