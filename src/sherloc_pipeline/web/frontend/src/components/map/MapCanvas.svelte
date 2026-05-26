<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher, afterUpdate } from 'svelte';
  import type {
    PointSet,
    ScalarLayer,
    Transform,
    GeometryMode,
    DisplayMode,
    DisplayPoint,
  } from '../../lib/types/map';
  import { BaseImageRenderer } from '../../lib/renderers/BaseImageRenderer';
  import { DataLayerRenderer } from '../../lib/renderers/DataLayerRenderer';
  import { OverlayRenderer } from '../../lib/renderers/OverlayRenderer';
  import type { OverlayOptions } from '../../lib/renderers/OverlayRenderer';
  import { getColormap, normalizeValue, colormapToRGBA } from '../../lib/colormaps';

  // --- Props ---
  export let scanId: string;
  export let pointSet: PointSet | null = null;
  export let layers: ScalarLayer[] = [];
  export let geometryMode: GeometryMode = 'voronoi';
  export let displayMode: DisplayMode = { type: 'all_domains' };
  export let brightness: number = 1.0;
  export let contrast: number = 1.0;
  // ACI images are pre-decoded by MapMode (via fetchAciImage, which attaches
  // Bearer auth and returns a decoded HTMLImageElement). MapCanvas just holds
  // the reference and passes it to the BaseImageRenderer. Pre-v4.1.8 this
  // component accepted URLs and did the fetch internally — that bypassed
  // Auth0 Bearer auth and produced 401s (Session 93 design memo §2.3).
  export let aciImage: HTMLImageElement | null = null;
  export let colorizedAciImage: HTMLImageElement | null = null;
  // When the parent is awaiting fetchAciImage, render a spinner instead of
  // the "No ACI image available" placeholder. False once the parent has
  // resolved (image present, auth-required state, or hard error).
  export let aciLoading: boolean = false;

  // Local toggle: Colorized variant when available, otherwise the original.
  // Matches the workbench AciViewer convention (button on the image itself).
  let useColorized: boolean = false;
  export let pixelScale: number = 10; // um per pixel, default ACI scale
  export let overlayOpacity: number = 0.7;
  export let showPointPositions: boolean = true;
  export let initialTransform: Transform | null = null;

  const dispatch = createEventDispatcher<{
    pointClick: { pointIndex: number };
    pointHover: { pointIndex: number | null };
    lassoComplete: { polygon: [number, number][] };
  }>();

  // --- Canvas refs ---
  let canvasBase: HTMLCanvasElement;
  let canvasData: HTMLCanvasElement;
  let canvasUI: HTMLCanvasElement;
  let container: HTMLDivElement;

  // --- Rendering contexts ---
  let ctxBase: CanvasRenderingContext2D | null = null;
  let ctxData: CanvasRenderingContext2D | null = null;
  let ctxUI: CanvasRenderingContext2D | null = null;

  // --- Renderers ---
  const baseRenderer = new BaseImageRenderer();
  const dataRenderer = new DataLayerRenderer();
  const overlayRenderer = new OverlayRenderer();

  // --- Transform state (shared across all canvases) ---
  let transform: Transform = initialTransform ?? { x: 0, y: 0, scale: 1 };
  let needsRedraw = true;
  let animFrameId: number | null = null;

  // --- Interaction state ---
  let isDragging = false;
  let dragStart = { x: 0, y: 0 };
  let lastTransform = { x: 0, y: 0 };
  let lassoPoints: [number, number][] = [];
  let isLassoing = false;
  let hoveredPoint: number | null = null;
  let tooltipX = 0;
  let tooltipY = 0;

  // --- Image loading ---
  let loading = false;
  let imageError = false;
  let loadGeneration = 0;
  // canvasReady gate: synchronous applyActiveImage() needs canvasBase /
  // ctxBase set. The reactive statement below can fire before onMount()
  // when activeImage is non-null at component instantiation; without this
  // flag the renderer would operate on uninitialized refs (Codex PR9 R1 F2).
  let canvasReady = false;

  // --- Lifecycle ---

  onMount(() => {
    setupCanvases();
    canvasReady = true;
    window.addEventListener('resize', handleResize);
    if (aciImage || colorizedAciImage) {
      applyActiveImage();
    }
  });

  onDestroy(() => {
    window.removeEventListener('resize', handleResize);
    if (animFrameId !== null) {
      cancelAnimationFrame(animFrameId);
    }
    baseRenderer.dispose();
  });

  afterUpdate(() => {
    scheduleRedraw();
  });

  // --- Reactive statements ---

  // Swap the active image when the colorized toggle or the source prop
  // changes. The image is already decoded by MapMode via fetchAciImage;
  // we just hand the reference to the BaseImageRenderer. Gated on
  // canvasReady because applyActiveImage touches canvas refs initialized
  // in onMount (Codex PR9 R1 F2 — without the gate, the reactive
  // statement can fire before onMount when activeImage is non-null at
  // component instantiation).
  //
  // When activeImage transitions to null (parent cleared the prop on a
  // new scan load or AuthRequiredError), explicitly dispose the renderer
  // and clear the canvas so prior-scan pixels do not remain on screen
  // during the placeholder / auth-required UI state (Codex PR9 R2 F4 —
  // residual of R1 F1; parent-side clearing alone is not enough).
  $: activeImage = useColorized && colorizedAciImage ? colorizedAciImage : aciImage;
  $: if (activeImage && canvasReady) applyActiveImage();
  $: if (!activeImage && canvasReady) clearActiveImage();

  // Invalidate data layer when geometry mode or display mode changes
  $: if (geometryMode || displayMode) {
    dataRenderer.invalidate();
    scheduleRedraw();
  }

  // Invalidate data layer when layers change
  $: if (layers) {
    dataRenderer.invalidate();
    scheduleRedraw();
  }

  // Redraw when overlay opacity changes (applied at blit time, no rebuild needed)
  $: if (overlayOpacity !== undefined) {
    scheduleRedraw();
  }

  // Initialize data renderer when point set loads (we know the image dimensions)
  $: if (pointSet && baseRenderer.loaded) {
    const w = baseRenderer.naturalWidth;
    const h = baseRenderer.naturalHeight;
    if (w > 0 && h > 0) {
      dataRenderer.init(w, h);
      dataRenderer.invalidate();
      scheduleRedraw();
    }
  }

  // --- Canvas setup ---

  function setupCanvases(): void {
    if (!canvasBase || !canvasData || !canvasUI || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    for (const cvs of [canvasBase, canvasData, canvasUI]) {
      cvs.width = w * dpr;
      cvs.height = h * dpr;
      cvs.style.width = `${w}px`;
      cvs.style.height = `${h}px`;
    }

    ctxBase = canvasBase.getContext('2d');
    ctxData = canvasData.getContext('2d');
    ctxUI = canvasUI.getContext('2d');

    if (ctxBase) ctxBase.scale(dpr, dpr);
    if (ctxData) ctxData.scale(dpr, dpr);
    if (ctxUI) ctxUI.scale(dpr, dpr);

    scheduleRedraw();
  }

  function handleResize(): void {
    setupCanvases();
    // Don't re-fit — maintain the user's current zoom/pan.
    // Just redraw at the new canvas size.
    scheduleRedraw();
  }

  // --- Image application ---

  // Clear renderer state + canvas pixels when the parent has nulled the
  // image props (Codex PR9 R2 F4). MapMode.svelte does this on every new
  // scan load and on auth/error paths; without the symmetric child-side
  // clear, prior-scan pixels remain visible on canvasBase during the
  // intermediate state.
  function clearActiveImage(): void {
    loadGeneration++; // invalidate any in-flight applyActiveImage
    imageError = false;
    baseRenderer.dispose();
    dataRenderer.invalidate();
    if (ctxBase && canvasBase) {
      const dpr = window.devicePixelRatio || 1;
      ctxBase.clearRect(0, 0, canvasBase.width / dpr, canvasBase.height / dpr);
    }
    if (ctxData && canvasData) {
      const dpr = window.devicePixelRatio || 1;
      ctxData.clearRect(0, 0, canvasData.width / dpr, canvasData.height / dpr);
    }
    if (ctxUI && canvasUI) {
      const dpr = window.devicePixelRatio || 1;
      ctxUI.clearRect(0, 0, canvasUI.width / dpr, canvasUI.height / dpr);
    }
  }

  // applyActiveImage replaces the legacy URL-based loadImage(): the image
  // is already decoded by MapMode via fetchAciImage (authenticated path).
  // We synchronously hand the HTMLImageElement to BaseImageRenderer and
  // run the same dataRenderer init + transform-fit lifecycle as before.
  // Stale-load guard via loadGeneration (rapid scan/colorized toggles).
  function applyActiveImage(): void {
    const img = activeImage;
    if (!img) return;

    const gen = ++loadGeneration;
    loading = false;
    imageError = false;

    try {
      baseRenderer.setImage(img);
      if (gen !== loadGeneration) return; // stale

      // Initialize data renderer at image dimensions
      const w = baseRenderer.naturalWidth;
      const h = baseRenderer.naturalHeight;
      if (w > 0 && h > 0) {
        dataRenderer.init(w, h);
      }

      // Restore saved transform or fit to image on first load
      if (initialTransform) {
        transform = { ...initialTransform };
        scheduleRedraw();
      } else {
        fitToImage();
      }
    } catch {
      if (gen !== loadGeneration) return;
      imageError = true;
    } finally {
      if (gen === loadGeneration) {
        scheduleRedraw();
      }
    }
  }

  function fitToImage(): void {
    if (!baseRenderer.loaded || !canvasBase) return;

    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvasBase.width / dpr;
    const canvasH = canvasBase.height / dpr;
    const imgW = baseRenderer.naturalWidth;
    const imgH = baseRenderer.naturalHeight;

    const scaleX = canvasW / imgW;
    const scaleY = canvasH / imgH;
    const fitScale = Math.min(scaleX, scaleY) * 0.99;

    transform = {
      x: (canvasW - imgW * fitScale) / 2,
      y: (canvasH - imgH * fitScale) / 2,
      scale: fitScale,
    };
    scheduleRedraw();
  }

  // --- Draw cycle ---

  function scheduleRedraw(): void {
    needsRedraw = true;
    if (animFrameId !== null) return;
    animFrameId = requestAnimationFrame(() => {
      animFrameId = null;
      if (needsRedraw) {
        needsRedraw = false;
        draw();
      }
    });
  }

  function draw(): void {
    drawBaseImage();
    drawDataLayers();
    drawOverlays();
  }

  function drawBaseImage(): void {
    if (!ctxBase || !canvasBase) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvasBase.width / dpr;
    const h = canvasBase.height / dpr;

    ctxBase.clearRect(0, 0, w, h);

    // Dark background
    ctxBase.fillStyle = '#0a0a1a';
    ctxBase.fillRect(0, 0, w, h);

    baseRenderer.draw(ctxBase, transform, brightness, contrast);
  }

  function drawDataLayers(): void {
    if (!ctxData || !canvasData) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvasData.width / dpr;
    const h = canvasData.height / dpr;

    ctxData.clearRect(0, 0, w, h);

    if (!pointSet || layers.length === 0) return;

    // Skip rendering entirely if no layer has loaded values (clean ACI view)
    const anyLayerHasValues = layers.some((l) => l.values.length > 0);
    if (!anyLayerHasValues) return;

    // Ensure data renderer is initialized (may not be if image loaded after layers)
    if (!dataRenderer.isInitialized && baseRenderer.loaded) {
      const iw = baseRenderer.naturalWidth;
      const ih = baseRenderer.naturalHeight;
      if (iw > 0 && ih > 0) {
        dataRenderer.init(iw, ih);
        dataRenderer.invalidate();
      }
    }

    const visibleLayers = getVisibleLayers();
    if (visibleLayers.length === 0) return;

    // Apply overlay opacity to the entire data layer
    ctxData.save();
    ctxData.globalAlpha = overlayOpacity;

    // For ring-only mode, draw rings directly as vector graphics on the
    // display canvas so they stay crisp at any zoom level (no pixelation).
    if (geometryMode === 'ring') {
      const isRgb = displayMode.type === 'rgb_mix';
      dataRenderer.drawRingsDirect(ctxData, visibleLayers, pointSet.points, transform, isRgb);
      ctxData.restore();
      return;
    }

    // Determine composite mode from display mode
    const compositeMode = displayMode.type === 'rgb_mix' ? 'rgb' as const : 'single' as const;

    // For voronoi/combined modes, use the OffscreenCanvas pipeline
    // (Voronoi polygons fill area so pixelation is less noticeable)
    if (dataRenderer.needsRebuild) {
      dataRenderer.renderFull(
        visibleLayers,
        pointSet.points,
        pointSet.voronoi,
        geometryMode,
        compositeMode,
      );
    }

    // Blit offscreen to visible canvas
    dataRenderer.blitTo(ctxData, transform);

    // For combined mode, also draw rings directly on top for crisp circles
    if (geometryMode === 'combined') {
      const isRgb = displayMode.type === 'rgb_mix';
      dataRenderer.drawRingsDirect(ctxData, visibleLayers, pointSet.points, transform, isRgb);
    }

    ctxData.restore();
  }

  function drawOverlays(): void {
    if (!ctxUI || !canvasUI) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvasUI.width / dpr;
    const h = canvasUI.height / dpr;

    ctxUI.clearRect(0, 0, w, h);

    // Draw scan point positions as transparent gray circles
    if (showPointPositions && pointSet && pointSet.points.length > 0) {
      drawPointPositions(ctxUI);
    }

    // Draw hover highlight in world space
    if (hoveredPoint !== null && pointSet) {
      drawHoverHighlight(ctxUI);
    }

    // Build overlay options
    const options: OverlayOptions = {};

    // Color scale from first visible layer
    const visibleLayers = getVisibleLayers();
    if (visibleLayers.length > 0 && displayMode.type !== 'rgb_mix') {
      const layer = visibleLayers[0];
      options.colorScale = {
        colormap: getColormap(layer.colormap.name),
        range: layer.colormap.range,
        label: layer.label,
        unit: layer.value_type === 'snr' ? 'SNR' : layer.value_type,
      };
    }

    // Scale bar
    options.scaleBar = {
      pixelScale,
      transform,
    };

    // Lasso polygon (in screen space)
    if (isLassoing && lassoPoints.length > 1) {
      options.selectionPolygon = lassoPoints;
    }

    overlayRenderer.draw(ctxUI, w, h, options);
  }

  /**
   * Draw scan point positions as transparent gray circles (vector, crisp at any zoom).
   * Shows the spatial footprint of the scan before any fitting data is loaded.
   */
  function drawPointPositions(ctx: CanvasRenderingContext2D): void {
    if (!pointSet) return;

    const radiusPx = 50 / pixelScale; // 100 µm diameter laser spot

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    const strokeWidth = Math.max(0.5, 1 / transform.scale);
    ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)';
    ctx.lineWidth = strokeWidth;

    for (const pt of pointSet.points) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, radiusPx, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  /**
   * Draw a highlight ring around the hovered point in world space.
   */
  function drawHoverHighlight(ctx: CanvasRenderingContext2D): void {
    if (hoveredPoint === null || !pointSet) return;

    const pt = pointSet.points.find((p) => p.index === hoveredPoint);
    if (!pt) return;

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    const radius = 7; // image-space pixels
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 220, 50, 0.9)';
    ctx.lineWidth = Math.max(0.5, 2 / transform.scale);
    ctx.stroke();

    // Second outer ring
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius + 2 / transform.scale, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 220, 50, 0.4)';
    ctx.lineWidth = Math.max(0.5, 1 / transform.scale);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Filter layers based on current display mode.
   */
  function getVisibleLayers(): ScalarLayer[] {
    // Helper: find the domain-level "all" layer (class_id=null) with loaded values
    function domainAllLayer(domain: string): ScalarLayer | undefined {
      return layers.find(
        (l) => l.domain === domain && l.class_id === null && l.values.length > 0
          && l.values.some(v => v.status === 'measured' && v.value !== null),
      );
    }

    switch (displayMode.type) {
      case 'all_domains': {
        // One domain-level layer per domain showing dominant assignment per point
        const seen = new Set<string>();
        const result: ScalarLayer[] = [];
        for (const l of layers) {
          if (seen.has(l.domain)) continue;
          const all = domainAllLayer(l.domain);
          if (all) {
            seen.add(l.domain);
            result.push(all);
          }
        }
        return result;
      }

      case 'domain': {
        // Single domain-level layer showing dominant assignment per point
        const all = domainAllLayer(displayMode.domain);
        if (all) return [all];
        return [];
      }

      case 'class':
        return layers.filter(
          (l) =>
            l.domain === displayMode.domain &&
            l.class_id === displayMode.class_id &&
            l.values.length > 0,
        );

      case 'rgb_mix': {
        // Return layers matching the RGB channel refs
        const result: ScalarLayer[] = [];
        const { channels } = displayMode;
        for (const ref of [channels.red, channels.green, channels.blue]) {
          if (!ref) continue;
          const match = layers.find(
            (l) =>
              l.domain === ref.domain &&
              l.class_id === ref.class_id &&
              l.value_type === ref.value_type &&
              l.values.length > 0,
          );
          if (match) result.push(match);
        }
        return result;
      }

      default:
        return layers.filter((l) => l.values.length > 0);
    }
  }

  // --- Coordinate transforms ---

  function screenToImage(clientX: number, clientY: number): { x: number; y: number } {
    const rect = canvasUI.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    return {
      x: (cx - transform.x) / transform.scale,
      y: (cy - transform.y) / transform.scale,
    };
  }

  function screenToCanvas(clientX: number, clientY: number): { x: number; y: number } {
    const rect = canvasUI.getBoundingClientRect();
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  /**
   * Find nearest point within a scale-adaptive hit radius.
   */
  function findNearestPoint(imgX: number, imgY: number): number | null {
    if (!pointSet) return null;

    let nearest: number | null = null;
    let minDist = Infinity;
    const hitRadius = Math.max(8, 12 / transform.scale);

    for (const pt of pointSet.points) {
      const dx = pt.x - imgX;
      const dy = pt.y - imgY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < hitRadius && dist < minDist) {
        minDist = dist;
        nearest = pt.index;
      }
    }
    return nearest;
  }

  // --- Mouse interaction handlers ---

  function handleMouseDown(e: MouseEvent): void {
    if (e.shiftKey) {
      // Shift+drag = lasso selection
      isLassoing = true;
      lassoPoints = [];
      const screen = screenToCanvas(e.clientX, e.clientY);
      lassoPoints.push([screen.x, screen.y]);
      return;
    }

    isDragging = true;
    dragStart = { x: e.clientX, y: e.clientY };
    lastTransform = { x: transform.x, y: transform.y };
  }

  function handleMouseMove(e: MouseEvent): void {
    if (isLassoing) {
      const screen = screenToCanvas(e.clientX, e.clientY);
      lassoPoints = [...lassoPoints, [screen.x, screen.y]];
      scheduleRedraw();
      return;
    }

    if (isDragging) {
      transform = {
        ...transform,
        x: lastTransform.x + (e.clientX - dragStart.x),
        y: lastTransform.y + (e.clientY - dragStart.y),
      };
      scheduleRedraw();
      return;
    }

    // Hover detection
    const img = screenToImage(e.clientX, e.clientY);
    const nearest = findNearestPoint(img.x, img.y);
    if (nearest !== hoveredPoint) {
      hoveredPoint = nearest;
      const rect = canvasUI.getBoundingClientRect();
      tooltipX = e.clientX - rect.left;
      tooltipY = e.clientY - rect.top;
      dispatch('pointHover', { pointIndex: nearest });
      scheduleRedraw();
    }
  }

  function handleMouseUp(e: MouseEvent): void {
    if (isLassoing) {
      isLassoing = false;
      if (lassoPoints.length > 2) {
        // Convert screen-space lasso to image-space polygon
        const imagePolygon: [number, number][] = lassoPoints.map(([sx, sy]) => {
          const ix = (sx - transform.x) / transform.scale;
          const iy = (sy - transform.y) / transform.scale;
          return [ix, iy];
        });
        dispatch('lassoComplete', { polygon: imagePolygon });
      }
      lassoPoints = [];
      scheduleRedraw();
      return;
    }

    const wasDragging = isDragging;
    isDragging = false;

    // Click vs drag: only fire click if movement < 4px
    const dx = Math.abs(e.clientX - dragStart.x);
    const dy = Math.abs(e.clientY - dragStart.y);
    if (wasDragging && dx < 4 && dy < 4) {
      const img = screenToImage(e.clientX, e.clientY);
      const nearest = findNearestPoint(img.x, img.y);
      if (nearest !== null) {
        dispatch('pointClick', { pointIndex: nearest });
      }
    }
  }

  function handleWheel(e: WheelEvent): void {
    const rect = canvasUI.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const newScale = Math.max(0.1, Math.min(20, transform.scale * zoomFactor));

    // Zoom centered on cursor
    transform = {
      x: cx - (cx - transform.x) * (newScale / transform.scale),
      y: cy - (cy - transform.y) * (newScale / transform.scale),
      scale: newScale,
    };
    scheduleRedraw();
  }

  function handleMouseLeave(): void {
    isDragging = false;
    if (isLassoing) {
      isLassoing = false;
      lassoPoints = [];
    }
    if (hoveredPoint !== null) {
      hoveredPoint = null;
      dispatch('pointHover', { pointIndex: null });
    }
    scheduleRedraw();
  }

  // --- Public methods exposed for parent components ---

  /** Reset the view to fit the full image. */
  export function resetView(): void {
    fitToImage();
  }

  /** Zoom to the extents of the current point set. */
  export function zoomToPoints(): void {
    if (!pointSet || pointSet.points.length === 0 || !canvasUI) return;

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const pt of pointSet.points) {
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }
    if (!isFinite(minX)) return;

    const pad = 50;
    minX -= pad;
    minY -= pad;
    maxX += pad;
    maxY += pad;

    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvasUI.width / dpr;
    const canvasH = canvasUI.height / dpr;
    const spanX = Math.max(maxX - minX, 2 * pad);
    const spanY = Math.max(maxY - minY, 2 * pad);

    const scaleX = canvasW / spanX;
    const scaleY = canvasH / spanY;
    const newScale = Math.min(scaleX, scaleY);

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    transform = {
      x: canvasW / 2 - cx * newScale,
      y: canvasH / 2 - cy * newScale,
      scale: newScale,
    };
    scheduleRedraw();
  }

  /** Zoom to native resolution (1:1 pixel mapping), centered on image. */
  export function zoomNative(): void {
    if (!baseRenderer.loaded || !canvasBase) return;

    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvasBase.width / dpr;
    const canvasH = canvasBase.height / dpr;
    const imgW = baseRenderer.naturalWidth;
    const imgH = baseRenderer.naturalHeight;

    // 1:1 means scale = 1.0 (one image pixel = one CSS pixel)
    transform = {
      x: (canvasW - imgW) / 2,
      y: (canvasH - imgH) / 2,
      scale: 1.0,
    };
    scheduleRedraw();
  }

  /** Get current transform for saving state across remounts. */
  export function getTransform(): Transform {
    return { ...transform };
  }

  // renderIncrementalPoint removed — fitting results now flow through mapLayers
</script>

<div class="map-canvas-container" bind:this={container}>
  {#if loading || aciLoading}
    <div class="map-loading">
      <div class="spinner"></div>
      <span>Loading ACI image...</span>
    </div>
  {:else if imageError || !aciImage}
    <div class="map-placeholder">
      {#if !aciImage}
        No ACI image available
      {:else}
        Failed to load ACI image
      {/if}
    </div>
  {/if}

  <canvas
    bind:this={canvasBase}
    class="map-canvas base"
    class:hidden={loading || aciLoading || imageError || !aciImage}
    style="z-index: 1"
  />
  <canvas
    bind:this={canvasData}
    class="map-canvas data"
    class:hidden={loading || aciLoading || imageError || !aciImage}
    style="z-index: 2"
  />
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <canvas
    bind:this={canvasUI}
    class="map-canvas ui"
    class:hidden={loading || aciLoading || imageError || !aciImage}
    style="z-index: 3"
    on:mousedown={handleMouseDown}
    on:mousemove={handleMouseMove}
    on:mouseup={handleMouseUp}
    on:wheel|preventDefault={handleWheel}
    on:mouseleave={handleMouseLeave}
  />

  {#if hoveredPoint !== null && pointSet}
    <div
      class="point-tooltip"
      style="left: {tooltipX + 14}px; top: {tooltipY - 10}px"
    >
      Point {hoveredPoint}
    </div>
  {/if}

  {#if !loading && !aciLoading && !imageError && aciImage}
    <div class="map-controls">
      {#if colorizedAciImage}
        <button
          class="ctrl-btn"
          on:click={() => (useColorized = !useColorized)}
          title={useColorized ? 'Show grayscale ACI' : 'Show colorized ACI'}
        >
          {useColorized ? 'Grayscale' : 'Colorized'}
        </button>
      {/if}
      <button class="ctrl-btn" on:click={resetView} title="Fit full image">
        Fit
      </button>
      <button class="ctrl-btn" on:click={zoomToPoints} title="Zoom to scan points">
        Scan
      </button>
      <button class="ctrl-btn" on:click={zoomNative} title="Native resolution (1:1)">
        1:1
      </button>
      <span class="zoom-label">{(transform.scale * 100).toFixed(0)}%</span>
    </div>
  {/if}
</div>

<style>
  .map-canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 300px;
    overflow: hidden;
    background: #0a0a1a;
  }

  .map-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }

  .map-canvas.ui {
    cursor: grab;
  }

  .map-canvas.ui:active {
    cursor: grabbing;
  }

  .map-canvas.hidden {
    display: none;
  }

  .map-loading {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    color: #94a3b8;
    font-size: 0.85rem;
    z-index: 10;
  }

  .map-placeholder {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #64748b;
    font-size: 0.9rem;
    text-align: center;
    padding: 24px;
    z-index: 10;
  }

  .point-tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(15, 23, 42, 0.9);
    color: #e2e8f0;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 4px;
    white-space: nowrap;
    z-index: 20;
  }

  .map-controls {
    position: absolute;
    bottom: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    background: rgba(15, 23, 42, 0.85);
    padding: 4px 8px;
    border-radius: 6px;
    z-index: 15;
  }

  .ctrl-btn {
    font-size: 0.75rem;
    padding: 3px 8px;
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    cursor: pointer;
  }

  .ctrl-btn:hover {
    background: rgba(255, 255, 255, 0.2);
  }

  .zoom-label {
    font-family: ui-monospace, monospace;
    font-size: 0.7rem;
    color: #94a3b8;
    margin-left: 4px;
    min-width: 36px;
    text-align: right;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(148, 163, 184, 0.3);
    border-top-color: #94a3b8;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
