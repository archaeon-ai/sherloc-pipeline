<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher, afterUpdate } from 'svelte';
  import type { ScanPoint } from '../lib/types';
  import { fetchAciImage, AuthRequiredError } from '../lib/api';
  import { getAciPixel, computeAciBBox } from '../lib/aciCoords';
  import { OverlayRenderer } from '../lib/renderers/OverlayRenderer';
  import { buildAciOverlayOptions } from '../lib/aciOverlay';

  export let scanId: string = '';
  export let points: ScanPoint[] = [];
  export let selectedIndices: number[] = [];
  export let selectionMode: 'average' | 'subset' | 'point' = 'average';
  export let selectedPointIdx: number | null = null;
  // µm per pixel. Default 10 mirrors Map mode's MapCanvas default (ACI
  // nominal pixel scale); per-scan `pixel_scale_um` plumbing for BOTH
  // viewers is tracked as issue #28 (deferred to v1.0.1; touching only
  // Workbench here would diverge from Map mode and violate the parity
  // contract issue #14 requires).
  export let pixelScale: number = 10;
  // True iff the backend reports a sol_NNNN_colorized/ R2 sibling for
  // this scan's ACI. False (default) keeps the "Colorized" button
  // visible but disabled with an explanatory tooltip — clicking used
  // to silently re-serve grayscale because `find_colorized_key`
  // returned None for 170 of 205 historical sols, which read as a
  // broken button.
  export let colorizedAvailable: boolean = false;

  const dispatch = createEventDispatcher<{
    pointClick: { pointIndex: number };
  }>();

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;
  let container: HTMLDivElement;
  let aciImage: HTMLImageElement | null = null;
  let colorized = false;
  // Defensive: if the parent re-binds `colorizedAvailable` to false
  // while the user already had Colorized active (e.g. router-navigated
  // to a different scan without remount), force the toggle off and let
  // the next loadImage call serve grayscale. Without this, the canvas
  // would be requesting `?colorized=true` against a scan whose backend
  // would silently fall back to grayscale — the exact UX this fix
  // exists to prevent.
  $: if (!colorizedAvailable && colorized) {
    colorized = false;
    if (scanId) loadImage();
  }
  let pointOpacity = 0.8;
  let pointHue = 180; // cyan default
  let loading = false;
  let imageError = false;
  let authRequired = false;
  let hoveredPoint: number | null = null;
  let tooltipX = 0;
  let tooltipY = 0;

  // Pan/zoom state
  let transform = { x: 0, y: 0, scale: 1 };
  let isDragging = false;
  let dragStart = { x: 0, y: 0 };
  let lastTransform = { x: 0, y: 0 };

  // Rendering
  let animFrameId: number | null = null;
  const overlayRenderer = new OverlayRenderer();

  let loadGeneration = 0;

  onMount(() => {
    if (canvas) {
      ctx = canvas.getContext('2d');
      setupCanvasSize();
    }
    window.addEventListener('resize', handleResize);
    if (scanId) {
      loadImage();
    }
  });

  onDestroy(() => {
    window.removeEventListener('resize', handleResize);
    if (animFrameId !== null) {
      cancelAnimationFrame(animFrameId);
    }
  });

  // Re-render when any visual state changes
  afterUpdate(() => {
    scheduleRender();
  });

  function setupCanvasSize() {
    if (!canvas || !container) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
    scheduleRender();
  }

  function handleResize() {
    setupCanvasSize();
  }

  async function loadImage() {
    if (!scanId) return;
    const gen = ++loadGeneration;
    loading = true;
    imageError = false;
    authRequired = false;
    aciImage = null;

    try {
      // fetchAciImage attaches Authorization: Bearer; throws AuthRequiredError
      // if no authenticated session (Session 93 design memo §2.2).
      const img = await fetchAciImage(scanId, { colorized });
      if (gen !== loadGeneration) return; // stale
      aciImage = img;
      fitToContainer();
    } catch (e) {
      if (gen !== loadGeneration) return;
      if (e instanceof AuthRequiredError) {
        authRequired = true;
      } else {
        imageError = true;
      }
      aciImage = null;
    } finally {
      if (gen === loadGeneration) {
        loading = false;
        scheduleRender();
      }
    }
  }

  function fitToContainer() {
    if (!aciImage || !canvas) return;
    const canvasW = canvas.width / (window.devicePixelRatio || 1);
    const canvasH = canvas.height / (window.devicePixelRatio || 1);
    const scaleX = canvasW / aciImage.width;
    const scaleY = canvasH / aciImage.height;
    const fitScale = Math.min(scaleX, scaleY) * 0.95;
    transform = {
      x: (canvasW - aciImage.width * fitScale) / 2,
      y: (canvasH - aciImage.height * fitScale) / 2,
      scale: fitScale,
    };
    scheduleRender();
  }

  function scheduleRender() {
    if (animFrameId !== null) return;
    animFrameId = requestAnimationFrame(() => {
      animFrameId = null;
      render();
    });
  }

  function render() {
    if (!ctx || !canvas) return;
    const canvasW = canvas.width / (window.devicePixelRatio || 1);
    const canvasH = canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, canvasW, canvasH);

    // Background
    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, canvasW, canvasH);

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    // Draw ACI image
    if (aciImage) {
      ctx.drawImage(aciImage, 0, 0);
    }

    // Draw measurement points
    drawPoints();

    ctx.restore();

    // Screen-space overlays (scale bar) — Map-mode parity overlay. The
    // gate-on-image + scaleBar contract lives in buildAciOverlayOptions
    // so the wiring is unit-testable without mounting the component
    // (Codex /code-review PR #27 R1 F2 helper-boundary gate).
    overlayRenderer.draw(
      ctx,
      canvasW,
      canvasH,
      buildAciOverlayOptions({ aciImage, pixelScale, transform }),
    );
  }

  function drawPoints() {
    if (!ctx || points.length === 0) return;

    // Laser spot is ~100 μm, ACI pixel scale ~10 μm/px → radius ≈ 5 px in image space
    const pointRadius = 5;
    const lineWidth = Math.max(0.5, 1.5 / transform.scale);

    for (const pt of points) {
      // Coordinate-frame discipline: only render points resolvable to
      // ACI image-pixel space. Loupe scanner_workspace values (~±0.5)
      // are NOT valid pixel coords and must be skipped (issue #16).
      const coord = getAciPixel(pt);
      if (coord === null) continue;
      const x = coord.x;
      const y = coord.y;

      const isSelected = selectedIndices.includes(pt.point_index);
      const isCurrent = selectedPointIdx === pt.point_index;
      const isHovered = hoveredPoint === pt.point_index;

      ctx.beginPath();
      ctx.arc(x, y, pointRadius, 0, Math.PI * 2);

      if (isCurrent) {
        // Current point: brighter, thicker outline
        ctx.strokeStyle = `hsla(${pointHue}, 100%, 65%, ${pointOpacity})`;
        ctx.lineWidth = lineWidth * 2.5;
        ctx.stroke();
      } else if (isSelected) {
        // Selected: user-chosen hue outline
        ctx.strokeStyle = `hsla(${pointHue}, 100%, 50%, ${pointOpacity})`;
        ctx.lineWidth = lineWidth * 2;
        ctx.stroke();
      } else {
        // Non-selected: white outline
        ctx.strokeStyle = `rgba(255, 255, 255, ${pointOpacity * 0.4})`;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
      }

      // Hover highlight ring
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, pointRadius + 2 / transform.scale, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${(pointHue + 60) % 360}, 100%, 60%, ${pointOpacity})`;
        ctx.lineWidth = lineWidth * 1.5;
        ctx.stroke();
      }
    }
  }

  // --- Interaction handlers ---

  function canvasToImage(clientX: number, clientY: number): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;
    return {
      x: (cx - transform.x) / transform.scale,
      y: (cy - transform.y) / transform.scale,
    };
  }

  function findNearestPoint(imgX: number, imgY: number): number | null {
    let nearest: number | null = null;
    let minDist = Infinity;
    const hitRadius = Math.max(8, 12 / transform.scale);

    for (const pt of points) {
      const coord = getAciPixel(pt);
      if (coord === null) continue;
      const dx = coord.x - imgX;
      const dy = coord.y - imgY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < hitRadius && dist < minDist) {
        minDist = dist;
        nearest = pt.point_index;
      }
    }
    return nearest;
  }

  function handleMouseDown(e: MouseEvent) {
    isDragging = true;
    dragStart = { x: e.clientX, y: e.clientY };
    lastTransform = { x: transform.x, y: transform.y };
  }

  function handleMouseMove(e: MouseEvent) {
    if (isDragging) {
      transform = {
        ...transform,
        x: lastTransform.x + (e.clientX - dragStart.x),
        y: lastTransform.y + (e.clientY - dragStart.y),
      };
      scheduleRender();
      return;
    }

    // Hover detection
    const img = canvasToImage(e.clientX, e.clientY);
    const nearest = findNearestPoint(img.x, img.y);
    if (nearest !== hoveredPoint) {
      hoveredPoint = nearest;
      const rect = canvas.getBoundingClientRect();
      tooltipX = e.clientX - rect.left;
      tooltipY = e.clientY - rect.top;
      scheduleRender();
    }
  }

  function handleMouseUp(e: MouseEvent) {
    const wasDragging = isDragging;
    isDragging = false;

    // Only fire click if mouse didn't move much (no drag)
    const dx = Math.abs(e.clientX - dragStart.x);
    const dy = Math.abs(e.clientY - dragStart.y);
    if (wasDragging && dx < 4 && dy < 4) {
      const img = canvasToImage(e.clientX, e.clientY);
      const nearest = findNearestPoint(img.x, img.y);
      if (nearest !== null) {
        dispatch('pointClick', { pointIndex: nearest });
      }
    }
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
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
    scheduleRender();
  }

  function handleMouseLeave() {
    isDragging = false;
    hoveredPoint = null;
    scheduleRender();
  }

  function toggleColorized() {
    colorized = !colorized;
    loadImage();
  }

  function resetView() {
    fitToContainer();
  }

  function exportPng() {
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `aci_${scanId}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  }

  function zoomToScan() {
    if (!aciImage || !canvas || points.length === 0) {
      return;
    }

    // Compute bbox using ONLY points whose coords are ACI-pixel-frame.
    // Returns null if no point resolves — typically Loupe scans without
    // a server-side `spatial.csv` (x_aci_pixel null, x_pixel in
    // scanner_workspace frame). In that case the frontend has no valid
    // image-pixel data to zoom to — leave the existing transform in
    // place rather than masquerade as a "Scan" zoom by fitting the full
    // image. The fix for those scans must come from the backend
    // `_compute_aci_pixels` resolver populating x_aci_pixel.
    const bbox = computeAciBBox(points);
    if (bbox === null) {
      return;
    }

    const pad = 50; // pixels padding in image space
    let minX = bbox.minX - pad;
    let minY = bbox.minY - pad;
    let maxX = bbox.maxX + pad;
    let maxY = bbox.maxY + pad;

    // Clamp to image bounds. Skip clamp on whichever axis has degenerate
    // image dimensions (defensive — `loadImage` ensures aciImage is fully
    // decoded before this runs, but a hostile image could still surface
    // 0/NaN dims and we don't want clamp to invert the span).
    if (aciImage.width > 0) {
      minX = Math.max(0, minX);
      maxX = Math.min(aciImage.width, maxX);
    }
    if (aciImage.height > 0) {
      minY = Math.max(0, minY);
      maxY = Math.min(aciImage.height, maxY);
    }

    const dpr = window.devicePixelRatio || 1;
    const canvasW = canvas.width / dpr;
    const canvasH = canvas.height / dpr;
    if (!(canvasW > 0) || !(canvasH > 0)) {
      // Canvas not yet sized — defer; afterUpdate will retry render.
      return;
    }

    const spanX = maxX - minX;
    const spanY = maxY - minY;

    // For line scans where one dimension is very narrow, ensure a minimum
    // 100px (2*pad) span so the view doesn't degenerate to a pinpoint
    // (and so clamp-induced span shrinkage doesn't go negative).
    const effSpanX = Math.max(spanX, 2 * pad);
    const effSpanY = Math.max(spanY, 2 * pad);

    const scaleX = canvasW / effSpanX;
    const scaleY = canvasH / effSpanY;
    const newScale = Math.min(scaleX, scaleY);

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const newX = canvasW / 2 - cx * newScale;
    const newY = canvasH / 2 - cy * newScale;

    // Final sanity gate: if anything went non-finite or non-positive,
    // refuse to apply the broken transform — leave the current view
    // alone rather than silently fitting (which would mask "Scan zoom
    // unavailable" as "Scan = Fit").
    if (
      !isFinite(newScale) || newScale <= 0 ||
      !isFinite(newX) || !isFinite(newY)
    ) {
      return;
    }

    transform = { x: newX, y: newY, scale: newScale };
    scheduleRender();
  }
</script>

<div class="aci-viewer" bind:this={container}>
  {#if loading}
    <div class="aci-loading">
      <div class="spinner"></div>
      <span>Loading ACI image...</span>
    </div>
  {:else if authRequired}
    <div class="aci-placeholder">
      Log in to view ACI image
    </div>
  {:else if imageError || !scanId}
    <div class="aci-placeholder">
      {#if !scanId}
        No scan selected
      {:else}
        No ACI image available for this scan
      {/if}
    </div>
  {/if}

  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <canvas
    bind:this={canvas}
    class="aci-canvas"
    class:hidden={!aciImage || loading}
    on:mousedown={handleMouseDown}
    on:mousemove={handleMouseMove}
    on:mouseup={handleMouseUp}
    on:wheel={handleWheel}
    on:mouseleave={handleMouseLeave}
  ></canvas>

  {#if hoveredPoint !== null}
    <div
      class="point-tooltip"
      style="left: {tooltipX + 12}px; top: {tooltipY - 8}px"
    >
      Point {hoveredPoint}
    </div>
  {/if}

  {#if aciImage && !loading}
    <div class="aci-controls">
      <button
        class="btn-secondary btn-sm"
        on:click={toggleColorized}
        disabled={!colorizedAvailable}
        title={!colorizedAvailable
          ? 'No colorized ACI variant published for this scan'
          : colorized
            ? 'Show grayscale'
            : 'Show colorized'}
      >
        {colorized ? 'Grayscale' : 'Colorized'}
      </button>
      <button
        class="btn-secondary btn-sm"
        on:click={resetView}
        title="Fit full image in view"
      >
        Fit
      </button>
      <button
        class="btn-secondary btn-sm"
        on:click={zoomToScan}
        title="Zoom to scan point extents"
      >
        Scan
      </button>
      <button
        class="btn-secondary btn-sm"
        on:click={exportPng}
        title="Export current view as PNG"
      >
        PNG
      </button>
      <span class="hue-control">
        <input
          type="range"
          min="0"
          max="360"
          step="1"
          bind:value={pointHue}
          on:input={scheduleRender}
          title="Point color: {pointHue}°"
          class="hue-slider"
        />
      </span>
      <span class="opacity-control">
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          bind:value={pointOpacity}
          on:input={scheduleRender}
          title="Point opacity: {(pointOpacity * 100).toFixed(0)}%"
        />
      </span>
      <span class="zoom-label">{(transform.scale * 100).toFixed(0)}%</span>
    </div>
  {/if}
</div>

<style>
  .aci-viewer {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 300px;
    background: #1e1e2e;
    border-radius: var(--radius-lg);
    overflow: hidden;
  }

  .aci-canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: grab;
  }

  .aci-canvas:active {
    cursor: grabbing;
  }

  .aci-canvas.hidden {
    display: none;
  }

  .aci-loading {
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
  }

  .aci-placeholder {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #64748b;
    font-size: 0.9rem;
    text-align: center;
    padding: 24px;
  }

  .aci-controls {
    position: absolute;
    bottom: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    background: rgba(30, 41, 59, 0.85);
    padding: 4px 8px;
    border-radius: var(--radius-md);
  }

  .aci-controls button {
    font-size: 0.75rem;
    padding: 3px 8px;
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-sm);
  }

  .aci-controls button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.2);
  }

  .aci-controls button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .zoom-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: #94a3b8;
    margin-left: 4px;
    min-width: 36px;
    text-align: right;
  }

  .hue-control {
    display: flex;
    align-items: center;
  }

  .hue-slider {
    width: 50px;
    height: 3px;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
    background: linear-gradient(to right, #f00, #ff0, #0f0, #0ff, #00f, #f0f, #f00);
    border-radius: 2px;
    outline: none;
  }

  .hue-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.3);
    cursor: pointer;
  }

  .hue-slider::-moz-range-thumb {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.3);
    cursor: pointer;
  }

  .opacity-control {
    display: flex;
    align-items: center;
  }

  .opacity-control input[type='range'] {
    width: 50px;
    height: 3px;
    cursor: pointer;
    accent-color: #94a3b8;
  }

  .point-tooltip {
    position: absolute;
    pointer-events: none;
    background: rgba(30, 41, 59, 0.9);
    color: #e2e8f0;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: var(--radius-sm);
    white-space: nowrap;
    z-index: 10;
  }
</style>
