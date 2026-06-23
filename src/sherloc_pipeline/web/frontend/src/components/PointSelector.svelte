<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { ScanPoint } from '../lib/types';

  export let points: ScanPoint[] = [];
  export let selectedIndices: number[] = [];

  const dispatch = createEventDispatcher<{
    select: { indices: number[] };
  }>();

  let selectAll = false;

  function togglePoint(idx: number) {
    if (selectedIndices.includes(idx)) {
      selectedIndices = selectedIndices.filter((i) => i !== idx);
    } else {
      selectedIndices = [...selectedIndices, idx];
    }
    selectAll = selectedIndices.length === points.length;
    dispatch('select', { indices: selectedIndices });
  }

  function toggleAll() {
    if (selectAll) {
      selectedIndices = [];
    } else {
      selectedIndices = points.map((p) => p.point_index);
    }
    selectAll = !selectAll;
    dispatch('select', { indices: selectedIndices });
  }

  function selectRange(start: number, end: number) {
    const lo = Math.min(start, end);
    const hi = Math.max(start, end);
    selectedIndices = [];
    for (let i = lo; i <= hi; i++) {
      selectedIndices.push(i);
    }
    selectAll = selectedIndices.length === points.length;
    dispatch('select', { indices: selectedIndices });
  }

  let rangeStart = 0;
  let rangeEnd = 0;

  // Funnel range-end reset through a helper so the `$:` block's
  // invalidation guard does not turn `rangeEnd` into a tracked dep
  // of itself. Same pattern as ProcessingChain.svelte:69-90; see
  // docs/FRONTEND_HAZARDS.md for the full hazard explanation.
  function resetRangeEnd(p: ScanPoint[]): void {
    if (p.length > 0) {
      rangeEnd = p.length - 1;
    }
  }
  $: resetRangeEnd(points);

  let goToPoint = '';

  function handleGoToPoint() {
    const idx = parseInt(goToPoint, 10);
    if (isNaN(idx)) return;
    const exists = points.some((p) => p.point_index === idx);
    if (!exists) return;
    selectedIndices = [idx];
    selectAll = false;
    dispatch('select', { indices: [idx] });
  }

  function handleGoToKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleGoToPoint();
  }
</script>

<div class="point-selector">
  <div class="go-to-point">
    <label>
      Point:
      <input
        type="number"
        bind:value={goToPoint}
        on:keydown={handleGoToKeydown}
        placeholder="e.g. 91"
        min="0"
        max={points.length - 1}
        style="width: 72px"
      />
      <button class="btn-primary btn-sm" on:click={handleGoToPoint}>Go</button>
    </label>
  </div>

  <div class="selector-header">
    <label>
      <input type="checkbox" checked={selectAll} on:change={toggleAll} />
      Select All ({points.length} points)
    </label>
    <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
      {selectedIndices.length} selected
    </span>
  </div>

  <div class="range-selector">
    <label>Range:
      <input type="number" bind:value={rangeStart} min="0" max={points.length - 1} style="width: 60px" />
      <span>to</span>
      <input type="number" bind:value={rangeEnd} min="0" max={points.length - 1} style="width: 60px" />
      <button class="btn-secondary btn-sm" on:click={() => selectRange(rangeStart, rangeEnd)}>
        Apply
      </button>
    </label>
  </div>

  <div class="point-grid">
    {#each points as point}
      <button
        class="point-chip"
        class:selected={selectedIndices.includes(point.point_index)}
        on:click={() => togglePoint(point.point_index)}
        title="Point {point.point_index}"
      >
        {point.point_index}
      </button>
    {/each}
  </div>
</div>

<style>
  .point-selector {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .selector-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .selector-header label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    margin-bottom: 0;
  }

  .range-selector {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.85rem;
  }

  .range-selector label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 0;
  }

  .point-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    max-height: 180px;
    overflow-y: auto;
  }

  .point-chip {
    width: 32px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text-secondary);
    cursor: pointer;
    padding: 0;
  }

  .point-chip:hover {
    border-color: var(--color-primary);
  }

  .point-chip.selected {
    background: var(--color-primary);
    color: white;
    border-color: var(--color-primary);
  }

  .go-to-point {
    margin-bottom: 4px;
  }

  .go-to-point label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 0;
  }
</style>
