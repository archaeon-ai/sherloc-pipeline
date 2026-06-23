<script lang="ts">
  import { mapGeometryMode } from '../../lib/stores/mapStore';
  import type { GeometryMode } from '../../lib/types/map';

  const modes: { value: GeometryMode; label: string }[] = [
    { value: 'voronoi', label: 'Voronoi' },
    { value: 'ring', label: 'Ring' },
    { value: 'combined', label: 'Combined' },
  ];

  function setMode(mode: GeometryMode) {
    mapGeometryMode.set(mode);
  }
</script>

<div class="geometry-toggle">
  {#each modes as mode}
    <button
      class="toggle-btn"
      class:active={$mapGeometryMode === mode.value}
      on:click={() => setMode(mode.value)}
    >
      {mode.label}
    </button>
  {/each}
</div>

<style>
  .geometry-toggle {
    display: flex;
    gap: 2px;
    background: var(--color-background);
    border-radius: var(--radius-md);
    padding: 2px;
  }

  .toggle-btn {
    flex: 1;
    padding: 5px 8px;
    font-size: 0.78rem;
    font-weight: 500;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--color-text-secondary);
    border: none;
    cursor: pointer;
    transition: all 0.15s;
  }

  .toggle-btn:hover {
    color: var(--color-text);
    background: rgba(255, 255, 255, 0.05);
  }

  .toggle-btn.active {
    background: var(--color-surface);
    color: var(--color-text);
    box-shadow: var(--shadow-sm);
  }
</style>
