<script lang="ts">
  import { mapDisplayMode, mapOverlayOpacity, mapShowPointPositions } from '../../lib/stores/mapStore';
  import DisplayModeSelector from './DisplayModeSelector.svelte';
  // GeometryToggle hidden for E1 — Ring is the only mode
  // import GeometryToggle from './GeometryToggle.svelte';
  import RGBMixPanel from './RGBMixPanel.svelte';

  let showRgbPanel = false;

  function handleRgbMix() {
    showRgbPanel = true;
  }

  // Close RGB panel when switching away from rgb_mix mode
  $: if ($mapDisplayMode.type !== 'rgb_mix') {
    showRgbPanel = false;
  }
</script>

<div class="map-controls-panel">
  <DisplayModeSelector on:rgbMix={handleRgbMix} />

  {#if showRgbPanel}
    <RGBMixPanel />
  {/if}

  <!-- Geometry toggle hidden for E1 — Ring mode only -->
  <!--
  <div class="control-group">
    <div class="section-label">Geometry</div>
    <GeometryToggle />
  </div>
  -->

  <div class="control-group">
    <label class="toggle-label">
      <input type="checkbox" bind:checked={$mapShowPointPositions} />
      <span>Show Point Positions</span>
    </label>
  </div>

  <div class="control-group">
    <label class="opacity-label">
      <span class="section-label">Overlay Opacity: {$mapOverlayOpacity.toFixed(2)}</span>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        bind:value={$mapOverlayOpacity}
      />
    </label>
  </div>
</div>

<style>
  .map-controls-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    cursor: pointer;
    margin-bottom: 0;
  }

  .toggle-label input[type='checkbox'] {
    width: 16px;
    height: 16px;
    accent-color: var(--color-primary);
  }

  .opacity-label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 0;
  }

  .opacity-label input[type='range'] {
    width: 100%;
  }
</style>
