<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { postBackground } from '../lib/api';
  import { ApiError } from '../lib/api';
  import InfoTooltip from './InfoTooltip.svelte';

  const bgRef = `<div class="ref-title">Dark frame background subtraction</div>
Subtracts instrument background using pre-measured reference spectra,
scaled by laser line intensity ratio in the 600–700 cm⁻¹ window.
<b>Arm-Stowed:</b> Post-anomaly 900 PPP averaged dark spectrum.
<b>Fused Silica:</b> Corning 7980 air-subtracted reference.
<div class="ref-cite">Bhartia, R. et al. (2021). Perseverance's Scanning Habitable Environments with
Raman and Luminescence for Organics and Chemicals (SHERLOC) Investigation.
<i>Space Sci. Rev.</i>, 217, 58.
<a href="https://doi.org/10.1007/s11214-021-00812-z" target="_blank">doi:10.1007/s11214-021-00812-z</a></div>`;

  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  export let scanPpp: number = 1;
  export let collapsed: boolean = true;
  export let isSinglePoint: boolean = false;
  // Generation counter from ProcessingChain; bumped on every raw-input
  // change. Capture at request start, re-check before dispatch — drops
  // stale responses started before a point/modality switch.
  export let inputGeneration: number = 0;

  const dispatch = createEventDispatcher<{
    apply: { subtracted: number[]; backgroundScaled: number[]; scaleUsed: number; bgType: string };
    toggle: { enabled: boolean };
  }>();

  // Parameters
  // bgType is bound from ProcessingChain so the parent can reset it to 'none'
  // when the raw spectrum input changes (point switch, modality change) —
  // keeps the radio selection in sync with what's actually applied to the
  // displayed spectrum.
  export let bgType: 'none' | 'as' | 'fs' = 'none';
  let autoScale = true;
  let manualScale = 1.0;

  let computing = false;
  let error = '';
  let lastScaleUsed: number | null = null;

  $: enabled = bgType !== 'none';

  // Debounce timer
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function debouncedApply() {
    if (!enabled) return;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => applyBackground(), 300);
  }

  function onBgTypeChange() {
    if (bgType === 'none') {
      dispatch('toggle', { enabled: false });
      lastScaleUsed = null;
    } else {
      dispatch('toggle', { enabled: true });
      if (wavenumber.length > 0) {
        applyBackground();
      }
    }
  }

  function toggleCollapsed() {
    collapsed = !collapsed;
  }

  async function applyBackground() {
    if (bgType === 'none' || wavenumber.length === 0) return;
    const gen = inputGeneration;
    computing = true;
    error = '';
    try {
      const result = await postBackground({
        wavenumber,
        intensity,
        bg_type: bgType as 'as' | 'fs',
        scale: autoScale ? 'auto' : manualScale,
        scan_ppp: scanPpp,
      });
      if (gen !== inputGeneration) return;
      lastScaleUsed = result.scale_used;
      dispatch('apply', {
        subtracted: result.subtracted,
        backgroundScaled: result.background_scaled,
        scaleUsed: result.scale_used,
        bgType: result.bg_type,
      });
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Background subtraction failed';
      }
    } finally {
      computing = false;
    }
  }
</script>

<div class="step-card" class:step-enabled={enabled}>
  <button class="step-header" on:click={toggleCollapsed}>
    <div class="step-header-left">
      <span class="step-indicator" class:active={enabled}></span>
      <span class="step-title">2. Background</span>
      {#if computing}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="step-header-right">
      {#if lastScaleUsed !== null && enabled}
        <span class="step-badge mono">scale: {lastScaleUsed.toFixed(3)}</span>
      {/if}
      <span class="collapse-icon">{collapsed ? '+' : '-'}</span>
    </div>
  </button>

  {#if !collapsed}
    <div class="step-body">
      {#if error}
        <div class="step-error">{error}</div>
      {/if}

      {#if isSinglePoint}
        <div class="step-warning">
          Background subtraction on single points may introduce artifacts. Consider using averaged spectra.
        </div>
      {/if}

      <div class="param-group">
        <label>Background Type</label>
        <div class="radio-group">
          <label class="radio-label">
            <input type="radio" bind:group={bgType} value="none" on:change={onBgTypeChange} />
            None
          </label>
          <label class="radio-label">
            <input type="radio" bind:group={bgType} value="as" on:change={onBgTypeChange} />
            Arm-Stowed
          </label>
          <label class="radio-label">
            <input type="radio" bind:group={bgType} value="fs" on:change={onBgTypeChange} />
            Fused Silica
          </label>
        </div>
      </div>

      {#if enabled}
        <div class="param-group">
          <label class="checkbox-label">
            <input type="checkbox" bind:checked={autoScale} on:change={debouncedApply} />
            Auto Scale
            {#if autoScale && lastScaleUsed !== null}
              <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                (PPP: {lastScaleUsed.toFixed(3)})
              </span>
            {/if}
          </label>
        </div>

        {#if !autoScale}
          <div class="param-group">
            <label for="bg-scale">Manual Scale: {manualScale.toFixed(2)}</label>
            <input
              id="bg-scale"
              type="range"
              min="0.1"
              max="5.0"
              step="0.05"
              bind:value={manualScale}
              on:input={debouncedApply}
            />
            <div class="range-labels">
              <span>0.1</span>
              <span>5.0</span>
            </div>
          </div>
        {/if}
      {/if}
    </div>
  {/if}
</div>

<style>
  .step-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface);
    overflow: hidden;
  }

  .step-card.step-enabled {
    border-color: var(--color-primary);
  }

  .step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    padding: 10px 12px;
    background: var(--color-background);
    border: none;
    border-radius: 0;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .step-header:hover {
    background: var(--color-primary-light);
  }

  .step-header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--color-border);
    border: 1px solid var(--color-border-strong);
  }

  .step-indicator.active {
    background: var(--color-primary);
    border-color: var(--color-primary);
  }

  .step-title {
    color: var(--color-text);
  }

  .step-badge {
    font-size: 0.75rem;
    padding: 1px 6px;
    background: var(--color-info-light);
    color: var(--color-info);
    border-radius: 9999px;
  }

  .collapse-icon {
    font-size: 1rem;
    color: var(--color-text-tertiary);
    width: 20px;
    text-align: center;
    font-family: var(--font-mono);
  }

  .step-body {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-top: 1px solid var(--color-border);
  }

  .step-error {
    background: var(--color-error-light);
    color: var(--color-error);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
  }

  .step-warning {
    background: var(--color-warning-light);
    color: var(--color-warning);
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
  }

  .radio-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .radio-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    margin-bottom: 0;
  }

  .checkbox-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    margin-bottom: 0;
  }

  .param-group input[type="range"] {
    width: 100%;
  }

  .range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--color-text-tertiary);
    font-family: var(--font-mono);
  }
</style>
