<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { postDespike } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { DespikeParams } from '../lib/types';
  import InfoTooltip from './InfoTooltip.svelte';

  const despikeRef = `<div class="ref-title">Rolling-median MAD z-score despiking</div>
Robust cosmic ray removal using rolling median filter with MAD-derived threshold.
Protected regions guard sulfate peaks and laser line from false flagging.
<div class="ref-cite">Whitaker, D.A. &amp; Hayes, K. (2018). A simple algorithm for despiking Raman spectra.
<i>Chemometrics and Intelligent Laboratory Systems</i>, 179, 82–84.
<a href="https://doi.org/10.1016/j.chemolab.2018.06.009" target="_blank">doi:10.1016/j.chemolab.2018.06.009</a></div>`;

  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  export let enabled: boolean = false;
  export let collapsed: boolean = true;
  export let isAverageMode: boolean = true;

  const dispatch = createEventDispatcher<{
    apply: { despiked: number[]; spikeMask: boolean[]; nSpikes: number; params: DespikeParams };
    toggle: { enabled: boolean };
  }>();

  // Parameters
  let windowSize = 7;
  let zThreshold = 6.0;
  let maxIterations = 1;
  let sulfateGuard = true;

  let computing = false;
  let error = '';
  let lastNSpikes: number | null = null;

  // Default: off for averages, on for single points
  $: if (isAverageMode && enabled) {
    // Keep current state, don't auto-toggle
  }

  // Debounce timer
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function debouncedApply() {
    if (!enabled) return;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => applyDespike(), 300);
  }

  function toggleEnabled() {
    enabled = !enabled;
    dispatch('toggle', { enabled });
    if (enabled && wavenumber.length > 0) {
      applyDespike();
    }
  }

  function toggleCollapsed() {
    collapsed = !collapsed;
  }

  // Enforce odd-only for window size
  function onWindowSizeChange() {
    if (windowSize % 2 === 0) {
      windowSize = windowSize + 1;
    }
    debouncedApply();
  }

  async function applyDespike() {
    if (!enabled || wavenumber.length === 0) return;
    computing = true;
    error = '';
    try {
      const result = await postDespike({
        wavenumber,
        intensity,
        params: {
          window_size: windowSize,
          zscore_threshold: zThreshold,
          max_iterations: maxIterations,
          sulfate_guard: sulfateGuard,
        },
      });
      lastNSpikes = result.n_spikes;
      dispatch('apply', {
        despiked: result.despiked,
        spikeMask: result.spike_mask,
        nSpikes: result.n_spikes,
        params: result.params_used,
      });
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Despiking failed';
      }
    } finally {
      computing = false;
    }
  }
</script>

<div class="step-card" class:step-enabled={enabled}>
  <button class="step-header" on:click={toggleCollapsed}>
    <div class="step-header-left">
      <label class="step-toggle" on:click|stopPropagation>
        <input type="checkbox" checked={enabled} on:change={toggleEnabled} />
      </label>
      <span class="step-title">1. Despike</span>
      {#if computing}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="step-header-right">
      {#if lastNSpikes !== null && enabled}
        <span class="step-badge mono">{lastNSpikes} spike{lastNSpikes !== 1 ? 's' : ''}</span>
      {/if}
      <span class="collapse-icon">{collapsed ? '+' : '-'}</span>
    </div>
  </button>

  {#if !collapsed}
    <div class="step-body">
      {#if error}
        <div class="step-error">{error}</div>
      {/if}

      <div class="param-group">
        <label for="ds-window">Window Size: {windowSize}</label>
        <input
          id="ds-window"
          type="range"
          min="3"
          max="15"
          step="2"
          bind:value={windowSize}
          on:input={onWindowSizeChange}
          disabled={!enabled}
        />
        <div class="range-labels">
          <span>3</span>
          <span>15</span>
        </div>
      </div>

      <div class="param-group">
        <label for="ds-zthresh">Z-Score Threshold: {zThreshold.toFixed(1)}</label>
        <input
          id="ds-zthresh"
          type="range"
          min="2.0"
          max="10.0"
          step="0.5"
          bind:value={zThreshold}
          on:input={debouncedApply}
          disabled={!enabled}
        />
        <div class="range-labels">
          <span>2.0</span>
          <span>10.0</span>
        </div>
      </div>

      <div class="param-group">
        <label for="ds-maxiter">Max Iterations: {maxIterations}</label>
        <input
          id="ds-maxiter"
          type="range"
          min="1"
          max="5"
          step="1"
          bind:value={maxIterations}
          on:input={debouncedApply}
          disabled={!enabled}
        />
        <div class="range-labels">
          <span>1</span>
          <span>5</span>
        </div>
      </div>

      <div class="param-group">
        <label class="checkbox-label">
          <input
            type="checkbox"
            bind:checked={sulfateGuard}
            on:change={debouncedApply}
            disabled={!enabled}
          />
          Sulfate Guard
        </label>
      </div>
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

  .step-toggle {
    display: inline-flex;
    align-items: center;
    margin-bottom: 0;
    cursor: pointer;
  }

  .step-toggle input[type="checkbox"] {
    cursor: pointer;
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

  .checkbox-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    margin-bottom: 0;
  }
</style>
