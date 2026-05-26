<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { postBaseline } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { BaselineParams } from '../lib/types';
  import InfoTooltip from './InfoTooltip.svelte';

  const baselineRef = `<div class="ref-title">Adaptive smoothness penalized least squares (asPLS)</div>
Iteratively reweighted penalized least squares baseline fitting.
Protected mineral windows (sulfate, carbonate, OH-stretch) reduce penalty
to avoid fitting through genuine Raman peaks.
<div class="ref-cite">Zhang, F. et al. (2020). Baseline correction for infrared spectra using adaptive
smoothness parameter penalized least squares method. <i>Spectroscopy Letters</i>, 53(3), 222–233.
<a href="https://doi.org/10.1080/00387010.2020.1730908" target="_blank">doi:10.1080/00387010.2020.1730908</a>
<br>Implementation: pybaselines (Erb, 2022).
<a href="https://doi.org/10.5281/zenodo.5608581" target="_blank">doi:10.5281/zenodo.5608581</a></div>`;

  export let wavenumber: number[] = [];
  export let intensity: number[] = [];
  export let enabled: boolean = true;
  export let collapsed: boolean = true;
  // Generation counter from ProcessingChain; bumped on every raw-input
  // change. Capture at request start, re-check before dispatch — if the
  // raw input changed mid-flight, drop the stale response instead of
  // re-applying baseline state to the now-reset spectrum.
  export let inputGeneration: number = 0;

  const dispatch = createEventDispatcher<{
    apply: { corrected: number[]; baseline: number[]; params: BaselineParams };
    toggle: { enabled: boolean };
  }>();

  // Parameters
  let lam = 1e6;
  let logLam = Math.log10(lam);
  let maxIter = 10;

  let computing = false;
  let error = '';

  // Debounce timer
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function debouncedApply() {
    if (!enabled) return;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => applyBaseline(), 300);
  }

  function onLogLamChange() {
    lam = Math.pow(10, logLam);
    debouncedApply();
  }

  function onMaxIterChange() {
    debouncedApply();
  }

  function toggleEnabled() {
    enabled = !enabled;
    dispatch('toggle', { enabled });
    if (enabled && wavenumber.length > 0) {
      applyBaseline();
    }
  }

  function toggleCollapsed() {
    collapsed = !collapsed;
  }

  async function applyBaseline() {
    if (!enabled || wavenumber.length === 0) return;
    const gen = inputGeneration;
    computing = true;
    error = '';
    try {
      const result = await postBaseline({
        wavenumber,
        intensity,
        params: {
          method: 'aspls',
          lam,
          max_iter: maxIter,
        },
      });
      if (gen !== inputGeneration) return;
      dispatch('apply', {
        corrected: result.corrected,
        baseline: result.baseline,
        params: result.params_used,
      });
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Baseline correction failed';
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
      <span class="step-title">3. Baseline</span>
      {#if computing}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="step-header-right">
      {#if enabled}
        <span class="step-badge mono">&lambda;={lam.toExponential(0)}</span>
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
        <label for="bl-lam">
          Smoothness (&lambda;): {lam.toExponential(1)}
        </label>
        <input
          id="bl-lam"
          type="range"
          min="4"
          max="8"
          step="0.1"
          bind:value={logLam}
          on:input={onLogLamChange}
          disabled={!enabled}
        />
        <div class="range-labels">
          <span>10<sup>4</sup></span>
          <span>10<sup>8</sup></span>
        </div>
      </div>

      <div class="param-group">
        <label for="bl-iter">Max Iterations: {maxIter}</label>
        <input
          id="bl-iter"
          type="range"
          min="1"
          max="50"
          bind:value={maxIter}
          on:input={onMaxIterChange}
          disabled={!enabled}
        />
        <div class="range-labels">
          <span>1</span>
          <span>50</span>
        </div>
      </div>

      <div class="param-group">
        <label>Method</label>
        <div class="mono" style="font-size: 0.85rem; color: var(--color-text-secondary)">asPLS</div>
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
</style>
