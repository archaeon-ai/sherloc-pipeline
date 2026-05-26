<script lang="ts">
  import { onMount } from 'svelte';
  import { navigate, baselineParams } from '../lib/stores';
  import { getScan, getAverageSpectrum, postBaseline } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { ScanDetail, SpectrumResponse, BaselineResponse } from '../lib/types';
  import SpectrumPlot from './SpectrumPlot.svelte';

  export let scanId: string;

  let scan: ScanDetail | null = null;
  let spectrum: SpectrumResponse | null = null;
  let baselineResult: BaselineResponse | null = null;
  let loading = true;
  let computing = false;
  let error = '';

  // Parameters (initialized from store)
  let lam = $baselineParams.lam ?? 1e6;
  let maxIter = $baselineParams.max_iter ?? 10;
  let logLam = Math.log10(lam);

  // Debounce timer
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  onMount(async () => {
    loading = true;
    try {
      const res = await getScan(scanId);
      scan = res.scan;
      spectrum = await getAverageSpectrum(scanId, { region: 'R1' });
      await computeBaseline();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  function onLogLamChange() {
    lam = Math.pow(10, logLam);
    debouncedCompute();
  }

  function onMaxIterChange() {
    debouncedCompute();
  }

  function debouncedCompute() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => computeBaseline(), 300);
  }

  async function computeBaseline() {
    if (!spectrum) return;
    computing = true;
    try {
      baselineResult = await postBaseline({
        wavenumber: spectrum.wavenumber,
        intensity: spectrum.intensity,
        params: {
          method: 'aspls',
          lam,
          max_iter: maxIter,
        },
      });
    } catch (e) {
      // Silently handle preview errors
      console.error('Baseline computation error:', e);
    } finally {
      computing = false;
    }
  }

  function useForFitting() {
    baselineParams.set({
      method: 'aspls',
      lam,
      max_iter: maxIter,
    });
    navigate(`#/scan/${scanId}/fit`);
  }
</script>

<div class="page-container">
  <div class="breadcrumb">
    <button class="btn-link" on:click={() => navigate('#/')}>Scans</button>
    <span class="breadcrumb-sep">/</span>
    <button class="btn-link" on:click={() => navigate(`#/scan/${scanId}`)}>
      {scan?.target ?? '...'} Sol {scan?.sol_number ?? ''}
    </button>
    <span class="breadcrumb-sep">/</span>
    <span>Baseline</span>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="empty-state"><span class="spinner"></span> Loading...</div>
  {:else}
    <div class="workspace-layout">
      <div class="workspace-main">
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <span>Baseline Preview</span>
            {#if computing}
              <span class="spinner"></span>
            {/if}
          </div>
          <div class="card-body">
            {#if spectrum}
              <SpectrumPlot
                wavenumber={spectrum.wavenumber}
                intensity={spectrum.intensity}
                baseline={baselineResult?.baseline ?? null}
                showBaseline={true}
                showFit={false}
                title="{scan?.target ?? ''} Sol {scan?.sol_number ?? ''} {scan?.scan_name ?? ''} — Baseline"
                height={480}
              />
            {/if}

            {#if baselineResult}
              <div style="margin-top: 16px">
                <h3 style="font-size: 0.95rem; margin-bottom: 8px">Baseline-Corrected</h3>
                <SpectrumPlot
                  wavenumber={baselineResult.wavenumber}
                  intensity={baselineResult.corrected}
                  title="Corrected Spectrum"
                  height={300}
                />
              </div>
            {/if}
          </div>
        </div>
      </div>

      <div class="workspace-sidebar">
        <div class="card">
          <div class="card-header">Baseline Parameters</div>
          <div class="card-body param-controls">
            <div class="param-group">
              <label for="b-lam">
                Smoothness (&lambda;): {lam.toExponential(1)}
              </label>
              <input
                id="b-lam"
                type="range"
                min="4"
                max="8"
                step="0.1"
                bind:value={logLam}
                on:input={onLogLamChange}
              />
              <div class="range-labels">
                <span>10<sup>4</sup></span>
                <span>10<sup>8</sup></span>
              </div>
            </div>

            <div class="param-group">
              <label for="b-iter">Max Iterations: {maxIter}</label>
              <input
                id="b-iter"
                type="range"
                min="1"
                max="50"
                bind:value={maxIter}
                on:input={onMaxIterChange}
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

            {#if baselineResult}
              <div class="param-group" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                <strong>Params used:</strong>
                <div class="mono">
                  &lambda;={baselineResult.params_used.lam?.toExponential(1)},
                  iter={baselineResult.params_used.max_iter}
                </div>
              </div>
            {/if}

            <hr style="border: none; border-top: 1px solid var(--color-border); margin: 8px 0" />

            <button class="btn-primary" style="width: 100%" on:click={useForFitting}>
              Use for Fitting
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 16px;
    font-size: 0.9rem;
    color: var(--color-text-secondary);
  }

  .breadcrumb-sep { color: var(--color-text-tertiary); }

  .btn-link {
    background: none;
    border: none;
    color: var(--color-primary);
    padding: 0;
    cursor: pointer;
    font-size: inherit;
  }
  .btn-link:hover { text-decoration: underline; }

  .workspace-layout {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 16px;
    align-items: start;
  }

  @media (max-width: 1024px) {
    .workspace-layout { grid-template-columns: 1fr; }
  }

  .workspace-main { min-width: 0; }

  .param-controls {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .param-group select,
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
