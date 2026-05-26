<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import SpectrumPlot from '../SpectrumPlot.svelte';
  import {
    AuthRequiredError,
    getMapData,
    getPointSpectrum,
    postSpectraSubset,
  } from '../../lib/api';

  export let scanId: string;
  export let mode: 'empty' | 'class_average' | 'single_point' = 'empty';
  export let classInfo: { domain: string; class_id: string; label: string } | null = null;
  export let pointIndex: number | null = null;

  const dispatch = createEventDispatcher<{
    backToAverage: void;
    openWorkbench: void;
  }>();

  let loading = false;
  let error = '';
  let wavenumber: number[] = [];
  let intensity: number[] = [];
  let nPoints = 0;
  let titleText = '';

  // Track what we last fetched to avoid redundant requests
  let lastFetchKey = '';

  $: fetchKey = mode === 'class_average'
    ? `avg:${scanId}:${classInfo?.domain}:${classInfo?.class_id}`
    : mode === 'single_point'
      ? `pt:${scanId}:${pointIndex}`
      : '';

  $: if (fetchKey && fetchKey !== lastFetchKey) {
    lastFetchKey = fetchKey;
    if (mode === 'class_average' && classInfo) {
      fetchClassAverage(classInfo.domain, classInfo.class_id, classInfo.label);
    } else if (mode === 'single_point' && pointIndex !== null) {
      fetchPointSpectrum(pointIndex);
    }
  }

  $: if (mode === 'empty') {
    wavenumber = [];
    intensity = [];
    titleText = '';
    lastFetchKey = '';
  }

  async function fetchClassAverage(domain: string, classId: string, label: string) {
    loading = true;
    error = '';
    try {
      // Step 1: Get measured point indices for this class via the authed
      // helper (issue #17 R1 F1: the raw `/api/map/data/...` fetch on this
      // path was the sibling auth-blind defect to the point-spectrum one
      // we just fixed).
      const data = await getMapData(scanId, domain, 'snr', classId);
      const measured = (data.points ?? []).filter(
        (p: { status: string }) => p.status === 'measured',
      );
      const indices = measured.map((p: { index: number }) => p.index);

      if (indices.length === 0) {
        wavenumber = [];
        intensity = [];
        nPoints = 0;
        titleText = `${label} — no detections`;
        return;
      }

      // Step 2: Fetch subset average via auth-attaching helper. Raw
      // fetch produced 401 under Auth0 Bearer-token mode (Codex PR9 R3 F5).
      const avg = await postSpectraSubset<{ wavenumber?: number[]; intensity?: number[] }>(
        scanId,
        {
          point_indices: indices,
          region: 'R1',
          averaging_method: 'trim_mean',
        },
      );

      wavenumber = avg.wavenumber ?? [];
      intensity = avg.intensity ?? [];
      nPoints = indices.length;
      titleText = `${label} average (${nPoints} pts)`;
    } catch (e) {
      if (e instanceof AuthRequiredError) {
        error = 'Log in to view spectrum';
      } else {
        error = e instanceof Error ? e.message : 'Failed to load spectrum';
      }
      wavenumber = [];
      intensity = [];
    } finally {
      loading = false;
    }
  }

  async function fetchPointSpectrum(idx: number) {
    loading = true;
    error = '';
    try {
      const data = await getPointSpectrum(scanId, idx, { region: 'R1' });

      wavenumber = data.wavenumber ?? [];
      intensity = data.intensity ?? [];
      nPoints = 1;
      titleText = `Point ${idx}`;
    } catch (e) {
      if (e instanceof AuthRequiredError) {
        error = 'Log in to view spectrum';
      } else {
        error = e instanceof Error ? e.message : 'Failed to load spectrum';
      }
      wavenumber = [];
      intensity = [];
    } finally {
      loading = false;
    }
  }
</script>

<div class="spectrum-panel">
  <div class="panel-header">
    <div class="title-area">
      {#if mode === 'single_point' && classInfo}
        <button class="btn-link" on:click={() => dispatch('backToAverage')}>
          &larr; Back to average
        </button>
        <span class="sep">|</span>
      {/if}
      <span class="title-text">{titleText || 'Spectrum'}</span>
    </div>
    <div class="actions">
      {#if mode === 'single_point' && pointIndex !== null}
        <button class="btn-link" on:click={() => dispatch('openWorkbench')}>
          Open in Workbench &rarr;
        </button>
      {/if}
    </div>
  </div>

  <div class="panel-body">
    {#if mode === 'empty'}
      <div class="placeholder">Select a class or click a point</div>
    {:else if loading}
      <div class="placeholder">
        <div class="spinner-sm"></div>
        Loading...
      </div>
    {:else if error}
      <div class="placeholder error">{error}</div>
    {:else if wavenumber.length > 0}
      <SpectrumPlot
        {wavenumber}
        {intensity}
        compact={true}
        height={200}
      />
    {:else}
      <div class="placeholder">No spectrum data</div>
    {/if}
  </div>
</div>

<style>
  .spectrum-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 8px;
    border-bottom: 1px solid var(--color-border);
    background: var(--color-background);
    flex-shrink: 0;
  }

  .title-area {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    min-width: 0;
  }

  .title-text {
    color: var(--color-text);
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .sep {
    color: var(--color-text-tertiary);
  }

  .actions {
    flex-shrink: 0;
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--color-primary);
    padding: 0;
    cursor: pointer;
    font-size: 0.72rem;
    white-space: nowrap;
  }

  .btn-link:hover {
    text-decoration: underline;
  }

  .panel-body {
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  .placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--color-text-tertiary);
    font-size: 0.8rem;
    gap: 8px;
  }

  .placeholder.error {
    color: var(--color-error, #dc2626);
  }

  .spinner-sm {
    width: 14px;
    height: 14px;
    border: 2px solid var(--color-border);
    border-top-color: var(--color-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
