<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { mapFitJob } from '../../lib/stores/mapStore';

  export let scanId: string;

  let domains: Record<string, boolean> = {
    minerals: true,
    organics: false,
    hydration: false,
    fluorescence: false,
  };

  const dispatch = createEventDispatcher<{
    startFit: { domains: string[] };
    cancelFit: void;
  }>();

  $: isRunning = ($mapFitJob?.status === 'running' || $mapFitJob?.status === 'queued')
    && !($mapFitJob?.fitted === $mapFitJob?.total && $mapFitJob?.total > 0);
  $: isComplete = $mapFitJob?.status === 'complete';
  $: isFailed = $mapFitJob?.status === 'failed';
  $: selectedDomains = Object.entries(domains).filter(([, v]) => v).map(([k]) => k);
  $: canStart = selectedDomains.length > 0 && !isRunning;
  $: progress = $mapFitJob;

  function handleStart() {
    if (!canStart) return;
    dispatch('startFit', { domains: selectedDomains });
  }

  function handleCancel() {
    dispatch('cancelFit');
  }

  // Suppress unused export warning
  void scanId;
</script>

<div class="fit-controls">
  <div class="section-header">Fitting Domains</div>

  <div class="domain-checkboxes">
    <label class="domain-label">
      <input type="checkbox" bind:checked={domains.minerals} disabled={isRunning} />
      <span>Minerals</span>
    </label>
    <label class="domain-label">
      <input type="checkbox" bind:checked={domains.organics} disabled={isRunning} />
      <span>Organics</span>
    </label>
    <label class="domain-label">
      <input type="checkbox" bind:checked={domains.hydration} disabled={isRunning} />
      <span>Hydration</span>
    </label>
    <label class="domain-label">
      <input type="checkbox" bind:checked={domains.fluorescence} disabled={isRunning} />
      <span>Fluorescence</span>
    </label>
  </div>

  {#if selectedDomains.length === 0 && !isRunning}
    <div class="hint">Select at least one domain to fit</div>
  {/if}

  <div class="fit-actions">
    {#if isRunning}
      <button class="btn-cancel" on:click={handleCancel}>
        Cancel Fitting
      </button>
    {:else}
      <button class="btn-primary fit-btn" on:click={handleStart} disabled={!canStart}>
        Start Fitting
      </button>
    {/if}
  </div>

  {#if progress && (isRunning || isComplete || isFailed)}
    <div class="progress-summary">
      <span class="mono">
        {progress.fitted} / {progress.total} points
      </span>
      {#if isComplete}
        <span class="status-complete">Complete</span>
      {:else if isFailed}
        <span class="status-failed">Failed</span>
      {/if}
    </div>
  {/if}
</div>

<style>
  .fit-controls {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .section-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .domain-checkboxes {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .domain-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    cursor: pointer;
    margin-bottom: 0;
  }

  .domain-label input[type='checkbox'] {
    width: 16px;
    height: 16px;
    accent-color: var(--color-primary);
  }

  .domain-label input:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .hint {
    font-size: 0.78rem;
    color: var(--color-text-tertiary);
    font-style: italic;
  }

  .fit-actions {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 4px;
  }

  .fit-btn {
    width: 100%;
    padding: 8px 12px;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .btn-cancel {
    width: 100%;
    padding: 8px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    background: var(--color-error-light, #fef2f2);
    color: var(--color-error, #dc2626);
    border: 1px solid var(--color-error, #dc2626);
    border-radius: var(--radius-md);
    cursor: pointer;
  }

  .btn-cancel:hover {
    background: var(--color-error, #dc2626);
    color: white;
  }

  .progress-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    padding: 6px 0;
  }

  .status-complete {
    color: var(--color-success, #16a34a);
    font-weight: 600;
    font-size: 0.78rem;
  }

  .status-failed {
    color: var(--color-error, #dc2626);
    font-weight: 600;
    font-size: 0.78rem;
  }
</style>
