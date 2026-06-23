<script lang="ts">
  import { healthStatus } from '../lib/stores';
  import { getHealth } from '../lib/api';

  let loading = false;

  async function refresh() {
    loading = true;
    try {
      const h = await getHealth();
      healthStatus.set(h);
    } catch {
      // ignore
    } finally {
      loading = false;
    }
  }

  $: h = $healthStatus;
</script>

<div class="card">
  <div class="card-header flex items-center justify-between">
    <span>System Health</span>
    <button class="btn-secondary btn-sm" on:click={refresh} disabled={loading}>
      {#if loading}<span class="spinner"></span>{:else}Refresh{/if}
    </button>
  </div>
  <div class="card-body">
    {#if h}
      <div class="checks">
        {#each Object.entries(h.checks) as [name, check]}
          <div class="check-row">
            <span
              class="status-dot"
              class:ok={check.status === 'ok'}
              class:error={check.status === 'error'}
            ></span>
            <span class="check-name">{name.replace(/_/g, ' ')}</span>
            <span class="check-detail mono">
              {#if check.status === 'error'}
                {check.error ?? 'error'}
              {:else if name === 'database' && 'n_scans' in check}
                {check.n_scans} scans
              {:else if name === 'job_queue' && 'running' in check}
                {check.running} running, {check.queued} queued
              {:else if name === 'unprocessed_scans' && 'n_unprocessed' in check}
                {check.n_unprocessed} unprocessed
              {:else}
                ok
              {/if}
            </span>
          </div>
        {/each}
      </div>
      <div class="version mono" style="margin-top: 12px; font-size: 0.8rem; color: var(--color-text-tertiary)">
        v{h.pipeline_version}
      </div>
    {:else}
      <p class="empty-state" style="padding: 16px">No health data available</p>
    {/if}
  </div>
</div>

<style>
  .checks {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--color-text-tertiary);
  }

  .status-dot.ok {
    background: var(--color-success);
  }

  .status-dot.error {
    background: var(--color-error);
  }

  .check-name {
    font-size: 0.85rem;
    text-transform: capitalize;
    min-width: 140px;
  }

  .check-detail {
    font-size: 0.8rem;
    color: var(--color-text-secondary);
  }
</style>
