<script lang="ts">
  import { mapFitJob, mapLogEntries } from '../../lib/stores/mapStore';

  let expanded = false;
  const MAX_VISIBLE = 4;

  $: progress = $mapFitJob;
  $: logs = $mapLogEntries;
  $: visibleLogs = expanded ? logs : logs.slice(-MAX_VISIBLE);
  $: pct = progress ? Math.round((progress.fitted / Math.max(progress.total, 1)) * 100) : 0;
  $: etaText = progress && progress.etaSeconds > 0
    ? formatEta(progress.etaSeconds)
    : '';
  $: isActive = progress && (progress.status === 'running' || progress.status === 'queued');
  $: isComplete = progress?.status === 'complete';
  $: isFailed = progress?.status === 'failed';

  function formatEta(seconds: number): string {
    if (seconds < 60) return `~${Math.round(seconds)}s remaining`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `~${m}m ${s}s remaining`;
  }

  function toggleExpanded() {
    expanded = !expanded;
  }
</script>

{#if progress}
  <div class="progress-panel" class:expanded>
    <!-- Log area -->
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="log-area" on:click={toggleExpanded}>
      {#if visibleLogs.length === 0}
        <div class="log-line dimmed">
          {#if isActive}Waiting for results...{:else}No log entries{/if}
        </div>
      {:else}
        {#each visibleLogs as line}
          <div class="log-line">{line}</div>
        {/each}
      {/if}
      {#if logs.length > MAX_VISIBLE && !expanded}
        <div class="log-expand-hint">
          Click to show all {logs.length} entries
        </div>
      {/if}
    </div>

    <!-- Status / progress bar -->
    {#if isComplete}
      <div class="completion-banner">
        Fitting complete — {progress.fitted} points fitted. Select a class from Display Mode to view results.
      </div>
    {:else if isFailed}
      <div class="failure-banner">
        Fitting failed. Check log for details.
      </div>
    {:else}
      <div class="progress-bar-container">
        <div
          class="progress-bar-fill"
          style="width: {pct}%"
        >
          <span class="progress-pct">{pct}%</span>
        </div>
        {#if etaText && isActive}
          <span class="progress-eta">{etaText}</span>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .progress-panel {
    display: flex;
    flex-direction: column;
    background: #1a1a2e;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .log-area {
    padding: 8px 12px;
    max-height: 80px;
    overflow-y: hidden;
    cursor: pointer;
    transition: max-height 0.3s ease;
  }

  .progress-panel.expanded .log-area {
    max-height: 150px;
    overflow-y: auto;
  }

  .log-line {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 0.75rem;
    line-height: 1.5;
    color: #4ade80;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .log-line.dimmed {
    color: #64748b;
    font-style: italic;
  }

  .log-expand-hint {
    font-size: 0.7rem;
    color: #64748b;
    text-align: center;
    padding-top: 2px;
  }

  .progress-bar-container {
    position: relative;
    height: 24px;
    background: rgba(255, 255, 255, 0.06);
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }

  .progress-bar-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    transition: width 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 40px;
  }

  .progress-bar-fill.complete {
    background: linear-gradient(90deg, #16a34a, #22c55e);
  }

  .progress-bar-fill.failed {
    background: linear-gradient(90deg, #dc2626, #ef4444);
  }

  .progress-pct {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }

  .completion-banner {
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #4ade80;
    background: rgba(74, 222, 128, 0.08);
    border-top: 1px solid rgba(74, 222, 128, 0.2);
  }

  .failure-banner {
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #f87171;
    background: rgba(248, 113, 113, 0.08);
    border-top: 1px solid rgba(248, 113, 113, 0.2);
  }

  .progress-eta {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    font-size: 0.7rem;
    color: #94a3b8;
  }
</style>
