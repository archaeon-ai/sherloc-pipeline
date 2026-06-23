<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { ProcessingStage } from '../lib/types';

  export let currentStage: ProcessingStage = 'raw';

  const dispatch = createEventDispatcher<{
    jumpTo: { stage: ProcessingStage };
  }>();

  interface StageInfo {
    key: ProcessingStage;
    label: string;
  }

  const stages: StageInfo[] = [
    { key: 'raw', label: 'Raw' },
    { key: 'despiked', label: 'Despiked' },
    { key: 'bg_subtracted', label: 'BG Sub' },
    { key: 'baseline_corrected', label: 'Baseline' },
    { key: 'raman_fitted', label: 'Fit' },
  ];

  const stageOrder: ProcessingStage[] = ['raw', 'despiked', 'bg_subtracted', 'baseline_corrected', 'raman_fitted'];

  function stageIndex(stage: ProcessingStage): number {
    return stageOrder.indexOf(stage);
  }

  function isCompleted(stage: ProcessingStage): boolean {
    return stageIndex(stage) < stageIndex(currentStage);
  }

  function isCurrent(stage: ProcessingStage): boolean {
    return stage === currentStage;
  }

  function isFuture(stage: ProcessingStage): boolean {
    return stageIndex(stage) > stageIndex(currentStage);
  }

  function onClick(stage: ProcessingStage) {
    if (isFuture(stage)) return;
    dispatch('jumpTo', { stage });
  }
</script>

<div class="history-bar">
  {#each stages as stage, i}
    {#if i > 0}
      <span class="history-arrow" class:completed={stageIndex(stages[i].key) <= stageIndex(currentStage)}>&rarr;</span>
    {/if}
    <button
      class="history-step"
      class:completed={isCompleted(stage.key)}
      class:current={isCurrent(stage.key)}
      class:future={isFuture(stage.key)}
      disabled={isFuture(stage.key)}
      on:click={() => onClick(stage.key)}
      title={isFuture(stage.key) ? 'Not yet reached' : `Jump to ${stage.label}`}
    >
      {stage.label}
    </button>
  {/each}
</div>

<style>
  .history-bar {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 8px 16px;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    overflow-x: auto;
  }

  .history-arrow {
    color: var(--color-text-tertiary);
    font-size: 0.8rem;
    flex-shrink: 0;
  }

  .history-arrow.completed {
    color: var(--color-primary);
  }

  .history-step {
    padding: 4px 10px;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: 9999px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-text-secondary);
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .history-step:hover:not(:disabled) {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }

  .history-step.completed {
    background: var(--color-primary-light);
    border-color: var(--color-primary);
    color: var(--color-primary);
  }

  .history-step.current {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
    font-weight: 600;
  }

  .history-step.future {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .history-step:disabled {
    cursor: not-allowed;
  }
</style>
