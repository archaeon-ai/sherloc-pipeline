<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import classWindows from '../../lib/classWindows.json';
  import type { ClassificationRule, ClassificationProfile } from '../../lib/types/map';

  export let hasFittedData = false;

  const dispatch = createEventDispatcher<{
    profileApply: { profile: ClassificationProfile };
    profileReset: void;
  }>();

  function getDefaults(): ClassificationRule[] {
    return classWindows.map(({ lo, hi, ...rule }) => ({
      ...rule,
      center: (lo + hi) / 2,
      range: (hi - lo) / 2,
      disabled: false,
    }));
  }

  let rules: ClassificationRule[] = getDefaults();
  const defaults = getDefaults();

  // Track whether anything has changed from defaults
  $: hasChanges = rules.some((r, i) => {
    const d = defaults[i];
    return r.center !== d.center || r.range !== d.range
      || r.snr_threshold !== d.snr_threshold || r.disabled !== d.disabled;
  });

  function isChanged(rule: ClassificationRule, idx: number, field: 'center' | 'range' | 'snr_threshold'): boolean {
    return rule[field] !== defaults[idx][field];
  }

  function handleApply() {
    dispatch('profileApply', {
      profile: { name: 'custom', rules: [...rules.map((r) => ({ ...r }))] },
    });
  }

  function handleReset() {
    rules = getDefaults();
    dispatch('profileReset');
  }

  const DOMAIN_LABELS: Record<string, string> = {
    minerals: 'Minerals (cm⁻¹)',
    organics: 'Organics (cm⁻¹)',
    hydration: 'Hydration (cm⁻¹)',
    fluorescence: 'Fluorescence (nm)',
  };

  // Group rules by domain
  $: domainGroups = Object.entries(
    rules.reduce<Record<string, { rule: ClassificationRule; idx: number }[]>>((acc, rule, idx) => {
      (acc[rule.domain] ??= []).push({ rule, idx });
      return acc;
    }, {}),
  );
</script>

<div class="class-editor">
  <div class="editor-header">
    <span class="section-header">Classification Rules</span>
    <div class="btn-group">
      <button
        class="btn-sm btn-primary"
        on:click={handleApply}
        disabled={!hasFittedData}
        title={hasFittedData ? 'Apply profile to fitted results' : 'Run fitting first'}
      >Apply</button>
      <button
        class="btn-sm"
        on:click={handleReset}
        disabled={!hasChanges && hasFittedData}
      >Reset</button>
    </div>
  </div>
  <p class="editor-note">Edits apply to this page only; they are lost on reload.</p>

  {#each domainGroups as [domain, entries]}
    <div class="domain-section">
      <div class="domain-label">{DOMAIN_LABELS[domain] ?? domain}</div>
      <table class="class-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Center</th>
            <th>±Range</th>
            <th>SNR</th>
            <th class="col-disable">Off</th>
          </tr>
        </thead>
        <tbody>
          {#each entries as { rule, idx }}
            <tr class:disabled-row={rule.disabled}>
              <td class="mono">{rule.class_id}</td>
              <td>
                <input
                  type="number"
                  class="num-input"
                  class:changed={isChanged(rule, idx, 'center')}
                  bind:value={rules[idx].center}
                  step="1"
                />
              </td>
              <td>
                <input
                  type="number"
                  class="num-input"
                  class:changed={isChanged(rule, idx, 'range')}
                  bind:value={rules[idx].range}
                  step="1"
                />
              </td>
              <td>
                <input
                  type="number"
                  class="num-input"
                  class:changed={isChanged(rule, idx, 'snr_threshold')}
                  bind:value={rules[idx].snr_threshold}
                  step="0.5"
                />
              </td>
              <td class="col-disable">
                <input
                  type="checkbox"
                  bind:checked={rules[idx].disabled}
                />
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/each}
</div>

<style>
  .class-editor {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .editor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .editor-note {
    margin: 0;
    font-size: 0.7rem;
    color: var(--color-text-secondary);
  }

  .section-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .btn-group {
    display: flex;
    gap: 4px;
  }

  .btn-sm {
    font-size: 0.7rem;
    padding: 2px 8px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text);
    cursor: pointer;
  }

  .btn-sm:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--color-primary);
    color: white;
    border-color: var(--color-primary);
  }

  .btn-primary:disabled {
    background: var(--color-primary);
    opacity: 0.4;
  }

  .domain-section {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }

  .domain-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--color-text-secondary);
    padding: 3px 6px;
    background: var(--color-background);
    border-bottom: 1px solid var(--color-border);
  }

  .class-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72rem;
  }

  .class-table th {
    padding: 2px 4px;
    text-align: left;
    font-weight: 600;
    color: var(--color-text-secondary);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .class-table td {
    padding: 1px 4px;
    border-top: 1px solid var(--color-border);
    vertical-align: middle;
  }

  .mono {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    white-space: nowrap;
  }

  .col-disable {
    width: 28px;
    text-align: center;
  }

  .num-input {
    width: 54px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 1px 3px;
    border: 1px solid var(--color-border);
    border-radius: 2px;
    background: var(--color-surface);
    color: var(--color-text);
    text-align: right;
  }

  .num-input.changed {
    background: rgba(59, 130, 246, 0.08);
    border-color: var(--color-primary);
  }

  .disabled-row {
    opacity: 0.4;
  }

  .disabled-row .num-input {
    pointer-events: none;
  }
</style>
