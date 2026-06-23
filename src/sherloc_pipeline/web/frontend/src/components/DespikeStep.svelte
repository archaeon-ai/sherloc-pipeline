<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { postDespike } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { DespikeParams, DespikeMethod } from '../lib/types';
  import InfoTooltip from './InfoTooltip.svelte';

  const despikeRef = `<div class="ref-title">Cosmic-ray despiking</div>
<b>ML</b> — certified cosmic-ray classifier applied at ingest; the server returns
the despiked spectrum from stored per-(point,region) masks (no inference on the
serving host). Covers R1/R2/R3; composite views are all-or-none.
<b>modz</b> — live rolling-median MAD z-score despike (R1 only), tunable below.
Protected regions guard sulfate peaks and the laser line from false flagging.
<div class="ref-cite">Whitaker, D.A. &amp; Hayes, K. (2018). A simple algorithm for despiking Raman spectra.
<i>Chemometrics and Intelligent Laboratory Systems</i>, 179, 82&ndash;84.
<a href="https://doi.org/10.1016/j.chemolab.2018.06.009" target="_blank">doi:10.1016/j.chemolab.2018.06.009</a></div>`;

  // --- Inputs ---
  // `intensity` is the chain input. When method==='ml' this is ALREADY the
  // server-despiked array (the fetch applied the masks); the step is a pure
  // pass-through that only DISPLAYS provenance. When method==='modz' this is
  // the raw array and the client-side compute runs against it.
  export let wavenumber: number[] = [];
  export let intensity: number[] = [];

  // Method selector state (owned by the workbench, threaded through the chain).
  export let method: DespikeMethod = 'none';
  // Gating inputs.
  export let mlMaskCount: number = 0; // 0/undefined → ML disabled
  export let region: string = 'R1'; // modz disabled when not R1
  export let collapsed: boolean = true;
  // Raw-input generation from the chain (PR #7 review F4). applyDespike
  // captures it at entry and drops its result after the await if the chain's
  // input was replaced while the POST was in flight — without this, a slow
  // modz response computed from the PREVIOUS point's array could land after
  // a point switch (method still 'modz') and render stale despiked data for
  // the current selection. Capture-at-entry is sound because the chain bumps
  // the counter in the same synchronous helper that swaps the input arrays,
  // and runModz is microtask-deferred, so the captured generation is always
  // paired with the matching wavenumber/intensity props.
  export let inputGeneration: number = 0;

  // ML provenance from the most recent fetch response (set by the workbench).
  // These are only meaningful when method==='ml'.
  export let mlApplied: boolean = false;
  export let mlMethodLabel: string | null = null;
  export let mlNMaskedChannels: number = 0;
  export let mlMissingRegions: string[] = [];
  // True when ML is active on an aggregate (average/subset) view (issue #12).
  // The union `masked_positions` spans the contributing points, so on-trace
  // markers and the per-row CSV column are suppressed upstream; here we replace
  // the per-position provenance line with an aggregate-count line that points
  // the user to a single point for exact positions.
  export let mlAggregateView: boolean = false;

  const dispatch = createEventDispatcher<{
    // Client-side modz result (only ever emitted when method==='modz').
    // `generation` echoes the inputGeneration captured when the compute
    // started; the chain requires it to match its current generation before
    // accepting the result (authoritative chain-side staleness check, F4).
    apply: { despiked: number[]; spikeMask: boolean[]; nSpikes: number; params: DespikeParams; generation: number };
    // Method changed by the user. `prev` lets the chain distinguish
    // none→modz (no refetch happens, so the chain must kick the client
    // compute itself) from ml→modz (the workbench refetch path kicks it
    // once the raw fetch lands — kicking earlier would compute modz over
    // the stale ML array). The workbench decides whether to refetch (it
    // must when crossing the ML boundary) and clears downstream chain state.
    methodChange: { method: DespikeMethod; prev: DespikeMethod };
  }>();

  $: mlAvailable = (mlMaskCount ?? 0) > 0;
  $: modzAvailable = region === 'R1';

  // modz parameters
  let windowSize = 7;
  let zThreshold = 6.0;
  let maxIterations = 1;
  let sulfateGuard = true;

  let computing = false;
  let error = '';
  let lastNSpikes: number | null = null;

  // Debounce timer for modz param tweaks
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  function debouncedApply() {
    if (method !== 'modz') return;
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => applyDespike(), 300);
  }

  // Single funnel for a user method change. Per FRONTEND_HAZARDS H1 we never
  // write `method` from a reactive block — only from this explicit handler.
  function selectMethod(next: DespikeMethod): void {
    if (next === method) return;
    // Guard against selecting a disabled option (defensive; the radios are
    // also `disabled` in the markup).
    if (next === 'ml' && !mlAvailable) return;
    if (next === 'modz' && !modzAvailable) return;
    const prev = method;
    method = next;
    // Clear any stale modz result indicator; the modz compute (if newly
    // selected) is kicked by the chain (none→modz) or by the workbench
    // refetch path (ml→modz) — see the methodChange payload doc above.
    if (next !== 'modz') {
      lastNSpikes = null;
      error = '';
    }
    dispatch('methodChange', { method: next, prev });
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

  // Run the client-side modz compute. Called by the workbench (via the
  // exported runModz) once it has confirmed the fetch is raw, and on local
  // param tweaks. Hard-gated on method==='modz' so it can NEVER run while ML
  // is selected — this is the structural half of the no-double-despike
  // invariant (the ML half is that the fetch returns the despiked array and
  // this compute is skipped).
  async function applyDespike() {
    if (method !== 'modz' || wavenumber.length === 0) return;
    // Pair this compute with the input arrays it reads (F4): if the chain
    // swaps its input (point/region switch) while the POST is in flight,
    // inputGeneration moves on and this result is dropped below.
    const gen = inputGeneration;
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
      // Drop stale results: a late method switch could have flipped us off
      // modz (double-apply hazard over an ML or none fetch), or the input
      // could have been replaced mid-flight (previous point's despiked array
      // landing on the current selection, F4).
      if (method !== 'modz' || gen !== inputGeneration) return;
      lastNSpikes = result.n_spikes;
      dispatch('apply', {
        despiked: result.despiked,
        spikeMask: result.spike_mask,
        nSpikes: result.n_spikes,
        params: result.params_used,
        generation: gen,
      });
    } catch (e) {
      // Same staleness rule for failures: a superseded compute's error is
      // irrelevant — don't surface it over the current input's state.
      if (method !== 'modz' || gen !== inputGeneration) return;
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Despiking failed';
      }
    } finally {
      computing = false;
    }
  }

  // Public entry the chain calls to (re)run modz against the current raw
  // input — after a refetch lands while modz is active (applyRawInputReset)
  // or on a none→modz selection (onDespikeMethodChange kick).
  export function runModz(): void {
    if (method === 'modz' && wavenumber.length > 0) {
      applyDespike();
    }
  }

  // Reset the modz result indicator when the parent's raw input is replaced
  // (point/region switch). Called by the workbench; not a reactive write.
  export function resetModzIndicator(): void {
    lastNSpikes = null;
    error = '';
  }

  // Header badge text for the active method.
  $: headerBadge =
    method === 'ml'
      ? mlApplied
        ? `ML · ${mlNMaskedChannels} ch`
        : 'ML · no mask'
      : method === 'modz' && lastNSpikes !== null
        ? `${lastNSpikes} spike${lastNSpikes !== 1 ? 's' : ''}`
        : null;
</script>

<div class="step-card" class:step-enabled={method !== 'none'}>
  <button class="step-header" on:click={toggleCollapsed}>
    <div class="step-header-left">
      <span class="step-title">1. Despike</span>
      {#if computing}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="step-header-right">
      {#if headerBadge}
        <span
          class="step-badge mono"
          class:badge-warn={method === 'ml' && !mlApplied}
        >{headerBadge}</span>
      {/if}
      <span class="collapse-icon">{collapsed ? '+' : '-'}</span>
    </div>
  </button>

  {#if !collapsed}
    <div class="step-body">
      <!-- Method selector: None | ML | modz -->
      <div class="method-row">
        <span class="method-label">
          Method
          <InfoTooltip text={despikeRef} />
        </span>
        <div class="method-segmented" role="radiogroup" aria-label="Despike method">
          <label class="seg-option" class:active={method === 'none'}>
            <input
              type="radio"
              name="despike-method"
              value="none"
              checked={method === 'none'}
              on:change={() => selectMethod('none')}
            />
            None
          </label>
          <label
            class="seg-option"
            class:active={method === 'ml'}
            class:disabled={!mlAvailable}
            title={mlAvailable ? 'Stored ML cosmic-ray masks (R1/R2/R3)' : 'No ML masks stored for this scan'}
          >
            <input
              type="radio"
              name="despike-method"
              value="ml"
              checked={method === 'ml'}
              disabled={!mlAvailable}
              on:change={() => selectMethod('ml')}
            />
            ML
          </label>
          <label
            class="seg-option"
            class:active={method === 'modz'}
            class:disabled={!modzAvailable}
            title={modzAvailable ? 'Live rolling-median MAD despike' : 'R1 only'}
          >
            <input
              type="radio"
              name="despike-method"
              value="modz"
              checked={method === 'modz'}
              disabled={!modzAvailable}
              on:change={() => selectMethod('modz')}
            />
            modz
          </label>
        </div>
      </div>

      {#if !mlAvailable}
        <div class="method-hint">No ML masks stored for this scan.</div>
      {/if}
      {#if !modzAvailable}
        <div class="method-hint">modz is R1 only — switch the spectral region to use it.</div>
      {/if}

      {#if error}
        <div class="step-error">{error}</div>
      {/if}

      <!-- ML provenance (pass-through display only) -->
      {#if method === 'ml'}
        <div class="ml-provenance" data-testid="ml-provenance">
          {#if mlApplied}
            {#if mlAggregateView}
              <!-- issue #12: union mask across the contributing points — the
                   per-position markers are misleading on a mean/trim-mean
                   trace, so report the aggregate count and steer to a point. -->
              <div class="prov-line ok" data-testid="ml-aggregate-note">
                Masks applied on {mlNMaskedChannels} channel{mlNMaskedChannels !== 1 ? 's' : ''} across the contributing points — select a single point to see positions.
              </div>
            {:else}
              <div class="prov-line ok">
                CR despiked
                {#if mlMethodLabel}<span class="mono">({mlMethodLabel})</span>{/if}
                · {mlNMaskedChannels} channel{mlNMaskedChannels !== 1 ? 's' : ''} masked
              </div>
            {/if}
          {:else}
            <div class="prov-line warn">No stored ML mask was applied to this view.</div>
          {/if}
          {#if mlMissingRegions && mlMissingRegions.length > 0}
            <div class="prov-line warn" data-testid="ml-missing-regions">
              Composite views are all-or-none: missing masks for {mlMissingRegions.join(', ')} —
              this view is not despiked.
            </div>
          {/if}
        </div>
      {/if}

      <!-- modz tunable params (visible only when modz is selected) -->
      {#if method === 'modz'}
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
            />
            Sulfate Guard
          </label>
        </div>
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

  .step-badge.badge-warn {
    background: var(--color-warning-light, #fff3e0);
    color: var(--color-warning, #b26a00);
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

  .method-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .method-label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    font-weight: 500;
  }

  .method-segmented {
    display: inline-flex;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--color-background);
  }

  .seg-option {
    position: relative;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-family: var(--font-mono);
    color: var(--color-text-secondary);
    cursor: pointer;
    user-select: none;
    margin-bottom: 0;
    border-left: 1px solid var(--color-border);
  }

  .seg-option:first-child {
    border-left: none;
  }

  .seg-option.active {
    background: var(--color-primary);
    color: white;
  }

  .seg-option.disabled {
    color: var(--color-text-tertiary);
    cursor: not-allowed;
    opacity: 0.6;
  }

  /* Hide the native radio; the label is the control surface. */
  .seg-option input[type='radio'] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    margin: 0;
  }

  .method-hint {
    font-size: 0.74rem;
    color: var(--color-text-tertiary);
    font-style: italic;
  }

  .ml-provenance {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.78rem;
  }

  .prov-line.ok {
    color: var(--color-success, #2e7d32);
  }

  .prov-line.warn {
    color: var(--color-warning, #b26a00);
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
