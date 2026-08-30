<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type {
    ProcessingSnapshot,
    ProcessingStage,
    Peak,
    BaselineParams,
    DespikeParams,
    DespikeMethod,
    FitParams,
  } from '../lib/types';
  import DespikeStep from './DespikeStep.svelte';
  import BackgroundStep from './BackgroundStep.svelte';
  import BaselineStep from './BaselineStep.svelte';
  import RamanFitStep from './RamanFitStep.svelte';

  export let wavenumber: number[] = [];
  export let wavelength: number[] | null = null;
  export let intensity: number[] = [];
  export let scanPpp: number = 1;
  export let isAverageMode: boolean = true;
  export let isSinglePoint: boolean = false;
  export let onRegionSwitch: ((region: string) => void) | null = null;
  // Forwarded to RamanFitStep → /api/process/fit so the backend quality
  // classifier applies the calibration-scan downgrade rule (v4.1.12).
  export let targetType: string | null = null;
  export let visibleRange: [number, number] | null = null;

  // --- Despike method selector (issue #6) ---
  // The method is owned by the workbench (it owns the fetch); the chain only
  // threads it down to DespikeStep and routes the chain input accordingly.
  //   - 'none' / 'ml' : the `intensity` prop IS the chain input as-fetched.
  //                     For 'ml' that fetched array is already server-despiked,
  //                     so the chain's despike step is a pass-through and
  //                     downstream steps operate on the (despiked) fetch.
  //   - 'modz'        : `intensity` is raw; the client-side modz compute runs
  //                     in DespikeStep and its result feeds downstream.
  export let despikeMethod: DespikeMethod = 'none';
  export let mlMaskCount: number = 0;
  export let region: string = 'R1';
  // ML provenance from the latest fetch (only meaningful when method==='ml').
  export let mlApplied: boolean = false;
  export let mlMethodLabel: string | null = null;
  export let mlNMaskedChannels: number = 0;
  export let mlMissingRegions: string[] = [];
  // True when ML is active on an aggregate (average/subset) view (issue #12).
  // The union mask spans many contributing points, so on-trace markers are
  // misleading and suppressed; DespikeStep shows the aggregate-count copy
  // instead of per-position provenance. Owned by the Workbench (it owns the
  // selection mode) and threaded down purely for the provenance display.
  export let mlAggregateView: boolean = false;

  const dispatch = createEventDispatcher<{
    stateUpdate: ProcessingSnapshot;
    // Bubbles a user despike-method change up to the workbench, which decides
    // whether a refetch is required (crossing the ML boundary) and resets.
    despikeMethodChange: { method: DespikeMethod };
  }>();

  // Step enable states. baselineEnabled defaults to false so the checkbox
  // accurately reflects "no baseline applied yet" on initial mount; checking
  // the box triggers applyBaseline() via BaselineStep's toggle handler.
  let bgEnabled = false;
  let baselineEnabled = false;
  // bgType is owned here (not in BackgroundStep) so the reactive reset block
  // below can clear it back to 'none' when raw spectrum input changes.
  let bgType: 'none' | 'as' | 'fs' = 'none';

  // Generation counter bumped on every raw-input change. Child steps
  // (DespikeStep, BaselineStep, BackgroundStep) capture the current
  // generation when they kick off an async POST and check it before
  // dispatching the apply event — guards against stale responses from a
  // request that started before a point/modality switch landing after the
  // reset and clobbering the now-raw spectrum state. For despike, the
  // generation also rides in the apply payload and onDespikeApply enforces
  // the match chain-side (PR #7 review F4).
  let inputGeneration = 0;

  // Track output at each stage so downstream steps get correct input
  let rawWavenumber: number[] = [];
  let rawIntensity: number[] = [];
  // Client-side modz result (only ever set when despikeMethod === 'modz').
  let afterDespikeIntensity: number[] | null = null;
  let afterBgIntensity: number[] | null = null;
  let afterBaselineIntensity: number[] | null = null;

  // Artifacts accumulated through the chain
  let spikeMask: boolean[] | undefined;
  let background: number[] | undefined;
  let backgroundScaled: number[] | undefined;
  let baseline: number[] | undefined;
  let appliedBaselineRange: [number, number] | undefined;

  let despikeStep: DespikeStep;

  // Whether the client-side modz compute is the active despike. ML and none
  // both leave the chain's despike step a pass-through — for ML the fetched
  // `intensity` is already despiked, for none it is raw. This is the routing
  // half of the no-double-despike invariant: `afterDespikeIntensity` is only
  // consulted (and only populated) when method === 'modz'.
  $: modzActive = despikeMethod === 'modz';

  // All resets are funnelled through this function so Svelte 4's reactive
  // dependency tracker does NOT see the writes as reads (FRONTEND_HAZARDS H1).
  function applyRawInputReset(w: number[], i: number[]): void {
    rawWavenumber = w;
    rawIntensity = i;
    afterDespikeIntensity = null;
    afterBgIntensity = null;
    afterBaselineIntensity = null;
    spikeMask = undefined;
    background = undefined;
    backgroundScaled = undefined;
    baseline = undefined;
    appliedBaselineRange = undefined;
    baselineEnabled = false;
    bgEnabled = false;
    bgType = 'none';
    // Clear the modz step's local indicator (named call, not a reactive write).
    if (despikeStep) despikeStep.resetModzIndicator();
    // Bump generation last so any in-flight apply request started under
    // the prior generation will be ignored when it returns.
    inputGeneration += 1;
    // If modz is the active method, kick off the client compute against the
    // freshly-arrived raw input. (ML/none need no compute — the fetch is the
    // chain input.) Defer to a microtask so DespikeStep has the new props.
    if (despikeMethod === 'modz') {
      queueMicrotask(() => {
        if (despikeStep) despikeStep.runModz();
      });
    }
  }

  // Update raw input when props change (point switch, modality change that
  // forces region reload, ML/none/modz refetch, etc.). Only wavenumber +
  // intensity are tracked dependencies — the writes hide inside the helper.
  $: applyRawInputReset(wavenumber, intensity);

  // Compute what each step receives as input. The despike step only diverts
  // the stream when modz is the active method AND it has produced a result.
  $: despikeInput = rawIntensity;
  $: bgInput = modzActive && afterDespikeIntensity ? afterDespikeIntensity : rawIntensity;
  $: baselineInput = bgEnabled && afterBgIntensity
    ? afterBgIntensity
    : modzActive && afterDespikeIntensity
      ? afterDespikeIntensity
      : rawIntensity;
  $: fitInput = baselineEnabled && afterBaselineIntensity
    ? afterBaselineIntensity
    : bgEnabled && afterBgIntensity
      ? afterBgIntensity
      : modzActive && afterDespikeIntensity
        ? afterDespikeIntensity
        : rawIntensity;

  function computeCurrentStage(): ProcessingStage {
    // The "latest completed" stage
    if (afterBaselineIntensity && baselineEnabled) return 'baseline_corrected';
    if (afterBgIntensity && bgEnabled) return 'bg_subtracted';
    if (afterDespikeIntensity && modzActive) return 'despiked';
    return 'raw';
  }

  function emitState(
    stage: ProcessingStage,
    outputIntensity: number[],
    extraArtifacts?: Record<string, unknown>,
    stepParams?: Record<string, unknown>,
  ) {
    const snapshot: ProcessingSnapshot = {
      stage,
      raman: { wavenumber: rawWavenumber, intensity: outputIntensity },
      params: stepParams ?? {},
      artifacts: {
        spikeMask,
        baseline,
        baselineRange: appliedBaselineRange,
        background,
        backgroundScaled,
        ...(extraArtifacts as Record<string, unknown> | undefined),
      },
    };
    dispatch('stateUpdate', snapshot);
  }

  // --- Step handlers ---

  // Client-side modz result. Only ever fires when method === 'modz' (the step
  // hard-gates its compute), so this can never double-apply over an ML fetch.
  // The generation echo is the authoritative staleness check (PR #7 review
  // F4): a result computed from a previous input generation — e.g. a slow
  // POST from before a point switch — is rejected even though modz is still
  // the active method.
  function onDespikeApply(e: CustomEvent<{ despiked: number[]; spikeMask: boolean[]; nSpikes: number; params: DespikeParams; generation: number }>) {
    if (!modzActive) return; // defensive: ignore a late result after a switch
    if (e.detail.generation !== inputGeneration) return; // stale input (F4)
    afterDespikeIntensity = e.detail.despiked;
    spikeMask = e.detail.spikeMask;
    // Downstream steps need to re-run; reset their results since input changed.
    afterBgIntensity = null;
    afterBaselineIntensity = null;
    emitState('despiked', e.detail.despiked, undefined, {
      step: 'despike',
      method: 'modz',
      n_spikes: e.detail.nSpikes,
      ...(e.detail.params as Record<string, unknown>),
    });
  }

  // User picked a despike method in the step. Bubble to the workbench, which
  // owns the fetch + decides on a refetch. Locally, clear any modz-only chain
  // state so leaving modz can't strand an old client-despiked array, and emit
  // a fresh raw stage so the view returns to the fetched array immediately.
  function onDespikeMethodChange(e: CustomEvent<{ method: DespikeMethod; prev: DespikeMethod }>) {
    afterDespikeIntensity = null;
    spikeMask = undefined;
    afterBgIntensity = null;
    afterBaselineIntensity = null;
    bgEnabled = false;
    baselineEnabled = false;
    bgType = 'none';
    emitState('raw', rawIntensity);
    dispatch('despikeMethodChange', { method: e.detail.method });
    // none→modz causes no refetch (both fetch raw), so applyRawInputReset
    // never fires and nothing else would kick the client compute — kick it
    // here (PR #7 review F1). ml→modz is deliberately EXCLUDED: that crossing
    // refetches (the current array is the stale ML-despiked one), and
    // applyRawInputReset kicks runModz once the raw fetch lands. Deferred to
    // a microtask (same pattern as applyRawInputReset) so the workbench's
    // synchronous method update has flushed before the compute dispatches.
    if (e.detail.method === 'modz' && e.detail.prev !== 'ml') {
      queueMicrotask(() => {
        if (despikeStep) despikeStep.runModz();
      });
    }
  }

  function onBgApply(e: CustomEvent<{ subtracted: number[]; backgroundScaled: number[]; scaleUsed: number; bgType: string }>) {
    afterBgIntensity = e.detail.subtracted;
    backgroundScaled = e.detail.backgroundScaled;
    // Reset downstream
    afterBaselineIntensity = null;
    emitState('bg_subtracted', e.detail.subtracted, undefined, {
      step: 'bg_subtract',
      bg_type: e.detail.bgType,
      scale_used: e.detail.scaleUsed,
    });
  }

  function onBgToggle(e: CustomEvent<{ enabled: boolean }>) {
    bgEnabled = e.detail.enabled;
    if (!bgEnabled) {
      afterBgIntensity = null;
      backgroundScaled = undefined;
      background = undefined;
      afterBaselineIntensity = null;
      const stage = computeCurrentStage();
      const output = modzActive && afterDespikeIntensity ? afterDespikeIntensity : rawIntensity;
      emitState(stage, output);
    }
  }

  function onBaselineApply(e: CustomEvent<{ corrected: number[]; baseline: number[]; params: BaselineParams }>) {
    afterBaselineIntensity = e.detail.corrected;
    baseline = e.detail.baseline;
    appliedBaselineRange = e.detail.params.wavenumber_range;
    emitState('baseline_corrected', e.detail.corrected, undefined, {
      step: 'baseline',
      ...(e.detail.params as Record<string, unknown>),
    });
  }

  function onBaselineToggle(e: CustomEvent<{ enabled: boolean }>) {
    baselineEnabled = e.detail.enabled;
    if (!baselineEnabled) {
      afterBaselineIntensity = null;
      baseline = undefined;
      appliedBaselineRange = undefined;
      const stage = computeCurrentStage();
      const output = bgEnabled && afterBgIntensity
        ? afterBgIntensity
        : modzActive && afterDespikeIntensity
          ? afterDespikeIntensity
          : rawIntensity;
      emitState(stage, output);
    }
  }

  function onFitApply(e: CustomEvent<{
    peaks: Peak[];
    fitCurve: number[];
    residual: number[];
    corrected: number[];
    baseline: number[];
    rSquared: number;
    modelSelectionMethod: string;
    fitWavenumber: number[];
    params: FitParams;
  }>) {
    const d = e.detail;
    // Don't include baseline from fit response — fit no longer applies a baseline
    // (workbench's BaselineStep is the canonical source). Letting d.baseline (all zeros)
    // through would clobber the BaselineStep baseline already in artifacts.
    emitState(
      'raman_fitted',
      d.corrected,
      {
        fitCurve: d.fitCurve,
        peaks: d.peaks,
        residual: d.residual,
        rSquared: d.rSquared,
        modelSelectionMethod: d.modelSelectionMethod,
        fitRange: d.params.wavenumber_range as [number, number],
      },
      {
        step: 'raman_fit',
        r_squared: d.rSquared,
        model_selection_method: d.modelSelectionMethod,
        n_peaks: d.peaks.length,
        ...(d.params as Record<string, unknown>),
      },
    );
  }
</script>

<div class="chain-container">
  <DespikeStep
    bind:this={despikeStep}
    wavenumber={rawWavenumber}
    intensity={despikeInput}
    method={despikeMethod}
    {mlMaskCount}
    {region}
    {mlApplied}
    {mlMethodLabel}
    {mlNMaskedChannels}
    {mlMissingRegions}
    {mlAggregateView}
    {inputGeneration}
    on:apply={onDespikeApply}
    on:methodChange={onDespikeMethodChange}
  />

  <BackgroundStep
    wavenumber={rawWavenumber}
    intensity={bgInput}
    {scanPpp}
    {isSinglePoint}
    bind:bgType
    {inputGeneration}
    on:apply={onBgApply}
    on:toggle={onBgToggle}
  />

  <BaselineStep
    wavenumber={rawWavenumber}
    intensity={baselineInput}
    enabled={baselineEnabled}
    {inputGeneration}
    {visibleRange}
    on:apply={onBaselineApply}
    on:toggle={onBaselineToggle}
  />

  <RamanFitStep
    wavenumber={rawWavenumber}
    {wavelength}
    intensity={fitInput}
    {onRegionSwitch}
    {targetType}
    on:apply={onFitApply}
  />
</div>

<style>
  .chain-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
</style>
