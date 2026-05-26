<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type {
    ProcessingSnapshot,
    ProcessingStage,
    Peak,
    BaselineParams,
    DespikeParams,
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

  const dispatch = createEventDispatcher<{
    stateUpdate: ProcessingSnapshot;
  }>();

  // Step enable states. baselineEnabled defaults to false so the checkbox
  // accurately reflects "no baseline applied yet" on initial mount; checking
  // the box triggers applyBaseline() via BaselineStep's toggle handler.
  let despikeEnabled = false;
  let bgEnabled = false;
  let baselineEnabled = false;
  // bgType is owned here (not in BackgroundStep) so the reactive reset block
  // below can clear it back to 'none' when raw spectrum input changes.
  let bgType: 'none' | 'as' | 'fs' = 'none';

  // Generation counter bumped on every raw-input change. Child steps
  // (BaselineStep, BackgroundStep) capture the current generation when
  // they kick off an async POST and check it before dispatching the apply
  // event — guards against stale responses from a request that started
  // before a point/modality switch landing after the reset and clobbering
  // the now-raw spectrum state.
  let inputGeneration = 0;

  // Track output at each stage so downstream steps get correct input
  let rawWavenumber: number[] = [];
  let rawIntensity: number[] = [];
  let afterDespikeIntensity: number[] | null = null;
  let afterBgIntensity: number[] | null = null;
  let afterBaselineIntensity: number[] | null = null;

  // Artifacts accumulated through the chain
  let spikeMask: boolean[] | undefined;
  let background: number[] | undefined;
  let backgroundScaled: number[] | undefined;
  let baseline: number[] | undefined;

  // All resets are funnelled through this function so Svelte 4's reactive
  // dependency tracker does NOT see the writes as reads. The reactive
  // block below would otherwise become reactive to baselineEnabled,
  // bgEnabled, bgType, and inputGeneration — Svelte 4's invalidation
  // guard (`if (x !== <new>) x = <new>`) reads those variables, and the
  // block would re-fire every time a user interaction set them, undoing
  // the user's input.
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
    baselineEnabled = false;
    bgEnabled = false;
    bgType = 'none';
    // Bump generation last so any in-flight apply request started under
    // the prior generation will be ignored when it returns.
    inputGeneration += 1;
  }

  // Update raw input when props change (point switch, modality change that
  // forces region reload, etc.). Only wavenumber + intensity are tracked
  // dependencies — the writes hide inside applyRawInputReset.
  $: applyRawInputReset(wavenumber, intensity);

  // Compute what each step receives as input
  $: despikeInput = rawIntensity;
  $: bgInput = despikeEnabled && afterDespikeIntensity ? afterDespikeIntensity : rawIntensity;
  $: baselineInput = bgEnabled && afterBgIntensity
    ? afterBgIntensity
    : despikeEnabled && afterDespikeIntensity
      ? afterDespikeIntensity
      : rawIntensity;
  $: fitInput = baselineEnabled && afterBaselineIntensity
    ? afterBaselineIntensity
    : bgEnabled && afterBgIntensity
      ? afterBgIntensity
      : despikeEnabled && afterDespikeIntensity
        ? afterDespikeIntensity
        : rawIntensity;

  function computeCurrentStage(): ProcessingStage {
    // The "latest completed" stage
    if (afterBaselineIntensity && baselineEnabled) return 'baseline_corrected';
    if (afterBgIntensity && bgEnabled) return 'bg_subtracted';
    if (afterDespikeIntensity && despikeEnabled) return 'despiked';
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
        background,
        backgroundScaled,
        ...(extraArtifacts as Record<string, unknown> | undefined),
      },
    };
    dispatch('stateUpdate', snapshot);
  }

  // --- Step handlers ---

  function onDespikeApply(e: CustomEvent<{ despiked: number[]; spikeMask: boolean[]; nSpikes: number; params: DespikeParams }>) {
    afterDespikeIntensity = e.detail.despiked;
    spikeMask = e.detail.spikeMask;
    // Downstream steps need to re-run, but we emit the current stage
    // Reset downstream results since input changed
    afterBgIntensity = null;
    afterBaselineIntensity = null;
    emitState('despiked', e.detail.despiked, undefined, {
      step: 'despike',
      n_spikes: e.detail.nSpikes,
      ...(e.detail.params as Record<string, unknown>),
    });
  }

  function onDespikeToggle(e: CustomEvent<{ enabled: boolean }>) {
    despikeEnabled = e.detail.enabled;
    if (!despikeEnabled) {
      afterDespikeIntensity = null;
      spikeMask = undefined;
      afterBgIntensity = null;
      afterBaselineIntensity = null;
      emitState('raw', rawIntensity);
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
      const output = despikeEnabled && afterDespikeIntensity ? afterDespikeIntensity : rawIntensity;
      emitState(stage, output);
    }
  }

  function onBaselineApply(e: CustomEvent<{ corrected: number[]; baseline: number[]; params: BaselineParams }>) {
    afterBaselineIntensity = e.detail.corrected;
    baseline = e.detail.baseline;
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
      const stage = computeCurrentStage();
      const output = bgEnabled && afterBgIntensity
        ? afterBgIntensity
        : despikeEnabled && afterDespikeIntensity
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
    wavenumber={rawWavenumber}
    intensity={despikeInput}
    enabled={despikeEnabled}
    {isAverageMode}
    on:apply={onDespikeApply}
    on:toggle={onDespikeToggle}
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
