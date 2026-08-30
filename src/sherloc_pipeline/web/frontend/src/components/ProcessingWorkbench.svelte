<script lang="ts">
  import { onMount } from 'svelte';
  import { navigate } from '../lib/stores';
  import {
    getScan,
    getScanPoints,
    getAverageSpectrum,
    getPointSpectrum,
    postSubsetAverage,
    getBadpixChannels,
  } from '../lib/api';
  import { ApiError } from '../lib/api';
  import type { BadpixChannel } from '../lib/types';
  import type {
    ScanDetail,
    SpectrumResponse,
    ScanPoint,
    ProcessingSnapshot,
    ProcessingStage,
    DespikeMethod,
    Peak,
  } from '../lib/types';
  import {
    processingState,
    undoStack,
    pushUndo,
    undo,
    resetProcessing,
    pointSelection,
  } from '../lib/processingStore';
  import PointSelector from './PointSelector.svelte';
  import ProcessingChain from './ProcessingChain.svelte';
  import ProcessingHistory from './ProcessingHistory.svelte';
  import PeakTable from './PeakTable.svelte';

  import RamanView from './RamanView.svelte';
  import AciViewer from './AciViewer.svelte';
  import ExportMenu from './ExportMenu.svelte';

  export let scanId: string;
  export let queryParams: Record<string, string> = {};

  let scan: ScanDetail | null = null;
  let points: ScanPoint[] = [];
  let spectrum: SpectrumResponse | null = null;
  let loading = true;
  let error = '';

  // Point selection
  let selectionMode: 'average' | 'subset' | 'point' = 'average';
  let selectedIndices: number[] = [];
  let selectedPointIdx: number | null = null;

  // Selection context OF THE COMMITTED spectrum (PR #13 review F1, issue #12).
  // The live selectionMode flips BEFORE the refetch commits (onPointSelect →
  // reloadAndReset is fire-and-forget), so during the pending window — or
  // indefinitely if the refetch rejects — gating ML markers/export on the live
  // flag would let the still-committed AGGREGATE spectrum's union mask render
  // and export as a point-view mask. This mirror is written ONLY inside
  // loadSpectrum's commit paths (under the generation guard, alongside the
  // `spectrum` assignment) — a named async function, never a reactive block
  // (H1). The live flag keeps driving selection UI; eligibility for ML markers
  // and the CSV spike_mask column derives from what is actually committed.
  let committedSelectionMode: 'average' | 'subset' | 'point' = 'average';

  // Averaging method
  let averagingMethod: 'mean' | 'trim_mean' | 'median' = 'trim_mean';
  let trimPct: string = '';  // empty = auto

  // Region selector
  let selectedRegion: 'R1' | 'R2' | 'R3' | 'R123' = 'R1';
  // Current interactive R1 viewport. BaselineStep defaults to full range and
  // only consumes this value when the operator selects "Current zoom".
  let visibleBaselineRange: [number, number] | null = null;

  // Despike method selector (issue #6). Owned here because the fetch carries
  // it (ML applies stored masks server-side; modz/none fetch raw). Initialised
  // to 'none' as a placeholder; applyDespikeDefault() (called from loadScan
  // after the scan metadata lands) sets the live default — 'ml' when the scan
  // has stored masks, 'none' otherwise. Never set from a $: reactive block (H1).
  let despikeMethod: DespikeMethod = 'none';
  // Count of stored ML masks for the active scan — gates the ML option.
  // Tolerate a legacy backend that omits the field by treating it as 0.
  $: mlMaskCount = scan?.ml_mask_count ?? 0;
  // ML despike provenance surfaced from the latest fetch response (only
  // meaningful when despikeMethod === 'ml').
  $: mlApplied = spectrum?.despike_applied ?? false;
  $: mlMethodLabel = spectrum?.despike_method ?? null;
  $: mlNMaskedChannels = spectrum?.n_masked_channels ?? 0;
  $: mlMissingRegions = spectrum?.despike_missing_regions ?? [];

  // --- Known-noisy channel annotation (issue #9) -----------------------------
  // A STATIC per-region list of known-noisy detector channels, fetched once per
  // region view (cached client-side in api.ts) and rendered as a separate,
  // default-OFF overlay in RamanView. Entirely independent of despike state:
  // the fetch carries no despike params and the overlay never alters the trace.
  // The overlay toggle (`showBadpixOverlay`) is owned HERE (the export
  // authority) so the CSV badpix column is gated on the same toggle that drives
  // the overlay — mirroring how the spike column is gated by the Workbench, not
  // by RamanView's internal state.
  let badpix: BadpixChannel[] = [];
  let showBadpixOverlay = false;

  // Generation counter for the badpix stale-response guard (PR #11 review
  // F1), mirroring spectrumRequestGen below. Badpix positions are REGION-
  // SPECIFIC served-array indices, so on a fast region switch an older
  // region's slower response must never overwrite the current region's
  // annotations — neither on the plot nor through the export gate
  // (exportBadpix). Each loadBadpix call claims a generation; a response
  // commits only if no newer request has started since. The call sites only
  // ever pass the then-current selectedRegion, so "newest generation" ==
  // "current region's request".
  let badpixRequestGen = 0;

  async function loadBadpix(reg: string): Promise<void> {
    const gen = ++badpixRequestGen;
    // Region transition: clear immediately so the previous region's
    // annotations neither render nor export while the new region's list is
    // in flight (their positions would be wrong for the new served array).
    badpix = [];
    try {
      const res = await getBadpixChannels(reg);
      // Stale-response guard: a newer request started while this one was in
      // flight — drop this response entirely.
      if (gen !== badpixRequestGen) return;
      badpix = res.badpix;
    } catch {
      // Annotation layer is best-effort: a failed fetch must never break the
      // spectrum view. The list was already cleared at request start, so the
      // (now-absent) overlay simply stays hidden.
    }
  }

  // References panel
  let showRefs = false;

  // Raw snapshot for reset
  let rawSnapshot: ProcessingSnapshot | null = null;

  // Current state derived from store
  $: currentState = $processingState;
  $: currentStage = currentState?.stage ?? 'raw';
  $: currentWavenumber = currentState?.raman.wavenumber ?? [];
  $: currentIntensity = currentState?.raman.intensity ?? [];
  $: fitPeaks = (currentState?.artifacts?.peaks as Peak[] | undefined) ?? [];
  $: fitCurve = (currentState?.artifacts?.fitCurve as number[] | undefined) ?? null;
  $: residual = (currentState?.artifacts?.residual as number[] | undefined) ?? null;
  $: baselineOverlay = (currentState?.artifacts?.baseline as number[] | undefined) ?? null;
  $: appliedBaselineRange =
    (currentState?.artifacts?.baselineRange as [number, number] | undefined) ?? null;
  $: rSquared = (currentState?.artifacts?.rSquared as number | undefined) ?? null;
  $: modelMethod = (currentState?.artifacts?.modelSelectionMethod as string | undefined) ?? null;
  $: fitRange = (currentState?.artifacts?.fitRange as [number, number] | undefined) ?? null;
  $: undoAvailable = $undoStack.length > 0;

  // Build processing_chain for export from undoStack history + current state.
  // Skips the initial 'raw' snapshot (no processing applied).
  $: processingChain = [...$undoStack, ...(currentState ? [currentState] : [])]
    .filter((s) => s.stage !== 'raw' && s.params && Object.keys(s.params).length > 0)
    .map((s) => ({ stage: s.stage, ...s.params }));

  // Show fit overlay when in fitted state
  $: showFit = currentStage === 'raman_fitted';
  $: showBaseline = currentStage === 'baseline_corrected' && baselineOverlay !== null;

  // --- Spike markers (issue #8) ---------------------------------------------
  // The red-triangle spike markers render in both ML and modz mode, but the
  // mask comes from different sources and the two never co-exist:
  //   - modz: the client compute in the chain emits a full-length boolean mask
  //     into the processing snapshot artifacts (currentState.artifacts.spikeMask).
  //   - ml:   the server returns masked_positions (served-array indices); we
  //     build a full-length boolean mask from them. ML markers sit on the
  //     already-cleaned fetched trace ("removal happened here").
  // Built via a pure helper (no reactive writes to user-mutated state — H1).

  /** Boolean mask (length = served array) with `true` at each ML position. */
  function buildMlSpikeMask(
    positions: number[] | undefined | null,
    length: number,
  ): boolean[] | null {
    if (!positions || positions.length === 0 || length === 0) return null;
    const m = new Array<boolean>(length).fill(false);
    for (const p of positions) {
      if (p >= 0 && p < length) m[p] = true;
    }
    return m;
  }

  /**
   * Set the initial despike-method selector from the scan's stored mask count.
   * Defaults to 'ml' when masks are available; falls back to 'none' otherwise.
   * Called once from loadScan() after the scan metadata lands — NEVER from a
   * $: reactive block (FRONTEND_HAZARDS H1: writes to user-mutated vars inside
   * a $: block register as tracked deps and stomp the user's selection on
   * any subsequent reactive re-fire).
   */
  function applyDespikeDefault(maskCount: number): void {
    despikeMethod = maskCount > 0 ? 'ml' : 'none';
  }

  // ML masks are per-(point,region). On a single point the union
  // `masked_positions` is exactly the channels repaired on the trace shown, so
  // the markers are true. On an aggregate view (average/subset) the union spans
  // ~N contributing points: each on-trace marker would assert "removal happened
  // here" for a mean/trim-mean trace where any single point's repair
  // contributes only ~1/N (and trim-mean often already excluded the outlier) —
  // visually broken and quantitatively misleading (issue #12). So the ML mask
  // is gated on point view — keyed on the COMMITTED spectrum's selection
  // context, not the live selectionMode flag (PR #13 review F1): the live flag
  // flips before the refetch lands, and the mask must describe the trace that
  // is actually displayed/exported (spectrum), which is commit-bound.
  // modz is unaffected — it computes on the DISPLAYED (aggregate) trace, so its
  // markers are exact for every view (see effectiveSpikeMask below).
  $: mlPointView = committedSelectionMode === 'point';
  // True when ML is selected on an aggregate (average/subset) view: markers and
  // the CSV column are suppressed, and DespikeStep shows the aggregate-count
  // copy instead of per-position provenance.
  $: mlAggregateView = despikeMethod === 'ml' && !mlPointView;

  // ML mask keyed off the fetched spectrum's served length (the ML trace is the
  // fetched array, so the mask aligns to spectrum.wavenumber === currentWavenumber
  // at the raw stage). Recomputed whenever the fetch lands. Built only on a
  // point view (issue #12 — aggregate union markers are misleading).
  $: mlSpikeMask =
    despikeMethod === 'ml' && mlPointView
      ? buildMlSpikeMask(spectrum?.masked_positions, spectrum?.wavenumber.length ?? 0)
      : null;

  // The effective mask + provenance handed to RamanView/ExportMenu, gated
  // STRICTLY three-way on the active method (PR #10 review F1): the ML mask
  // only when the method is 'ml', the modz client mask only when 'modz', and
  // 'none' yields null/null — no markers, no CSV column, no provenance
  // header. The explicit 'modz' arm matters: a modz artifact can outlive the
  // method (apply modz → switch to 'none' → undo restores the despiked
  // snapshot whose artifacts still carry spikeMask); without the gate that
  // stale mask would render and export as modz provenance while the active
  // method is 'none'. The ML arm additionally yields null on aggregate views
  // (mlSpikeMask is already null there), so effectiveSpikeMethod must match:
  // an 'ml' label with a null mask would leave a dangling provenance header on
  // an aggregate CSV (issue #12) — pair the method with mask presence.
  $: effectiveSpikeMask =
    despikeMethod === 'ml'
      ? mlSpikeMask
      : despikeMethod === 'modz'
        ? (currentState?.artifacts?.spikeMask ?? null)
        : null;
  $: effectiveSpikeMethod = (
    despikeMethod === 'ml' && mlPointView
      ? 'ml'
      : despikeMethod === 'modz'
        ? 'modz'
        : null
  ) as 'ml' | 'modz' | null;

  // --- Badpix export gating (issue #9) --------------------------------------
  // The CSV badpix column is emitted iff the overlay toggle is ON (and the
  // list is non-empty). This derived value is the single export authority,
  // mirroring effectiveSpikeMask: null means no column, no provenance header.
  // Pure read of toggle + fetched list — no write to user-mutated state (H1).
  $: exportBadpix =
    showBadpixOverlay && badpix.length > 0 ? badpix : null;

  onMount(async () => {
    await loadScan();
    await loadBadpix(selectedRegion);
  });

  async function loadScan() {
    loading = true;
    error = '';
    try {
      const res = await getScan(scanId);
      scan = res.scan;
      // Apply ML-or-none default once, before the first loadSpectrum call.
      // Funnel through the named helper — never set despikeMethod in a $: block
      // (FRONTEND_HAZARDS H1). The user can override via the despike selector;
      // applyDespikeDefault is never called again after mount.
      applyDespikeDefault(scan.ml_mask_count ?? 0);
      const ptRes = await getScanPoints(scanId);
      points = ptRes.points;

      // Pre-select point from query param (before first loadSpectrum)
      const pointParam = queryParams?.point;
      if (pointParam !== undefined) {
        const idx = parseInt(pointParam, 10);
        if (!isNaN(idx) && points.some((p) => p.point_index === idx)) {
          selectionMode = 'point';
          selectedPointIdx = idx;
          selectedIndices = [idx];
          pointSelection.set({ mode: 'point', pointIdx: idx, indices: [idx] });
        }
      }

      await loadSpectrum();
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Failed to load scan';
      }
    } finally {
      loading = false;
    }
  }

  // Request generation counter for the stale-response guard (PR #7 review
  // F3): each loadSpectrum call claims a generation; a response only commits
  // if no newer request has started since. Without this, a quick none→ml→none
  // toggle can let the slower ML response land last and overwrite the newer
  // raw response.
  let spectrumRequestGen = 0;

  async function loadSpectrum() {
    const gen = ++spectrumRequestGen;
    // The selection context this request is FOR (PR #13 review F1): captured
    // at entry, synchronously with the branch below that reads selectionMode,
    // so the commit records the request's own mode even if the live flag moves
    // on while the fetch is in flight (a newer request would bump the
    // generation and drop this commit anyway).
    const requestMode = selectionMode;
    try {
      const trimVal = trimPct !== '' ? parseFloat(trimPct) : undefined;
      // Despike at the fetch: only 'ml' asks the server to alter the returned
      // array (stored masks + interpolation). Under 'modz' the Workbench
      // DELIBERATELY fetches raw ('none') and runs the modified-z compute
      // client-side in the chain step with its tunable params — the backend's
      // server-side modz path (which live-computes on the served array) exists
      // for API consumers, not the Workbench. Sending 'modz' here would
      // double-despike every reload: server modz on the served array, then the
      // client compute again on top (PR #7 review F2).
      const fetchMethod: DespikeMethod = despikeMethod === 'modz' ? 'none' : despikeMethod;
      let result: SpectrumResponse;
      if (selectionMode === 'point' && selectedPointIdx !== null) {
        result = await getPointSpectrum(scanId, selectedPointIdx, {
          region: selectedRegion,
          despike_method: fetchMethod,
        });
      } else if (selectionMode === 'subset' && selectedIndices.length > 0) {
        result = await postSubsetAverage(scanId, {
          point_indices: selectedIndices,
          region: selectedRegion,
          averaging_method: averagingMethod,
          trim_pct: trimVal,
          despike_method: fetchMethod,
        });
      } else {
        result = await getAverageSpectrum(scanId, {
          region: selectedRegion,
          averaging_method: averagingMethod,
          trim_pct: trimVal,
          despike_method: fetchMethod,
        });
      }
      // Stale-response guard: a newer request started while this one was in
      // flight — drop this response entirely (do not touch spectrum or the
      // processing state).
      if (gen !== spectrumRequestGen) return;
      spectrum = result;
      // Commit-bound selection context (PR #13 review F1): recorded only here
      // and in the 404 arm below, in lockstep with the `spectrum` assignment,
      // so the ML marker/export gate keys on what is actually committed.
      committedSelectionMode = requestMode;
      initProcessingState();
    } catch (e) {
      // Stale-failure guard (PR #7 review F5): if a newer request superseded
      // this one, its failure is irrelevant — neither surface an error UI
      // for an obsolete request nor clear a newer spectrum.
      if (gen !== spectrumRequestGen) return;
      if (e instanceof ApiError && e.status === 404) {
        spectrum = null;
        // A committed "no spectrum" for this selection: record its mode so the
        // gate never inherits a stale context from the previous commit.
        committedSelectionMode = requestMode;
      } else {
        // Non-404 failure: the committed spectrum (and its recorded selection
        // context) remain those of the LAST successful commit — the displayed
        // trace is unchanged, so marker/export eligibility must be too.
        throw e;
      }
    }
  }

  function initProcessingState() {
    if (!spectrum) return;
    const snapshot: ProcessingSnapshot = {
      stage: 'raw',
      raman: {
        wavenumber: spectrum.wavenumber,
        intensity: spectrum.intensity,
      },
      params: {},
    };
    rawSnapshot = snapshot;
    processingState.set(snapshot);
    undoStack.set([]);
  }

  // Point selection handlers
  function onPointSelect(e: CustomEvent<{ indices: number[] }>) {
    const indices = e.detail.indices;
    if (indices.length === 0) {
      selectionMode = 'average';
      selectedIndices = [];
      selectedPointIdx = null;
    } else if (indices.length === 1) {
      selectionMode = 'point';
      selectedPointIdx = indices[0];
      selectedIndices = indices;
    } else {
      selectionMode = 'subset';
      selectedIndices = indices;
      selectedPointIdx = null;
    }
    pointSelection.set({
      mode: selectionMode,
      indices: selectedIndices.length > 0 ? selectedIndices : undefined,
      pointIdx: selectedPointIdx ?? undefined,
    });
    // Reload spectrum and reset chain
    reloadAndReset();
  }

  async function switchToAverage() {
    selectionMode = 'average';
    selectedIndices = [];
    selectedPointIdx = null;
    pointSelection.set({ mode: 'average' });
    await reloadAndReset();
  }

  async function reloadAndReset() {
    try {
      await loadSpectrum();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to reload spectrum';
    }
  }

  // Switch spectral region. modz is R1-only, so a switch to a non-R1 region
  // while modz is active falls the method back to 'none' (the fetch is raw
  // either way, so no extra refetch is needed beyond the region reload). ML
  // stays valid across regions (masks cover R1/R2/R3; composite is all-or-
  // none and surfaces a warning). Funnelled through a named helper so the
  // method write is never a reactive assignment (FRONTEND_HAZARDS H1).
  function switchRegion(reg: string): void {
    selectedRegion = reg as 'R1' | 'R2' | 'R3' | 'R123';
    visibleBaselineRange = null;
    if (reg !== 'R1' && despikeMethod === 'modz') {
      despikeMethod = 'none';
    }
    // The badpix list is region-specific (served positions differ per region),
    // so refresh it on a region switch. Independent of the spectrum reload and
    // of despike state — a failure here never blocks the reload.
    loadBadpix(selectedRegion);
    reloadAndReset();
  }

  // Handle a despike-method change bubbled up from the chain's DespikeStep.
  // A refetch is required exactly when crossing the ML boundary (none↔ml or
  // modz↔ml), because only 'ml' changes the bytes the server returns. A
  // none↔modz switch does NOT refetch: both fetch the raw spectrum and the
  // modz path is computed client-side in the chain step. This is the second
  // structural guarantee of the no-double-despike invariant — selecting modz
  // leaves the fetch raw (despike_method omitted/none from the server's POV),
  // and selecting ML refetches the masked array while the chain's modz compute
  // is hard-gated off.
  async function onDespikeMethodChange(e: CustomEvent<{ method: DespikeMethod }>) {
    const prev = despikeMethod;
    const next = e.detail.method;
    despikeMethod = next;
    const crossesMlBoundary = (prev === 'ml') !== (next === 'ml');
    if (crossesMlBoundary) {
      await reloadAndReset();
    }
    // none↔modz: no refetch. The chain already reset its downstream state and
    // emitted a raw stage; if modz was selected, the chain runs its compute
    // against the existing raw fetch on its own.
  }

  // Processing chain update
  function onChainStateUpdate(e: CustomEvent<ProcessingSnapshot>) {
    const newState = e.detail;
    // Push current state to undo before updating
    if (currentState) {
      pushUndo(currentState);
    }
    processingState.set(newState);
  }

  function handleUndo() {
    undo();
  }

  function handleReset() {
    if (rawSnapshot) {
      resetProcessing(rawSnapshot);
    }
  }

  function handleHistoryJump(e: CustomEvent<{ stage: ProcessingStage }>) {
    // Undo until we reach the target stage
    const targetStage = e.detail.stage;
    const stack = $undoStack;
    // Find the most recent snapshot in the undo stack matching the target stage
    for (let i = stack.length - 1; i >= 0; i--) {
      if (stack[i].stage === targetStage) {
        // Pop everything above this index
        const target = stack[i];
        undoStack.set(stack.slice(0, i));
        processingState.set(target);
        return;
      }
    }
    // If target is raw and not found in stack, just reset
    if (targetStage === 'raw' && rawSnapshot) {
      resetProcessing(rawSnapshot);
    }
  }

  function getPlotTitle(): string {
    const base = `${scan?.target ?? 'Unknown'} Sol ${scan?.sol_number ?? ''} ${scan?.scan_name ?? ''}`;
    if (selectionMode === 'point' && selectedPointIdx !== null) {
      return `${base} — Point ${selectedPointIdx}`;
    }
    if (selectionMode === 'subset') {
      return `${base} — ${selectedIndices.length} pts`;
    }
    return base;
  }
</script>

<div class="page-container">
  <!-- Breadcrumb -->
  <div class="breadcrumb">
    <button class="btn-link" on:click={() => navigate('#/')}>Scans</button>
    <span class="breadcrumb-sep">/</span>
    <span>{scan?.target ?? '...'} Sol {scan?.sol_number ?? '...'}</span>
    <ExportMenu
      {scanId}
      scanName={scan?.scan_name ?? ''}
      target={scan?.target ?? ''}
      solNumber={scan?.sol_number ?? 0}
      stage={currentStage}
      wavenumber={currentWavenumber}
      intensity={currentIntensity}
      artifacts={currentState?.artifacts ?? null}
      spikeMask={effectiveSpikeMask}
      spikeMethod={effectiveSpikeMethod}
      badpix={exportBadpix}
      {processingChain}
      scanNPoints={scan?.n_points ?? 0}
      {selectionMode}
      {selectedIndices}
      {selectedPointIdx}
    />
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading}
    <div class="empty-state"><span class="spinner"></span> Loading scan...</div>
  {:else if scan}
    <div class="workbench-layout">
      <!-- Sidebar -->
      <div class="workbench-sidebar">
        <!-- Scan Info Panel (inline) -->
        <div class="card">
          <div class="card-header">Scan Info</div>
          <div class="card-body">
            <dl class="meta-list">
              <dt>Sol</dt><dd class="mono">{scan.sol_number}</dd>
              <dt>Target</dt><dd>{scan.target ?? '--'}</dd>
              <dt>Scan</dt><dd class="mono">{scan.scan_name}</dd>
              <dt>Points</dt><dd class="mono">{scan.n_points}</dd>
              <dt>PPP</dt><dd class="mono">{scan.shots_per_point}</dd>
              <dt>Class</dt><dd><span class="badge badge-neutral">{scan.scan_class}</span></dd>
              <dt>Type</dt><dd>{scan.scan_type ?? '--'}</dd>
              <dt>Status</dt>
              <dd>
                <span
                  class="badge"
                  class:badge-success={scan.processing_status === 'completed'}
                  class:badge-error={scan.processing_status === 'failed'}
                  class:badge-neutral={!scan.processing_status}
                >
                  {scan.processing_status ?? 'unprocessed'}
                </span>
              </dd>
            </dl>
          </div>
        </div>

        <!-- Point Selector -->
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <span>Points</span>
            <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
              {#if selectionMode === 'point'}
                Point {selectedPointIdx}
              {:else if selectionMode === 'subset'}
                {selectedIndices.length} selected
              {:else}
                Average (all)
              {/if}
            </span>
          </div>
          <div class="card-body">
            <div class="view-toggle">
              <button
                class="btn-sm"
                class:btn-primary={selectionMode === 'average'}
                class:btn-secondary={selectionMode !== 'average'}
                on:click={switchToAverage}
              >
                Average
              </button>
            </div>
            {#if selectionMode !== 'point'}
              <div class="averaging-controls">
                <select bind:value={averagingMethod} on:change={reloadAndReset}>
                  <option value="trim_mean">Trim Mean</option>
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                </select>
                {#if averagingMethod === 'trim_mean'}
                  <input
                    type="number"
                    bind:value={trimPct}
                    on:change={reloadAndReset}
                    placeholder="auto"
                    min="0"
                    max="0.49"
                    step="0.01"
                    title="Trim fraction per tail (empty = auto)"
                    class="trim-input"
                  />
                {/if}
              </div>
            {/if}
            <PointSelector
              {points}
              {selectedIndices}
              on:select={onPointSelect}
            />
          </div>
        </div>

        <!-- Processing Chain -->
        <div class="card">
          <div class="card-header">Processing Chain</div>
          <div class="card-body chain-body">
            {#if spectrum}
              <ProcessingChain
                wavenumber={spectrum.wavenumber}
                wavelength={spectrum.wavelength ?? null}
                intensity={spectrum.intensity}
                scanPpp={scan.shots_per_point ?? 900}
                isAverageMode={selectionMode === 'average'}
                isSinglePoint={selectionMode === 'point'}
                targetType={scan.target_type ?? null}
                despikeMethod={despikeMethod}
                {mlMaskCount}
                region={selectedRegion}
                {mlApplied}
                {mlMethodLabel}
                {mlNMaskedChannels}
                {mlMissingRegions}
                {mlAggregateView}
                visibleRange={selectedRegion === 'R1' ? visibleBaselineRange : null}
                onRegionSwitch={(reg) => switchRegion(reg)}
                on:stateUpdate={onChainStateUpdate}
                on:despikeMethodChange={onDespikeMethodChange}
              />
            {:else}
              <div class="empty-state" style="padding: 16px; font-size: 0.85rem">
                No spectrum loaded
              </div>
            {/if}
          </div>
        </div>

        <!-- Undo / Reset -->
        <div class="undo-controls">
          <button
            class="btn-secondary btn-sm"
            on:click={handleUndo}
            disabled={!undoAvailable}
            title="Undo last processing step"
          >
            Undo
          </button>
          <button
            class="btn-secondary btn-sm"
            on:click={handleReset}
            title="Reset to raw spectrum"
          >
            Reset
          </button>
        </div>
      </div>

      <!-- Main Content -->
      <div class="workbench-main">
        <!-- Raman View -->
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <div class="flex gap-2 items-center">
              <span>Spectrum</span>
              {#if spectrum?.laser_normalized}
                <span class="badge badge-success" title="Photodiode laser power normalization applied">LN</span>
              {/if}
              <div class="region-selector">
                {#each ['R1', 'R2', 'R3', 'R123'] as reg}
                  <button
                    class="region-btn"
                    class:active={selectedRegion === reg}
                    on:click={() => switchRegion(reg)}
                  >
                    {reg === 'R123' ? 'full' : reg.toLowerCase()}
                  </button>
                {/each}
              </div>
            </div>
            <div class="flex gap-2 items-center">
              {#if showFit && rSquared !== null}
                <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                  R² = {rSquared.toFixed(4)}
                  {#if modelMethod}| {modelMethod}{/if}
                </span>
              {/if}
            </div>
          </div>
          <div class="card-body">
            {#if currentWavenumber.length > 0}
              <RamanView
                wavenumber={currentWavenumber}
                intensity={currentIntensity}
                rawIntensity={rawSnapshot?.raman.intensity ?? []}
                stage={currentStage}
                spikeMask={effectiveSpikeMask}
                spikeMethod={effectiveSpikeMethod}
                {badpix}
                bind:showBadpix={showBadpixOverlay}
                background={currentState?.artifacts?.background ?? null}
                backgroundScaled={currentState?.artifacts?.backgroundScaled ?? null}
                baseline={baselineOverlay}
                baselineRange={appliedBaselineRange}
                {fitCurve}
                peaks={showFit ? fitPeaks : []}
                {residual}
                showRaw={currentStage !== 'raw'}
                showResidual={showFit && residual !== null}
                title={getPlotTitle()}
                height={480}
                {selectionMode}
                {selectedIndices}
                {selectedPointIdx}
                nPoints={scan?.n_points ?? 0}
                {fitRange}
                wavelength={spectrum?.wavelength ?? null}
                region={selectedRegion}
                {averagingMethod}
                on:viewRangeChange={(e) => {
                  if (selectedRegion === 'R1') visibleBaselineRange = e.detail.range;
                }}
              />
            {:else}
              <div class="empty-state" style="padding: 60px">
                No spectrum data available
              </div>
            {/if}
          </div>
        </div>

        <!-- Peak Table (shown when fitted) -->
        {#if showFit && fitPeaks.length > 0}
          <div class="card" style="margin-top: 16px">
            <div class="card-header flex items-center justify-between">
              <span>Peak Results</span>
              <span class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                {fitPeaks.length} peak{fitPeaks.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div class="card-body">
              <PeakTable peaks={fitPeaks} />
            </div>
          </div>
        {/if}

        <!-- ACI Viewer -->
        <div class="card" style="margin-top: 16px">
          <div class="card-header">ACI Context Image</div>
          <div class="card-body aci-body">
            <AciViewer
              {scanId}
              {points}
              {selectedIndices}
              selectionMode={selectionMode}
              {selectedPointIdx}
              colorizedAvailable={scan?.colorized_aci_available ?? false}
              on:pointClick={(e) => {
                const idx = e.detail.pointIndex;
                if (selectedIndices.includes(idx)) {
                  // Deselect: remove from group
                  selectedIndices = selectedIndices.filter((i) => i !== idx);
                } else {
                  // Add to group
                  selectedIndices = [...selectedIndices, idx];
                }
                if (selectedIndices.length === 0) {
                  selectionMode = 'average';
                  selectedPointIdx = null;
                } else if (selectedIndices.length === 1) {
                  selectionMode = 'point';
                  selectedPointIdx = selectedIndices[0];
                } else {
                  selectionMode = 'subset';
                  selectedPointIdx = null;
                }
                pointSelection.set({
                  mode: selectionMode,
                  indices: selectedIndices.length > 0 ? selectedIndices : undefined,
                  pointIdx: selectedPointIdx ?? undefined,
                });
                reloadAndReset();
              }}
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Processing History breadcrumb -->
    <div class="history-container">
      <ProcessingHistory
        currentStage={currentStage}
        on:jumpTo={handleHistoryJump}
      />
    </div>
  {/if}

  <!-- Footer: References + Support -->
  <div class="page-footer">
    {#if showRefs}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="refs-backdrop" on:click={() => showRefs = false}></div>
    <div class="refs-panel">
      <div class="ref-section">
        <b>0. SHERLOC instrument and Loupe software</b>
        <div class="ref-cite">Bhartia, R. et al. (2021). Perseverance's Scanning Habitable Environments with Raman and Luminescence for Organics and Chemicals (SHERLOC) Investigation. <i>Space Sci. Rev.</i>, 217, 58. <a href="https://doi.org/10.1007/s11214-021-00812-z" target="_blank">doi:10.1007/s11214-021-00812-z</a></div>
        <div class="ref-cite">Uckert, K. et al. Loupe — SHERLOC data visualization and analysis tool (NASA/JPL). <a href="https://github.com/nasa/Loupe" target="_blank">github.com/nasa/Loupe</a></div>
      </div>
      <div class="ref-section">
        <b>1. Despiking</b> — Rolling-median MAD z-score
        <div class="ref-cite">Whitaker, D.A. &amp; Hayes, K. (2018). A simple algorithm for despiking Raman spectra. <i>Chemometrics Intell. Lab. Syst.</i>, 179, 82–84. <a href="https://doi.org/10.1016/j.chemolab.2018.06.009" target="_blank">doi:10.1016/j.chemolab.2018.06.009</a></div>
      </div>
      <div class="ref-section">
        <b>2. Background</b> — Dark frame subtraction (arm-stowed / fused silica)
        <div class="ref-cite">Jakubek, R.S. et al. (2025). Spectral Background Calibration of SHERLOC Spectrometer Onboard Perseverance Enables Identification of a Ubiquitous Martian Spectral Component. <i>Appl. Spectrosc.</i>, 79, 904–918. <a href="https://doi.org/10.1177/00037028241280081" target="_blank">doi:10.1177/00037028241280081</a></div>
      </div>
      <div class="ref-section">
        <b>3. Baseline</b> — Adaptive smoothness penalized least squares (asPLS)
        <div class="ref-cite">Zhang, F. et al. (2020). Baseline correction for infrared spectra using adaptive smoothness parameter penalized least squares method. <i>Spectroscopy Letters</i>, 53(3), 222–233. <a href="https://doi.org/10.1080/00387010.2020.1730908" target="_blank">doi:10.1080/00387010.2020.1730908</a></div>
        <div class="ref-cite">Implementation: pybaselines (Erb, 2022). <a href="https://doi.org/10.5281/zenodo.5608581" target="_blank">doi:10.5281/zenodo.5608581</a></div>
      </div>
      <div class="ref-section">
        <b>4. Raman Fit</b> — Multi-Gaussian with sequential F-test
        <div class="ref-cite">Branch, M.A., Coleman, T.F. &amp; Li, Y. (1999). A subspace, interior, and conjugate gradient method for large-scale bound-constrained minimization problems. <i>SIAM J. Sci. Comput.</i>, 21(1), 1–23. <a href="https://doi.org/10.1137/S1064827595289108" target="_blank">doi:10.1137/S1064827595289108</a></div>
        <div class="ref-cite">Burnham, K.P. &amp; Anderson, D.R. (2002). <i>Model Selection and Multimodel Inference</i>. Springer.</div>
      </div>
    </div>
    {/if}
    <div class="footer-buttons">
      <button class="footer-btn" on:click={() => showRefs = !showRefs} class:active={showRefs}>References</button>
      <a class="footer-btn" href="https://github.com/archaeon-ai/sherloc-pipeline/issues" target="_blank">Support</a>
    </div>
  </div>
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

  .breadcrumb-sep {
    color: var(--color-text-tertiary);
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--color-primary);
    padding: 0;
    cursor: pointer;
    font-size: inherit;
  }

  .btn-link:hover {
    text-decoration: underline;
  }

  .workbench-layout {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    align-items: start;
  }

  @media (max-width: 1024px) {
    .workbench-layout {
      grid-template-columns: 1fr;
    }
  }

  .workbench-sidebar {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .workbench-main {
    min-width: 0;
  }

  .meta-list {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 10px;
    font-size: 0.83rem;
  }

  .meta-list dt {
    color: var(--color-text-secondary);
    font-weight: 500;
  }

  .meta-list dd {
    margin: 0;
  }

  .view-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .region-selector {
    display: flex;
    gap: 2px;
    background: var(--color-background);
    border-radius: var(--radius-sm);
    padding: 2px;
  }

  .region-btn {
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--color-text-secondary);
    cursor: pointer;
  }

  .region-btn:hover {
    background: var(--color-primary-light);
  }

  .region-btn.active {
    background: var(--color-primary);
    color: white;
  }

  .averaging-controls {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    font-size: 0.8rem;
  }

  .averaging-controls select {
    font-size: 0.8rem;
    padding: 2px 4px;
  }

  .trim-input {
    width: 55px;
    font-size: 0.8rem;
    font-family: var(--font-mono);
    padding: 2px 4px;
    text-align: center;
  }

  .chain-body {
    padding: 12px;
  }

  .undo-controls {
    display: flex;
    gap: 8px;
  }

  .undo-controls button {
    flex: 1;
  }

  .page-footer {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin-top: 12px;
  }

  .footer-buttons {
    display: flex;
    gap: 8px;
  }

  .footer-btn {
    font-size: 0.75rem;
    padding: 4px 12px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface);
    color: var(--color-text-secondary);
    cursor: pointer;
    text-decoration: none;
  }

  .footer-btn:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }

  .footer-btn.active {
    border-color: var(--color-primary);
    color: var(--color-primary);
  }

  .refs-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 99;
  }

  .refs-panel {
    position: absolute;
    bottom: 40px;
    right: 0;
    z-index: 100;
    width: 480px;
    max-height: 400px;
    overflow-y: auto;
    padding: 16px;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    font-size: 0.8rem;
    line-height: 1.6;
  }

  .refs-panel :global(.ref-section) {
    margin-bottom: 10px;
  }

  .refs-panel :global(.ref-section:last-child) {
    margin-bottom: 0;
  }

  .refs-panel :global(.ref-cite) {
    font-size: 0.73rem;
    color: var(--color-text-secondary);
    margin-left: 12px;
  }

  .refs-panel :global(a) {
    color: var(--color-primary);
    text-decoration: none;
  }

  .refs-panel :global(a:hover) {
    text-decoration: underline;
  }

  .history-container {
    margin-top: 16px;
  }

  .aci-body {
    padding: 0;
    aspect-ratio: 1648 / 1200;
  }
</style>
