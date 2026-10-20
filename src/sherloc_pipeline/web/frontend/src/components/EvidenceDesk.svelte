<script lang="ts">
  import type { ProcessingSnapshot } from '../lib/types';
  import { onDestroy } from 'svelte';
  import { appConfig, accessMode } from '../lib/stores';
  import { ApiError, getReferences, getReference, identifyStoredPoint } from '../lib/api';
  import { referenceEvidence } from '../lib/identification';
  import { identificationPlot } from '../lib/identification';
  import type { IdentificationEvidence, ReferencePage } from '../lib/identification';
  import type { EvidenceTrace } from '../lib/evidenceTypes';
  import type { EvidenceDomain, EvidenceTheme } from '../lib/evidenceTypes';
  import { comparisonDisplays, evidenceColor, evidenceTrace, pinnedTraces } from '../lib/evidenceStore';
  import RamanView from './RamanView.svelte';
  import EvidenceRail from './EvidenceRail.svelte';
  import ExportMenu from './ExportMenu.svelte';

  export let snapshot: ProcessingSnapshot | null = null;
  export let contextKey: string;
  export let showBadpix = false;
  let view = 'processing';
  let theme: EvidenceTheme = 'system';
  let plot: RamanView;
  let result: IdentificationEvidence | null = null;
  let reference: EvidenceTrace | null = null;
  let page: ReferencePage | null = null;
  let chosenReference = '';
  let busy = false;
  let message = '';
  let generation = 0;
  let stateKey = '';
  $: referencesEnabled = $accessMode === 'internal' && $appConfig?.tier === 'team' && $appConfig?.capabilities?.references === true;
  $: identifyEnabled = referencesEnabled && $appConfig?.capabilities?.identification === true && $appConfig?.capabilities?.experimental_candidates === true;
  $: selectionKey = JSON.stringify([contextKey, snapshot?.trace, snapshot?.stage, snapshot?.processingChain]);
  $: policyKey = JSON.stringify([$accessMode, $appConfig?.tier, $appConfig?.capabilities, $appConfig?.evidence_policy_digest, $appConfig?.reference_release_id]);
  $: resetFor(selectionKey + policyKey);
  $: identifyAvailable = identifyEnabled && domain === 'raman_shift' && !unsupportedRegion;
  $: canIdentify = snapshot?.trace?.selection.mode === 'point' && snapshot?.stage === 'raw' &&
    !snapshot?.processingChain?.length && !snapshot?.trace?.provenance.processing_chain?.some(step => step.step === 'despike') &&
    !!snapshot?.trace?.provenance.source_digest;
  $: comparisons = [...$pinnedTraces, ...(reference && referencesEnabled ? [reference] : [])];

  function resetFor(key: string) {
    if (stateKey === key) return;
    stateKey = key; generation += 1;
    result = null; reference = null; page = null; message = ''; busy = false; chosenReference = '';
  }
  onDestroy(() => { generation += 1; });

  async function loadReferences(cursor?: string) {
    const release = $appConfig?.reference_release_id;
    if (!referencesEnabled || !release) return;
    const token = generation;
    try {
      const loaded = await getReferences(release, cursor);
      if (token !== generation) return;
      page = loaded;
      chosenReference = page.items[0]?.trace_revision_id ?? '';
    } catch { if (token === generation) message = 'Reference request failed.'; }
  }

  async function addReference(identity: string, release = $appConfig?.reference_release_id) {
    if (!referencesEnabled || !release || !identity) return;
    const token = generation;
    try {
      const loaded = await getReference(release, identity);
      if (token !== generation || loaded.release_id !== release || loaded.trace_revision_id !== identity) return;
      reference = referenceEvidence(loaded);
      if (reference) view = reference.nativeDomain === 'emission_wavelength' ? 'emission' : 'overlay';
    } catch { if (token === generation) message = 'Reference request failed.'; }
  }

  async function identify() {
    if (!identifyAvailable || !canIdentify || busy || !snapshot?.trace) return;
    const trace = snapshot.trace;
    const token = generation;
    const sourceDigest = trace.provenance.source_digest;
    busy = true; message = ''; result = null;
    try {
      const release = $appConfig?.reference_release_id;
      if (!release) throw new Error('Missing reference release');
      const metadata = await getReferences(release);
      if (token !== generation) return;
      const binding = metadata.binding;
      if (!binding || binding.reference_release_id !== release) throw new Error('Pinned method unavailable');
      const loaded = await identifyStoredPoint({ ...binding, scan_id: trace.source.scan_id,
        selection: { mode: 'point', index: (trace.selection.indices as number[])[0] }, region: 'R1',
        spectrum_type: trace.provenance.source_processing?.spectrum_type, expected_source_digest: sourceDigest });
      if (token !== generation) return;
      if (loaded.source_digest !== sourceDigest || loaded.reference_release_id !== binding.reference_release_id ||
          loaded.reference_subset_id !== binding.reference_subset_id || loaded.method.release_id !== binding.method_release_id) {
        message = 'Stale identification rejected. Reload the stored selection.'; return;
      }
      result = loaded;
    } catch (error) {
      if (token === generation) message = error instanceof ApiError && error.status === 409
        ? 'Source changed. Reload the stored selection.' : 'Identification execution failed. No scientific result was produced.';
    } finally { if (token === generation) busy = false; }
  }
  $: primary = evidenceTrace(snapshot, {
    spike_mask: $$restProps.spikeMask ?? null, spike_method: $$restProps.spikeMethod ?? null,
    badpix: showBadpix ? $$restProps.badpix : null,
  });
  $: unsupportedRegion = snapshot?.trace?.region === 'R123';
  $: nativeDomain = primary?.nativeDomain ?? (
    ($$restProps.region ?? snapshot?.trace?.region ?? 'R1') !== 'R1' && snapshot?.trace?.wavelength?.length
      ? 'emission_wavelength' : snapshot?.raman.wavenumber.length ? 'raman_shift' : 'emission_wavelength');
  $: domain = (view === 'emission' ? 'emission_wavelength' : view === 'processing' ? nativeDomain : 'raman_shift') as EvidenceDomain;
  $: displays = comparisonDisplays(primary, comparisons, domain, view === 'stacked').map(item => ({ ...item, trace: evidenceColor(item.trace, theme) }));
  $: committedSelection = snapshot?.trace?.selection;
  $: committedMode = (committedSelection?.mode ?? 'average') as 'average' | 'subset' | 'point';
  $: committedIndices = Array.isArray(committedSelection?.indices) ? committedSelection.indices as number[] : [];
  $: committedAverage = snapshot?.trace?.provenance.averaging_method ?? 'unknown';
  $: comparing = !unsupportedRegion && view !== 'processing' && view !== 'identification';
  $: methodPlot = identificationPlot(result);
  $: if (view === 'identification' && !methodPlot) view = 'processing';
  $: if (unsupportedRegion) view = 'processing';
</script>

<section aria-label="Evidence desk">
  <div class="evidence-controls">
    <label>View <select bind:value={view}>
      <option value="processing">Processing / fit and residual</option>
      {#if methodPlot}<option value="identification">{methodPlot.label} and residual</option>{/if}
      <option value="overlay" disabled={unsupportedRegion}>Raman overlay</option>
      <option value="stacked" disabled={unsupportedRegion}>Raman stacked</option>
      <option value="emission" disabled={unsupportedRegion}>Luminescence (nm)</option>
    </select></label>
    <details><summary>Traces</summary>
      <EvidenceRail {primary} {contextKey} pinDisabled={unsupportedRegion} onPin={() => view = nativeDomain === 'emission_wavelength' ? 'emission' : 'overlay'}
        {result} onCandidate={(identity) => addReference(identity, result?.reference_release_id)} />
      {#if referencesEnabled}
        <button class="btn-secondary btn-sm" on:click={() => loadReferences()}>Browse references</button>
        {#if page}
          <label>Permitted reference <select bind:value={chosenReference}>
            {#each page.items as item}<option value={item.trace_revision_id}>{item.provenance.species_name ?? item.provenance.source_label}</option>{/each}
          </select></label>
          <button class="btn-secondary btn-sm" disabled={!chosenReference} on:click={() => addReference(chosenReference)}>Add reference</button>
          {#if page.next_cursor}<button class="btn-secondary btn-sm" on:click={() => loadReferences(page?.next_cursor ?? undefined)}>Next references</button>{/if}
        {/if}
        {#if reference}<p>{reference.label} · Citation: {String(reference.snapshot.trace.provenance.source?.citation ?? 'missing')}</p>
          <button class="btn-secondary btn-sm" on:click={() => reference = null}>Remove reference</button>{/if}
      {/if}
    </details>
    {#if identifyAvailable}<button class="btn-secondary btn-sm" disabled={!canIdentify || busy} on:click={identify}>{busy ? 'Identifying…' : 'Identify'}</button>{/if}
    <label>Theme <select bind:value={theme}>
      <option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option>
    </select></label>
    <ExportMenu identification={result} evidenceDesk={true} trace={snapshot?.trace ?? null} stage={snapshot?.stage ?? 'raw'}
      wavenumber={snapshot?.raman.wavenumber ?? []} intensity={snapshot?.raman.intensity ?? []}
      artifacts={snapshot?.artifacts ?? null} processingChain={snapshot?.processingChain ?? []}
      spikeMask={$$restProps.spikeMask ?? null} spikeMethod={$$restProps.spikeMethod ?? null}
      badpix={showBadpix ? $$restProps.badpix : null}
      evidenceMode={comparing} evidenceDisplays={comparing ? displays : []}
      exportFigure={(format) => plot?.downloadAs(format)} />
  </div>
  {#if identifyAvailable && !canIdentify}<p class="display-note">Identify requires one raw stored point without manual processing. The frozen recipe uses that source selection.</p>{/if}
  {#if message}<p role="alert">{message}</p>{/if}
  {#if result}<p class="display-note">{result.status === 'abstained' ? 'Scientific abstention' : 'Provisional candidate support'} · provisional threshold {result.method.support_min.toFixed(2)}
    {#each result.reasons as reason}<span>{reason.replaceAll('_', ' ')}</span>{/each}</p>{/if}
  {#if unsupportedRegion}<p class="display-note">R123 comparison and pinning require an authoritative native domain.</p>{/if}
  {#if comparing}
    <p class="display-note">Display only: native maximum in {domain === 'raman_shift' ? '800–1800 cm⁻¹' : '280–350 nm'}; fixed while zooming.
      {view === 'stacked' ? 'Stacked offsets: 1.25 per trace.' : ''}
      {#each displays.filter(item => !item.scalable) as item}<span>{item.trace.label}: unscalable; native values retained.</span>{/each}
      {#if !displays.length}<span>No compatible native axis for this view.</span>{/if}
    </p>
  {/if}
  <RamanView identification={result} showIdentificationFit={view === 'identification'} evidenceDesk={true} bind:this={plot} {...$$restProps} xDomain={domain} traceIdentity={primary?.id ?? ''}
    selectionMode={committedMode} selectedIndices={committedIndices}
    selectedPointIdx={committedMode === 'point' ? committedIndices[0] ?? null : null}
    nPoints={Number(committedSelection?.n_points ?? committedIndices.length)} averagingMethod={committedAverage}
    processingTrace={primary} bind:showBadpix evidenceMode={comparing}
    comparisonTraces={comparing ? comparisons : []} primaryTrace={comparing ? primary : null}
    stacked={view === 'stacked'} {theme} on:viewRangeChange />
</section>

<style>
  .evidence-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: start; margin-bottom: 8px; }
  label, summary { font-size: 0.8rem; }
  select { max-width: 100%; }
  details { flex: 1; min-width: 110px; }
  summary { cursor: pointer; padding: 4px; }
  .display-note { font-size: 0.75rem; color: var(--color-text-secondary); }
  .display-note span { display: block; }
  @media (max-width: 600px) { .evidence-controls > label { width: 100%; } }
</style>
