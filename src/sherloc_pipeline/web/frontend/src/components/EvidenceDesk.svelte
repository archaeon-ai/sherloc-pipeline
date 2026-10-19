<script lang="ts">
  import type { ProcessingSnapshot } from '../lib/types';
  import type { EvidenceDomain, EvidenceTheme } from '../lib/evidenceTypes';
  import { comparisonDisplays, evidenceTrace, pinnedTraces } from '../lib/evidenceStore';
  import RamanView from './RamanView.svelte';
  import EvidenceRail from './EvidenceRail.svelte';
  import ExportMenu from './ExportMenu.svelte';

  export let snapshot: ProcessingSnapshot | null = null;
  export let contextKey: string;
  export let showBadpix = false;
  let view = 'processing';
  let theme: EvidenceTheme = 'system';
  let plot: RamanView;
  $: primary = evidenceTrace(snapshot, {
    spike_mask: $$restProps.spikeMask ?? null, spike_method: $$restProps.spikeMethod ?? null,
    badpix: showBadpix ? $$restProps.badpix : null,
  });
  $: nativeDomain = snapshot?.trace?.wavelength?.length && snapshot.trace.provenance.region && snapshot.trace.provenance.region !== 'R1'
    ? 'emission_wavelength' : snapshot?.raman.wavenumber.length ? 'raman_shift' : 'emission_wavelength';
  $: domain = (view === 'emission' ? 'emission_wavelength' : view === 'processing' ? nativeDomain : 'raman_shift') as EvidenceDomain;
  $: displays = comparisonDisplays(primary, $pinnedTraces, domain, view === 'stacked');
  $: committedSelection = snapshot?.trace?.selection;
  $: committedMode = (committedSelection?.mode ?? 'average') as 'average' | 'subset' | 'point';
  $: committedIndices = Array.isArray(committedSelection?.indices) ? committedSelection.indices as number[] : [];
  $: committedAverage = snapshot?.trace?.provenance.averaging_method ?? 'unknown';
  $: comparing = view !== 'processing';
</script>

<section aria-label="Evidence desk">
  <div class="evidence-controls">
    <label>View <select bind:value={view}>
      <option value="processing">Processing / fit and residual</option>
      <option value="overlay">Raman overlay</option>
      <option value="stacked">Raman stacked</option>
      <option value="emission">Luminescence (nm)</option>
    </select></label>
    <details><summary>Traces</summary>
      <EvidenceRail {primary} {contextKey} onPin={() => view = nativeDomain === 'emission_wavelength' ? 'emission' : 'overlay'} />
    </details>
    <label>Theme <select bind:value={theme}>
      <option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option>
    </select></label>
    <ExportMenu trace={snapshot?.trace ?? null} stage={snapshot?.stage ?? 'raw'}
      wavenumber={snapshot?.raman.wavenumber ?? []} intensity={snapshot?.raman.intensity ?? []}
      artifacts={snapshot?.artifacts ?? null} processingChain={snapshot?.processingChain ?? []}
      spikeMask={$$restProps.spikeMask ?? null} spikeMethod={$$restProps.spikeMethod ?? null}
      badpix={showBadpix ? $$restProps.badpix : null}
      evidenceMode={comparing} evidenceDisplays={comparing ? displays : []}
      exportFigure={(format) => plot?.downloadAs(format)} />
  </div>
  {#if comparing}
    <p class="display-note">Display only: native maximum in {domain === 'raman_shift' ? '800–1800 cm⁻¹' : '280–350 nm'}; fixed while zooming.
      {view === 'stacked' ? 'Stacked offsets: 1.25 per trace.' : ''}
      {#each displays.filter(item => !item.scalable) as item}<span>{item.trace.label}: unscalable; native values retained.</span>{/each}
      {#if !displays.length}<span>No compatible native axis for this view.</span>{/if}
    </p>
  {/if}
  <RamanView bind:this={plot} {...$$restProps} xDomain={domain} traceIdentity={primary?.id ?? ''}
    selectionMode={committedMode} selectedIndices={committedIndices}
    selectedPointIdx={committedMode === 'point' ? committedIndices[0] ?? null : null}
    nPoints={Number(committedSelection?.n_points ?? committedIndices.length)} averagingMethod={committedAverage}
    processingTrace={primary} bind:showBadpix evidenceMode={comparing}
    comparisonTraces={comparing ? $pinnedTraces : []} primaryTrace={comparing ? primary : null}
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
