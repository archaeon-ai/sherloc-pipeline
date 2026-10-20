<script lang="ts">
  import { pinTrace, pinnedTraces, removeTrace, toggleTrace, MAX_PINNED_TRACES } from '../lib/evidenceStore';
  import type { EvidenceTrace } from '../lib/evidenceTypes';

  import type { IdentificationEvidence } from '../lib/identification';
  export let result: IdentificationEvidence | null = null;
  export let onCandidate: (identity: string) => void = () => {};
  export let primary: EvidenceTrace | null = null;
  export let contextKey: string;
  export let pinDisabled = false;
  export let onPin: () => void = () => {};
</script>

<div class="evidence-rail">
  <button class="btn-secondary btn-sm" disabled={pinDisabled || !primary || $pinnedTraces.length >= MAX_PINNED_TRACES || $pinnedTraces.some(item => item.id === primary?.id)}
    on:click={() => { if (primary && pinTrace(primary, contextKey)) onPin(); }}>Pin current trace</button>
  <p>Read-only comparisons: current selection plus up to three pins. Pins stay in this session for this target.</p>
  {#if result}
    <p>Support · provisional threshold {result.method.support_min.toFixed(2)}</p>
    {#each result.ranking.slice(0, 5) as candidate, index}
      <div class="pin"><span>{index === 0 ? 'Leading candidate' : 'Alternative'}: {candidate.species_name} · support {candidate.support.toFixed(3)}</span>
        {#each candidate.trace_revision_ids as identity}
          <button class="btn-secondary btn-sm" on:click={() => onCandidate(identity)}>Add {candidate.species_name} trace</button>
          <small>Trace: {identity} · Citation: {candidate.citations.find(item => item.trace_revision_id === identity)?.citation_text ?? 'missing'}</small>
        {/each}
      </div>
    {/each}
  {/if}
  {#each $pinnedTraces as trace (trace.id)}
    <div class="pin">
      <label><input type="checkbox" checked={trace.visible} on:change={() => toggleTrace(trace.id)} /> {trace.label}</label>
      <button class="btn-secondary btn-sm" aria-label={`Remove ${trace.label}`} on:click={() => removeTrace(trace.id)}>Remove</button>
      <small>Source: {trace.snapshot.trace.provenance.data_origin ?? 'unknown'} · Citation: {String(trace.snapshot.trace.provenance.source?.citation ?? 'missing')}</small>
    </div>
  {/each}
</div>

<style>
  .evidence-rail { padding: 8px 0; overflow-wrap: anywhere; }
  p, small { font-size: 0.75rem; color: var(--color-text-secondary); }
  .pin { display: flex; flex-wrap: wrap; gap: 8px; padding: 6px 0; }
  label { flex: 1; min-width: 160px; font-size: 0.8rem; }
  small { width: 100%; }
</style>
