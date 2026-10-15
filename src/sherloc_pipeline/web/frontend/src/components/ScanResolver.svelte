<script lang="ts">
  import { onMount } from 'svelte';
  import { getScans, getScanPoints } from '../lib/api';
  import { currentHash } from '../lib/stores';
  import { parseScanLink, validateLinkSelection, workbenchLink } from '../lib/scanLink';
  import type { ScanListItem } from '../lib/types';

  export let queryParams: Record<string, string> = {};

  let matches: ScanListItem[] = [];
  let loading = true;
  let error = '';
  let link: ReturnType<typeof parseScanLink>;
  let active = true;

  onMount(() => {
    resolve();
    return () => { active = false; };
  });

  async function resolve() {
    try {
      link = parseScanLink(queryParams);
      // Read every page so duplicates beyond the first page remain visible.
      let offset = 0;
      do {
        const result = await getScans({ scan_name: link.scanName, sol: link.sol, offset, limit: 200 });
        if (!active) return;
        matches = [...matches, ...result.scans];
        offset += result.scans.length;
        if (offset >= result.total) break;
        if (!result.scans.length) throw new Error('The scan list changed. Reload this link to try again.');
      } while (active);

      if (matches.length === 0) {
        error = `No scan found for "${link.scanName}" on sol ${link.sol} in this tier.`;
      } else if (matches.length === 1) {
        await openScan(matches[0]);
      }
    } catch (e) {
      if (active) error = e instanceof Error ? e.message : 'Unable to look up this scan.';
    } finally {
      if (active) loading = false;
    }
  }

  async function openScan(scan: ScanListItem) {
    loading = true;
    error = '';
    try {
      if (link.selection.mode !== 'average') {
        const result = await getScanPoints(scan.id);
        if (!active) return;
        validateLinkSelection(link.selection, result.points);
      }
      if (!active) return;
      const hash = workbenchLink(scan.id, link.selection);
      // Replace the resolver entry so Back cannot immediately redirect again.
      window.history.replaceState(null, '', hash);
      currentHash.set(hash);
    } catch (e) {
      if (active) error = e instanceof Error ? e.message : 'Unable to open this scan.';
    } finally {
      if (active) loading = false;
    }
  }
</script>

<div class="page-container">
  <h1>Open in SHERLOC Pipeline</h1>
  {#if loading}<p role="status">Looking up scan…</p>{/if}
  {#if error}<div class="error-banner" role="alert">{error}</div>{/if}
  {#if matches.length > 1}
    <p>More than one scan matches this name and sol. Choose a scan to open:</p>
    <ul>
      {#each matches as scan (scan.id)}
        <li>
          <button class="btn-secondary" disabled={loading} on:click={() => openScan(scan)}>
            {scan.target ?? 'Unnamed target'} — {scan.scan_name}, sol {scan.sol_number}
            ({scan.n_points} points; {scan.scan_class}; {scan.scan_type ?? 'unknown type'})
            — {scan.scan_id} [ID: {scan.id}]
          </button>
        </li>
      {/each}
    </ul>
  {/if}
  <p><a href="#/">Back to scan browser</a></p>
</div>

<style>
  li { margin-bottom: 0.75rem; }
</style>
