<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { navigate, accessMode } from '../lib/stores';
  import { getPdsAvailableSols, postPdsDownload } from '../lib/api';
  import { ApiError } from '../lib/api';
  import { JobWebSocket, createPollingFallback } from '../lib/websocket';
  import type { PdsSol, PdsAvailableResponse, WsMessage } from '../lib/types';

  let available: PdsAvailableResponse | null = null;
  let loading = false;
  let error = '';
  let ingestingSol: number | null = null;
  let ingestJobId: string | null = null;
  let ingestProgress = 0;
  let ingestStatus = '';
  let ingestMessage = '';
  let ingestPhase = '';
  let ingestResult: Record<string, unknown> | null = null;
  let lastDownloadedSol: number | null = null;
  let ws: JobWebSocket | null = null;
  let pollingStop: (() => void) | null = null;

  let filterText = '';
  let filterIngested: 'all' | 'ingested' | 'not_ingested' = 'all';

  // Sort state: default descending by sol number
  let sortAsc = false;

  // Force-reingest toggle: tracks which sol (if any) has the toggle active
  let forceReingestSol: number | null = null;

  $: ingestedSet = new Set(available?.already_ingested ?? []);

  $: filteredSols = (() => {
    if (!available) return [];
    let sols = available.sols.filter((s) => {
      if (filterText && !String(s.sol).includes(filterText)) return false;
      if (filterIngested === 'ingested' && !ingestedSet.has(s.sol)) return false;
      if (filterIngested === 'not_ingested' && ingestedSet.has(s.sol)) return false;
      return true;
    });
    sols = [...sols].sort((a, b) => sortAsc ? a.sol - b.sol : b.sol - a.sol);
    return sols;
  })();

  onMount(() => {
    loadSols();
  });

  onDestroy(() => {
    ws?.close();
    pollingStop?.();
  });

  async function loadSols() {
    loading = true;
    error = '';
    try {
      available = await getPdsAvailableSols();
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Failed to load PDS data';
      }
    } finally {
      loading = false;
    }
  }

  async function downloadSol(sol: number, force: boolean) {
    lastDownloadedSol = sol;
    ingestingSol = sol;
    ingestProgress = 0;
    ingestStatus = 'starting';
    ingestMessage = '';
    ingestPhase = '';
    ingestResult = null;
    forceReingestSol = null;
    error = '';

    try {
      const res = await postPdsDownload({ sol, force_reingest: force });
      ingestJobId = res.job_id;
      ingestStatus = res.status;

      // Connect WebSocket
      try {
        ws = new JobWebSocket(
          res.job_id,
          (msg: WsMessage) => {
            if (msg.type === 'progress') {
              ingestProgress = Math.round(msg.progress * 100);
              ingestMessage = msg.message;
              ingestPhase = msg.phase;
              ingestStatus = 'running';
            } else if (msg.type === 'complete') {
              ingestStatus = 'completed';
              ingestProgress = 100;
              ingestResult = msg.result as unknown as Record<string, unknown>;
              ingestingSol = null;
              loadSols();
            } else if (msg.type === 'error') {
              ingestStatus = 'failed';
              ingestMessage = msg.error;
              ingestingSol = null;
            } else if (msg.type === 'heartbeat') {
              // No-op, connection is alive
            }
          },
          () => {
            if (ingestStatus !== 'completed' && ingestStatus !== 'failed' && ingestJobId) {
              startPolling(ingestJobId);
            }
          },
        );
      } catch {
        if (ingestJobId) startPolling(ingestJobId);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        error = e.message;
      } else {
        error = 'Failed to start download';
      }
      ingestingSol = null;
    }
  }

  function startPolling(jobId: string) {
    pollingStop?.();
    pollingStop = createPollingFallback(jobId, (data: unknown) => {
      const d = data as Record<string, unknown>;
      const type = d.type as string;
      if (type === 'progress') {
        ingestStatus = 'running';
        ingestProgress = Math.round(((d.progress as number) ?? 0) * 100);
        ingestMessage = (d.message as string) ?? '';
        ingestPhase = (d.phase as string) ?? '';
      } else if (type === 'complete') {
        ingestStatus = 'completed';
        ingestProgress = 100;
        ingestResult = d.result as Record<string, unknown>;
        ingestingSol = null;
        loadSols();
      } else if (type === 'error') {
        ingestStatus = 'failed';
        ingestMessage = (d.error as string) ?? 'Unknown error';
        ingestingSol = null;
      }
    });
  }

  function goToScanBrowser(sol: number) {
    navigate(`#/?sol=${sol}`);
  }

  function handleSolClick(sol: number) {
    if (ingestedSet.has(sol)) {
      navigate(`#/?sol=${sol}`);
    } else {
      // Trigger download instead of navigating to empty browser
      downloadSol(sol, false);
    }
  }

  function toggleSortSol() {
    sortAsc = !sortAsc;
  }

  function toggleForceReingest(sol: number) {
    forceReingestSol = forceReingestSol === sol ? null : sol;
  }
</script>

<div class="page-container">
  <h1 class="page-title">PDS Browser</h1>
  <p class="page-subtitle">Browse and ingest SHERLOC data from the Planetary Data System</p>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if ingestStatus === 'running' || ingestStatus === 'starting'}
    <div class="card" style="margin-bottom: 16px">
      <div class="card-header flex items-center justify-between">
        <span>Downloading Sol {ingestingSol}</span>
        <span class="badge badge-info">{ingestStatus}</span>
      </div>
      <div class="card-body">
        <div class="progress-bar">
          <div class="progress-bar-fill" style="width: {ingestProgress}%"></div>
        </div>
        <div class="mono" style="margin-top: 8px; font-size: 0.8rem; color: var(--color-text-secondary)">
          {ingestProgress}% {ingestPhase ? `[${ingestPhase}]` : ''} {ingestMessage ? `— ${ingestMessage}` : ''}
        </div>
      </div>
    </div>
  {/if}

  {#if ingestResult}
    <div class="card" style="margin-bottom: 16px; border-color: var(--color-success)">
      <div class="card-header" style="color: var(--color-success)">Ingestion Complete</div>
      <div class="card-body mono" style="font-size: 0.85rem">
        {#if ingestResult.n_scans !== undefined}
          {ingestResult.n_scans} scans, {ingestResult.n_spectra} spectra ingested
          {#if ingestResult.n_aci > 0}, {ingestResult.n_aci} ACI images{/if}
        {/if}
        {#if ingestResult.warnings && ingestResult.warnings.length > 0}
          <div style="margin-top: 4px; color: var(--color-warning)">
            {ingestResult.warnings.length} warning(s): {ingestResult.warnings.join('; ')}
          </div>
        {/if}
      </div>
    </div>
  {/if}

  {#if ingestStatus === 'failed'}
    <div class="card" style="margin-bottom: 16px; border-color: var(--color-error)">
      <div class="card-header" style="color: var(--color-error)">Download Failed</div>
      <div class="card-body">
        <p class="mono" style="font-size: 0.85rem">{ingestMessage}</p>
        <button class="btn-sm btn-primary" style="margin-top: 8px" on:click={() => { if (lastDownloadedSol !== null) downloadSol(lastDownloadedSol, false); }}>
          Retry
        </button>
      </div>
    </div>
  {/if}

  <div class="card">
    <div class="card-header flex items-center justify-between">
      <span>
        {available ? `${available.total} sols available` : 'Loading...'}
        {#if available}
          ({available.already_ingested.length} ingested)
        {/if}
      </span>
      <div class="flex gap-2 items-center">
        <input
          type="text"
          placeholder="Filter by sol..."
          bind:value={filterText}
          style="width: 140px"
        />
        <select bind:value={filterIngested} style="width: 140px">
          <option value="all">All</option>
          <option value="ingested">Ingested</option>
          <option value="not_ingested">Not Ingested</option>
        </select>
        <button class="btn-secondary btn-sm" on:click={loadSols} disabled={loading}>
          {#if loading}<span class="spinner"></span>{:else}Refresh{/if}
        </button>
      </div>
    </div>
    <div class="table-scroll" style="max-height: 600px; overflow-y: auto">
      <table>
        <thead>
          <tr>
            <th>
              <button class="sort-btn" on:click={toggleSortSol}>
                Sol {sortAsc ? '▲' : '▼'}
              </button>
            </th>
            <th>Scans</th>
            <th>Size (MB)</th>
            <th>Status</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {#each filteredSols as sol}
            <tr>
              <td class="mono">
                <button class="sol-link" on:click={() => handleSolClick(sol.sol)}>
                  {sol.sol}
                </button>
              </td>
              <td class="mono">{sol.n_scans}</td>
              <td class="mono">{sol.data_volume_mb?.toFixed(1) ?? '--'}</td>
              <td>
                {#if ingestedSet.has(sol.sol)}
                  <span class="badge badge-success">ingested</span>
                {:else}
                  <span class="badge badge-neutral">available</span>
                {/if}
              </td>
              <td>
                {#if $accessMode === 'public'}
                  <span class="badge badge-neutral" title="Downloads disabled in public mode">view only</span>
                {:else if ingestedSet.has(sol.sol)}
                  <div class="action-cell">
                    <label class="force-toggle">
                      <input
                        type="checkbox"
                        checked={forceReingestSol === sol.sol}
                        on:change={() => toggleForceReingest(sol.sol)}
                        disabled={ingestingSol !== null}
                      />
                      Force re-ingest
                    </label>
                    {#if forceReingestSol === sol.sol}
                      <button
                        class="btn-sm btn-primary"
                        disabled={ingestingSol !== null}
                        on:click={() => downloadSol(sol.sol, true)}
                      >
                        Re-ingest
                      </button>
                    {:else}
                      <button
                        class="btn-sm btn-secondary"
                        disabled={ingestingSol !== null}
                        on:click={() => downloadSol(sol.sol, false)}
                      >
                        Re-ingest
                      </button>
                    {/if}
                  </div>
                {:else}
                  <button
                    class="btn-sm btn-primary"
                    disabled={ingestingSol !== null}
                    on:click={() => downloadSol(sol.sol, false)}
                  >
                    Download & Ingest
                  </button>
                {/if}
              </td>
            </tr>
          {:else}
            <tr>
              <td colspan="5" class="empty-state">
                {loading ? 'Loading...' : 'No sols found'}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .table-scroll {
    overflow-x: auto;
  }

  .sort-btn {
    background: none;
    border: none;
    cursor: pointer;
    font-weight: inherit;
    font-size: inherit;
    color: inherit;
    padding: 0;
    text-decoration: underline dotted;
  }

  .sort-btn:hover {
    color: var(--color-primary, #4a9eff);
  }

  .sol-link {
    background: none;
    border: none;
    cursor: pointer;
    font-family: inherit;
    font-size: inherit;
    color: var(--color-primary, #4a9eff);
    padding: 0;
    text-decoration: underline;
  }

  .sol-link:hover {
    opacity: 0.8;
  }

  .action-cell {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .force-toggle {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    cursor: pointer;
    white-space: nowrap;
  }

  .force-toggle input[type='checkbox'] {
    cursor: pointer;
  }
</style>
