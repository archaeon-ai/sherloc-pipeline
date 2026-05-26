<script lang="ts">
  import { onMount } from 'svelte';
  import { navigate } from '../lib/stores';
  import { getScans } from '../lib/api';
  import type { ScanListItem, ScanFilterParams } from '../lib/types';

  let scans: ScanListItem[] = [];
  let total = 0;
  let loading = false;
  let error = '';
  // Per spec §12.2: backend sets `message` only on the unfiltered
  // empty-DB response. When present we render the onboarding panel
  // in place of the standard table.
  let emptyDbMessage: string | null = null;

  // Filters
  let filterSol: string = '';
  let filterTarget: string = '';
  let filterScanClass: string = '';
  let filterScanType: string = '';
  let filterProcessingStatus: string = '';
  let offset = 0;
  let limit = 50;

  $: totalPages = Math.ceil(total / limit);
  $: currentPage = Math.floor(offset / limit) + 1;

  onMount(() => {
    // Read optional sol query param from hash (e.g. #/?sol=921)
    const hashQuery = window.location.hash.replace(/^#\/?/, '');
    const params = new URLSearchParams(hashQuery);
    if (params.has('sol')) {
      filterSol = params.get('sol') ?? '';
    }
    fetchScans();
  });

  async function fetchScans() {
    loading = true;
    error = '';
    try {
      const params: ScanFilterParams = { offset, limit };
      if (filterSol) params.sol = parseInt(filterSol, 10);
      if (filterTarget) params.target = filterTarget;
      if (filterScanClass) params.scan_class = filterScanClass;
      if (filterScanType) params.scan_type = filterScanType;
      if (filterProcessingStatus) params.processing_status = filterProcessingStatus;

      const res = await getScans(params);
      scans = res.scans;
      total = res.total;
      emptyDbMessage = res.message ?? null;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load scans';
    } finally {
      loading = false;
    }
  }

  function applyFilters() {
    offset = 0;
    fetchScans();
  }

  function clearFilters() {
    filterSol = '';
    filterTarget = '';
    filterScanClass = '';
    filterScanType = '';
    filterProcessingStatus = '';
    offset = 0;
    fetchScans();
  }

  function goPage(page: number) {
    offset = (page - 1) * limit;
    fetchScans();
  }

  function openScan(scanId: string) {
    navigate(`#/scan/${scanId}/workbench`);
  }

  function statusBadge(status: string | null): string {
    if (status === 'completed') return 'badge-success';
    if (status === 'failed') return 'badge-error';
    return 'badge-neutral';
  }
</script>

<div class="page-container">
  <h1 class="page-title">Scan Browser</h1>
  <p class="page-subtitle">Browse and filter SHERLOC measurement scans</p>

  <!-- Filters -->
  <div class="card" style="margin-bottom: 16px">
    <div class="card-body">
      <div class="filter-row">
        <div class="filter-field">
          <label for="f-sol">Sol</label>
          <input id="f-sol" type="number" placeholder="e.g. 921" bind:value={filterSol} style="width: 100px" />
        </div>
        <div class="filter-field">
          <label for="f-target">Target</label>
          <input id="f-target" type="text" placeholder="Search target..." bind:value={filterTarget} style="width: 180px" />
        </div>
        <div class="filter-field">
          <label for="f-class">Scan Class</label>
          <select id="f-class" bind:value={filterScanClass}>
            <option value="">All</option>
            <option value="primary">Primary</option>
            <option value="sub_scan">Sub-scan</option>
            <option value="composite">Composite</option>
          </select>
        </div>
        <div class="filter-field">
          <label for="f-type">Scan Type</label>
          <select id="f-type" bind:value={filterScanType}>
            <option value="">All</option>
            <option value="detail">Detail</option>
            <option value="line">Line</option>
            <option value="hdr">HDR</option>
            <option value="survey">Survey</option>
          </select>
        </div>
        <div class="filter-field">
          <label for="f-status">Status</label>
          <select id="f-status" bind:value={filterProcessingStatus}>
            <option value="">All</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="null">Unprocessed</option>
          </select>
        </div>
        <div class="filter-actions">
          <button class="btn-primary" on:click={applyFilters}>Filter</button>
          <button class="btn-secondary" on:click={clearFilters}>Clear</button>
        </div>
      </div>
    </div>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if emptyDbMessage}
    <!-- Empty-DB onboarding panel — spec §12.2. Only rendered when the
         backend reports an unfiltered empty database. The standard
         table is hidden in this state since there is nothing to show. -->
    <div class="card onboarding-card">
      <div class="card-body">
        <h2 class="onboarding-title">No data ingested yet</h2>
        <p class="onboarding-body">{emptyDbMessage}</p>
        <p class="onboarding-body">
          To get started:
        </p>
        <ol class="onboarding-list">
          <li>
            Run <code>sherloc init</code> to create the database schema.
          </li>
          <li>
            For PDS-only workflows, run
            <code>sherloc pds-download --sol &lt;n&gt;</code> followed by
            <code>sherloc pds-ingest --sol &lt;n&gt;</code> to load a sol.
          </li>
          <li>
            Reload this page once ingestion completes.
          </li>
        </ol>
        <p class="onboarding-body">
          See <code>sherloc init --help</code> and
          <code>sherloc --help</code> for full options.
        </p>
      </div>
    </div>
  {:else}

  <!-- Results -->
  <div class="card">
    <div class="card-header flex items-center justify-between">
      <span>{total.toLocaleString()} scan{total !== 1 ? 's' : ''}</span>
      {#if loading}
        <span class="spinner"></span>
      {/if}
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Sol</th>
            <th>Target</th>
            <th>Scan Name</th>
            <th>Points</th>
            <th>Class</th>
            <th>Type</th>
            <th>Status</th>
            <th>Processed</th>
          </tr>
        </thead>
        <tbody>
          {#each scans as scan}
            <tr class="clickable" on:click={() => openScan(scan.id)}>
              <td class="mono">{scan.sol_number}</td>
              <td>{scan.target ?? '--'}</td>
              <td class="mono">{scan.scan_name}</td>
              <td class="mono">{scan.n_points}</td>
              <td><span class="badge badge-neutral">{scan.scan_class}</span></td>
              <td>{scan.scan_type ?? '--'}</td>
              <td>
                <span class="badge {statusBadge(scan.processing_status)}">
                  {scan.processing_status ?? 'unprocessed'}
                </span>
              </td>
              <td class="mono" style="font-size: 0.8rem; color: var(--color-text-secondary)">
                {scan.processed_at ? new Date(scan.processed_at).toLocaleDateString() : '--'}
              </td>
            </tr>
          {:else}
            <tr>
              <td colspan="8" class="empty-state">
                {loading ? 'Loading...' : 'No scans found'}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    {#if totalPages > 1}
      <div class="pagination">
        <button
          class="btn-secondary btn-sm"
          disabled={currentPage <= 1}
          on:click={() => goPage(currentPage - 1)}
        >
          Previous
        </button>
        <span class="mono" style="font-size: 0.85rem; color: var(--color-text-secondary)">
          Page {currentPage} of {totalPages}
        </span>
        <button
          class="btn-secondary btn-sm"
          disabled={currentPage >= totalPages}
          on:click={() => goPage(currentPage + 1)}
        >
          Next
        </button>
      </div>
    {/if}
  </div>
  {/if}
</div>

<style>
  .filter-row {
    display: flex;
    align-items: flex-end;
    gap: 12px;
    flex-wrap: wrap;
  }

  .filter-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .filter-actions {
    display: flex;
    gap: 8px;
    padding-bottom: 1px;
  }

  .table-scroll {
    overflow-x: auto;
  }

  .clickable {
    cursor: pointer;
  }

  .pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 12px;
    border-top: 1px solid var(--color-border);
  }

  .onboarding-card {
    border-left: 4px solid var(--color-primary, #2563eb);
  }

  .onboarding-title {
    margin: 0 0 12px 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .onboarding-body {
    margin: 8px 0;
    line-height: 1.5;
    color: var(--color-text-secondary);
  }

  .onboarding-list {
    margin: 8px 0 8px 24px;
    line-height: 1.7;
    color: var(--color-text-secondary);
  }

  .onboarding-list code,
  .onboarding-body code {
    font-family: var(--font-mono, ui-monospace, monospace);
    font-size: 0.9em;
    padding: 2px 6px;
    background: var(--color-bg-secondary, #f1f5f9);
    border-radius: 3px;
  }
</style>
