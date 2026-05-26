<script lang="ts">
  import type { Peak } from '../lib/types';

  export let peaks: Peak[] = [];
  export let showExport: boolean = true;

  type SortKey = 'center_cm1' | 'amplitude' | 'snr' | 'fwhm_cm1' | 'mineral_assignment';
  let sortKey: SortKey = 'center_cm1';
  let sortAsc = true;

  $: sortedPeaks = [...peaks].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (aVal === null && bVal === null) return 0;
    if (aVal === null) return 1;
    if (bVal === null) return -1;
    const cmp = typeof aVal === 'string' ? (aVal as string).localeCompare(bVal as string) : (aVal as number) - (bVal as number);
    return sortAsc ? cmp : -cmp;
  });

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      sortAsc = !sortAsc;
    } else {
      sortKey = key;
      sortAsc = true;
    }
  }

  function sortIndicator(key: SortKey): string {
    if (sortKey !== key) return '';
    return sortAsc ? ' \u2191' : ' \u2193';
  }

  function exportCsv() {
    const header = 'center_cm1,fwhm_cm1,amplitude,snr,r_squared,assignment,modality,sharpness_ratio,pass_sharpness,quality';
    const rows = peaks.map(p =>
      [
        p.center_cm1?.toFixed(1) ?? '',
        p.fwhm_cm1?.toFixed(1) ?? '',
        p.amplitude.toFixed(1),
        p.snr?.toFixed(1) ?? '',
        p.fit_quality?.toFixed(4) ?? '',
        p.mineral_assignment ?? '',
        p.fit_modality,
        p.sharpness_ratio?.toFixed(2) ?? '',
        p.pass_sharpness ?? '',
        p.quality ?? '',
      ].join(','),
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'peaks.csv';
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="peak-table-container">
  {#if peaks.length === 0}
    <p class="empty-state" style="padding: 24px">No peaks detected</p>
  {:else}
    {#if showExport}
      <div class="table-actions">
        <span class="mono" style="font-size: 0.85rem; color: var(--color-text-secondary)">
          {peaks.length} peak{peaks.length !== 1 ? 's' : ''}
        </span>
        <button class="btn-secondary btn-sm" on:click={exportCsv}>Export CSV</button>
      </div>
    {/if}
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th class="sortable" on:click={() => toggleSort('center_cm1')}>
              Center (cm<sup>-1</sup>){sortIndicator('center_cm1')}
            </th>
            <th class="sortable" on:click={() => toggleSort('fwhm_cm1')}>
              FWHM (cm<sup>-1</sup>){sortIndicator('fwhm_cm1')}
            </th>
            <th class="sortable" on:click={() => toggleSort('amplitude')}>
              Amplitude{sortIndicator('amplitude')}
            </th>
            <th class="sortable" on:click={() => toggleSort('snr')}>
              SNR{sortIndicator('snr')}
            </th>
            <th>R&sup2;</th>
            <th class="sortable" on:click={() => toggleSort('mineral_assignment')}>
              Assignment{sortIndicator('mineral_assignment')}
            </th>
            <th>Modality</th>
            <th>Quality</th>
          </tr>
        </thead>
        <tbody>
          {#each sortedPeaks as peak}
            <tr>
              <td class="mono">{peak.center_cm1?.toFixed(1) ?? '--'}</td>
              <td class="mono">{peak.fwhm_cm1?.toFixed(1) ?? '--'}</td>
              <td class="mono">{peak.amplitude.toFixed(1)}</td>
              <td class="mono">{peak.snr?.toFixed(1) ?? '--'}</td>
              <td class="mono">{peak.fit_quality?.toFixed(3) ?? '--'}</td>
              <td>
                {#if peak.mineral_assignment}
                  <span class="badge badge-info">{peak.mineral_assignment}</span>
                {:else}
                  <span class="badge badge-neutral">unassigned</span>
                {/if}
              </td>
              <td>
                <span class="badge badge-neutral">{peak.fit_modality}</span>
              </td>
              <td>
                {#if peak.quality === 'pass'}
                  <span class="badge badge-success">pass</span>
                {:else if peak.quality === 'review'}
                  <span class="badge badge-warning">review</span>
                {:else if peak.quality === 'fail'}
                  <span class="badge badge-error">fail</span>
                {:else}
                  --
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .peak-table-container {
    width: 100%;
  }

  .table-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
  }

  .table-scroll {
    overflow-x: auto;
  }

  .sortable {
    cursor: pointer;
    user-select: none;
  }

  .sortable:hover {
    color: var(--color-primary);
  }

  td {
    white-space: nowrap;
  }
</style>
