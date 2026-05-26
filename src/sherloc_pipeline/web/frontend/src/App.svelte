<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { get } from 'svelte/store';
  import { currentHash, parsedRoute, accessMode, features, navigate } from './lib/stores';
  import { getAccessMode } from './lib/api';
  import Navigation from './components/Navigation.svelte';
  import ScanBrowser from './components/ScanBrowser.svelte';
  import ScanDetail from './components/ScanDetail.svelte';
  import FittingWorkspace from './components/FittingWorkspace.svelte';
  import BaselineWorkspace from './components/BaselineWorkspace.svelte';
  import ProcessingWorkbench from './components/ProcessingWorkbench.svelte';
  import PdsBrowser from './components/PdsBrowser.svelte';
  import MapMode from './components/map/MapMode.svelte';

  $: route = $parsedRoute;

  // Issue #21 — when the PDS Browser feature is env-disabled, any
  // hash-route landing on `#/pds` (direct paste, bookmark, stale tab)
  // bounces synchronously back to the default route. `navigate()`
  // updates both `window.location.hash` and the `currentHash` store
  // in one step, so the `parsedRoute` derived store re-computes
  // before the next render — no transient soft-404 frame.
  $: if (route.page === 'pds' && !$features.pds_browser) {
    navigate('#/');
  }

  function onHashChange() {
    currentHash.set(window.location.hash || '#/');
  }

  onMount(async () => {
    window.addEventListener('hashchange', onHashChange);
    // Set initial hash
    if (!window.location.hash) {
      window.location.hash = '#/';
    }
    onHashChange();

    // Fetch access mode
    try {
      const mode = await getAccessMode();
      accessMode.set(mode.access_mode);
      // In public mode default to PDS browser — UNLESS PDS is feature-hidden
      // (issue #21 prod gate). Then fall through to the default route (Scans).
      if (
        mode.access_mode === 'public'
        && get(features).pds_browser
        && (!window.location.hash || window.location.hash === '#/')
      ) {
        window.location.hash = '#/pds';
      }
    } catch {
      // ignore — default to internal
    }
  });

  onDestroy(() => {
    window.removeEventListener('hashchange', onHashChange);
  });
</script>

<Navigation />

<main>
  {#if route.page === 'browser'}
    <ScanBrowser />
  {:else if route.page === 'scan' && route.scanId}
    {#key route.scanId}
      <ScanDetail scanId={route.scanId} />
    {/key}
  {:else if route.page === 'workbench' && route.scanId}
    {#key route.scanId + (route.queryParams?.point ?? '')}
      <ProcessingWorkbench scanId={route.scanId} queryParams={route.queryParams ?? {}} />
    {/key}
  {:else if route.page === 'fit' && route.scanId}
    {#key route.scanId}
      <FittingWorkspace scanId={route.scanId} />
    {/key}
  {:else if route.page === 'baseline' && route.scanId}
    {#key route.scanId}
      <BaselineWorkspace scanId={route.scanId} />
    {/key}
  {:else if route.page === 'map' && route.scanId}
    {#key route.scanId}
      <MapMode scanId={route.scanId} />
    {/key}
  {:else if route.page === 'pds' && $features.pds_browser}
    <PdsBrowser />
  {:else}
    <div class="page-container">
      <div class="empty-state">Page not found</div>
    </div>
  {/if}
</main>

<style>
  main {
    flex: 1;
  }
</style>
