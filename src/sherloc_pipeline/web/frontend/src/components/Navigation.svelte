<script lang="ts">
  import { navigate, parsedRoute, healthStatus, accessMode, features } from '../lib/stores';
  import { getHealth } from '../lib/api';
  import { getSession } from '../lib/auth';
  import { onMount } from 'svelte';

  $: route = $parsedRoute;

  // Login/Logout UI is gated on auth0 mode (§13.4). cf-access and dev
  // both have showsLoginUi === false so the buttons never render.
  let showsLoginUi = false;
  let isAuthed = false;

  function isActive(page: string): boolean {
    return route.page === page;
  }

  onMount(() => {
    // Periodic health check
    const check = async () => {
      try {
        const h = await getHealth();
        healthStatus.set(h);
      } catch {
        // ignore
      }
    };
    check();
    const interval = setInterval(check, 30000);

    // Wire auth UI once the bootstrap has resolved.
    const session = getSession();
    if (session !== null) {
      showsLoginUi = session.showsLoginUi;
      session.isAuthenticated().then((v) => (isAuthed = v));
    }

    return () => clearInterval(interval);
  });

  async function handleLogin() {
    const session = getSession();
    if (session) await session.login();
  }

  async function handleLogout() {
    const session = getSession();
    if (session) await session.logout();
  }

  $: statusColor = $healthStatus
    ? $healthStatus.status === 'ok'
      ? 'var(--color-success)'
      : $healthStatus.status === 'degraded'
        ? 'var(--color-warning)'
        : 'var(--color-error)'
    : 'var(--color-text-tertiary)';
</script>

<nav class="nav-bar">
  <div class="nav-inner">
    <div class="nav-brand" on:click={() => navigate('#/')} on:keydown={() => {}} role="button" tabindex="0">
      <span class="brand-mark">S</span>
      <span class="brand-text">SHERLOC Pipeline{#if $accessMode === 'public'}&nbsp;(Public){/if}</span>
    </div>

    <div class="nav-links">
      <button
        class="nav-link"
        class:active={isActive('browser')}
        on:click={() => navigate('#/')}
      >
        Scans
      </button>
      {#if $features.pds_browser}
        <button
          class="nav-link"
          class:active={isActive('pds')}
          on:click={() => navigate('#/pds')}
        >
          PDS Browser
        </button>
      {/if}
      {#if route.scanId}
        <span class="nav-sep">|</span>
        <button
          class="nav-link"
          class:active={route.page === 'workbench'}
          on:click={() => navigate(`#/scan/${route.scanId}/workbench`)}
        >
          Workbench
        </button>
        <button
          class="nav-link"
          class:active={route.page === 'map'}
          on:click={() => navigate(`#/scan/${route.scanId}/map`)}
        >
          Map
        </button>
      {/if}
    </div>

    <div class="nav-status">
      {#if showsLoginUi}
        {#if isAuthed}
          <button class="auth-btn" on:click={handleLogout}>Log out</button>
        {:else}
          <button class="auth-btn auth-btn-primary" on:click={handleLogin}>Log in</button>
        {/if}
      {/if}
      <span class="status-dot" style="background: {statusColor}"></span>
      <span class="status-label">
        {#if $healthStatus}
          {$healthStatus.status}
        {:else}
          ...
        {/if}
      </span>
    </div>
  </div>
</nav>

<style>
  .nav-bar {
    background: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
    box-shadow: var(--shadow-sm);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .nav-inner {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    align-items: center;
    height: 48px;
    gap: 32px;
  }

  .nav-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
  }

  .brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--color-primary);
    color: white;
    border-radius: var(--radius-md);
    font-weight: 700;
    font-size: 1rem;
    font-family: var(--font-mono);
  }

  .brand-text {
    font-weight: 600;
    font-size: 1rem;
    color: var(--color-text);
  }

  .nav-links {
    display: flex;
    gap: 4px;
  }

  .nav-link {
    background: none;
    border: none;
    padding: 8px 14px;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--color-text-secondary);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.15s;
  }

  .nav-link:hover {
    background: var(--color-background);
    color: var(--color-text);
  }

  .nav-link.active {
    background: var(--color-primary-light);
    color: var(--color-primary);
  }

  .nav-sep {
    color: var(--color-text-tertiary);
    font-size: 0.8rem;
    user-select: none;
  }

  .nav-status {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--color-text-secondary);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .auth-btn {
    background: none;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 4px 12px;
    font-size: 0.8rem;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .auth-btn:hover {
    background: var(--color-background);
    color: var(--color-text);
  }

  .auth-btn-primary {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
  }

  .auth-btn-primary:hover {
    background: var(--color-primary);
    color: white;
    opacity: 0.9;
  }
</style>
