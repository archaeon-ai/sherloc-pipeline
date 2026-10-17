<script lang="ts">
  // Shown instead of the app when Auth0 returns to /auth/callback with an
  // error (main.ts). The wording mirrors the PHASE Worker's access pages
  // (m2020-phase viewer-worker/src/oauth.ts ACCESS_DENIED_BODIES).
  import { getSession, type AuthCallbackError } from '../lib/auth';

  export let error: AuthCallbackError;

  const PUBLIC_SITE = 'https://sherloc-public.m2020-phase.net/';

  const COPY: Record<AuthCallbackError['kind'], { title: string; heading: string; body: string }> = {
    team_required: {
      title: 'Team membership required — PHASE',
      heading: 'Team membership required',
      body: 'This page is for the Mars 2020 team. The account you are signed in with does not have team access.',
    },
    verify_email: {
      title: 'Verify your email — PHASE',
      heading: 'Please verify your email address',
      body:
        'We sent a verification link to the email address you signed up with — ' +
        'check your inbox (and spam folder). Click the link, then return here and sign in again.',
    },
    team_pending: {
      title: 'Team access under review — PHASE',
      heading: 'Team access under review',
      body:
        'Thanks for requesting Mars 2020 team-tier access. The platform operator received a notification of your ' +
        'signup and will review the request. You’ll get an email once your account is approved.',
    },
    denied: {
      title: 'Access denied — PHASE',
      heading: 'Access denied',
      body:
        'Your sign-in attempt was rejected. ' +
        'If you just signed up, you may need to verify your email address first — check your inbox for a verification link. ' +
        'Otherwise, contact the platform operator.',
    },
    generic: {
      title: 'Sign-in failed — PHASE',
      heading: 'Sign-in did not complete',
      body: 'The sign-in could not be completed. Please try again.',
    },
  };

  $: copy = COPY[error.kind];
  $: document.title = copy.title;

  function signIn(prompt?: 'login') {
    void getSession()?.login({ prompt, returnTo: error.returnTo });
  }
</script>

<main class="page-container">
  <div class="card access-card">
    <h1>{copy.heading}</h1>
    <p>{copy.body}</p>
    {#if error.kind === 'team_required'}
      <p><a href={PUBLIC_SITE}>Go to the public SHERLOC site</a></p>
    {:else if error.kind === 'team_pending'}
      <p>In the meantime, you can explore the public-tier tools at <a href={PUBLIC_SITE}>sherloc-public.m2020-phase.net</a>.</p>
    {/if}
    {#if error.kind === 'team_required'}
      <p><button type="button" class="link-button" on:click={() => signIn('login')}>Sign in with a different account</button></p>
    {:else if error.kind !== 'team_pending'}
      <p><button type="button" class="link-button" on:click={() => signIn()}>Sign in again</button></p>
    {/if}
    <p class="contact">
      Please contact <a href="mailto:support@m2020-phase.net">support@m2020-phase.net</a> with any questions.
    </p>
  </div>
</main>

<style>
  .access-card {
    max-width: 520px;
    margin: 48px auto;
    padding: 32px 28px;
  }
  h1 {
    margin: 0 0 16px;
    font-size: 1.4rem;
  }
  p {
    margin: 0 0 12px;
    line-height: 1.5;
  }
  .link-button {
    background: none;
    border: none;
    padding: 0;
    color: var(--color-primary, inherit);
    text-decoration: underline;
    cursor: pointer;
    font: inherit;
  }
  .contact {
    margin-top: 20px;
    font-size: 0.85rem;
    color: var(--color-text-secondary);
  }
</style>
