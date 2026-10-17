import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/svelte';
import AccessError from './AccessError.svelte';
import * as auth from '../lib/auth';

const login = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  login.mockClear();
  vi.spyOn(auth, 'getSession').mockReturnValue({ login } as unknown as auth.AuthSession);
});

describe('AccessError (#308)', () => {
  // The Worker's copy (m2020-phase viewer-worker/src/oauth.ts ACCESS_DENIED_BODIES).
  const WORKER = {
    team: 'This page is for the Mars 2020 team. The account you are signed in with does not have team access.',
    verify:
      'We sent a verification link to the email address you signed up with — check your inbox (and spam folder). ' +
      'Click the link, then return here and sign in again.',
    denied:
      'Your sign-in attempt was rejected. If you just signed up, you may need to verify your email address first — ' +
      'check your inbox for a verification link. Otherwise, contact the platform operator.',
  };

  it('team_required: Worker wording, public site link, different-account sign-in', async () => {
    render(AccessError, { props: { error: { kind: 'team_required', returnTo: '/#/open?scan_name=x' } } });
    expect(screen.getByRole('heading', { name: 'Team membership required' })).toBeInTheDocument();
    expect(screen.getByText(WORKER.team)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Go to the public SHERLOC site' })).toHaveAttribute(
      'href',
      'https://sherloc-public.m2020-phase.net/',
    );
    await fireEvent.click(screen.getByRole('button', { name: 'Sign in with a different account' }));
    expect(login).toHaveBeenCalledWith({ prompt: 'login', returnTo: '/#/open?scan_name=x' });
    expect(document.title).toBe('Team membership required — PHASE');
  });

  it('generic: plain message and a sign-in retry', async () => {
    render(AccessError, { props: { error: { kind: 'generic', returnTo: '/' } } });
    expect(screen.getByRole('heading', { name: 'Sign-in did not complete' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'Go to the public SHERLOC site' })).toBeNull();
    await fireEvent.click(screen.getByRole('button', { name: 'Sign in again' }));
    expect(login).toHaveBeenCalledWith({ prompt: undefined, returnTo: '/' });
  });

  it('verify_email: Worker wording and a plain sign-in retry', async () => {
    render(AccessError, { props: { error: { kind: 'verify_email', returnTo: '/' } } });
    expect(screen.getByRole('heading', { name: 'Please verify your email address' })).toBeInTheDocument();
    expect(screen.getByText(WORKER.verify)).toBeInTheDocument();
    await fireEvent.click(screen.getByRole('button', { name: 'Sign in again' }));
    expect(login).toHaveBeenCalledWith({ prompt: undefined, returnTo: '/' });
  });

  it('denied: the Worker generic access-denied wording', () => {
    render(AccessError, { props: { error: { kind: 'denied', returnTo: '/' } } });
    expect(screen.getByRole('heading', { name: 'Access denied' })).toBeInTheDocument();
    expect(screen.getByText(WORKER.denied)).toBeInTheDocument();
    expect(document.title).toBe('Access denied — PHASE');
  });

  it('team_pending: no sign-in button, public site link', () => {
    render(AccessError, { props: { error: { kind: 'team_pending', returnTo: '/' } } });
    expect(screen.getByRole('heading', { name: 'Team access under review' })).toBeInTheDocument();
    expect(screen.queryByRole('button')).toBeNull();
    expect(screen.getByRole('link', { name: 'sherloc-public.m2020-phase.net' })).toHaveAttribute(
      'href',
      'https://sherloc-public.m2020-phase.net/',
    );
  });
});
