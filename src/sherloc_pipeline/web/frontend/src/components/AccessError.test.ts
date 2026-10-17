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
  it('team_required: Worker wording, public site link, different-account sign-in', async () => {
    render(AccessError, { props: { error: { kind: 'team_required', returnTo: '/#/open?scan_name=x' } } });
    expect(screen.getByRole('heading', { name: 'Team membership required' })).toBeInTheDocument();
    expect(screen.getByText(/does not have team access/)).toBeInTheDocument();
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

  it('team_pending: no sign-in button, public site link', () => {
    render(AccessError, { props: { error: { kind: 'team_pending', returnTo: '/' } } });
    expect(screen.getByRole('heading', { name: 'Team access under review' })).toBeInTheDocument();
    expect(screen.queryByRole('button')).toBeNull();
    expect(screen.getByRole('link', { name: 'Go to the public SHERLOC site' })).toBeInTheDocument();
  });
});
