import { afterEach, expect, it, vi } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import { get } from 'svelte/store';
import App from './App.svelte';
import * as api from './lib/api';
import { accessModeResolved, clearSessionState, currentHash } from './lib/stores';

afterEach(() => { vi.restoreAllMocks(); });

it('keeps the access-mode store unresolved after logout during startup', async () => {
  let reject!: (reason: Error) => void;
  vi.spyOn(api, 'getAccessMode').mockReturnValue(new Promise((_, r) => { reject = r; }));
  vi.spyOn(api, 'getScan').mockRejectedValue(new Error('synthetic unavailable'));
  vi.spyOn(api, 'getBadpixChannels').mockRejectedValue(new Error('synthetic unavailable'));
  vi.spyOn(api, 'getHealth').mockRejectedValue(new Error('synthetic unavailable'));
  currentHash.set('#/scan/synthetic-scan/workbench');
  window.location.hash = '#/scan/synthetic-scan/workbench';
  const component = render(App);
  await waitFor(() => expect(api.getAccessMode).toHaveBeenCalled());
  component.unmount();
  clearSessionState();
  reject(new api.AuthRequiredError());
  await new Promise((r) => setTimeout(r, 0));
  expect(get(accessModeResolved)).toBe(false);
});
