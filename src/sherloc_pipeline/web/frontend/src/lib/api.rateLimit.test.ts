import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

vi.mock('./auth', () => ({
  bootstrapAuthReady: Promise.resolve(null),
  getSession: () => null,
}));

import { ApiError, AuthRequiredError, getScans, postBaseline } from './api';
import { clearSessionState, processingNotice } from './stores';

const request = { wavenumber: [1, 2, 3], intensity: [4, 5, 6] };
const data = { schema_version: '1.0.0', corrected: [3, 4, 5], baseline: [1, 1, 1] };

function rateLimited(retryAfter: string | null = '2'): Response {
  return new Response(JSON.stringify({ error: 'rate_limited', detail: 'Too many requests' }), {
    status: 429,
    headers: retryAfter === null ? {} : { 'Retry-After': retryAfter },
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  processingNotice.set(null);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  processingNotice.set(null);
});

describe('processing rate limit retry', () => {
  it('resolves a 429 then 200 after the delay and shows the notice once', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(rateLimited())
      .mockResolvedValueOnce(new Response(JSON.stringify(data)));
    const noticeSpy = vi.spyOn(processingNotice, 'set');
    const pending = postBaseline(request);

    await vi.advanceTimersByTimeAsync(0);
    expect(noticeSpy).toHaveBeenCalledTimes(1);
    expect(noticeSpy).toHaveBeenCalledWith({ message: 'Server busy, retrying in 2 s' });
    expect(get(processingNotice)?.message).toBe('Server busy, retrying in 2 s');
    await vi.advanceTimersByTimeAsync(1999);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(pending).resolves.toEqual(data);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(fetchSpy.mock.calls[1]).toEqual(fetchSpy.mock.calls[0]);
    expect(noticeSpy).toHaveBeenCalledTimes(1);
    expect(get(processingNotice)).toBeNull();
  });

  it('rejects a second 429 with the normal ApiError and stops retrying', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => rateLimited());
    const noticeSpy = vi.spyOn(processingNotice, 'set');
    const pending = postBaseline(request);
    const rejected = expect(pending).rejects.toMatchObject({
      name: 'ApiError', status: 429, errorClass: 'rate_limited', detail: 'Too many requests',
    });
    await vi.advanceTimersByTimeAsync(2000);
    await rejected;
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    expect(noticeSpy).toHaveBeenCalledTimes(1);
    expect(get(processingNotice)).toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('does not retry a non-process path', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(rateLimited());
    const noticeSpy = vi.spyOn(processingNotice, 'set');
    await expect(getScans()).rejects.toBeInstanceOf(ApiError);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(noticeSpy).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each<[string | null, number]>([
    [null, 5], ['invalid', 5], ['', 5], ['120', 60], ['0', 1], ['1.2', 2], ['-3', 5],
  ])('bounds Retry-After %s to %s seconds', async (header, seconds) => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(rateLimited(header))
      .mockResolvedValueOnce(new Response(JSON.stringify(data)));
    const pending = postBaseline(request);
    await vi.advanceTimersByTimeAsync(0);
    expect(get(processingNotice)?.message).toBe(`Server busy, retrying in ${seconds} s`);
    await vi.advanceTimersByTimeAsync(seconds * 1000 - 1);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(pending).resolves.toEqual(data);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it('does not replay a waiting request after logout', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(rateLimited());
    const pending = postBaseline(request);
    const rejected = expect(pending).rejects.toBeInstanceOf(AuthRequiredError);
    await vi.advanceTimersByTimeAsync(0);
    clearSessionState();
    expect(get(processingNotice)).toBeNull();
    await vi.advanceTimersByTimeAsync(2000);
    await rejected;
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});
