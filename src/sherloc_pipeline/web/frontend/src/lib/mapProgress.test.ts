import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  MapProgressTracker,
  MapJobPoller,
  isTerminalJobStatus,
  jobLivenessNote,
} from './mapProgress';
import type { MapJobStatus } from './mapProgress';

describe('MapProgressTracker', () => {
  it('counts distinct points as they stream in', () => {
    const p = new MapProgressTracker();
    for (const i of [0, 1, 2]) p.notePoint(i, i + 1);
    expect(p.fitted).toBe(3);
    expect(p.pointsReceived).toBe(3);
  });

  it('does not double-count the backlog handed to a late connection', () => {
    // Server has fitted 500 of 1000 points when the client connects: the
    // immediate heartbeat reports 500, then the 500 queued per-point frames
    // the client never saw are delivered behind it.
    const p = new MapProgressTracker();
    p.noteServerCount(500);
    expect(p.fitted).toBe(500);

    const progression: number[] = [];
    for (let i = 0; i < 500; i++) {
      p.notePoint(i, i + 1);
      progression.push(p.fitted);
    }

    expect(p.fitted).toBe(500); // not 1000
    // Never runs backwards while the backlog drains.
    expect(progression.every((v) => v === 500)).toBe(true);

    // Live points after the backlog advance the count again.
    p.notePoint(500, 501);
    expect(p.fitted).toBe(501);
  });

  it('reports a new point exactly once', () => {
    const p = new MapProgressTracker();
    expect(p.notePoint(7, 1)).toBe(true);
    expect(p.notePoint(7, 1)).toBe(false); // replayed after a reconnect
    expect(p.fitted).toBe(1);
  });

  it('keeps the count monotonic across mixed signals', () => {
    const p = new MapProgressTracker();
    p.noteServerCount(40);
    p.notePoint(0, 1); // stale absolute count from a replayed frame
    expect(p.fitted).toBe(40);
    p.noteServerCount(12); // out-of-order/stale heartbeat
    expect(p.fitted).toBe(40);
  });

  it('ignores absent or non-finite server counts', () => {
    const p = new MapProgressTracker();
    p.notePoint(0); // server frame without a `fitted` field
    p.noteServerCount(undefined);
    p.noteServerCount(NaN);
    expect(p.fitted).toBe(1);
  });

  it('resets between jobs', () => {
    const p = new MapProgressTracker();
    p.noteServerCount(500);
    p.notePoint(3, 500);
    p.reset();
    expect(p.fitted).toBe(0);
    expect(p.notePoint(3, 1)).toBe(true);
    expect(p.fitted).toBe(1);
  });
});

describe('jobLivenessNote', () => {
  it('names how many jobs a queued fit is waiting behind', () => {
    expect(jobLivenessNote({ status: 'queued', queue_position: 2 })).toContain(
      'Queued behind 2 active fit jobs',
    );
    expect(jobLivenessNote({ status: 'queued', queue_position: 1 })).toContain(
      'Queued behind 1 active fit job —',
    );
  });

  it('reports a queued job with nothing ahead of it', () => {
    expect(jobLivenessNote({ status: 'queued' })).toBe(
      'Queued — waiting for the fitting worker.',
    );
  });

  it('reports how long a stalled job has been silent', () => {
    const note = jobLivenessNote({
      status: 'running',
      stalled: true,
      since_last_message_s: 137.4,
      fitted: 12,
      total: 900,
    });
    expect(note).toContain('137s');
    expect(note).toContain('12/900');
  });

  it('says nothing about a job that is simply running', () => {
    expect(jobLivenessNote({ status: 'running', fitted: 5, total: 10 })).toBe('');
  });

  it('flags a timed-out stream ahead of every other condition', () => {
    const note = jobLivenessNote({ status: 'queued', timed_out: true });
    expect(note).toContain('timed out');
  });
});

describe('isTerminalJobStatus', () => {
  it('separates finished jobs from live ones', () => {
    expect(['complete', 'failed', 'cancelled'].map(isTerminalJobStatus)).toEqual([
      true,
      true,
      true,
    ]);
    expect(['queued', 'running', '', undefined].map(isTerminalJobStatus)).toEqual([
      false,
      false,
      false,
      false,
    ]);
  });
});

describe('MapJobPoller', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  const status = (over: Partial<MapJobStatus> = {}): MapJobStatus => ({
    job_id: 'mf_1',
    status: 'running',
    fitted: 0,
    total: 10,
    ...over,
  });

  it('polls until the job reaches a terminal status, then stops', async () => {
    vi.useFakeTimers();
    const responses = [
      status({ status: 'queued', queue_position: 1 }),
      status({ status: 'running', fitted: 4 }),
      status({ status: 'complete', fitted: 10, results_available: true }),
    ];
    const fetchStatus = vi.fn(async () => responses.shift()!);
    const seen: MapJobStatus[] = [];
    const terminal: MapJobStatus[] = [];

    const poller = new MapJobPoller(
      'mf_1',
      fetchStatus,
      { onStatus: (s) => seen.push(s), onTerminal: (s) => terminal.push(s) },
      1000,
    );
    poller.start();

    await vi.advanceTimersByTimeAsync(3000);

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(seen.map((s) => s.status)).toEqual(['queued', 'running', 'complete']);
    expect(terminal).toHaveLength(1);

    // No further polls once terminal.
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
  });

  it('preserves the queued and stalled signals the WebSocket carries', async () => {
    vi.useFakeTimers();
    const responses = [
      status({ status: 'queued', queue_position: 3 }),
      status({
        status: 'running',
        fitted: 2,
        stalled: true,
        since_last_message_s: 180,
      }),
      status({ status: 'complete', fitted: 10 }),
    ];
    const seen: MapJobStatus[] = [];
    const poller = new MapJobPoller(
      'mf_1',
      async () => responses.shift()!,
      { onStatus: (s) => seen.push(s) },
      1000,
    );
    poller.start();
    await vi.advanceTimersByTimeAsync(3000);

    expect(jobLivenessNote(seen[0])).toContain('Queued behind 3');
    expect(jobLivenessNote(seen[1])).toContain('180s');
  });

  it('keeps polling after a transient failure', async () => {
    vi.useFakeTimers();
    let call = 0;
    const errors: unknown[] = [];
    const fetchStatus = vi.fn(async () => {
      call++;
      if (call === 1) throw new Error('network down');
      return status({ status: 'running', fitted: call });
    });

    const poller = new MapJobPoller(
      'mf_1',
      fetchStatus,
      { onStatus: () => {}, onError: (e) => errors.push(e) },
      1000,
    );
    poller.start();
    await vi.advanceTimersByTimeAsync(3000);
    poller.stop();

    expect(errors).toHaveLength(1);
    expect(fetchStatus.mock.calls.length).toBeGreaterThan(1);
  });

  it('gives up after repeated failures instead of retrying forever', async () => {
    vi.useFakeTimers();
    // A job the server no longer knows about (reaped, or lost to a
    // restart) 404s on every poll; retrying it forever would leave the
    // panel claiming progress nobody is reporting.
    const fetchStatus = vi.fn(async () => {
      throw new Error('404 Job not found');
    });
    const gaveUp: unknown[] = [];

    const poller = new MapJobPoller(
      'mf_gone',
      fetchStatus,
      { onStatus: () => {}, onGiveUp: (e) => gaveUp.push(e) },
      1000,
      3,
    );
    poller.start();
    await vi.advanceTimersByTimeAsync(30_000);

    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(gaveUp).toHaveLength(1);
  });

  it('forgets earlier failures once a poll succeeds', async () => {
    vi.useFakeTimers();
    const script: Array<MapJobStatus | Error> = [
      new Error('blip'),
      new Error('blip'),
      status({ status: 'running', fitted: 1 }),
      new Error('blip'),
      new Error('blip'),
      status({ status: 'complete', fitted: 10 }),
    ];
    const fetchStatus = vi.fn(async () => {
      const next = script.shift()!;
      if (next instanceof Error) throw next;
      return next;
    });
    const gaveUp: unknown[] = [];
    const terminal: MapJobStatus[] = [];

    const poller = new MapJobPoller(
      'mf_1',
      fetchStatus,
      {
        onStatus: () => {},
        onTerminal: (s) => terminal.push(s),
        onGiveUp: (e) => gaveUp.push(e),
      },
      1000,
      3,
    );
    poller.start();
    await vi.advanceTimersByTimeAsync(10_000);

    expect(gaveUp).toHaveLength(0);
    expect(terminal).toHaveLength(1);
  });

  it('stops on request and ignores a later start', async () => {
    vi.useFakeTimers();
    const fetchStatus = vi.fn(async () => status());
    const poller = new MapJobPoller('mf_1', fetchStatus, { onStatus: () => {} }, 1000);

    poller.start();
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    poller.stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(1);

    poller.start(); // stopped is final — a torn-down panel must not resurrect
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });
});
