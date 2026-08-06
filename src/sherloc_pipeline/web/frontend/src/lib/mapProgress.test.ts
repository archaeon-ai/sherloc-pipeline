import { describe, it, expect } from 'vitest';
import { MapProgressTracker } from './mapProgress';

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
