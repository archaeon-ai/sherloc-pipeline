import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MapWebSocket } from './mapWebSocket';
import type { MapWSHandlers } from './mapWebSocket';

// ============================================================
// The fit socket can close while the job is still running (proxy idle
// timeout, the handler's own 30-minute cap, a flaky link). Map Mode used
// to treat every close as final, so the per-point results produced for
// the rest of the job never reached the client — and they exist nowhere
// else, because fitting streams results and never writes them to the
// database. These tests pin the resume behaviour that recovers the live
// stream, and the give-up behaviour that hands over to REST polling.
// ============================================================

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readyState = FakeWebSocket.OPEN;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
  }

  /** Server pushes a frame. */
  emit(msg: object): void {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }

  /** Connection goes away. 1006 is an abnormal (unannounced) close. */
  drop(code = 1006): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code });
  }
}

function makeHandlers(): MapWSHandlers {
  return {
    onJobStarted: vi.fn(),
    onPointFitted: vi.fn(),
    onProgress: vi.fn(),
    onLog: vi.fn(),
    onComplete: vi.fn(),
    onFailed: vi.fn(),
    onCancelled: vi.fn(),
    onDisconnect: vi.fn(),
    onHeartbeat: vi.fn(),
    onReconnecting: vi.fn(),
    onReconnected: vi.fn(),
  };
}

const point = (seq: number, pointIndex: number) => ({
  type: 'point_fitted',
  seq,
  point_index: pointIndex,
  x: 0,
  y: 0,
  results: {},
});

/** Let every scheduled reconnect delay elapse. */
async function settleReconnects(): Promise<void> {
  await vi.advanceTimersByTimeAsync(10_000);
}

describe('MapWebSocket reconnect', () => {
  let realWebSocket: unknown;

  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    realWebSocket = (globalThis as Record<string, unknown>).WebSocket;
    (globalThis as Record<string, unknown>).WebSocket = FakeWebSocket;
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).WebSocket = realWebSocket;
    vi.useRealTimers();
  });

  it('resumes from the last sequence number after an unexpected close', async () => {
    const handlers = makeHandlers();
    const client = new MapWebSocket('/ws/map/mf_1', handlers, 5, 10);

    FakeWebSocket.instances[0].emit(point(7, 3));
    expect(handlers.onPointFitted).toHaveBeenCalledTimes(1);

    FakeWebSocket.instances[0].drop();
    // A drop is not the end of the job: the fallback must not take over
    // before the socket has been given a chance to come back.
    expect(handlers.onDisconnect).not.toHaveBeenCalled();
    expect(handlers.onReconnecting).toHaveBeenCalledWith(1, 5);

    await settleReconnects();

    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(FakeWebSocket.instances[1].url).toContain('last_seq=7');
    FakeWebSocket.instances[1].onopen?.();
    expect(handlers.onReconnected).toHaveBeenCalledWith(1);

    // The replay the server sends back is deduped against what was seen.
    FakeWebSocket.instances[1].emit(point(7, 3));
    expect(handlers.onPointFitted).toHaveBeenCalledTimes(1);
    FakeWebSocket.instances[1].emit(point(8, 4));
    expect(handlers.onPointFitted).toHaveBeenCalledTimes(2);

    client.close();
  });

  it('does not resume once the job has finished', async () => {
    const handlers = makeHandlers();
    new MapWebSocket('/ws/map/mf_2', handlers, 5, 10);

    FakeWebSocket.instances[0].emit({
      type: 'complete',
      seq: 2,
      summary: { total_points: 1, detections: {}, elapsed_s: 1 },
    });
    // The server closes the socket right after the terminal frame.
    FakeWebSocket.instances[0].drop(1000);
    await settleReconnects();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(handlers.onReconnecting).not.toHaveBeenCalled();
    expect(handlers.onDisconnect).toHaveBeenCalledTimes(1);
  });

  it('hands the cancel acknowledgement to the caller intact', async () => {
    const handlers = makeHandlers();
    new MapWebSocket('/ws/map/mf_cancel', handlers, 5, 10);

    // Sent by the fitting thread once it has stopped, so it carries a seq
    // like every other streamed frame. `results_final` is what tells the
    // caller whether the retained results it fetches next are complete.
    FakeWebSocket.instances[0].emit({
      type: 'cancelled',
      seq: 4,
      job_id: 'mf_cancel',
      fitted: 2,
      total: 5,
      results_final: true,
    });
    FakeWebSocket.instances[0].drop(1000);
    await settleReconnects();

    expect(handlers.onCancelled).toHaveBeenCalledWith(
      expect.objectContaining({ type: 'cancelled', fitted: 2, results_final: true }),
    );
    // The job is over: the close behind the acknowledgement is expected.
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(handlers.onReconnecting).not.toHaveBeenCalled();
  });

  it('does not retry a refusal the server will only repeat', async () => {
    const handlers = makeHandlers();
    new MapWebSocket('/ws/map/mf_gone', handlers, 5, 10);

    FakeWebSocket.instances[0].drop(4004); // job not found
    await settleReconnects();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(handlers.onDisconnect).toHaveBeenCalledTimes(1);
  });

  it('hands over to the caller once the attempt budget is spent', async () => {
    const handlers = makeHandlers();
    new MapWebSocket('/ws/map/mf_3', handlers, 2, 10);

    FakeWebSocket.instances[0].drop();
    await settleReconnects();
    FakeWebSocket.instances[1].drop();
    await settleReconnects();
    FakeWebSocket.instances[2].drop();
    await settleReconnects();

    // Two reconnects, then the REST fallback takes the job over rather
    // than the client retrying a socket that clearly is not coming back.
    expect(FakeWebSocket.instances).toHaveLength(3);
    expect(handlers.onDisconnect).toHaveBeenCalledTimes(1);
  });

  it('backs off between attempts', async () => {
    const handlers = makeHandlers();
    new MapWebSocket('/ws/map/mf_4', handlers, 3, 100);

    FakeWebSocket.instances[0].drop();
    await vi.advanceTimersByTimeAsync(99);
    expect(FakeWebSocket.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeWebSocket.instances).toHaveLength(2);

    FakeWebSocket.instances[1].drop();
    await vi.advanceTimersByTimeAsync(150); // second delay is 200ms
    expect(FakeWebSocket.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(50);
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it('stops reconnecting when the client closes the socket itself', async () => {
    const handlers = makeHandlers();
    const client = new MapWebSocket('/ws/map/mf_5', handlers, 5, 10);

    client.close();
    await settleReconnects();

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(handlers.onReconnecting).not.toHaveBeenCalled();
  });

  it('reports a cancel that could not be sent', async () => {
    const handlers = makeHandlers();
    const client = new MapWebSocket('/ws/map/mf_6', handlers, 5, 10);

    expect(client.sendCancel()).toBe(true);
    expect(FakeWebSocket.instances[0].sent).toEqual([JSON.stringify({ type: 'cancel' })]);

    // Between reconnect attempts there is no socket to carry the cancel,
    // and the caller has to be able to say so rather than leaving the
    // button silently inert.
    FakeWebSocket.instances[0].drop();
    expect(client.isReconnecting).toBe(true);
    expect(client.sendCancel()).toBe(false);

    client.close();
  });
});
