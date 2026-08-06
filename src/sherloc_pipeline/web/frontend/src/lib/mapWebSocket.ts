// ============================================================
// WebSocket client for Map Mode fitting protocol.
// Push-based: receives per-point results as they complete.
// ============================================================

import type {
  WSServerMessage,
  WSHeartbeat,
  WSJobStarted,
  WSPointFitted,
  WSProgress,
  WSLog,
  WSJobComplete,
  WSJobFailed,
} from './types/map';

/**
 * Reconnect attempts before the client gives up on the socket and leaves
 * the job to the REST fallback.
 *
 * Budgeted for the whole job rather than reset on each successful
 * connection: a server that accepts and immediately drops would otherwise
 * spin forever. Five attempts covers the ordinary causes (proxy idle
 * timeout, the handler's own 30-minute cap, a brief network loss) and
 * exhausting them is not a failure — polling still reports progress and
 * still recovers the results at the end.
 */
export const MAP_WS_MAX_RECONNECTS = 5;

/** First reconnect delay; doubles on each subsequent attempt. */
export const MAP_WS_RECONNECT_BASE_MS = 1000;

/**
 * Close codes the server uses to refuse a socket outright (public mode,
 * unknown job). Retrying those just repeats the refusal.
 */
const FATAL_CLOSE_CODES = [4003, 4004];

export interface MapWSHandlers {
  onJobStarted: (msg: WSJobStarted) => void;
  onPointFitted: (msg: WSPointFitted) => void;
  onProgress: (msg: WSProgress) => void;
  onLog: (msg: WSLog) => void;
  onComplete: (msg: WSJobComplete) => void;
  onFailed: (msg: WSJobFailed) => void;
  onCancelled: () => void;
  /** Called once, when reconnection has been abandoned or is not wanted. */
  onDisconnect: () => void;
  /** Server-side job liveness frame (status, queue position, stall flag). */
  onHeartbeat?: (msg: WSHeartbeat) => void;
  /** A reconnect has been scheduled after an unexpected close. */
  onReconnecting?: (attempt: number, maxAttempts: number) => void;
  /** The socket came back up and resumed from `last_seq`. */
  onReconnected?: (attempt: number) => void;
}

export class MapWebSocket {
  private ws: WebSocket | null = null;
  private lastSeq = -1;
  private handlers: MapWSHandlers;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnects = 0;
  private closedByClient = false;
  private jobFinished = false;

  constructor(
    private readonly wsUrl: string,
    handlers: MapWSHandlers,
    private readonly maxReconnects: number = MAP_WS_MAX_RECONNECTS,
    private readonly baseDelayMs: number = MAP_WS_RECONNECT_BASE_MS,
  ) {
    this.handlers = handlers;
    this.connect();
  }

  /**
   * Open the socket, resuming from the last sequence number seen.
   *
   * The handler replays its buffered frames past `last_seq`, so a
   * reconnect picks the stream back up rather than restarting it. The
   * buffer is bounded, so a long outage can still leave holes — the caller
   * closes those from the retained-results endpoint once the job ends.
   */
  private connect(): void {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const sep = this.wsUrl.includes('?') ? '&' : '?';
    const resume = this.lastSeq >= 0 ? `${sep}last_seq=${this.lastSeq}` : '';
    const fullUrl = `${protocol}//${window.location.host}${this.wsUrl}${resume}`;
    const ws = new WebSocket(fullUrl);
    this.ws = ws;

    const attempt = this.reconnects;
    ws.onopen = () => {
      if (attempt > 0) this.handlers.onReconnected?.(attempt);
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const msg: WSServerMessage = JSON.parse(event.data);
        this.dispatch(msg);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = (event: CloseEvent) => {
      this.handleClose(event?.code);
    };

    ws.onerror = () => {
      // A close event always follows, which is where reconnect is decided.
    };
  }

  private handleClose(code?: number): void {
    this.ws = null;
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }

    const fatal = typeof code === 'number' && FATAL_CLOSE_CODES.includes(code);
    const retryable =
      !this.closedByClient &&
      !this.jobFinished &&
      !fatal &&
      this.reconnects < this.maxReconnects;

    if (retryable) {
      this.reconnects++;
      const delay = this.baseDelayMs * 2 ** (this.reconnects - 1);
      this.handlers.onReconnecting?.(this.reconnects, this.maxReconnects);
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (!this.closedByClient) this.connect();
      }, delay);
      return;
    }

    this.handlers.onDisconnect();
  }

  private dispatch(msg: WSServerMessage): void {
    if ('seq' in msg && typeof msg.seq === 'number') {
      if (msg.seq <= this.lastSeq) return; // dedup
      this.lastSeq = msg.seq;
    }

    switch (msg.type) {
      case 'job_started':
        this.handlers.onJobStarted(msg as WSJobStarted);
        break;
      case 'point_fitted':
        this.handlers.onPointFitted(msg as WSPointFitted);
        break;
      case 'point_fitted_batch':
        if ('points' in msg) {
          for (const pt of (msg as { points: WSPointFitted[] }).points) {
            this.handlers.onPointFitted(pt);
          }
        }
        break;
      case 'progress':
        this.handlers.onProgress(msg as WSProgress);
        break;
      case 'log':
        this.handlers.onLog(msg as WSLog);
        break;
      case 'job_complete':
      case 'complete':
        // The job is over: the close that follows is expected, not a drop.
        this.jobFinished = true;
        this.handlers.onComplete(msg as WSJobComplete);
        break;
      case 'job_failed':
      case 'error':
        this.jobFinished = true;
        this.handlers.onFailed(msg as WSJobFailed);
        break;
      case 'job_cancelled':
      case 'cancelled':
        this.jobFinished = true;
        this.handlers.onCancelled();
        break;
      case 'heartbeat':
        this.handlers.onHeartbeat?.(msg as WSHeartbeat);
        break;
      case 'ping':
        this.ws?.send(JSON.stringify({ type: 'pong' }));
        break;
    }
  }

  /**
   * Ask the server to cancel the job.
   *
   * @returns false when the socket is not open, so the caller can say the
   *   cancel did not reach the server instead of leaving the button
   *   silently inert.
   */
  sendCancel(): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify({ type: 'cancel' }));
    return true;
  }

  /** True while the client is between reconnect attempts. */
  get isReconnecting(): boolean {
    return this.reconnectTimer !== null;
  }

  close(): void {
    this.closedByClient = true;
    this.cleanup();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }

  private cleanup(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
