// ============================================================
// WebSocket client for Map Mode fitting protocol.
// Push-based: receives per-point results as they complete.
// ============================================================

import type {
  WSServerMessage,
  WSJobStarted,
  WSPointFitted,
  WSProgress,
  WSLog,
  WSJobComplete,
  WSJobFailed,
} from './types/map';

export interface MapWSHandlers {
  onJobStarted: (msg: WSJobStarted) => void;
  onPointFitted: (msg: WSPointFitted) => void;
  onProgress: (msg: WSProgress) => void;
  onLog: (msg: WSLog) => void;
  onComplete: (msg: WSJobComplete) => void;
  onFailed: (msg: WSJobFailed) => void;
  onCancelled: () => void;
  onDisconnect: () => void;
}

export class MapWebSocket {
  private ws: WebSocket | null = null;
  private lastSeq = -1;
  private handlers: MapWSHandlers;
  private pingTimer: ReturnType<typeof setInterval> | null = null;

  constructor(wsUrl: string, handlers: MapWSHandlers) {
    this.handlers = handlers;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const fullUrl = `${protocol}//${window.location.host}${wsUrl}`;
    this.ws = new WebSocket(fullUrl);

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg: WSServerMessage = JSON.parse(event.data);
        this.dispatch(msg);
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this.cleanup();
      this.handlers.onDisconnect();
    };

    this.ws.onerror = () => {
      this.cleanup();
    };
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
        this.handlers.onComplete(msg as WSJobComplete);
        break;
      case 'job_failed':
      case 'error':
        this.handlers.onFailed(msg as WSJobFailed);
        break;
      case 'job_cancelled':
      case 'cancelled':
        this.handlers.onCancelled();
        break;
      case 'ping':
        this.ws?.send(JSON.stringify({ type: 'pong' }));
        break;
    }
  }

  sendCancel(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'cancel' }));
    }
  }

  close(): void {
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
  }
}
