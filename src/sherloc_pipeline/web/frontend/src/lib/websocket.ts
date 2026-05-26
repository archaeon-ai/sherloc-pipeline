// ============================================================
// WebSocket client for real-time job progress
// ============================================================

import type { WsMessage } from './types';

export type WsCallback = (msg: WsMessage) => void;

export class JobWebSocket {
  private ws: WebSocket | null = null;
  private jobId: string;
  private onMessage: WsCallback;
  private onClose: (() => void) | null;
  private reconnectAttempts = 0;
  private maxReconnects = 3;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;

  constructor(jobId: string, onMessage: WsCallback, onClose?: () => void) {
    this.jobId = jobId;
    this.onMessage = onMessage;
    this.onClose = onClose ?? null;
    this.connect();
  }

  private getWsUrl(): string {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/api/ws/jobs/${this.jobId}`;
  }

  private connect(): void {
    if (this.closed) return;

    try {
      this.ws = new WebSocket(this.getWsUrl());
    } catch {
      this.handleReconnect();
      return;
    }

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        this.onMessage(msg);
        // Reset reconnect counter on successful message
        this.reconnectAttempts = 0;
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      if (!this.closed) {
        this.handleReconnect();
      }
      this.onClose?.();
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  private handleReconnect(): void {
    if (this.closed || this.reconnectAttempts >= this.maxReconnects) return;
    this.reconnectAttempts++;
    const delay = 1000 * this.reconnectAttempts;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  sendCancel(submitterToken: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: 'cancel',
          submitter_token: submitterToken,
        }),
      );
    }
  }

  close(): void {
    this.closed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}

/**
 * Fallback: poll job status at intervals when WebSocket is unavailable.
 */
export function createPollingFallback(
  jobId: string,
  onStatus: (status: unknown) => void,
  intervalMs = 2000,
): () => void {
  let active = true;
  const poll = async () => {
    if (!active) return;
    try {
      // Auth-attaching via getJobStatus() — raw fetch produced 401 under
      // Auth0 Bearer-token mode (Codex PR9 R3 F5).
      const { getJobStatus } = await import('./api');
      const data = await getJobStatus(jobId);
      onStatus(data);
      const t = (data as { type?: string }).type;
      if (t === 'complete' || t === 'error' || t === 'cancelled') {
        active = false;
        return;
      }
    } catch {
      // Ignore fetch errors (incl. AuthRequiredError), retry on next interval
    }
    if (active) {
      setTimeout(poll, intervalMs);
    }
  };
  poll();
  return () => {
    active = false;
  };
}
