// ============================================================
// Progress reconciliation for a single Map Mode fitting job.
//
// Two independent progress signals arrive on the same stream:
//   * per-point `point_fitted` frames, and
//   * server-side counts in `progress` / `heartbeat` frames.
//
// They overlap. A client that connects after fitting has already started
// gets a status frame with the count so far and then, right behind it,
// every per-point frame it missed — so incrementing a local counter on
// each point frame double-counts, while trusting the point frames alone
// makes progress jump backwards from the status frame's count.
//
// Track point identity instead, and let the displayed count be the larger
// of "distinct points seen" and "highest count the server reported". Both
// are monotonic, so the count never runs backwards and never exceeds what
// the server has actually fitted.
// ============================================================

export class MapProgressTracker {
  private seen = new Set<number>();
  private serverFitted = 0;

  /**
   * Record a per-point result.
   *
   * @returns true if this point had not been seen before (replayed and
   *   re-delivered frames return false, so callers can skip re-ingesting).
   */
  notePoint(pointIndex: number, serverFitted?: number): boolean {
    this.noteServerCount(serverFitted);
    if (this.seen.has(pointIndex)) return false;
    this.seen.add(pointIndex);
    return true;
  }

  /** Record an authoritative server-side count (progress/heartbeat frame). */
  noteServerCount(fitted?: number): void {
    if (typeof fitted !== 'number' || !Number.isFinite(fitted)) return;
    if (fitted > this.serverFitted) this.serverFitted = fitted;
  }

  /** Points fitted so far, reconciled across both signals. */
  get fitted(): number {
    return Math.max(this.seen.size, this.serverFitted);
  }

  /** Distinct per-point frames actually received by this client. */
  get pointsReceived(): number {
    return this.seen.size;
  }

  /** Clear all state — call when a new job starts. */
  reset(): void {
    this.seen.clear();
    this.serverFitted = 0;
  }
}

/**
 * Pick the retained results this client has not already received, and
 * record them as seen.
 *
 * Recovery after a dropped fit stream re-delivers everything the job has
 * produced, including the points that did arrive live. Filtering by point
 * identity keeps those from being ingested (and cached, and re-coloured)
 * twice, and leaves the reconciled count correct: the tracker's `fitted`
 * already covers the points it has seen.
 */
export function selectMissedResults<T extends { point_index: number }>(
  points: T[],
  progress: MapProgressTracker,
): T[] {
  return points.filter((p) => progress.notePoint(p.point_index));
}

// ============================================================
// Job liveness, shared by the WebSocket heartbeat and the REST fallback.
//
// Both surfaces carry the same fields, so both render the same sentence:
// a user whose WebSocket dropped should not get a different (or worse, no)
// explanation of why the progress bar is not moving.
// ============================================================

/** Liveness fields common to a WS heartbeat frame and a REST job status. */
export interface MapJobLiveness {
  status?: string;
  fitted?: number;
  total?: number;
  queue_position?: number;
  since_last_message_s?: number;
  stalled?: boolean;
  timed_out?: boolean;
}

/** Full REST payload from `GET /api/map/jobs/{job_id}`. */
export interface MapJobStatus extends MapJobLiveness {
  job_id?: string;
  status: string;
  fitted: number;
  total: number;
  results_available?: boolean;
  /**
   * Per-point results the server still holds for this job. Non-zero means
   * the frames missed while the stream was down are still fetchable from
   * `GET /api/map/jobs/{job_id}/results`.
   */
  results_retained?: number;
  /**
   * False while the fitting thread may still add a point to that store.
   *
   * A terminal `status` is not the same signal: a cancel is recorded when
   * it is requested, but the fitting loop only notices between points, so
   * the point it was on is retained afterwards. Reading the results in
   * that window loses a finished measurement (issue #6). Absent on
   * servers that predate the field, which is why the poller only waits on
   * an explicit `false`.
   */
  results_final?: boolean;
}

export const TERMINAL_JOB_STATUSES = ['complete', 'failed', 'cancelled'] as const;

export function isTerminalJobStatus(status: string | undefined): boolean {
  return (TERMINAL_JOB_STATUSES as readonly string[]).includes(status ?? '');
}

/**
 * One-line explanation of why a job is not producing results, or `''` when
 * there is nothing to say (it is simply running normally).
 */
export function jobLivenessNote(msg: MapJobLiveness): string {
  if (msg.timed_out) {
    return 'Fitting stream timed out; the job may still be running on the server.';
  }
  if (msg.status === 'queued') {
    const ahead = msg.queue_position ?? 0;
    return ahead > 0
      ? `Queued behind ${ahead} active fit job${ahead === 1 ? '' : 's'} — fitting runs one scan at a time.`
      : 'Queued — waiting for the fitting worker.';
  }
  if (msg.stalled) {
    const silent = Math.round(msg.since_last_message_s ?? 0);
    return `No new results for ${silent}s (${msg.fitted ?? 0}/${msg.total ?? 0} points) — this scan is fitting slowly.`;
  }
  return '';
}

/** Default gap between REST status polls, in milliseconds. */
export const MAP_JOB_POLL_INTERVAL_MS = 5000;

/**
 * Consecutive failed polls after which the fallback gives up.
 *
 * Bounded because the failure may be permanent — the job was reaped from
 * the registry, or the server restarted and never knew about it — and an
 * unbounded retry loop would keep a dead job's panel spinning forever,
 * which is the same lie the fallback exists to stop telling.
 */
export const MAP_JOB_POLL_MAX_ERRORS = 5;

/**
 * Extra polls allowed after a terminal status while the server still
 * reports its retained results as not final.
 *
 * The gap is real but short: the fitting thread notices a cancel between
 * points and finishes the one it is on. Waiting through it is what stops
 * a finished measurement from being read past (issue #6). Bounded for the
 * same reason as MAP_JOB_POLL_MAX_ERRORS — a wedged fitting thread must
 * not keep this poller alive forever; the caller is handed the last
 * status either way and can say the map may be a point short.
 */
export const MAP_JOB_POLL_MAX_SETTLE_POLLS = 3;

export interface MapJobPollerHandlers {
  /** Called with each successful status read. */
  onStatus: (status: MapJobStatus) => void;
  /** Called once when the job reaches a terminal status. */
  onTerminal?: (status: MapJobStatus) => void;
  /** Called when a poll fails; the poller keeps trying. */
  onError?: (err: unknown) => void;
  /** Called once when polling is abandoned after repeated failures. */
  onGiveUp?: (err: unknown) => void;
}

/**
 * REST polling fallback for a fit job whose WebSocket is unavailable.
 *
 * The fit WebSocket is the primary channel, but a dropped or blocked
 * connection used to leave Map Mode with no signal at all — the progress
 * panel simply froze, which is the symptom issue #6 is about. Polling
 * `GET /api/map/jobs/{job_id}` recovers status, progress, queue position
 * and the stall flag; only the per-point results stop flowing, so the map
 * is refreshed from the database once the job finishes.
 *
 * `fetchStatus` is injected so the loop is testable without a network.
 */
export class MapJobPoller {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private inFlight = false;
  private consecutiveErrors = 0;
  private settlePolls = 0;

  constructor(
    private readonly jobId: string,
    private readonly fetchStatus: (jobId: string) => Promise<MapJobStatus>,
    private readonly handlers: MapJobPollerHandlers,
    private readonly intervalMs: number = MAP_JOB_POLL_INTERVAL_MS,
    private readonly maxErrors: number = MAP_JOB_POLL_MAX_ERRORS,
    private readonly maxSettlePolls: number = MAP_JOB_POLL_MAX_SETTLE_POLLS,
  ) {}

  /** Begin polling. Safe to call once; further calls are ignored. */
  start(): void {
    if (this.stopped || this.timer !== null || this.inFlight) return;
    void this.tick();
  }

  /** Stop polling. Idempotent, and safe to call from a handler. */
  stop(): void {
    this.stopped = true;
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private async tick(): Promise<void> {
    if (this.stopped) return;
    this.timer = null;
    this.inFlight = true;
    try {
      const status = await this.fetchStatus(this.jobId);
      if (this.stopped) return;
      this.consecutiveErrors = 0;
      this.handlers.onStatus(status);
      if (isTerminalJobStatus(status.status)) {
        // Terminal, but the fitting thread may still be finishing the
        // point it was on when the job was stopped — reading the retained
        // results now would read past it. Wait, briefly and boundedly.
        if (status.results_final === false && this.settlePolls < this.maxSettlePolls) {
          this.settlePolls++;
        } else {
          this.stop();
          this.handlers.onTerminal?.(status);
          return;
        }
      }
    } catch (err) {
      if (this.stopped) return;
      // A transient failure (server restart, brief network loss) must not
      // end the fallback — that would put the UI back in the frozen state
      // this poller exists to prevent. A persistent one must, though:
      // see MAP_JOB_POLL_MAX_ERRORS.
      this.consecutiveErrors++;
      this.handlers.onError?.(err);
      if (this.consecutiveErrors >= this.maxErrors) {
        this.stop();
        this.handlers.onGiveUp?.(err);
        return;
      }
    } finally {
      this.inFlight = false;
    }
    if (this.stopped) return;
    this.timer = setTimeout(() => void this.tick(), this.intervalMs);
  }
}
