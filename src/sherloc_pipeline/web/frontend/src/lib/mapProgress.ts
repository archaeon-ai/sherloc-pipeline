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
