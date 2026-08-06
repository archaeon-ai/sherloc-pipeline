"""Map Mode WebSocket handler -- per-point fitting result streaming.

This WebSocket is push-based: the fitting thread pushes messages onto
an asyncio.Queue, and the handler awaits them. This differs from the
existing ws.py which polls JobState.

Protocol: see docs/specs/MAP_MODE_SPEC.md section 3.2
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sherloc_pipeline.services.map_fitting import PointFitResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

HEARTBEAT_INTERVAL = 30.0  # seconds
TIMEOUT = 1800  # 30 minutes
RECONNECT_BUFFER_SIZE = 2000  # max messages to retain for replay
RECONNECT_BUFFER_SECONDS = 300  # 5 min buffer for resume

# Ceiling on per-point results retained for REST retrieval.
#
# ``MapFitService`` streams results and never writes them to the database,
# so a frame lost to a disconnect is lost outright: reloading the map from
# ``fitted_peaks`` afterwards shows whatever an earlier pipeline run wrote,
# not this job's output. Every point is therefore retained on the job
# context, keyed by point index, and served by
# ``GET /api/map/jobs/{job_id}/results``.
#
# Keying by point index bounds the store at the scan's point count (map
# scans run to ~1600 points), and the registry drops it with the job an
# hour after it terminates. The cap is a backstop against a pathological
# scan, not an expected limit -- when it bites, the job reports
# ``truncated`` rather than silently serving a partial map.
MAX_RETAINED_RESULTS = 5000

# A running job that has emitted nothing for this long is reported as
# stalled in the heartbeat frame. The fitting thread is NOT killed --
# the flag exists so the UI can distinguish "slow/queued" from "frozen"
# instead of showing a dead progress panel forever (issue #6).
STALL_WARN_SECONDS = 120.0

# Statuses a job never leaves. Retention and queue accounting both key off
# this set.
TERMINAL_STATUSES = ("complete", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "running")


# ---------------------------------------------------------------------------
# Map job context and registry
# ---------------------------------------------------------------------------


@dataclass
class MapJobContext:
    """Shared state between fitting thread and WebSocket handler."""

    job_id: str
    scan_id: str
    queue: asyncio.Queue  # fitting thread puts messages here
    cancel_event: threading.Event
    message_buffer: deque  # ring buffer for reconnect replay
    created_at: float
    loop: asyncio.AbstractEventLoop  # the event loop that owns the queue
    voronoi: Optional[dict] = None  # set by fitting thread after computation
    status: str = "queued"  # queued | running | complete | failed | cancelled
    n_points: int = 0
    queue_position: int = 0  # fallback when no registry owns this context
    fitted: int = 0  # points streamed so far (authoritative progress counter)
    started_at: Optional[float] = None  # monotonic time the fitting thread began
    terminal_at: Optional[float] = None  # monotonic time the job reached a terminal state
    last_activity: float = 0.0  # monotonic time of the last emitted message
    submit_order: int = 0  # registry-assigned FIFO rank on the map executor
    # Per-point results, keyed by point index, for REST retrieval after a
    # disconnect (see MAX_RETAINED_RESULTS).
    results: dict[int, dict] = field(default_factory=dict)
    results_truncated: bool = False
    registry: Optional["MapJobRegistry"] = field(
        default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.last_activity == 0.0:
            self.last_activity = self.created_at

    def set_status(self, new_status: str) -> bool:
        """Thread-safe status update. Returns True if it was applied.

        Terminal states are sticky. A user can cancel a job while it is
        still waiting its turn on the single map executor, and the fitting
        thread only learns about that when it finally runs -- without
        stickiness its ``set_status("running")`` would resurrect a job the
        user already stopped, putting it back in the active set and
        restarting its retention clock.
        """
        with self._lock:
            if self.status in TERMINAL_STATUSES:
                # First terminal transition wins: a user cancel marks the job
                # cancelled from the WebSocket handler, and the fitting thread
                # re-marks it when it unwinds.
                return False
            self.status = new_status
            if new_status == "running" and self.started_at is None:
                self.started_at = time.monotonic()
            if new_status in TERMINAL_STATUSES:
                self.terminal_at = time.monotonic()
            return True

    def get_status(self) -> str:
        """Thread-safe status read."""
        with self._lock:
            return self.status

    def terminal_since(self) -> Optional[float]:
        """Monotonic time this job reached a terminal state, else ``None``.

        Read atomically with the status so retention decisions can't see a
        terminal status without its timestamp. A context whose status was
        assigned without going through ``set_status`` has no stamp and falls
        back to ``created_at``.
        """
        with self._lock:
            if self.status not in TERMINAL_STATUSES:
                return None
            return self.terminal_at if self.terminal_at is not None else self.created_at

    def note_activity(self, *, point_fitted: bool = False) -> int:
        """Record that the fitting thread produced output (thread-safe).

        Called from the fitting thread on every emitted message so the
        WebSocket heartbeat can report how long the job has been silent.

        Returns the authoritative fitted-point count after this call.
        """
        with self._lock:
            self.last_activity = time.monotonic()
            if point_fitted:
                self.fitted += 1
            return self.fitted

    def retain_result(self, payload: dict) -> None:
        """Retain one point's fit result for later REST retrieval.

        Keyed by point index so a re-emitted point overwrites rather than
        accumulates, and so the store can never outgrow the scan.
        """
        point_index = payload.get("point_index")
        if point_index is None:
            return
        with self._lock:
            if (
                point_index not in self.results
                and len(self.results) >= MAX_RETAINED_RESULTS
            ):
                self.results_truncated = True
                return
            self.results[point_index] = payload

    def results_snapshot(self) -> tuple[list[dict], bool]:
        """Retained per-point results in point order, plus a truncation flag.

        A shallow copy of the list is enough: the payloads are built once by
        the fitting thread and never mutated afterwards.
        """
        with self._lock:
            return (
                [self.results[i] for i in sorted(self.results)],
                self.results_truncated,
            )

    def results_retained(self) -> int:
        """How many per-point results are available for REST retrieval."""
        with self._lock:
            return len(self.results)

    def live_queue_position(self) -> int:
        """Jobs currently ahead of this one on the single map executor.

        Recomputed on every read rather than frozen at submission: with two
        jobs queued behind a running fit, the second one's position has to
        drop as the jobs ahead of it terminate, or heartbeats keep reporting
        finished jobs as still ahead.
        """
        if self.get_status() != "queued":
            return 0
        if self.registry is None:
            return self.queue_position
        return self.registry.position_of(self.job_id)

    def progress_snapshot(self) -> dict:
        """Thread-safe progress/liveness snapshot for status frames."""
        # Computed before the instance lock is taken: the registry walks every
        # job (locking each in turn), so acquiring in the opposite order here
        # would invert the registry -> context lock ordering.
        queue_position = self.live_queue_position()
        now = time.monotonic()
        with self._lock:
            status = self.status
            fitted = self.fitted
            silent_for = now - self.last_activity
            elapsed = now - (self.started_at if self.started_at is not None else self.created_at)
        return {
            "status": status,
            "fitted": fitted,
            "total": self.n_points,
            "queue_position": queue_position,
            "elapsed_s": round(elapsed, 1),
            "since_last_message_s": round(silent_for, 1),
            "stalled": status == "running" and silent_for >= STALL_WARN_SECONDS,
        }


class MapJobRegistry:
    """Thread-safe registry of active map fitting jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, MapJobContext] = {}
        self._submit_counter = 0
        self._lock = threading.Lock()

    def create(
        self,
        job_id: str,
        scan_id: str,
        loop: asyncio.AbstractEventLoop,
        n_points: int = 0,
    ) -> MapJobContext:
        """Create and register a new map job context."""
        ctx = MapJobContext(
            job_id=job_id,
            scan_id=scan_id,
            queue=asyncio.Queue(),
            cancel_event=threading.Event(),
            message_buffer=deque(maxlen=RECONNECT_BUFFER_SIZE),
            created_at=time.monotonic(),
            loop=loop,
            n_points=n_points,
            registry=self,
        )
        with self._lock:
            self._submit_counter += 1
            ctx.submit_order = self._submit_counter
            self._jobs[job_id] = ctx
        return ctx

    def get(self, job_id: str) -> Optional[MapJobContext]:
        """Look up a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        """Remove a job from the registry."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def position_of(self, job_id: str) -> int:
        """Number of still-active jobs submitted ahead of ``job_id``.

        The map executor is single-threaded and FIFO, so submission order is
        execution order: the answer is exactly how many jobs this one waits
        on. Jobs that have since terminated drop out of the count.
        """
        with self._lock:
            ctx = self._jobs.get(job_id)
            if ctx is None:
                return 0
            order = ctx.submit_order
            return sum(
                1
                for other in self._jobs.values()
                if other.submit_order < order
                and other.get_status() in ACTIVE_STATUSES
            )

    def find_active_for_scan(self, scan_id: str) -> Optional[MapJobContext]:
        """Find an active (queued/running) job for a given scan."""
        with self._lock:
            for ctx in self._jobs.values():
                status = ctx.get_status()
                if ctx.scan_id == scan_id and status in ACTIVE_STATUSES:
                    return ctx
            return None

    def cleanup_stale(self, max_age_seconds: float = 3600.0) -> int:
        """Drop terminal jobs that finished more than max_age_seconds ago.

        Retention is measured from the terminal transition, not from
        creation: a fit that ran for two hours is exactly the one whose
        result a client most needs to fetch afterwards, and keying off
        ``created_at`` would reap it the moment the next fit starts --
        taking its REST status and reconnect buffer with it.

        Returns the number of removed jobs.
        """
        now = time.monotonic()
        to_remove = []
        with self._lock:
            for job_id, ctx in self._jobs.items():
                finished_at = ctx.terminal_since()
                if finished_at is not None and now - finished_at > max_age_seconds:
                    to_remove.append(job_id)
            for job_id in to_remove:
                del self._jobs[job_id]
        return len(to_remove)


# ---------------------------------------------------------------------------
# Thread-to-async bridge: callbacks for the fitting thread
# ---------------------------------------------------------------------------


def make_fitting_callbacks(
    ctx: MapJobContext,
):
    """Create callbacks that bridge from the fitting thread to asyncio queue.

    The fitting thread (sync) calls these callbacks, which use
    ``loop.call_soon_threadsafe`` to push messages onto the asyncio Queue.

    Returns:
        (on_point_fitted, on_progress, on_log) callback tuple.
    """
    seq_counter = [0]
    loop = ctx.loop

    def _enqueue(msg: dict, *, point_fitted: bool = False) -> None:
        """Thread-safe push to the asyncio queue."""
        fitted = ctx.note_activity(point_fitted=point_fitted)
        if point_fitted:
            # Carry the authoritative running total. A client that connects
            # after fitting started gets a status frame with the count so
            # far AND, right behind it, the queued per-point frames it never
            # saw; counting those on top of the status frame would double.
            msg["fitted"] = fitted
        ctx.message_buffer.append(msg)
        try:
            loop.call_soon_threadsafe(ctx.queue.put_nowait, msg)
        except RuntimeError:
            # Event loop is closed (client disconnected)
            pass

    def on_point_fitted(result: PointFitResult) -> None:
        seq_counter[0] += 1
        # Retained separately from the wire frame so a client that missed
        # this point entirely can still fetch it: the ring buffer above is
        # sized in messages and wraps on a long scan, while the retention
        # store is keyed by point and holds the whole job.
        payload = {
            "point_index": result.point_index,
            "x": result.x,
            "y": result.y,
            "results": {
                domain: {"status": dr.status, "peaks": dr.peaks}
                for domain, dr in result.results.items()
            },
        }
        ctx.retain_result(payload)
        msg = {"type": "point_fitted", "seq": seq_counter[0], **payload}
        _enqueue(msg, point_fitted=True)

    def on_progress(fitted: int, total: int, elapsed: float, eta: float) -> None:
        seq_counter[0] += 1
        pct = round(fitted / total * 100, 1) if total > 0 else 0.0
        msg = {
            "type": "progress",
            "seq": seq_counter[0],
            "fitted": fitted,
            "total": total,
            "pct": pct,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1),
        }
        # Progress messages are not buffered for replay (transient)
        ctx.note_activity()
        try:
            loop.call_soon_threadsafe(ctx.queue.put_nowait, msg)
        except RuntimeError:
            pass

    def on_log(point_index: int, message: str) -> None:
        seq_counter[0] += 1
        msg = {
            "type": "log",
            "seq": seq_counter[0],
            "point_index": point_index,
            "message": message,
        }
        _enqueue(msg)

    def send_job_started(domains: list[str]) -> None:
        """Announce that the fitting thread actually started running.

        Emitted when the job leaves the map executor queue, not when the
        POST returns: a job submitted behind a long-running fit can sit
        queued for minutes, and the UI needs to tell the two apart.
        """
        seq_counter[0] += 1
        msg = {
            "type": "job_started",
            "seq": seq_counter[0],
            "job_id": ctx.job_id,
            "n_points": ctx.n_points,
            "domains": list(domains),
            "voronoi": ctx.voronoi,
        }
        _enqueue(msg)

    def send_complete(summary_dict: dict) -> None:
        """Send the terminal 'complete' message."""
        seq_counter[0] += 1
        msg = {
            "type": "complete",
            "seq": seq_counter[0],
            "summary": summary_dict,
        }
        _enqueue(msg)

    def send_error(error_msg: str) -> None:
        """Send the terminal 'error' message."""
        seq_counter[0] += 1
        msg = {
            "type": "error",
            "seq": seq_counter[0],
            "error": error_msg,
        }
        _enqueue(msg)

    # Attach the lifecycle senders to the callbacks for use by the job runner
    on_point_fitted.send_job_started = send_job_started  # type: ignore[attr-defined]
    on_point_fitted.send_complete = send_complete  # type: ignore[attr-defined]
    on_point_fitted.send_error = send_error  # type: ignore[attr-defined]

    return on_point_fitted, on_progress, on_log


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/map/{job_id}")
async def map_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream map-mode fitting results over WebSocket.

    Push-based: the fitting thread pushes messages onto an asyncio.Queue;
    this handler awaits them and forwards to the client.

    Supports:
    - Resume via ``last_seq`` query parameter
    - Cancel via ``{"type": "cancel"}`` client message
    - Heartbeat every 30 seconds
    """
    # Block public mode
    access_mode = getattr(websocket.app.state, "access_mode", "internal")
    if access_mode == "public":
        await websocket.close(code=4003, reason="WebSocket not available in public mode")
        return

    # Look up job in registry
    registry: MapJobRegistry = websocket.app.state.map_registry
    ctx = registry.get(job_id)
    if ctx is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()

    # Handle resume: replay buffered messages since last_seq
    last_seq_param = websocket.query_params.get("last_seq")
    if last_seq_param is not None:
        try:
            resume_from = int(last_seq_param)
        except ValueError:
            resume_from = 0
        # Replay buffered messages with seq > resume_from
        for msg in ctx.message_buffer:
            if msg.get("seq", 0) > resume_from:
                try:
                    await websocket.send_json(msg)
                except Exception:
                    return

    start_time = time.monotonic()

    # Send an immediate status frame so a client that connects behind a
    # long-running job sees "queued (N ahead)" instead of an empty panel.
    await websocket.send_json({"type": "heartbeat", **ctx.progress_snapshot()})

    # One long-lived receive task instead of re-arming a 10 ms
    # ``wait_for(receive_text())`` every iteration: cancelling a pending
    # Starlette receive can drop the frame it just picked up, which is how
    # a user's "cancel" could vanish and leave the UI stuck on a job it
    # could no longer stop.
    receive_task: asyncio.Task = asyncio.create_task(websocket.receive_text())
    queue_task: Optional[asyncio.Task] = None

    try:
        while True:
            # Check timeout
            if time.monotonic() - start_time > TIMEOUT:
                await websocket.send_json({
                    "type": "heartbeat",
                    **ctx.progress_snapshot(),
                    "timed_out": True,
                })
                await websocket.close(code=1000, reason="Timeout")
                return

            if queue_task is None:
                queue_task = asyncio.create_task(ctx.queue.get())

            done, _pending = await asyncio.wait(
                {queue_task, receive_task},
                timeout=HEARTBEAT_INTERVAL,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if queue_task in done:
                msg = queue_task.result()
                queue_task = None
                await websocket.send_json(msg)

                # If terminal message, close cleanly
                if msg.get("type") in ("complete", "error"):
                    await websocket.close()
                    return

            if receive_task in done:
                # Raises WebSocketDisconnect when the client goes away,
                # which the handler below turns into a clean exit.
                raw = receive_task.result()
                receive_task = asyncio.create_task(websocket.receive_text())
                try:
                    client_msg = json.loads(raw)
                except (ValueError, TypeError):
                    client_msg = {}
                if client_msg.get("type") == "cancel":
                    ctx.cancel_event.set()
                    ctx.set_status("cancelled")
                    await websocket.send_json({
                        "type": "cancelled",
                        "job_id": job_id,
                    })
                    await websocket.close()
                    return

            if not done:
                # Nothing from either side within the heartbeat interval:
                # report server-side job state so a silent job is
                # distinguishable from a dead connection.
                await websocket.send_json({
                    "type": "heartbeat",
                    **ctx.progress_snapshot(),
                })

    except WebSocketDisconnect:
        logger.debug("Map WS client disconnected for job %s", job_id)
    except Exception:
        logger.debug("Map WS error for job %s", job_id, exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        for task in (queue_task, receive_task):
            if task is not None and not task.done():
                task.cancel()
