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

# A running job that has emitted nothing for this long is reported as
# stalled in the heartbeat frame. The fitting thread is NOT killed --
# the flag exists so the UI can distinguish "slow/queued" from "frozen"
# instead of showing a dead progress panel forever (issue #6).
STALL_WARN_SECONDS = 120.0


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
    queue_position: int = 0  # jobs ahead of this one on the map executor
    fitted: int = 0  # points streamed so far (authoritative progress counter)
    started_at: Optional[float] = None  # monotonic time the fitting thread began
    last_activity: float = 0.0  # monotonic time of the last emitted message
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.last_activity == 0.0:
            self.last_activity = self.created_at

    def set_status(self, new_status: str) -> None:
        """Thread-safe status update."""
        with self._lock:
            self.status = new_status
            if new_status == "running" and self.started_at is None:
                self.started_at = time.monotonic()

    def get_status(self) -> str:
        """Thread-safe status read."""
        with self._lock:
            return self.status

    def note_activity(self, *, point_fitted: bool = False) -> None:
        """Record that the fitting thread produced output (thread-safe).

        Called from the fitting thread on every emitted message so the
        WebSocket heartbeat can report how long the job has been silent.
        """
        with self._lock:
            self.last_activity = time.monotonic()
            if point_fitted:
                self.fitted += 1

    def progress_snapshot(self) -> dict:
        """Thread-safe progress/liveness snapshot for status frames."""
        now = time.monotonic()
        with self._lock:
            status = self.status
            fitted = self.fitted
            silent_for = now - self.last_activity
            elapsed = now - (self.started_at if self.started_at is not None else self.created_at)
            queue_position = self.queue_position
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
        )
        with self._lock:
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

    def count_active(self) -> int:
        """Number of jobs currently queued or running on the map executor."""
        with self._lock:
            return sum(
                1
                for ctx in self._jobs.values()
                if ctx.get_status() in ("queued", "running")
            )

    def find_active_for_scan(self, scan_id: str) -> Optional[MapJobContext]:
        """Find an active (queued/running) job for a given scan."""
        with self._lock:
            for ctx in self._jobs.values():
                status = ctx.get_status()
                if ctx.scan_id == scan_id and status in ("queued", "running"):
                    return ctx
            return None

    def cleanup_stale(self, max_age_seconds: float = 3600.0) -> int:
        """Remove jobs older than max_age_seconds that are in terminal state.

        Returns the number of removed jobs.
        """
        now = time.monotonic()
        to_remove = []
        with self._lock:
            for job_id, ctx in self._jobs.items():
                status = ctx.get_status()
                age = now - ctx.created_at
                if age > max_age_seconds and status in (
                    "complete",
                    "failed",
                    "cancelled",
                ):
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

    def _enqueue(msg: dict) -> None:
        """Thread-safe push to the asyncio queue."""
        ctx.message_buffer.append(msg)
        ctx.note_activity(point_fitted=msg.get("type") == "point_fitted")
        try:
            loop.call_soon_threadsafe(ctx.queue.put_nowait, msg)
        except RuntimeError:
            # Event loop is closed (client disconnected)
            pass

    def on_point_fitted(result: PointFitResult) -> None:
        seq_counter[0] += 1
        msg = {
            "type": "point_fitted",
            "seq": seq_counter[0],
            "point_index": result.point_index,
            "x": result.x,
            "y": result.y,
            "results": {
                domain: {"status": dr.status, "peaks": dr.peaks}
                for domain, dr in result.results.items()
            },
        }
        _enqueue(msg)

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
