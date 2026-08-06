"""Map Mode WebSocket handler -- per-point fitting result streaming.

This WebSocket is push-based: the fitting thread broadcasts messages to
every connected client, and each handler awaits its own queue. This
differs from the existing ws.py which polls JobState.

Protocol: see docs/specs/MAP_MODE_SPEC.md section 3.2
"""

from __future__ import annotations

import asyncio
import enum
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

# Message types that end the stream. Each is emitted by the fitting thread
# once it has unwound, so everything it produced is already on the queue
# (and in the retention store) behind it.
TERMINAL_MESSAGE_TYPES = ("complete", "error", "cancelled")

# How long the handler holds a client's cancel acknowledgement while it
# waits for the fitting thread's own ``cancelled`` frame.
#
# The thread checks the cancel event between points, so a cancel that
# lands mid-point is only observed after that point has been fitted and
# retained. Acknowledging before then loses it: the client fetches the
# retained results as soon as it sees the acknowledgement, map fitting
# never writes results to the database, and no later frame announces the
# straggler (issue #6).
#
# Bounded, because an unacknowledged cancel is the same frozen panel this
# change exists to remove: past the window the handler acknowledges anyway
# and flags the results as not yet final.
CANCEL_DRAIN_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Map job context and registry
# ---------------------------------------------------------------------------


@dataclass
class MapJobContext:
    """Shared state between fitting thread and WebSocket handler.

    Frames are *broadcast*: the fitting thread hands every message to
    ``publish``, which fans it out to one queue per connected client.
    There is deliberately no single job-wide queue. A client that drops
    and resumes overlaps with its own predecessor -- the server only
    learns the old socket is gone when it next touches it -- and with one
    shared queue the two handlers race to dequeue: whichever won took the
    frame, and if that was the dying one the frame died with it. Losing
    the terminal frame that way leaves the resumed client waiting on a
    job that already finished, which is exactly the frozen panel issue #6
    is about.
    """

    job_id: str
    scan_id: str
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
    # True once no fitting thread can add to ``results`` any more. See
    # mark_results_final().
    results_final: bool = False
    registry: Optional["MapJobRegistry"] = field(
        default=None, repr=False, compare=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # One queue per connected client, and the lock that keeps them in step
    # with the replay buffer. Separate from ``_lock`` so a broadcast never
    # waits on a progress snapshot, and so the two are never nested.
    _subscribers: list[asyncio.Queue] = field(
        default_factory=list, repr=False, compare=False
    )
    _stream_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

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

    def request_cancel(self) -> tuple[bool, str]:
        """Mark the job cancelled, reporting the status it replaced.

        Returns ``(applied, previous_status)``. ``applied`` is False when
        the job was already terminal.

        The previous status is read under the same lock as the write
        because the caller acts on it: a job seen as ``queued`` here can
        never start (the fitting thread's ``set_status("running")`` now
        loses to this terminal state), so nothing is in flight and its
        results are already final. Reading the status in a separate call
        would leave exactly the gap that guarantee is meant to close.
        """
        with self._lock:
            previous = self.status
            if previous in TERMINAL_STATUSES:
                return False, previous
            self.status = "cancelled"
            self.terminal_at = time.monotonic()
            if previous == "queued":
                # The fitting thread had not started, and this terminal
                # status means it never will, so it can add nothing to the
                # retention store.
                self.results_final = True
            return True, previous

    def mark_results_final(self) -> None:
        """Record that the fitting thread has stopped producing results.

        Set as the thread emits its terminal message, which it does after
        the last point it retained. Until then a job can *look* finished
        while a point is still in flight -- a cancel is recorded the moment
        it is requested, but ``run_map_fit`` only notices between points,
        so the point it was on is retained afterwards. A client that reads
        the retention store in that window silently loses that finished
        measurement (issue #6), so both the WebSocket handler and the REST
        status report this rather than status alone.
        """
        with self._lock:
            self.results_final = True

    def results_are_final(self) -> bool:
        """Whether the retained results can no longer change."""
        with self._lock:
            return self.results_final

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

    # -- broadcast stream ---------------------------------------------------
    #
    # ``publish`` runs on the fitting thread; ``subscribe`` / ``unsubscribe``
    # run on the event loop. All three mutate the buffer and the subscriber
    # list under ``_stream_lock``, which is what makes a connect atomic with
    # respect to a broadcast: see subscribe().

    def publish(self, msg: dict, *, replayable: bool = True) -> None:
        """Broadcast one frame to every connected client.

        ``replayable`` messages also go into the reconnect ring buffer.
        Progress frames do not: they are a running total that the next one
        supersedes, so replaying them after a reconnect only re-reports
        numbers the client has already moved past.
        """
        with self._stream_lock:
            if replayable:
                self.message_buffer.append(msg)
            targets = list(self._subscribers)
        for queue in targets:
            try:
                self.loop.call_soon_threadsafe(queue.put_nowait, msg)
            except RuntimeError:
                # Event loop is closed (app shutting down).
                pass

    def subscribe(self, resume_from: int = 0) -> tuple[asyncio.Queue, list[dict]]:
        """Attach a client and take its replay backlog in one step.

        Returns ``(queue, backlog)``: the frames buffered past
        ``resume_from``, and the queue that receives everything after
        them. ``resume_from`` of 0 replays the whole buffer, which is what
        a client connecting mid-job with no history wants.

        Both happen under ``_stream_lock``, so a frame published
        concurrently is either already in the backlog (and not yet
        broadcast to this queue) or not in it (and broadcast to this
        queue) -- never both and never neither. Snapshotting the buffer
        outside the lock would also iterate a deque the fitting thread is
        appending to, which raises ``RuntimeError: deque mutated during
        iteration`` and drops the client mid-replay.
        """
        queue: asyncio.Queue = asyncio.Queue()
        with self._stream_lock:
            backlog = [m for m in self.message_buffer if m.get("seq", 0) > resume_from]
            self._subscribers.append(queue)
        return queue, backlog

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Detach a client. Idempotent."""
        with self._stream_lock:
            for i, existing in enumerate(self._subscribers):
                if existing is queue:
                    del self._subscribers[i]
                    return

    def subscriber_count(self) -> int:
        """How many clients are currently attached to the stream."""
        with self._stream_lock:
            return len(self._subscribers)

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
    """Create callbacks that bridge from the fitting thread to the stream.

    The fitting thread (sync) calls these callbacks, which broadcast the
    frame to every connected client via ``ctx.publish``.

    Returns:
        (on_point_fitted, on_progress, on_log) callback tuple.
    """
    seq_counter = [0]

    def _enqueue(msg: dict, *, point_fitted: bool = False) -> None:
        """Thread-safe broadcast of one replayable frame."""
        fitted = ctx.note_activity(point_fitted=point_fitted)
        if point_fitted:
            # Carry the authoritative running total. A client that connects
            # after fitting started gets a status frame with the count so
            # far AND, right behind it, the buffered per-point frames it
            # never saw; counting those on top of the status frame would
            # double.
            msg["fitted"] = fitted
        if msg.get("type") in TERMINAL_MESSAGE_TYPES:
            # Marked before the frame goes out, so a client that acts on it
            # cannot observe the job as finished while the store it is
            # about to read still says otherwise.
            ctx.mark_results_final()
        ctx.publish(msg)

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
        ctx.publish(msg, replayable=False)

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

    def send_cancelled() -> None:
        """Send the terminal 'cancelled' message.

        Emitted by the fitting thread once it has stopped, so it is
        enqueued *behind* the last point that thread retained. The
        WebSocket handler holds a client's cancel acknowledgement until
        this frame arrives: the client fetches the server's retained
        results the moment it is acknowledged, and a cancel that lands
        mid-point is only noticed after that point has been fitted --
        acknowledging any earlier drops it for good (issue #6).
        """
        seq_counter[0] += 1
        snapshot = ctx.progress_snapshot()
        msg = {
            "type": "cancelled",
            "seq": seq_counter[0],
            "job_id": ctx.job_id,
            "fitted": snapshot["fitted"],
            "total": snapshot["total"],
            # This frame follows the thread's last retained point, so the
            # store the client is about to read is complete. ``_enqueue``
            # records the same thing on the context for REST pollers.
            "results_final": True,
        }
        _enqueue(msg)

    # Attach the lifecycle senders to the callbacks for use by the job runner
    on_point_fitted.send_job_started = send_job_started  # type: ignore[attr-defined]
    on_point_fitted.send_complete = send_complete  # type: ignore[attr-defined]
    on_point_fitted.send_error = send_error  # type: ignore[attr-defined]
    on_point_fitted.send_cancelled = send_cancelled  # type: ignore[attr-defined]

    return on_point_fitted, on_progress, on_log


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


def _cancel_ack(ctx: MapJobContext) -> dict:
    """The handler's own cancel acknowledgement frame.

    Carries no ``seq``: it is not part of the fitting thread's sequenced
    stream. Used only where the handler has to answer without the thread's
    own ``cancelled`` frame -- a job that never started, and a thread that
    did not stop inside ``CANCEL_DRAIN_SECONDS``. ``results_final`` is read
    from the context rather than asserted here, so it says what is
    actually true of the store in both cases.
    """
    snapshot = ctx.progress_snapshot()
    return {
        "type": "cancelled",
        "job_id": ctx.job_id,
        "fitted": snapshot["fitted"],
        "total": snapshot["total"],
        "results_final": ctx.results_are_final(),
    }




class _Control(enum.Enum):
    """Handler-internal items multiplexed onto a client's own stream queue.

    The client's socket and the fitting thread are two independent wake-up
    sources, and the pump used to await both with a single
    ``asyncio.wait`` over a queue task and a receive task. Re-arming those
    two tasks around every branch is what let a client's ``cancel`` and a
    freshly published frame wake the pump in an order that left the other
    one sitting unread until the next timeout -- a queued terminal frame
    with nobody looking at it is the stall this change exists to remove.

    A dedicated reader task now turns client input into these control
    items and posts them to the same per-connection queue the fitting
    thread broadcasts to, so the pump has exactly one wake-up source and
    the two streams are ordered rather than raced.
    """

    CLIENT_GONE = "client_gone"


@dataclass(frozen=True)
class _CancelRequest:
    """A client ``cancel``, with the outcome the reader already applied.

    The cancel is applied to the context by the reader the instant it
    arrives -- the fitting thread should stop as early as possible -- but
    it is *answered* by the pump, in order behind the frames already
    published. ``applied`` / ``previous_status`` are the atomic result of
    ``request_cancel``, carried here so the pump does not re-read a status
    that has since moved on.
    """

    applied: bool
    previous_status: str


async def _client_reader(
    websocket: WebSocket, ctx: MapJobContext, stream: asyncio.Queue
) -> None:
    """Read client messages until the socket closes, posting to ``stream``.

    One long-lived task rather than a per-iteration
    ``wait_for(receive_text())``: cancelling a pending Starlette receive
    can drop the frame it just picked up, which is how a user's "cancel"
    could vanish and leave the UI stuck on a job it could no longer stop.
    """
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                client_msg = json.loads(raw)
            except (ValueError, TypeError):
                client_msg = {}
            if client_msg.get("type") != "cancel":
                continue
            ctx.cancel_event.set()
            applied, previous_status = ctx.request_cancel()
            stream.put_nowait(_CancelRequest(applied, previous_status))
    except WebSocketDisconnect:
        logger.debug("Map WS client disconnected for job %s", ctx.job_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Map WS receive error for job %s", ctx.job_id, exc_info=True)
    stream.put_nowait(_Control.CLIENT_GONE)


@router.websocket("/ws/map/{job_id}")
async def map_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream map-mode fitting results over WebSocket.

    Push-based: the fitting thread broadcasts messages to every connected
    client; this handler drains its own queue and forwards to the socket.

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

    # Resume point. Absent (or unparseable) means "replay everything
    # buffered": a client connecting mid-job has no history, and the
    # frames already fitted are the map it has to draw. Clients that do
    # hold history always send ``last_seq``, and the frontend drops any
    # frame whose seq it has seen, so an overlapping replay costs nothing.
    resume_from = 0
    last_seq_param = websocket.query_params.get("last_seq")
    if last_seq_param is not None:
        try:
            resume_from = int(last_seq_param)
        except ValueError:
            resume_from = 0

    # Attach before anything is sent, and take the replay backlog in the
    # same atomic step, so a frame published while this handler is
    # starting up is delivered exactly once (see MapJobContext.subscribe).
    stream, backlog = ctx.subscribe(resume_from)

    start_time = time.monotonic()
    reader_task: Optional[asyncio.Task] = None
    pump_task: Optional[asyncio.Task] = None
    # Set when a client cancel is being drained: the deadline by which the
    # fitting thread's own terminal frame has to arrive.
    cancel_deadline: Optional[float] = None

    try:
        # Send an immediate status frame so a client that connects behind a
        # long-running job sees "queued (N ahead)" instead of an empty
        # panel. It carries the authoritative fitted count, which the
        # replay behind it can only converge on, never exceed.
        await websocket.send_json({"type": "heartbeat", **ctx.progress_snapshot()})

        for msg in backlog:
            await websocket.send_json(msg)
            if msg.get("type") in TERMINAL_MESSAGE_TYPES:
                # The job ended before this client attached. Say so and
                # close instead of heartbeating a finished job until the
                # 30-minute cap.
                await websocket.close()
                return

        reader_task = asyncio.create_task(_client_reader(websocket, ctx, stream))

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

            if cancel_deadline is not None and time.monotonic() >= cancel_deadline:
                # The fitting thread did not unwind in time. Acknowledge
                # anyway rather than leaving the client watching a job it
                # has already cancelled, but say the retention store may
                # still be missing the point that was in flight.
                await websocket.send_json(_cancel_ack(ctx))
                await websocket.close()
                return

            wait_timeout = HEARTBEAT_INTERVAL
            if cancel_deadline is not None:
                # Wake at the drain deadline so the acknowledgement above
                # is not held for a whole heartbeat interval past it.
                wait_timeout = max(
                    0.0, min(wait_timeout, cancel_deadline - time.monotonic())
                )

            # The pending get is kept across iterations rather than
            # recreated: ``asyncio.wait`` leaves an unfinished task alone
            # on timeout, so the queue always has a registered waiter and
            # an item put while the pump was elsewhere still wakes it.
            if pump_task is None:
                pump_task = asyncio.create_task(stream.get())
            done, _pending = await asyncio.wait({pump_task}, timeout=wait_timeout)

            if pump_task not in done:
                if cancel_deadline is None:
                    # Nothing from either side within the heartbeat
                    # interval: report server-side job state so a silent
                    # job is distinguishable from a dead connection.
                    # Suppressed while a cancel is draining -- that wait
                    # ends in a terminal frame either way, and a heartbeat
                    # in front of it only reports a status the client
                    # already asked for.
                    await websocket.send_json({
                        "type": "heartbeat",
                        **ctx.progress_snapshot(),
                    })
                continue

            item = pump_task.result()
            pump_task = None

            if item is _Control.CLIENT_GONE:
                return

            if isinstance(item, _CancelRequest):
                if not item.applied:
                    # Already terminal -- a cancel racing the end of the
                    # job. The fitting thread's own complete / error /
                    # cancelled frame is on its way (or has already been
                    # replayed); keep draining so it, and the points ahead
                    # of it, still reach the client instead of closing on
                    # an acknowledgement that would contradict it. Answer
                    # with the real state so the request is not met with
                    # silence either.
                    await websocket.send_json({
                        "type": "heartbeat",
                        **ctx.progress_snapshot(),
                    })
                    continue
                if item.previous_status == "queued":
                    # The fitting thread had not started, and the terminal
                    # status it now carries means it never will: nothing is
                    # in flight, so the retention store is already final.
                    # Acknowledging here matters because the single map
                    # executor may not reach this job for minutes.
                    await websocket.send_json(_cancel_ack(ctx))
                    await websocket.close()
                    return
                # Running: the thread is mid-point and retains that point
                # *after* this message was received. Hold the
                # acknowledgement until its own terminal frame arrives
                # behind that point (issue #6).
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + CANCEL_DRAIN_SECONDS
                continue

            await websocket.send_json(item)

            # If terminal message, close cleanly. "cancelled" is one of
            # these: the fitting thread emits it once it has stopped, so
            # it lands behind the last point it retained and is the signal
            # a cancelling client is waiting on.
            if item.get("type") in TERMINAL_MESSAGE_TYPES:
                await websocket.close()
                return

    except WebSocketDisconnect:
        logger.debug("Map WS client disconnected for job %s", job_id)
    except Exception:
        logger.debug("Map WS error for job %s", job_id, exc_info=True)
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # Detached first: the fitting thread must stop broadcasting into a
        # queue nobody drains before the tasks that drained it go away.
        ctx.unsubscribe(stream)
        for task in (pump_task, reader_task):
            if task is not None and not task.done():
                task.cancel()
