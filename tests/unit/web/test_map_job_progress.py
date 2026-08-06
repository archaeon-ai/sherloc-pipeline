"""Tests for map-fit job liveness reporting (issue #6).

Map Mode fitting runs on a single-threaded executor and streams results
over a WebSocket. When a job is queued behind another scan's fit, or is
simply slow, the client previously received nothing but a contentless
``{"type": "heartbeat"}`` — indistinguishable from a frozen UI.

These tests pin the observable signals that fix that:

* ``MapJobContext.progress_snapshot()`` reports status, progress, queue
  position and how long the job has been silent.
* The snapshot's ``fitted`` counter survives the reconnect ring buffer
  wrapping (the old count-the-buffer approach silently undercounted).
* ``MapJobRegistry.position_of()`` reports a *live* queue position that
  shrinks as the jobs ahead of a waiting one terminate.
* Terminal jobs are retained relative to when they finished, not when
  they were created, so a fit that ran for hours survives long enough to
  be read back.
* ``make_fitting_callbacks`` emits a ``job_started`` frame when the
  fitting thread actually starts, keeps the liveness fields current, and
  stamps each ``point_fitted`` frame with the authoritative count so a
  late-connecting client can reconcile the backlog it is handed.
* Terminal statuses are sticky, so a job cancelled while it waits its
  turn on the single-threaded map executor cannot be restarted by the
  fitting thread that eventually picks it up.
* The ``/ws/map/{job_id}`` handler itself: connect frame, resume replay,
  client cancel, and the two refusal paths.
* ``GET /api/map/jobs/{job_id}``, the REST fallback, carries the same
  queued / stalled / progress signals rather than collapsing them.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from sherloc_pipeline.services.map_fitting import DomainResult, PointFitResult
from sherloc_pipeline.web.routes.map import get_map_job_status
from sherloc_pipeline.web.ws_map import (
    RECONNECT_BUFFER_SIZE,
    STALL_WARN_SECONDS,
    MapJobContext,
    MapJobRegistry,
    make_fitting_callbacks,
    map_ws,
)


def _make_ctx(n_points: int = 10, loop=None) -> MapJobContext:
    """A standalone context, i.e. one no registry owns."""
    return MapJobContext(
        job_id="mf_test",
        scan_id="scan-1",
        queue=asyncio.Queue(),
        cancel_event=threading.Event(),
        message_buffer=collections.deque(maxlen=RECONNECT_BUFFER_SIZE),
        created_at=time.monotonic(),
        loop=loop,
        n_points=n_points,
    )


class _InlineLoop:
    """Stand-in for the app event loop used by the fitting-thread bridge.

    The real callbacks hand messages to the loop that owns the queue. When
    no client is connected yet nothing is awaiting the queue, so running
    the callback inline is equivalent -- and lets a test build up the
    undelivered backlog a late-connecting client will be handed.
    """

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


def _fit_points(on_point_fitted, indices) -> None:
    for i in indices:
        on_point_fitted(
            PointFitResult(
                point_index=i,
                x=float(i),
                y=0.0,
                results={"minerals": DomainResult(status="below_threshold")},
            )
        )


# ---------------------------------------------------------------------------
# progress_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reports_queued_job_with_queue_position():
    # No registry owns this context, so the snapshot falls back to the
    # statically-assigned position (see the registry tests below for the
    # live one).
    ctx = _make_ctx(n_points=42)
    ctx.queue_position = 2

    snap = ctx.progress_snapshot()

    assert snap["status"] == "queued"
    assert snap["fitted"] == 0
    assert snap["total"] == 42
    assert snap["queue_position"] == 2
    # A queued job is waiting, not stalled — the distinction is the point.
    assert snap["stalled"] is False


def test_snapshot_counts_points_and_clears_silence_on_activity():
    ctx = _make_ctx()
    ctx.set_status("running")
    ctx.last_activity = time.monotonic() - (STALL_WARN_SECONDS + 5)

    assert ctx.progress_snapshot()["stalled"] is True

    ctx.note_activity(point_fitted=True)
    snap = ctx.progress_snapshot()

    assert snap["stalled"] is False
    assert snap["fitted"] == 1
    assert snap["since_last_message_s"] < 1.0


def test_snapshot_stall_flag_only_applies_to_running_jobs():
    ctx = _make_ctx()
    ctx.last_activity = time.monotonic() - (STALL_WARN_SECONDS * 3)

    # Still queued: silence is expected, not a stall.
    assert ctx.progress_snapshot()["stalled"] is False

    ctx.set_status("complete")
    assert ctx.progress_snapshot()["stalled"] is False


def test_fitted_counter_survives_ring_buffer_wrap():
    """The old REST fallback counted point_fitted messages in the ring
    buffer, so scans longer than the buffer looked like they stopped."""
    ctx = _make_ctx(n_points=RECONNECT_BUFFER_SIZE + 100)
    ctx.set_status("running")

    for _ in range(RECONNECT_BUFFER_SIZE + 100):
        ctx.message_buffer.append({"type": "point_fitted"})
        ctx.note_activity(point_fitted=True)

    assert len(ctx.message_buffer) == RECONNECT_BUFFER_SIZE
    assert ctx.progress_snapshot()["fitted"] == RECONNECT_BUFFER_SIZE + 100


def test_set_status_running_starts_the_elapsed_clock_once():
    ctx = _make_ctx()
    assert ctx.started_at is None

    ctx.set_status("running")
    first = ctx.started_at
    assert first is not None

    ctx.set_status("running")
    assert ctx.started_at == first


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_cleanup_stale_removes_only_aged_terminal_jobs():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        old_done = registry.create("mf_old", "scan-a", loop=loop)
        old_running = registry.create("mf_run", "scan-b", loop=loop)
        old_done.set_status("complete")
        old_running.set_status("running")
        old_done.created_at -= 7200
        old_done.terminal_at -= 7200
        old_running.created_at -= 7200

        removed = registry.cleanup_stale(max_age_seconds=3600.0)

        assert removed == 1
        assert registry.get("mf_old") is None
        assert registry.get("mf_run") is not None
    finally:
        loop.close()


def test_cleanup_stale_retains_a_long_running_job_that_just_finished():
    """Retention runs from the terminal transition, not from creation.

    A fit that ran longer than the retention window is exactly the one a
    client most needs to read back; keying off ``created_at`` reaped it the
    moment the next fit called ``cleanup_stale()``.
    """
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        long_job = registry.create("mf_long", "scan-a", loop=loop, n_points=5000)
        long_job.created_at -= 7200  # started two hours ago
        long_job.set_status("running")
        long_job.set_status("complete")  # ... and finished just now

        assert registry.cleanup_stale(max_age_seconds=3600.0) == 0
        assert registry.get("mf_long") is not None

        # It is reaped once the retention window elapses after completion.
        long_job.terminal_at -= 7200
        assert registry.cleanup_stale(max_age_seconds=3600.0) == 1
        assert registry.get("mf_long") is None
    finally:
        loop.close()


def test_queue_position_shrinks_as_jobs_ahead_terminate():
    """Queued jobs must not keep reporting finished jobs as still ahead."""
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        first = registry.create("mf_1", "scan-a", loop=loop)
        second = registry.create("mf_2", "scan-b", loop=loop)
        third = registry.create("mf_3", "scan-c", loop=loop)
        first.set_status("running")

        assert first.progress_snapshot()["queue_position"] == 0
        assert second.progress_snapshot()["queue_position"] == 1
        assert third.progress_snapshot()["queue_position"] == 2

        # First job finishes; the executor picks up the second.
        first.set_status("complete")
        second.set_status("running")

        assert second.progress_snapshot()["queue_position"] == 0
        assert third.progress_snapshot()["queue_position"] == 1

        second.set_status("failed")
        assert third.progress_snapshot()["queue_position"] == 0

        # A terminal job is not "queued behind" anything.
        assert first.progress_snapshot()["queue_position"] == 0
    finally:
        loop.close()


def test_queue_position_counts_only_jobs_submitted_earlier():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        first = registry.create("mf_1", "scan-a", loop=loop)
        first.set_status("running")
        second = registry.create("mf_2", "scan-b", loop=loop)
        third = registry.create("mf_3", "scan-c", loop=loop)

        # Three jobs are active, but only one precedes mf_2 in the FIFO.
        assert second.progress_snapshot()["queue_position"] == 1
        assert third.progress_snapshot()["queue_position"] == 2
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_started_frame_is_emitted_and_buffered():
    loop = asyncio.get_running_loop()
    ctx = _make_ctx(n_points=3, loop=loop)
    on_point_fitted, _on_progress, _on_log = make_fitting_callbacks(ctx)

    on_point_fitted.send_job_started(["minerals", "organics"])
    await asyncio.sleep(0)

    msg = ctx.queue.get_nowait()
    assert msg["type"] == "job_started"
    assert msg["job_id"] == "mf_test"
    assert msg["n_points"] == 3
    assert msg["domains"] == ["minerals", "organics"]
    # Buffered so a reconnecting client can replay it.
    assert ctx.message_buffer[-1]["type"] == "job_started"


@pytest.mark.asyncio
async def test_point_fitted_callback_updates_liveness_counters():
    loop = asyncio.get_running_loop()
    ctx = _make_ctx(n_points=2, loop=loop)
    ctx.set_status("running")
    ctx.last_activity = time.monotonic() - (STALL_WARN_SECONDS + 5)
    on_point_fitted, on_progress, _on_log = make_fitting_callbacks(ctx)

    on_point_fitted(
        PointFitResult(
            point_index=0,
            x=1.0,
            y=2.0,
            results={"minerals": DomainResult(status="below_threshold")},
        )
    )
    await asyncio.sleep(0)

    snap = ctx.progress_snapshot()
    assert snap["fitted"] == 1
    assert snap["stalled"] is False

    # Progress frames are transient (not buffered) but still count as
    # liveness — a job emitting only progress is not stalled.
    ctx.last_activity = time.monotonic() - (STALL_WARN_SECONDS + 5)
    on_progress(1, 2, 3.0, 3.0)
    await asyncio.sleep(0)
    assert ctx.progress_snapshot()["stalled"] is False
    assert ctx.progress_snapshot()["fitted"] == 1


@pytest.mark.asyncio
async def test_point_fitted_frames_carry_the_authoritative_count():
    loop = asyncio.get_running_loop()
    ctx = _make_ctx(n_points=3, loop=loop)
    ctx.set_status("running")
    on_point_fitted, _on_progress, _on_log = make_fitting_callbacks(ctx)

    _fit_points(on_point_fitted, range(3))
    await asyncio.sleep(0)

    counts = [ctx.queue.get_nowait()["fitted"] for _ in range(3)]
    assert counts == [1, 2, 3]


# ---------------------------------------------------------------------------
# Terminal statuses are sticky
# ---------------------------------------------------------------------------


def test_a_cancelled_job_cannot_be_reactivated():
    """A job cancelled while queued must stay cancelled.

    The fitting thread only learns it was cancelled when the single map
    executor finally reaches it. If its ``set_status("running")`` won, the
    job would rejoin the active set, restart its retention clock, and hold
    up every job behind it in the queue position count.
    """
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        ctx = registry.create("mf_cancelled", "scan-a", loop=loop, n_points=10)
        behind = registry.create("mf_behind", "scan-b", loop=loop, n_points=10)
        assert behind.progress_snapshot()["queue_position"] == 1

        ctx.cancel_event.set()
        assert ctx.set_status("cancelled") is True
        cancelled_at = ctx.terminal_at

        # The fitting thread wakes up and tries to start.
        assert ctx.set_status("running") is False

        assert ctx.get_status() == "cancelled"
        assert ctx.terminal_at == cancelled_at
        assert ctx.started_at is None
        assert behind.progress_snapshot()["queue_position"] == 0
    finally:
        loop.close()


def test_terminal_status_survives_a_later_failure_report():
    """The unwinding fitting thread must not overwrite the user's cancel."""
    ctx = _make_ctx()
    ctx.set_status("running")
    ctx.set_status("cancelled")

    assert ctx.set_status("failed") is False
    assert ctx.get_status() == "cancelled"


# ---------------------------------------------------------------------------
# WebSocket endpoint: connecting after fitting has already started
#
# The endpoint coroutine is driven directly against an in-process stub
# rather than through ``TestClient.websocket_connect``. That harness runs
# the ASGI app on a portal thread behind httpx and the ``websockets``
# sans-io layer, and a blocking read on it has no test-side deadline: how
# it behaves against a long-lived handler depends on which versions of
# that stack are installed, and a failure mode there costs the whole suite
# rather than one test. ``map_ws`` only calls accept / send_json /
# receive_text / close, so a plain stub drives the same code on one event
# loop with an explicit timeout on every wait -- and reaches the
# client-cancel path, which needs a bidirectional exchange with a handler
# that never returns on its own.
# ---------------------------------------------------------------------------


class _StubApp:
    def __init__(self, registry: MapJobRegistry, access_mode: str = "internal"):
        self.state = SimpleNamespace(map_registry=registry, access_mode=access_mode)


class _StubWebSocket:
    """Minimal stand-in for a Starlette WebSocket."""

    def __init__(
        self,
        registry: MapJobRegistry,
        *,
        access_mode: str = "internal",
        query_params: dict | None = None,
        client_messages: list[str] | None = None,
    ):
        self.app = _StubApp(registry, access_mode)
        self.query_params = query_params or {}
        self.sent: list[dict] = []
        self.accepted = False
        self.closed: tuple[int, str] | None = None
        self._inbound: asyncio.Queue = asyncio.Queue()
        for raw in client_messages or []:
            self._inbound.put_nowait(raw)

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive_text(self) -> str:
        return await self._inbound.get()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed is None:
            self.closed = (code, reason)


async def _await_frames(ws: _StubWebSocket, count: int, timeout: float = 2.0) -> None:
    """Wait until the handler has sent ``count`` frames, or fail loudly.

    Bounded rather than open-ended: a regression that leaves the handler
    silent must surface as a failed assertion, never as a hung suite.
    """
    deadline = time.monotonic() + timeout
    while len(ws.sent) < count:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"handler sent {len(ws.sent)}/{count} frames in {timeout}s: {ws.sent}"
            )
        await asyncio.sleep(0.005)


@asynccontextmanager
async def _connected(ws: _StubWebSocket, job_id: str):
    """Run ``map_ws`` for the duration of the block, then always tear down."""
    task = asyncio.create_task(map_ws(ws, job_id))
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # Let the handler's own cancelled child tasks settle before the
        # test's event loop is torn down.
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_late_connect_backlog_reconciles_to_the_status_frame_count():
    """A client that connects mid-job gets a count *and* the frames behind it.

    The immediate status frame reports the authoritative fitted count, and
    the per-point frames the client never received are then drained from
    the job queue. Those frames describe the very points the count already
    covers, so a client that adds them to it reports twice the real
    progress. Each frame therefore identifies its point and carries the
    running total, so reconciling by identity (or by max) converges on the
    status frame's number instead of doubling it.
    """
    registry = MapJobRegistry()
    ctx = registry.create("mf_late", "scan-late", loop=_InlineLoop(), n_points=10)
    ctx.set_status("running")
    on_point_fitted, _on_progress, _on_log = make_fitting_callbacks(ctx)

    # Four points fitted before any client connected: their frames sit
    # undelivered on the job queue.
    _fit_points(on_point_fitted, range(4))
    assert ctx.progress_snapshot()["fitted"] == 4

    ws = _StubWebSocket(registry)
    async with _connected(ws, "mf_late"):
        await _await_frames(ws, 5)

    status_frame, *backlog = ws.sent[:5]
    assert ws.accepted is True
    assert status_frame["type"] == "heartbeat"
    assert status_frame["status"] == "running"
    assert status_frame["fitted"] == 4
    assert status_frame["total"] == 10

    assert [f["type"] for f in backlog] == ["point_fitted"] * 4
    indices = [f["point_index"] for f in backlog]
    assert indices == [0, 1, 2, 3]
    # Absolute, never ahead of the count the status frame already reported.
    assert [f["fitted"] for f in backlog] == [1, 2, 3, 4]

    reconciled = max(len(set(indices)), max(f["fitted"] for f in backlog))
    assert reconciled == status_frame["fitted"]  # 4, not 8


@pytest.mark.asyncio
async def test_late_connect_to_a_queued_job_reports_live_queue_position():
    """The connect frame tells a waiting client it is queued, not frozen."""
    registry = MapJobRegistry()
    running = registry.create("mf_running", "scan-a", loop=_InlineLoop(), n_points=10)
    running.set_status("running")
    waiting = registry.create("mf_waiting", "scan-b", loop=_InlineLoop(), n_points=7)

    first = _StubWebSocket(registry)
    async with _connected(first, "mf_waiting"):
        await _await_frames(first, 1)

    # The job ahead finishes; a reconnecting client sees position 0.
    running.set_status("complete")
    second = _StubWebSocket(registry)
    async with _connected(second, "mf_waiting"):
        await _await_frames(second, 1)

    queued_frame = first.sent[0]
    after_frame = second.sent[0]
    assert queued_frame["status"] == "queued"
    assert queued_frame["queue_position"] == 1
    assert queued_frame["stalled"] is False  # waiting is not stalling
    assert after_frame["queue_position"] == 0
    assert waiting.get_status() == "queued"


@pytest.mark.asyncio
async def test_client_cancel_marks_the_job_and_closes():
    """Cancelling a *queued* job must stop it, not merely flag it.

    ``_run_fit`` checks ``cancel_event`` before it starts, so the event has
    to be set here — a status-only cancel would let the executor start the
    job anyway when it reached it.
    """
    registry = MapJobRegistry()
    ctx = registry.create("mf_cancel", "scan-a", loop=_InlineLoop(), n_points=10)

    ws = _StubWebSocket(registry, client_messages=['{"type": "cancel"}'])
    async with _connected(ws, "mf_cancel") as task:
        await _await_frames(ws, 2)
        await asyncio.wait_for(task, timeout=2.0)

    assert ctx.cancel_event.is_set() is True
    assert ctx.get_status() == "cancelled"
    assert ws.sent[-1] == {"type": "cancelled", "job_id": "mf_cancel"}
    assert ws.closed is not None
    # And the executor, reaching it later, cannot put it back to work.
    assert ctx.set_status("running") is False


@pytest.mark.asyncio
async def test_resume_replays_only_frames_after_last_seq():
    registry = MapJobRegistry()
    ctx = registry.create("mf_resume", "scan-a", loop=_InlineLoop(), n_points=10)
    ctx.set_status("running")
    on_point_fitted, _on_progress, _on_log = make_fitting_callbacks(ctx)
    _fit_points(on_point_fitted, range(3))
    # Drain the live queue so only the replay path can produce these.
    while not ctx.queue.empty():
        ctx.queue.get_nowait()

    ws = _StubWebSocket(registry, query_params={"last_seq": "1"})
    async with _connected(ws, "mf_resume"):
        await _await_frames(ws, 3)

    replayed = [f for f in ws.sent if f["type"] == "point_fitted"]
    assert [f["seq"] for f in replayed] == [2, 3]


@pytest.mark.asyncio
async def test_unknown_job_is_refused_without_accepting():
    ws = _StubWebSocket(MapJobRegistry())
    await asyncio.wait_for(map_ws(ws, "mf_nope"), timeout=2.0)

    assert ws.accepted is False
    assert ws.closed == (4004, "Job not found")


@pytest.mark.asyncio
async def test_public_mode_refuses_the_socket():
    registry = MapJobRegistry()
    registry.create("mf_public", "scan-a", loop=_InlineLoop())
    ws = _StubWebSocket(registry, access_mode="public")

    await asyncio.wait_for(map_ws(ws, "mf_public"), timeout=2.0)

    assert ws.accepted is False
    assert ws.closed is not None and ws.closed[0] == 4003


# ---------------------------------------------------------------------------
# REST polling fallback: GET /api/map/jobs/{job_id}
#
# The route reads nothing but ``request.app.state``, so it is called
# directly rather than through an HTTP stack.
# ---------------------------------------------------------------------------


def _status_request(registry: MapJobRegistry):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(map_registry=registry, job_queue=None))
    )


def test_rest_status_reports_queued_with_its_queue_position():
    """The fallback must not collapse "waiting" into "running".

    A client polling because its WebSocket is unavailable is in exactly
    the situation issue #6 describes; reporting a queued job as running
    with 0/N points is the frozen-panel symptom, restated over REST.
    """
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        running = registry.create("mf_a", "scan-a", loop=loop, n_points=10)
        running.set_status("running")
        registry.create("mf_b", "scan-b", loop=loop, n_points=7)

        resp = get_map_job_status(_status_request(registry), "mf_b")

        assert resp.status == "queued"
        assert resp.queue_position == 1
        assert resp.stalled is False
        assert resp.total == 7
        assert resp.fitted == 0
        assert resp.results_available is False
    finally:
        loop.close()


def test_rest_status_reports_a_stalled_running_job():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        ctx = registry.create("mf_slow", "scan-a", loop=loop, n_points=500)
        ctx.set_status("running")
        ctx.note_activity(point_fitted=True)
        ctx.last_activity = time.monotonic() - (STALL_WARN_SECONDS + 30)

        resp = get_map_job_status(_status_request(registry), "mf_slow")

        assert resp.status == "running"
        assert resp.stalled is True
        assert resp.since_last_message_s >= STALL_WARN_SECONDS
        assert resp.fitted == 1
        assert resp.queue_position == 0
    finally:
        loop.close()


def test_rest_status_reports_completion_and_offers_results():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        ctx = registry.create("mf_done", "scan-a", loop=loop, n_points=3)
        ctx.set_status("running")
        for _ in range(3):
            ctx.note_activity(point_fitted=True)
        ctx.set_status("complete")

        resp = get_map_job_status(_status_request(registry), "mf_done")

        assert resp.status == "complete"
        assert resp.results_available is True
        assert (resp.fitted, resp.total) == (3, 3)
        assert resp.stalled is False
    finally:
        loop.close()
