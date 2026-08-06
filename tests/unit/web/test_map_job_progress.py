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
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sherloc_pipeline.services.map_fitting import DomainResult, PointFitResult
from sherloc_pipeline.web.ws_map import (
    RECONNECT_BUFFER_SIZE,
    STALL_WARN_SECONDS,
    MapJobContext,
    MapJobRegistry,
    make_fitting_callbacks,
    router as ws_map_router,
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
# WebSocket endpoint: connecting after fitting has already started
# ---------------------------------------------------------------------------


def _ws_app(registry: MapJobRegistry) -> FastAPI:
    app = FastAPI()
    app.include_router(ws_map_router)
    app.state.access_mode = "internal"
    app.state.map_registry = registry
    return app


def test_late_connect_backlog_reconciles_to_the_status_frame_count():
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

    with TestClient(_ws_app(registry)) as client:
        with client.websocket_connect("/ws/map/mf_late") as ws:
            status_frame = ws.receive_json()
            backlog = [ws.receive_json() for _ in range(4)]

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


def test_late_connect_to_a_queued_job_reports_live_queue_position():
    """The connect frame tells a waiting client it is queued, not frozen."""
    registry = MapJobRegistry()
    running = registry.create("mf_running", "scan-a", loop=_InlineLoop(), n_points=10)
    running.set_status("running")
    waiting = registry.create("mf_waiting", "scan-b", loop=_InlineLoop(), n_points=7)

    app = _ws_app(registry)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/map/mf_waiting") as ws:
            queued_frame = ws.receive_json()

        # The job ahead finishes; a reconnecting client sees position 0.
        running.set_status("complete")
        with client.websocket_connect("/ws/map/mf_waiting") as ws:
            after_frame = ws.receive_json()

    assert queued_frame["status"] == "queued"
    assert queued_frame["queue_position"] == 1
    assert queued_frame["stalled"] is False  # waiting is not stalling
    assert after_frame["queue_position"] == 0
    assert waiting.get_status() == "queued"
