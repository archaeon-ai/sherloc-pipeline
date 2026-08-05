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
* ``MapJobRegistry.count_active()`` counts only queued/running jobs, so
  a new job can report how many are ahead of it.
* ``make_fitting_callbacks`` emits a ``job_started`` frame when the
  fitting thread actually starts, and keeps the liveness fields current.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time

import pytest

from sherloc_pipeline.services.map_fitting import DomainResult, PointFitResult
from sherloc_pipeline.web.ws_map import (
    RECONNECT_BUFFER_SIZE,
    STALL_WARN_SECONDS,
    MapJobContext,
    MapJobRegistry,
    make_fitting_callbacks,
)


def _make_ctx(n_points: int = 10, loop=None) -> MapJobContext:
    return MapJobContext(
        job_id="mf_test",
        scan_id="scan-1",
        queue=asyncio.Queue() if loop is None else asyncio.Queue(),
        cancel_event=threading.Event(),
        message_buffer=collections.deque(maxlen=RECONNECT_BUFFER_SIZE),
        created_at=time.monotonic(),
        loop=loop,
        n_points=n_points,
    )


# ---------------------------------------------------------------------------
# progress_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reports_queued_job_with_queue_position():
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


def test_count_active_ignores_terminal_jobs():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        running = registry.create("mf_a", "scan-a", loop=loop, n_points=5)
        registry.create("mf_b", "scan-b", loop=loop, n_points=5)  # queued
        done = registry.create("mf_c", "scan-c", loop=loop, n_points=5)

        running.set_status("running")
        done.set_status("complete")

        assert registry.count_active() == 2
    finally:
        loop.close()


def test_cleanup_stale_removes_only_aged_terminal_jobs():
    registry = MapJobRegistry()
    loop = asyncio.new_event_loop()
    try:
        old_done = registry.create("mf_old", "scan-a", loop=loop)
        old_running = registry.create("mf_run", "scan-b", loop=loop)
        old_done.set_status("complete")
        old_running.set_status("running")
        old_done.created_at -= 7200
        old_running.created_at -= 7200

        removed = registry.cleanup_stale(max_age_seconds=3600.0)

        assert removed == 1
        assert registry.get("mf_old") is None
        assert registry.get("mf_run") is not None
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
