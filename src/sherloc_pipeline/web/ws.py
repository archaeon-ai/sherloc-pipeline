"""WebSocket handler for job progress updates.

Change-detection polling (500ms), heartbeat every 30s, 30-min timeout.
Public mode blocked with close code 4003.
"""

import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sherloc_pipeline.web.jobs import JobQueue, JobStatus

router = APIRouter(tags=["websocket"])

# -- tunables --
POLL_INTERVAL = 0.5  # seconds
HEARTBEAT_INTERVAL = 30.0  # seconds
TIMEOUT = 1800  # 30 minutes


@router.websocket("/api/ws/jobs/{job_id}")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream job progress updates over WebSocket with change detection."""
    # Block public mode
    access_mode = getattr(websocket.app.state, "access_mode", "internal")
    if access_mode == "public":
        await websocket.close(code=4003, reason="WebSocket not available in public mode")
        return

    job_queue: JobQueue = websocket.app.state.job_queue
    state = job_queue.get(job_id)

    if state is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()

    last_seq = -1
    last_heartbeat = time.monotonic()
    start_time = time.monotonic()

    try:
        while True:
            # Check timeout
            if time.monotonic() - start_time > TIMEOUT:
                await websocket.close(code=1000, reason="Timeout")
                return

            # Check for client messages (cancel) within the poll interval
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=POLL_INTERVAL
                )
                msg = json.loads(raw)
                if msg.get("type") == "cancel":
                    token = msg.get("submitter_token", "")
                    if job_queue.cancel(job_id, token):
                        await websocket.send_json({
                            "type": "cancelled",
                            "job_id": job_id,
                        })
                        await websocket.close()
                        return
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": "unauthorized",
                            "detail": (
                                "submitter_token does not match or "
                                "job is not cancellable"
                            ),
                        })
            except asyncio.TimeoutError:
                pass

            # Get snapshot and check for changes
            snap = state.snapshot()

            if snap["seq"] != last_seq:
                last_seq = snap["seq"]
                await _send_progress(websocket, snap)
                last_heartbeat = time.monotonic()

                # If terminal, send and close
                if snap["status"] in ("completed", "failed", "cancelled"):
                    await websocket.close()
                    return
            elif time.monotonic() - last_heartbeat > HEARTBEAT_INTERVAL:
                await websocket.send_json({"type": "heartbeat"})
                last_heartbeat = time.monotonic()

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


async def _send_progress(websocket: WebSocket, snap: dict) -> None:
    """Format and send a progress message matching the spec protocol."""
    status = snap["status"]

    if status in ("queued", "running"):
        await websocket.send_json({
            "type": "progress",
            "phase": snap.get("phase") or "unknown",
            "progress": round((snap.get("progress_pct") or 0) / 100, 3),
            "message": snap.get("message") or snap.get("step_label") or "",
        })
    elif status == "completed":
        await websocket.send_json({
            "type": "complete",
            "result": snap.get("result") or {},
        })
    elif status == "failed":
        error = snap.get("error")
        if isinstance(error, dict):
            error_msg = error.get("message", "Unknown error")
        else:
            error_msg = str(error or "Unknown error")
        await websocket.send_json({
            "type": "error",
            "error": error_msg,
        })
    elif status == "cancelled":
        await websocket.send_json({
            "type": "cancelled",
            "job_id": snap.get("job_id"),
        })
