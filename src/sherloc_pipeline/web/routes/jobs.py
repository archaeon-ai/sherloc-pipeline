"""REST endpoint for job status (polling fallback)."""

from fastapi import APIRouter, HTTPException, Request

from sherloc_pipeline.web.jobs import JobQueue
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/{job_id}")
def get_job_status(request: Request, job_id: str):
    """Return current job status.  Used as polling fallback when WS unavailable."""
    job_queue: JobQueue = request.app.state.job_queue
    state = job_queue.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    snap = state.snapshot()

    # Format response to match WS message protocol
    status = snap["status"]
    response = {"schema_version": API_SCHEMA_VERSION, "job_id": job_id, "status": status}

    if status == "running":
        response["type"] = "progress"
        response["phase"] = snap.get("phase")
        response["progress"] = round((snap.get("progress_pct") or 0) / 100, 3)
        response["message"] = snap.get("message") or snap.get("step_label") or ""
    elif status == "completed":
        response["type"] = "complete"
        response["progress"] = 1.0
        response["result"] = snap.get("result")
    elif status == "failed":
        response["type"] = "error"
        error = snap.get("error")
        if isinstance(error, dict):
            response["error"] = error.get("message", "Unknown error")
        else:
            response["error"] = str(error or "Unknown error")
    elif status == "queued":
        response["type"] = "queued"
        response["queue_position"] = job_queue.queue_position(job_id)

    return response
