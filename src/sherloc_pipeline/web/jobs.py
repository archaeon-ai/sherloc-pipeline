"""Job queue for long-running operations (e.g. PDS download).

FIFO queue with depth 3, sequential execution via ThreadPoolExecutor(max_workers=1).
Jobs are in-memory only -- lost on server restart.
"""

import logging
import secrets
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_QUEUE_DEPTH = 3


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobState:
    """Mutable state for a single job.

    All mutable fields are guarded by ``_lock``.  Background threads call
    :meth:`update` to mutate progress fields; the async WS handler calls
    :meth:`snapshot` to read them.  The monotonic ``seq`` counter lets
    consumers detect changes without diffing the full dict.
    """

    job_id: str
    scan_id: str
    status: JobStatus = JobStatus.QUEUED
    submitter_token: str = ""
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    step: Optional[str] = None
    step_label: Optional[str] = None
    step_index: Optional[int] = None
    progress_pct: Optional[int] = None
    phase: Optional[str] = None
    message: Optional[str] = None
    seq: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    _start_time: Optional[float] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self._start_time is None:
            return None
        end = time.time()
        return round(end - self._start_time, 1)

    def update(
        self,
        *,
        phase: Optional[str] = None,
        message: Optional[str] = None,
        progress_pct: Optional[float] = None,
        step: Optional[str] = None,
        step_label: Optional[str] = None,
    ) -> None:
        """Thread-safe update of progress fields.  Bumps ``seq``."""
        with self._lock:
            if phase is not None:
                self.phase = phase
            if message is not None:
                self.message = message
            if progress_pct is not None:
                self.progress_pct = progress_pct
            if step is not None:
                self.step = step
            if step_label is not None:
                self.step_label = step_label
            self.seq += 1

    def snapshot(self) -> dict:
        """Thread-safe read of all progress fields."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status.value,
                "phase": self.phase,
                "message": self.message,
                "progress_pct": self.progress_pct,
                "seq": self.seq,
                "step": self.step,
                "step_label": self.step_label,
                "result": self.result,
                "error": self.error,
                "elapsed_seconds": self.elapsed_seconds,
                "scan_id": self.scan_id,
            }


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_job_id() -> str:
    return f"j_{secrets.token_hex(16)}"


def _make_token() -> str:
    return f"t_{secrets.token_hex(16)}"


class JobQueue:
    """In-memory FIFO job queue with single-worker executor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobState] = {}
        self._pending: deque = deque(maxlen=MAX_QUEUE_DEPTH)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running_job_id: Optional[str] = None
        self._futures: Dict[str, Future] = {}

    # -- public API --

    def submit(
        self,
        scan_id: str,
        func: Callable[..., Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> JobState:
        """Submit a job.  Returns the initial JobState.

        Raises ValueError if queue is full.
        """
        with self._lock:
            self._evict_stale()

            # Count pending + running
            active = sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            )
            if active >= MAX_QUEUE_DEPTH:
                raise ValueError("queue_full")

            job_id = _make_job_id()
            token = _make_token()
            state = JobState(
                job_id=job_id,
                scan_id=scan_id,
                submitter_token=token,
                created_at=_utc_iso(),
            )
            self._jobs[job_id] = state

            # If nothing is running, start immediately
            if self._running_job_id is None:
                self._start_job(state, func, args, kwargs)
            else:
                self._pending.append((state, func, args, kwargs))

        return state

    def get(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def find_active_by_scan_id(self, scan_id: str) -> Optional[JobState]:
        """Find an active (queued/running) job for the given scan_id."""
        with self._lock:
            for j in self._jobs.values():
                if j.scan_id == scan_id and j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    return j
            return None

    def cancel(self, job_id: str, submitter_token: str) -> bool:
        """Cancel a queued job.  Returns True if cancelled."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return False
            if state.status != JobStatus.QUEUED:
                return False
            if state.submitter_token != submitter_token:
                return False
            state.status = JobStatus.CANCELLED
            state.completed_at = _utc_iso()
            # Remove from pending
            self._pending = deque(
                (s, f, a, k)
                for s, f, a, k in self._pending
                if s.job_id != job_id
            )
            return True

    def queue_position(self, job_id: str) -> Optional[int]:
        """1-based position.  Running = 1.  None if not active."""
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            if state.status == JobStatus.RUNNING:
                return 1
            if state.status == JobStatus.QUEUED:
                for i, (s, _, _, _) in enumerate(self._pending):
                    if s.job_id == job_id:
                        return i + 2  # +1 for running, +1 for 1-based
                return None
            return None

    @property
    def stats(self) -> Dict[str, int]:
        running = 1 if self._running_job_id else 0
        queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
        return {"running": running, "queued": queued, "max_depth": MAX_QUEUE_DEPTH}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    # -- internal --

    def _evict_stale(self) -> None:
        """Remove terminal-state jobs older than 30 minutes.  Must hold ``self._lock``."""
        now = time.time()
        stale = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            and j.completed_at is not None
            and (now - (j._start_time or now)) > 1800
        ]
        for jid in stale:
            del self._jobs[jid]
            self._futures.pop(jid, None)

    def _start_job(
        self,
        state: JobState,
        func: Callable,
        args: tuple,
        kwargs: dict,
    ) -> None:
        state.status = JobStatus.RUNNING
        state.started_at = _utc_iso()
        state._start_time = time.time()
        state.phase = "starting"
        self._running_job_id = state.job_id

        def _run() -> None:
            try:
                result = func(*args, job_state=state, **kwargs)
                with self._lock:
                    with state._lock:
                        state.status = JobStatus.COMPLETED
                        state.completed_at = _utc_iso()
                        state.progress_pct = 100
                        state.result = result
                        state.seq += 1
            except Exception as exc:
                logger.exception("Job %s failed", state.job_id)
                with self._lock:
                    with state._lock:
                        state.status = JobStatus.FAILED
                        state.completed_at = _utc_iso()
                        state.error = {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "step": state.step,
                        }
                        state.seq += 1
            finally:
                self._on_job_done()

        future = self._executor.submit(_run)
        self._futures[state.job_id] = future

    def _on_job_done(self) -> None:
        with self._lock:
            self._running_job_id = None
            if self._pending:
                next_state, func, args, kwargs = self._pending.popleft()
                if next_state.status == JobStatus.QUEUED:
                    self._start_job(next_state, func, args, kwargs)
