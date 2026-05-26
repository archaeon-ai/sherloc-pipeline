"""Job queue logic tests."""

import time
import threading

import pytest

from sherloc_pipeline.web.jobs import JobQueue, JobState, JobStatus, MAX_QUEUE_DEPTH


class TestJobQueue:
    def test_submit_and_get(self):
        q = JobQueue()
        state = q.submit("scan_1", lambda job_state=None: {"ok": True})
        assert state.job_id.startswith("j_")
        assert state.submitter_token.startswith("t_")
        assert state.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        # Allow completion
        time.sleep(0.3)
        retrieved = q.get(state.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.COMPLETED
        q.shutdown()

    def test_get_nonexistent(self):
        q = JobQueue()
        assert q.get("nonexistent") is None
        q.shutdown()

    def test_queue_full(self):
        q = JobQueue()
        blocker = threading.Event()

        def slow_job(job_state=None):
            blocker.wait(timeout=5)
            return {}

        # Fill the queue
        for _ in range(MAX_QUEUE_DEPTH):
            q.submit("scan", slow_job)

        with pytest.raises(ValueError, match="queue_full"):
            q.submit("scan", slow_job)

        blocker.set()
        time.sleep(0.5)
        q.shutdown()

    def test_cancel_queued_job(self):
        q = JobQueue()
        blocker = threading.Event()

        def slow_job(job_state=None):
            blocker.wait(timeout=5)
            return {}

        # First job blocks the executor
        q.submit("scan_1", slow_job)
        time.sleep(0.1)

        # Second job stays queued
        state2 = q.submit("scan_2", slow_job)
        assert state2.status == JobStatus.QUEUED

        # Cancel with correct token
        assert q.cancel(state2.job_id, state2.submitter_token)
        assert q.get(state2.job_id).status == JobStatus.CANCELLED

        blocker.set()
        time.sleep(0.3)
        q.shutdown()

    def test_cancel_wrong_token_fails(self):
        q = JobQueue()
        blocker = threading.Event()

        def slow_job(job_state=None):
            blocker.wait(timeout=5)
            return {}

        q.submit("scan_1", slow_job)
        time.sleep(0.1)

        state2 = q.submit("scan_2", slow_job)
        assert not q.cancel(state2.job_id, "wrong_token")

        blocker.set()
        time.sleep(0.3)
        q.shutdown()

    def test_queue_position(self):
        q = JobQueue()
        blocker = threading.Event()

        def slow_job(job_state=None):
            blocker.wait(timeout=5)
            return {}

        s1 = q.submit("scan_1", slow_job)
        time.sleep(0.1)
        s2 = q.submit("scan_2", slow_job)

        # s1 is running (pos 1), s2 is queued (pos 2)
        assert q.queue_position(s1.job_id) == 1
        assert q.queue_position(s2.job_id) == 2

        blocker.set()
        time.sleep(0.5)
        q.shutdown()

    def test_stats(self):
        q = JobQueue()
        stats = q.stats
        assert stats["running"] == 0
        assert stats["queued"] == 0
        assert stats["max_depth"] == MAX_QUEUE_DEPTH
        q.shutdown()

    def test_failed_job(self):
        q = JobQueue()

        def failing_job(job_state=None):
            raise RuntimeError("boom")

        state = q.submit("scan_1", failing_job)
        time.sleep(0.5)
        retrieved = q.get(state.job_id)
        assert retrieved.status == JobStatus.FAILED
        assert retrieved.error["error_type"] == "RuntimeError"
        assert "boom" in retrieved.error["message"]
        q.shutdown()


# ---------------------------------------------------------------------------
# JobState unit tests
# ---------------------------------------------------------------------------


class TestJobState:
    def test_job_state_update_bumps_seq(self):
        state = JobState(job_id="j_test", scan_id="scan_1")
        assert state.seq == 0
        state.update(phase="downloading", progress_pct=50, message="test")
        assert state.phase == "downloading"
        assert state.progress_pct == 50
        assert state.message == "test"
        assert state.seq == 1

    def test_job_state_snapshot_consistency(self):
        state = JobState(job_id="j_snap", scan_id="scan_2")
        state.update(phase="ingesting", progress_pct=75, message="halfway")
        snap = state.snapshot()
        assert "job_id" in snap
        assert "status" in snap
        assert "phase" in snap
        assert "message" in snap
        assert "progress_pct" in snap
        assert "seq" in snap
        assert snap["job_id"] == "j_snap"
        assert snap["phase"] == "ingesting"
        assert snap["progress_pct"] == 75
        assert snap["message"] == "halfway"
        assert snap["seq"] == 1

    def test_job_state_thread_safe_update(self):
        state = JobState(job_id="j_threads", scan_id="scan_3")
        n_threads = 10
        updates_per_thread = 100

        def worker(i):
            for _ in range(updates_per_thread):
                state.update(progress_pct=i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state.seq == n_threads * updates_per_thread

    def test_find_active_by_scan_id(self):
        q = JobQueue()
        blocker = threading.Event()

        def slow_job(job_state=None):
            blocker.wait(timeout=5)
            return {}

        state = q.submit("test_scan", slow_job)
        time.sleep(0.1)

        found = q.find_active_by_scan_id("test_scan")
        assert found is not None
        assert found.job_id == state.job_id

        not_found = q.find_active_by_scan_id("other")
        assert not_found is None

        blocker.set()
        time.sleep(0.3)
        q.shutdown()

    def test_eviction_of_stale_jobs(self):
        q = JobQueue()

        # Submit and complete a job
        completed_state = q.submit("scan_stale", lambda job_state=None: {})
        time.sleep(0.3)
        assert q.get(completed_state.job_id).status == JobStatus.COMPLETED

        # Manually backdate _start_time to 2 hours ago so it looks stale
        completed_state._start_time = time.time() - 7200

        # Submit another job — triggers _evict_stale() internally
        new_state = q.submit("scan_new", lambda job_state=None: {})

        # The old completed job should have been evicted
        assert q.get(completed_state.job_id) is None

        time.sleep(0.3)
        q.shutdown()
