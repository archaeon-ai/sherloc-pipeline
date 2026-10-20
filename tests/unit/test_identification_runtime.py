"""Synthetic disposable-worker receipt and cleanup regressions."""

import multiprocessing
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import pytest

from sherloc_pipeline.runtime import worker as runtime


# Keep spawned deadline helpers independent of API and scientific imports.
def _hang(path):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(path, 'w') as stream:
        stream.write(str(os.getpid()))
    while True:
        time.sleep(.01)


class HangingStore:
    def __init__(self, path):
        self.path = path

    def read_verified(self, locator, sha256):
        _hang(self.path)


def _crash():
    os._exit(7)


def _large_payload(length):
    return b"synthetic" * length


def _stalled_sender(connection, function, args):
    pid_path, partial = args
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(connection.fileno(), partial)
    Path(pid_path).write_text(str(os.getpid()))
    # Finite even under the broken implementation, so a regression fails
    # instead of hanging the suite indefinitely.
    time.sleep(8)
    connection.close()


def test_receives_a_payload_larger_than_the_pipe_buffer():
    expected = b"synthetic" * 262144
    assert runtime.run_process(_large_payload, (262144,), 5) == expected


@pytest.mark.parametrize("partial", [
    b"\x00",
    struct.pack("!i", 1048576) + b"synthetic-partial-body",
    struct.pack("!i", -1) + b"\x00\x00",
    struct.pack("!i", -1) + struct.pack("!Q", 1048576) + b"synthetic-partial-body",
], ids=["partial-header", "partial-body", "partial-extended-header", "partial-extended-body"])
def test_stalled_receipt_times_out_and_kills_and_reaps_the_worker(monkeypatch, tmp_path, partial):
    monkeypatch.setattr(runtime, "_worker", _stalled_sender)
    pid_path = tmp_path / "synthetic-worker.pid"
    started = time.monotonic()
    with pytest.raises(runtime.ExecutionFailed, match="time limit"):
        runtime.run_process(_large_payload, (str(pid_path), partial), 3)
    assert time.monotonic() - started < 5
    # This proves startup and the partial write happened before the timeout.
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    # Reaping frees the caller to start another worker successfully.
    monkeypatch.undo()
    assert runtime.run_process(_large_payload, (1,), 5) == b"synthetic"


def test_services_reexport_the_worker_objects():
    # Imported here, not at module level: spawned targets in this module are pickled by
    # reference, so its top-level imports load in every child.
    from sherloc_pipeline.services import identification_runtime

    for name in ("ExecutionFailed", "WorkRejected", "_receive", "_worker", "run_process"):
        assert getattr(identification_runtime, name) is getattr(runtime, name)


def test_worker_module_imports_neither_services_nor_scientific_stacks():
    # A spawned child imports this module before its target; keep that cost small.
    probe = ("import sys, sherloc_pipeline.runtime.worker; "
             "print(sorted(m for m in sys.modules if m == 'sherloc_pipeline.services' or "
             "m.split('.')[0] in {'matplotlib', 'sklearn', 'scipy'}))")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", probe], cwd=root / "src", capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
