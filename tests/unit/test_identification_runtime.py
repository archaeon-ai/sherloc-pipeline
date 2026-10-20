"""Synthetic disposable-worker receipt and cleanup regressions."""

import multiprocessing
import os
from pathlib import Path
import signal
import struct
import time

import pytest

from sherloc_pipeline.services import identification_runtime as runtime


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
