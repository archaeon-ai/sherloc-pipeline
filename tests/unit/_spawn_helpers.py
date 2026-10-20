"""Synthetic spawned-worker targets; standard library only so a spawned child starts quickly."""

import os
import signal
import time


def _write_atomic(path, text):
    partial = f"{path}.partial"
    with open(partial, 'w') as stream:
        stream.write(text)
    os.replace(partial, path)


def ready_path(path):
    return f"{path}.ready"


def _hang(path):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # time.monotonic() reads CLOCK_MONOTONIC, which is system-wide on Linux, so the
    # parent can compare this readiness time with its own clock (not guaranteed on macOS).
    _write_atomic(ready_path(path), repr(time.monotonic()))
    _write_atomic(path, str(os.getpid()))
    while True:
        time.sleep(.01)


class HangingStore:
    def __init__(self, path):
        self.path = path

    def read_verified(self, locator, sha256):
        _hang(self.path)


def _crash():
    os._exit(7)
