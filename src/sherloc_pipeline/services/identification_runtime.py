"""Process-local admission and bounded caches; disposable workers live in runtime.worker."""

from collections import OrderedDict, deque
import copy
import threading
import time

# Re-exported so existing imports keep working; the implementation lives outside services,
# so monkeypatch runtime.worker (not these aliases) to affect run_process.
from sherloc_pipeline.runtime.worker import ExecutionFailed, WorkRejected, _receive, _worker, run_process

__all__ = ["ExecutionFailed", "IdentificationRuntime", "ResponseCache", "WorkRejected", "_receive", "_worker",
           "run_process"]


class ResponseCache:
    def __init__(self, capacity=16, ttl=120):
        self.capacity, self.ttl = capacity, ttl
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            expires, payload = value
            if time.monotonic() >= expires:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(payload)

    def put(self, key, payload):
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl, copy.deepcopy(payload))
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)


class IdentificationRuntime:
    def __init__(self, max_concurrency=1):
        self.slots = threading.BoundedSemaphore(max_concurrency)
        self.cache = ResponseCache()
        self._requests = OrderedDict()
        self._lock = threading.Lock()

    def admit_rate(self, identity):
        now = time.monotonic()
        with self._lock:
            while self._requests and now - next(iter(self._requests.values()))[-1] >= 60:
                self._requests.popitem(last=False)
            if identity not in self._requests and len(self._requests) >= 1024:
                raise WorkRejected("identification rate limit reached")
            times = self._requests.setdefault(identity, deque())
            while times and now - times[0] >= 60:
                times.popleft()
            if len(times) >= 10:
                raise WorkRejected("identification rate limit reached")
            times.append(now)
            self._requests.move_to_end(identity)
