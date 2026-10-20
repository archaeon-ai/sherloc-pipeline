"""Process-local admission, bounded caches and disposable compute workers."""

from collections import OrderedDict, deque
import copy
import multiprocessing
from multiprocessing.reduction import ForkingPickler
import os
import select
import struct
import threading
import time


class WorkRejected(ValueError):
    pass


class ExecutionFailed(ValueError):
    pass


def _worker(connection, function, args):
    try:
        connection.send((True, function(*args)))
    except WorkRejected as exc:
        connection.send((False, str(exc)))
    except Exception:
        connection.send((False, None))
    finally:
        connection.close()


def _receive(connection, deadline):
    """Read the complete Connection frame without blocking beyond its deadline."""
    descriptor = connection.fileno()
    os.set_blocking(descriptor, False)

    def read_exact(length):
        payload = bytearray()
        while len(payload) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExecutionFailed("identification time limit reached")
            if not select.select([descriptor], [], [], remaining)[0]:
                raise ExecutionFailed("identification time limit reached")
            try:
                chunk = os.read(descriptor, min(length - len(payload), 65536))
            except BlockingIOError:
                continue
            if not chunk:
                raise ExecutionFailed("identification worker failed")
            payload.extend(chunk)
        return payload

    # multiprocessing.Connection uses a signed four-byte size, or -1 followed
    # by an unsigned eight-byte size for messages larger than its short frame.
    length = struct.unpack("!i", read_exact(4))[0]
    if length == -1:
        length = struct.unpack("!Q", read_exact(8))[0]
    elif length < 0:
        raise ExecutionFailed("identification worker failed")
    payload = read_exact(length)
    if time.monotonic() >= deadline:
        raise ExecutionFailed("identification time limit reached")
    result = ForkingPickler.loads(payload)
    if time.monotonic() >= deadline:
        raise ExecutionFailed("identification time limit reached")
    return result


def run_process(function, args, timeout):
    """Spawn with detached arguments; always reap, including crash and timeout."""
    if timeout <= 0:
        raise ExecutionFailed("identification time limit reached")
    deadline = time.monotonic() + timeout
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, function, args), daemon=True)
    try:
        process.start()
        writer.close()
        success, result = _receive(reader, deadline)
        if not success:
            if result is not None:
                raise WorkRejected(result)
            raise ExecutionFailed("identification worker failed")
        return result
    finally:
        reader.close()
        writer.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(0.2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


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
