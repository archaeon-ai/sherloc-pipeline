"""Process-local admission, bounded caches and disposable compute workers."""

from collections import OrderedDict, deque
import copy
import multiprocessing
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
        if not reader.poll(max(0., deadline - time.monotonic())):
            raise ExecutionFailed("identification time limit reached")
        try:
            success, result = reader.recv()
        except EOFError as exc:
            raise ExecutionFailed("identification worker failed") from exc
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
