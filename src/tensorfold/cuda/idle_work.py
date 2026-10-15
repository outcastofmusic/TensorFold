"""GPU work done between requests, a step at a time, that a request never waits on for longer than one step.

``busy()`` wraps every request. ``delay`` seconds after the last one ends, a worker thread asks ``plan()`` for the
steps (callables, or None), and runs them one at a time, each holding the same lock a request holds. It stops before
a step when a request is waiting or one ran since the plan, and plans afresh after the next request.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager


class IdleWork:
    def __init__(self, plan: Callable[[], Iterable[Callable[[], None]] | None], delay: float = 1.0,
                 setup: Callable[[], None] | None = None) -> None:
        self.plan, self.delay, self.setup = plan, float(delay), setup
        self.lock = threading.Lock()              # the GPU: a request or one step at a time
        self.count = threading.Lock()
        self.waiting = 0
        self.served = 0                           # requests finished: a step planned before the last one is stale
        self.last = time.monotonic()
        self.steps = 0
        self.errors: list[BaseException] = []
        self.wake = threading.Event()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="idle-work", daemon=True)
        self.thread.start()

    @contextmanager
    def busy(self):
        with self.count:
            self.waiting += 1
        self.lock.acquire()
        with self.count:
            self.waiting -= 1
        try:
            yield
        finally:
            self.served += 1
            self.last = time.monotonic()
            self.lock.release()
            self.wake.set()

    def close(self) -> None:
        self.closed = True
        self.wake.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        if self.setup is not None:
            self.setup()
        while not self.closed:
            self.wake.wait()
            self.wake.clear()
            while not self.closed:                 # a request during the delay restarts it
                left = self.delay - (time.monotonic() - self.last)
                if left <= 0:
                    break
                time.sleep(left)
            if self.closed or self.waiting:
                continue
            served = self.served
            with self.lock:
                steps = list(self.plan() or ())
            for step in steps:
                if self.closed or self.waiting or self.served != served:
                    break
                if not self.lock.acquire(blocking=False):
                    break
                try:
                    if self.waiting or self.served != served:
                        break
                    step()
                    self.steps += 1
                except RuntimeError as exc:         # a CUDA failure (out of memory, a refused capture): the work is
                    self.errors.append(exc)          # optional, so it stops for good and requests do it lazily
                    self.closed = True
                    print(f"[tensorfold] idle work off after: {type(exc).__name__}: {exc}", flush=True)
                    break
                finally:
                    self.lock.release()


__all__ = ["IdleWork"]
