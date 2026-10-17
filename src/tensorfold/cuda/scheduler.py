"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
from collections import deque
from typing import Any, Callable

from .memory_gate import NoRoom
from .streams import Stream

IDLE_DELAY = 0.15                # seconds with no stream live before the decoder's idle work (warm.DELAY)
WAKE = object()                  # a ``Scheduler.call`` waits for the worker


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def put_many(self, items) -> None:
        """Publish a group before waking the worker, preserving arrival and background priority."""

        with self.not_empty:
            for item in items:
                self._put((1 if item[0].background else 0, next(self._order), item))
                self.unfinished_tasks += 1
            self.not_empty.notify()

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        super().put((2, next(self._order), None))

    def wake(self) -> None:
        """Wake an idle worker for a ``call``: after every waiting request."""

        super().put((3, next(self._order), WAKE))

    def foreground(self) -> bool:
        """Whether a foreground request waits."""

        with self.mutex:
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    def __init__(self, decoder: Any, *, max_streams: int = 4, idle: Callable[[Callable[[], bool]], Any] | None = None
                 ) -> None:
        self.decoder = decoder
        self.max_streams = max_streams
        # ``idle(stop)``: the decoder's work once no stream is live (its kept states' disk write), IDLE_DELAY after a
        # request ends; ``stop()`` turns true when a request arrives, and the work then stops between its steps
        self.idle = idle
        self.owed = False                            # a request ended since ``idle`` last ran
        self.jobs: deque = deque()                   # ``call``s waiting for the worker: (fn, box)
        self.waiting = Waiting()
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self.boxes: dict[int, queue.Queue] = {}
        self.yields = 0                              # background streams that gave up their lane
        if hasattr(decoder, "arrived"):              # a decoder filling prompts lets a new request in between passes
            decoder.arrived = self.waiting.foreground
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn()`` on the worker once no stream is live and return what it returns, or raise what it raises."""

        box: queue.Queue = queue.Queue()
        self.jobs.append((fn, box))
        self.waiting.wake()
        kind, value = box.get()
        if kind == "error":
            raise value
        return value

    def close(self) -> None:
        """Stop the worker once idle and let go of the decoder (its thread held it, and its weights, until now)."""

        self.waiting.stop()
        self.thread.join()
        self.decoder = None

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, probabilities: Any = None,
               splice: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats.

        ``splice`` (a decoder with ``splices``) runs on the worker with each round's new tokens before they commit."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities, splice=splice)
        cancel = [False]
        stream.emit = lambda new: (box.put(("tokens", new)), cancel[0])[1]
        self.waiting.put((stream, box))
        while True:
            kind, value = box.get()
            if kind == "tokens":
                if not cancel[0] and emit(value):
                    cancel[0] = True                 # the client left: the stream ends after its next round
            elif kind == "error":
                raise value
            else:
                return value

    def submit_many(self, requests: list[dict]) -> list[dict]:
        """Queue isolated one-shot requests atomically and return ordered stats, draining errors too."""

        boxes = []
        for r in requests:
            box: queue.Queue = queue.Queue()
            stream = Stream(list(r["prompt"]), max(1, r["count"]), r["sampling"], draft=r.get("draft", True),
                            stop_eos=r.get("stop_eos", True), probabilities=r.get("probabilities"))
            stream.emit = lambda new: False
            boxes.append((stream, box))
        self.waiting.put_many(boxes)
        results, error = [], None
        for _, box in boxes:
            while True:
                kind, value = box.get()
                if kind == "tokens":
                    continue
                if kind == "error":
                    error = error or value
                    results.append(None)
                else:
                    results.append(value)
                break
        if error is not None:
            raise error
        return results

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                (stream, box), first = first, None
            elif self.held is not None:
                (stream, box), self.held = self.held, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
                if item is WAKE:                     # a call waits for no stream live: ``_next`` runs it
                    continue
                stream, box = item
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom as exc:
                self.boxes.pop(id(stream))
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    self.held = (stream, box)
                    break
                box.put(("error", exc))
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
                continue
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes.pop(id(stream))
        self.decoder.finish([stream])
        self.yields += 1
        self.waiting.put((stream.continued(), box))

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            box.put((kind, value))

    def _next(self):
        """With no stream live: the next request, or None to stop. ``call`` jobs run first, and ``idle`` runs once
        IDLE_DELAY passes with no request after one ended."""

        while True:
            while self.jobs:
                fn, box = self.jobs.popleft()
                try:
                    box.put(("done", fn()))
                except Exception as exc:             # noqa: BLE001  (raised again on the calling thread)
                    box.put(("error", exc))
            try:
                item = self.waiting.get(timeout=IDLE_DELAY if self.owed and self.idle is not None else None)
            except queue.Empty:
                self.owed = False
                try:
                    self.idle(lambda: not self.waiting.empty())
                except RuntimeError as exc:          # a CUDA failure: the work is optional, so it stops for good
                    self.idle = None
                    print(f"[tensorfold] idle work off after: {type(exc).__name__}: {exc}", flush=True)
                continue
            if item is None:
                while self.jobs:
                    self.jobs.popleft()[1].put(("error", RuntimeError("the scheduler closed")))
                return None
            if item is not WAKE:
                return item

    def _loop(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and self.held is None
            first = self._next() if idle else None                                    # idle: wait for a request
            if idle and first is None:
                return                                                                # close()
            done = self._admit(first)
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            self.owed = self.owed or bool(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))
