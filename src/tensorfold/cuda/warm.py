"""After a chat reply, the next turn's shared prompt prefilled in the background, so that turn resumes past the reply.

The next turn starts with the conversation and the reply as the client sends it back, then a user or a tool message.
Rendered with each of those after the reply, the prompts agree up to that message's start: that common prefix is
prefilled once the engine has had no request for ``DELAY`` seconds, as a background request a foreground one stops
between chunks. ``TENSORFOLD_WARM_REPLIES=0`` turns it off. Upstream TensorFold 1.0.9 and 1.0.12 do the same (warm.zig).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from tensorfold.env_flags import flag

DELAY = 0.15                    # seconds without a request before a pass starts (upstream's warm_quiet)


def enabled() -> bool:
    return flag("TENSORFOLD_WARM_REPLIES", True)


def reply_message(result: dict[str, Any]) -> dict[str, Any]:
    """The reply as the chat route sends it (``http``), which is how a client sends it back."""

    message: dict[str, Any] = {"role": "assistant", "content": result.get("content") or None}
    if result.get("reasoning"):
        message["reasoning_content"] = result["reasoning"]
    if result.get("calls"):
        message["tool_calls"] = result["calls"]
    return message


def next_turn_prefix(render: Callable[[list[dict[str, Any]]], Sequence[int]], messages: list[dict[str, Any]],
                     reply: dict[str, Any]) -> list[int]:
    """The tokens every next turn shares: the conversation and ``reply`` rendered with a user message "A", one "B",
    and after a call a tool result, cut where the renderings part."""

    after = [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    calls = reply.get("tool_calls") or []
    if calls:
        after.append({"role": "tool", "tool_call_id": calls[0].get("id", ""), "content": "A"})
    ids = [list(render([*messages, reply, turn])) for turn in after]
    n = min(len(x) for x in ids)
    cut = next((i for i in range(n) if any(x[i] != ids[0][i] for x in ids[1:])), n)
    return ids[0][:cut]


class Warmer:
    """One pending pass at a time: a newer reply's replaces it, and a request that ran since it was queued makes it
    stale. A pass is queued as ``make()``, which renders (prompt, born) on this thread, so the reply's last chunk does
    not wait on three renderings of the conversation; None skips it. ``take``/``give``/``wanted`` are the engine's
    turns; ``run(prompt, born, stop)`` does the prefill."""

    def __init__(self, run: Callable[[list[int], int, Callable[[], bool]], Any], take: Callable[[], None],
                 give: Callable[[], None], wanted: Callable[[], bool], delay: float = DELAY) -> None:
        self.run, self.take, self.give, self.wanted, self.delay = run, take, give, wanted, float(delay)
        self.cv = threading.Condition()
        self.served = 0                       # requests finished: a pass queued before the latest is stale
        self.pending: tuple[Callable[[], tuple[list[int], int] | None], int] | None = None
        self.passes: list[int] = []           # tokens each finished pass prefilled (0: kept already or stopped)
        self.thread = threading.Thread(target=self._loop, name="warm-replies", daemon=True)
        self.thread.start()

    def finished(self) -> None:
        """A request ended: a pass queued before it describes a turn the conversation has moved past."""

        with self.cv:
            self.served += 1
            self.pending = None

    def queue(self, make: Callable[[], tuple[list[int], int] | None]) -> None:
        with self.cv:
            self.pending = (make, self.served)
            self.cv.notify_all()

    def _stale(self, served: int) -> bool:
        with self.cv:
            return self.served != served

    def _loop(self) -> None:
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                make, served = self.pending
                self.pending = None
            made = make()
            time.sleep(self.delay)
            if made is None or self._stale(served) or self.wanted():
                continue
            prompt, born = made
            self.take()
            try:
                if self._stale(served):
                    continue
                done = self.run(prompt, born, self.wanted)
            finally:
                self.give()
            with self.cv:
                self.passes.append(int(done or 0))
