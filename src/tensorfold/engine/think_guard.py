"""A thinking guard: a decision model reads each finished paragraph of thinking, and its yes closes the block."""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# the decision model rereads its whole state on every check, so the state keeps only the newest reasoning
MAX_REASONING_BYTES = 24000
TIMEOUT = 30.0
QUESTION = {"enough": {
    "type": "noul",
    "instructions": "Is the reasoning so far enough to write a correct and complete final answer to the user's "
                    "request, so that more thinking would only repeat or re-check it?",
    "criteria": {"true": "The reasoning already contains everything the final answer needs.",
                 "false": "The reasoning still lacks a step, a calculation, or a decision the final answer needs."},
}}

# checks wait on HTTP, never on the GPU: a few threads serve every stream's guard
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="think-guard")


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """``--thinking-guard``: the decision server's ``/v1`` base, its model, and the default threshold."""

    url: str
    model: str
    threshold: float = 0.8


def guard_state(request: str, reasoning: str) -> str:
    """What the decision model reads: the user's request, then the newest reasoning."""

    data = reasoning.encode()
    if len(data) > MAX_REASONING_BYTES:
        reasoning = "[earlier reasoning left out]\n" + data[-MAX_REASONING_BYTES:].decode(errors="ignore")
    return f"User's request:\n{request}\n\nReasoning so far:\n{reasoning}"


def decide(config: GuardConfig, request: str, reasoning: str) -> float:
    """The decision model's probability that ``reasoning`` is enough to answer ``request`` (SystemOne ``noul``)."""

    body = json.dumps({"model": config.model, "state": guard_state(request, reasoning), "questions": QUESTION})
    call = urllib.request.Request(config.url.rstrip("/") + "/systemone", body.encode(),
                                  {"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=TIMEOUT) as reply:      # an HTTP error raises, and ends the checks
        answer = json.load(reply)["answers"]["enough"]["noul"]
    return float(answer)


def _ends(tail: str, piece: str) -> tuple[list[int], str]:
    """(where in ``piece`` each paragraph that ``tail + piece`` finishes ends, just past its blank line; the new tail).

    The tail is the text since the last paragraph end. A blank paragraph (blank lines in a row) is no paragraph."""

    text, start, ends = tail + piece, 0, []
    while (at := text.find("\n\n", start)) >= 0:
        if text[start:at].strip():
            ends.append(at + 2 - len(tail))
        start = at + 2
    return ends, text[start:]


class ThinkGuard:
    """After each paragraph of thinking, ``ask(reasoning)`` runs on a background thread; once one returns a
    probability at ``threshold`` or above, the next paragraph end closes the think block. Decoding never waits for a
    check: one runs at a time, and a paragraph that ends while one runs replaces any paragraph still waiting."""

    def __init__(self, ask: Callable[[str], float], threshold: float, *, think_end: int,
                 decode: Callable[[list[int]], str], encode: Callable[[str], list[int]], model: str = "",
                 submit: Callable[..., Future] | None = None) -> None:
        self.ask, self.threshold, self.think_end = ask, float(threshold), int(think_end)
        self.decode, self.encode, self.submit = decode, encode, submit or _POOL.submit
        self.tokens: list[int] = []         # the reasoning so far
        self.tail = ""                      # its text since the last paragraph end
        self.paragraphs = 0
        self.open = True                    # until the think block closes
        self.running: tuple[int, int, float, Future] | None = None   # (paragraph, characters, start, check)
        self.waiting: tuple[int, int] | None = None                  # (paragraph, tokens): the newest unchecked
        self.yes = 0                        # the paragraph a check said yes to
        self.failed = False
        self.record: dict[str, Any] = {"model": model, "threshold": self.threshold, "checks": []}

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        """(index in the next committed ``tokens`` the close replaces, the close), as ``CallGate.cut``: after a yes,
        the token that ends a paragraph becomes its text up to the blank line, the think end and a blank line."""

        if not self.open:
            return None
        self._collect()
        if not self.yes:
            return None
        tail = self.tail
        if not tail.strip():                # the yes came back at a paragraph end: close it there
            return 0, [self.think_end, *self.encode("\n\n")]
        for i, token in enumerate(int(t) for t in tokens):
            if token == self.think_end:
                return None
            piece = self.decode([token])
            ends, tail = _ends(tail, piece)
            if ends:
                return i, [*self.encode(piece[:ends[0]]), self.think_end, *self.encode("\n\n")]
        return None

    def observe(self, token: int) -> None:
        """Follow a committed token: count paragraph ends, and start the check of the newest one."""

        if not self.open:
            return
        token = int(token)
        if token == self.think_end:
            self.open = False
            self.record["closed_at_paragraph"] = self.paragraphs
            return
        self.tokens.append(token)
        ends, self.tail = _ends(self.tail, self.decode([token]))
        if ends:
            self.paragraphs += len(ends)
            if not self.yes and not self.failed:
                self.waiting = (self.paragraphs, len(self.tokens))
        self._collect()

    def _collect(self) -> None:
        """Take a finished check's answer, then start the waiting paragraph's check."""

        if self.running is not None and self.running[3].done():
            paragraph, chars, start, check = self.running
            self.running = None
            try:
                p = float(check.result())
            # refused or unreachable (URLError), slow (TimeoutError), or an answer of another shape: no more checks
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.failed = True
                self.record["error"] = f"{type(exc).__name__}: {exc}"
                return
            self.record["checks"].append({"paragraph": paragraph, "chars": chars, "p": round(p, 4),
                                          "ms": round((time.perf_counter() - start) * 1000)})
            if p >= self.threshold and not self.yes:
                self.yes = paragraph
                self.record["yes_paragraph"] = paragraph
        if self.running is None and self.waiting is not None and not self.yes and not self.failed:
            paragraph, count = self.waiting
            self.waiting = None
            reasoning = self.decode(self.tokens[:count])
            self.running = (paragraph, len(reasoning), time.perf_counter(), self.submit(self.ask, reasoning))

    def finish(self) -> dict[str, Any]:
        """The record of every check that answered while the reply ran. A check still running is dropped."""

        if self.open and self.running is not None:
            self._collect()
        return self.record


__all__ = ["GuardConfig", "ThinkGuard", "decide", "guard_state"]
