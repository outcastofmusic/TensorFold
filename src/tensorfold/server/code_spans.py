"""Markdown code spans and fences in a reply, so tool-call markup quoted inside them stays prose (upstream #3588).

A reply that explains a tool may write ``<tool_call>`` in backticks or show a whole call in a ```` ```xml ```` fence.
Without this, the parsers ran that call and the stream hid it. ``Quoted.check`` says whether an opener's place is
prose, code, or (while the reply streams) not known yet: an inline span that may still close, or a backtick run at the
end that may still grow. The parsers ``jump`` past each call they take, so Markdown inside its arguments is not read."""

from __future__ import annotations

import bisect
import re

PENDING = -1                    # ``check``: streaming, and the place may still turn out to be code
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_TICKS = re.compile(r"`+")
_BLANK = re.compile(r"[ \t\r]*")


def scan(text: str, start: int = 0, finished: bool = True) -> tuple[list[tuple[int, int]], int]:
    """The code in ``text[start:]``: [(start, end)] of fenced blocks and inline spans in order, and where undecided
    text begins (``len(text)`` when none, always when ``finished``). CommonMark's rules: a fence is 3+ backticks or
    tildes at a line start (3 spaces at most before it, no backtick after a backtick fence's run on its line) and runs
    to a line of at least as many of its character, or to the end. An inline span runs from a backtick run to the next
    run of the same length in its paragraph; a run with none is a literal backtick."""

    spans: list[tuple[int, int]] = []
    n, hold = len(text), len(text)

    def inline(first: int, last: int, growing: bool) -> None:
        """The spans of the paragraph ``text[first:last]``; ``growing``: it is the reply's last and may go on."""

        nonlocal hold
        runs = [m.span() for m in _TICKS.finditer(text, first, last)]
        after: list[int | None] = [None] * len(runs)       # the next run of the same length (linear, not a search)
        seen: dict[int, int] = {}
        for k in range(len(runs) - 1, -1, -1):
            width = runs[k][1] - runs[k][0]
            after[k], seen[width] = seen.get(width), k
        k = 0
        while k < len(runs):
            begin, end = runs[k]
            close = after[k]
            if growing and (end == n or close is None or runs[close][1] == n):
                hold = min(hold, begin)                     # a run that may grow, or a span that may still close
                return
            if close is not None:
                spans.append((begin, runs[close][1]))
                k = close + 1
            else:
                k += 1

    paragraph = None
    i = start
    while i < n:
        newline = text.find("\n", i)
        line_end = n if newline < 0 else newline
        fence = _FENCE.match(text, i, line_end) if i == 0 or text[i - 1] == "\n" else None
        if fence and fence.group(1)[0] == "`" and "`" in text[fence.end():line_end]:
            fence = None                                    # ```a`b is inline code, not a fence
        if fence and newline < 0 and not finished:          # an opening line still being written
            if paragraph is not None:
                inline(paragraph, i, True)
            return spans, min(hold, fence.start(1))
        if fence and newline >= 0:
            if paragraph is not None:
                inline(paragraph, i, False)
                paragraph = None
            mark = fence.group(1)
            closer = re.compile(rf"^ {{0,3}}{re.escape(mark[0])}{{{len(mark)},}}[ \t\r]*$", re.MULTILINE)
            found = closer.search(text, newline + 1)
            if found is None or (found.end() == n and not finished):
                spans.append((fence.start(1), n))           # open to the end (a last line may still grow)
                return spans, hold
            i = min(n, found.end() + 1)
            spans.append((fence.start(1), i))
            continue
        if (newline >= 0 or finished) and _BLANK.fullmatch(text, i, line_end):
            if paragraph is not None:
                inline(paragraph, i, False)
                paragraph = None
        elif paragraph is None:
            paragraph = i
        i = line_end + 1
    if paragraph is not None:
        inline(paragraph, n, not finished)
    return spans, hold


class Quoted:
    """Whether places in one reply text are quoted as code, scanned once from the last ``jump``."""

    def __init__(self, text: str, finished: bool = True, start: int = 0) -> None:
        self.text, self.finished = text, finished
        self.jump(start)

    def jump(self, start: int) -> None:
        """Read Markdown again only from ``start``: what came before was a call's envelope, consumed whole."""

        self.start, self._scanned = start, None

    def check(self, at: int) -> int | None:
        """None when ``at`` is prose, PENDING while it may still become code, else where the code holding it ends."""

        if at < self.start:
            return None
        if self._scanned is None:
            spans, hold = scan(self.text, self.start, self.finished)
            self._scanned = spans, [s for s, _ in spans], hold
        spans, starts, hold = self._scanned
        if at >= hold:
            return PENDING
        k = bisect.bisect_right(starts, at) - 1
        return spans[k][1] if k >= 0 and at < spans[k][1] else None


def live_find(text: str, needle: str, quoted: Quoted, at: int = 0) -> int:
    """The first place of ``needle`` from ``at`` that is not quoted as code (one PENDING counts), or -1."""

    found = text.find(needle, at)
    while found >= 0 and (end := quoted.check(found)) not in (None, PENDING):
        found = text.find(needle, end)
    return found
