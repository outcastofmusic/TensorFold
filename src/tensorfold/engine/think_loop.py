"""A reasoning block that repeats itself is closed: the reply's next token becomes a hidden phrase and ``</think>``.

On by default for chat replies that think; TENSORFOLD_THINK_LOOP=0 turns it off. Three arms watch the block's
tokens:

- a pattern: the last ``max(4, ceil(128 / s))`` blocks of ``s`` tokens are equal, for some ``s`` from 1 to 16 (each
  token then equals the one ``s`` back for ``(count - 1) * s`` tokens in a row, so a pattern must span 128 tokens);
- history: past 2,048 tokens, every 64th token, the share of the last 256 tokens' 16-grams that already occurred before
  them; two checks in a row at 0.7 or more fire;
- novelty: past 320 tokens, every 64th token, the share of the last 128 tokens' 8-grams that the 128 before them do not
  hold; under 0.10 fires.

The close is Qwen's own budget-forcing phrase, then ``</think>`` and a blank line. The model reads the phrase, and the
client's reasoning text leaves it out (``hide``). One close a reply; once the block ends the gate watches nothing.

The arms and their thresholds come from TandemLLM's reasoning-loop detector (AGPL-3.0, ``engine/spec.py`` and
``ops/supervisor.yaml``: 3 of 3 failed turns caught, 4 of 80 healthy long turns closed); this is an independent
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tensorfold.env_flags import flag

ENV = "TENSORFOLD_THINK_LOOP"
CLOSE_TEXT = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
              "directly now.\n")

SIZES = range(1, 17)
SPAN, MIN_COUNT = 128, 4
GRAM, WINDOW, EVERY, AFTER, ACT, IN_A_ROW = 16, 256, 64, 2048, 0.7, 2
NOVEL_GRAM, NOVEL_WINDOW, NOVEL_AFTER, NOVEL_BELOW = 8, 128, 320, 0.10
# a 16-gram is history once it started this many times before the window. With one, code the model wrote out again
# as its final check reached a 0.84 to 0.91 share and fired: 3 of 7 fires on 183 saved blocks (2026-10-15), all
# false. With two those copies peak at 0.64 and the 4 real loops still reach 1.0, one repeat later.
SEEN = 2


def enabled() -> bool:
    return flag(ENV, default=True)


def _need(size: int) -> int:
    """Tokens in a row that equal the one ``size`` back, for a pattern of ``size`` to fire."""

    return (max(MIN_COUNT, -(-SPAN // size)) - 1) * size


class LoopDetector:
    """The three arms over one reasoning block, fed a token at a time. ``observe`` commits a token and returns the arm
    that fired on it, or None; ``ahead`` answers the same for tokens not committed yet, changing nothing."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.runs = [0] * (max(SIZES) + 1)     # size -> tokens in a row equal to the one size back
        self.grams: list[int] = []             # hash of the 16-gram starting at each index
        self.starts: dict[int, list[int]] = {}  # 16-gram hash -> the first SEEN indexes it starts at
        self.streak = 0                        # history checks in a row at ACT or more
        self.share: float | None = None        # the last history check's share

    def observe(self, token: int) -> str | None:
        token = int(token)
        why, self.runs, self.streak, share = self._step(token, self.runs, self.streak, [])
        self.tokens.append(token)
        start = len(self.tokens) - GRAM
        if start >= 0:
            h = hash(tuple(self.tokens[start:]))
            self.grams.append(h)
            seen = self.starts.setdefault(h, [])
            if len(seen) < SEEN:
                seen.append(start)
        if share is not None:
            self.share = share
        return why

    def ahead(self, tokens: Sequence[int]) -> int | None:
        """The index in ``tokens`` (following the committed ones) of the first token an arm would fire on, or None."""

        runs, streak, pending = list(self.runs), self.streak, []
        for i, token in enumerate(int(t) for t in tokens):
            why, runs, streak, _ = self._step(token, runs, streak, pending)
            if why is not None:
                return i
            pending.append(token)
        return None

    def why(self, token: int) -> str | None:
        """The arm that would fire on ``token`` next, changing nothing."""

        return self._step(int(token), self.runs, self.streak, [])[0]

    def _at(self, i: int, pending: list[int]) -> int:
        n = len(self.tokens)
        return self.tokens[i] if i < n else pending[i - n]

    def _gram(self, start: int, pending: list[int]) -> int:
        if start < len(self.grams):
            return self.grams[start]
        return hash(tuple(self._at(j, pending) for j in range(start, start + GRAM)))

    def _step(self, token: int, runs: list[int], streak: int, pending: list[int]):
        """Arms on ``token`` after the committed tokens and ``pending``: (why, runs, streak, history share)."""

        n = len(self.tokens) + len(pending)          # the token's index
        why = None
        runs = list(runs)
        for size in SIZES:
            if size <= n and self._at(n - size, pending) == token:
                runs[size] += 1
                if why is None and runs[size] >= _need(size):
                    why = f"loop(size={size} count={_need(size) // size + 1})"
            else:
                runs[size] = 0
        count, share = n + 1, None
        if count >= AFTER and count % EVERY == 0:
            share = self._history(count, token, pending)
            streak = streak + 1 if share >= ACT else 0
            if why is None and streak >= IN_A_ROW:
                why = f"repeat{GRAM}(share={share:.2f} at={count})"
        if why is None and count >= NOVEL_AFTER and count % EVERY == 0 and count >= 2 * NOVEL_WINDOW:
            novel = self._novelty(count, token, pending)
            if novel < NOVEL_BELOW:
                why = f"novelty({novel:.2f})"
        return why, runs, streak, share

    def _history(self, count: int, token: int, pending: list[int]) -> float:
        """Share of the 16-grams starting in the last WINDOW tokens that already started SEEN times at or before
        ``count - WINDOW - GRAM`` (grams that ended before the window). Grams not committed yet start inside the
        window, so only committed starts can count."""

        seq = pending + [token]
        last = count - GRAM                        # the newest gram's start
        before = count - WINDOW - GRAM
        hits = total = 0
        for start in range(count - WINDOW, last + 1):
            seen = self.starts.get(self._gram(start, seq), ())
            hits += len(seen) >= SEEN and seen[SEEN - 1] <= before
            total += 1
        return hits / total

    def _novelty(self, count: int, token: int, pending: list[int]) -> float:
        seq = pending + [token]
        get = lambda i: self._at(i, seq)          # noqa: E731
        w, g = NOVEL_WINDOW, NOVEL_GRAM
        new = {tuple(get(i) for i in range(s, s + g)) for s in range(count - w, count - g + 1)}
        old = {tuple(get(i) for i in range(s, s + g)) for s in range(count - 2 * w, count - w - g + 1)}
        return len(new - old) / len(new)


class LoopGate:
    """A ``generate_gated`` gate: the token an arm fires on becomes ``close`` (the phrase, ``</think>``, a blank
    line), once a reply; ``record`` says which arm fired and at which reasoning token."""

    def __init__(self, close: Sequence[int], think_end: int, log: Callable[[str], None] | None = None) -> None:
        self.close, self.think_end = [int(t) for t in close], int(think_end)
        self.detector = LoopDetector()
        self.open = True
        self.fired = False
        self.expect: tuple[int, int] | None = None     # (reasoning index, token) the last cut replaces
        self.record: dict = {}
        self.log = log

    def cut(self, tokens: Sequence[int]) -> tuple[int, list[int]] | None:
        if not self.open or self.fired:
            return None
        ends = next((i for i, t in enumerate(tokens) if int(t) == self.think_end), len(tokens))
        at = self.detector.ahead(list(tokens[:ends]))
        if at is None:
            self.expect = None
            return None
        self.expect = (len(self.detector.tokens) + at, int(tokens[at]))
        return at, list(self.close)

    def observe(self, token: int) -> None:
        if not self.open:
            return
        token = int(token)
        if token == self.think_end:
            self.open = False
            return
        if self.fired:                                 # the close's own tokens
            return
        n = len(self.detector.tokens)
        if self.expect is not None and self.expect[0] == n and token == self.close[0]:
            self.fired = True                          # this cut's close, written in place of the firing token
            self.record = {"reason": self.detector.why(self.expect[1]), "at": n}
            if self.log is not None:
                self.log(f"[tensorfold] think loop closed at reasoning token {n}: {self.record['reason']}")
            return
        self.detector.observe(token)


def hide(reasoning: str, phrase: str = CLOSE_TEXT, *, finished: bool) -> str:
    """The reasoning without the close's phrase. The gate writes it last, so only the end is cut: the phrase, or
    the start of it that is all a stream has sent so far or all ``max_tokens`` left room for. The same text is cut
    whether or not the reply ``finished``, so the final reasoning is the one the stream sent."""

    if reasoning.endswith(phrase):
        return reasoning[:-len(phrase)]
    for k in range(min(len(phrase) - 1, len(reasoning)), 0, -1):
        if reasoning.endswith(phrase[:k]):
            return reasoning[:-k]
    return reasoning


__all__ = ["CLOSE_TEXT", "ENV", "LoopDetector", "LoopGate", "enabled", "hide"]
