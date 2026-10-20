"""Copy drafts for Flash Next on CUDA: a repeated tail drafts its earlier continuation, the same chain on both ranks."""

from __future__ import annotations

import os
from collections.abc import Sequence

COPY_MATCH = 8          # the context's last this many tokens must have appeared before, and at least this many follow
ROWS_ENV = "TENSORFOLD_COPY_ROWS"


def enabled(default: bool = True) -> bool:
    """TENSORFOLD_COPY_DRAFTS=0 (or off, false, no) turns copy drafts off."""
    value = os.environ.get("TENSORFOLD_COPY_DRAFTS")
    return default if value is None else value.strip().lower() not in ("0", "off", "false", "no")


def most_rows(depth: int) -> int:
    """TENSORFOLD_COPY_ROWS: the widest window a copy verifies; unset, an MTP chain's (depth + 1)."""
    value = os.environ.get(ROWS_ENV, "").strip()
    return max(depth + 1, int(value)) if value else depth + 1


def next_rows(rows: int, landed_whole: bool, base: int, most: int) -> int:
    """A copy's next window: twice as wide after a whole copy, half after a broken one, within [base, most].

    The qwen3_5 path's rule (``next_copy_rows``) starts at 16 rows. Here the floor is the MTP chain's window, because
    on Flash Next a row past 7 costs 1.8-1.9 ms (tf-runs/verify-rows, 2026-10-20) and a 16-row window pays only once
    it commits 10 tokens: a copy widens only after one lands whole."""
    return min(most, max(rows, base) * 2) if landed_whole else max(base, min(rows, most) // 2)


class CopyIndex:
    """Longest continuation of an earlier copy of the last ``min_match`` tokens; n-gram starts keep a round O(new).

    With ``rows`` (the MTP chain's window), a copy drafts at most its own window less the anchor row, which widens
    toward ``most`` while copies land whole (``verified``); without, it drafts as many as asked."""

    def __init__(self, context: Sequence[int], min_match: int = COPY_MATCH, *, rows: int | None = None,
                 most: int | None = None):
        self.m = int(min_match)
        self.base = self.rows = rows
        self.most = max(rows, most or rows) if rows is not None else None
        self.ctx: list[int] = []
        self.starts: dict[tuple, list[int]] = {}
        self.extend(context)

    def extend(self, tokens: Sequence[int]) -> None:
        for t in tokens:
            self.ctx.append(int(t))
            s = len(self.ctx) - self.m
            if s >= 0:
                self.starts.setdefault(tuple(self.ctx[s:]), []).append(s)

    def chain(self, max_nodes: int) -> list[int]:
        """Up to ``max_nodes`` tokens (and the window's) after the tail's latest earlier copy; [] without one or
        under ``min_match``."""

        ctx, m = self.ctx, self.m
        if self.rows is not None:
            max_nodes = min(max_nodes, self.rows - 1)
        if len(ctx) < 2 * m or max_nodes < 1:
            return []
        # A round with fewer than min_match slots (Flash Next drafts 6) still copies: the earlier copy must run
        # min_match tokens on, and the round takes the first max_nodes of them.
        want = max(max_nodes, m)
        best: list[int] = []
        for start in reversed(self.starts.get(tuple(ctx[-m:]), [])):
            if start > len(ctx) - m - 1:                 # the needle itself
                continue
            cont = ctx[start + m:start + m + want]
            if len(cont) > len(best):
                best = cont
                if len(best) == want:
                    break
        return best[:max_nodes] if len(best) >= m else []

    def verified(self, drafted: int, kept: int) -> None:
        """A copy window's outcome: ``kept`` of its ``drafted`` tokens landed."""
        if self.rows is not None:
            self.rows = next_rows(self.rows, kept == drafted, self.base, self.most)
