"""Copy drafts for Flash Next on CUDA: a repeated tail drafts its earlier continuation, the same chain on both ranks."""

from __future__ import annotations

import os
from collections.abc import Sequence

COPY_MATCH = 8          # the context's last this many tokens must have appeared before, and at least this many follow
ROWS_ENV = "TENSORFOLD_COPY_ROWS"
COPY_ROWS = 32          # the widest copy window: 1.25x one stream, 1.23x four (tf-runs/copy-windows, 2026-10-20)
# TENSORFOLD_COPY_BACK: a match this many tokens further back counts as the same text, and the longest continuation
# among those wins; 0 takes the latest occurrence with the longest continuation, the choice before 2026-10-20
BACK = int(os.environ.get("TENSORFOLD_COPY_BACK", "16").strip() or 16)
CANDIDATES = 32         # the latest occurrences a choice reads


def enabled(default: bool = True) -> bool:
    """TENSORFOLD_COPY_DRAFTS=0 (or off, false, no) turns copy drafts off."""
    value = os.environ.get("TENSORFOLD_COPY_DRAFTS")
    return default if value is None else value.strip().lower() not in ("0", "off", "false", "no")


def most_rows(depth: int) -> int:
    """TENSORFOLD_COPY_ROWS: the widest window a copy verifies, COPY_ROWS unset; 0 keeps an MTP chain's (depth + 1)."""
    value = os.environ.get(ROWS_ENV, "").strip()
    return max(depth + 1, int(value) if value else COPY_ROWS)


def next_rows(rows: int, landed_whole: bool, base: int, most: int) -> int:
    """A copy's next window: twice as wide after a whole copy, half after a broken one, within [base, most].

    The qwen3_5 path's rule (``next_copy_rows``) starts at 16 rows. Here the floor is the MTP chain's window, because
    on Flash Next a row past 7 costs 1.8-1.9 ms (tf-runs/verify-rows, 2026-10-20) and a 16-row window pays only once
    it commits 10 tokens: a copy widens only after one lands whole."""
    return min(most, max(rows, base) * 2) if landed_whole else max(base, min(rows, most) // 2)


class CopyIndex:
    """The continuation of the earlier copy of the last ``min_match`` tokens whose match runs furthest back; n-gram
    starts keep a round O(new).

    Following the latest occurrence copied the previous line of a list whose lines share a template (" EUR and ships
    in 3 days") where the prompt's own line was meant: a replay of the A/B replies gave 47% of copy drafts kept on GLM
    and 72% on Qwen, against 100% for the occurrence that matches furthest back (tf-runs/glm-copy-windows, copy_sim).
    ``min_back`` (GLM's TF_GLM_COPY_BACK, at most BACK): no copy unless the match runs this many tokens past
    ``min_match``.

    With ``rows`` (the MTP chain's window), a copy drafts at most its own window less the anchor row, which widens
    toward ``most`` while copies land whole (``verified``); without, it drafts as many as asked."""

    def __init__(self, context: Sequence[int], min_match: int = COPY_MATCH, *, rows: int | None = None,
                 most: int | None = None, min_back: int = 0):
        self.m, self.min_back = int(min_match), int(min_back)
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
        """Up to ``max_nodes`` tokens (and the window's) after the earlier copy whose match runs furthest back, the
        latest on a tie, among those matching at least ``min_back`` further; [] without one or under ``min_match``."""

        ctx, m = self.ctx, self.m
        if self.rows is not None:
            max_nodes = min(max_nodes, self.rows - 1)
        if len(ctx) < 2 * m or max_nodes < 1:
            return []
        # A round with fewer than min_match slots (Flash Next drafts 6) still copies: the earlier copy must run
        # min_match tokens on, and the round takes the first max_nodes of them.
        want = max(max_nodes, m)
        best, score = [], (False, -1, 0)
        for start in reversed(self.starts.get(tuple(ctx[-m:]), [])[-CANDIDATES:]):
            if start > len(ctx) - m - 1:                 # the needle itself
                continue
            cont = ctx[start + m:start + m + want]
            back = self._back(start)
            if len(cont) < m or back < self.min_back:
                continue
            # a continuation that fills the window first (in periodic text the latest occurrences run into the tail
            # and every match reaches far back), then the match's reach, then the continuation's length; ties go to
            # the later occurrence, seen first
            here = (len(cont) == want, back, len(cont))
            if here > score:
                best, score = cont, here
                if here == (True, BACK, want):
                    break
        return best[:max_nodes] if len(best) >= m else []

    def _back(self, start: int) -> int:
        """Tokens before the occurrence at ``start`` that equal those before the tail's match, up to BACK."""
        ctx, n = self.ctx, 0
        i, j = start - 1, len(self.ctx) - self.m - 1
        while n < BACK and i >= 0 and ctx[i] == ctx[j]:
            n, i, j = n + 1, i - 1, j - 1
        return n

    def verified(self, drafted: int, kept: int) -> None:
        """A copy window's outcome: ``kept`` of its ``drafted`` tokens landed."""
        if self.rows is not None:
            self.rows = next_rows(self.rows, kept == drafted, self.base, self.most)
