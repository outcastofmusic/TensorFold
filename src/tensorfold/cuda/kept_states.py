"""Kept prompt states in GPU memory, for Nemotron 3.5 Bolt and LFM2-MoE: a prompt resumes from the longest kept
prefix instead of prefilling it.

A kept state is a family engine's copy of its state after a prompt position (``Engine.state``, whose contents each
family defines), with a drafter's context when the prefill fed one. A recurrent state exists only after the last row a
pass committed, so a prefill takes states as it passes their positions (``keep_points``): the message starts
``tensorfold.cuda.markers`` finds, and the prompt end less one token, since a next turn sent back without its reasoning
renders that last token differently. Only prompt rows are kept, since decode and prefill bits differ.

How close a resumed prompt ends to a fresh prefill depends on the family. Bolt's prompt kernels are chunk-invariant,
so it resumes bit-exact. LFM2-MoE resumes within its tolerance: a resume can leave a short last chunk, and chunks of
up to ``PAIRS_ROWS`` (64) rows run the experts on ``pairs`` where a fresh prefill's long chunk runs ``dot``, which
rounds differently (tests/cuda/test_lfm2_moe.py checks resumed logits against the same bounds as fresh ones).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch


def tensor_bytes(obj) -> int:
    """Bytes of the tensors in nested dicts, lists and tuples."""

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def keep_points(prompt: Sequence[int], begin: int, marks: Callable[[Sequence[int]], list[int]] | None) -> list[int]:
    """Where a prefill from ``begin`` keeps states: message starts at least ``MIN_GAP`` past ``begin``, and the prompt
    end less one token unless a kept message start sits within ``MIN_GAP`` before it."""

    from tensorfold.cuda.markers import MIN_GAP

    end = max(1, len(prompt) - 1)
    points = [p for p in (marks(prompt) if marks is not None else []) if begin + MIN_GAP <= p < end]
    if end > begin and not (points and end - points[-1] < MIN_GAP):
        points.append(end)
    return points


@dataclass(eq=False)
class Kept:
    ids: tuple[int, ...]
    state: dict                    # Engine.state
    drafter: dict | None           # a drafter's context, None when the prefill fed no drafter
    nbytes: int

    def starts(self, prompt: Sequence[int]) -> bool:
        """Whether ``prompt`` strictly extends these ids (one prompt token is always left to prefill)."""

        return len(self.ids) < len(prompt) and tuple(prompt[:len(self.ids)]) == self.ids


class Store:
    """At most ``entries`` kept states in at most ``budget`` bytes, least recently used first."""

    def __init__(self, entries: int, budget: int) -> None:
        self.entries, self.budget = int(entries), int(budget)
        self.kept: list[Kept] = []

    @property
    def on(self) -> bool:
        return self.entries > 0 and self.budget > 0

    @property
    def held(self) -> int:
        return sum(k.nbytes for k in self.kept)

    def longest(self, prompt: Sequence[int], drafter: bool) -> Kept | None:
        """The longest kept prefix of ``prompt`` (with a drafter's context when ``drafter``), now the newest entry."""

        best = max((k for k in self.kept if k.starts(prompt) and (k.drafter is not None or not drafter)),
                   key=lambda k: len(k.ids), default=None)
        if best is not None:
            self.kept = [k for k in self.kept if k is not best] + [best]
        return best

    def add(self, ids: Sequence[int], state: dict, drafter: dict | None) -> None:
        """Keep a state, newest last; past the caps a superseded entry goes first, else the oldest."""

        entry = Kept(tuple(int(t) for t in ids), state, drafter, tensor_bytes(state) + tensor_bytes(drafter))
        if not self.on or entry.nbytes > self.budget:
            return
        self.kept = [k for k in self.kept if k.ids != entry.ids] + [entry]
        while len(self.kept) > self.entries or self.held > self.budget:
            self.kept.remove(self._superseded() or self.kept[0])

    def _superseded(self) -> Kept | None:
        """The oldest entry that a newer entry extends and an entry it extends precedes: a conversation's earlier turn.

        A chain's shortest entry (a system block's end, which other conversations resume from) and its newest turn
        stay; only a regenerated or edited turn resumes from one between them.
        """

        for i, k in enumerate(self.kept):
            n = len(k.ids)
            if any(len(d.ids) > n and d.ids[:n] == k.ids for d in self.kept[i + 1:]) and \
                    any(len(f.ids) < n and k.ids[:len(f.ids)] == f.ids for f in self.kept):
                return k
        return None
