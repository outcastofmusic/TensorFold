"""Nemotron 3.5 Bolt's verify windows as CUDA graphs: the eager window's kernels and launch shapes, replayed.

A window's graph holds ``Engine._forward(rows)``: every layer, the bf16 head, the grammar bias when masked, and the
keyed sampler, about 1,300 launches. Every input it reads is a buffer the engine keeps for its lifetime: the token ids
and ``meta`` (pos, parity, prev_keep), which ``verify`` copies in from pinned memory before each replay, the KV caches
and Mamba states, the grammar bias and the sampling parameters. Each kernel takes its position from ``meta`` on the
device (the KV write, the attention chunks, the Mamba conv and scan with their kept-row replay, the sampler's key), so
one graph serves every position and a replay gives the eager window's bits.

What a graph compiles in from the host is its key: the row count, the sampling mode (greedy, or keyed with its top_k,
top-p and min-p switches, ``sampling_mode``) and whether a grammar masks the window. A key is captured on its first
window, right after that window ran eagerly: the eager run is the warm-up (Triton compiles, the IQ2_XXS occupancy
cache fills, the selector table reaches the GPU), and a capture executes nothing, so the state the eager window left
stays as it is.

Graphs share one memory pool. Replays never overlap, and a graph's temporaries are dead once its replay ends, so the
pool holds one window's largest temporaries plus each graph's tap rows, which ``tap_rows`` reads after the replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SETS = 4                 # (sampling mode, masked) sets of graphs kept; past it the least recently used set goes
EXEC_BYTES = 2 << 20     # a captured window's own device memory (its kernel nodes): an estimate the GPU test prints


@dataclass
class Captured:
    graph: Any                                       # torch.cuda.CUDAGraph
    taps: list = field(default_factory=list)         # the tap streams its replay writes, in tap-slot order


class WindowGraphs:
    """Captured windows by set (sampling mode, masked) and row count; at most ``sets`` sets."""

    def __init__(self, sets: int = SETS) -> None:
        if sets < 1:
            raise ValueError("window graphs need at least one set")
        self.sets = int(sets)
        self.by_set: dict[tuple, dict[int, Captured]] = {}
        self.pool = None
        self.captures = 0

    def __len__(self) -> int:
        return sum(len(g) for g in self.by_set.values())

    def get(self, key: tuple, rows: int) -> Captured | None:
        """The graph of a ``rows``-row window in set ``key``, now the most recently used set (None: not captured)."""

        graphs = self.by_set.pop(key, None)
        if graphs is None:
            return None
        self.by_set[key] = graphs
        return graphs.get(rows)

    def add(self, key: tuple, rows: int, captured: Captured) -> None:
        """Keep a capture; past ``sets`` sets the least recently used set's graphs are released."""

        graphs = self.by_set.pop(key, {})
        graphs[rows] = captured
        self.by_set[key] = graphs
        self.captures += 1
        while len(self.by_set) > self.sets:
            del self.by_set[next(iter(self.by_set))]


def graph_bytes(c, max_len: int, rows: int, taps: int, sets: int = SETS) -> int:
    """An upper bound on what the window graphs hold on the GPU: the shared pool (one ``rows``-row window's
    temporaries with a quarter for the allocator's rounding, and every graph's tap rows) and each graph's own nodes."""

    from tensorfold.families.nemotron_h.cuda.attention import CHUNK

    head = rows * c.vocab * (2 + 8 * 9)            # bf16 logits, then the nucleus sampler's fp64 and int64 rows
    attention = rows * (max_len // CHUNK) * c.heads * (c.head_dim + 2) * 4 + rows * c.heads * c.head_dim * 2
    experts = max((k * (2 * c.latent + 2 * w) * 2 for k, w in zip(c.top_k, c.moe_width)), default=0) * rows
    rest = rows * (c.proj_dim * 2 + 2 * c.conv_dim * 2 + c.xd * 4 + 8 * c.hidden * 2)
    kept = sets * sum(range(1, rows + 1)) * taps * c.hidden * 2
    return (head + attention + experts + rest) * 5 // 4 + kept + sets * rows * EXEC_BYTES
