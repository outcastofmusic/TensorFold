"""The NVFP4 shared expert on a second CUDA stream beside the routed experts (TF_FLASHNEXT_SHARED_STREAM).

The idea is MiaAI-Lab's Qwen3.8 Flash Next recipe, patch 0016 (decode rounds 104.6 -> 108.8 tok/s greedy prose at 5
streams, with its lane-batched DeltaNet, which ``gdn_multi`` already has here); this is a reimplementation on torch
streams. Only the schedule moves: the side stream runs the kernels the compute stream ran, on the same arguments, so
every byte out is the same. The shared expert reads x (rows the compute stream wrote before the fork) and writes slot
``top_k`` of ``buf.act`` and ``buf.y``, which the routed kernels skip (``skip=ex.routed``); its temporaries are
allocated and freed on the side stream, so the caching allocator hands them to nothing else until the side stream
has run past them. The compute stream joins before anything reads ``buf.y``, so the next layer's write-back, the
next round's writes into x and every host wait see the shared expert's work done.

Under CUDA graph capture the fork's wait on the capturing stream pulls the side stream into the capture, and the
join puts it back before ``moe`` returns, so a graph ends with no stream left outside it. One stream and two events a
device, made on first use: nothing grows with --parallel or with rounds. The side stream's freed temporaries stay in
its own pool (decode windows: under 1 MB; prompt chunks with mode 2: tens of MB at 3,072 rows).

The EXL3 and MLX checkpoints run the shared expert as expert E inside the grouped launch, so there is nothing to move.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from tensorfold.env_flags import choice

ENV = "TF_FLASHNEXT_SHARED_STREAM"
OFF, DECODE, ALL = 0, 1, 2          # 1: decode windows and head passes; 2: prompt chunks too


@lru_cache(maxsize=1)
def mode() -> int:
    """TF_FLASHNEXT_SHARED_STREAM: 0 one stream, 1 (default) decode windows and head passes, 2 every pass. Read
    once a process: captured graphs keep the schedule they were captured with (tests clear the cache). FP4-Local,
    mode 1 against 0: 1.033x at one request and 1.038x at four, replies identical 280/280; mode 2 added nothing
    (tf-runs/shared-1028, 2026-10-28)."""

    return int(choice(ENV, ("0", "1", "2"), "1"))


def applies(prompt_pass: bool, m: int | None = None) -> bool:
    """Whether a pass (``prompt_pass``: a prompt chunk's buffers) runs its shared experts on the side stream."""

    m = mode() if m is None else m
    return m == ALL or (m == DECODE and not prompt_pass)


class Side:
    """A device's side stream and its fork and join events (recorded again each layer; a wait takes the record in
    place when it is queued, so reuse orders nothing wrongly)."""

    def __init__(self, device: torch.device) -> None:
        self.stream = torch.cuda.Stream(device)
        self.forked = torch.cuda.Event()
        self.joined = torch.cuda.Event()

    def fork(self) -> torch.cuda.Stream:
        """The side stream, behind everything the current stream has queued; run the shared expert under it."""

        self.forked.record(torch.cuda.current_stream(self.stream.device))
        self.stream.wait_event(self.forked)
        return self.stream

    def join(self) -> None:
        """The current stream continues behind everything queued on the side stream."""

        self.joined.record(self.stream)
        torch.cuda.current_stream(self.stream.device).wait_event(self.joined)


_SIDES: dict[int, Side] = {}


def side(device: torch.device, prompt_pass: bool) -> Side | None:
    """The device's Side when this pass uses it, else None (made on first use)."""

    if device.type != "cuda" or not applies(prompt_pass):
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    got = _SIDES.get(index)
    if got is None:
        got = _SIDES[index] = Side(torch.device("cuda", index))
    return got
