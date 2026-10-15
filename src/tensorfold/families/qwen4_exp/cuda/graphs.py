"""Capture static-buffer forwards by window size, context bucket and DeltaNet buffer parity, staging new inputs before replay with the eager path's kernels and launch parameters."""

from __future__ import annotations

import torch

from .forward import compute, stage
from .mtp import mtp_compute, mtp_stage


class Graphs:
    def __init__(self, e, *, max_rows: int = 8) -> None:
        self.e = e
        self.max_rows = max_rows
        self.main: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp_out: dict[tuple[int, int], torch.Tensor] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.captures = 0
        self.rows = max_rows                      # the window sizes warm() captured, which idle captures follow

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        import gc

        torch.cuda.synchronize()
        gc.collect()
        g = torch.cuda.CUDAGraph()
        # Collecting old graphs calls cuGraphExecDestroy and invalidates an active capture.
        enabled = gc.isenabled()
        gc.disable()
        try:
            # thread-local: NCCL's helper threads (tensor parallel) may call CUDA while this thread captures
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        self.captures += 1
        return g

    def _bucket(self, end: int) -> int:
        return min(self.e.st.capacity, max(8192, 1 << (end - 1).bit_length()))

    @torch.no_grad()
    def forward(self, tokens) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.buf
        segs = stage(w, b, [(st, tokens)])
        R = segs[-1][2]
        if R > self.max_rows:
            return compute(w, segs, b)
        context = self._bucket(st.pos + R)
        key = (R, st.cur[0] if st.cur else 0, context)
        g = self.main.get(key)
        if g is None:
            compute(w, segs, b, context=context)     # eager warm-up: compiles this launch shape
            g = self._capture(lambda: compute(w, segs, b, context=context))
            self.main[key] = g
        g.replay()
        return b.logits[:R]

    @torch.no_grad()
    def mtp_forward(self, next_tokens, streams: torch.Tensor) -> torch.Tensor:
        e = self.e
        w, st, b = e.w, e.st, e.mbuf
        segs = mtp_stage(w, b, [(st, next_tokens, streams)])
        n = segs[-1][2]
        if n > self.max_rows:
            return mtp_compute(w, segs, b)
        context = self._bucket(st.mtp_len + n)
        key = (n, context)
        g = self.mtp.get(key)
        if g is None:
            out = mtp_compute(w, segs, b, context=context)     # eager warm-up; its result is the view replays fill
            g = self._capture(lambda: mtp_compute(w, segs, b, context=context))
            self.mtp[key] = g
            self.mtp_out[key] = out
        g.replay()
        return self.mtp_out[key]

    @torch.no_grad()
    def warm(self, rows: int | None = None) -> int:
        """Capture all decode windows at both DeltaNet parities and all MTP step sizes before decoding; this dirties sequence state, so prefill afterward."""

        e = self.e
        st = e.st
        rows = rows or self.max_rows
        self.rows = rows
        saved = list(st.cur)
        before = self.captures
        for parity in (0, 1):
            st.cur = [parity] * len(st.cur)
            for R in range(1, rows + 1):
                self.forward([0] * R)
        st.cur = saved
        if e.mbuf is not None:
            for n in range(1, rows + 1):
                self.mtp_forward([0] * n, e.buf.streams[:n])
        torch.cuda.synchronize()
        return self.captures - before


    def missing(self, end: int) -> list[tuple]:
        """The graphs a round ending near ``end`` and the next bucket up would capture, not captured yet: ("main",
        rows, parity, context) and ("mtp", rows, context), for each window size warm() captured."""

        st = self.e.st
        now = self._bucket(end)
        out: list[tuple] = []
        for context in sorted({now, min(st.capacity, 2 * now)}):
            if context < end:
                continue
            out += [("main", R, parity, context) for parity in (0, 1) for R in range(1, self.rows + 1)
                    if (R, parity, context) not in self.main]
            if self.e.mbuf is not None:
                out += [("mtp", n, context) for n in range(1, self.rows + 1) if (n, context) not in self.mtp]
        return out

    def idle_steps(self, end: int) -> list:
        """One capture a step for ``missing(end)``. A capture writes only rows past the committed ones and the
        recurrent buffers, which a next request restores from a kept snapshot or resets, as warm() leaves them."""

        return [lambda key=key: self._capture_key(key) for key in self.missing(end)]

    @torch.no_grad()
    def _capture_key(self, key: tuple) -> None:
        e = self.e
        w, st = e.w, e.st
        if key[0] == "main":
            _, R, parity, context = key
            if (R, parity, context) in self.main or st.pos + R > context:
                return
            saved = list(st.cur)
            st.cur = [parity] * len(st.cur)
            try:
                segs = stage(w, e.buf, [(st, [0] * R)])
                compute(w, segs, e.buf, context=context)
                self.main[(R, parity, context)] = self._capture(lambda: compute(w, segs, e.buf, context=context))
            finally:
                st.cur = saved
            return
        _, n, context = key
        if (n, context) in self.mtp or st.mtp_len + n > context:
            return
        b = e.mbuf
        segs = mtp_stage(w, b, [(st, [0] * n, e.buf.streams[:n])])
        out = mtp_compute(w, segs, b, context=context)
        self.mtp[(n, context)] = self._capture(lambda: mtp_compute(w, segs, b, context=context))
        self.mtp_out[(n, context)] = out
