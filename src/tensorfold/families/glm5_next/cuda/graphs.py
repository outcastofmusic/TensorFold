"""CUDA graphs preserve eager bits by replaying static buffers, device positions, and collectives, with empty attention chunks skipped."""

from __future__ import annotations

import torch

from . import latent, prof
from .forward import compute
from .mtp import mtp_compute


def pool_buckets(np_max: int) -> list[int]:
    """Every value ``sparse.pool_bucket`` can return for a sparse step (past TOPK_POOLS complete pools)."""
    out, b = [], 1024
    while b < np_max:
        out.append(b)
        b *= 2
    return out + [np_max]


class Graphs:
    def __init__(self, e, main_rows=(1, 2, 3, 4), mtp_rows=(1, 2, 3, 4)) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.main: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mtp: dict[int, torch.cuda.CUDAGraph] = {}
        self.sparse: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}      # (rows, parity, pool bucket)
        self.sparse_mtp: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}       # (rows, pool bucket)
        w, st = e.w, e.st
        self.load_mtp = w.mtp is not None and not w.meta.get("target_only", False) and w.meta.get("load_mtp", True)
        prof.active = False
        saved = list(st.cur)
        parities = (0, 1) if st.cur else (0,)
        self.warmed: set[str] = set()
        # TF_GLM_LAZY_SPARSE=1: a long-context shape is captured after its first step, which runs eagerly. At a
        # 98,304-token window start-up captures 72 of them in 7 s, and a reply uses the 12 of its own pool bucket.
        from tensorfold.env_flags import flag

        self.lazy = flag("TF_GLM_LAZY_SPARSE")
        self.rows = tuple(main_rows)
        with torch.no_grad():
            for R in main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(self._warm("main")):
                        compute(w, st, e.buf, R)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R)
                    self.main[(R, parity)] = g
            st.cur = saved
            if self.load_mtp:
                e.mbuf.zero_first = False
                for n in mtp_rows:
                    for _ in range(self._warm("mtp")):
                        mtp_compute(w, st, e.mbuf, n)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n)
                    self.mtp[n] = g
            if st.index is not None and latent.ENABLED:
                self._capture_sparse(e, main_rows, mtp_rows, parities)
        torch.cuda.synchronize()

    def capture_sparse(self, e, R: int, bucket: int) -> None:
        """Capture the sparse graph of the step that just ran eagerly (``lazy``), for the next step of its shape."""

        if not self.lazy or R not in self.rows:
            return
        with torch.no_grad():
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                compute(e.w, e.st, e.buf, R, sparse_np=bucket)
        self.sparse[(R, e.st.parity, bucket)] = g

    def _warm(self, kind: str) -> int:
        """Runs before a capture: 2 for a kind's first shape (lazy handles and workspaces), 1 for its later shapes.

        Start-up captures 12 shapes a sparse bucket, and a second run of each cost 5 of GLM's 50 s load.
        """

        first = kind not in self.warmed
        self.warmed.add(kind)
        return 2 if first else 1

    def _capture_sparse(self, e, main_rows, mtp_rows, parities) -> None:
        w, st = e.w, e.st
        np_max = st.index[0][2].shape[0] - 2
        saved = list(st.cur)
        for bucket in pool_buckets(np_max):
            for R in () if self.lazy else main_rows:
                for parity in parities:
                    st.cur = [parity] * len(st.cur)
                    for _ in range(self._warm("sparse")):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        compute(w, st, e.buf, R, sparse_np=bucket)
                    self.sparse[(R, parity, bucket)] = g
            if self.load_mtp:
                for n in mtp_rows:
                    for _ in range(self._warm("sparse_mtp")):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self.pool):
                        mtp_compute(w, st, e.mbuf, n, sparse_np=bucket)
                    self.sparse_mtp[(n, bucket)] = g
        st.cur = saved
