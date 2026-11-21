"""Hy4-flash's forward for one stream: prompt chunks, then one-row decode steps, sampled by the keyed sampler.

The layer order is transformers PR #50903's: an Engram memory added to the four streams (layers 1, 11, 21, 31), the
attention hyper-connection's collapse, the input norm, gated MLA, the write-back; the MLP hyper-connection's collapse,
the post-attention norm, the MLP (dense in layer 0, the latent MoE in the rest), the write-back. The head collapses the
streams through its own gates, norms and multiplies in fp32.

Attention keeps one 576-wide row a token and layer: the 512-wide normed latent (keys and values) and the 64 rotated
key channels. A query's no-rope part is absorbed into the latent through kv_b's key rows, and the output's latent
expanded through its value rows, so no layer materializes per-head keys. A sliding layer reads its last 512 keys. A
sparse layer reads every key below position 2,048, and from there the indexer's 511 best closed groups of four
tokens plus the row's own group: at most 2,048 keys a row at any context.

Every operation also runs on the CPU in fp32 torch ops, so the tests drive this code against transformers' reference.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from tensorfold.env_flags import flag

from ..config import SPARSE
from ..engram import ngram_tokens
from . import kernels as K
from .weights import DECODE_ROWS, Attention, Dense, Engram, Indexer, Layer, MoE, Weights, gather_engram

PREFILL_ROWS = 2048
# Engram's norms are nn.RMSNorm without an epsilon, which torch computes with fp32's (1.19e-7) even on bf16 rows, as
# Tencent's llama.cpp patch spells out. bf16's 2^-7 sat above the streams' mean square (0.002 at layer 1), shrank the
# query norm, divided the gate logits by about 3 and put layer 1's streams 69% (mean) off transformers' (21 Nov)
ENGRAM_EPS = float(os.environ.get("TF_HY4_ENGRAM_EPS", str(torch.finfo(torch.float32).eps)))


def _silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """bf16(bf16(silu(g)) * u), transformers' SwiGLU on bf16 tensors."""

    return (torch.nn.functional.silu(g.float()).to(torch.bfloat16).float() * u.float()).to(torch.bfloat16)


class Engine:
    """One stream's caches: ``forward`` commits a prompt chunk or decode rows, ``sample`` draws the next token."""

    def __init__(self, w: Weights, *, max_len: int = 8192, prefill_rows: int = PREFILL_ROWS):
        c = self.c = w.config
        self.w = w
        dev = self.device = w.device
        self.cuda = dev.type == "cuda"
        if max_len % c.index_pool:
            raise ValueError(f"max_len must be a multiple of {c.index_pool}")
        self.max_len, self.prefill_rows = max_len, prefill_rows
        self.latent = [torch.zeros((max_len, K.LW + K.RW), dtype=torch.bfloat16, device=dev) for _ in w.layers]
        self.group_topk = c.index_topk // c.index_pool                           # 512: the open group plus 511
        sparse = [i for i, layer in enumerate(w.layers) if layer.attn.indexer is not None]
        self.tkeys = {i: torch.zeros((max_len, c.index_dim), dtype=torch.bfloat16, device=dev) for i in sparse}
        self.gkeys = {i: torch.zeros((max_len // c.index_pool, c.index_dim), dtype=torch.bfloat16, device=dev)
                      for i in sparse}
        self.inv = {kind: (1.0 / (h.theta ** (torch.arange(0, h.rope, 2, dtype=torch.float32) / h.rope))).to(dev)
                    for kind, h in ((SPARSE, c.sparse), ("sliding", c.sliding))}
        self.idx_inv = (1.0 / (c.sparse.theta ** (torch.arange(0, c.sparse.rope, 2, dtype=torch.float32)
                                                  / c.sparse.rope))).to(dev)
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)            # meta[0]: the call's first position
        self.offsets = torch.arange(max(prefill_rows, DECODE_ROWS), dtype=torch.int32, device=dev)
        self.tok = torch.zeros(DECODE_ROWS, dtype=torch.int32, device=dev)   # a graphed call's input tokens
        self.ids: list[int] = []                                            # the committed tokens (n-grams, states)
        width = max(c.sparse.heads, c.sliding.heads)
        rows = max(prefill_rows, DECODE_ROWS)
        # a sparse row's 2,048 keys, or a draft block row's 2,048 and its block
        self.scratch = K.Scratch(rows, width, -(-(c.index_topk + 16) // K.CHUNK), dev) if self.cuda else None
        heads = max((len(s) for s in c.engram_sizes), default=0)
        self.eg_rows = (torch.zeros((len(c.engram_layers), rows, heads, c.engram_dim), dtype=torch.float32,
                                    device=dev) if heads else None)        # a call's Engram rows, staged on the host
        self.graphs = flag("TF_HY4_GRAPHS", True) and self.cuda
        from concurrent.futures import ThreadPoolExecutor

        self._readers = ThreadPoolExecutor(len(c.engram_layers) or 1, thread_name_prefix="engram-layer")
        self._ahead_pool = ThreadPoolExecutor(1, thread_name_prefix="engram-ahead")
        self._ahead: tuple | None = None
        # one graph a decode width, captured at its first use. Each keeps the normed rows and watched lists it writes,
        # which a drafter reads after it replays. A shared pool lets one width's temporaries sit in another's kept
        # rows, and widths replay in any order, so each gets its own unless TF_HY4_GRAPH_POOL=shared.
        # TF_HY4_GRAPH_WIDTHS=1 graphs one-row calls only (verify windows run eagerly)
        self.graph: dict[int, tuple] = {}
        self.graph_pool = (torch.cuda.graph_pool_handle()
                           if self.cuda and os.environ.get("TF_HY4_GRAPH_POOL") == "shared" else None)
        self.graph_widths = os.environ.get("TF_HY4_GRAPH_WIDTHS", "all")
        self.sampling = None
        self.pos = 0
        self.prompt = False
        # what a drafter reads after each call: the normed rows, the watched sparse layers' key lists by row
        # (sel, tok, cnt), and the row of the last committed token (``rewind`` moves it back)
        self.watch: set[int] = set()
        self.lists: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.normed: torch.Tensor | None = None
        self.src_row = 0
        # tests only: expert picks [T, top_k] by layer to use at positions 0.. (``replay``), and the engine's own (``seen``)
        self.replay: dict[int, torch.Tensor] | None = None
        self.seen: dict[int, list[torch.Tensor]] | None = None
        # tests only: the indexer's closed groups [T, 511] by sparse layer (``replay_groups``), and its own with the
        # margin between its last kept and first dropped group (``seen_groups``)
        self.replay_groups: dict[int, torch.Tensor] | None = None
        self.seen_groups: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] | None = None

    # -- state ---------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.pos = 0                       # keys past pos are never read: the caches need no clearing
        self.ids = []

    def state(self, n: int) -> dict:
        """A copy of the committed state after positions 0 .. n - 1 (n == pos): the caches' first n rows."""

        if n != self.pos or n <= 0:
            raise ValueError(f"no committed state at {n} (the engine is at {self.pos})")
        g = n // self.c.index_pool
        return {"pos": n, "ids": list(self.ids), "latent": [x[:n].clone() for x in self.latent],
                "tkeys": {i: t[:n].clone() for i, t in self.tkeys.items()},
                "gkeys": {i: t[:g].clone() for i, t in self.gkeys.items()}}

    def restore(self, st: dict) -> None:
        n = int(st["pos"])
        for dst, src in zip(self.latent, st["latent"]):
            dst[:n].copy_(src)
        for i, t in st["tkeys"].items():
            self.tkeys[i][:n].copy_(t)
        for i, t in st["gkeys"].items():
            self.gkeys[i][:t.shape[0]].copy_(t)
        self.ids = list(st["ids"])
        self.pos = n

    @staticmethod
    def state_bytes(st: dict) -> int:
        ts = list(st["latent"]) + list(st["tkeys"].values()) + list(st["gkeys"].values())
        return sum(t.numel() * t.element_size() for t in ts)

    def set_sampling(self, sampling) -> None:
        self.sampling = sampling

    def rewind(self, n: int) -> None:
        """Drop the last ``n`` committed rows (a window's rejected drafts); a later call overwrites their cache rows."""

        if n:
            self.pos -= n
            del self.ids[-n:]
            self.src_row -= n

    @property
    def eos(self) -> tuple[int, ...]:
        return self.c.eos

    # -- positions -----------------------------------------------------------------------------------------------
    def _positions(self, rows: int) -> torch.Tensor:
        """The call's positions from the device meta, so a captured step serves every position."""

        return self.meta[0] + self.offsets[:rows]

    @staticmethod
    def _rope(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
        """Interleaved pairs (x[2i], x[2i+1]) turned by pos * inv[i], with transformers' bf16 cos and sin."""

        ang = pos.float()[:, None] * inv[None, :]
        cos, sin = ang.cos().to(torch.bfloat16).float(), ang.sin().to(torch.bfloat16).float()
        shape = [cos.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[1]]
        cos, sin = cos.view(shape), sin.view(shape)
        f = x.float()
        a, b = f[..., 0::2], f[..., 1::2]
        out = torch.stack([a * cos - b * sin, b * cos + a * sin], dim=-1).flatten(-2)
        return out.to(torch.bfloat16)

    # -- layers --------------------------------------------------------------------------------------------------
    def _attention(self, li: int, a: Attention, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        c = self.c
        rows = x.shape[0]
        heads = c.heads_of(li)
        pos = self._positions(rows)
        inv = self.inv[SPARSE if a.kind == SPARSE else "sliding"]
        gate = a.gate(x, prompt)
        qr = K.rmsnorm(a.q_a(x, prompt), a.q_norm, c.eps)
        q = a.q_b(qr, prompt).view(rows, heads.heads, heads.qk)
        ckv = a.kv_a(x, prompt)
        lat = torch.empty((rows, K.LW + K.RW), dtype=torch.bfloat16, device=x.device)
        lat[:, :K.LW] = K.rmsnorm(ckv[:, :c.kv_rank], a.kv_norm, c.eps)
        lat[:, K.LW:] = self._rope(ckv[:, c.kv_rank:], pos, inv)
        self.latent[li].index_copy_(0, pos.long(), lat)
        qa = torch.empty((rows, heads.heads, K.LW + K.RW), dtype=torch.bfloat16, device=x.device)
        qa[:, :, :K.LW] = K.absorb(q[:, :, :heads.nope], a.wk, torch.empty(
            (rows, heads.heads, K.LW), dtype=torch.bfloat16, device=x.device))
        qa[:, :, K.LW:] = self._rope(q[:, :, heads.nope:], pos, inv)
        lo, cnt, sel, tok = self._keys(li, a, x, qr, pos, prompt)
        o_lat = torch.empty((rows, heads.heads, K.LW), dtype=torch.bfloat16, device=x.device)
        K.attention(qa, self.latent[li], lo, cnt, sel, tok, a.sinks, heads.qk ** -0.5, o_lat, self.scratch)
        v = K.expand(o_lat, a.wv, torch.empty((rows, heads.heads, heads.v), dtype=torch.bfloat16, device=x.device))
        sig = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
        out = (v.reshape(rows, -1).float() * sig).to(torch.bfloat16)
        return K.rmsnorm(a.o(out, prompt), a.post_norm, c.eps)

    def _keys(self, li: int, a: Attention, x: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, prompt: bool):
        """Each row's key list: (lo, cnt, sel, tok) for ``kernels.attention``."""

        c = self.c
        rows = x.shape[0]
        p32 = pos.to(torch.int32)
        if a.indexer is None:
            lo = torch.clamp(p32 - (c.window - 1), min=0)
            tok = torch.zeros((rows, 1), dtype=torch.int32, device=x.device)
            return lo, p32 - lo + 1, torch.zeros_like(p32), tok
        tok, sel, cnt = self._index(li, a.indexer, x, qr, pos, prompt)
        cnt = torch.where(sel != 0, cnt, p32 + 1)
        if li in self.watch:
            self.lists[li] = (sel, tok, cnt)
        return torch.zeros_like(p32), cnt, sel, tok

    def _index(self, li: int, ix: Indexer, x: torch.Tensor, qr: torch.Tensor, pos: torch.Tensor, prompt: bool):
        """The compressed indexer: cache the rows' token keys, pool the groups they touch, and for rows past
        ``index_topk`` keys pick the 511 best closed groups plus the row's own group (tokens ascending)."""

        c = self.c
        rows = x.shape[0]
        pool, dim, rope = c.index_pool, c.index_dim, c.sparse.rope
        tk = K.rmsnorm(ix.wk(x, prompt), ix.k_norm, c.eps)
        self.tkeys[li].index_copy_(0, pos.long(), tk)
        # every group the call touches: a group's key is right once its last token is in, and no row reads a group
        # before it closes
        groups = (rows + pool - 1) // pool + 1
        g0 = self.meta[0] // pool
        gids = (g0 + torch.arange(groups, device=x.device, dtype=torch.int32)).clamp(max=self.max_len // pool - 1)
        gk = K.pool(self.tkeys[li], gids, ix.pv, ix.pg, ix.ape, ix.g_norm, c.eps)
        gk = torch.cat([gk[:, :dim - rope], self._rope(gk[:, dim - rope:], gids * pool, self.idx_inv)], -1)
        self.gkeys[li].index_copy_(0, gids.long(), gk)
        closed = (pos // pool).to(torch.int32)
        sel = (closed >= self.group_topk).to(torch.int32)
        tok = torch.zeros((rows, c.index_topk), dtype=torch.int32, device=x.device)
        cnt = (self.group_topk - 1) * pool + pos.to(torch.int32) % pool + 1
        if not self._sparse_rows(rows):
            return tok, sel, cnt
        qi = ix.wq(qr, prompt).view(rows, c.index_heads, dim)
        qi = torch.cat([qi[..., :dim - rope], self._rope(qi[..., dim - rope:], pos, self.idx_inv)], -1)
        hw = ix.wh(x, prompt).float() * (c.index_heads ** -0.5 * dim ** -0.5)
        groups = self.max_len // pool if self.graph_capturing else (self.pos + rows) // pool + 1
        logits = K.group_scores(qi, hw, self.gkeys[li], closed, groups)
        ranked = logits.topk(min(self.group_topk, logits.shape[-1]), dim=-1)
        top = ranked.indices[:, :self.group_topk - 1].sort(-1).values.to(torch.int32)                   # [R, 511]
        if self.seen_groups is not None:
            margin = ranked.values[:, self.group_topk - 2] - ranked.values[:, -1]
            self.seen_groups.setdefault(li, []).append((top.clone(), margin.clone()))
        if self.replay_groups is not None:
            forced = self.replay_groups[li][self.pos:self.pos + rows].to(top.device, torch.int32)
            top = torch.where(sel[:, None] != 0, forced, top)
        expand = (top[:, :, None] * pool + torch.arange(pool, device=x.device, dtype=torch.int32)).view(rows, -1)
        own = closed[:, None] * pool + torch.arange(pool, device=x.device, dtype=torch.int32)
        tok[:, :expand.shape[1]] = expand
        tok[:, expand.shape[1]:] = own
        return tok, sel, cnt

    def _sparse_rows(self, rows: int) -> bool:
        """Whether any row of the call can sit past the dense limit (host-known; a graph always scores)."""

        return self.graph_capturing or self.pos + rows - 1 >= self.c.index_topk

    graph_capturing = False

    def _engram(self, j: int, e: Engram, x: torch.Tensor) -> torch.Tensor:
        """The memory's residual for the streams x [R, S, D] from the rows staged in ``eg_rows[j]``."""

        c = self.c
        rows = x.shape[0]
        heads = e.head_norms.shape[0]
        emb = self.eg_rows[j, :rows, :heads]                                        # fp32 [R, heads, dim]
        flat = K.rms(emb, e.head_norms, ENGRAM_EPS).to(torch.bfloat16).reshape(rows, -1)
        out = torch.empty_like(x)
        main = e.pre_gamma.float() * K.rms(e.value(flat, self.prompt), None, ENGRAM_EPS)
        for s, key_proj in enumerate(e.keys):
            key = K.rms(key_proj(flat, self.prompt), e.norm_key[s], ENGRAM_EPS).to(torch.bfloat16)
            query = K.rms(x[:, s], e.norm_query[s], ENGRAM_EPS).to(torch.bfloat16)
            g = (K.dot_rows(key, query) * c.hidden ** -0.5).to(torch.bfloat16).float()
            g = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
            out[:, s] = ((g[:, None] + e.hc_scales[s].float()) * main).to(torch.bfloat16)
        return out

    def _mlp(self, li: int, m, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        c = self.c
        if isinstance(m, Dense):
            return K.rmsnorm(m.down(_silu_mul(m.gate(x, prompt), m.up(x, prompt)), prompt), m.post_norm, c.eps)
        return self._moe(li, m, x, prompt)

    def _moe(self, li: int, m: MoE, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        c = self.c
        rows = x.shape[0]
        if self.cuda:
            from tensorfold.cuda import moe as moe_mod

            logits = moe_mod.router(x, m.router)
        else:
            logits = x.float() @ m.router.float().t()
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + m.bias, c.top_k, dim=-1).indices
        if self.seen is not None:
            self.seen.setdefault(li, []).append(idx.clone())
        if self.replay is not None:
            idx = self.replay[li][self.pos:self.pos + rows].to(idx.device)
        wts = K.norm_weights(scores.gather(1, idx), c.scaling)
        lat = m.pre(x, prompt)
        routed = self._experts(m, lat, idx, wts)
        shared = m.s_down(_silu_mul(m.s_gate(x, prompt), m.s_up(x, prompt)), prompt)
        h = (m.post(routed, prompt).float() + shared.float()).to(torch.bfloat16)
        return K.rmsnorm(h, m.combine_norm, c.eps)

    def _experts(self, m: MoE, lat: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        """Each row's top-k experts on the latent, combined by their weights in fp32 -> [R, latent] bf16."""

        c = self.c
        rows = lat.shape[0]
        if not self.cuda:
            gate, up, down = m.experts
            out = torch.zeros((rows, c.latent), dtype=torch.float32)
            for r in range(rows):
                for k in range(c.top_k):
                    e = int(idx[r, k])
                    g = (lat[r].float() @ gate[e].float().t()).to(torch.bfloat16)
                    u = (lat[r].float() @ up[e].float().t()).to(torch.bfloat16)
                    if c.swiglu_limit:
                        g, u = g.clamp(max=c.swiglu_limit), u.clamp(-c.swiglu_limit, c.swiglu_limit)
                    act = _silu_mul(g, u)
                    out[r] += (act.float() @ down[e].float().t()).to(torch.bfloat16).float() * wts[r, k]
            return out.to(torch.bfloat16)
        from tensorfold.cuda import experts as grouped
        from tensorfold.cuda.nvfp4 import experts as nvx

        prompt = rows > DECODE_ROWS
        plan = grouped.Plan(rows, c.top_k, c.experts, lat.device, prefill=prompt)
        tile = prompt and rows >= nvx.TILE_ROWS
        grouped.route(idx.to(torch.int32).contiguous(), plan, nvx.TILE_ITEM if tile else nvx.PREFILL_TILE)
        act = torch.empty((rows * c.top_k, c.moe_width), dtype=torch.bfloat16, device=lat.device)
        nvx.gate_up(lat.contiguous(), m.experts, plan, act, rows, tile=tile)
        y = torch.empty((rows * c.top_k, c.latent), dtype=torch.float32, device=lat.device)
        nvx.down(act, m.experts, plan, y, rows, tile=tile)
        return K.combine(y.view(rows, c.top_k, c.latent), wts)

    def _layers(self, ids: torch.Tensor, prompt: bool) -> torch.Tensor:
        """Every layer over the rows ``ids`` at positions pos..: the final-normed rows [R, D] bf16."""

        c, w = self.c, self.w
        self.prompt = prompt
        rows = ids.shape[0]
        x = w.embed[ids.long()][:, None, :].expand(rows, c.hc, c.hidden).contiguous()
        for i, layer in enumerate(w.layers):
            x = self._layer(i, layer, x, prompt)
        _, h = K.hc_pre(x, w.hc_head.fn, w.hc_head.base, w.hc_head.scale, eps=c.eps, hc_eps=c.hc_eps, head=True)
        return K.rmsnorm(h, w.norm, c.eps)

    def _layer(self, i: int, layer: Layer, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        c = self.c
        if layer.engram is not None:
            x = (x.float() + self._engram(c.engram_layers.index(i), layer.engram, x).float()).to(torch.bfloat16)
        hc = layer.attn_hc
        post, h = K.hc_pre(x, hc.fn, hc.base, hc.scale, eps=c.eps, hc_eps=c.hc_eps, magnitude=c.hc_magnitude)
        y = self._attention(i, layer.attn, K.rmsnorm(h, layer.input_norm, c.eps), prompt)
        K.hc_post(x, post, y)
        hc = layer.ffn_hc
        post, h = K.hc_pre(x, hc.fn, hc.base, hc.scale, eps=c.eps, hc_eps=c.hc_eps, magnitude=c.hc_magnitude)
        y = self._mlp(i, layer.mlp, K.rmsnorm(h, layer.post_norm, c.eps), prompt)
        K.hc_post(x, post, y)
        return x

    def head(self, normed: torch.Tensor) -> torch.Tensor:
        """fp32 logits [n, V]: the bf16 rows times the bf16 head, summed in fp32 (``enable_lm_head_fp32``)."""

        return K.head(normed, self.w.head)

    # -- Engram rows ---------------------------------------------------------------------------------------------
    def _read_engram(self, tokens: Sequence[int], pos: int, previous: Sequence[int]) -> list[torch.Tensor]:
        """A call's Engram rows, every layer's table read at once: fp32 [R, heads, dim] a layer, on the host."""

        c = self.c
        grams = ngram_tokens(np.asarray(tokens), pos, np.asarray(previous[-(c.engram_ngram - 1):]), c.engram_ngram,
                             c.pad)
        layers = [self.w.layers[li].engram for li in c.engram_layers if li < len(self.w.layers)]
        jobs = [self._readers.submit(gather_engram, e.table, e.hasher.rows_of(grams), c.engram_dim) for e in layers]
        return [j.result() for j in jobs]

    def prefetch(self, tokens: Sequence[int], pos: int, previous: Sequence[int]) -> None:
        """Start reading the Engram rows of a later call (the next prompt chunk) while this one computes."""

        if self.eg_rows is not None:
            self._ahead = (pos, tuple(int(t) for t in tokens),
                           self._ahead_pool.submit(self._read_engram, list(tokens), pos, list(previous)))

    def _stage_engram(self, tokens: Sequence[int]) -> None:
        """Copy the call's Engram rows into ``eg_rows`` (before a graph replays): prefetched when ``prefetch`` read
        them, else read now."""

        if self.eg_rows is None:
            return
        ahead = self._ahead
        if ahead is not None and ahead[0] == self.pos and ahead[1] == tuple(int(t) for t in tokens):
            self._ahead = None
            got = ahead[2].result()
        else:                                     # a prefetch for a later call stays for it
            got = self._read_engram(tokens, self.pos, self.ids)
        rows = len(tokens)
        for j, part in enumerate(got):
            self.eg_rows[j, :rows, :part.shape[1]].copy_(part.to(self.device), non_blocking=False)

    # -- calls ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, tokens: Sequence[int], *, prompt: bool, every: bool = False) -> torch.Tensor:
        """Commit ``tokens`` at positions pos..; the logits of the last row, or of every row with ``every``."""

        rows = len(tokens)
        if not 1 <= rows <= (self.prefill_rows if prompt else DECODE_ROWS):
            raise ValueError(f"a call holds 1..{self.prefill_rows} prompt rows or 1..{DECODE_ROWS} decode rows")
        if self.pos + rows > self.max_len:
            raise ValueError("the KV cache is full")
        self.meta[0] = self.pos
        self._stage_engram(tokens)
        if self.graphs and not prompt and (rows == 1 or self.graph_widths == "all"):
            self.tok[:rows] = torch.as_tensor(list(tokens), dtype=torch.int32)
            logits = self._graphed(rows)
            logits = logits if every else logits[rows - 1:rows]
        else:
            ids = torch.as_tensor(list(tokens), dtype=torch.int32).to(self.device)
            normed = self._layers(ids, prompt)
            logits = self.head(normed if every else normed[rows - 1:rows])
            self.normed = normed
        self.src_row = rows - 1
        self.pos += rows
        self.ids.extend(int(t) for t in tokens)
        return logits

    def _step(self, rows: int) -> torch.Tensor:
        """``rows`` decode rows from ``tok``: every row's logits; the normed rows and watched lists it keeps."""

        self.normed = self._layers(self.tok[:rows], False)
        return self.head(self.normed)

    def _graphed(self, rows: int) -> torch.Tensor:
        """``_step`` by graph replay. A width's first call runs eagerly (it compiles the kernels), then captures it;
        capturing launches nothing, so the state moves once. The capture points ``normed`` and ``lists`` at buffers
        nothing has written yet, so the eager call's are put back: a drafter reads them next (served DFly2 indexed
        those unwritten lists and tripped a device-side assert on its first request, 21 Nov)."""

        have = self.graph.get(rows)
        if have is None:
            out = self._step(rows)
            normed, lists = self.normed, dict(self.lists)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            self.graph_capturing = True
            try:
                with torch.cuda.graph(g, pool=self.graph_pool, capture_error_mode="thread_local"):
                    captured = self._step(rows)
            finally:
                self.graph_capturing = False
            self.graph[rows] = (g, captured, self.normed, dict(self.lists))
            self.normed, self.lists = normed, lists
            return out
        g, captured, normed, lists = have
        g.replay()
        self.normed, self.lists = normed, dict(lists)
        return captured.clone()

    def sample(self, logits: torch.Tensor) -> int:
        """The token after position pos - 1 from its logits [1, V], by the request's keyed rule (greedy at T = 0)."""

        s = self.sampling
        if self.cuda:                         # the rule a drafted window's rows sample by (``draft_decode``)
            from tensorfold.cuda.sampling import sample_rows

            return sample_rows(logits[-1:], [self.pos], s)[0]
        if s is None or s.temperature <= 0:
            return int(torch.argmax(logits[0]))
        from tensorfold.engine.exact_sampling import MARGIN, choose

        vals, ids = torch.topk(logits[0].float(), min(logits.shape[-1], (s.top_k or logits.shape[-1]) + MARGIN))
        return choose(vals.numpy(), ids.numpy(), self.pos, s)


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float

    @property
    def rounds(self) -> int:
        return len(self.tokens) - 1

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(e: Engine) -> None:
    if e.cuda:
        torch.cuda.synchronize()


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling, *, resume: dict | None = None,
            keeps: Sequence[int] = (), keep: Callable[[int, dict], None] | None = None) -> int:
    """Commit the prompt in chunks and return its first sampled token (position len(prompt)).

    ``resume`` (an ``Engine.state`` of a strict prefix) starts at its length; ``keep(n, state)`` gets the state after
    each of ``keeps``, which cut the chunks there."""

    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prefill needs at least one token")
    begin = 0 if resume is None else int(resume["pos"])
    if not 0 <= begin < len(prompt):
        raise ValueError("a resumed prompt must extend the kept positions")
    e.reset()
    if resume is not None:
        e.restore(resume)
    e.set_sampling(sampling)
    stops = sorted({int(p) for p in keeps if begin < int(p) < len(prompt)} | {len(prompt)})
    s = begin
    logits = None
    for stop in stops:
        while s < stop:
            chunk = prompt[s:min(stop, s + e.prefill_rows)]
            after = s + len(chunk)
            if after < len(prompt):              # the next chunk's Engram rows, read while this one computes
                nxt = prompt[after:min(after + e.prefill_rows, next(p for p in stops if p > after))]
                e.prefetch(nxt, after, prompt[max(0, after - 2):after])
            logits = e.forward(chunk, prompt=True)
            s += len(chunk)
        if keep is not None and stop < len(prompt):
            keep(stop, e.state(stop))
    pending = e.sample(logits)
    if keep is not None and len(prompt) in keeps:
        keep(len(prompt), e.state(len(prompt)))
    return pending


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling, *, stop_eos: bool = True,
                  on_tokens: Callable[[list[int]], bool | None] | None = None) -> DecodeResult:
    """``count`` tokens (the first is ``pending``), one row a step; stops after an end token or ``on_tokens``' True."""

    e.set_sampling(sampling)
    out = [int(pending)]
    eos = set(e.c.eos)
    _sync(e)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in eos):
        tok = e.sample(e.forward([out[-1]], prompt=False))
        out.append(tok)
        if on_tokens is not None and on_tokens([tok]):
            break
    _sync(e)
    return DecodeResult(out, time.perf_counter() - start)


FIRST_COPY_ROWS = 16          # a copy's first window; it doubles while copies land whole, up to DECODE_ROWS


@torch.no_grad()
def draft_decode(e: Engine, context: Sequence[int], pending: int, count: int, sampling, *, stop_eos: bool = True,
                 on_tokens: Callable[[list[int]], bool | None] | None = None, drafter=None,
                 allow_copy: bool = True, trace: list | None = None) -> DecodeResult:
    """Drafted chains verified as one window, each token the one serial decoding gives: a copy from the context when
    one matches, else the ``drafter``'s block (DFly2), else one row. Every window row samples by the serial rule at
    its position, and rows past the first rejected draft are rewound."""

    from tensorfold.cuda.sampling import sample_rows
    from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, next_copy_rows

    e.set_sampling(sampling)
    out = [int(pending)]
    ctx = list(context) + out
    eos = set(e.c.eos)
    copies = CopyIndex() if allow_copy else None
    rows = FIRST_COPY_ROWS
    stats = {"rounds": 0, "drafted": 0, "accepted": 0, "copy": 0, "draft_s": 0.0}
    _sync(e)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in eos):
        t0 = time.perf_counter()
        room = min(count - len(out) - 1, DECODE_ROWS - 1)
        guesses = copies.propose(ctx, min(rows, DECODE_ROWS) - 1)[:room] if copies is not None else []
        copied = bool(guesses)
        if not guesses and drafter is not None and room > 0:
            guesses = drafter.propose(e, out[-1], sampling)[:room]
        stats["draft_s"] += time.perf_counter() - t0
        tokens = [out[-1]] + guesses
        base = e.pos
        logits = e.forward(tokens, prompt=False, every=True)
        sampled = sample_rows(logits, [base + 1 + i for i in range(len(tokens))], sampling)
        n = 0
        while n < len(guesses) and guesses[n] == sampled[n] and not (stop_eos and sampled[n] in eos):
            n += 1
        if trace is not None:
            trace.append({"base": base, "tokens": tokens, "sampled": sampled, "accepted": n, "copied": copied})
        e.rewind(len(tokens) - (n + 1))
        if drafter is not None:
            drafter.commit(e, n)
        new = guesses[:n] + [sampled[n]]
        out.extend(new)
        ctx.extend(new)
        if copied:
            rows = next_copy_rows(rows, n == len(guesses), FIRST_COPY_ROWS, DECODE_ROWS)
            stats["copy"] += 1
        stats["rounds"] += 1
        stats["drafted"] += len(guesses)
        stats["accepted"] += n
        if on_tokens is not None and on_tokens(new):
            break
    _sync(e)
    res = DecodeResult(out, time.perf_counter() - start)
    res.stats = stats
    return res
