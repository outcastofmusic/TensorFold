"""GLM-5.3-Flash's tensor-parallel forward and commit; kernels keep rows apart, so row r has the serial step's bits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3.experts import Scratch as Exl3Scratch
from tensorfold.cuda.geometry import MLA_PROMPT_ATT_ROWS as PROMPT_ATT_ROWS   # a dense latent call's prompt rows
from tensorfold.cuda.kernels import prefill_attention, qmm as shared

from . import exl3_generic, glue, kda as kda_mod, latent, prof, qmm, route_log, sparse, packed
from .prefill import routed_micro_windows
from tensorfold.cuda.exl3 import experts as generic_exl3
from .attention import AttnScratch, attention, kv_write
from .weights import LayerW, Weights

# Expert skipping (changes replies; a measurement tool). TF_GLM_SKIP_SHARE=0.05: a routed pick under 5% of its row's routed
# weight is not read. TF_GLM_SKIP_LAYERS=A:B: only layers A <= index < B. TF_GLM_SKIP_RENORM=0: the kept picks keep their weights.
# TF_GLM_SKIP_FILE=path: a JSON file read before each request, {"share": x or [x per layer], "layers": [A, B], "renorm": bool,
# "keep": k (picks after a row's k heaviest are not read), "prompt": bool, "decode": bool (which rows skip)}, so one boot
# sweeps settings.
SKIP_SHARE = float(os.environ.get("TF_GLM_SKIP_SHARE", "0") or 0)
SKIP_LAYERS = tuple(int(x) for x in (os.environ.get("TF_GLM_SKIP_LAYERS") or "0:1000").split(":"))
SKIP_RENORM = os.environ.get("TF_GLM_SKIP_RENORM", "1") != "0"
SKIP_FILE = os.environ.get("TF_GLM_SKIP_FILE", "")
SKIP_ON = SKIP_SHARE > 0 or bool(SKIP_FILE)
_skip: dict = {}


def skip_table(device) -> torch.Tensor:
    """[68] on the device: a share per layer index, then the renormalize flag, the kept-pick count, and whether prompt rows
    and decode rows skip (graphs read it live)."""

    if "t" not in _skip:
        t = torch.ones((68,), dtype=torch.float32, device=device)
        t[:64] = 0.0
        t[SKIP_LAYERS[0]:min(SKIP_LAYERS[1], 64)] = SKIP_SHARE
        t[64], t[65] = float(SKIP_RENORM), 64.0
        _skip["t"] = t
    return _skip["t"]


def skip_reload() -> None:
    """Before a request: the skip file's settings into the table, when the file changed."""

    if not SKIP_FILE:
        return
    skip_table(torch.device("cuda"))         # eager engines make it in their first forward, too late for that request
    try:
        stamp = os.stat(SKIP_FILE).st_mtime_ns
        if stamp == _skip.get("stamp"):
            return
        with open(SKIP_FILE) as f:
            cfg = __import__("json").load(f)
    except (OSError, ValueError):
        return
    share = cfg.get("share", 0.0)
    a, b = cfg.get("layers", [0, 64])
    row = [float(x) for x in share] if isinstance(share, list) else [float(share) if a <= i < b else 0.0 for i in range(64)]
    row = (row + [0.0] * 64)[:64] + [float(bool(cfg.get("renorm", True))), float(cfg.get("keep", 64)),
                                     float(bool(cfg.get("prompt", True))), float(bool(cfg.get("decode", True)))]
    _skip["t"].copy_(torch.tensor(row, dtype=torch.float32))
    _skip["stamp"] = stamp


@dataclass
class Cut:
    """The recurrent state and convolution windows at an interior prompt row."""

    point: int
    rec: torch.Tensor
    conv: torch.Tensor


class Buffers:
    """Scratch for windows of up to ``rows`` rows, sliced [:R] for smaller ones; ``prefill`` for prompt chunks."""

    def __init__(self, w: Weights, rows: int, capacity: int = 2560, *, prefill: bool = False) -> None:
        c = w.cfg
        dev = w.device
        bf, f32 = torch.bfloat16, torch.float32
        D, S = c.hidden, c.streams
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.rows, self.prefill = rows, prefill
        head_rows = 1 if prefill else rows
        self.world = w.world
        self.ids = torch.zeros((rows,), dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((rows,), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.staged = torch.cuda.Event() if torch.cuda.is_available() else None
        if latent.ENABLED:
            # Dense attention only ever covers contexts up to the dense limit; longer rows go sparse.
            self.attn = None
            # a prompt chunk's dense pass runs PROMPT_ATT_ROWS rows at a time: its fp32 partials hold that many rows
            self.lat_s = latent.LatentScratch(rows, HL, latent.chunks_for(min(capacity, 2560) + rows), dev,
                                              lw=c.kv_lora, part_rows=PROMPT_ATT_ROWS if prefill else rows)
        else:
            self.attn = AttnScratch(1 if prefill else rows, HL, c.qk_dim, capacity, dev)
        if prefill:
            kda_layers = [l for l in w.layers if l.kind == "kda"]
            width = kda_layers[0].kda.proj.n if kda_layers else 0
            self.kproj = torch.zeros((1, rows, width), dtype=bf, device=dev)
            self.kscratch = kda_mod.KDAScratch(rows, LL, dev)
        self.hin = torch.empty((rows, c.hidden), dtype=torch.bfloat16, device=dev) if w.mtp is not None else None
        self.zero_first = False          # MTP: this step starts at position 0 (its embedding is zeroed)
        self.fp32_residual = packed.fp32_residual(w)
        self.x = torch.empty((rows, S * D), dtype=f32 if self.fp32_residual else bf, device=dev)
        self.normed = torch.empty((rows, D), dtype=bf, device=dev)
        self.xs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.post = torch.empty((rows, S), dtype=f32, device=dev)
        self.comb = torch.empty((rows, S * S), dtype=f32, device=dev)
        self.hcpart = torch.empty((rows, glue.HC_BLOCKS, 32), dtype=f32, device=dev)
        # KDA
        self.ka = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.kg = torch.empty((rows, LL * 128), dtype=bf, device=dev)
        self.xs_fa = torch.empty((rows, 2), dtype=f32, device=dev)
        self.xs_ga = torch.empty((rows, 2), dtype=f32, device=dev)
        self.kxs = torch.empty((rows, LL * 128 // 64), dtype=f32, device=dev)
        # DSA
        self.dp = torch.empty((rows, c.q_lora + c.kv_lora), dtype=bf, device=dev)
        self.qr = torch.empty((rows, c.q_lora), dtype=bf, device=dev)
        self.xs_qr = torch.empty((rows, c.q_lora // 64), dtype=f32, device=dev)
        self.lat = torch.empty((rows, c.kv_lora), dtype=bf, device=dev)
        self.xs_lat = torch.empty((rows, c.kv_lora // 64), dtype=f32, device=dev)
        self.q = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.kn = torch.empty((rows, HL, c.qk_dim), dtype=bf, device=dev)
        self.vn = torch.empty((rows, HL, c.v_dim), dtype=bf, device=dev)
        self.xs_ao = torch.empty((rows, HL * c.v_dim // 64), dtype=f32, device=dev)
        # DSA indexer (long contexts)
        self.ikr = torch.empty((rows, c.index_dim + c.index_heads), dtype=bf, device=dev)
        self.igr = torch.empty((rows, c.index_dim), dtype=f32, device=dev)
        self.qi = torch.empty((rows, c.index_heads * c.index_dim), dtype=bf, device=dev)
        # dense MLP
        dl = c.dense_width // w.world
        self.gu = torch.empty((rows, 2 * dl), dtype=bf, device=dev)
        self.act = torch.empty((rows, dl), dtype=bf, device=dev)
        self.xs_act = torch.empty((rows, dl // 64), dtype=f32, device=dev)
        # MoE
        slots = c.top_k + 1
        ml = c.moe_width // w.world
        self.mlog = torch.empty((rows, c.experts), dtype=f32, device=dev)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=dev)
        self.wts = torch.empty((rows, slots), dtype=f32, device=dev)
        self.eact = torch.empty((rows * slots, ml), dtype=bf, device=dev)
        exl3 = c.quant == "exl3"
        self.ey = None if exl3 else torch.empty((rows, slots, D), dtype=bf if prefill else f32, device=dev)
        self.plan = None if exl3 else grouped.Plan(rows, slots, c.experts + 1, dev, prefill=prefill)
        self.exl3 = None
        if c.quant == "exl3":            # EXL3 routed experts, and the shared expert as a BF16 MLP
            sl = c.shared_width // w.world
            if w.meta.get("packed_exl3"):
                # packed TP1: a prompt chunk's routed experts run in windows of up to MOE_WINDOW rows, copied into ey
                ex = next((l.moe.experts for l in w.layers if l.moe is not None),
                          w.mtp.layer.moe.experts if w.mtp is not None else None)
                self.exl3 = generic_exl3.Scratch(ex, min(rows, packed.MOE_WINDOW), slots, device=dev) if ex is not None else None
                self.ey = torch.empty((rows, slots, D), dtype=f32, device=dev)
            else:
                shape = SimpleNamespace(dims=D, width=ml, count=c.experts)
                self.exl3 = Exl3Scratch(shape, rows, slots, device=dev)
                self.ey = self.exl3.y.view(rows, slots, D)
            self.sgu = torch.empty((rows, 2 * sl), dtype=bf, device=dev)
            self.sact = torch.empty((rows, sl), dtype=bf, device=dev)
            self.sxs = torch.empty((rows, sl // 64), dtype=f32, device=dev)
            self.sy = torch.empty((rows, D), dtype=f32, device=dev)
        # rank partials
        self.part = torch.empty((rows, D), dtype=f32, device=dev)
        self.gath = torch.empty((w.world * rows * D,), dtype=f32, device=dev)
        self.sk = torch.empty((1 if w.meta.get("packed_exl3") or (prefill and not exl3) else 8 * rows * 16384,), dtype=f32, device=dev)
        # final
        self.hidden = torch.empty((rows, D), dtype=bf, device=dev)
        self.fnormed = torch.empty((rows, D), dtype=bf, device=dev)
        self.fxs = torch.empty((rows, D // 64), dtype=f32, device=dev)
        self.logits = torch.empty((head_rows, w.head.n), dtype=bf, device=dev)
        # MTP
        self.me = self.mcat = self.mxs = self.mx = None
        if w.mtp is not None:
            self.me = torch.empty((rows, D), dtype=bf, device=dev)
            self.mcat = torch.empty((rows, 2 * D), dtype=bf, device=dev)
            self.mxs = torch.empty((rows, 2 * D // 64), dtype=f32, device=dev)
            self.mx = torch.empty((rows, D), dtype=bf, device=dev)
        self._parents: dict[int, torch.Tensor] = {}
        # DFlash2 taps: the mean of the streams after chosen layers (``set_taps``), filled by every forward
        self.taps: list[torch.Tensor] = []
        self.tap_at: dict[int, list[int]] = {}
        self.experts = c.experts
        self.top_k = c.top_k
        self.projection_scratch = (packed.ProjectionScratch(list(packed.projections(w)), rows, dev, prefill)
                                   if w.meta.get("packed_exl3") else None)
        if self.projection_scratch is not None:
            self.projection_scratch.heads = {id(p) for p in packed.leaves(w.head)}

    def set_taps(self, layers: tuple[int, ...], hidden: int) -> None:
        self.tap_at = {}
        for i, layer in enumerate(layers):
            self.tap_at.setdefault(layer, []).append(i)
        self.taps = [torch.empty((self.rows, hidden), dtype=torch.bfloat16, device=self.ids.device) for _ in layers]

    def parents(self, R: int) -> torch.Tensor:
        p = self._parents.get(R)
        if p is None:
            p = torch.arange(-1, R - 1, dtype=torch.int32, device=self.ids.device)
            self._parents[R] = p
        return p


class State:
    """Committed caches of one sequence (and of the MTP head's attention layer)."""

    def __init__(self, w: Weights, capacity: int, rows: int) -> None:
        c = w.cfg
        dev = w.device
        HL = c.heads // w.world
        LL = c.lin_heads // w.world
        self.capacity = capacity
        self.pos = 0
        self.pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.mtp_pos_dev = torch.zeros((1,), dtype=torch.int32, device=dev)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        dsa_layers = [l for l in w.layers if l.kind == "dsa"]
        self.kda_index = {l.index: i for i, l in enumerate(kda_layers)}
        self.dsa_index = {l.index: i for i, l in enumerate(dsa_layers)}
        n = len(kda_layers)
        width = kda_layers[0].kda.proj.n if kda_layers else 0
        self.conv = torch.zeros((n, c.conv - 1, 3 * LL * 128), dtype=torch.bfloat16, device=dev)
        self.rec = torch.zeros((2, n, LL, 128, 128), dtype=torch.float32, device=dev)
        self.cur = [0] * n
        self.proj = torch.zeros((n, rows, width), dtype=torch.bfloat16, device=dev)
        self.scratch_set = kda_mod.KDAScratchSet(n, rows, LL, dev) if n else None
        self.scratch = self.scratch_set.views if n else []
        self.latent = latent.ENABLED
        if self.latent:              # one 512-wide latent a token and layer (kc), no separate values (vc)
            self.kc = [torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
            self.vc = [None for _ in dsa_layers]
        else:
            self.kc = [torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
            self.vc = [torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev) for _ in dsa_layers]
        self.mtp_len = 0
        self.mtp_drafted = 0
        if w.mtp is not None:
            if self.latent:
                self.mtp_kc = torch.zeros((capacity, c.kv_lora), dtype=torch.bfloat16, device=dev)
                self.mtp_vc = None
            else:
                self.mtp_kc = torch.zeros((capacity, HL, c.qk_dim), dtype=torch.bfloat16, device=dev)
                self.mtp_vc = torch.zeros((capacity, HL, c.v_dim), dtype=torch.bfloat16, device=dev)
        # DSA indexer caches (long contexts only): per layer (and the MTP layer, last) keys, gates, pool keys
        self.index = None
        if w.meta.get("long_context"):
            n_idx = len(dsa_layers) + (1 if w.mtp is not None else 0)
            mk = lambda n: torch.zeros((n, c.index_dim), dtype=torch.bfloat16, device=dev)   # noqa: E731
            self.index = [(mk(capacity), mk(capacity), mk(capacity // 4 + 2)) for _ in range(n_idx)]

    def reset(self) -> None:
        self.conv.zero_()
        self.rec.zero_()
        self.cur = [0] * len(self.cur)
        self.set_pos(0)
        self.set_mtp_len(0)
        self.mtp_drafted = 0

    def set_pos(self, pos: int) -> None:
        self.pos = pos
        self.pos_dev.fill_(pos)

    def set_mtp_len(self, n: int) -> None:
        self.mtp_len = n
        self.mtp_pos_dev.fill_(n)

    @property
    def parity(self) -> int:
        return self.cur[0] if self.cur else 0

    def clone(self) -> "State":
        import copy

        other = copy.copy(self)
        other.conv = self.conv.clone()
        other.rec = self.rec.clone()
        other.cur = list(self.cur)
        other.pos_dev = self.pos_dev.clone()
        other.mtp_pos_dev = self.mtp_pos_dev.clone()
        other.kc = [x.clone() for x in self.kc]
        other.vc = [x.clone() if x is not None else None for x in self.vc]
        if self.index is not None:
            other.index = [tuple(x.clone() for x in trio) for trio in self.index]
        if hasattr(self, "mtp_kc"):
            other.mtp_kc = self.mtp_kc.clone()
            other.mtp_vc = self.mtp_vc.clone() if self.mtp_vc is not None else None
        return other


# -- blocks ---------------------------------------------------------------------------------------------------
def gather(w: Weights, b: Buffers, R: int) -> torch.Tensor:
    """Every rank's fp32 partial b.part[:R] in rank order: [world, R, D] (summed rank 0 first by the consumer)."""

    d = b.part.shape[1]
    if w.comm is None:
        return b.part[:R].view(1, R, d)
    out = b.gath[:b.world * R * d]
    w.comm.all_gather(b.part[:R].reshape(-1), out)
    return out.view(b.world, R, d)


def mm(b: Buffers, x: torch.Tensor, q, xs: torch.Tensor | None, out: torch.Tensor, f32: bool = False) -> torch.Tensor:
    """A projection: 4-bit ones of a prompt chunk on the shared prefill matmul, the rest on ``qmm.matmul``."""

    if isinstance(q, (packed.PlainLinear, packed.ComposedLinear, packed.Exl3Linear, packed.Affine6)):
        with prof.timed("mm: packed"):
            return packed.matmul(q, x, out, b.projection_scratch)
    if b.prefill and isinstance(q, qmm.Q4):
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return qmm.matmul(x, q, xs, out=out, f32=f32, part=b.sk)


def out_proj(w: Weights, b: Buffers, x: torch.Tensor, q: qmm.Q4, xs: torch.Tensor, R: int) -> torch.Tensor:
    mm(b, x, q, xs, b.part[:R], f32=True)
    return gather(w, b, R)


def kda_block(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, cuts: Sequence[Cut] = ()) -> torch.Tensor:
    c = w.cfg
    k = layer.kda
    li = st.kda_index[layer.index]
    p = b.kproj[0, :R] if b.prefill else st.proj[li, :R]
    mm(b, b.normed[:R], k.proj, b.xs[:R], p)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    pre = b.prefill
    mm(b, fa, k.fb, None if pre else qmm.group_sums(fa, b.xs_fa[:R]), b.ka[:R])
    mm(b, ga, k.gb, None if pre else qmm.group_sums(ga, b.xs_ga[:R]), b.kg[:R])
    cur = st.cur[li]
    if not cuts:
        out = kda_mod.chain(p, k.b_off, b.ka[:R], b.kg[:R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                            k.dt_bias, k.norm, c.eps, c.lower, R, b.kscratch if pre else st.scratch[li],
                            st.rec[1 - cur, li])
    else:
        # the chain runs piece by piece, each from the last cut's state and conv window: every piece's rows are those
        # of the whole chunk, and each cut keeps the state a prefill ending at its row leaves
        pieces, conv, rec, a = [], st.conv[li], st.rec[cur, li], 0
        for cut in cuts:
            n = cut.point
            pieces.append(kda_mod.chain(p[a:n], k.b_off, b.ka[a:n], b.kg[a:n], conv, k.conv, rec, k.a_log,
                                        k.dt_bias, k.norm, c.eps, c.lower, n - a, b.kscratch, cut.rec[li]).clone())
            _shift_conv(cut.conv[li:li + 1], b.kproj[:, :n], n)
            conv, rec, a = cut.conv[li], cut.rec[li], n
        pieces.append(kda_mod.chain(p[a:], k.b_off, b.ka[a:R], b.kg[a:R], conv, k.conv, rec, k.a_log, k.dt_bias,
                                    k.norm, c.eps, c.lower, R - a, b.kscratch, st.rec[1 - cur, li]))
        out = torch.cat(pieces)
    if pre:                              # a prompt chunk keeps every row: the layer commits now
        st.cur[li] = 1 - cur
        _shift_conv(st.conv[li:li + 1], b.kproj[:, :R], R)
    return out_proj(w, b, out, k.o, None if pre else qmm.group_sums(out, b.kxs[:R]), R)


def dsa_block(layer: LayerW, w: Weights, kc: torch.Tensor, vc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers,
              R: int, nch: int | None, index=None, host_pos: int | None = None,
              sparse_np: int | None = None) -> torch.Tensor:
    """Write every index key and pool; rows past the dense limit attend to their top-512 pools (host_pos eager, or sparse_np in a captured graph)."""

    c = w.cfg
    a = layer.dsa
    mm(b, b.normed[:R], a.proj, b.xs[:R], b.dp[:R])
    glue.rmsnorm(b.dp[:R, :c.q_lora], a.q_norm, c.eps, b.qr[:R], b.xs_qr[:R])
    glue.rmsnorm(b.dp[:R, c.q_lora:], a.kv_norm, c.eps, b.lat[:R], b.xs_lat[:R])
    HL = a.heads
    mm(b, b.qr[:R], a.q_b, b.xs_qr[:R], b.q[:R].view(R, HL * c.qk_dim))
    if a.absorb is not None:
        return _dsa_latent(a, w, kc, pos_dev, b, R, nch, index, host_pos, sparse_np)
    if sparse_np is not None:
        raise ValueError("sparse CUDA graphs need the latent cache (TF_GLM_LATENT=1)")
    mm(b, b.lat[:R], a.kv_k, b.xs_lat[:R], b.kn[:R].view(R, HL * c.qk_dim))
    mm(b, b.lat[:R], a.kv_v, b.xs_lat[:R], b.vn[:R].view(R, HL * c.v_dim))
    kv_write(b.kn[:R], b.vn[:R], kc, vc, pos_dev)
    sparse_rows = index is not None and host_pos is not None and host_pos + R - 1 >= c.dense_limit
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
        if w.meta.get("packed_exl3"):
            mm(b, b.normed[:R], packed.PlainLinear(ix.gate), None, b.igr[:R], f32=True)
        else:
            glue.router(b.normed[:R], ix.gate, b.igr[:R])
        sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    if sparse_rows and host_pos >= c.dense_limit:     # every row sparse: the dense pass is skipped
        o = torch.empty((R, HL, c.v_dim), dtype=torch.bfloat16, device=b.q.device) if b.prefill else b.attn.out[:R]
    elif b.prefill:
        o = prefill_attention.attention(b.q[:R], kc, vc, host_pos, scale=c.qk_dim ** -0.5)
    else:
        o = attention(b.q[:R], kc, vc, pos_dev, b.attn, scale=c.qk_dim ** -0.5, nch=nch)
    if sparse_rows:
        mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
        tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                              pk.shape[0] - 2, pos_dev)
        sparse.sparse_attention(b.q[:R], kc, vc, tokens, counts, o, c.qk_dim ** -0.5)
    o = o.view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, None if b.prefill else qmm.group_sums(o, b.xs_ao[:R]), R)


def dense_attention(qa: torch.Tensor, lc: torch.Tensor, pos_dev: torch.Tensor, s: latent.LatentScratch, *,
                    scale: float, nch: int, out: torch.Tensor) -> torch.Tensor:
    """``latent.attention`` in blocks of ``part_rows`` rows, each at its first row's position: one call's bits a row."""

    R, step = qa.shape[0], s.part_rows
    hb = latent.head_block(R)
    if R <= step:
        return latent.attention(qa, lc, pos_dev, s, scale=scale, nch=nch, out=out, hb=hb)
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        at = pos_dev if r0 == 0 else pos_dev + r0          # the block's first row's position, on the device
        latent.attention(qa[r0:r1], lc, at, s, scale=scale, nch=nch, out=out[r0:r1], hb=hb)
    return out


def _dsa_latent(a, w: Weights, lc: torch.Tensor, pos_dev: torch.Tensor, b: Buffers, R: int, nch: int | None,
                index, host_pos: int | None, sparse_np: int | None = None) -> torch.Tensor:
    """DSA on the latent cache: the same indexer and selection, attention over latents with kv_b's key blocks absorbed into the query."""

    c = w.cfg
    HL = a.heads
    s = b.lat_s
    with prof.timed("dsa: latent write"):
        latent.latent_write(b.lat[:R], lc, pos_dev)
    # sparse_np: every row is past the dense limit (a captured graph); else the host position decides
    all_sparse = sparse_np is not None or (host_pos is not None and host_pos >= c.dense_limit)
    sparse_rows = index is not None and (all_sparse or (host_pos is not None and host_pos + R - 1 >= c.dense_limit))
    if index is not None:
        ik, ig, pk = index
        ix = a.index
        with prof.timed("dsa: indexer update"):
            mm(b, b.normed[:R], ix.kw, b.xs[:R], b.ikr[:R])
            if w.meta.get("packed_exl3"):
                mm(b, b.normed[:R], packed.PlainLinear(ix.gate), None, b.igr[:R], f32=True)
            else:
                glue.router(b.normed[:R], ix.gate, b.igr[:R])
            sparse.index_update(b.ikr[:R, :c.index_dim], b.igr[:R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk, pos_dev)
    with prof.timed("dsa: absorb"):
        qa = latent.absorb_q(b.q[:R], a.absorb, s.qa[:R])
    ol = s.ol[:R]
    scale = c.qk_dim ** -0.5
    if not all_sparse:
        # Rows past the dense limit are recomputed sparsely below, so the dense pass needs only the chunks up to it.
        with prof.timed("dsa: dense attention"):
            dense_attention(qa, lc, pos_dev, s, scale=scale, nch=min(nch or s.nch, s.nch), out=ol)
    if sparse_rows:
        with prof.timed("dsa: select tokens"):
            mm(b, b.qr[:R], ix.qb, b.xs_qr[:R], b.qi[:R])
            tokens, counts = sparse.select_tokens(b.qi[:R], b.ikr[:R, c.index_dim:], pk, host_pos, R,
                                                  pk.shape[0] - 2, pos_dev, bucket=sparse_np)
        with prof.timed("dsa: sparse attention"):
            latent.sparse_attention(qa, lc, tokens, counts, ol, scale)
    with prof.timed("dsa: expand"):
        o = latent.expand_v(ol, a.absorb, b.vn[:R]).view(R, HL * c.v_dim)
    return out_proj(w, b, o, a.o, qmm.group_sums(o, b.xs_ao[:R]), R)


def mlp_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    m = layer.mlp
    mm(b, b.normed[:R], m.gu, b.xs[:R], b.gu[:R])
    glue.swiglu(b.gu[:R], b.act[:R], b.xs_act[:R], w.cfg.limit)
    return out_proj(w, b, b.act[:R], m.down, b.xs_act[:R], R)


def moe_block(layer: LayerW, w: Weights, b: Buffers, R: int) -> torch.Tensor:
    c = w.cfg
    m = layer.moe
    with prof.timed("moe: route"):
        if w.meta.get("packed_exl3"):
            mm(b, b.normed[:R], packed.PlainLinear(m.router), None, b.mlog[:R], f32=True)
        else:
            glue.router(b.normed[:R], m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
        if m.shared is None:
            grouped.route(b.pick[:R], b.plan)
    route_log.note(m, b.pick, b.wts, R, c.experts, c.top_k, b.prefill)
    if SKIP_ON and m.shared is not None and w.meta.get("packed_exl3"):
        # a pick under the layer's share of its row's routed weight, or past the kept count, is not read (the kernels skip
        # id >= experts); a zero share and a full count change nothing
        t = skip_table(b.wts.device)
        wts = b.wts[:R, :c.top_k]
        total = wts.sum(1, keepdim=True)
        on = t[66 if b.prefill else 67]
        low = wts < t[layer.index] * on * total
        order = wts.argsort(1, descending=True).argsort(1).float()
        low |= (order >= t[65]) & (on > 0)
        b.pick[:R, :c.top_k].masked_fill_(low, c.experts)
        wts.masked_fill_(low, 0.0)
        # a row whose every pick is skipped keeps zero weights instead of dividing by their zero sum
        kept = wts.sum(1, keepdim=True).clamp_min(torch.finfo(wts.dtype).tiny)
        wts.mul_(torch.where(t[64] > 0, total / kept, torch.ones_like(total)))
    if m.shared is not None:
        # EXL3: the routed slots through the trellis kernels, the shared expert (last slot) through BF16 matmuls
        if w.meta.get("packed_exl3"):
            routed_micro_windows(b.normed[:R], b.pick[:R], m.experts, b.exl3, b.ey[:R], R, c.limit)
        else:
            exl3_generic.routed(b.normed[:R], b.pick, m.experts, b.exl3, R, c.limit)
        s = m.shared
        mm(b, b.normed[:R], s.gu, b.xs[:R], b.sgu[:R])
        glue.swiglu(b.sgu[:R], b.sact[:R], b.sxs[:R], c.limit)
        mm(b, b.sact[:R], s.down, b.sxs[:R], b.sy[:R], f32=True)
        b.ey[:R, c.top_k].copy_(b.sy[:R])
        glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
        return gather(w, b, R)
    with prof.timed("moe: gate/up"):
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)
    with prof.timed("moe: combine"):
        glue.combine(b.ey[:R], b.wts[:R], b.part[:R])
    with prof.timed("moe: all-gather"):
        return gather(w, b, R)


def layer_forward(layer: LayerW, w: Weights, st: State, b: Buffers, R: int, nch: int | None = None,
                  host_pos: int | None = None, sparse_np: int | None = None, cuts: Sequence[Cut] = ()) -> None:
    c = w.cfg
    x = b.x[:R]
    h = layer.attn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    if layer.kind == "kda":
        with prof.timed("kda"):
            g = kda_block(layer, w, st, b, R, cuts)
    else:
        di = st.dsa_index[layer.index]
        with prof.timed("dsa (total)"):
            g = dsa_block(layer, w, st.kc[di], st.vc[di], st.pos_dev, b, R, nch,
                          st.index[di] if st.index is not None else None, host_pos, sparse_np)
    with prof.timed("hc"):
        glue.hc_post(x, x, g, b.post[:R], b.comb[:R], fp32_residual=b.fp32_residual)
        h = layer.ffn_hc
        glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters)
    with prof.timed("moe (total)" if layer.mlp is None else "mlp"):
        g = mlp_block(layer, w, b, R) if layer.mlp is not None else moe_block(layer, w, b, R)
    glue.hc_post(x, x, g, b.post[:R], b.comb[:R], fp32_residual=b.fp32_residual)


def check_room(w: Weights, st: State, R: int, pos: int | None = None) -> None:
    pos = st.pos if pos is None else pos
    if pos + R > w.cfg.dense_limit and st.index is None:
        raise ValueError(f"context {pos + R} past {w.cfg.dense_limit} tokens: this engine was started without long "
                         "contexts (DSA's sparse top-k)")
    if pos + R > st.capacity:
        raise ValueError("context past the cache capacity")


def stage(w: Weights, st: State, b: Buffers, tokens: Sequence[int]) -> int:
    """Host work before a forward: the token ids into the static device buffer (pinned copy)."""

    R = len(tokens)
    if R > b.rows:
        raise ValueError(f"window of {R} rows, buffers hold {b.rows}")
    check_room(w, st, R)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = list(tokens)
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()
    return R


def compute(w: Weights, st: State, b: Buffers, R: int, *, logits: bool = True, nch: int | None = None,
            host_pos: int | None = None, sparse_np: int | None = None, cuts: Sequence[Cut] = ()):
    """Run capturable GPU work on static buffers and device positions; eager long contexts use host_pos (graphs sparse_np) to select sparse attention."""

    points = [cut.point for cut in cuts]
    if cuts and (not b.prefill or not 0 < points[0] or points[-1] >= R or points != sorted(set(points))):
        raise ValueError("prompt cuts must lie inside a prefill chunk, in increasing order")
    c = w.cfg
    glue.embed(b.ids[:R], w.embed, c.hidden, c.streams, b.x[:R])
    for layer in w.layers:
        layer_forward(layer, w, st, b, R, nch, host_pos, sparse_np, cuts)
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:R], b.taps[slot][:R])
    glue.stream_mean(b.x[:R], b.hidden[:R])
    if not logits:
        return None
    glue.rmsnorm(b.hidden[:R], w.norm, c.eps, b.fnormed[:R], b.fxs[:R])
    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)
        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])
    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])


def chunks_for(st: State, R: int) -> int:
    from .attention import CHUNK

    return -(-(st.pos + R) // CHUNK)


@torch.no_grad()
def forward(w: Weights, st: State, b: Buffers, tokens: Sequence[int], *, logits: bool = True) -> torch.Tensor | None:
    """Return logits and hidden buffer views for token rows, leaving committed state unchanged until commit."""

    R = stage(w, st, b, tokens)
    return compute(w, st, b, R, logits=logits, nch=chunks_for(st, R), host_pos=st.pos)


@triton.jit
def _row(CONV, PROJ, l, src, c, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr):
    old = tl.load(CONV + l * conv_layer + src * C + c, mask=(src < TAPS) & (c < C), other=0.0)
    new = tl.load(PROJ + l * proj_layer + (src - TAPS) * proj_row + c, mask=(src >= TAPS) & (c < C), other=0.0)
    return tl.where(src < TAPS, old, new)


@triton.jit
def _conv_shift(CONV, PROJ, keep, conv_layer, proj_layer, proj_row, C: tl.constexpr, TAPS: tl.constexpr,
                BLOCK: tl.constexpr):
    """Program (layer, channel block): the 3 window rows become rows keep .. keep + 2 of [old window; new rows]."""

    l = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v0 = _row(CONV, PROJ, l, keep, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v1 = _row(CONV, PROJ, l, keep + 1, c, conv_layer, proj_layer, proj_row, C, TAPS)
    v2 = _row(CONV, PROJ, l, keep + 2, c, conv_layer, proj_layer, proj_row, C, TAPS)
    tl.store(CONV + l * conv_layer + c, v0, mask=c < C)
    tl.store(CONV + l * conv_layer + C + c, v1, mask=c < C)
    tl.store(CONV + l * conv_layer + 2 * C + c, v2, mask=c < C)


def _shift_conv(conv: torch.Tensor, proj: torch.Tensor, keep: int) -> None:
    """conv [L, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [L, R, W >= C])."""

    n, taps, C = conv.shape
    if taps != 3:
        raise ValueError("the conv shift kernel is written for 4-tap convolutions")
    _conv_shift[(n, triton.cdiv(C, 1024))](conv, proj, keep, conv.stride(0), proj.stride(0), proj.stride(1), C=C,
                                           TAPS=taps, BLOCK=1024, num_warps=4)


@torch.no_grad()
def commit(w: Weights, st: State, b: Buffers, R: int, keep: int) -> None:
    """Keep the last forward's first ``keep`` rows; a prompt chunk keeps all, its KDA layers already committed."""

    if not 1 <= keep <= R or (b.prefill and keep != R):
        raise ValueError("keep must be in 1..R, and all of a prompt chunk")
    n = 0 if b.prefill else len(st.cur)
    if n:
        cur = st.cur[0]
        if keep < R:
            kda_mod.replay_layers(st.rec[cur], st.scratch_set, keep, st.rec[1 - cur])
        st.cur = [1 - cur] * n
        _shift_conv(st.conv, st.proj, keep)
    st.set_pos(st.pos + keep)
