"""Hy4-flash's own row kernels: transformers' RMSNorm rounding, the identity hyper-connections, and MLA attention over the
576-wide latent with learned sinks, the sliding window and the indexer's token lists.

Each program reads one row (or one row's head block), so a row's bits never depend on the rows beside it, and every
function has a CPU branch in fp32 torch ops that the tests drive against transformers' reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

LW, RW = 512, 64       # the latent's normed width (values too) and its rotated width: a key is 576 wide
CHUNK = 512            # keys a chunk program reads; a row's chunk partials merge in chunk order
KT = 32                # keys a tensor-core tile
HB = 16                # heads a program


# --------------------------------------------------------------------------------------------- RMSNorm ---

@triton.jit
def _rmsnorm(X, W, Y, x_stride, y_stride, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * x_stride + offs, mask=ok, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * y_stride + offs, (n * w).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """transformers' RMSNorm: bf16(w * bf16(x / rms(x))) over the last dim of bf16 rows [R, D]. ``out`` (CUDA) may
    be a row-strided view, such as the normed part of a wider row."""

    d = x.shape[-1]
    if not x.is_cuda:
        f = x.float()
        n = (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        return (w.float() * n.float()).to(torch.bfloat16)
    x2 = x.reshape(-1, d)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = out if out is not None else torch.empty((x2.shape[0], d), dtype=torch.bfloat16, device=x.device)
    if y.dim() != 2 or y.stride(-1) != 1 or y.shape != x2.shape:
        raise ValueError(f"rmsnorm out must be [{x2.shape[0]}, {d}] with unit column stride")
    _rmsnorm[(x2.shape[0],)](x2, w, y, x2.stride(0), y.stride(0), eps, D=d, BLOCK=triton.next_power_of_2(d),
                             num_warps=8)
    return y if out is not None else y.view(*x.shape[:-1], d)


# ---------------------------------------------------------------------------------------- hyper-connections ---

@triton.jit
def _hc_pre(X, FN, BASE, SCALE, POST, H, eps, hc_eps, mag, D: tl.constexpr, S: tl.constexpr, M: tl.constexpr,
            BD: tl.constexpr, HEAD: tl.constexpr):
    """Program r: the mixes of row r's S streams (fp32 dots with M rows of FN), their sigmoid gates, the collapse."""

    r = tl.program_id(0)
    m = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    ss = tl.zeros((BD,), tl.float32)
    for s in tl.static_range(S):
        for d0 in range(0, D, BD):
            d = d0 + tl.arange(0, BD)
            x = tl.load(X + (r * S + s) * D + d).to(tl.float32)
            ss += x * x
            f = tl.load(FN + m[:, None] * (S * D) + s * D + d[None, :])
            acc += tl.sum(f * x[None, :], axis=1)
    inv = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / (S * D) + eps)
    mixes = acc * inv
    base = tl.load(BASE + m)
    if HEAD:
        pre = tl.sigmoid(mixes * tl.load(SCALE) + base) + hc_eps
    else:
        sc = tl.where(m < S, tl.load(SCALE), tl.load(SCALE + 1))
        gate = tl.sigmoid(mixes * sc + base)
        pre = gate + hc_eps
        post = mag * gate + hc_eps
        tl.store(POST + r * S + (m - S), post, mask=m >= S)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        out = tl.zeros((BD,), tl.float32)
        for s in tl.static_range(S):
            ps = tl.sum(tl.where(m == s, pre, 0.0), axis=0)
            out += ps * tl.load(X + (r * S + s) * D + d).to(tl.float32)
        tl.store(H + r * D + d, out.to(tl.bfloat16))


def hc_pre(x: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, *, eps: float, hc_eps: float,
           magnitude: float = 2.0, head: bool = False) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Streams x [R, S, D] bf16 -> (post [R, S] fp32, or None for the head; the collapsed row [R, D] bf16).

    ``HYV4FlashHyperConnection``: mixes = (flat x) . fn * rsqrt(mean(x^2) + eps) in fp32, pre = sigmoid(mixes * scale
    + base) + hc_eps over the first S, post = magnitude * sigmoid(...) + hc_eps over the next S; the row is sum pre * x.
    The head (``HYV4FlashHyperHead``) has S mixes and one scale, and no post."""

    rows, s, d = x.shape
    if not x.is_cuda:
        flat = x.reshape(rows, s * d).float()
        mixes = (flat @ fn.float().t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        if head:
            pre, post = torch.sigmoid(mixes * scale.float()[0] + base.float()) + hc_eps, None
        else:
            pre = torch.sigmoid(mixes[:, :s] * scale.float()[0] + base.float()[:s]) + hc_eps
            post = magnitude * torch.sigmoid(mixes[:, s:] * scale.float()[1] + base.float()[s:]) + hc_eps
        return post, (pre[..., None] * x.float()).sum(1).to(torch.bfloat16)
    m = fn.shape[0]
    post = None if head else torch.empty((rows, s), dtype=torch.float32, device=x.device)
    h = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    _hc_pre[(rows,)](x.contiguous(), fn, base, scale, post if post is not None else h, h, eps, hc_eps, magnitude,
                     D=d, S=s, M=m, BD=256, HEAD=head, num_warps=8)
    return post, h


@triton.jit
def _hc_post(X, POST, Y, D: tl.constexpr, S: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    p = tl.load(POST + r * S + s)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        y = tl.load(Y + r * D + d).to(tl.float32)
        x = tl.load(X + (r * S + s) * D + d).to(tl.float32)
        tl.store(X + (r * S + s) * D + d, (p * y + x).to(tl.bfloat16))


def hc_post(x: torch.Tensor, post: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x [R, S, D] bf16 <- bf16(post[..., None] * y + x) in fp32 (the decoder layer's write-back), in place."""

    if not x.is_cuda:
        x.copy_((post[..., None].float() * y.float()[:, None, :] + x.float()).to(torch.bfloat16))
        return x
    rows, s, d = x.shape
    bd = min(1024, triton.next_power_of_2(d))
    if d % bd:
        raise ValueError(f"hc_post: width {d} is not a multiple of {bd}")
    _hc_post[(rows, s)](x, post, y.contiguous(), D=d, S=s, BD=bd, num_warps=4)
    return x


# ------------------------------------------------------------------------------------------------- head ---

@triton.jit
def _head(X, W, OUT, R, V, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Program (row tile, column block): OUT[r, n] = sum_k X[r, k] W[n, k] in fp32, K in BK steps in order."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * D + (k0 + rk)[None, :], mask=rm[:, None] < R, other=0.0)
        w = tl.load(W + rn[:, None] * D + (k0 + rk)[None, :], mask=rn[:, None] < V, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rm[:, None] * V + rn[None, :], acc, mask=(rm[:, None] < R) & (rn[None, :] < V))


def head(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 logits [R, V] of bf16 rows x [R, D] times the bf16 head w [V, D]: one fixed 16-row tile and K order, so a
    row's logits are the same bits alone and in any window (``enable_lm_head_fp32``: the products are exact)."""

    rows, d = x.shape
    v = w.shape[0]
    if not x.is_cuda:
        return x.float() @ w.float().t()
    x = x.contiguous()
    out = torch.empty((rows, v), dtype=torch.float32, device=x.device)
    _head[(triton.cdiv(rows, 16), triton.cdiv(v, 64))](x, w, out, rows, v, D=d, BM=16, BN=64, BK=128, num_warps=4,
                                                     num_stages=4)
    return out


@triton.jit
def _e2m1(c):
    """e2m1 codes (int32, sign in bit 3) -> fp32 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives."""

    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * m))
    return tl.where((c & 8) != 0, -mag, mag)


@triton.jit
def _head4(X, C, S, OUT, R, V, scale2, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """``_head`` over an NVFP4 head: codes C [V, D/2] (low nibble first) and e4m3 scales S [V, D/16]. Each weight is
    code x block scale, exact in bf16, so the bf16 mma sums the same products as a dense head would; the global
    scale multiplies the fp32 sum once."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rh = tl.arange(0, BK // 2)
    rs = tl.arange(0, BK // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * D + (k0 + rk)[None, :], mask=rm[:, None] < R, other=0.0)
        c = tl.load(C + rn[:, None] * (D // 2) + (k0 // 2 + rh)[None, :], mask=rn[:, None] < V, other=0).to(tl.int32)
        w = tl.reshape(tl.join(_e2m1(c & 15), _e2m1(c >> 4)), (BN, BK))
        s = tl.load(S + rn[:, None] * (D // 16) + (k0 // 16 + rs)[None, :], mask=rn[:, None] < V, other=0)
        s = s.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        s = tl.reshape(tl.broadcast_to(s[:, :, None], (BN, BK // 16, 16)), (BN, BK))
        acc = tl.dot(x, tl.trans((w * s).to(tl.bfloat16)), acc)
    tl.store(OUT + rm[:, None] * V + rn[None, :], acc * scale2, mask=(rm[:, None] < R) & (rn[None, :] < V))


class Head4:
    """An NVFP4 head (codes uint8 [V, D/2], e4m3 scales [V, D/16], an fp32 global scale): fp32 logits like ``head``,
    with the same fixed 16-row tile and K order, so a row's logits are the same bits alone and in any window."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, scale2: float) -> None:
        self.codes = codes.contiguous().view(torch.uint8)
        self.scales = scales.contiguous().view(torch.uint8)
        self.scale2 = float(scale2)
        self.shape = (self.codes.shape[0], self.codes.shape[1] * 2)

    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel()

    def dense(self) -> torch.Tensor:
        """The head's values in fp32 [V, D] (the CPU path, and tests)."""

        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
                           device=self.codes.device)
        c = self.codes.to(torch.int64)
        v = lut[torch.stack([c & 15, c >> 4], -1).flatten(-2)]
        s = self.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        return v * s * self.scale2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows, d = x.shape
        v = self.shape[0]
        if not x.is_cuda:
            return (x.float() @ (self.dense() / self.scale2).to(torch.bfloat16).float().t()) * self.scale2
        x = x.contiguous()
        out = torch.empty((rows, v), dtype=torch.float32, device=x.device)
        _head4[(triton.cdiv(rows, 16), triton.cdiv(v, 64))](x, self.codes, self.scales, out, rows, v, self.scale2,
                                                            D=d, BM=16, BN=64, BK=128, num_warps=4, num_stages=4)
        return out


# ----------------------------------------------------------------------------------- MLA absorb and expand ---

ROW_KERNEL = 16          # calls of up to this many rows take the per-row kernels; prompt chunks batched matmuls


@triton.jit
def _absorb(Q, WK, SK, QA, R, q_stride, qa_stride, H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr,
            LW: tl.constexpr, BN: tl.constexpr, F8: tl.constexpr):
    """Program (head, latent block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n], one fp32 sum a row. ``F8``: WK holds
    e4m3 codes and SK [H, D] each row's fp32 scale."""

    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, DP)
    ok = k < D
    if F8:
        w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :], mask=ok[:, None], other=0)
        w = w.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(SK + h * D + k, mask=ok, other=0.0)[:, None]
    else:
        w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    for r in range(R):
        q = tl.load(Q + r * q_stride + h * D + k, mask=ok, other=0.0).to(tl.float32)
        tl.store(QA + (r * H + h) * qa_stride + n, tl.sum(q[:, None] * w, axis=0).to(tl.bfloat16))


@triton.jit
def _expand(OL, WV, SV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr,
            F8: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k], one fp32 sum a row. ``F8``: WV holds
    e4m3 codes and SV [H, DV] each output row's fp32 scale, which multiplies the sum."""

    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, LW)
    if F8:
        w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sv = tl.load(SV + h * DV + n)
    else:
        w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)
    for r in range(R):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        y = tl.sum(w * o[None, :], axis=1)
        if F8:
            y = y * sv
        tl.store(OUT + (r * H + h) * DV + n, y.to(tl.bfloat16))


class RowF8:
    """kv_b's key or value rows in e4m3 with an fp32 scale a row: codes uint8 [H, rows, LW], scales [H, rows]."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor) -> None:
        self.codes, self.scales = codes.contiguous().view(torch.uint8), scales.float().contiguous()
        self.shape = self.codes.shape

    def dense(self) -> torch.Tensor:
        return self.codes.view(torch.float8_e4m3fn).float() * self.scales[..., None]

    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel() * 4


def absorb(q: torch.Tensor, wk, out: torch.Tensor) -> torch.Tensor:
    """q [R, H, D] (rows may be strided) times kv_b's key rows wk [H, D, LW] (bf16 or ``RowF8``) -> out [R, H, LW];
    on the row kernel ``out`` may be the first LW columns of wider (row, head) rows, such as the absorbed query's."""

    rows, heads, d = q.shape
    lw = wk.shape[2]
    f8 = isinstance(wk, RowF8)
    if not q.is_cuda or rows > ROW_KERNEL:
        if f8:                          # fp32 products of the exact weights
            out.copy_(torch.bmm(q.transpose(0, 1).float(), wk.dense()).transpose(0, 1))
        elif not q.is_cuda:
            out.copy_(torch.einsum("rhd,hdn->rhn", q.float(), wk.float()).to(torch.bfloat16))
        else:
            out.copy_(torch.bmm(q.transpose(0, 1), wk).transpose(0, 1))
        return out
    if q.stride(2) != 1 or q.stride(1) != d:
        q = q.contiguous()
    step = out.stride(1)
    if out.stride(2) != 1 or (rows > 1 and out.stride(0) != heads * step):
        raise ValueError("absorb out must hold evenly strided (row, head) rows")
    _absorb[(heads, lw // 64)](q, wk.codes if f8 else wk, wk.scales if f8 else wk, out, rows, q.stride(0), step,
                               H=heads, D=d, DP=triton.next_power_of_2(d), LW=lw, BN=64, F8=f8, num_warps=4)
    return out


def expand(o_lat: torch.Tensor, wv, out: torch.Tensor) -> torch.Tensor:
    """o_lat [R, H, LW] times kv_b's value rows wv [H, DV, LW] (bf16 or ``RowF8``) -> out [R, H, DV] bf16."""

    rows, heads, lw = o_lat.shape
    dv = wv.shape[1]
    f8 = isinstance(wv, RowF8)
    if not o_lat.is_cuda or rows > ROW_KERNEL:
        if f8:
            out.copy_(torch.bmm(o_lat.transpose(0, 1).float(), wv.dense().transpose(1, 2)).transpose(0, 1))
        elif not o_lat.is_cuda:
            out.copy_(torch.einsum("rhn,hdn->rhd", o_lat.float(), wv.float()).to(torch.bfloat16))
        else:
            out.copy_(torch.bmm(o_lat.transpose(0, 1), wv.transpose(1, 2)).transpose(0, 1))
        return out
    _expand[(heads, dv // 16)](o_lat, wv.codes if f8 else wv, wv.scales if f8 else wv, out, rows, H=heads, DV=dv,
                               LW=lw, BN=16, F8=f8, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------- MLA attention ---

@triton.jit
def _chunks(QA, LC, LO, CNT, SEL, TOK, PO, PM, PL, R, W: tl.constexpr, H: tl.constexpr, scale,
            HBT: tl.constexpr, KTT: tl.constexpr, CH: tl.constexpr):
    """Program (row, head block, chunk): keys j in [c CH, (c + 1) CH) of row r's list, token(j) = TOK[r, j] when
    SEL[r] else LO[r] + j, valid for j < CNT[r]; online-softmax partials of the latent values."""

    r = tl.program_id(0)
    hb = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(CNT + r)
    lo = tl.load(LO + r)
    sel = tl.load(SEL + r)
    hh = hb * HBT + tl.arange(0, HBT)
    hok = hh < H
    kl = tl.arange(0, 512)
    kr = tl.arange(0, 64)
    m = tl.full((HBT,), float("-inf"), tl.float32)
    l = tl.zeros((HBT,), tl.float32)
    o = tl.zeros((HBT, 512), tl.float32)
    if c * CH < n:
        q = tl.load(QA + (r * H + hh[:, None]) * 576 + kl[None, :], mask=hok[:, None], other=0.0)
        qr = tl.load(QA + (r * H + hh[:, None]) * 576 + 512 + kr[None, :], mask=hok[:, None], other=0.0)
        for t in range(CH // KTT):
            j = c * CH + t * KTT + tl.arange(0, KTT)
            ok = j < n
            listed = tl.load(TOK + r * W + j, mask=ok & (sel != 0), other=0)
            tok = tl.where(sel != 0, listed, lo + j).to(tl.int64)
            kv = tl.load(LC + tok[:, None] * 576 + kl[None, :], mask=ok[:, None], other=0.0)
            kp = tl.load(LC + tok[:, None] * 576 + 512 + kr[None, :], mask=ok[:, None], other=0.0)
            s = (tl.dot(q, tl.trans(kv)) + tl.dot(qr, tl.trans(kp))) * scale
            s = tl.where(ok[None, :], s, float("-inf"))
            tm = tl.max(s, 1)
            act = tm != float("-inf")
            nm = tl.where(act, tl.maximum(m, tm), m)
            alpha = tl.where(act, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)), 1.0)
            p = tl.where(ok[None, :] & act[:, None], tl.exp(s - nm[:, None]), 0.0)
            o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
            l = l * alpha + tl.sum(p, 1)
            m = nm
    base = (c * R + r) * H + hh
    tl.store(PO + base[:, None] * 512 + kl[None, :], o, mask=hok[:, None])
    tl.store(PM + base, m, mask=hok)
    tl.store(PL + base, l, mask=hok)


@triton.jit
def _merge(PO, PM, PL, SINK, OUT, CNT, R, H: tl.constexpr, NCH: tl.constexpr, CH: tl.constexpr):
    """Program (row, head): the chunk partials in order, then the head's sink as one more logit with a zero value."""

    r = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.load(CNT + r)
    k = tl.arange(0, 512)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((512,), tl.float32)
    for c in range(NCH):
        if c * CH < n:
            base = (c * R + r) * H + h
            cm = tl.load(PM + base)
            cl = tl.load(PL + base)
            act = cl > 0.0
            nm = tl.where(act, tl.maximum(m, cm), m)
            a = tl.where(act, tl.where(m == float("-inf"), 0.0, tl.exp(m - nm)), 1.0)
            b = tl.where(act, tl.exp(cm - nm), 0.0)
            o = o * a + tl.load(PO + base * 512 + k) * b
            l = l * a + cl * b
            m = nm
    sink = tl.load(SINK + h)
    nm = tl.maximum(m, sink)
    a = tl.where(m == float("-inf"), 0.0, tl.exp(m - nm))
    tl.store(OUT + (r * H + h) * 512 + k, (o * a / (l * a + tl.exp(sink - nm))).to(tl.bfloat16))


class Scratch:
    """Chunk partials for up to ``rows`` rows of ``heads`` heads over ``chunks`` chunks."""

    def __init__(self, rows: int, heads: int, chunks: int, device) -> None:
        self.rows, self.heads, self.chunks = rows, heads, chunks
        n = chunks * rows * heads
        self.po = torch.empty((n * LW,), dtype=torch.float32, device=device)
        self.pm = torch.empty((n,), dtype=torch.float32, device=device)
        self.pl = torch.empty((n,), dtype=torch.float32, device=device)


def attention(qa: torch.Tensor, cache: torch.Tensor, lo: torch.Tensor, cnt: torch.Tensor, sel: torch.Tensor,
              tok: torch.Tensor, sinks: torch.Tensor, scale: float, out: torch.Tensor,
              scratch: Scratch | None = None) -> torch.Tensor:
    """qa [R, H, 576] bf16 over the latent cache [N, 576]: row r reads keys token(j) for j < cnt[r] (``tok[r]`` when
    ``sel[r]``, else lo[r] + j), values the keys' first 512 columns, and the head's sink -> out [R, H, 512] bf16.

    Every key a row lists must be at or before its own position: the caller builds the lists causal."""

    rows, heads, _ = qa.shape
    width = tok.shape[1]
    if not qa.is_cuda:
        return _attention_cpu(qa, cache, lo, cnt, sel, tok, sinks, scale, out)
    nch = triton.cdiv(width, CHUNK)
    s = scratch
    if s is None or s.rows < rows or s.heads < heads or s.chunks < nch:
        s = Scratch(rows, heads, nch, qa.device)
    _chunks[(rows, triton.cdiv(heads, HB), nch)](qa, cache, lo, cnt, sel, tok, s.po, s.pm, s.pl, rows, W=width,
                                                 H=heads, scale=scale, HBT=HB, KTT=KT, CH=CHUNK, num_warps=8,
                                                 num_stages=1)
    _merge[(rows, heads)](s.po, s.pm, s.pl, sinks, out, cnt, rows, H=heads, NCH=nch, CH=CHUNK, num_warps=4)
    return out


def _attention_cpu(qa, cache, lo, cnt, sel, tok, sinks, scale, out):
    for r in range(qa.shape[0]):
        n = int(cnt[r])
        ids = tok[r, :n].long() if int(sel[r]) else torch.arange(int(lo[r]), int(lo[r]) + n)
        keys = cache[ids].float()
        s = qa[r].float() @ keys.t() * scale                                       # [H, n]
        full = torch.cat([s, sinks.float()[:, None]], 1)
        p = (full - full.max(-1, keepdim=True).values).softmax(-1)[:, :n]
        out[r] = (p @ keys[:, :LW]).to(torch.bfloat16)
    return out


# ------------------------------------------------------------------- row-invariant reductions (verify windows) ---
# torch picks its reduction strategy by the tensor's shape: a 16-row window's Engram residual differed from a one-row
# step's by 3.8e-6 (window 6, 21 Nov), enough to flip a near tie. Every reduction a decode row needs runs here, one
# program a row, in one fixed order.

@triton.jit
def _rms(X, W, Y, x_stride, y_stride, eps, D: tl.constexpr, BLOCK: tl.constexpr, HAS_W: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + r * x_stride + offs, mask=ok, other=0.0).to(tl.float32)
    y = x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps))
    if HAS_W:
        y = y * tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + r * y_stride + offs, y, mask=ok)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """fp32 x / rms(x) (times w) over the last dim of rows [..., D], each row in one fixed order; fp32 out."""

    shape, d = x.shape, x.shape[-1]
    if not x.is_cuda:
        f = x.float()
        n = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
        return n if w is None else n * w.float()
    x2 = x.reshape(-1, d)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if w is not None and w.dim() == 2:                       # a weight a row group: [G, D] over rows [..., G, D]
        y = torch.empty((x2.shape[0], d), dtype=torch.float32, device=x.device)
        g = w.shape[0]
        for j in range(g):                                    # G is small (16 Engram heads)
            _rms[(x2.shape[0] // g,)](x2[j::g], w[j], y[j::g], x2.stride(0) * g, d * g, eps, D=d,
                                      BLOCK=triton.next_power_of_2(d), HAS_W=True, num_warps=4)
        return y.view(*shape[:-1], d)
    y = torch.empty((x2.shape[0], d), dtype=torch.float32, device=x.device)
    _rms[(x2.shape[0],)](x2, w if w is not None else x2, y, x2.stride(0), d, eps, D=d,
                         BLOCK=triton.next_power_of_2(d), HAS_W=w is not None, num_warps=8 if d > 1024 else 4)
    return y.view(*shape[:-1], d)


@triton.jit
def _dot(A, B, Y, a_stride, b_stride, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    a = tl.load(A + r * a_stride + offs, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(B + r * b_stride + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + r, tl.sum(a * b, axis=0))


def dot_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 sum(a * b) over the last dim of rows [R, D], each row in one fixed order."""

    if not a.is_cuda:
        return (a.float() * b.float()).sum(-1)
    a, b = (t if t.stride(-1) == 1 else t.contiguous() for t in (a, b))
    y = torch.empty(a.shape[0], dtype=torch.float32, device=a.device)
    d = a.shape[-1]
    _dot[(a.shape[0],)](a, b, y, a.stride(0), b.stride(0), D=d, BLOCK=triton.next_power_of_2(d), num_warps=8)
    return y


@triton.jit
def _norm_weights(W, Y, scale, K: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    w = tl.load(W + r * K + k, mask=k < K, other=0.0)
    tl.store(Y + r * K + k, w / (tl.sum(w, axis=0) + 1e-20) * scale, mask=k < K)


def norm_weights(w: torch.Tensor, scale: float) -> torch.Tensor:
    """Routing weights [R, K] fp32 over their row's sum (+1e-20), times ``scale``."""

    if not w.is_cuda:
        return w / (w.sum(-1, keepdim=True) + 1e-20) * scale
    w = w.contiguous()
    y = torch.empty_like(w)
    _norm_weights[(w.shape[0],)](w, y, scale, K=w.shape[1], BLOCK=triton.next_power_of_2(w.shape[1]), num_warps=1)
    return y


@triton.jit
def _combine(Yp, W, OUT, K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    acc = tl.zeros((BD,), tl.float32)
    for k in range(K):                                       # slots in order
        acc += tl.load(Yp + (r * K + k) * D + d) * tl.load(W + r * K + k)
    tl.store(OUT + r * D + d, acc.to(tl.bfloat16))


def combine(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Experts' outputs y [R, K, D] fp32 weighted by w [R, K], summed over K in slot order -> bf16 [R, D]."""

    rows, k, d = y.shape
    if not y.is_cuda:
        acc = y[:, 0] * w[:, :1]
        for j in range(1, k):
            acc = acc + y[:, j] * w[:, j:j + 1]
        return acc.to(torch.bfloat16)
    bd = min(256, triton.next_power_of_2(d))
    if d % bd:
        raise ValueError(f"combine: width {d} is not a multiple of {bd}")
    out = torch.empty((rows, d), dtype=torch.bfloat16, device=y.device)
    _combine[(rows, d // bd)](y.contiguous(), w.contiguous(), out, K=k, D=d, BD=bd, num_warps=4)
    return out


@triton.jit
def _pool(TK, GIDS, PV, PG, APE, GN, TMP, OUT, eps, D: tl.constexpr, P: tl.constexpr, BN: tl.constexpr):
    """Program g: group GIDS[g]'s P token keys pooled by softmax gates over its slots, BN channels at a time (the slot
    softmax is a channel's own), then the key norm over all D (fp32)."""

    g = tl.program_id(0)
    gid = tl.load(GIDS + g).to(tl.int64)
    rows = tl.arange(0, 16)
    k = tl.arange(0, D)
    ok = rows < P
    m = tl.load(TK + (gid * P + rows[:, None]) * D + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    for b in tl.static_range(D // BN):
        n = b * BN + tl.arange(0, BN)
        vals = tl.dot(m, tl.trans(tl.load(PV + n[:, None] * D + k[None, :])), input_precision="ieee")
        gates = tl.dot(m, tl.trans(tl.load(PG + n[:, None] * D + k[None, :])), input_precision="ieee")
        gates += tl.load(APE + rows[:, None] * D + n[None, :], mask=ok[:, None], other=0.0)
        gates = tl.where(ok[:, None], gates, float("-inf"))
        e = tl.exp(gates - tl.max(gates, 0)[None, :])
        tl.store(TMP + g * D + n, tl.sum(vals * (e / tl.sum(e, 0)[None, :]), 0))
    tl.debug_barrier()
    pooled = tl.load(TMP + g * D + k)
    out = pooled * (1.0 / tl.sqrt(tl.sum(pooled * pooled, 0) / D + eps)) * tl.load(GN + k)
    tl.store(OUT + g * D + k, out.to(tl.bfloat16))


def pool(tkeys: torch.Tensor, gids: torch.Tensor, pv: torch.Tensor, pg: torch.Tensor, ape: torch.Tensor,
         g_norm: torch.Tensor, eps: float) -> torch.Tensor:
    """The compressed indexer's group keys for groups ``gids`` [G] from token keys [N, D] -> bf16 [G, D]."""

    pool_n, d = ape.shape
    if not tkeys.is_cuda:
        members = tkeys[(gids.long()[:, None] * pool_n + torch.arange(pool_n)).view(-1)].view(-1, pool_n, d).float()
        vals = members @ pv.t()
        gates = members @ pg.t() + ape
        pooled = (vals * gates.softmax(dim=1)).sum(1)
        return (g_norm * (pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + eps))).to(torch.bfloat16)
    out = torch.empty((gids.shape[0], d), dtype=torch.bfloat16, device=tkeys.device)
    tmp = torch.empty((gids.shape[0], d), dtype=torch.float32, device=tkeys.device)
    _pool[(gids.shape[0],)](tkeys, gids, pv, pg, ape, g_norm, tmp, out, eps, D=d, P=pool_n, BN=64, num_warps=4)
    return out


@triton.jit
def _scores(QI, HW, GK, CLOSED, OUT, G, H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BG: tl.constexpr):
    """Program (row, group block): sum over heads of w_h relu(q_h . k_g) for groups below the row's closed count."""

    r = tl.program_id(0)
    g = tl.program_id(1) * BG + tl.arange(0, BG)
    h = tl.arange(0, HP)
    k = tl.arange(0, D)
    closed = tl.load(CLOSED + r)
    q = tl.load(QI + (r * H + h[:, None]) * D + k[None, :], mask=(h < H)[:, None], other=0.0)
    keys = tl.load(GK + g[:, None].to(tl.int64) * D + k[None, :], mask=(g < closed)[:, None], other=0.0)
    s = tl.maximum(tl.dot(q, tl.trans(keys)), 0.0)                              # [H, BG] fp32
    w = tl.load(HW + r * H + h, mask=h < H, other=0.0)
    logit = tl.sum(s * w[:, None], 0)
    tl.store(OUT + r * G + g, tl.where(g < closed, logit, float("-inf")), mask=g < G)


def group_scores(qi: torch.Tensor, hw: torch.Tensor, gkeys: torch.Tensor, closed: torch.Tensor,
                 groups: int) -> torch.Tensor:
    """logits [R, groups] fp32 of queries qi [R, H, D] bf16 (head weights hw [R, H]) against group keys [>= groups, D];
    -inf from each row's closed count on. Each (row, group) in one fixed order, whatever R and groups are."""

    rows, heads, d = qi.shape
    if not qi.is_cuda:
        keys = gkeys[:groups].float()
        logits = torch.einsum("rh,rhg->rg", hw, torch.relu(torch.einsum("rhd,gd->rhg", qi.float(), keys)))
        return logits.masked_fill(torch.arange(groups)[None, :] >= closed[:, None], float("-inf"))
    out = torch.empty((rows, groups), dtype=torch.float32, device=qi.device)
    _scores[(rows, triton.cdiv(groups, 64))](qi.contiguous(), hw.contiguous(), gkeys, closed.to(torch.int32), out,
                                             groups, H=heads, HP=max(16, triton.next_power_of_2(heads)), D=d,
                                             BG=64, num_warps=4)
    return out


# ------------------------------------------------------------------------------------------- small fusions ---
# Each replaces a chain of torch launches with one program a row block and keeps its bits: libdevice's cosf, sinf and
# expf are the functions torch's kernels call, and every product and sum is rounded on its own (``mul_rn``,
# ``add_rn``), as torch's separate launches round them, so no FMA contraction changes a value.

@triton.jit
def _rope(X, POS, INV, OUT, N, x_stride, o_stride, H, RW: tl.constexpr, BR: tl.constexpr):
    """Program (row block): rows n of X [N, RW] (stride x_stride) at position POS[n // H], pairs (x[2i], x[2i+1])
    turned by bf16(cos), bf16(sin) of pos * INV[i]."""

    n = tl.program_id(0) * BR + tl.arange(0, BR)
    ok = n < N
    i = tl.arange(0, RW // 2)
    p = tl.load(POS + n // H, mask=ok, other=0).to(tl.float32)
    ang = libdevice.mul_rn(p[:, None], tl.load(INV + i)[None, :])
    c = libdevice.cos(ang).to(tl.bfloat16).to(tl.float32)
    s = libdevice.sin(ang).to(tl.bfloat16).to(tl.float32)
    a = tl.load(X + n[:, None] * x_stride + 2 * i[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    b = tl.load(X + n[:, None] * x_stride + 2 * i[None, :] + 1, mask=ok[:, None], other=0.0).to(tl.float32)
    o0 = libdevice.sub_rn(libdevice.mul_rn(a, c), libdevice.mul_rn(b, s))
    o1 = libdevice.add_rn(libdevice.mul_rn(b, c), libdevice.mul_rn(a, s))
    tl.store(OUT + n[:, None] * o_stride + 2 * i[None, :], o0.to(tl.bfloat16), mask=ok[:, None])
    tl.store(OUT + n[:, None] * o_stride + 2 * i[None, :] + 1, o1.to(tl.bfloat16), mask=ok[:, None])


def _rows(t: torch.Tensor) -> int | None:
    """The stride between t's RW-wide rows when every leading index steps evenly (slices of wider rows), else None."""

    step = t.stride(-2) if t.dim() > 1 else t.shape[-1]
    for i in range(t.dim() - 2, 0, -1):
        if t.shape[i] > 1 and t.stride(i - 1) != t.shape[i] * t.stride(i):
            return None
        if t.shape[i] == 1 and t.dim() > 2 and t.stride(i - 1) != step:
            return None
    return step if t.stride(-1) == 1 else None


def rope(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``Engine._rope`` in one launch: x [R, ..., RW] bf16 (rows of RW evenly strided, as slices of a wider row are),
    pos [R] int, inv fp32 [RW / 2] -> a new contiguous bf16 tensor of x's shape, or ``out`` (x's shape, its rows
    evenly strided: the rotated part of a wider row) written in place."""

    rw = x.shape[-1]
    heads = x[0].numel() // rw if x.shape[0] else 1
    flat = x.reshape(-1, rw)                     # a view when the rows are evenly strided, else a copy
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    if out is None:
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    step = _rows(out)
    if out.shape != x.shape or out.dtype != torch.bfloat16 or step is None:
        raise ValueError("rope out must have x's shape and evenly strided bf16 rows")
    n = flat.shape[0]
    br = 16
    _rope[(triton.cdiv(n, br),)](flat, pos, inv, out, n, flat.stride(0), step, heads, RW=rw, BR=br, num_warps=4)
    return out


@triton.jit
def _silu_mul_k(G, U, OUT, N, BLOCK: tl.constexpr):
    """bf16(bf16(silu(g)) * u): torch's silu is x / (1 + expf(-x)) in fp32."""

    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = o < N
    g = tl.load(G + o, mask=ok, other=0.0).to(tl.float32)
    u = tl.load(U + o, mask=ok, other=0.0).to(tl.float32)
    act = libdevice.div_rn(g, libdevice.add_rn(1.0, libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + o, libdevice.mul_rn(act, u).to(tl.bfloat16), mask=ok)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``model._silu_mul`` in one launch (contiguous bf16 g and u of one shape)."""

    g, u = g.contiguous(), u.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    _silu_mul_k[(triton.cdiv(n, 1024),)](g, u, out, n, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _gate_mul_k(V, G, OUT, N, BLOCK: tl.constexpr):
    """bf16(v * bf16(sigmoid(gate))): torch's sigmoid is 1 / (1 + expf(-x)) in fp32."""

    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = o < N
    g = tl.load(G + o, mask=ok, other=0.0).to(tl.float32)
    v = tl.load(V + o, mask=ok, other=0.0).to(tl.float32)
    sig = libdevice.div_rn(1.0, libdevice.add_rn(1.0, libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + o, libdevice.mul_rn(v, sig).to(tl.bfloat16), mask=ok)


def gate_mul(v: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gated MLA's output: v [R, ...] times bf16(sigmoid(gate [R, n])) -> [R, n] bf16, one launch."""

    v, gate = v.reshape(gate.shape).contiguous(), gate.contiguous()
    out = torch.empty_like(gate, dtype=torch.bfloat16)
    n = gate.numel()
    _gate_mul_k[(triton.cdiv(n, 1024),)](v, gate, out, n, BLOCK=1024, num_warps=4)
    return out


@triton.jit
def _add_rmsnorm(A, B, W, Y, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    """``_rmsnorm`` of bf16(a + b) in fp32: the MoE's routed plus shared sum and its combine norm."""

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    a = tl.load(A + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    b = tl.load(B + row * D + offs, mask=ok, other=0.0).to(tl.float32)
    x = libdevice.add_rn(a, b).to(tl.bfloat16).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, (n * w).to(tl.bfloat16), mask=ok)


def add_rmsnorm(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """rmsnorm(bf16(a + b), w) for bf16 rows [R, D], one launch."""

    a, b = a.contiguous(), b.contiguous()
    d = a.shape[-1]
    y = torch.empty_like(a)
    _add_rmsnorm[(a.shape[0],)](a, b, w, y, eps, D=d, BLOCK=triton.next_power_of_2(d), num_warps=8)
    return y


@triton.jit
def _hc_boundary(X, PP, YP, FN, BASE, SCALE, POST, H, NW, OUT, eps, hc_eps, mag, rms_eps, D: tl.constexpr,
                 S: tl.constexpr, M: tl.constexpr, BD: tl.constexpr, BLOCK: tl.constexpr, WRITE: tl.constexpr):
    """Program r: a sublayer boundary in one launch. ``WRITE``: the previous sublayer's write-back (``_hc_post``'s
    expression) into the streams; then ``_hc_pre``'s mixes, gates and collapse into H; then ``_rmsnorm`` of that row
    into OUT. Each phase is its kernel's code and reads what the last wrote through global memory after a barrier."""

    r = tl.program_id(0)
    if WRITE:
        for s in tl.static_range(S):
            p = tl.load(PP + r * S + s)
            for d0 in range(0, D, BD):
                d = d0 + tl.arange(0, BD)
                y = tl.load(YP + r * D + d).to(tl.float32)
                x = tl.load(X + (r * S + s) * D + d).to(tl.float32)
                tl.store(X + (r * S + s) * D + d, (p * y + x).to(tl.bfloat16))
        tl.debug_barrier()
    m = tl.arange(0, M)
    acc = tl.zeros((M,), tl.float32)
    ss = tl.zeros((BD,), tl.float32)
    for s in tl.static_range(S):
        for d0 in range(0, D, BD):
            d = d0 + tl.arange(0, BD)
            x = tl.load(X + (r * S + s) * D + d).to(tl.float32)
            ss += x * x
            f = tl.load(FN + m[:, None] * (S * D) + s * D + d[None, :])
            acc += tl.sum(f * x[None, :], axis=1)
    inv = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / (S * D) + eps)
    mixes = acc * inv
    base = tl.load(BASE + m)
    sc = tl.where(m < S, tl.load(SCALE), tl.load(SCALE + 1))
    gate = tl.sigmoid(mixes * sc + base)
    pre = gate + hc_eps
    post = mag * gate + hc_eps
    tl.store(POST + r * S + (m - S), post, mask=m >= S)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        out = tl.zeros((BD,), tl.float32)
        for s in tl.static_range(S):
            ps = tl.sum(tl.where(m == s, pre, 0.0), axis=0)
            out += ps * tl.load(X + (r * S + s) * D + d).to(tl.float32)
        tl.store(H + r * D + d, out.to(tl.bfloat16))
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(H + r * D + offs, mask=ok, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + rms_eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(NW + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + offs, (n * w).to(tl.bfloat16), mask=ok)


def hc_boundary(x: torch.Tensor, pending: tuple[torch.Tensor, torch.Tensor] | None, fn: torch.Tensor,
                base: torch.Tensor, scale: torch.Tensor, norm: torch.Tensor, *, eps: float, hc_eps: float,
                magnitude: float, rms_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_post(x, *pending)`` (when pending), ``hc_pre`` and ``rmsnorm`` of its row in one launch, the same bits:
    streams x [R, S, D] bf16, updated in place -> (post [R, S] fp32, the normed row [R, D] bf16)."""

    rows, s, d = x.shape
    m = fn.shape[0]
    post = torch.empty((rows, s), dtype=torch.float32, device=x.device)
    h = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((rows, d), dtype=torch.bfloat16, device=x.device)
    pp, yp = pending if pending is not None else (post, h)
    _hc_boundary[(rows,)](x, pp, yp.contiguous(), fn, base, scale, post, h, norm, out, eps, hc_eps, magnitude,
                          rms_eps, D=d, S=s, M=m, BD=256, BLOCK=triton.next_power_of_2(d), WRITE=pending is not None,
                          num_warps=8)
    return post, out
