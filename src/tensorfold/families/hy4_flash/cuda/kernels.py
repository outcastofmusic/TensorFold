"""Hy4-flash's own row kernels: transformers' RMSNorm rounding, the identity hyper-connections, and MLA attention over the
576-wide latent with learned sinks, the sliding window and the indexer's token lists.

Each program reads one row (or one row's head block), so a row's bits never depend on the rows beside it, and every
function has a CPU branch in fp32 torch ops that the tests drive against transformers' reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LW, RW = 512, 64       # the latent's normed width (values too) and its rotated width: a key is 576 wide
CHUNK = 512            # keys a chunk program reads; a row's chunk partials merge in chunk order
KT = 32                # keys a tensor-core tile
HB = 16                # heads a program


# --------------------------------------------------------------------------------------------- RMSNorm ---

@triton.jit
def _rmsnorm(X, W, Y, x_stride, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ok = offs < D
    x = tl.load(X + row * x_stride + offs, mask=ok, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(Y + row * D + offs, (n * w).to(tl.bfloat16), mask=ok)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """transformers' RMSNorm: bf16(w * bf16(x / rms(x))) over the last dim of bf16 rows [R, D]."""

    d = x.shape[-1]
    if not x.is_cuda:
        f = x.float()
        n = (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        return (w.float() * n.float()).to(torch.bfloat16)
    x2 = x.reshape(-1, d)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = out if out is not None else torch.empty((x2.shape[0], d), dtype=torch.bfloat16, device=x.device)
    _rmsnorm[(x2.shape[0],)](x2, w, y, x2.stride(0), eps, D=d, BLOCK=triton.next_power_of_2(d), num_warps=8)
    return y.view(*x.shape[:-1], d)


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


# ----------------------------------------------------------------------------------- MLA absorb and expand ---

ROW_KERNEL = 16          # calls of up to this many rows take the per-row kernels; prompt chunks batched matmuls


@triton.jit
def _absorb(Q, WK, QA, R, q_stride, H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, LW: tl.constexpr,
            BN: tl.constexpr):
    """Program (head, latent block): QA[r, h, n] = sum_k Q[r, h, k] WK[h, k, n], one fp32 sum a row."""

    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, DP)
    ok = k < D
    w = tl.load(WK + (h * D + k[:, None]) * LW + n[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    for r in range(R):
        q = tl.load(Q + r * q_stride + h * D + k, mask=ok, other=0.0).to(tl.float32)
        tl.store(QA + (r * H + h) * LW + n, tl.sum(q[:, None] * w, axis=0).to(tl.bfloat16))


@triton.jit
def _expand(OL, WV, OUT, R, H: tl.constexpr, DV: tl.constexpr, LW: tl.constexpr, BN: tl.constexpr):
    """Program (head, output block): OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k], one fp32 sum a row."""

    h = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, LW)
    w = tl.load(WV + (h * DV + n[:, None]) * LW + k[None, :]).to(tl.float32)
    for r in range(R):
        o = tl.load(OL + (r * H + h) * LW + k).to(tl.float32)
        tl.store(OUT + (r * H + h) * DV + n, tl.sum(w * o[None, :], axis=1).to(tl.bfloat16))


def absorb(q: torch.Tensor, wk: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """q [R, H, D] (rows may be strided) times kv_b's key rows wk [H, D, LW] -> out [R, H, LW] bf16."""

    rows, heads, d = q.shape
    lw = wk.shape[2]
    if not q.is_cuda:
        out.copy_(torch.einsum("rhd,hdn->rhn", q.float(), wk.float()).to(torch.bfloat16))
        return out
    if rows > ROW_KERNEL:
        out.copy_(torch.bmm(q.transpose(0, 1), wk).transpose(0, 1))
        return out
    if q.stride(2) != 1 or q.stride(1) != d:
        q = q.contiguous()
    _absorb[(heads, lw // 64)](q, wk, out, rows, q.stride(0), H=heads, D=d, DP=triton.next_power_of_2(d), LW=lw,
                               BN=64, num_warps=4)
    return out


def expand(o_lat: torch.Tensor, wv: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """o_lat [R, H, LW] times kv_b's value rows wv [H, DV, LW] -> out [R, H, DV] bf16."""

    rows, heads, lw = o_lat.shape
    dv = wv.shape[1]
    if not o_lat.is_cuda:
        out.copy_(torch.einsum("rhn,hdn->rhd", o_lat.float(), wv.float()).to(torch.bfloat16))
        return out
    if rows > ROW_KERNEL:
        out.copy_(torch.bmm(o_lat.transpose(0, 1), wv.transpose(1, 2)).transpose(0, 1))
        return out
    _expand[(heads, dv // 16)](o_lat, wv, out, rows, H=heads, DV=dv, LW=lw, BN=16, num_warps=4)
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
