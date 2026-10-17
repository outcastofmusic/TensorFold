"""LFM2-MoE's forward for one stream: prompt chunks, then one-row decode steps, sampled by the keyed sampler.

A conv layer keeps its last ``conv_L_cache - 1`` products B * x [taps - 1, D] bf16; an attention layer keeps its
rotated keys and values [max_len, KV, HD] bf16. Every operation reads its rows in a fixed order: block FP8 linears on
the fork's lane matmul (``Fp8BlockLinear``), bf16 ones on ``b16``, the norms on Qwen's row-invariant Triton kernel,
prompt attention on ``prefill_attention`` and decode attention on Nemotron-H's chunked kernel. A token samples with the
CUDA families' keyed rule (seed, position, logits), so a seed gives the same reply here as on the others' engines.

Every tensor operation here also runs on the CPU, with fp32 matmuls and attention in place of the kernels, so the
tests drive this same code against the fp32 reference.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from . import DECODE_ROWS, PREFILL_ROWS
from . import experts as X
from .weights import Attention, Block, Conv, Weights

CHUNK = 512          # Nemotron-H's decode attention: keys a chunk; the cache holds whole chunks


def add_rmsnorm(h: torch.Tensor, delta: torch.Tensor | None, w: torch.Tensor,
                eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(h + delta in bf16, rmsnorm of it times w in bf16)."""

    if h.is_cuda:
        from tensorfold.families.qwen3_5.cuda import glue

        h, y, _ = glue.add_rmsnorm(h, delta, w, eps)
        return h, y
    if delta is not None:
        h = (h.float() + delta.float()).to(torch.bfloat16)
    x = h.float()
    return h, (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def head_norm_rope(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """x [R, H, HD] bf16: each head RMS-normed (transformers' rounding: bf16, then times w), then rotated in fp32."""

    f = x.float()
    n = ((f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16) * w).float()
    half = n.shape[-1] // 2
    turned = torch.cat([-n[..., half:], n[..., :half]], -1)
    return (n * cos[:, None] + turned * sin[:, None]).to(torch.bfloat16)


class Engine:
    """One stream's caches: ``forward`` commits a prompt chunk or decode rows, ``sample`` draws the next token."""

    def __init__(self, w: Weights, *, max_len: int = 8192, prefill_rows: int = PREFILL_ROWS):
        if max_len % CHUNK:
            raise ValueError(f"max_len must be a multiple of {CHUNK}")
        c = self.c = w.config
        self.w = w
        dev = self.device = w.device
        self.cuda = dev.type == "cuda"
        self.max_len, self.prefill_rows = max_len, prefill_rows
        self.slot = {layer: j for j, layer in enumerate(c.attention_layers)}
        self.slot.update({layer: j for j, layer in enumerate(c.conv_layers)})
        na, nc = len(c.attention_layers), len(c.conv_layers)
        self.k_cache = [torch.zeros((max_len, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)
                        for _ in range(na)]
        self.v_cache = [torch.zeros_like(k) for k in self.k_cache]
        self.conv_state = torch.zeros((nc, c.conv_taps - 1, c.hidden), dtype=torch.bfloat16, device=dev)
        # one fp32 buffer a prompt chunk's experts decode into in turn: w13 [2I, D] is the wider of an expert's two
        self.decoded = torch.empty(max((b.moe.experts.w13[0].numel() for b in w.blocks if b.moe is not None),
                                       default=0), dtype=torch.float32, device=dev)
        self.inv_freq = (1.0 / (c.theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim))).to(dev)
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.sampled = torch.zeros(1, dtype=torch.int32, device=dev)
        self.sampling = None
        self.params = None
        if self.cuda:
            from tensorfold.families.nemotron_h.cuda import sampler as S

            self.params = S.Params(dev)
        self.bias, self.masked = None, False          # a reply grammar's -inf per disallowed column
        self.pos = 0

    # -- state ---------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.conv_state.zero_()
        self.pos = 0                                   # keys past pos are never read: the caches need no clearing

    def set_sampling(self, sampling) -> None:
        self.sampling = sampling
        if self.params is not None:
            self.params.set(sampling)

    @property
    def eos(self) -> tuple[int, ...]:
        return self.c.eos

    @property
    def head_width(self) -> tuple[int, int]:
        return self.c.vocab, 0

    # -- layers --------------------------------------------------------------------------------------------------
    def _rope(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(self.pos, self.pos + rows, device=self.device, dtype=torch.float32)
        freqs = positions[:, None] * self.inv_freq[None, :]
        angles = torch.cat([freqs, freqs], -1)
        return angles.cos(), angles.sin()

    def _conv(self, cv: Conv, x: torch.Tensor, j: int, prompt: bool) -> torch.Tensor:
        rows, d = x.shape
        b, gate, v = cv.in_proj(x, prompt).split(d, dim=-1)
        seq = torch.cat([self.conv_state[j], b * v])                    # B * x in bf16, as transformers rounds it
        taps = cv.taps.shape[1]
        acc = seq[0:rows].float() * cv.taps[:, 0]
        for t in range(1, taps):
            acc = acc + seq[t:t + rows].float() * cv.taps[:, t]
        self.conv_state[j].copy_(seq[rows:])
        return cv.out_proj(gate * acc.to(torch.bfloat16), prompt)

    def _attention(self, a: Attention, x: torch.Tensor, j: int, prompt: bool) -> torch.Tensor:
        c = self.c
        rows = x.shape[0]
        qd, kd = c.heads * c.head_dim, c.kv_heads * c.head_dim
        qkv = a.qkv(x, prompt)
        cos, sin = self._rope(rows)
        q = head_norm_rope(qkv[:, :qd].view(rows, c.heads, c.head_dim), a.q_norm, cos, sin, c.eps)
        k = head_norm_rope(qkv[:, qd:qd + kd].view(rows, c.kv_heads, c.head_dim), a.k_norm, cos, sin, c.eps)
        kc, vc = self.k_cache[j], self.v_cache[j]
        kc[self.pos:self.pos + rows] = k
        vc[self.pos:self.pos + rows] = qkv[:, qd + kd:].view(rows, c.kv_heads, c.head_dim)
        scale = c.head_dim ** -0.5
        if not self.cuda:
            out = _attend(q, kc[:self.pos + rows], vc[:self.pos + rows], self.pos, scale)
        elif prompt:
            from tensorfold.cuda.kernels import prefill_attention

            out = prefill_attention.attention(q.contiguous(), kc, vc, self.pos, scale=scale).view(rows, qd)
        else:
            from tensorfold.families.nemotron_h.cuda import attention as A

            self.meta[0] = self.pos
            # the kernel reads each row's q from its first heads * head_dim columns, rows its width apart
            out, _ = A.attention(q.view(rows, qd), kc, vc, self.meta, rows, heads=c.heads, kv_heads=c.kv_heads,
                                 head_dim=c.head_dim, max_chunks=self.max_len // CHUNK)
        return a.out(out, prompt)

    def _ffn(self, b: Block, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        if b.moe is not None:
            return X.moe(x, b.moe, self.c.top_k, self.c.scaling, few=DECODE_ROWS, work=self.decoded)
        g, u = b.w13(x, prompt).chunk(2, dim=-1)
        act = (torch.nn.functional.silu(g.float()).to(torch.bfloat16).float() * u.float()).to(torch.bfloat16)
        return b.w2(act, prompt)

    def _layers(self, ids: torch.Tensor, prompt: bool) -> torch.Tensor:
        """Every layer over the rows ``ids`` at positions pos..: the final-normed rows [R, D] bf16."""

        c, w = self.c, self.w
        h = w.embed[ids.long()]
        delta = None
        for i, b in enumerate(w.blocks):
            h, x = add_rmsnorm(h, delta, b.operator_norm, c.eps)
            j = self.slot[i]
            delta = self._conv(b.conv, x, j, prompt) if b.kind == "conv" else self._attention(b.attn, x, j, prompt)
            h, x = add_rmsnorm(h, delta, b.ffn_norm, c.eps)
            delta = self._ffn(b, x, prompt)
        return add_rmsnorm(h, delta, w.norm, c.eps)[1]

    def head(self, normed: torch.Tensor) -> torch.Tensor:
        """The tied head over bf16 rows [n, D] -> [n, V] (bf16 on CUDA, fp32 on the CPU)."""

        if not self.cuda:
            return normed.float() @ self.w.embed.float().t()
        from tensorfold.families.qwen3_5.cuda import b16

        return b16.matmul(normed, self.w.embed)

    # -- calls ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, tokens: Sequence[int], *, prompt: bool, every: bool = False) -> torch.Tensor:
        """Commit ``tokens`` at positions pos..; the logits of the last row, or of every row with ``every``."""

        rows = len(tokens)
        if not 1 <= rows <= (self.prefill_rows if prompt else DECODE_ROWS):
            raise ValueError("a call holds 1..prefill_rows prompt rows or 1..16 decode rows")
        if self.pos + rows > self.max_len:
            raise ValueError("the KV cache is full")
        ids = torch.as_tensor(list(tokens), dtype=torch.int32).to(self.device)
        normed = self._layers(ids, prompt)
        self.pos += rows
        logits = self.head(normed if every else normed[rows - 1:rows])
        if self.masked:
            logits = logits + self.bias[:1]
        return logits

    def sample(self, logits: torch.Tensor) -> int:
        """The token after position pos - 1 from its logits [1, V], by the request's keyed rule (greedy at T = 0)."""

        s = self.sampling
        if self.cuda:
            from tensorfold.families.nemotron_h.cuda import sampler as S

            self.meta[0] = self.pos - 1                  # the row samples meta[0] + 1: the next position
            S.sample(logits, self.meta, self.params, self.sampled)
            return int(self.sampled[0])
        if s is None or s.temperature <= 0:
            return int(torch.argmax(logits[0]))
        from tensorfold.engine.exact_sampling import MARGIN, choose

        vals, ids = torch.topk(logits[0].float(), min(logits.shape[-1], (s.top_k or logits.shape[-1]) + MARGIN))
        return choose(vals.numpy(), ids.numpy(), self.pos, s)

    def mask(self, constraint, window) -> None:
        """The next call's sampling masked by a reply's grammar; ``None`` lifts it."""

        if constraint is None or not window.rows:
            self.masked = False
            return
        width, offset = self.head_width
        allowed = constraint.allowed(window, width, self.device, offset)
        self.bias = torch.zeros((1, width), dtype=torch.float32, device=self.device).masked_fill_(~allowed[:1],
                                                                                                 float("-inf"))
        self.masked = True


def _attend(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, p0: int, scale: float) -> torch.Tensor:
    """The CPU's causal GQA in fp32: q [R, H, HD] at positions p0.. over keys and values [T, KV, HD] -> [R, H * HD]."""

    rows, heads, hd = q.shape
    group = heads // keys.shape[1]
    kk = keys.float().repeat_interleave(group, dim=1)
    vv = values.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("rhd,thd->hrt", q.float(), kk) * scale
    seen = torch.arange(keys.shape[0])[None, :] <= torch.arange(p0, p0 + rows)[:, None]
    scores = scores.masked_fill(~seen[None], float("-inf"))
    return torch.einsum("hrt,thd->rhd", scores.softmax(-1), vv).reshape(rows, heads * hd).to(torch.bfloat16)


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
def prefill(e: Engine, prompt: Sequence[int], sampling, *, constraint=None) -> int:
    """Reset, commit the prompt in chunks and return its first sampled token (position len(prompt))."""

    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prefill needs at least one token")
    e.reset()
    e.set_sampling(sampling)
    try:
        for s in range(0, len(prompt), e.prefill_rows):
            last = s + e.prefill_rows >= len(prompt)
            if last and constraint is not None:        # a reply's grammar masks the first token's row
                e.mask(constraint, constraint.window([0], [-1]))
            logits = e.forward(prompt[s:s + e.prefill_rows], prompt=True)
        pending = e.sample(logits)
    finally:
        e.mask(None, None)
    if constraint is not None:
        constraint.advance([pending])
    return pending


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling, *, stop_eos: bool = True,
                  on_tokens: Callable[[list[int]], bool | None] | None = None, constraint=None) -> DecodeResult:
    """``count`` tokens (the first is ``pending``), one row a step; stops after an end token or ``on_tokens``' True."""

    e.set_sampling(sampling)
    out = [int(pending)]
    eos = set(e.c.eos)
    _sync(e)
    start = time.perf_counter()
    try:
        while len(out) < count and not (stop_eos and out[-1] in eos):
            if constraint is not None:
                e.mask(constraint, constraint.window([out[-1]], [-1]))
            tok = e.sample(e.forward([out[-1]], prompt=False))
            out.append(tok)
            if constraint is not None:
                constraint.advance([tok])
            if on_tokens is not None and on_tokens([tok]):
                break
    finally:
        e.mask(None, None)
    _sync(e)
    return DecodeResult(out, time.perf_counter() - start)
