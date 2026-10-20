"""LFM2-MoE's forward for one stream: prompt chunks, then one-row decode steps, sampled by the keyed sampler.

A conv layer keeps its last ``conv_L_cache - 1`` products B * x [taps - 1, D] bf16; an attention layer keeps its
rotated keys and values [max_len, KV, HD] bf16. Every operation reads its rows in a fixed order: block FP8 linears on
the fork's lane matmul (``Fp8BlockLinear``), bf16 ones on ``b16``, the norms on Qwen's row-invariant Triton kernel,
prompt attention on ``prefill_attention`` and decode attention on Nemotron-H's chunked kernel. A token samples with the
CUDA families' keyed rule (seed, position, logits), so a seed gives the same reply here as on the others' engines.

A one-row decode step replays a CUDA graph (``TF_LFM2_GRAPHS=0`` runs it eagerly): it reads its position from the
device ``meta`` (RoPE, the KV row it writes, attention's length) and its token from a static buffer, so one capture
serves every position and the replay launches the eager step's kernels with its shapes, giving its bits.

``TF_LFM2_DECODE_FUSE`` (default on; ``0`` restores the torch chains) runs a decode call's conv and MoE combine as
one kernel each (``fused.py``, the chains' bits), computes RoPE's angles and the KV rows once a call instead of once
an attention layer, and drops two launches a MoE layer that change no value: route's int32 copy of the ids and its
multiply by a scaling of 1.0. A one-row step launched about 2,040 kernels and launches about 1,020 with it.

Every tensor operation here also runs on the CPU, with fp32 matmuls and attention in place of the kernels, so the
tests drive this same code against the fp32 reference.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tensorfold.env_flags import flag

from . import DECODE_ROWS, PAIRS_ROWS, PREFILL_ROWS, fused
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
        # the expert kernels of prompt chunks (experts.path): TF_LFM2_PROMPT_KERNEL=fp32 keeps the reference path
        self.prompt_kernel = os.environ.get("TF_LFM2_PROMPT_KERNEL", "auto").strip() or "auto"
        X.path(self.prompt_kernel, 1, 0, 0)            # refuses an unknown value at startup
        self.fuse = flag("TF_LFM2_DECODE_FUSE", True) and self.cuda      # read once: a graph keeps what it captured
        self.combine_fits = fused.combine_fits(c.hidden)                 # LFM2.5-24B's 2,048 channels fit
        self.call_rope: tuple[torch.Tensor, ...] | None = None           # a fused call's (cos, sin, rows' positions)
        self.slot = {layer: j for j, layer in enumerate(c.attention_layers)}
        self.slot.update({layer: j for j, layer in enumerate(c.conv_layers)})
        na, nc = len(c.attention_layers), len(c.conv_layers)
        self.k_cache = [torch.zeros((max_len, c.kv_heads, c.head_dim), dtype=torch.bfloat16, device=dev)
                        for _ in range(na)]
        self.v_cache = [torch.zeros_like(k) for k in self.k_cache]
        self.conv_state = torch.zeros((nc, c.conv_taps - 1, c.hidden), dtype=torch.bfloat16, device=dev)
        # one fp32 buffer a prompt chunk's experts decode into in turn: w13 [2I, D] is the wider of an expert's two.
        # Only the fp32 path (``grouped``: always on the CPU, TF_LFM2_PROMPT_KERNEL=fp32 on CUDA) reads it.
        self.decoded = (torch.empty(max((b.moe.experts.w13[0].numel() for b in w.blocks if b.moe is not None),
                                        default=0), dtype=torch.float32, device=dev)
                        if not self.cuda or self.prompt_kernel == "fp32" else None)
        self.inv_freq = (1.0 / (c.theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim))).to(dev)
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)       # meta[0]: the call's first position
        self.sampled = torch.zeros(1, dtype=torch.int32, device=dev)
        self.offsets = torch.arange(max(prefill_rows, DECODE_ROWS), dtype=torch.int32, device=dev)
        self.tok = torch.zeros(1, dtype=torch.int32, device=dev)       # a graphed step's input token
        self.graphs = self.cuda and os.environ.get("TF_LFM2_GRAPHS", "1").strip() != "0"
        self.graph = None
        self.graph_out: torch.Tensor | None = None
        self.sampling = None
        self.params = None
        if self.cuda:
            from tensorfold.families.nemotron_h.cuda import sampler as S

            self.params = S.Params(dev)
        self.bias, self.masked = None, False          # a reply grammar's -inf per disallowed column
        self.pos = 0
        self.parts: dict[int, torch.Tensor] = {}       # the last prompt chunk's conv states at its inner cuts

    # -- state ---------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.conv_state.zero_()
        self.pos = 0                                   # keys past pos are never read: the caches need no clearing

    def state(self, n: int, part: torch.Tensor | None = None) -> dict:
        """A copy of the committed state after positions 0 .. n - 1: the conv states (``part``, a prompt chunk's at an
        inner cut, else the live ones at n == pos) and the attention rows, which are the caches' first n rows."""

        if not 0 < n <= self.pos or (part is None and n != self.pos):
            raise ValueError(f"no committed conv state at {n} (the engine is at {self.pos})")
        conv = part if part is not None else self.conv_state.clone()
        return {"conv": conv, "k": [k[:n].clone() for k in self.k_cache], "v": [v[:n].clone() for v in self.v_cache],
                "pos": n}

    def restore(self, st: dict) -> None:
        """The engine as a prefill of the kept state's positions left it."""

        n = int(st["pos"])
        self.conv_state.copy_(st["conv"])
        for dst, src in zip(self.k_cache + self.v_cache, st["k"] + st["v"]):
            dst[:n].copy_(src)
        self.pos = n

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
        # from the device meta, so a captured step serves every position; whole numbers below 2^24 add exactly
        positions = (self.meta[:1] + self.offsets[:rows]).float()
        freqs = positions[:, None] * self.inv_freq[None, :]
        angles = torch.cat([freqs, freqs], -1)
        return angles.cos(), angles.sin()

    def _conv(self, cv: Conv, x: torch.Tensor, j: int, prompt: bool) -> torch.Tensor:
        rows, d = x.shape
        if self.fuse and not prompt and not self.parts and fused.conv_fits(rows, self.conv_state[j], cv.taps):
            return cv.out_proj(fused.conv_step(cv.in_proj(x, prompt), self.conv_state[j], cv.taps), prompt)
        b, gate, v = cv.in_proj(x, prompt).split(d, dim=-1)
        seq = torch.cat([self.conv_state[j], b * v])                    # B * x in bf16, as transformers rounds it
        taps = cv.taps.shape[1]
        acc = seq[0:rows].float() * cv.taps[:, 0]
        for t in range(1, taps):
            acc = acc + seq[t:t + rows].float() * cv.taps[:, t]
        for q, part in self.parts.items():          # the state after q rows: the taps - 1 products ending at row q
            part[j].copy_(seq[q:q + taps - 1])
        self.conv_state[j].copy_(seq[rows:])
        return cv.out_proj(gate * acc.to(torch.bfloat16), prompt)

    def _attention(self, a: Attention, x: torch.Tensor, j: int, prompt: bool) -> torch.Tensor:
        c = self.c
        rows = x.shape[0]
        qd, kd = c.heads * c.head_dim, c.kv_heads * c.head_dim
        qkv = a.qkv(x, prompt)
        if self.fuse and not prompt:                # every attention layer of a call reads the same positions
            if self.call_rope is None:
                self.call_rope = (*self._rope(rows), (self.meta[:1] + self.offsets[:rows]).long())
            cos, sin, at = self.call_rope
        else:
            cos, sin = self._rope(rows)
            at = None
        q = head_norm_rope(qkv[:, :qd].view(rows, c.heads, c.head_dim), a.q_norm, cos, sin, c.eps)
        k = head_norm_rope(qkv[:, qd:qd + kd].view(rows, c.kv_heads, c.head_dim), a.k_norm, cos, sin, c.eps)
        kc, vc = self.k_cache[j], self.v_cache[j]
        if at is None:
            at = (self.meta[:1] + self.offsets[:rows]).long()
        kc.index_copy_(0, at, k)
        vc.index_copy_(0, at, qkv[:, qd + kd:].view(rows, c.kv_heads, c.head_dim))
        scale = c.head_dim ** -0.5
        if not self.cuda:
            out = _attend(q, kc[:self.pos + rows], vc[:self.pos + rows], self.pos, scale)
        elif prompt:
            from tensorfold.cuda.kernels import prefill_attention

            out = prefill_attention.attention(q.contiguous(), kc, vc, self.pos, scale=scale).view(rows, qd)
        else:
            from tensorfold.families.nemotron_h.cuda import attention as A

            # the kernel reads each row's q from its first heads * head_dim columns, rows its width apart
            out, _ = A.attention(q.view(rows, qd), kc, vc, self.meta, rows, heads=c.heads, kv_heads=c.kv_heads,
                                 head_dim=c.head_dim, max_chunks=self.max_len // CHUNK)
        return a.out(out, prompt)

    def _ffn(self, b: Block, x: torch.Tensor, prompt: bool) -> torch.Tensor:
        if b.moe is not None:
            return X.moe(x, b.moe, self.c.top_k, self.c.scaling, few=DECODE_ROWS, work=self.decoded,
                         kernel=self.prompt_kernel, pairs_rows=PAIRS_ROWS,
                         fuse=self.fuse and self.combine_fits and not prompt)
        g, u = b.w13(x, prompt).chunk(2, dim=-1)
        act = (torch.nn.functional.silu(g.float()).to(torch.bfloat16).float() * u.float()).to(torch.bfloat16)
        return b.w2(act, prompt)

    def _layers(self, ids: torch.Tensor, prompt: bool) -> torch.Tensor:
        """Every layer over the rows ``ids`` at positions pos..: the final-normed rows [R, D] bf16."""

        c, w = self.c, self.w
        self.call_rope = None
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
    def forward(self, tokens: Sequence[int], *, prompt: bool, every: bool = False,
                cuts: Sequence[int] = ()) -> torch.Tensor:
        """Commit ``tokens`` at positions pos..; the logits of the last row, or of every row with ``every``.

        ``parts`` then holds the conv states after each of ``cuts`` (rows inside the call), for ``state``.
        """

        rows = len(tokens)
        if not 1 <= rows <= (self.prefill_rows if prompt else DECODE_ROWS):
            raise ValueError("a call holds 1..prefill_rows prompt rows or 1..16 decode rows")
        if self.pos + rows > self.max_len:
            raise ValueError("the KV cache is full")
        if any(not 0 < int(q) < rows for q in cuts):
            raise ValueError("a cut lies inside the call")
        self.parts = {int(q): torch.empty_like(self.conv_state) for q in cuts}
        self.meta[0] = self.pos
        if self.graphs and not prompt and rows == 1:
            self.tok[0] = int(tokens[0])
            logits = self._graphed()
        else:
            ids = torch.as_tensor(list(tokens), dtype=torch.int32).to(self.device)
            normed = self._layers(ids, prompt)
            logits = self.head(normed if every else normed[rows - 1:rows])
        self.pos += rows
        if self.masked:
            logits = logits + self.bias[:1]
        return logits

    def _step(self) -> torch.Tensor:
        """One decode row: the token in ``tok`` at position meta[0] -> its logits [1, V]."""

        return self.head(self._layers(self.tok, False))

    def _graphed(self) -> torch.Tensor:
        """``_step`` by graph replay. The first call runs the step eagerly (it compiles the kernels and commits its
        row), then captures it; capturing launches nothing, so the state moves once."""

        if self.graph is None:
            out = self._step()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):   # the server's threads keep running
                self.graph_out = self._step()
            self.graph = g
            return out
        self.graph.replay()
        return self.graph_out.clone()               # the next replay rewrites graph_out

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
def prefill(e: Engine, prompt: Sequence[int], sampling, *, constraint=None, resume: dict | None = None,
            keeps: Sequence[int] = (), keep: Callable[[int, dict], None] | None = None) -> int:
    """Commit the prompt in chunks and return its first sampled token (position len(prompt)).

    A fresh prompt starts from a reset; ``resume`` (an ``Engine.state`` of a strict prefix) starts at its length.
    ``keep(n, state)`` gets the engine's state after each of ``keeps`` (positions past the start).
    """

    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prefill needs at least one token")
    begin = 0 if resume is None else int(resume["pos"])
    if not 0 <= begin < len(prompt):
        raise ValueError("a resumed prompt must extend the kept positions")
    points = sorted({int(p) for p in keeps if begin < int(p) <= len(prompt)})
    if points and keep is None:
        raise ValueError("kept positions need a callback")
    e.reset()
    if resume is not None:
        e.restore(resume)
    e.set_sampling(sampling)
    try:
        for s in range(begin, len(prompt), e.prefill_rows):
            chunk = prompt[s:s + e.prefill_rows]
            rows = len(chunk)
            if s + rows >= len(prompt) and constraint is not None:      # a reply's grammar masks the first token's row
                e.mask(constraint, constraint.window([0], [-1]))
            here = [p - s for p in points if s < p <= s + rows]
            logits = e.forward(chunk, prompt=True, cuts=[q for q in here if q < rows])
            for q in here:
                keep(s + q, e.state(s + q, e.parts.get(q)))
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
