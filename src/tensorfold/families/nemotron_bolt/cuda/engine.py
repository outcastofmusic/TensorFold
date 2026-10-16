"""Nemotron 3.5 Bolt's forward on CUDA: a verify window's row r gets the bits a serial step at its position gets.

Every kernel a window row passes through reads only that row, or rows in a fixed order: the NVFP4 lane matmul, the
bf16 linear (``b16.matmul``), Nemotron-H's Mamba-2 and attention kernels (the state lags a window and replays its
kept rows first) and the keyed sampler. So a draft accepted because it equals ``tokens()[r]`` is the token serial
decoding samples there, and a drafted reply equals a ``draft: false`` one token for token.

The routed experts go through ``iq2xxs_grouped`` and the shared expert through ``iq2xxs_linear``: the KERNELS part's
``iq2xxs.py`` when present, else ``fallback.py`` (the same signatures).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import triton

from tensorfold.cuda.kernels import prefill_attention
from tensorfold.families.nemotron_h.cuda import attention as A
from tensorfold.families.nemotron_h.cuda import glue as G
from tensorfold.families.nemotron_h.cuda import mamba as M
from tensorfold.families.nemotron_h.cuda import sampler as S
from tensorfold.families.nemotron_h.cuda.engine import sampling_mode
from tensorfold.families.qwen3_5.cuda import b16
from tensorfold.families.qwen3_5.cuda import glue as base

from .weights import Attention, Block, Mamba, MoE, Weights

try:                                    # the KERNELS part's kernels, once merged; the contract's signatures either way
    from .iq2xxs import iq2xxs_grouped, iq2xxs_linear
    EXPERTS = "iq2xxs kernels"
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.iq2xxs":
        raise
    from .fallback import iq2xxs_grouped, iq2xxs_linear
    EXPERTS = "fallback (dequantize the called experts, bf16 linear)"

ROWS = 16            # the window limit
PREFILL_ROWS = 2048  # rows of a prompt chunk


def route(x: torch.Tensor, moe: MoE, *, scaling: float, norm: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """x (R, D) bf16 -> (expert ids (R, top_k) int32, fp32 weights): sigmoid scores, corrected pick, scaled weights.

    Nemotron-H's router kernels: fp32 dots in fixed K slices a row, then the top-k in pick order (its two extra slots,
    the folded shared expert's, are cut off here).
    """

    rows, d = x.shape
    e, k = moe.experts, moe.top_k
    ns = triton.next_power_of_2(k + 2)
    sk = G._divisor(d // 64, 6)
    part = torch.empty((sk, rows, e), dtype=torch.float32, device=x.device)
    ids = torch.zeros((rows, ns), dtype=torch.int32, device=x.device)
    wts = torch.zeros((rows, ns), dtype=torch.float32, device=x.device)
    G._router[(triton.cdiv(rows, 16), triton.cdiv(e, 16), sk)](x, moe.router, part, rows, D=d, E=e, SK=sk, BE=16,
                                                               num_warps=4)
    G._topk[(rows,)](part, moe.bias, ids, wts, rows, float(scaling), E=e, EP=triton.next_power_of_2(e), SK=sk,
                     TOPK=k, NS=ns, NORM=norm, num_warps=4)
    return ids[:, :k], wts[:, :k]


def dense(x: torch.Tensor, w: torch.Tensor, prompt: bool) -> torch.Tensor:
    """bf16 rows times a bf16 weight [N, K]: the row-invariant linear in a window, the prompt mma in a chunk."""

    return b16.prompt(x, w) if prompt else b16.matmul(x, w)


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Nemotron's relu(x)^2 in bf16, as transformers' ReLUSquaredActivation rounds it."""

    r = torch.relu(x)
    return r * r


def latent_moe(moe: MoE, normed: torch.Tensor, *, scaling: float, norm: bool, prompt: bool) -> torch.Tensor:
    """The MoE block's output (R, D) bf16: routed experts in the latent space, then fc2, plus the shared expert.

    As modeling_nemotron_h.NemotronHMoE: the router reads the block input, the routed experts read fc1(input) and
    their weighted sum (fp32, rounded to bf16) goes through fc2; the shared expert reads the block input.
    """

    rows = normed.shape[0]
    ids, wts = route(normed, moe, scaling=scaling, norm=norm)
    k = moe.top_k
    latent = dense(normed, moe.fc1, prompt)
    pairs = latent.repeat_interleave(k, dim=0)                     # row r's k copies, in pick order
    flat_ids = ids.reshape(-1)
    act = relu2(iq2xxs_grouped(pairs, moe.up, flat_ids))
    y = iq2xxs_grouped(act, moe.down, flat_ids).view(rows, k, -1)
    acc = y[:, 0].float() * wts[:, 0:1]
    for j in range(1, k):                                          # a fixed order, so a row's sum ignores R
        acc = acc + y[:, j].float() * wts[:, j:j + 1]
    routed = dense(acc.to(torch.bfloat16), moe.fc2, prompt)
    shared = iq2xxs_linear(relu2(iq2xxs_linear(normed, moe.shared_up)), moe.shared_down)
    return routed + shared


class Engine:
    """One sequence's caches and buffers; ``verify`` runs a window, ``prefill_chunk`` a prompt chunk."""

    STATE = ("ssm", "conv_base", "raw", "xc", "dt")

    def __init__(self, w: Weights, *, max_len: int = 8192, max_rows: int = ROWS, prefill_rows: int = PREFILL_ROWS,
                 taps: Sequence[int] = ()):
        if max_rows > ROWS:
            raise ValueError(f"windows hold at most {ROWS} rows")
        if max_len % A.CHUNK:
            raise ValueError(f"max_len must be a multiple of {A.CHUNK}")
        c = self.c = w.config
        bad = [t for t in taps if not 0 <= int(t) < len(c.pattern)]
        if bad:
            raise ValueError(f"tap layers {bad} are outside the model's {len(c.pattern)} layers")
        self.w = w
        dev = self.device = w.norm_f.device
        self.max_len, self.max_rows, self.max_chunks = max_len, max_rows, max_len // A.CHUNK
        self.prefill_rows = prefill_rows
        nm = c.pattern.count("M")
        self.meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.p_meta = torch.zeros(4, dtype=torch.int32, device=dev)
        self.k_cache = [torch.zeros((max_len, kv, c.head_dim), dtype=torch.bfloat16, device=dev) for kv in c.kv_heads]
        self.v_cache = [torch.zeros_like(k) for k in self.k_cache]
        self.ssm = torch.zeros((nm, c.m_heads, c.m_head_dim, c.m_state), dtype=torch.float32, device=dev)
        self.conv_base = torch.zeros((nm, c.conv_kernel - 1, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.raw = torch.zeros((nm, 2, max_rows, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.xc = torch.zeros_like(self.raw)
        self.dt = torch.zeros((nm, 2, max_rows, c.m_heads), dtype=torch.float32, device=dev)
        self.p_xc = torch.zeros((prefill_rows, c.conv_dim), dtype=torch.bfloat16, device=dev)
        self.tap_layers = tuple(int(t) for t in taps)
        self._tap_slots: dict[int, list[int]] = {}
        for slot, layer in enumerate(self.tap_layers):
            self._tap_slots.setdefault(layer, []).append(slot)
        self._taps: list[torch.Tensor] = []
        self.hidden = torch.zeros((0, c.hidden), dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros((max_rows, c.vocab), dtype=torch.bfloat16, device=dev)
        self.sampled = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        self.p_sampled = torch.zeros(1, dtype=torch.int32, device=dev)
        self.params = S.Params(dev)
        self.mode: tuple = ("greedy",)
        self.bias, self.masked = None, False      # a reply grammar's -inf per disallowed column, rows of the window
        self._ready = torch.cuda.Event()
        self.pos = self.parity = self.prev_keep = 0
        self._rows = 0

    # -- state ---------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        for name in self.STATE:
            getattr(self, name).zero_()
        for t in self.k_cache + self.v_cache:
            t.zero_()
        self.pos = self.parity = self.prev_keep = 0
        self._rows = 0

    def set_sampling(self, sampling) -> None:
        """The request's sampling: seed, temperature and top-p live on the device for the keyed sampler."""

        self.params.set(sampling)
        self.mode = sampling_mode(sampling)

    # -- one layer ----------------------------------------------------------------------------------------------
    def _mamba(self, m: Mamba, normed: torch.Tensor, rows: int, j: int, prompt: bool) -> torch.Tensor:
        c = self.c
        params = {"heads": c.m_heads, "head_dim": c.m_head_dim, "groups": c.m_groups, "state_dim": c.m_state,
                  "lo": c.dt_min, "hi": c.dt_max}
        if prompt:
            proj = m.in_proj.prefill(normed)
            M.conv_rows(proj, self.conv_base[j], self.p_xc[:rows], m.conv_w, m.conv_b, rows, xd=c.xd)
            y = M.scan_rows(proj, self.p_xc[:rows], self.ssm[j], m.a, m.d, m.dt_bias, rows, **params)
        else:
            proj = m.in_proj(normed)
            M.conv(proj, self.conv_base[j], self.raw[j], self.xc[j], m.conv_w, m.conv_b, self.meta, rows, xd=c.xd)
            y = M.scan(proj, self.xc[j], self.dt[j], self.ssm[j], m.a, m.d, m.dt_bias, self.meta, rows, **params)
        g, _ = M.group_rmsnorm(y, m.gnorm, c.eps, c.m_groups)
        return m.out_proj.prefill(g) if prompt else m.out_proj(g)

    def _attention(self, a: Attention, normed: torch.Tensor, rows: int, j: int, prompt: bool) -> torch.Tensor:
        c = self.c
        qd = c.heads * c.head_dim
        qkv = dense(normed, a.qkv, prompt)
        kc, vc = self.k_cache[j], self.v_cache[j]
        meta = self._meta_at(self.pos) if prompt else self.meta
        A.kv_write(qkv, kc, vc, meta, rows, q_dim=qd)
        if prompt:
            q = qkv[:, :qd].contiguous().view(rows, c.heads, c.head_dim)
            out = prefill_attention.attention(q, kc, vc, self.pos, scale=c.head_dim ** -0.5).view(rows, qd)
        else:
            out, _ = A.attention(qkv, kc, vc, self.meta, rows, heads=c.heads, kv_heads=a.kv_heads,
                                 head_dim=c.head_dim, max_chunks=self.max_chunks)
        return dense(out, a.o, prompt)

    def _block(self, blk: Block, normed: torch.Tensor, rows: int, js: list[int], prompt: bool) -> torch.Tensor:
        if blk.kind == "M":
            js[0] += 1
            return self._mamba(blk.mamba, normed, rows, js[0] - 1, prompt)
        if blk.kind == "*":
            js[1] += 1
            return self._attention(blk.attn, normed, rows, js[1] - 1, prompt)
        return latent_moe(blk.moe, normed, scaling=self.c.scaling, norm=self.c.norm_topk, prompt=prompt)

    def _layers(self, ids: torch.Tensor, prompt: bool) -> torch.Tensor:
        """Every layer over the rows ``ids``: the final-normed hidden rows; each tap layer's stream into ``_taps``."""

        w, c = self.w, self.c
        rows = ids.shape[0]
        x = w.embed[ids.long()]
        delta = None
        js = [0, 0]
        self._taps = [x] * len(self.tap_layers)
        for i, blk in enumerate(w.blocks):
            x, normed, _ = base.add_rmsnorm(x, delta, blk.norm, c.eps)
            for slot in self._tap_slots.get(i - 1, ()):       # the stream after layer i - 1
                self._taps[slot] = x
            delta = self._block(blk, normed, rows, js, prompt)
        h, normed, _ = base.add_rmsnorm(x, delta, w.norm_f, c.eps)
        for slot in self._tap_slots.get(len(w.blocks) - 1, ()):
            self._taps[slot] = h
        return normed

    def _meta_at(self, pos: int) -> torch.Tensor:
        self.p_meta.fill_(pos)
        return self.p_meta

    # -- windows ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def verify(self, tokens: Sequence[int]) -> torch.Tensor:
        """Logits (R, V) bf16 of the window ``tokens`` at positions pos .. pos + R - 1, sampled into ``tokens()``.

        Nothing is committed: ``commit(keep)`` keeps the window's first ``keep`` rows.
        """

        rows = len(tokens)
        if not 1 <= rows <= self.max_rows:
            raise ValueError(f"a window holds 1..{self.max_rows} rows")
        if self.pos + rows > self.max_len:
            raise ValueError("the KV cache is full")
        self.meta.copy_(torch.tensor([self.pos, self.parity, self.prev_keep, 0], dtype=torch.int32))
        ids = torch.as_tensor(list(tokens), dtype=torch.int32).to(self.device)
        normed = self._layers(ids, prompt=False)
        self.hidden = normed
        self.logits[:rows].copy_(b16.matmul(normed, self.w.head))
        if self.masked:
            self.logits[:rows].add_(self.bias[:rows])
        # row r samples position pos + r + 1 with the keyed sampler: the same token serial decoding samples there
        S.sample(self.logits[:rows], self.meta, self.params, self.sampled[:rows])
        self._ready.record()
        self.parity ^= 1
        self._rows = rows
        return self.logits[:rows]

    def tokens(self) -> list[int]:
        """The last window's sampled tokens: row r is the token at position pos + r + 1."""

        self._ready.synchronize()
        return self.sampled[:self._rows].tolist()

    def commit(self, keep: int) -> None:
        """Keep the first ``keep`` rows of the last window (the Mamba states replay them in the next window)."""

        if not 1 <= keep <= self._rows:
            raise ValueError("keep must be between 1 and the window's rows")
        self.pos += keep
        self.prev_keep = keep

    def tap_rows(self, n: int) -> torch.Tensor:
        """The last window's or prompt chunk's first ``n`` rows of taps, layers in ``tap_layers`` order: [n, T * D]."""

        if not self.tap_layers:
            raise ValueError("this engine was made without tap layers")
        return torch.cat([t[:n] for t in self._taps], dim=1)

    def hidden_rows(self, n: int) -> torch.Tensor:
        """The last window's or prompt chunk's first ``n`` final-normed hidden rows [n, D] bf16."""

        return self.hidden[:n]

    def mask(self, constraint, window) -> None:
        """The next window's sampling masked by a reply's grammar; ``None`` lifts it."""

        if constraint is None or not window.rows:
            self.masked = False
            return
        width, offset = self.head_width
        if self.bias is None:
            self.bias = torch.zeros((self.max_rows, width), dtype=torch.bfloat16, device=self.device)
        self.bias.zero_()
        allowed = constraint.allowed(window, width, self.device, offset)
        index = torch.tensor(window.rows, device=self.device)
        self.bias[index] = torch.zeros_like(allowed, dtype=torch.bfloat16).masked_fill_(~allowed, float("-inf"))
        self.masked = True

    @property
    def head_width(self) -> tuple[int, int]:
        return self.c.vocab, 0

    # -- prompt chunks ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def prefill_chunk(self, tokens: Sequence[int]) -> None:
        """Commit a prompt chunk at positions pos .. and sample the token after its last row into ``prefill_token``."""

        rows = len(tokens)
        if not 1 <= rows <= self.prefill_rows or self.pos + rows > self.max_len:
            raise ValueError("a prompt chunk must fit the chunk buffers and the KV cache")
        if self.prev_keep:
            raise RuntimeError("a prompt chunk runs from a committed state (no window rows left to replay)")
        ids = torch.as_tensor(list(tokens), dtype=torch.int32).to(self.device)
        normed = self._layers(ids, prompt=True)
        self.hidden = normed
        self._rows = rows
        self.pos += rows
        logits = b16.matmul(normed[rows - 1:rows], self.w.head)
        if self.masked:
            logits = logits + self.bias[:1]
        S.sample(logits, self._meta_at(self.pos - 1), self.params, self.p_sampled)
        self._ready.record()

    def prefill_token(self) -> int:
        self._ready.synchronize()
        return int(self.p_sampled[0])


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    widths: list[int] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling, *, on_taps: Callable[[torch.Tensor], None] | None = None,
            constraint=None, stop: Callable[[], bool] | None = None) -> int:
    """Reset, commit the prompt in chunks and return its first sampled token (position len(prompt)).

    ``on_taps`` gets each chunk's taps [R, T * D] in order (a drafter's context); ``stop()`` is asked before each
    chunk and abandons the prompt with ``RequestCancelled``.
    """

    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("prefill needs at least one token")
    e.reset()
    e.set_sampling(sampling)
    if constraint is not None:                           # a reply's grammar masks the first token's row
        e.mask(constraint, constraint.window([0], [-1]))
    try:
        for s in range(0, len(prompt), e.prefill_rows):
            if stop is not None and stop():
                from tensorfold.server.cancellation import RequestCancelled

                raise RequestCancelled("the client left while its prompt was filling")
            chunk = prompt[s:s + e.prefill_rows]
            e.prefill_chunk(chunk)
            if on_taps is not None:
                on_taps(e.tap_rows(len(chunk)))
        pending = e.prefill_token()
    finally:
        if constraint is not None:
            e.mask(None, None)
    if constraint is not None:
        constraint.advance([pending])
    return pending


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling, *, stop_eos: bool = True,
                  on_tokens: Callable[[list[int]], bool | None] | None = None, constraint=None) -> DecodeResult:
    """``count`` tokens (the first is ``pending``), one window of one row a token: the reference drafts must equal."""

    e.set_sampling(sampling)
    out = [int(pending)]
    eos = set(e.c.eos)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        while len(out) < count and not (stop_eos and out[-1] in eos):
            if constraint is not None:
                e.mask(constraint, constraint.window([out[-1]], [-1]))
            e.verify([out[-1]])
            tok = e.tokens()[0]
            e.commit(1)
            out.append(tok)
            if constraint is not None:
                constraint.advance([tok])
            if on_tokens is not None and on_tokens([tok]):
                break
    finally:
        e.mask(None, None)
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, widths=[1] * (len(out) - 1))
