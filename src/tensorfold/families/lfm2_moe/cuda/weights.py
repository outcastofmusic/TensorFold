"""LFM2-MoE weights on the device: block FP8 linears on the fork's lane matmul, bf16 ones on ``b16``, experts stacked.

Attention's q, k and v and the dense SwiGLU's w1 and w3 are each stacked into one ``Fp8BlockLinear`` (their row
counts are whole 128-row blocks, so each keeps its own block scales). compressed-tensors' ``weight_scale`` is the
same multiplier DeepSeek's ``weight_scale_inv`` is (bf16 here, fp32 there), so ``Fp8BlockLinear.column_scales`` reads
it as is. A layer's routed experts stay in their stored layout, stacked: w1 over w3 as ``w13`` [E, 2I, D] and w2 as
[E, D, I], e4m3 bytes with fp32 block scales (bf16 weights from the bf16 checkpoint), for ``experts.py``.

On the CPU (``device="cpu"``, the tests' engine) every linear is its exact fp32 decode instead of a kernel layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import PREFIX, dequantize, layer
from ..config import Config


@dataclass
class Linear:
    """y = x W^T in bf16: ``fp8`` (an ``Fp8BlockLinear``), else ``w`` (bf16 on CUDA, the fp32 decode on the CPU)."""

    n: int
    k: int
    fp8: Any = None
    w: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        if self.fp8 is not None:
            return self.fp8.prefill(x) if prompt else self.fp8(x)
        if not x.is_cuda:
            return (x.float() @ self.w.float().t()).to(torch.bfloat16)
        from tensorfold.families.qwen3_5.cuda import b16

        return b16.prompt(x, self.w) if prompt else b16.matmul(x, self.w)

    def nbytes(self) -> int:
        return int(self.fp8.nbytes()) if self.fp8 is not None else self.w.numel() * self.w.element_size()


@dataclass
class Experts:
    """One layer's routed experts as stored: FP8 bytes with fp32 block scales, or bf16 with ``s13 = s2 = None``."""

    w13: torch.Tensor                # [E, 2I, D] uint8 (e4m3) or bf16: w1's rows, then w3's
    w2: torch.Tensor                 # [E, D, I]
    s13: torch.Tensor | None         # [E, 2I / 128, D / 128] fp32
    s2: torch.Tensor | None          # [E, D / 128, I / 128] fp32

    @property
    def count(self) -> int:
        return int(self.w13.shape[0])

    @property
    def width(self) -> int:
        return int(self.w2.shape[2])

    @property
    def fp8(self) -> bool:
        return self.s13 is not None

    def dense(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert ``e``'s (w13, w2) in fp32: e4m3 x scale, exact."""

        if not self.fp8:
            return self.w13[e].float(), self.w2[e].float()
        return dequantize(self.w13[e], self.s13[e]), dequantize(self.w2[e], self.s2[e])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13, self.w2, self.s13, self.s2) if t is not None)


@dataclass
class Conv:
    in_proj: Linear                  # [3D, D]: B, C, x
    out_proj: Linear
    taps: torch.Tensor               # [D, taps] fp32, the oldest input's tap first


@dataclass
class Attention:
    qkv: Linear                      # [(H + 2 KV) HD, D]: q, k, v
    out: Linear
    q_norm: torch.Tensor             # [HD] bf16
    k_norm: torch.Tensor


@dataclass
class Moe:
    router: torch.Tensor             # [E, D] fp32 (bf16 as stored, widened once)
    bias: torch.Tensor               # [E] fp32
    experts: Experts


@dataclass
class Block:
    kind: str                        # "conv" or "full_attention"
    operator_norm: torch.Tensor      # [D] bf16
    ffn_norm: torch.Tensor
    conv: Conv | None = None
    attn: Attention | None = None
    w13: Linear | None = None        # the dense SwiGLU's w1 over w3 (num_dense_layers)
    w2: Linear | None = None
    moe: Moe | None = None


@dataclass
class Weights:
    config: Config
    embed: torch.Tensor              # [V, D] bf16, the head too (tied)
    norm: torch.Tensor               # [D] bf16, embedding_norm (the final norm)
    blocks: list[Block]
    extra: dict = field(default_factory=dict)

    @property
    def device(self) -> torch.device:
        return self.embed.device


def stack_fp8(parts: list[tuple[torch.Tensor, torch.Tensor]], names: tuple[str, ...] = ()):
    """FP8 linears of one input, each (e4m3 bytes [N, K], block scales [N / 128, K / 128]), as an ``Fp8BlockLinear``."""

    from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear

    if any(w.shape[0] % 128 for w, _ in parts[:-1]):
        raise ValueError(f"{names}: a stacked FP8 part must be whole 128-row blocks")
    cols = torch.cat([Fp8BlockLinear.column_scales(s, *w.shape) for w, s in parts])
    return Fp8BlockLinear.from_rows(torch.cat([w for w, _ in parts]).view(torch.float8_e4m3fn), cols)


class Loader:
    """A checkpoint's tensors by name, each read once to ``device``; every one must be used."""

    def __init__(self, model_dir: Path, device: str):
        from tensorfold.cuda.direct_read import SafeTensors

        self.c = Config.read(model_dir)
        self.device = device
        self.r = SafeTensors(_shards(model_dir))
        self.used: set[str] = set()
        self.staging = None

    def get(self, name: str) -> torch.Tensor:
        if name not in self.r:
            raise ValueError(f"the checkpoint has no tensor {name}")
        self.used.add(name)
        return self.r.get(name, self.device)

    def stored(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(weight, block scales as fp32) of FP8; (bf16 weight, None) of a bf16 linear."""

        w = self.get(prefix + ".weight")
        if w.dtype != torch.float8_e4m3fn:
            return w.to(torch.bfloat16), None
        if prefix + ".weight_scale" not in self.r:
            raise ValueError(f"{prefix}: an FP8 weight without its weight_scale (compressed-tensors block FP8)")
        return w.view(torch.uint8), self.get(prefix + ".weight_scale").float()

    def linear(self, *prefixes: str) -> Linear:
        """The linears ``prefixes`` stacked by rows (one input), as one ``Linear``."""

        parts = [self.stored(p) for p in prefixes]
        k = parts[0][0].shape[1]
        n = sum(w.shape[0] for w, _ in parts)
        fp8 = [s is not None for _, s in parts]
        if any(fp8) and not all(fp8):
            raise ValueError(f"{prefixes}: a stack mixes FP8 and bf16 weights")
        if self.device == "cpu":
            dense = [w.float() if s is None else dequantize(w, s) for w, s in parts]
            return Linear(n, k, w=torch.cat(dense).contiguous())
        if not all(fp8):
            return Linear(n, k, w=torch.cat([w for w, _ in parts]).contiguous())
        return Linear(n, k, fp8=stack_fp8(parts, prefixes))

    def vector(self, name: str, dtype=torch.bfloat16) -> torch.Tensor:
        return self.get(name).to(dtype).contiguous()

    def experts(self, p: str) -> Experts:
        c = self.c
        w13 = w2 = s13 = s2 = None
        for e in range(c.experts):
            parts = [self.stored(p + f"experts.{e}.{name}") for name in ("w1", "w3", "w2")]
            if w13 is None:
                (g, gs), (u, _), (d, ds) = parts
                w13 = torch.empty((c.experts, g.shape[0] + u.shape[0], g.shape[1]), dtype=g.dtype, device=g.device)
                w2 = torch.empty((c.experts, *d.shape), dtype=d.dtype, device=d.device)
                if gs is not None:
                    s13 = torch.empty((c.experts, 2 * gs.shape[0], gs.shape[1]), dtype=torch.float32,
                                      device=g.device)
                    s2 = torch.empty((c.experts, *ds.shape), dtype=torch.float32, device=g.device)
            (g, gs), (u, us), (d, ds) = parts
            if (gs is None) != (s13 is None):
                raise ValueError(f"{p}experts.{e}: experts of one layer mix FP8 and bf16")
            w13[e, :g.shape[0]], w13[e, g.shape[0]:], w2[e] = g, u, d
            if s13 is not None:
                if g.shape[0] % 128:
                    raise ValueError(f"{p}experts: a width of {g.shape[0]} is not whole 128-row blocks")
                s13[e, :gs.shape[0]], s13[e, gs.shape[0]:], s2[e] = gs, us, ds
        return Experts(w13, w2, s13, s2)

    def block(self, i: int) -> Block:
        c, p = self.c, layer(i)
        b = Block(c.layers[i], self.vector(p + "operator_norm.weight"), self.vector(p + "ffn_norm.weight"))
        if b.kind == "conv":
            taps = self.get(p + "conv.conv.weight")
            b.conv = Conv(self.linear(p + "conv.in_proj"), self.linear(p + "conv.out_proj"),
                          taps.reshape(taps.shape[0], -1).float().contiguous())
        else:
            a = p + "self_attn."
            b.attn = Attention(self.linear(a + "q_proj", a + "k_proj", a + "v_proj"), self.linear(a + "out_proj"),
                               self.vector(a + "q_layernorm.weight"), self.vector(a + "k_layernorm.weight"))
        f = p + "feed_forward."
        if not c.moe(i):
            b.w13, b.w2 = self.linear(f + "w1", f + "w3"), self.linear(f + "w2")
        else:
            b.moe = Moe(self.vector(f + "gate.weight", torch.float32), self.vector(f + "expert_bias", torch.float32),
                        self.experts(f))
        return b

    def load(self, layers: int | None = None) -> Weights:
        c = self.c
        count = len(c.layers) if layers is None else int(layers)
        blocks = []
        for i in range(count):
            blocks.append(self.block(i))
            if self.device != "cpu" and i % 8 == 7:
                torch.cuda.empty_cache()
        embed = self.vector(PREFIX + "embed_tokens.weight")
        w = Weights(c, embed, self.vector(PREFIX + "embedding_norm.weight"), blocks)
        skipped = PREFIX + "layers."
        left = sorted(n for n in self.r.where if n not in self.used
                      and not (layers is not None and n.startswith(skipped) and int(n.split(".")[2]) >= count))
        self.r.close()
        if left:
            raise ValueError(f"unused checkpoint tensors: {left[:5]}")
        if count < len(c.layers):
            from dataclasses import replace

            w.config = replace(c, layers=c.layers[:count])
        return w


def _shards(model_dir: Path) -> list[Path]:
    import json

    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return [model_dir / f for f in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
    return sorted(model_dir.glob("model*.safetensors"))


def load(model_dir: str | Path, device: str = "cuda", *, layers: int | None = None) -> Weights:
    """The checkpoint in ``model_dir`` on ``device`` (its first ``layers`` layers when given, for tests)."""

    return Loader(Path(model_dir), device).load(layers)


def weight_bytes(model_dir: str | Path) -> int:
    """Device bytes of the loaded weights: tensors as stored, block scales widened to fp32 per 64 inputs."""

    from tensorfold.cuda.capacity import headers

    total = 0
    for name, info in headers(model_dir).items():
        n = math.prod(info["shape"])
        if name.endswith(".weight_scale") and ".experts." not in name:
            total += n * 128 * 2 * 4                  # Fp8BlockLinear: an fp32 scale per row and 64 inputs
        else:
            total += n * {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}.get(info["dtype"], 4)
    return total
