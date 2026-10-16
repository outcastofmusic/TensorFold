"""Nemotron 3.5 Bolt weights from the ModelOpt checkpoint: Q8_0 to bf16 at load, NVFP4 Mamba, IQ2_XXS experts packed.

``hf_quant_config.json`` names each quantized linear's format. Q8_0 (attention, latent projections, lm_head) is
dequantized to bf16 here, following llama.cpp's ``dequantize_row_q8_0`` (ggml/src/ggml-quants.c, lines 553-567:
weight j of a 32-weight block is int8 q[j] times the block's fp16 scale, in fp32). The Mamba projections stay NVFP4 on
the fork's row-invariant kernels (``Fp4Linear``). Routed and shared experts stay IQ2_XXS, a layer's routed experts
stacked into one uint8 table [E, N, K // 256 * 66] for ``iq2xxs_grouped``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

Q8_BLOCK, Q8_BYTES = 32, 34
IQ2_BLOCK, IQ2_BYTES = 256, 66
KINDS = {"mamba": "M", "attention": "*", "moe": "E"}


def _number(value: Any) -> float:
    """A config float; ``{"__float__": "Infinity"}`` is how the checkpoint writes inf."""

    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    return float(value)


def text_config(raw: dict) -> dict:
    """The language model's config: ``llm_config`` of the omni checkpoint, or the file itself for a text export."""

    return raw.get("llm_config") or raw


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    pattern: str                     # a character a layer: M Mamba-2, * attention, E latent MoE
    heads: int
    kv_heads: tuple[int, ...]        # by attention layer
    head_dim: int
    m_heads: int
    m_head_dim: int
    m_groups: int
    m_state: int
    conv_kernel: int
    experts: tuple[int, ...]         # by MoE layer
    top_k: tuple[int, ...]           # by MoE layer
    moe_width: tuple[int, ...]       # by MoE layer
    shared_width: int
    latent: int
    scaling: float
    norm_topk: bool
    eps: float
    dt_min: float = 0.0
    dt_max: float = float("inf")
    eos: tuple[int, ...] = (2, 11)

    @property
    def xd(self) -> int:
        return self.m_heads * self.m_head_dim

    @property
    def conv_dim(self) -> int:
        return self.xd + 2 * self.m_groups * self.m_state

    @property
    def proj_dim(self) -> int:
        return self.xd + self.conv_dim + self.m_heads

    @classmethod
    def read(cls, model_dir: str | Path) -> Config:
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        return cls.parse(raw, _eos(Path(model_dir), raw))

    @classmethod
    def parse(cls, raw: dict, eos: tuple[int, ...] = (2, 11)) -> Config:
        t = text_config(raw)
        blocks = t.get("block_configs") or [{"block_type": b} for b in t["layers_block_type"]]
        unknown = sorted({b["block_type"] for b in blocks} - set(KINDS))
        if unknown:
            raise ValueError(f"Nemotron Bolt's CUDA engine runs Mamba, attention and MoE layers, not {unknown}")
        moe = [b for b in blocks if b["block_type"] == "moe"]
        attn = [b for b in blocks if b["block_type"] == "attention"]
        limit = t.get("time_step_limit") or (0.0, float("inf"))
        return cls(
            hidden=int(t["hidden_size"]), vocab=int(t["vocab_size"]),
            pattern="".join(KINDS[b["block_type"]] for b in blocks),
            heads=int(t["num_attention_heads"]),
            kv_heads=tuple(int(b.get("num_key_value_heads") or t["num_key_value_heads"]) for b in attn),
            head_dim=int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]),
            m_heads=int(t["mamba_num_heads"]), m_head_dim=int(t["mamba_head_dim"]), m_groups=int(t["n_groups"]),
            m_state=int(t["ssm_state_size"]), conv_kernel=int(t["conv_kernel"]),
            experts=tuple(int(b.get("n_routed_experts") or t["n_routed_experts"]) for b in moe),
            top_k=tuple(int(b.get("num_experts_per_tok") or t["num_experts_per_tok"]) for b in moe),
            moe_width=tuple(int(b.get("moe_intermediate_size") or t["moe_intermediate_size"]) for b in moe),
            shared_width=int(t["moe_shared_expert_intermediate_size"]) * int(t.get("n_shared_experts") or 1),
            latent=int(t.get("moe_latent_size") or t["hidden_size"]),
            scaling=float(t.get("routed_scaling_factor") or 1.0), norm_topk=bool(t.get("norm_topk_prob", True)),
            eps=float(t.get("layer_norm_epsilon") or t.get("norm_eps") or 1e-5),
            dt_min=_number(limit[0]), dt_max=_number(limit[1]), eos=eos)


def _eos(model_dir: Path, raw: dict) -> tuple[int, ...]:
    """Every end token: generation_config.json's list (2 and <|im_end|> 11 on Bolt), else the configs'."""

    found: list[int] = []
    sources = [raw.get("eos_token_id"), text_config(raw).get("eos_token_id")]
    gen = model_dir / "generation_config.json"
    if gen.is_file():
        sources.insert(0, json.loads(gen.read_text()).get("eos_token_id"))
    for value in sources:
        for t in (value if isinstance(value, list) else [value]):
            if isinstance(t, int) and t not in found:
                found.append(t)
    return tuple(found) or (2,)


# -- formats ------------------------------------------------------------------------------------------------------

def dequantize_q8_0(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K // 32 * 34] -> bf16 [..., K]: q * d in fp32, rounded once."""

    if packed.dtype != torch.uint8 or packed.shape[-1] % Q8_BYTES:
        raise ValueError(f"Q8_0 rows are uint8 in blocks of {Q8_BYTES} bytes, not {packed.dtype} {tuple(packed.shape)}")
    nb = packed.shape[-1] // Q8_BYTES
    b = packed.reshape(-1, nb, Q8_BYTES)
    d = b[..., :2].contiguous().view(torch.float16).float()
    q = b[..., 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(*packed.shape[:-1], nb * Q8_BLOCK).to(torch.bfloat16)


def flat(packed: torch.Tensor, block_bytes: int) -> torch.Tensor:
    """A ggml-packed weight stored [N, K // block, bytes] as [N, K // block * bytes] (the kernels' layout)."""

    if packed.dim() == 3 and packed.shape[-1] == block_bytes:
        return packed.reshape(packed.shape[0], -1)
    if packed.dim() == 2 and packed.shape[-1] % block_bytes == 0:
        return packed
    raise ValueError(f"a ggml weight of {block_bytes}-byte blocks, not {tuple(packed.shape)}")


@dataclass
class Fp4:
    """An NVFP4 projection: ``lin`` (``Fp4Linear``) on CUDA; the stored tensors kept where ``raw`` is asked."""

    n: int
    k: int
    lin: Any = None
    raw: tuple[torch.Tensor, torch.Tensor, float] | None = None   # (codes [N, K/2], e4m3 scales [N, K/16], scale_2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin.prefill(x)

    def dequantized(self) -> torch.Tensor:
        """bf16 [N, K] from the stored tensors (ModelOpt: e2m1 code * e4m3 group scale * scale_2), for references."""

        if self.raw is None:
            raise ValueError("this projection did not keep its stored tensors (load with raw=True)")
        codes, scales, scale2 = self.raw
        table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                             device=codes.device)
        lo, hi = table[(codes & 15).long()], table[(codes >> 4).long()]
        values = torch.stack([lo, hi], dim=-1).reshape(self.n, self.k)        # the low nibble is the even input
        s = scales.float().repeat_interleave(16, dim=1) * scale2
        return (values * s).to(torch.bfloat16)

    def nbytes(self) -> int:
        if self.lin is not None:
            return int(self.lin.nbytes())
        return sum(t.numel() * t.element_size() for t in self.raw[:2]) if self.raw else 0


@dataclass
class Mamba:
    in_proj: Fp4                # (proj_dim, D): [z (XD) | x B C (conv_dim) | dt (H)]
    out_proj: Fp4               # (D, XD)
    conv_w: torch.Tensor        # (KC, conv_dim) fp32: tap k multiplies the input KC-1-k rows back
    conv_b: torch.Tensor        # (conv_dim,) fp32
    a: torch.Tensor             # (H,) fp32, -exp(A_log)
    d: torch.Tensor             # (H,) fp32
    dt_bias: torch.Tensor       # (H,) fp32
    gnorm: torch.Tensor         # (XD,) bf16, the gated RMSNorm's weight (groups of XD / n_groups)


@dataclass
class Attention:
    qkv: torch.Tensor           # (heads * hd + 2 * kv_heads * hd, D) bf16: [q | k | v]
    o: torch.Tensor             # (D, heads * hd) bf16
    kv_heads: int


@dataclass
class MoE:
    router: torch.Tensor        # (E, D) bf16
    bias: torch.Tensor          # (E,) fp32 score correction
    fc1: torch.Tensor           # (L, D) bf16: hidden into the latent space the routed experts read
    fc2: torch.Tensor           # (D, L) bf16
    up: torch.Tensor            # (E, I, L // 256 * 66) uint8 IQ2_XXS
    down: torch.Tensor          # (E, L, I // 256 * 66) uint8 IQ2_XXS
    shared_up: torch.Tensor     # (S, D // 256 * 66) uint8 IQ2_XXS
    shared_down: torch.Tensor   # (D, S // 256 * 66) uint8 IQ2_XXS
    top_k: int

    @property
    def experts(self) -> int:
        return int(self.router.shape[0])


@dataclass
class Block:
    kind: str                   # "M", "*" or "E"
    norm: torch.Tensor          # (D,) bf16, the block's input RMSNorm
    mamba: Mamba | None = None
    attn: Attention | None = None
    moe: MoE | None = None


@dataclass
class Weights:
    config: Config
    embed: torch.Tensor         # (V, D) bf16
    blocks: list[Block]
    norm_f: torch.Tensor        # (D,) bf16
    head: torch.Tensor          # (V, D) bf16
    extra: dict = field(default_factory=dict)


# -- loading ------------------------------------------------------------------------------------------------------

def quant_layers(model_dir: Path) -> dict[str, str]:
    """Linear name (without ``.weight``) -> quant_algo, from hf_quant_config.json or config.json's block."""

    path = model_dir / "hf_quant_config.json"
    if path.is_file():
        block = json.loads(path.read_text()).get("quantization") or {}
    else:
        block = json.loads((model_dir / "config.json").read_text()).get("quantization_config") or {}
    return {name: str(entry.get("quant_algo", "")).upper() for name, entry in (block.get("quantized_layers") or {}).items()}


class _Reader:
    """Tensors by name from every safetensors file in a folder, straight to the device, each read once."""

    def __init__(self, files: list[Path], device: str):
        from tensorfold.cuda.direct_read import SafeTensors

        self.files, self.device = SafeTensors(files), device
        self.used: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def get(self, name: str) -> torch.Tensor:
        self.used.add(name)
        return self.files.get(name, self.device)

    def skip(self, prefix: str) -> None:
        self.used.update(n for n in self.files.keys() if n.startswith(prefix))  # noqa: SIM118  (no __iter__)

    def unused(self) -> list[str]:
        return sorted(set(self.files.keys()) - self.used)

    def close(self) -> None:
        self.files.close()


TEXT_ONLY_SKIPS = ("vision_model.", "mlp1.", "sound_encoder.", "sound_projection.", "radio_model.")


class Loader:
    """One checkpoint's tensors by layer, each linear read in the format hf_quant_config.json names for it."""

    def __init__(self, model_dir: Path, device: str, raw: bool):
        self.dir, self.device, self.raw = model_dir, device, raw
        self.c = Config.read(model_dir)
        self.r = _Reader(sorted(model_dir.glob("model*.safetensors")), device)
        self.root = "language_model." if "language_model.lm_head.weight" in self.r else ""
        self.formats = quant_layers(model_dir)
        self.staging = None

    def name(self, short: str) -> str:
        return self.root + short

    def tensor(self, short: str) -> torch.Tensor:
        return self.r.get(self.name(short)).contiguous()

    def format(self, short: str) -> str:
        return self.formats.get(self.name(short), "BF16")

    def bf16(self, short: str) -> torch.Tensor:
        """A linear as bf16 [N, K]: Q8_0 dequantized, bf16 as stored."""

        kind = self.format(short)
        w = self.tensor(short + ".weight")
        if kind == "Q8_0":
            return dequantize_q8_0(flat(w, Q8_BYTES)).contiguous()
        if kind in ("BF16", "") and w.dtype in (torch.bfloat16, torch.float16, torch.float32):
            return w.to(torch.bfloat16).contiguous()
        raise ValueError(f"{self.name(short)}: Bolt reads this projection as Q8_0 or bf16, not {kind}")

    def iq2(self, short: str) -> torch.Tensor:
        kind = self.format(short)
        if kind != "IQ2_XXS":
            raise ValueError(f"{self.name(short)}: Bolt's experts are IQ2_XXS, not {kind}")
        return flat(self.tensor(short + ".weight"), IQ2_BYTES)

    def fp4(self, short: str) -> Fp4:
        kind = self.format(short)
        if kind != "W4A16_NVFP4":
            raise ValueError(f"{self.name(short)}: Bolt's Mamba projections are W4A16_NVFP4, not {kind}")
        codes = self.tensor(short + ".weight")
        scales = self.tensor(short + ".weight_scale")
        scale2 = float(self.tensor(short + ".weight_scale_2").float().reshape(-1)[0])
        n, k = codes.shape[0], codes.shape[1] * 2
        out = Fp4(n, k, raw=(codes, scales, scale2) if self.raw or self.device == "cpu" else None)
        if self.device != "cpu":
            from tensorfold.cuda.nvfp4.linear import Fp4Linear, Staging

            if self.staging is None:
                self.staging = Staging()          # one e4m3 copy at a time, shared by every NVFP4 projection
            out.lin = Fp4Linear.from_checkpoint(codes, scales, scale2)
            out.lin.staging = self.staging
        return out

    def mamba(self, p: str) -> Mamba:
        conv = self.tensor(p + "conv1d.weight")                 # torch layout (C, 1, K)
        conv = conv[:, 0, :] if conv.dim() == 3 else conv
        bias = self.name(p + "conv1d.bias")
        conv_b = self.r.get(bias).float() if bias in self.r else torch.zeros(conv.shape[0], device=conv.device)
        return Mamba(in_proj=self.fp4(p + "in_proj"), out_proj=self.fp4(p + "out_proj"),
                     conv_w=conv.float().t().contiguous(), conv_b=conv_b.contiguous(),
                     a=(-torch.exp(self.tensor(p + "A_log").float())).contiguous(),
                     d=self.tensor(p + "D").float().contiguous(),
                     dt_bias=self.tensor(p + "dt_bias").float().contiguous(),
                     gnorm=self.tensor(p + "norm.weight").to(torch.bfloat16).contiguous())

    def attention(self, p: str, kv_heads: int) -> Attention:
        qkv = torch.cat([self.bf16(p + name) for name in ("q_proj", "k_proj", "v_proj")]).contiguous()
        return Attention(qkv=qkv, o=self.bf16(p + "o_proj"), kv_heads=kv_heads)

    def moe(self, p: str, experts: int, top_k: int) -> MoE:
        first = self.iq2(p + "experts.0.up_proj")
        up = torch.empty((experts, *first.shape), dtype=torch.uint8, device=first.device)
        up[0] = first
        del first
        down = None
        for e in range(experts):
            if e:
                up[e] = self.iq2(p + f"experts.{e}.up_proj")
            d = self.iq2(p + f"experts.{e}.down_proj")
            if down is None:
                down = torch.empty((experts, *d.shape), dtype=torch.uint8, device=d.device)
            down[e] = d
        return MoE(router=self.tensor(p + "gate.weight").to(torch.bfloat16),
                   bias=self.tensor(p + "gate.e_score_correction_bias").float(),
                   fc1=self.bf16(p + "fc1_latent_proj"), fc2=self.bf16(p + "fc2_latent_proj"), up=up, down=down,
                   shared_up=self.iq2(p + "shared_experts.up_proj"),
                   shared_down=self.iq2(p + "shared_experts.down_proj"), top_k=top_k)

    def load(self, layers: int | None = None) -> Weights:
        c = self.c
        count = len(c.pattern) if layers is None else int(layers)
        blocks: list[Block] = []
        mi = ai = 0
        for i, kind in enumerate(c.pattern[:count]):
            p = f"backbone.layers.{i}."
            norm = self.tensor(p + "norm.weight").to(torch.bfloat16)
            if kind == "M":
                blocks.append(Block("M", norm, mamba=self.mamba(p + "mixer.")))
            elif kind == "*":
                blocks.append(Block("*", norm, attn=self.attention(p + "mixer.", c.kv_heads[ai])))
                ai += 1
            else:
                blocks.append(Block("E", norm, moe=self.moe(p + "mixer.", c.experts[mi], c.top_k[mi])))
                mi += 1
            if self.device != "cpu" and i % 8 == 7:
                torch.cuda.empty_cache()
        if count < len(c.pattern):                 # a truncated model: the later layers are read by nobody
            for i in range(count, len(c.pattern)):
                self.r.skip(self.name(f"backbone.layers.{i}."))
            c = _truncated(c, count)
        w = Weights(config=c, embed=self.tensor("backbone.embeddings.weight").to(torch.bfloat16), blocks=blocks,
                    norm_f=self.tensor("backbone.norm_f.weight").to(torch.bfloat16), head=self.bf16("lm_head"))
        # the omni checkpoint's vision, audio and projector tensors sit outside the language model's root
        left = [n for n in self.r.unused() if n.startswith(self.root) and not n.startswith(TEXT_ONLY_SKIPS)]
        self.r.close()
        if left:
            raise ValueError(f"unused checkpoint tensors: {left[:5]}")
        return w


def _truncated(c: Config, layers: int) -> Config:
    from dataclasses import replace

    pattern = c.pattern[:layers]
    na, nm = pattern.count("*"), pattern.count("E")
    return replace(c, pattern=pattern, kv_heads=c.kv_heads[:na], experts=c.experts[:nm], top_k=c.top_k[:nm],
                   moe_width=c.moe_width[:nm])


def load(model_dir: str | Path, device: str = "cuda", *, raw: bool = False, layers: int | None = None) -> Weights:
    """The language model of the checkpoint in ``model_dir`` (its first ``layers`` layers when given).

    ``raw`` keeps the NVFP4 projections' stored tensors beside their kernels' layout, for references and tests.
    """

    return Loader(Path(model_dir), device, raw).load(layers)


def weight_bytes(model_dir: str | Path, layers: int | None = None) -> int:
    """Device bytes the loaded language model holds: Q8_0 as bf16, the rest as stored (an upper bound for NVFP4)."""

    from tensorfold.cuda.capacity import headers

    model_dir = Path(model_dir)
    formats = quant_layers(model_dir)
    total = 0
    for name, info in headers(model_dir).items():
        if name.startswith(TEXT_ONLY_SKIPS) or (not name.startswith(("language_model.", "backbone.", "lm_head"))):
            continue
        if layers is not None and ".layers." in name and int(name.split(".layers.")[1].split(".")[0]) >= layers:
            continue
        shape, dtype = info["shape"], info["dtype"]
        n = math.prod(shape)
        if formats.get(name.removesuffix(".weight")) == "Q8_0":
            total += n // Q8_BYTES * Q8_BLOCK * 2
        else:
            total += n * (2 if dtype in ("BF16", "F16") else 4 if dtype == "F32" else 1)
    return total
