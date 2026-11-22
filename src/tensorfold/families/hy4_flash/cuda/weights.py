"""A Hy4-flash checkpoint as the forward reads it: projections on the fork's kernels on CUDA, fp32 on the CPU.

Every format the NVFP4 pack (Avifenesh/Hy4-flash-NVFP4) and tencent/Hy4-flash-FP8 store has a reader here:

* block FP8 (``weight`` e4m3 + ``weight_scale_inv`` per 128x128) on the lane matmul (``Fp8BlockLinear``);
* bf16 on ``b16`` (one warp an output for decode rows, the bf16 mma for prompt chunks);
* NVFP4 routed experts (``weight`` codes, e4m3 ``weight_scale`` per 16, fp32 ``weight_scale_2`` per expert) on the
  grouped NVFP4 kernel (``tensorfold.cuda.nvfp4.experts``), and bf16 or FP8-block routed experts quantized to NVFP4 at
  load the same way (the tests' tiny checkpoints).

On the CPU every weight decodes to fp32 (the tests' tiny checkpoints only). The Engram tables stay in their files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import SPARSE, Config
from ..engram import Hasher, RowReader

# calls of up to this many rows run the decode kernels, prompt chunks the prompt kernels. TF_HY4_DECODE_ROWS=32 lets a
# copy window grow to 32 rows: every decode kernel keeps a row's bits at any width up to it
DECODE_ROWS = int(__import__("os").environ.get("TF_HY4_DECODE_ROWS", "16"))
if not 16 <= DECODE_ROWS <= 32:
    raise ValueError(f"TF_HY4_DECODE_ROWS={DECODE_ROWS}: 16 to 32")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# --------------------------------------------------------------------------------------------- linears ---

class CpuLin:
    """fp32 weight [N, K] (and bias): bf16 rows in, bf16 rows out, an fp32 sum (the reference's rounding)."""

    def __init__(self, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.w, self.bias = w.float(), None if bias is None else bias.float()
        self.n, self.k = self.w.shape

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        y = x.float() @ self.w.t()
        if self.bias is not None:
            y = y.to(torch.bfloat16).float() + self.bias
        return y.to(torch.bfloat16)

    def nbytes(self) -> int:
        return 0


class B16Lin:
    """A bf16 projection on ``b16``: decode rows one warp an output, prompt chunks on the bf16 mma; a bias added in
    bf16 after the matmul, as ``nn.Linear`` adds it."""

    def __init__(self, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.w = w.to(torch.bfloat16).contiguous()
        self.bias = None if bias is None else bias.to(torch.bfloat16).contiguous()
        self.n, self.k = self.w.shape

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        from tensorfold.families.qwen3_5.cuda import b16

        y = b16.prompt(x, self.w) if prompt and x.shape[0] > DECODE_ROWS else b16.matmul(x, self.w)
        return y if self.bias is None else (y.float() + self.bias.float()).to(torch.bfloat16)

    def nbytes(self) -> int:
        return self.w.numel() * 2


class F8Lin:
    """A 128x128-block FP8 projection on the lane matmul (``Fp8BlockLinear``): the stored bytes exactly, fp32 sums."""

    def __init__(self, lin) -> None:
        self.lin = lin
        self.n, self.k = lin.n, lin.k

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        return self.lin.prefill(x) if prompt and x.shape[0] > DECODE_ROWS else self.lin(x)

    def nbytes(self) -> int:
        return self.lin.nbytes()


class F4Lin:
    """An NVFP4 projection on ``Fp4Linear`` (or a block FP8 one on ``Fp8BlockLinear``): decode rows on the lane matmul,
    prompt chunks on its exact prompt GEMM; a bias added in bf16 after the matmul, as ``nn.Linear`` adds it."""

    def __init__(self, lin, bias: torch.Tensor | None = None) -> None:
        self.lin = lin
        self.bias = None if bias is None else bias.to(torch.bfloat16).contiguous()
        self.n, self.k = lin.n, lin.k

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        y = self.lin.prefill(x) if prompt and x.shape[0] > DECODE_ROWS else self.lin(x)
        return y if self.bias is None else (y.float() + self.bias.float()).to(torch.bfloat16)

    def nbytes(self) -> int:
        return self.lin.nbytes()


def pair(a, b, x: torch.Tensor, prompt: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a(x, prompt), b(x, prompt))``: two bf16 projections of the same rows in one launch (``b16``'s pair kernels,
    each its single kernel's bits), any other pair one call each. TF_HY4_PAIR=0: one call each."""

    if PAIR and isinstance(a, B16Lin) and isinstance(b, B16Lin) and a.bias is None and b.bias is None and x.is_cuda:
        from tensorfold.families.qwen3_5.cuda import b16

        if prompt and x.shape[0] > DECODE_ROWS:
            ya, yb = b16.prompt_pair(x, a.w, b.w)
        else:
            ya, yb = b16.matmul_pair(x, a.w, b.w)
        return ya, yb
    return a(x, prompt), b(x, prompt)


PAIR = __import__("os").environ.get("TF_HY4_PAIR", "1") != "0"


# ---------------------------------------------------------------------------------------------- layers ---

@dataclass
class Indexer:
    wq: Any                    # q_resid [1024] -> 16 heads x 128
    wh: Any                    # x -> 16 head weights
    wk: Any                    # x -> the token key [128]
    k_norm: torch.Tensor       # token_key_norm, bf16 [128]
    pv: torch.Tensor           # key_value_projection fp32 [128, 128]
    pg: torch.Tensor           # key_gate_projection fp32 [128, 128]
    ape: torch.Tensor          # intra_group_ape fp32 [4, 128]
    g_norm: torch.Tensor       # key_norm fp32 [128]


@dataclass
class Attention:
    kind: str
    heads: int
    q_a: Any
    q_norm: torch.Tensor
    q_b: Any
    kv_a: Any
    kv_norm: torch.Tensor
    wk: Any                    # kv_b's key rows per head [H, nope, 512] bf16 or ``kernels.RowF8`` (absorbed into the query)
    wv: Any                    # kv_b's value rows per head [H, v, 512] bf16 or ``kernels.RowF8``
    gate: Any
    o: Any
    post_norm: torch.Tensor
    sinks: torch.Tensor        # fp32 [H]
    indexer: Indexer | None = None


@dataclass
class Dense:
    gate: Any
    up: Any
    down: Any
    post_norm: torch.Tensor    # post_fc2_norm: the dense layer's sandwich norm


@dataclass
class MoE:
    router: torch.Tensor       # bf16 [E, D]
    bias: torch.Tensor         # e_score_correction_bias fp32 [E]
    pre: Any                   # routed_expert_down_proj: D -> latent
    post: Any                  # routed_expert_up_proj: latent -> D
    experts: Any               # nvfp4 Experts4 on CUDA; fp32 (gate, up, down) [E, N, K] on the CPU
    s_gate: Any
    s_up: Any
    s_down: Any
    combine_norm: torch.Tensor


@dataclass
class Engram:
    hasher: Hasher
    table: Any                 # RowReader, or an in-memory fp32 [rows, dim] tensor (tests)
    head_norms: torch.Tensor   # fp32 [heads, dim]
    value: Any
    keys: list
    norm_key: torch.Tensor     # [S, D]
    norm_query: torch.Tensor   # [S, D]
    pre_gamma: torch.Tensor    # [D]
    hc_scales: torch.Tensor    # [S, D]


@dataclass
class HC:
    fn: torch.Tensor           # fp32 [2S, S * D] (the head's: [S, S * D])
    base: torch.Tensor         # fp32
    scale: torch.Tensor        # fp32 [2] (the head's: [1])


@dataclass
class Layer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor    # post_attention_layernorm
    attn_hc: HC
    ffn_hc: HC
    attn: Attention
    mlp: Any                   # MoE or Dense
    engram: Engram | None = None


@dataclass
class Weights:
    config: Config
    embed: torch.Tensor        # bf16 [V, D]
    layers: list[Layer]
    norm: torch.Tensor
    head: Any                  # bf16 [V, D], or an NVFP4 ``kernels.Head4`` (computed in fp32 either way)
    hc_head: HC
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def nbytes(self) -> int:
        total = self.embed.numel() * 2
        total += self.head.numel() * 2 if isinstance(self.head, torch.Tensor) else self.head.nbytes()
        for layer in self.layers:
            for part in (layer.attn, layer.mlp, layer.engram):
                if part is None:
                    continue
                for v in vars(part).values():
                    if isinstance(v, torch.Tensor):
                        total += v.numel() * v.element_size()
                    elif hasattr(v, "nbytes") and callable(v.nbytes):
                        total += v.nbytes()
        return total


# ------------------------------------------------------------------------------------------------ load ---

def _nvfp4_dense(words: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor, dtype=torch.float32,
                 chunk: int = 16) -> torch.Tensor:
    """NVFP4 codes [.., N, K/2] (low nibble first), e4m3 scales [.., N, K/16], fp32 scale2 [..] -> [.., N, K] in
    ``dtype`` (bf16 holds every value exactly), ``chunk`` experts at a time with int32 codes: a whole layer's int64
    codes and fp32 products took about 35 GB."""

    lut = torch.tensor(E2M1, dtype=torch.float32, device=words.device)
    if words.dim() == 2:
        return _nvfp4_dense(words[None], scales[None], scale2.reshape(1), dtype, chunk)[0]
    out = torch.empty((*words.shape[:-1], words.shape[-1] * 2), dtype=dtype, device=words.device)
    for e0 in range(0, words.shape[0], chunk):
        w = words[e0:e0 + chunk].to(torch.int32)
        v = lut[torch.stack([w & 0xF, w >> 4], dim=-1).flatten(-2)]
        v *= scales[e0:e0 + chunk].view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
        v *= scale2.reshape(-1)[e0:e0 + chunk].float().reshape(-1, 1, 1)
        out[e0:e0 + chunk] = v.to(dtype)
    return out


def _fp8_block_dense(w: torch.Tensor, s: torch.Tensor, block: int = 128) -> torch.Tensor:
    n, k = w.shape[-2:]
    sf = s.float().repeat_interleave(block, -2)[..., :n, :].repeat_interleave(block, -1)[..., :k]
    return w.float() * sf


class _Reader:
    """Checkpoint tensors by name, each read once; ``left`` names what nothing read."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from tensorfold.families.qwen3_5.cuda.weights import _Tensors

        skip = (lambda n: n.startswith(("vit.", "model.vit.", "model.dfly_layers.", "mtp."))
                or "engram_layer.multi_head_embedding" in n)
        self.t = _Tensors(model_dir, device, skip=skip)

    def __contains__(self, name: str) -> bool:
        return name in self.t

    def get(self, name: str) -> torch.Tensor:
        return self.t.pop(name)

    def left(self) -> list[str]:
        return list(self.t)

    def close(self) -> None:
        self.t.close()


class _Trunk:
    """NVFP4 replacements for the checkpoint's projections (``tools/hy4_trunk.py`` writes them from the BF16 release):
    ``<name>.weight`` codes, ``.weight_scale`` e4m3 and ``.weight_scale_2`` fp32 by the checkpoint's own names."""

    def __init__(self, folder: str | None, device: str, only: str | None = None, skip: str | None = None) -> None:
        import os
        import re

        from tensorfold.cuda.direct_read import SafeTensors

        self.device = device
        # TF_HY4_TRUNK_ONLY / TF_HY4_TRUNK_SKIP: regular expressions over projection names (``model.layers.7.
        # self_attn.o_proj``, ``lm_head``) that take a folder's replacement, or keep the checkpoint's: one class of
        # a trunk folder at a time (a sensitivity window), or a folder's layout less its sensitive tensors
        only = os.environ.get("TF_HY4_TRUNK_ONLY") if only is None else only
        skip = os.environ.get("TF_HY4_TRUNK_SKIP") if skip is None else skip
        self.only = re.compile(only) if only else None
        self.skip = re.compile(skip) if skip else None
        files = sorted(Path(folder).glob("*.safetensors")) if folder else []
        if folder and not files:
            raise ValueError(f"TF_HY4_TRUNK={folder}: no safetensors files there")
        self.t = SafeTensors(files) if files else None

    def taken(self, name: str) -> bool:
        """Whether the filters let projection ``name`` take the folder's replacement."""

        return ((self.only is None or self.only.search(name) is not None)
                and (self.skip is None or self.skip.search(name) is None))

    def __contains__(self, name: str) -> bool:
        return self.t is not None and name + ".weight" in self.t and self.taken(name)

    def fp8(self, name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(e4m3 [N, K], fp32 block scales) when ``name`` is stored in block FP8, else None."""

        if name + ".weight_scale_inv" not in self.t:
            return None
        return self.t.get(name + ".weight", self.device), self.t.get(name + ".weight_scale_inv", self.device)

    def rows(self, name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(e4m3 [N, K], fp32 scale a row [N]) when ``name`` is stored in row-scaled FP8 (kv_b), else None."""

        if self.t is None or name + ".weight_scale_row" not in self.t or not self.taken(name):
            return None
        return self.t.get(name + ".weight", self.device), self.t.get(name + ".weight_scale_row", self.device)

    def get(self, name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(codes uint8 [N, K/2], e4m3 scales [N, K/16], the global scale) of NVFP4 projection ``name``."""

        g = self.t.get
        return (g(name + ".weight", self.device).view(torch.uint8), g(name + ".weight_scale", self.device),
                float(g(name + ".weight_scale_2").float().reshape(-1)[0]))

    def close(self) -> None:
        if self.t is not None:
            self.t.close()


def load(model_dir: str | Path, device: str = "cuda", *, layers: int | None = None,
         trunk: str | None = None) -> Weights:
    """The checkpoint's text model on ``device`` (``layers``: the first so many, for tests and probes). ``trunk``
    (default ``TF_HY4_TRUNK``) names a folder of NVFP4 projections that replace the checkpoint's bf16 or FP8 ones."""

    import os

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    cuda = torch.device(device).type == "cuda"
    r = _Reader(model_dir, device)
    q = _Trunk(trunk if trunk is not None else os.environ.get("TF_HY4_TRUNK") or None, device)
    try:
        w = _build(model_dir, cfg, r, device, cuda, layers, q)
        if layers is None:
            left = [n for n in r.left() if not n.startswith("model.layers.") or int(n.split(".")[2]) < cfg.layers]
            if left:
                raise ValueError(f"unused checkpoint tensors: {left[:6]} ...")
    finally:
        r.close()
        q.close()
    if cuda:
        torch.cuda.empty_cache()
    return w


def _build(model_dir: Path, cfg: Config, r: _Reader, device: str, cuda: bool, count: int | None,
           trunk: _Trunk) -> Weights:
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8BlockLinear

    from .kernels import Head4, RowF8

    def bf16(name: str) -> torch.Tensor:
        return r.get(name).to(torch.bfloat16).contiguous()

    def f32(name: str) -> torch.Tensor:
        return r.get(name).float().contiguous()

    def lin(name: str, bias: bool = False):
        """A projection in whatever form the checkpoint stores it."""

        b = bf16(name + ".bias") if bias else None
        w = r.get(name + ".weight")
        if name in trunk:                  # the checkpoint's copy is read off its list and dropped
            if name + ".weight_scale_inv" in r:
                r.get(name + ".weight_scale_inv")
            f8 = trunk.fp8(name)
            if f8 is not None:
                return CpuLin(_fp8_block_dense(*f8), b) if not cuda else F4Lin(Fp8BlockLinear.from_checkpoint(*f8), b)
            codes, scales, scale2 = trunk.get(name)
            if not cuda:
                return CpuLin(_nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2)), b)
            return F4Lin(Fp4Linear.from_checkpoint(codes, scales.view(torch.uint8), scale2), b)
        if w.dtype == torch.float8_e4m3fn:
            s = r.get(name + ".weight_scale_inv")
            if not cuda:
                return CpuLin(_fp8_block_dense(w, s), b)
            return F8Lin(Fp8BlockLinear.from_checkpoint(w, s))
        if w.dtype == torch.uint8:
            dense = _nvfp4_dense(w, r.get(name + ".weight_scale"), r.get(name + ".weight_scale_2"))
            return CpuLin(dense, b) if not cuda else B16Lin(dense.to(torch.bfloat16), b)
        return CpuLin(w, b) if not cuda else B16Lin(w, b)

    def hc(prefix: str, head: bool = False) -> HC:
        if head:
            return HC(f32(prefix + "hc_head_fn"), f32(prefix + "hc_head_base"), f32(prefix + "hc_head_scale"))
        return HC(f32(prefix + "hc_fn"), f32(prefix + "hc_base"), f32(prefix + "hc_scale"))

    n = cfg.layers if count is None else min(count, cfg.layers)
    out: list[Layer] = []
    lookup = None
    for i in range(n):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        heads = cfg.heads_of(i)
        kv_b = bf16(a + "kv_b_proj.weight").view(heads.heads, heads.nope + heads.v, cfg.kv_rank)
        wk, wv = kv_b[:, :heads.nope].contiguous(), kv_b[:, heads.nope:].contiguous()
        f8 = trunk.rows(a + "kv_b_proj")
        if f8 is not None:                # e4m3 rows with a scale each, read by absorb and expand as stored
            del kv_b, wk, wv
            codes = f8[0].view(torch.uint8).view(heads.heads, heads.nope + heads.v, cfg.kv_rank)
            sc = f8[1].float().view(heads.heads, heads.nope + heads.v)
            wk, wv = RowF8(codes[:, :heads.nope], sc[:, :heads.nope]), RowF8(codes[:, heads.nope:], sc[:, heads.nope:])
        idx = None
        if cfg.kinds[i] == SPARSE:
            ci = a + "compressed_indexer."
            idx = Indexer(wq=lin(ci + "query_projection"), wh=lin(ci + "head_weight_projection"),
                          wk=lin(ci + "token_key_projection"), k_norm=bf16(ci + "token_key_norm.weight"),
                          pv=f32(ci + "key_value_projection.weight"), pg=f32(ci + "key_gate_projection.weight"),
                          ape=f32(ci + "intra_group_ape"), g_norm=f32(ci + "key_norm.weight"))
        attn = Attention(kind=cfg.kinds[i], heads=heads.heads, q_a=lin(a + "q_a_proj"),
                         q_norm=bf16(a + "q_a_layernorm.weight"), q_b=lin(a + "q_b_proj"),
                         kv_a=lin(a + "kv_a_proj_with_mqa"), kv_norm=bf16(a + "kv_a_layernorm.weight"),
                         wk=wk, wv=wv, gate=lin(a + "linear_gate"), o=lin(a + "o_proj"), post_norm=bf16(a + "post_proj_norm.weight"),
                         sinks=f32(a + "learnable_sink_param"), indexer=idx)
        m = p + "mlp."
        if cfg.moe(i):
            mlp = MoE(router=bf16(m + "gate.weight"), bias=f32(m + "gate.e_score_correction_bias"),
                      pre=lin(m + "routed_expert_down_proj"), post=lin(m + "routed_expert_up_proj"),
                      experts=_experts(r, m + "experts.", cfg, cuda), s_gate=lin(m + "shared_experts.gate_proj"),
                      s_up=lin(m + "shared_experts.up_proj"), s_down=lin(m + "shared_experts.down_proj"),
                      combine_norm=bf16(m + "combine_output_norm.weight"))
        else:
            mlp = Dense(gate=lin(m + "gate_proj"), up=lin(m + "up_proj"), down=lin(m + "down_proj"),
                        post_norm=bf16(m + "post_fc2_norm.weight"))
        eg = None
        if i in cfg.engram_layers:
            e = p + "engram_layer."
            j = cfg.engram_layers.index(i)
            lookup = r.get(e + "lookup_table").cpu().numpy()
            hasher = Hasher(lookup, cfg.engram_sizes[j], seed=cfg.engram_seed, layer_id=i + 1,
                            ngram=cfg.engram_ngram, heads=cfg.engram_heads, pad=cfg.pad)
            heads_n = len(cfg.engram_sizes[j])
            eg = Engram(hasher=hasher, table=_table(model_dir, e + "multi_head_embedding.embedding", hasher.rows,
                                                    cfg.engram_dim),
                        head_norms=torch.stack([f32(e + f"lookup_per_head_norms.{h}.weight") for h in range(heads_n)]),
                        value=lin(e + "value_proj", bias=True),
                        keys=[lin(e + f"key_projs.{s}", bias=True) for s in range(cfg.hc)],
                        norm_key=torch.stack([bf16(e + f"norm_key.{s}.weight") for s in range(cfg.hc)]),
                        norm_query=torch.stack([bf16(e + f"norm_query.{s}.weight") for s in range(cfg.hc)]),
                        pre_gamma=bf16(e + "unified_pre_gamma"),
                        hc_scales=torch.stack([bf16(e + f"unified_per_hc_scales.{s}") for s in range(cfg.hc)]))
        out.append(Layer(input_norm=bf16(p + "input_layernorm.weight"),
                         post_norm=bf16(p + "post_attention_layernorm.weight"),
                         attn_hc=hc(p + "hc_attn_layer.hc_pre."), ffn_hc=hc(p + "hc_mlp_layer.hc_pre."),
                         attn=attn, mlp=mlp, engram=eg))
    head = bf16("lm_head.weight")
    if "lm_head" in trunk:
        del head
        head = Head4(*trunk.get("lm_head"))
    return Weights(cfg, bf16("model.embed_tokens.weight"), out, bf16("model.norm.weight"), head,
                   hc("model.hc_head.", head=True), torch.device(device))


def _experts(r: _Reader, prefix: str, cfg: Config, cuda: bool):
    """A layer's routed experts: NVFP4 as stored, or bf16 / block FP8 ``gate_up_proj`` split and quantized to NVFP4
    by ModelOpt's recipe on CUDA; fp32 (gate, up, down) [E, N, K] on the CPU."""

    from tensorfold.cuda.nvfp4 import experts as nvx

    def stored(name: str):
        """(codes, e4m3 scale bytes, per-expert scales) of an NVFP4 tensor, or its dense fp32 form."""

        if prefix + name + ".weight_scale_2" in r:
            return (r.get(prefix + name + ".weight"), r.get(prefix + name + ".weight_scale").view(torch.uint8),
                    r.get(prefix + name + ".weight_scale_2").float())
        return None

    gate, up, down = stored("gate_proj"), stored("up_proj"), stored("down_proj")
    if gate is None:
        gu = r.get(prefix + "gate_up_proj")
        if gu.dtype == torch.float8_e4m3fn:
            gu = _fp8_block_dense(gu, r.get(prefix + "gate_up_proj_scale_inv"))
        dn = r.get(prefix + "down_proj")
        if dn.dtype == torch.float8_e4m3fn:
            dn = _fp8_block_dense(dn, r.get(prefix + "down_proj_scale_inv"))
        half = gu.shape[1] // 2
        if not cuda:
            return (gu[:, :half].to(torch.bfloat16), gu[:, half:].to(torch.bfloat16), dn.to(torch.bfloat16))
        gate, up, down = (nvx.quantize(t.to(torch.bfloat16)) for t in (gu[:, :half], gu[:, half:], dn))
    if not cuda:
        return tuple(_nvfp4_dense(*t, dtype=torch.bfloat16) for t in (gate, up, down))
    return nvx.make(gate, up, down, limit=cfg.swiglu_limit)


def _table(model_dir: Path, name: str, rows: int, dim: int):
    """The Engram table named ``name`` read from its file, refused unless it holds the config's row count."""

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    if name + ".weight" not in index:
        raise ValueError(f"no Engram table {name} in model.safetensors.index.json")
    return RowReader(model_dir / index[name + ".weight"], name, rows, dim)


def gather_engram(table, rows: np.ndarray, dim: int) -> torch.Tensor:
    """Rows [T, heads] -> fp32 [T, heads, dim] (FP8 times its row scale, or bf16), on the host."""

    from ..engram import decode_rows

    if isinstance(table, torch.Tensor):
        return table[torch.from_numpy(rows)].float()
    codes, scales = table.gather(rows)
    return torch.from_numpy(decode_rows(codes, scales, dim)).view(*rows.shape, dim)
