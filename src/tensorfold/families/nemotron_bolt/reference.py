"""A slow fp32 PyTorch forward of Nemotron 3.5 Bolt's language model, straight from the 24GB checkpoint.

Each layer's weights are decoded to fp32 when the layer runs and dropped after it, so the reference never holds more
than one layer: the routed experts alone would be about 132 GB in bf16. Every decode is exact (see ``quant``) and every
matmul runs in fp32, so the reference's gap to a bf16 engine is the engine's own rounding.

The math follows the checkpoint's modeling_nemotron_h.py: a block is ``x + mixer(rmsnorm(x))``; Mamba-2 runs as a
plain sequential scan with dt clamped to the config's time_step_limit, as the checkpoint's CUDA path and vLLM clamp it
((0, inf) on Bolt; its torch fallback clamps dt at time_step_min = 1e-3, see ``Config.dt_floor``); attention is GQA without positional encoding;
MoE routes on sigmoid scores plus the correction bias, weights the top 9 by their plain scores normalized and times
routed_scaling_factor, runs relu2 experts inside the 1024-wide latent (fc1_latent_proj in, fc2_latent_proj out) and
adds the shared relu2 expert on the block's input.

A ``State`` carries the conv windows, SSM states and attention keys between calls, so a prompt fed whole and the same
prompt fed a token at a time give the same logits up to fp32 summation order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from . import quant
from .cuda.weights import config_float

PREFIX = "language_model."
LAYERS = PREFIX + "backbone.layers."


@dataclass(frozen=True)
class Block:
    kind: str                     # "mamba", "attention" or "moe"
    experts: int = 0              # routed experts (moe)
    width: int = 0                # a routed expert's intermediate size (moe)
    top_k: int = 0                # experts a token (moe)


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    blocks: tuple[Block, ...]
    heads: int
    kv_heads: int
    head_dim: int
    m_heads: int
    m_head_dim: int
    m_groups: int
    m_state: int
    conv_kernel: int
    latent: int
    shared_width: int
    scaling: float
    norm_topk: bool
    eps: float
    eos: tuple[int, ...]
    dt_floor: float = 0.0         # time_step_limit[0]; 1e-3 reproduces the HF torch fallback's clamp
    dt_ceil: float = float("inf")  # time_step_limit[1]

    @property
    def xd(self) -> int:
        return self.m_heads * self.m_head_dim

    @property
    def conv_dim(self) -> int:
        return self.xd + 2 * self.m_groups * self.m_state

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> Config:
        model_dir = Path(model_dir)
        raw = json.loads((model_dir / "config.json").read_text())
        c = raw.get("llm_config", raw)
        blocks = tuple(Block(b["block_type"], b.get("n_routed_experts", 0), b.get("moe_intermediate_size", 0),
                             b.get("num_experts_per_tok", 0)) for b in c["block_configs"])
        eos = c["eos_token_id"]
        gen = model_dir / "generation_config.json"
        if gen.is_file():
            eos = json.loads(gen.read_text()).get("eos_token_id", eos)
        limit = c.get("time_step_limit") or (0.0, float("inf"))     # the checkpoint's CUDA path and vLLM clamp here
        return cls(dt_floor=config_float(limit[0]), dt_ceil=config_float(limit[1]),hidden=c["hidden_size"], vocab=c["vocab_size"], blocks=blocks, heads=c["num_attention_heads"],
                   kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"], m_heads=c["mamba_num_heads"],
                   m_head_dim=c["mamba_head_dim"], m_groups=c["n_groups"], m_state=c["ssm_state_size"],
                   conv_kernel=c["conv_kernel"], latent=c["moe_latent_size"],
                   shared_width=c["moe_shared_expert_intermediate_size"] * c.get("n_shared_experts", 1),
                   scaling=float(c["routed_scaling_factor"]), norm_topk=bool(c["norm_topk_prob"]),
                   eps=float(c["layer_norm_epsilon"]), eos=tuple(eos if isinstance(eos, list) else [eos]))


class Checkpoint:
    """The checkpoint's tensors by name, read on demand and decoded to fp32 on ``device`` by their quant recipe."""

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu"):
        from safetensors import safe_open

        self.dir = Path(model_dir)
        self.device = torch.device(device)
        self.config = Config.from_dir(self.dir)
        self._where = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        recipes = json.loads((self.dir / "hf_quant_config.json").read_text())["quantization"]["quantized_layers"]
        self._algo = {name: r["quant_algo"] for name, r in recipes.items()}
        self._files = {f: safe_open(str(self.dir / f), framework="pt") for f in sorted(set(self._where.values()))}

    def recipe(self, module: str) -> str | None:
        """``module``'s quant_algo in hf_quant_config.json, or None for a bf16 / fp32 tensor."""

        return self._algo.get(module)

    def raw(self, name: str) -> torch.Tensor:
        return self._files[self._where[name]].get_tensor(name)

    def rows(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """Rows ``ids`` of a bf16 table (the embeddings) as fp32 on the device."""

        table = self._files[self._where[name]].get_slice(name)
        return torch.stack([table[int(i)] for i in ids.tolist()]).to(self.device).float()

    def linear(self, module: str) -> torch.Tensor:
        """``module``'s weight [N, K] as exact fp32 on the device, decoded by its hf_quant_config recipe."""

        algo = self.recipe(module)
        w = self.raw(module + ".weight").to(self.device)
        if algo is None:
            return w.float()
        if algo == "Q8_0":
            return quant.dequantize_q8_0(w.flatten(-2), dtype=torch.float32)
        if algo == "IQ2_XXS":
            return quant.dequantize_iq2xxs(w.flatten(-2), dtype=torch.float32)
        if algo == "W4A16_NVFP4":
            scale = self.raw(module + ".weight_scale").to(self.device)
            scale2 = self.raw(module + ".weight_scale_2")
            return quant.dequantize_nvfp4(w, scale, scale2, dtype=torch.float32)
        raise ValueError(f"{module}: no decoder for {algo}")

    def vector(self, name: str) -> torch.Tensor:
        return self.raw(name).to(self.device).float()


@dataclass
class State:
    """What a sequence carries between calls: per layer, a conv window and SSM state (Mamba) or keys and values."""

    length: int = 0
    conv: dict[int, torch.Tensor] = field(default_factory=dict)     # [conv_kernel - 1, conv_dim]
    ssm: dict[int, torch.Tensor] = field(default_factory=dict)      # [m_heads, m_head_dim, m_state]
    keys: dict[int, torch.Tensor] = field(default_factory=dict)     # [length, kv_heads, head_dim]
    values: dict[int, torch.Tensor] = field(default_factory=dict)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def mamba(ck: Checkpoint, i: int, x: torch.Tensor, st: State) -> torch.Tensor:
    c, p, T = ck.config, f"{LAYERS}{i}.mixer.", x.shape[0]
    proj = x @ ck.linear(p + "in_proj").T
    z, xbc, dt = proj[:, :c.xd], proj[:, c.xd:c.xd + c.conv_dim], proj[:, c.xd + c.conv_dim:]
    k = c.conv_kernel
    window = st.conv.get(i, torch.zeros((k - 1, c.conv_dim), device=x.device))
    padded = torch.cat([window, xbc])
    st.conv[i] = padded[-(k - 1):]
    cw = ck.vector(p + "conv1d.weight")[:, 0, :]                            # [conv_dim, k]
    conv = ck.vector(p + "conv1d.bias") + sum(cw[:, j] * padded[j:j + T] for j in range(k))
    conv = F.silu(conv)
    gs = c.m_groups * c.m_state
    xs = conv[:, :c.xd].reshape(T, c.m_heads, c.m_head_dim)
    B = conv[:, c.xd:c.xd + gs].reshape(T, c.m_groups, c.m_state).repeat_interleave(c.m_heads // c.m_groups, 1)
    C = conv[:, c.xd + gs:].reshape(T, c.m_groups, c.m_state).repeat_interleave(c.m_heads // c.m_groups, 1)
    dt = F.softplus(dt + ck.vector(p + "dt_bias")).clamp(c.dt_floor, c.dt_ceil)  # [T, H]
    A = -torch.exp(ck.vector(p + "A_log"))
    D = ck.vector(p + "D")
    h = st.ssm.get(i, torch.zeros((c.m_heads, c.m_head_dim, c.m_state), device=x.device))
    ys = []
    for t in range(T):
        h = h * torch.exp(dt[t] * A)[:, None, None] + (dt[t][:, None] * xs[t])[:, :, None] * B[t][:, None, :]
        ys.append((h * C[t][:, None, :]).sum(-1) + D[:, None] * xs[t])
    st.ssm[i] = h
    y = torch.stack(ys).reshape(T, c.xd) * F.silu(z)
    y = rms(y.reshape(T, c.m_groups, -1), None, c.eps).reshape(T, c.xd) * ck.vector(p + "norm.weight")
    return y @ ck.linear(p + "out_proj").T


def attention(ck: Checkpoint, i: int, x: torch.Tensor, st: State) -> torch.Tensor:
    c, p, T = ck.config, f"{LAYERS}{i}.mixer.", x.shape[0]
    q = (x @ ck.linear(p + "q_proj").T).reshape(T, c.heads, c.head_dim)
    k = (x @ ck.linear(p + "k_proj").T).reshape(T, c.kv_heads, c.head_dim)
    v = (x @ ck.linear(p + "v_proj").T).reshape(T, c.kv_heads, c.head_dim)
    past = st.keys[i].shape[0] if i in st.keys else 0
    if past:
        k, v = torch.cat([st.keys[i], k]), torch.cat([st.values[i], v])
    st.keys[i], st.values[i] = k, v
    rep = c.heads // c.kv_heads
    kf, vf = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("thd,lhd->htl", q, kf) * c.head_dim ** -0.5
    future = torch.arange(k.shape[0], device=x.device)[None, :] > (past + torch.arange(T, device=x.device))[:, None]
    s = s.masked_fill(future, float("-inf"))
    o = torch.einsum("htl,lhd->thd", torch.softmax(s, -1), vf).reshape(T, c.heads * c.head_dim)
    return o @ ck.linear(p + "o_proj").T


def route(logits: torch.Tensor, bias: torch.Tensor, top_k: int, scaling: float, norm: bool):
    """Top-k experts by sigmoid score + bias (ties to the lower expert id, as the engine's router picks); weights are
    the chosen plain scores, normalized, times ``scaling``."""

    score = torch.sigmoid(logits)
    idx = torch.sort(score + bias, dim=-1, descending=True, stable=True).indices[:, :top_k]
    w = torch.gather(score, 1, idx)
    if norm:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return idx, w * scaling


def moe(ck: Checkpoint, i: int, x: torch.Tensor) -> torch.Tensor:
    c, b, p = ck.config, ck.config.blocks[i], f"{LAYERS}{i}.mixer."
    idx, wts = route(x @ ck.vector(p + "gate.weight").T, ck.vector(p + "gate.e_score_correction_bias"),
                     b.top_k, c.scaling, c.norm_topk)
    lat = x @ ck.linear(p + "fc1_latent_proj").T                           # [T, latent]
    out = torch.zeros_like(lat)
    for e in torch.unique(idx).tolist():
        rows, slot = (idx == e).nonzero(as_tuple=True)
        act = relu2(lat[rows] @ ck.linear(f"{p}experts.{e}.up_proj").T)
        out.index_add_(0, rows, wts[rows, slot][:, None] * (act @ ck.linear(f"{p}experts.{e}.down_proj").T))
    shared = relu2(x @ ck.linear(p + "shared_experts.up_proj").T) @ ck.linear(p + "shared_experts.down_proj").T
    return out @ ck.linear(p + "fc2_latent_proj").T + shared


def layer(ck: Checkpoint, i: int, x: torch.Tensor, st: State) -> torch.Tensor:
    """Block i on the residual stream x [T, hidden]: x + mixer(rmsnorm(x))."""

    h = rms(x, ck.vector(f"{LAYERS}{i}.norm.weight"), ck.config.eps)
    kind = ck.config.blocks[i].kind
    if kind == "mamba":
        return x + mamba(ck, i, h, st)
    if kind == "attention":
        return x + attention(ck, i, h, st)
    if kind == "moe":
        return x + moe(ck, i, h)
    raise ValueError(f"layer {i}: unknown block type {kind!r}")


def embed(ck: Checkpoint, tokens: torch.Tensor) -> torch.Tensor:
    return ck.rows(PREFIX + "backbone.embeddings.weight", tokens)


@torch.no_grad()
def hidden(ck: Checkpoint, tokens: torch.Tensor, st: State, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, dict]:
    """Feed ``tokens`` after ``st``; returns the final normed hidden states [T, hidden] and, for each layer in
    ``taps``, the residual stream after that layer [T, hidden]."""

    x = embed(ck, tokens)
    tapped = {}
    for i in range(len(ck.config.blocks)):
        x = layer(ck, i, x, st)
        if i in taps:
            tapped[i] = x.clone()
    st.length += tokens.numel()
    return rms(x, ck.vector(PREFIX + "backbone.norm_f.weight"), ck.config.eps), tapped


@torch.no_grad()
def logits(ck: Checkpoint, h: torch.Tensor, rows_per_slice: int = 16384) -> torch.Tensor:
    """The lm_head over normed hidden rows h [R, hidden] -> fp32 [R, vocab], decoding the Q8_0 head a slice at a time."""

    head = ck.raw(PREFIX + "lm_head.weight")
    parts = []
    for s0 in range(0, head.shape[0], rows_per_slice):
        w = quant.dequantize_q8_0(head[s0:s0 + rows_per_slice].to(ck.device).flatten(-2), dtype=torch.float32)
        parts.append(h @ w.T)
    return torch.cat(parts, dim=1)


@torch.no_grad()
def forward(ck: Checkpoint, tokens: torch.Tensor, st: State | None = None, last: int | None = None) -> torch.Tensor:
    """Logits fp32 [T, vocab] for ``tokens`` fed after ``st`` (a fresh sequence when None); ``last`` keeps only the
    final ``last`` rows' logits."""

    h, _ = hidden(ck, tokens, st if st is not None else State())
    return logits(ck, h if last is None else h[-last:])


@torch.no_grad()
def greedy(ck: Checkpoint, prompt: torch.Tensor, steps: int, *, last: int = 1, stop_eos: bool = False):
    """Greedy decode: returns (tokens [steps], prompt logits of the last ``last`` positions [last, vocab], the logits
    each token was chosen from [steps, vocab]). Row 0 of the step logits is the prompt's last row."""

    st = State()
    prompt_logits = forward(ck, prompt, st, last=last)
    row = prompt_logits[-1:]
    tokens, rows = [], []
    for _ in range(steps):
        tok = int(row[0].argmax())
        tokens.append(tok)
        rows.append(row[0])
        if stop_eos and tok in ck.config.eos:
            break
        if len(tokens) == steps:
            break
        row = forward(ck, torch.tensor([tok]), st)
    return torch.tensor(tokens), prompt_logits, torch.stack(rows)
