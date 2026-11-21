"""Hy4-flash's config.json read once, with the reasons a checkpoint is refused (no torch, so ``check`` runs anywhere).

The geometry follows transformers PR #50903 (``configuration_hy_v4_flash.py``): 42 layers of gated MLA with learned
sinks, sparse ("indexed") layers at 0, 1, 5, 9, ..., 41 and 512-key sliding layers between them, identity
hyper-connections over 4 streams, Engram n-gram memories before layers 1, 11, 21 and 31 (``engram_layer_ids`` counts
from 1), a dense MLP in layer 0 and a latent MoE (3,072 -> 1,536, 512 experts, top 16) in the rest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPARSE, SLIDING = "indexed_attention", "sliding_attention"
LEGACY = {"deepseek_sparse_attention": SPARSE, "full_attention": SPARSE, "sliding_attention": SLIDING}


@dataclass(frozen=True)
class Heads:
    """One attention kind's head geometry."""

    heads: int
    nope: int                 # qk_nope_head_dim
    rope: int                 # qk_rope_head_dim
    v: int                    # v_head_dim
    theta: float

    @property
    def qk(self) -> int:
        return self.nope + self.rope


@dataclass(frozen=True)
class Config:
    hidden: int
    vocab: int
    layers: int
    kinds: tuple[str, ...]           # SPARSE or SLIDING, by layer
    sparse: Heads
    sliding: Heads
    window: int                      # a sliding layer's keys: the row and the window - 1 before it
    q_rank: int                      # q_lora_rank
    kv_rank: int                     # kv_lora_rank: the cached latent's normed part
    dense_layers: tuple[int, ...]    # layers with the plain SwiGLU MLP (layer 0)
    intermediate: int                # the dense MLP's width
    experts: int
    top_k: int
    latent: int                      # routed_expert_hidden_size: the experts' input and output width
    moe_width: int                   # moe_intermediate_size: an expert's (and the shared expert's) SwiGLU width
    scaling: float                   # routed_scaling_factor
    swiglu_limit: float
    hc: int                          # hc_mult: hyper-connection streams
    hc_eps: float
    hc_magnitude: float
    eps: float                       # rms_norm_eps
    index_heads: int
    index_dim: int
    index_topk: int                  # keys a sparse layer keeps past this many (in groups of ``index_pool``)
    index_pool: int
    engram_layers: tuple[int, ...]   # zero-based decoder layers whose input gets an Engram memory
    engram_ngram: int                # engram_max_ngram_size: orders 2 .. this
    engram_heads: int                # hash heads an order
    engram_dim: int                  # a hash head's row width
    engram_sizes: tuple[tuple[int, ...], ...]   # per Engram layer: every hash head's table rows, order-major
    engram_seed: int
    pad: int                         # engram_pad_id: an n-gram's slot before the document's start
    eos: tuple[int, ...]
    bos: int
    context: int

    @property
    def sparse_layers(self) -> tuple[int, ...]:
        return tuple(i for i, k in enumerate(self.kinds) if k == SPARSE)

    def heads_of(self, layer: int) -> Heads:
        return self.sparse if self.kinds[layer] == SPARSE else self.sliding

    def moe(self, layer: int) -> bool:
        return layer not in self.dense_layers

    @classmethod
    def read(cls, model_dir: str | Path) -> Config:
        model_dir = Path(model_dir)
        raw = json.loads((model_dir / "config.json").read_text())
        why = refusal(raw)
        if why:
            raise ValueError(f"Hy4-flash cannot run this checkpoint: {why}")
        return cls.parse(raw, eos_tokens(model_dir, raw))

    @classmethod
    def parse(cls, raw: dict[str, Any], eos: tuple[int, ...] | None = None) -> Config:
        t = text(raw)
        n = int(t["num_hidden_layers"])
        kinds = layer_kinds(t)
        rope = t.get("rope_parameters") or {}
        theta = {LEGACY.get(k, k): float(v.get("rope_theta")) for k, v in rope.items() if isinstance(v, dict)}
        sparse = Heads(int(t["num_attention_heads"]), int(t.get("qk_nope_head_dim", 192)),
                       int(t.get("qk_rope_head_dim", 64)), int(t.get("v_head_dim", 256)),
                       theta.get(SPARSE, 1e7))
        sliding = Heads(int(t.get("swa_num_attention_heads", sparse.heads)),
                        int(t.get("swa_qk_nope_head_dim", sparse.nope)), int(t.get("swa_qk_rope_head_dim", sparse.rope)),
                        int(t.get("swa_v_head_dim", sparse.v)), theta.get(SLIDING, 1e4))
        mlp = t.get("mlp_layer_types") or ["dense"] + ["sparse"] * (n - 1)
        engram_ids = tuple(int(i) for i in (t.get("engram_layer_ids") or ()))
        table = t.get("engram_layer_head_vocab_sizes") or {}
        sizes = tuple(tuple(int(s) for order in table[str(i)] for s in order) for i in engram_ids)
        ids = eos if eos is not None else tuple(_ints(t.get("eos_token_id")))
        return cls(
            hidden=int(t["hidden_size"]), vocab=int(t["vocab_size"]), layers=n, kinds=kinds, sparse=sparse,
            sliding=sliding, window=int(t.get("sliding_window", 512)), q_rank=int(t["q_lora_rank"]),
            kv_rank=int(t["kv_lora_rank"]), dense_layers=tuple(i for i, m in enumerate(mlp) if m == "dense"),
            intermediate=int(t["intermediate_size"]), experts=int(t["n_routed_experts"]),
            top_k=int(t["num_experts_per_tok"]), latent=int(t["routed_expert_hidden_size"]),
            moe_width=int(t["moe_intermediate_size"]), scaling=float(t["routed_scaling_factor"]),
            swiglu_limit=float(t.get("swiglu_limit") or 0.0), hc=int(t["hc_mult"]), hc_eps=float(t["hc_eps"]),
            hc_magnitude=float(t.get("hc_magnitude", 2.0)), eps=float(t["rms_norm_eps"]),
            index_heads=int(t.get("compressed_indexer_num_attention_heads", t.get("index_n_heads", 16))),
            index_dim=int(t.get("compressed_indexer_head_dim", t.get("index_head_dim", 128))),
            index_topk=int(t.get("index_topk", 2048)),
            index_pool=int(t.get("compressed_indexer_compress_ratio", t.get("index_kpool", 4))),
            engram_layers=tuple(i - 1 for i in engram_ids), engram_ngram=int(t.get("engram_max_ngram_size", 3)),
            engram_heads=int(t.get("engram_n_head_per_ngram", 8)),
            engram_dim=int(t.get("engram_n_embed_per_ngram", 768)) // int(t.get("engram_n_head_per_ngram", 8)),
            engram_sizes=sizes, engram_seed=int(t.get("engram_seed", 0)), pad=int(t.get("engram_pad_id", 120002)),
            eos=tuple(ids), bos=int(t.get("bos_token_id", 120000)),
            context=int(t.get("max_position_embeddings", 1 << 20)))


def text(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("text_config") or raw


def layer_kinds(t: dict[str, Any]) -> tuple[str, ...]:
    """``layer_types`` with the release's legacy names, or the PR's default pattern (sparse at 0, 1 and 1 + 4k)."""

    n = int(t["num_hidden_layers"])
    named = t.get("layer_types")
    if named:
        return tuple(LEGACY.get(k, k) for k in named)
    return tuple(SPARSE if i < 2 or (i - 1) % 4 == 0 else SLIDING for i in range(n))


def _ints(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, list) else [int(value)]


def eos_tokens(model_dir: Path, raw: dict[str, Any]) -> tuple[int, ...]:
    """The union of config.json's and generation_config.json's end tokens (playbook 2.7), config's first."""

    found = _ints(text(raw).get("eos_token_id")) + _ints(raw.get("eos_token_id"))
    gen = model_dir / "generation_config.json"
    if gen.exists():
        found += _ints(json.loads(gen.read_text()).get("eos_token_id"))
    if not found:
        raise ValueError("no eos_token_id in config.json or generation_config.json")
    return tuple(dict.fromkeys(found))


# Each field the engine computes one way: a checkpoint that sets another value is refused by name.
FIXED = {
    "hc_mult": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "qk_nope_head_dim": 192, "v_head_dim": 256,
    "moe_router_use_sigmoid": True, "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "gated_mla": True,
    "learnable_sink": True, "sandwich_norm": True, "use_mla": True, "attention_bias": False, "hidden_act": "silu",
    "tie_word_embeddings": False, "engram_max_ngram_size": 3, "compressed_indexer_compress_ratio": 4,
    "compressed_indexer_skip_fine_select": True, "gating_type": "elementwise",
}


def refusal(raw: dict[str, Any]) -> str | None:
    """Why the CUDA engine cannot run a config, from config.json alone; None when it can."""

    if raw.get("model_type") != "hy_v4_flash":
        return f"model_type {raw.get('model_type')!r}: the engine runs hy_v4_flash"
    t = text(raw)
    for key, want in FIXED.items():
        if key in t and t[key] != want:
            return f"{key} {t[key]!r}: the engine computes {want!r} only"
    try:
        kinds = layer_kinds(t)
    except (KeyError, TypeError, ValueError) as e:
        return f"layer_types unreadable ({e})"
    if set(kinds) - {SPARSE, SLIDING}:
        return f"layer_types {sorted(set(kinds) - {SPARSE, SLIDING})}: the engine runs sparse and sliding MLA"
    for k in ("indexer_types",):
        if any(v != "full" for v in t.get(k) or ()):
            return f"{k} with shared indexers: the engine runs one indexer a sparse layer"
    if (t.get("engram_layer_ids") or ()) and not t.get("engram_layer_head_vocab_sizes"):
        return "Engram layers without engram_layer_head_vocab_sizes: the hash tables' sizes are unknown"
    for kind, rope in (t.get("rope_parameters") or {}).items():
        if isinstance(rope, dict) and rope.get("rope_type", "default") != "default":
            return f"rope_type {rope['rope_type']!r} on {kind}: the engine rotates by plain rope_theta only"
    mlp = t.get("mlp_layer_types") or []
    if mlp and set(mlp) - {"dense", "sparse"}:
        return f"mlp_layer_types {sorted(set(mlp) - {'dense', 'sparse'})}"
    return None
