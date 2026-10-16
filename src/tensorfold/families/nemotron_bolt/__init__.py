"""Nemotron 3.5 Bolt 74B-A9B (latent MoE Nemotron-H, ModelOpt mixed ggml/NVFP4 weights) on CUDA, text only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# the omni checkpoint's top-level type; a text-only export carries the language model's own
MODEL_TYPES = ("nemotron_h_omni", "nemotron_h_puzzle")
TITLE = "Nemotron 3.5 Bolt"
MODELS = ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",)
DRAFTER = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"   # used where it has been pulled (--drafter auto)
QUANT_METHODS = {"cuda": ("modelopt",)}
# the ModelOpt block holds ggml-packed Q8_0 and IQ2_XXS groups beside NVFP4: ``check`` reads it, not the NVFP4 reader
MODELOPT_GGML = True

FORMATS = {"Q8_0", "IQ2_XXS", "W4A16_NVFP4"}   # what the loader reads; everything unlisted is bf16 or fp32


def refusal(model_dir: str | Path) -> str | None:
    """Why the CUDA engine cannot read a checkpoint, from its JSON files alone; None when it can."""

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("llm_config") or config
    if text.get("model_type") != "nemotron_h_puzzle":
        return f"its language model is {text.get('model_type')!r}; this engine reads nemotron_h_puzzle (Bolt)"
    if text.get("moe_latent_size") is None:
        return "its MoE layers have no latent projections (moe_latent_size)"
    if text.get("mtp_block_configs") or int(text.get("num_nextn_predict_layers") or 0):
        return "it has MTP layers, which this engine does not read yet"
    from .cuda.weights import quant_layers

    formats = set(quant_layers(model_dir).values())
    if not formats:
        return "it names no quantized layers (hf_quant_config.json): the engine reads Bolt's 24 GB ModelOpt export"
    odd = sorted(formats - FORMATS)
    if odd:
        return f"it stores layers as {', '.join(odd)}; the engine reads {', '.join(sorted(FORMATS))} and bf16"
    return None


def check(model_dir: str | Path) -> None:
    why = refusal(model_dir)
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, no_drafts: bool = False,
                mtp_drafts: int | None = None, context: int | None = None, **options: Any):
    """One GPU; LiLiCorr drafts with ``drafter`` (its folder), serial rounds without one."""

    if tp != 1 or rank != 0:
        raise ValueError(f"{TITLE} runs on one GPU on CUDA")
    if options.get("vision"):
        raise ValueError(f"{TITLE} is served text only on CUDA")
    from .cuda import DRAFTS
    from .cuda.app import BoltEngine

    return BoltEngine(Path(model_dir), drafter="" if no_drafts else drafter,
                      drafts=DRAFTS if mtp_drafts is None else int(mtp_drafts), context=context,
                      context_explicit=bool(options.get("context_explicit")))
