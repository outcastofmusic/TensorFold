"""Liquid AI's LFM2.5-24B-A2B (model_type ``lfm2_moe``) on CUDA: gated short convs, GQA and 64 sigmoid-routed experts.

The weights are under the LFM Open License v1.0: commercial use is licensed only to an entity under US$10M annual
revenue (docs/recipes/lfm2.5-24b-a2b.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_TYPES = ("lfm2_moe",)
TITLE = "LFM2-MoE"
MODELS = ("LiquidAI/LFM2.5-24B-A2B-FP8", "LiquidAI/LFM2.5-24B-A2B")
QUANT_METHODS = {"cuda": ("compressed-tensors", None)}   # 128x128 block FP8, or the bf16 export


def refusal(model_dir: str | Path) -> str | None:
    """Why the CUDA engine cannot read a checkpoint, from config.json alone; None when it can."""

    from .config import refusal as config_refusal

    return config_refusal(json.loads((Path(model_dir) / "config.json").read_text()))


def check(model_dir: str | Path) -> None:
    why = refusal(model_dir)
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, context: int | None = None,
                context_explicit: bool = False, **options: Any):
    """One GPU, one stream, no drafts; a serve option it has no use for is refused."""

    if int(tp) != 1 or int(rank) != 0 or master:
        raise ValueError(f"{TITLE} runs on one GPU on CUDA: serve with --tp 1")
    if options.pop("vision", False):
        raise ValueError(f"{TITLE} is a text model")
    if drafter:
        raise ValueError(f"{TITLE} has no draft model on CUDA: drop --drafter")
    if int(options.pop("parallel", None) or 1) > 1:
        raise ValueError(f"{TITLE} serves one stream on CUDA: requests take turns (drop --parallel)")
    if options:
        raise ValueError(f"{TITLE} takes no {', '.join(sorted(options))} on CUDA: one stream, no drafts, a bf16 KV "
                         "cache")
    check(model_dir)
    from .cuda.app import Lfm2MoeEngine

    return Lfm2MoeEngine(Path(model_dir), context=context, context_explicit=bool(context_explicit))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so discovery never loads the CUDA server
        from .cuda.app import Lfm2MoeApp

        return Lfm2MoeApp
    raise AttributeError(name)
