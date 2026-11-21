"""Tencent's Hy4-flash (model_type ``hy_v4_flash``) on CUDA: gated MLA with sinks over sparse (indexed) and sliding
layers, identity hyper-connections, Engram n-gram memories read from disk, and a latent MoE of 512 experts.

Tested checkpoint: Avifenesh/Hy4-flash-NVFP4 (NVFP4 routed experts, block-FP8 attention projections, FP8 Engram tables
with a scale a row). tencent/Hy4-flash-FP8 reads too but its 148.6 GB of FP8 experts do not fit one GB10. The weights
are under the Tencent Hy License 1.0 (docs/recipes/hy4-flash.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_TYPES = ("hy_v4_flash",)
TITLE = "Hy4-flash"
MODELS = ("Avifenesh/Hy4-flash-NVFP4",)
QUANT_METHODS = {"cuda": ("modelopt", "fp8", None)}
CONTEXT = 131072              # the default window; the model reads 1M, the cache sizes to what memory allows


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
    """One GPU, one stream; a serve option it has no use for is refused."""

    if int(tp) != 1 or int(rank) != 0 or master:
        raise ValueError(f"{TITLE} runs on one GPU on CUDA: serve with --tp 1")
    if options.pop("vision", False):
        raise ValueError(f"{TITLE} serves text on CUDA: drop --vision")
    if drafter:
        raise ValueError(f"{TITLE} drafts with its own DFly2 layers, not a --drafter folder")
    if int(options.pop("parallel", None) or 1) > 1:
        raise ValueError(f"{TITLE} serves one stream on CUDA: requests take turns (drop --parallel)")
    if options:
        raise ValueError(f"{TITLE} takes no {', '.join(sorted(options))} on CUDA")
    check(model_dir)
    from .cuda.app import Hy4Engine

    return Hy4Engine(Path(model_dir), context=context, context_explicit=bool(context_explicit),
                     drafts=not no_drafts)


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so discovery never loads the CUDA server
        from .cuda.app import Hy4App

        return Hy4App
    raise AttributeError(name)
