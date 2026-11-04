"""Plain dense Llama checkpoints (``LlamaForCausalLM``) on CUDA: one GPU, one stream, context-copy drafts verified as
chains and, with ``--drafter``, DFlash2 trees; every drafted token the serial decode's."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_TYPES = ("llama",)
TITLE = "Llama dense"
MODELS = ("Nanbeige/Nanbeige4.3-33B",)
# bf16 as released; an EXL3 pack (exllamav3's convert) or a ModelOpt / compressed-tensors NVFP4 / FP8 export
QUANT_METHODS = {"cuda": ("exl3", "modelopt", "compressed-tensors", None)}
EXL3_VARIANT = "any"                           # every EXL3 codebook and width (tensorfold.families.EXL3_VARIANT_ANY)

HEAD_DIMS = (128, 256)                         # the verify window's tree attention
MAX_GROUP = 16                                 # query heads a KV head: tree attention's query tile


def refusal(config: dict[str, Any]) -> str | None:
    """Why the CUDA engine cannot run a config, from config.json alone; None when it can."""

    t = config.get("text_config") or config
    arch = config.get("architectures") or ["LlamaForCausalLM"]
    if arch != ["LlamaForCausalLM"]:
        return f"architectures {arch}: the engine runs LlamaForCausalLM"
    if t.get("rope_scaling"):
        return f"rope_scaling {t['rope_scaling']}: the engine rotates by plain rope_theta only"
    if (t.get("rope_parameters") or {}).get("rope_type", "default") != "default":
        return f"rope_parameters {t['rope_parameters']}: the engine rotates by plain rope_theta only"
    if t.get("attention_bias") or t.get("mlp_bias"):
        return "attention or MLP biases: the engine's projections have none"
    if str(t.get("hidden_act", "silu")) != "silu":
        return f"hidden_act {t['hidden_act']!r}: the engine's MLP is SwiGLU (silu)"
    if t.get("tie_word_embeddings"):
        return "a head tied to the embedding: the engine reads a separate lm_head"
    hidden, heads, kv = int(t["hidden_size"]), int(t["num_attention_heads"]), int(t.get("num_key_value_heads") or
                                                                                    t["num_attention_heads"])
    head_dim = int(t.get("head_dim") or hidden // heads)
    if head_dim not in HEAD_DIMS:
        return f"head_dim {head_dim}: verify attention takes {' or '.join(map(str, HEAD_DIMS))}"
    if heads % kv or heads // kv > MAX_GROUP:
        return f"{heads} query heads over {kv} KV heads: a KV head serves at most {MAX_GROUP}, evenly"
    if hidden % 64 or int(t["intermediate_size"]) % 64:
        return "hidden and intermediate sizes must be multiples of 64 (the row kernels' groups)"
    return None


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone, before any weight loads or downloads."""

    why = refusal(json.loads((Path(model_dir) / "config.json").read_text()))
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Tested: {', '.join(MODELS)}")


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, context: int | None = None,
                context_explicit: bool = False, **options: Any):
    """One GPU, one stream; copies from the context draft unless ``no_drafts``, and a DFlash2 ``drafter`` (a folder
    such as ProCreations/Nanbeige4.3-33B-DFlash2) adds trees. Unused serve options are refused."""

    if int(tp) != 1 or int(rank) != 0 or master:
        raise ValueError(f"{TITLE} runs on one GPU on CUDA: serve with --tp 1")
    if options.pop("vision", False):
        raise ValueError(f"{TITLE} is a text model: drop --vision")
    if int(options.pop("parallel", None) or 1) > 1:
        raise ValueError(f"{TITLE} serves one stream on CUDA: requests take turns (drop --parallel)")
    if options:
        raise ValueError(f"{TITLE} takes no {', '.join(sorted(options))} on CUDA")
    check(model_dir)
    from .cuda.engine import LlamaEngine

    draft = Path(drafter) if drafter and not no_drafts else None
    if draft is not None and not (draft / "config.json").exists():
        raise ValueError(f"--drafter {drafter}: no config.json there (a DFlash2 folder, e.g. "
                         "ProCreations/Nanbeige4.3-33B-DFlash2)")
    return LlamaEngine(Path(model_dir), draft_dir=draft, allow_copy=not no_drafts, context=context,
                       context_explicit=bool(context_explicit))
