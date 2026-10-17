"""LFM2-VL decision models (Liquid AI's d1-3B) on CUDA, in plain torch.

A d1 checkpoint answers typed questions about a state in one forward pass with zero output tokens: ``POST
/v1/systemone`` takes llama-server's System One request (``state``, ``questions`` keyed by id, pictures) and
``/v1/decisions`` the SGLang-style one every CUDA engine here serves. It writes no text, so chat and completions are
refused. The prompt format and the readout follow Liquid AI's own runner (``prompt.py``, ``hybrid.py`` in the
checkpoint) and plotarmordev/d1-tensorfold's Python 0.6 recipe (Apache 2.0), which this family ports without its
Triton kernels and CUDA graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("lfm2_vl",)
TITLE = "LFM2-VL (d1 decision model)"
LANES = False        # nothing decodes: a request is one forward over its state and its questions
MODELS = ("LiquidAI/d1-3B",)
QUANT_METHODS = {"cuda": (None,)}     # the checkpoint's own bf16 weights
# tokens one forward reads at most (TENSORFOLD_LFM2_TOKEN_BUDGET lowers it; --context caps it)
TOKEN_BUDGET = 16384


def read_text_config(model_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The checkpoint's config and its text config, refused from JSON alone when the forward does not read it."""

    from tensorfold.families import OWN_MODEL_HELP, describe_quantization, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) is not None:
        raise ValueError(f"LFM2 d1 runs the checkpoint's own bf16 weights ({MODELS[0]}); this checkpoint stores "
                         f"{describe_quantization(config)}. {OWN_MODEL_HELP}")
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    heads = int(text.get("num_attention_heads") or 0)
    kinds = {str(kind) for kind in (text.get("layer_types") or ())}
    bad = [what for what, odd in (
        ("a dtype other than bfloat16", str(text.get("dtype") or config.get("dtype") or "bfloat16") != "bfloat16"),
        ("no layer_types", not kinds),
        ("a layer that is neither conv nor full_attention", bool(kinds - {"conv", "full_attention"})),
        ("a convolution bias", bool(text.get("conv_bias"))),
        ("an attention bias", bool(text.get("attention_bias"))),
        ("no SwiGLU block", text.get("block_use_swiglu") is False),
        ("no short convolution (conv_L_cache)", int(text.get("conv_L_cache") or 0) < 2),
        ("no grouped attention", not 1 <= int(text.get("num_key_value_heads") or 0) <= heads
         or heads % int(text.get("num_key_value_heads") or 1)),
        ("a hidden size that is not a whole number of heads", not heads or int(text.get("hidden_size") or 0) % heads),
        ("no vocabulary", int(text.get("vocab_size") or 0) < 1),
        ("no MLP width (intermediate_size)", int(text.get("intermediate_size") or 0) < 1),
        ("no normalization epsilon", not (text.get("norm_eps") or text.get("block_norm_eps"))),
    ) if odd]
    if bad:
        raise ValueError(f"TensorFold's LFM2 forward covers Liquid AI's d1 decision checkpoints ({MODELS[0]}); this "
                         f"one has {', '.join(bad)}. {OWN_MODEL_HELP}")
    return config, text


def check(model_dir: str | Path) -> None:
    """Refuse a checkpoint the forward does not read, from config.json alone (``pull``, ``info`` and ``serve``)."""

    read_text_config(model_dir)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                no_drafts: bool = False, context: int | None = None, context_explicit: bool | None = None,
                master_port: int = 29551, **options: Any):
    """The decision engine on one GPU: one forward a request, ``--context`` bounding a forward's tokens."""

    import os

    if int(tp) != 1 or int(rank) != 0 or master:
        raise ValueError("LFM2 d1 runs on one GPU: serve with --tp 1")
    if drafter:
        raise ValueError("LFM2 d1 decodes nothing, so --drafter does not apply")
    if options.pop("vision", False):
        raise ValueError("LFM2 d1 reads pictures through its own tower without --vision: drop the flag")
    for flag in ("vision_urls", "vision_offload"):
        options.pop(flag, None)
    if options:
        raise ValueError(f"LFM2 d1 takes no {', '.join(sorted(options))}: a request is one forward (no --parallel "
                         "streams, no drafts, no KV cache)")
    budget = os.environ.get("TENSORFOLD_LFM2_TOKEN_BUDGET", "").strip()
    if budget and (not budget.isdecimal() or int(budget) < 1):
        raise ValueError(f"TENSORFOLD_LFM2_TOKEN_BUDGET={budget!r}: a positive token count")
    check(model_dir)
    from .engine import Engine

    return Engine(Path(model_dir), context=context, context_explicit=context_explicit,
                  token_budget=int(budget) if budget else TOKEN_BUDGET)


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .app import DecisionApp

        return DecisionApp
    raise AttributeError(name)
