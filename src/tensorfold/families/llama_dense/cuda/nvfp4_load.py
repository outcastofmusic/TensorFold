"""ModelOpt and compressed-tensors exports of a Llama checkpoint: each projection NVFP4, FP8 or bf16 by its tensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorfold.families.qwen3_5.cuda.nvfp4_load import SUFFIXES, maths


def load_nvfp4(model_dir: str | Path, device: str = "cuda"):
    """NVFP4 and FP8 projections at full precision or in the checkpoint's own math (``--precision``), bf16 as stored."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.nvfp4 import format as fmt
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear, Staging
    from tensorfold.families.qwen3_5.cuda.weights import Plain, _Tensors

    from .weights import Config, build, skipped

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    block = fmt.config_block(json.loads((model_dir / "config.json").read_text())) or {}
    reciprocal = str(block.get("quant_method", "")).lower() == "compressed-tensors"
    info = {n: (i["dtype"], i["shape"]) for n, i in headers(model_dir).items() if not skipped(n)}
    t = _Tensors(model_dir, device, skip=skipped)
    staging = Staging()
    own, line = maths()

    def act_scale(name: str, got: dict, kind: str) -> float | None:
        if not own[kind]:
            return None
        if "input_scale" in got:
            return float(got["input_scale"].float().reshape(-1)[0])
        if "input_global_scale" in got:
            return 1.0 / float(got["input_global_scale"].float().reshape(-1)[0])
        raise ValueError(f"{name}: no static input scale, so its own math is unknown; serve it with --precision full")

    def linear(name: str):
        parts = {s: info[f"{name}.{s}"] for s in SUFFIXES if f"{name}.{s}" in info}
        if not parts:
            raise ValueError(f"the checkpoint has no projection {name}")
        kind = fmt.scheme(parts)
        got = {s: t.pop(f"{name}.{s}") for s in parts}
        weight = got["weight"] if "weight" in got else got["weight_packed"]      # ``scheme`` found one of them
        if kind == "nvfp4":
            stored = got.get("weight_global_scale" if reciprocal else "weight_scale_2")
            if stored is None:
                raise ValueError(f"{name}: an NVFP4 weight without its global scale")
            g = float(stored.float().reshape(-1)[0])
            lin = Fp4Linear.from_checkpoint(weight, got["weight_scale"], 1.0 / g if reciprocal else g,
                                            act=act_scale(name, got, "nvfp4"))
            lin.staging = staging
            return lin
        if kind == "fp8":
            s = got["weight_scale"].float().reshape(-1)
            if s.numel() != 1:
                raise ValueError(f"{name}: FP8 with {s.numel()} scales; the CUDA engine reads one scale a tensor")
            return Fp8Linear.from_checkpoint(weight, float(s[0]), act=act_scale(name, got, "fp8"))
        if kind == "bf16":
            return Plain(weight.to(torch.bfloat16).contiguous())
        raise ValueError(f"{name}: {kind} projections are not read by the Llama engine")

    def tensor(name: str) -> torch.Tensor:
        return t.pop(name).contiguous()

    w = build(cfg, linear, tensor, device, "nvfp4")
    w.embed = Plain(w.embed.weight.to(torch.bfloat16))
    w.precision = "checkpoint" if any(own.values()) else "full"
    print(f"[tensorfold] precision: {line}", flush=True)
    left = list(t)
    t.close()
    if left:
        raise ValueError(f"unused checkpoint tensors: {left[:5]} ...")
    torch.cuda.empty_cache()
    return w
