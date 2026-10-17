"""LFM2-MoE checkpoint tensors by name, and the exact decode of compressed-tensors' 128x128 block FP8.

compressed-tensors writes a block-quantized linear as ``weight`` (e4m3 [N, K]) and ``weight_scale`` (bf16
[ceil(N / 128), K / 128]); the weight is ``e4m3 * scale``, the block's scale as stored (llm-compressor sets it to the
block's absmax / 448). DeepSeek and ModelOpt's FP8_PB_WO store the same multiplier in fp32 as ``weight_scale_inv``.
The bf16 checkpoint stores the same names with bf16 weights and no scales.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import BLOCK

PREFIX = "model."


def layer(i: int) -> str:
    return f"{PREFIX}layers.{i}."


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 [N, K] and its block scales [ceil(N / 128), K / 128] -> fp32 [N, K]; e4m3 x bf16 is exact in fp32."""

    n, k = weight.shape
    bn, bk = BLOCK
    if tuple(scale.shape) != (-(-n // bn), -(-k // bk)):
        raise ValueError(f"weight_scale {tuple(scale.shape)} does not tile a [{n}, {k}] weight in {bn}x{bk} blocks")
    s = scale.float().repeat_interleave(bn, dim=0)[:n].repeat_interleave(bk, dim=1)[:, :k]
    return weight.view(torch.float8_e4m3fn).float() * s


class Tensors:
    """Every safetensors shard of a folder, read on the CPU by name (memory-mapped)."""

    def __init__(self, model_dir: str | Path):
        from safetensors import safe_open

        model_dir = Path(model_dir)
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            files = sorted({model_dir / f for f in json.loads(index.read_text())["weight_map"].values()})
        else:
            files = sorted(model_dir.glob("model*.safetensors"))
        if not files:
            raise ValueError(f"{model_dir} holds no safetensors shards")
        self.where: dict[str, object] = {}
        self.files = [safe_open(str(f), framework="pt") for f in files]
        for f in self.files:
            for name in f.keys():  # noqa: SIM118  (safe_open has no __iter__)
                self.where[name] = f

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> torch.Tensor:
        if name not in self.where:
            raise KeyError(f"the checkpoint has no tensor {name}")
        return self.where[name].get_tensor(name)

    def stored(self, prefix: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A linear as stored: (weight, block scales) for FP8, (weight, None) for bf16."""

        w = self.get(prefix + ".weight")
        if w.dtype != torch.float8_e4m3fn:
            return w, None
        if prefix + ".weight_scale" not in self:
            raise ValueError(f"{prefix}: an FP8 weight without its weight_scale (compressed-tensors block FP8)")
        return w.view(torch.uint8), self.get(prefix + ".weight_scale")

    def linear(self, prefix: str) -> torch.Tensor:
        """A linear's weight [N, K] in fp32, FP8 decoded exactly."""

        w, scale = self.stored(prefix)
        return w.float() if scale is None else dequantize(w, scale)

    def vector(self, name: str) -> torch.Tensor:
        return self.get(name).float()
