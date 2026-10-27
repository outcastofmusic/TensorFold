"""An EXL3 pack of a Llama checkpoint (exllamav3's convert): trellis groups on the shared row-invariant linear, the
embedding, norms and any unquantized head as stored."""

from __future__ import annotations

from pathlib import Path

import torch


def load_exl3(model_dir: str | Path, device: str = "cuda"):
    from tensorfold.cuda.exl3 import format as fmt
    from tensorfold.cuda.exl3.prefill import Workspace
    from tensorfold.families.qwen3_5.cuda.exl3_load import _files, _read, _read_groups, _where
    from tensorfold.families.qwen3_5.cuda.weights import Plain

    from .weights import Config, build, skipped

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    ckpt = fmt.scan(model_dir, read_markers=False)
    if ckpt.bad:
        raise ValueError(f"unreadable EXL3 groups: {list(ckpt.bad.items())[:3]}")
    if not ckpt.groups:
        raise ValueError(f"{model_dir} has no EXL3 groups")
    where = _where(model_dir)
    files = _files(model_dir, where)
    plain = _read(files, where, [n for n in ckpt.plain if not skipped(n)], device)
    groups = _read_groups(files, ckpt.groups, device, Workspace())
    files.close()                                    # the reader's pinned staging goes back to the system

    def tensor(name: str) -> torch.Tensor:
        if name not in plain:
            raise ValueError(f"the checkpoint has no plain tensor {name}")
        return plain.pop(name).contiguous()

    def linear(name: str):
        if name in groups:
            return groups.pop(name)
        return Plain(tensor(name + ".weight").to(torch.bfloat16))      # a head left unquantized

    w = build(cfg, linear, tensor, device, "exl3")
    w.embed = Plain(w.embed.weight.to(torch.bfloat16))
    if plain or groups:
        raise ValueError(f"unused checkpoint tensors: {sorted(plain)[:3] + sorted(groups)[:3]}")
    return w
