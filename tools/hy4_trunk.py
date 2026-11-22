"""NVFP4 copies of Hy4-flash's trunk projections, quantized from the BF16 release one tensor at a time.

    python tools/hy4_trunk.py BF16_MODEL OUT_DIR [--classes attn,shared,latent,head] [--search 1,0.95,...]

The NVFP4 pack keeps the trunk as Tencent shipped it: linear_gate, o_proj and q_b in block FP8, the rest in bf16 (9.4
GB a decoded token). This writes NVFP4 codes, e4m3 scales per 16 inputs and an fp32 global scale for the chosen
classes, by the checkpoint's own names, one file a layer (``trunk-L07.safetensors``, ``trunk-head.safetensors``).
The engine reads the folder with ``TF_HY4_TRUNK=OUT_DIR`` and replaces those projections. Each tensor is read alone
from its shard, so the process holds one tensor and its codes at a time (the head's 742 MB at most).

A block's scale is the one of ``--search``'s factors times amax / 6 whose rounded codes give the smallest squared
error (data-free MSE scales); ``--search 1`` is the plain amax rule. ``quant.json`` records each tensor's relative
squared error.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from itertools import pairwise
from pathlib import Path

import torch

GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CLASSES = {
    "attn": r"self_attn\.(linear_gate|o_proj|q_b_proj)$",
    "shared": r"mlp\.shared_experts\.(gate|up|down)_proj$",
    "latent": r"mlp\.routed_expert_(down|up)_proj$",
    "head": r"^lm_head$",
    "kvb": r"self_attn\.kv_b_proj$",
    "qa": r"self_attn\.(q_a_proj|kv_a_proj_with_mqa)$",
    "dense": r"layers\.0\.mlp\.(gate|up|down)_proj$",
    "engram": r"engram_layer\.(value_proj|key_projs\.\d+)$",
}


def quantize(w: torch.Tensor, search: tuple[float, ...] = (1.0,), rows: int = 4096):
    """bf16 [N, K] -> (codes uint8 [N, K/2] low nibble first, e4m3 scales [N, K/16], fp32 global scale, relative
    squared error). Each 16-wide block takes the factor of ``search`` whose codes reconstruct it best."""

    n, k = w.shape
    amax = float(w.float().abs().max())
    scale2 = amax / (448.0 * 6.0) if amax > 0 else 1.0
    mids = torch.tensor([(a + b) / 2 for a, b in pairwise(GRID)])
    grid = torch.tensor(GRID)
    codes = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn)
    err = total = 0.0
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float().view(-1, k // 16, 16)
        top = x.abs().amax(-1)
        best_err = best_s = best_q = None
        for f in search:
            s = (top * f / 6.0 / scale2).clamp(max=448.0).to(torch.float8_e4m3fn)
            step = s.float()[..., None] * scale2
            y = torch.where(step > 0, x / step.clamp_min(1e-30), torch.zeros_like(x))
            q = torch.bucketize(y.abs().clamp(max=6.0), mids)
            e = ((grid[q] * y.sign() * step - x) ** 2).sum(-1)
            if best_err is None:
                best_err, best_s, best_q = e, s, torch.where(y < 0, q | 8, q)
            else:
                take = e < best_err
                best_err = torch.where(take, e, best_err)
                best_s = torch.where(take, s.float(), best_s.float()).to(torch.float8_e4m3fn)
                best_q = torch.where(take[..., None], torch.where(y < 0, q | 8, q), best_q)
        code = best_q.to(torch.uint8).view(-1, k)
        codes[r0:r0 + rows] = code[:, 0::2] | (code[:, 1::2] << 4)
        scales[r0:r0 + rows] = best_s
        err += float(best_err.sum())
        total += float((x ** 2).sum())
    return codes, scales, scale2, err / max(total, 1e-30)


def quantize_fp8(w: torch.Tensor, block: int = 128):
    """bf16 [N, K] -> (e4m3 [N, K], fp32 ``weight_scale_inv`` [N/128, K/128], relative squared error): Tencent's block
    FP8, a block's scale from its largest value."""

    n, k = w.shape
    x = w.float()
    nb, kb = -(-n // block), -(-k // block)
    pad = torch.zeros((nb * block, kb * block))
    pad[:n, :k] = x
    top = pad.view(nb, block, kb, block).abs().amax((1, 3))
    s = torch.where(top > 0, top / 448.0, torch.ones_like(top))
    full = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    codes = (x / full).to(torch.float8_e4m3fn)
    err = float(((codes.float() * full - x) ** 2).sum()) / max(float((x ** 2).sum()), 1e-30)
    return codes, s, err


def quantize_rows(w: torch.Tensor):
    """bf16 [N, K] -> (e4m3 [N, K], fp32 scale a row [N], relative squared error): the row's amax maps to 448. kv_b's
    form: the absorb and expand kernels scale a key row before, a value row after, its sum."""

    x = w.float()
    top = x.abs().amax(-1)
    s = torch.where(top > 0, top / 448.0, torch.ones_like(top))
    codes = (x / s[:, None]).to(torch.float8_e4m3fn)
    err = float(((codes.float() * s[:, None] - x) ** 2).sum()) / max(float((x ** 2).sum()), 1e-30)
    return codes, s, err


def main() -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--classes", default="attn,shared,latent,head")
    p.add_argument("--search", default="1.05,1.0,0.975,0.95,0.925,0.9,0.875,0.85,0.825,0.8,0.775,0.75,0.7")
    p.add_argument("--fp8", default="", help="classes written in block FP8 instead of NVFP4")
    p.add_argument("--layers", default="", help="first,last layer (inclusive); default every layer")
    a = p.parse_args()
    root, out = Path(a.model), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    search = tuple(float(f) for f in a.search.split(","))
    pats = [re.compile(CLASSES[c]) for c in a.classes.split(",")]
    fp8 = [re.compile(CLASSES[c]) for c in a.fp8.split(",") if c]
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted({n[:-len(".weight")] for n in index if n.endswith(".weight") and n.startswith(("model.layers.",
                                                                                                    "lm_head"))
                    and ".dfly" not in n and any(pt.search(n[:-len(".weight")]) for pt in pats)})
    lo, hi = (int(x) for x in a.layers.split(",")) if a.layers else (0, 1 << 30)
    groups: dict[str, list[str]] = {}
    for name in names:
        if name.startswith("model.layers."):
            li = int(name.split(".")[2])
            if lo <= li <= hi:
                groups.setdefault(f"L{li:02d}", []).append(name)
        else:
            groups.setdefault("head", []).append(name)
    report_path = out / "quant.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["search"] = list(search)
    t0 = time.perf_counter()
    for tag, members in groups.items():
        tensors: dict[str, torch.Tensor] = {}
        for name in members:
            with safe_open(str(root / index[name + ".weight"]), "pt", device="cpu") as f:
                w = f.get_tensor(name + ".weight")
            if w.dtype != torch.bfloat16:
                raise ValueError(f"{name}: {w.dtype}, not the BF16 release")
            if name.endswith("kv_b_proj"):
                codes, sc, rel = quantize_rows(w)
                tensors[name + ".weight"] = codes
                tensors[name + ".weight_scale_row"] = sc
                kind = "fp8-row"
            elif any(pt.search(name) for pt in fp8):
                codes, inv, rel = quantize_fp8(w)
                tensors[name + ".weight"] = codes
                tensors[name + ".weight_scale_inv"] = inv
                kind = "fp8"
            else:
                codes, scales, scale2, rel = quantize(w, search)
                tensors[name + ".weight"] = codes
                tensors[name + ".weight_scale"] = scales
                tensors[name + ".weight_scale_2"] = torch.tensor(scale2, dtype=torch.float32)
                kind = "nvfp4"
            report[name] = {"shape": list(w.shape), "format": kind, "rel_mse": rel}
            del w
        save_file(tensors, str(out / f"trunk-{tag}.safetensors"))
        report_path.write_text(json.dumps(report, indent=1))
        worst = max(report[n]["rel_mse"] for n in members)
        print(f"{tag}: {len(members)} tensors, worst relative error {worst:.5f}, {time.perf_counter() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
