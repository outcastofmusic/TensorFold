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

With ``--stats DIR`` (``tools/hy4_calib.py``'s input sums) the error counts where the model reads it. A projection
whose site keeps the full X^T X = H is quantized by GPTQ (Frantar et al. 2022): input columns in order, each column's
rounding error spread over the columns not yet rounded through H's inverse, and each 16-input block's scale picked
when the block starts, on the updated weights, by the same search weighted by H's diagonal. A site with a diagonal
only weights the search by it. ``quant.json`` then also records ``rel_out``: tr(dW H dW^T) / tr(W H W^T), the
relative error of the projection's output on the calibration rows (a diagonal H where only that is kept).
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


MIDS = torch.tensor([(a + b) / 2 for a, b in pairwise(GRID)])
GRID_T = torch.tensor(GRID)


def _blocks(x: torch.Tensor, scale2: float, search: tuple[float, ...], weight: torch.Tensor | None = None):
    """Blocks x [.., 16] fp32 -> (e4m3 scales [..], signed codes int64 [.., 16], decoded values [.., 16]): each block
    takes the factor of ``search`` whose codes give the smallest squared error, each input's weighted by ``weight``
    [.., 16] (broadcast) when given."""

    top = x.abs().amax(-1)
    best_err = best_s = best_q = best_v = None
    for f in search:
        s = (top * f / 6.0 / scale2).clamp(max=448.0).to(torch.float8_e4m3fn)
        step = s.float()[..., None] * scale2
        y = torch.where(step > 0, x / step.clamp_min(1e-30), torch.zeros_like(x))
        q = torch.bucketize(y.abs().clamp(max=6.0), MIDS)
        v = GRID_T[q] * y.sign() * step
        d = (v - x) ** 2
        e = (d if weight is None else d * weight).sum(-1)
        code = torch.where(y < 0, q | 8, q)
        if best_err is None:
            best_err, best_s, best_q, best_v = e, s, code, v
        else:
            take = e < best_err
            best_err = torch.where(take, e, best_err)
            best_s = torch.where(take, s.float(), best_s.float()).to(torch.float8_e4m3fn)
            best_q = torch.where(take[..., None], code, best_q)
            best_v = torch.where(take[..., None], v, best_v)
    return best_s, best_q, best_v


def _pack_codes(code: torch.Tensor) -> torch.Tensor:
    code = code.to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4)


def _scale2(w: torch.Tensor) -> float:
    amax = float(w.float().abs().max())
    return amax / (448.0 * 6.0) if amax > 0 else 1.0


def quantize(w: torch.Tensor, search: tuple[float, ...] = (1.0,), rows: int = 4096, diag: torch.Tensor | None = None):
    """bf16 [N, K] -> (codes uint8 [N, K/2] low nibble first, e4m3 scales [N, K/16], fp32 global scale, relative
    squared error). Each 16-wide block takes the factor of ``search`` whose codes reconstruct it best, its inputs'
    errors weighted by ``diag`` [K] (an input's mean square on calibration rows) when given."""

    n, k = w.shape
    scale2 = _scale2(w)
    codes = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn)
    wt = None if diag is None else diag.float().view(1, k // 16, 16)
    err = total = 0.0
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float().view(-1, k // 16, 16)
        s, q, v = _blocks(x, scale2, search, wt)
        codes[r0:r0 + rows] = _pack_codes(q.view(-1, k))
        scales[r0:r0 + rows] = s
        err += float(((v - x) ** 2).sum())
        total += float((x ** 2).sum())
    return codes, scales, scale2, err / max(total, 1e-30)


def quantize_gptq(w: torch.Tensor, h: torch.Tensor, search: tuple[float, ...] = (1.0,), damp: float = 0.01,
                  block: int = 128, rows: int = 8192):
    """bf16 [N, K] and its input's X^T X [K, K] -> ``quantize``'s tuple by GPTQ: columns in order, each one's error
    spread over the later columns through the inverse of H + damp * mean(diag H) I, each 16-input block's scale
    searched on the updated weights when the block starts, its inputs weighted by diag H. Rows are independent, so
    they go ``rows`` at a time (the head's 120,832)."""

    n, k = w.shape
    if k % block or block % 16:
        raise ValueError(f"GPTQ blocks of {block} do not tile K = {k} in 16-input groups")
    scale2 = _scale2(w)
    H = h.double().clone()
    dead = torch.diagonal(H) <= 0
    H[dead, dead] = 1.0
    H += damp * float(torch.diagonal(H).mean()) * torch.eye(k, dtype=torch.float64)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    dg = torch.diagonal(H).float()
    codes = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn)
    err = total = 0.0
    for r0 in range(0, n, rows):
        x = w[r0:r0 + rows].float()
        W = x.clone()
        W[:, dead] = 0.0
        m = W.shape[0]
        code = torch.empty((m, k), dtype=torch.uint8)
        for i1 in range(0, k, block):
            i2 = i1 + block
            W1 = W[:, i1:i2].clone()
            E1 = torch.empty_like(W1)
            h1 = hinv[i1:i2, i1:i2]
            step = None
            for j in range(block):
                c = i1 + j
                if c % 16 == 0:
                    s, _, _ = _blocks(W1[:, j:j + 16].view(m, 1, 16), scale2, search, dg[c:c + 16].view(1, 1, 16))
                    scales[r0:r0 + m, c // 16] = s[:, 0]
                    step = s.float()[:, 0] * scale2
                y = torch.where(step > 0, W1[:, j] / step.clamp_min(1e-30), torch.zeros_like(step))
                qi = torch.bucketize(y.abs().clamp(max=6.0), MIDS)
                v = GRID_T[qi] * y.sign() * step
                code[:, c] = torch.where(y < 0, qi | 8, qi).to(torch.uint8)
                err += float(((v - x[:, c]) ** 2).sum())
                e = (W1[:, j] - v) / h1[j, j]
                W1[:, j:] -= e[:, None] * h1[j, j:][None, :]
                E1[:, j] = e
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
        codes[r0:r0 + m] = _pack_codes(code)
        total += float((x ** 2).sum())
    return codes, scales, scale2, err / max(total, 1e-30)


def output_error(w: torch.Tensor, back: torch.Tensor, h: torch.Tensor) -> float:
    """tr(dW H dW^T) / tr(W H W^T) for H [K, K], or with H's diagonal [K]: the output's relative squared error on the
    calibration rows."""

    x, d = w.float(), back.float() - w.float()
    if h.dim() == 1:
        return float((d ** 2 @ h.float()).sum()) / max(float((x ** 2 @ h.float()).sum()), 1e-30)
    hf = h.float()
    return float(((d @ hf) * d).sum()) / max(float(((x @ hf) * x).sum()), 1e-30)


def dense_of(codes: torch.Tensor, scales: torch.Tensor, scale2: float) -> torch.Tensor:
    """NVFP4 codes and scales back to fp32 [N, K]."""

    c = codes.to(torch.int64)
    q = torch.stack([c & 15, c >> 4], -1).flatten(-2)
    lut = torch.tensor(GRID + tuple(-g for g in GRID))
    return lut[q] * scales.float().repeat_interleave(16, -1) * scale2


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


class Stats:
    """``tools/hy4_calib.py``'s sites: a projection's input sums by its name (H [K, K], or a diagonal [K])."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.names = json.loads((folder / "sites.json").read_text())["names"]
        self.cache: dict[str, dict[str, torch.Tensor]] = {}

    def get(self, name: str) -> torch.Tensor | None:
        from safetensors.torch import load_file

        site = self.names.get(name)
        if site is None:
            return None
        tag = f"L{int(site.split('.')[2]):02d}" if site.startswith("model.layers.") else "head"
        if tag not in self.cache:
            self.cache = {tag: load_file(str(self.folder / f"sites-{tag}.safetensors"))}   # one layer at a time
        t = self.cache[tag]
        for kind in ("h", "diag"):
            if f"{site}.{kind}" in t:
                n = float(t[f"{site}.n"])
                return t[f"{site}.{kind}"].double() / max(n, 1.0)
        return None


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
    p.add_argument("--stats", default="", help="tools/hy4_calib.py's folder: GPTQ, or diagonal-weighted scales")
    p.add_argument("--gptq", default="", help="classes quantized by GPTQ where --stats keeps a full H (default: all)")
    p.add_argument("--damp", type=float, default=0.01)
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
    stats = Stats(Path(a.stats)) if a.stats else None
    gptq = [re.compile(CLASSES[c]) for c in a.gptq.split(",") if c] if a.gptq else None
    if stats is not None:
        report["stats"] = {"dir": a.stats, "damp": a.damp, "gptq": a.gptq or "all"}
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
                h = stats.get(name) if stats is not None else None
                if h is not None and h.dim() == 2 and (gptq is None or any(pt.search(name) for pt in gptq)):
                    codes, scales, scale2, rel = quantize_gptq(w, h, search, a.damp)
                    kind = "nvfp4-gptq"
                else:
                    codes, scales, scale2, rel = quantize(w, search, diag=None if h is None else
                                                          (h if h.dim() == 1 else torch.diagonal(h)))
                    kind = "nvfp4" if h is None else "nvfp4-weighted"
                tensors[name + ".weight"] = codes
                tensors[name + ".weight_scale"] = scales
                tensors[name + ".weight_scale_2"] = torch.tensor(scale2, dtype=torch.float32)
                if h is not None:
                    out_err = output_error(w, dense_of(codes, scales, scale2), h)
            report[name] = {"shape": list(w.shape), "format": kind, "rel_mse": rel}
            if stats is not None and kind.startswith("nvfp4") and h is not None:
                report[name]["rel_out"] = out_err
            del w
        save_file(tensors, str(out / f"trunk-{tag}.safetensors"))
        report_path.write_text(json.dumps(report, indent=1))
        worst = max(report[n]["rel_mse"] for n in members)
        print(f"{tag}: {len(members)} tensors, worst relative error {worst:.5f}, {time.perf_counter() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
