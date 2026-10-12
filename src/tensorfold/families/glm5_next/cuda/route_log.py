"""TF_GLM_ROUTE_LOG=dir: each request's routed-expert picks, counted on the GPU and written when it ends (use --eager).

One .npz a request: counts [phase, layer, E] (phase 0 a decode window's rows, 1 prompt rows), wsum [phase, layer, k]
(each row's routed weights sorted high to low, summed), small [phase, layer, 3] (picks under 2%, 5% and 10% of their
row's routed weight), uniq / calls / rows [phase, layer] (distinct experts a call, calls, rows)."""

from __future__ import annotations

import json
import os

import torch

PATH = os.environ.get("TF_GLM_ROUTE_LOG", "")
_layers: dict[int, int] = {}
_t: dict[str, torch.Tensor] = {}
_n = 0
MAX_LAYERS = 64


def note(moe, pick: torch.Tensor, wts: torch.Tensor, R: int, experts: int, top_k: int, prefill: bool) -> None:
    if not PATH:
        return
    li = _layers.setdefault(id(moe), len(_layers))
    if not _t:
        dev = pick.device
        _t.update(counts=torch.zeros((2, MAX_LAYERS, experts + 1), dtype=torch.int64, device=dev),
                  wsum=torch.zeros((2, MAX_LAYERS, top_k), dtype=torch.float64, device=dev),
                  small=torch.zeros((2, MAX_LAYERS, 3), dtype=torch.int64, device=dev),
                  uniq=torch.zeros((2, MAX_LAYERS), dtype=torch.int64, device=dev),
                  calls=torch.zeros((2, MAX_LAYERS), dtype=torch.int64, device=dev),
                  rows=torch.zeros((2, MAX_LAYERS), dtype=torch.int64, device=dev))
    ph = int(bool(prefill))
    p = pick[:R, :top_k].long().clamp(0, experts)
    hit = torch.zeros((experts + 1,), dtype=torch.int64, device=p.device)
    hit.scatter_add_(0, p.reshape(-1), torch.ones_like(p.reshape(-1)))
    _t["counts"][ph, li] += hit
    w = wts[:R, :top_k].float()
    share = w / w.sum(dim=1, keepdim=True).clamp_min(1e-20)
    _t["wsum"][ph, li] += share.sort(dim=1, descending=True).values.double().sum(dim=0)
    _t["small"][ph, li] += torch.stack([(share < x).sum() for x in (0.02, 0.05, 0.10)])
    _t["uniq"][ph, li] += (hit[:experts] > 0).sum()
    _t["calls"][ph, li] += 1
    _t["rows"][ph, li] += R


def flush(meta: dict) -> None:
    """Write the request's counts and start the next request's."""

    global _n
    if not PATH or not _t:
        return
    import numpy as np

    os.makedirs(PATH, exist_ok=True)
    L = len(_layers)
    np.savez_compressed(os.path.join(PATH, f"req{_n:04d}.npz"), meta=json.dumps(meta),
                        **{k: v[:, :L].cpu().numpy() for k, v in _t.items()})
    for v in _t.values():
        v.zero_()
    _n += 1
