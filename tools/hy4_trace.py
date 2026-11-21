"""Hy4-flash's engine against transformers' per-layer streams (``hy4_golden.py reference`` with ``HY4_REF_DUMP``):
the first layer where the two part (playbook 4.4).

    python tools/hy4_trace.py MODEL LAYERS.npz [--rows 16]

``--rows 0`` runs the sequence as one prompt pass (the prompt kernels), ``--rows N`` as calls of N decode rows.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("layers")
    p.add_argument("--rows", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cpu: the engine's fp32 torch path (same semantics, no GPU)")
    p.add_argument("--layers", type=int, default=0, help="load and compare only the first N layers")
    a = p.parse_args()

    import torch

    from tensorfold.families.hy4_flash.cuda.model import Engine
    from tensorfold.families.hy4_flash.cuda.weights import load

    z = np.load(a.layers)
    ids = z["ids"].tolist()
    w = load(a.model, a.device, layers=a.layers or None)
    e = Engine(w, max_len=8192)
    e.graphs = False
    got: dict[int, list] = {}
    inner = e._layer

    def layer(i, lay, x, prompt):
        y = inner(i, lay, x, prompt)
        got.setdefault(i, []).append(y.clone())
        return y

    e._layer = layer
    sub: dict[str, list] = {}

    def keep(tag, fn):
        def wrapped(*args):
            out = fn(*args)
            if int(args[0]) == 1:                    # layer 1: the first Engram, MoE and FP8 attention
                sub.setdefault(tag, []).append(out.clone())
            return out
        return wrapped

    e._attention = keep("attn", e._attention)
    e._mlp = keep("mlp", e._mlp)
    engram = e._engram

    def first_engram(j, layer_eg, x):             # Engram layer 0 is decoder layer 1
        out = engram(j, layer_eg, x)
        if j == 0:
            sub.setdefault("engram", []).append(out.clone())
        return out

    e._engram = first_engram
    step = a.rows or len(ids)
    for s in range(0, len(ids), step):
        e.forward(ids[s:s + step], prompt=a.rows == 0, every=True)
    for tag, parts in sub.items():
        key = f"{tag}_1"
        if key not in z.files:
            continue
        mine = torch.cat(parts).float()
        want = torch.from_numpy(z[key]).view(torch.bfloat16).float().to(mine.device).reshape(mine.shape)
        rel = float((mine - want).abs().max() / want.abs().max())
        mean = float((mine - want).abs().mean() / want.abs().mean())
        print(f"sublayer {key} max {rel:.4f} mean {mean:.4f} |mine| {float(mine.abs().mean()):.4f} "
              f"|want| {float(want.abs().mean()):.4f}", flush=True)
    for i in range(len(w.layers)):
        mine = torch.cat(got[i]).float()
        want = torch.from_numpy(z[f"layer_{i}"]).view(torch.bfloat16).float().to(mine.device)
        rel = float((mine - want).abs().max() / want.abs().max())
        mean = float((mine - want).abs().mean() / want.abs().mean())
        cos = torch.nn.functional.cosine_similarity(mine.flatten(1), want.flatten(1), dim=1)
        print(f"layer {i:2d} {w.config.kinds[i][:6]} max {rel:.4f} mean {mean:.4f} cos min {float(cos.min()):.5f} "
              f"worst row {int(cos.argmin())}", flush=True)


if __name__ == "__main__":
    main()
