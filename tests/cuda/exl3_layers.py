"""The EXL3 GPU tests' layers: random routed experts at Flash Next EXL3's and GLM-5.3's shapes, their inputs, and
the dense linear shapes."""

from __future__ import annotations

import math

import torch

from tensorfold.cuda.exl3 import experts


def _trellis(k, n, words16, g):
    v = torch.randint(-32768, 32768, (k // 16, n // 16, words16), dtype=torch.int32, generator=g)
    return v.to(torch.int16).cuda().contiguous()


def _scale(n, mag, g):
    sign = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
    return (sign * (torch.rand((n,), generator=g) + 0.5) * mag).half().cuda()


def _flash_next(E=48, seed=1):
    """E routed experts at 3 bits (48 int16 a tile) and the shared expert (index E) at 5 bits (80), mul1."""

    D, I = 2560, 640
    g = torch.Generator().manual_seed(seed)
    mats = {p: [] for p in ("gate", "up", "down")}
    for e in range(E + 1):
        w16 = 80 if e == E else 48
        for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
            mats[p].append((_trellis(k, n, w16, g), _scale(k, 1 / math.sqrt(k), g), _scale(n, 0.5, g)))
    return experts.prepare(mats["gate"], mats["up"], mats["down"], "mul1"), 11, E, experts.ACT_F32


def _glm(words16, E=48, seed=2):
    """A stacked GLM-5.3 layer: 2.25 bits (36 int16 a tile) or 2.5 bits (40), mcg, 14-bit window."""

    D, I = 4096, 2048
    g = torch.Generator().manual_seed(seed)
    tr = [_trellis(k, n, words16, g).unsqueeze(0).expand(E, -1, -1, -1).clone()
          for k, n in ((D, I), (D, I), (I, D))]
    # every expert its own words (a stacked copy would hide a wrong expert stride)
    for t in tr:
        t.view(E, -1)[:, :97] = torch.randint(-32768, 32768, (E, 97), dtype=torch.int32, generator=g).to(torch.int16).cuda()

    def sc(n, mag):
        return torch.stack([_scale(n, mag, g) for _ in range(E)])

    ex = experts.prepare_stacked(tr[0], tr[1], tr[2], sc(D, 1 / 64), sc(D, 1 / 64), sc(I, 0.5), sc(I, 0.5),
                                 sc(I, 1 / 45), sc(D, 0.5), "mcg", window=14)
    return ex, 8, E, experts.ACT_BF16


LAYERS = {"flash_next": lambda: _flash_next(), "glm53_2.25bit": lambda: _glm(36), "glm53_2.5bit": lambda: _glm(40)}
_built: dict = {}


def _layer(name):
    if name not in _built:
        _built.clear()
        torch.cuda.empty_cache()
        _built[name] = LAYERS[name]()
    return _built[name]


def _inputs(name, R, seed):
    ex, slots, E, _ = _layer(name)
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((R, ex.dims), generator=g) * 0.5).to(torch.bfloat16).cuda()
    routed = slots - 1 if name == "flash_next" else slots
    sel = torch.stack([torch.randperm(E, generator=g)[:routed] for _ in range(R)]).to(torch.int32)
    w = torch.rand((R, slots), generator=g) * 0.2 + 0.05
    if name == "flash_next":                                  # the shared expert in the last slot
        sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
    return x, sel.cuda().contiguous(), w.float().cuda().contiguous()


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


# (K, N, bits): Flash Next's dense trellis projections (DeltaNet qkv, z, out; attention q, k/v, o; the shared expert;
# the indexer; the MTP fc), and GLM-5.3-sized ones
LINEAR = [(2560, 10240, 5), (2560, 6144, 5), (6144, 2560, 5), (2560, 12288, 5), (2560, 512, 5), (2560, 640, 5),
          (640, 2560, 5), (2560, 640, 3), (2560, 2560, 4), (4096, 2048, 4), (2048, 4096, 3), (4096, 512, 6)]
