"""The real Bolt LiLiCorr checkpoint: it loads at the expected shapes and drafts on the GPU from random target taps."""

from __future__ import annotations

import json
import os
import time
from contextlib import closing
from pathlib import Path

import pytest
import torch

from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_bolt.cuda.lilicorr import DraftConfig, Drafter, device_bytes, expected_shapes
from tensorfold.families.nemotron_bolt.cuda.weights import dequantize_q8_0_rows

DRAFT = Path(os.environ.get("TF_BOLT_DRAFT", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"))
TARGET = Path(os.environ.get("TF_BOLT_TARGET", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB"))

pytestmark = pytest.mark.skipif(not (DRAFT / "model.safetensors").exists() or
                                not (TARGET / "model.safetensors.index.json").exists(),
                                reason="needs the Bolt target and LiLiCorr draft checkpoints")


@pytest.fixture(scope="module")
def shared():
    """The target's bf16 embedding and its Q8_0 lm_head, dequantized: what LiLiCorr shares with the target."""

    index = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
    names = ("language_model.backbone.embeddings.weight", "language_model.lm_head.weight")
    with closing(SafeTensors(sorted({TARGET / index[n] for n in names}))) as f:
        embedding = f.get(names[0], "cuda")
        head = dequantize_q8_0_rows(f.get(names[1], "cuda").flatten(-2))
    return {"embed": lambda ids: embedding[ids], "lm_head": lambda h: h @ head.T}


@pytest.fixture(scope="module")
def drafter(shared):
    return Drafter(DRAFT, device="cuda", **shared)


def test_checkpoint_geometry():
    cfg = DraftConfig.from_json(json.loads((DRAFT / "config.json").read_text()))
    assert (cfg.hidden, cfg.layers, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.inter) == (4096, 5, 32, 2, 128, 12288)
    assert (cfg.block, cfg.window, cfg.top_k, cfg.mask_id, cfg.sinks) == (8, 1024, 8, 990, True)
    assert cfg.tap_layers == (1, 18, 35, 51, 68, 85)
    assert expected_shapes(cfg)["fc.weight"] == (4096, 6 * 4096)


def test_loads_and_drafts_on_the_gpu(drafter):
    d = drafter
    params = sum(t.numel() for t in d.p.values()) + sum(q.n * q.k for q in d.q.values())
    assert 1.10e9 < params < 1.12e9                          # the README's 1.113B draft parameters
    assert len(d.q) == 1 + 3 * 5                             # fc and every MLP projection stay NVFP4
    assert all(t.dtype == torch.bfloat16 and t.is_cuda for t in d.p.values())
    assert all(torch.isfinite(t).all() for t in d.p.values())
    assert d.nbytes() == device_bytes(DRAFT)                  # what admission charges for the drafter
    g = torch.Generator(device="cuda").manual_seed(0)
    taps = torch.randn((1500, 6 * 4096), generator=g, device="cuda", dtype=torch.bfloat16)
    d.reset()
    d.add_taps(taps[:1200])                                  # a prompt longer than the window
    d.add_taps(taps[1200:1203])
    assert d.context_end == 1203 and int((d.slot_pos >= 0).sum()) == 1024
    for sampling in (None, Sampling(seed=3, temperature=1.0, top_k=0, top_p=0.95)):
        drafts = d.propose(17, 7, sampling)
        assert len(drafts) == 7 and all(0 <= t < 131072 for t in drafts)
        assert torch.isfinite(d.last.scores).all() and d.last.scores.shape == (7, 8, 8)
        assert (d.last.log_probs <= 0).all()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        d.propose(17, 7, None)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    d.add_taps(taps[1203:1211])
    torch.cuda.synchronize()
    add = time.perf_counter() - t0
    print(f"\npropose(depth 7): {min(times) * 1e3:.1f} ms min of 5; add_taps(8 rows): {add * 1e3:.1f} ms; "
          f"weights {d.nbytes() / 2**30:.2f} GiB; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def test_packed_matches_dequantized(drafter):
    """Each kept NVFP4 projection on the fork's matmul against our dequantized weights, both against fp32.

    End to end through 5 layers on random taps the two paths differ by 4.4% (2026-10-16), the bf16 rounding of each
    projection amplified by random activations, so the check sits at the projection.
    """

    from tensorfold.families.nemotron_bolt.quant import dequantize_nvfp4

    with closing(SafeTensors([DRAFT / "model.safetensors"])) as f:
        for name in ("fc", "layers.0.mlp.gate_proj", "layers.4.mlp.down_proj"):
            w, s, s2 = (f.get(name + x, "cuda") for x in (".weight", ".weight_scale", ".weight_scale_2"))
            dense = dequantize_nvfp4(w, s, s2)
            for rows in (8, 300):
                x = torch.randn((rows, dense.shape[1]), device="cuda", dtype=torch.bfloat16)
                ref = x.float() @ dense.float().T
                got = drafter._lin(x, name + ".weight").float()
                rel = float((got - ref).norm() / ref.norm())
                assert rel < 4e-3, (name, rows, rel)            # bf16 F.linear sits at 1.7e-3


def _former_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, s2: float, dtype: torch.dtype) -> torch.Tensor:
    """The drafter's NVFP4 decoder before 2026-10-17 (lilicorr.dequantize_nvfp4), 2,048 rows at a time."""

    mags = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=torch.float32, device=weight.device)
    table = torch.cat([mags, -mags])
    out = torch.empty((weight.shape[0], 2 * weight.shape[1]), dtype=dtype, device=weight.device)
    for r in range(0, weight.shape[0], 2048):
        w = weight[r:r + 2048].to(torch.int64)
        codes = torch.stack([w & 0xF, w >> 4], dim=-1).reshape(w.shape[0], -1)
        scale = weight_scale[r:r + 2048].to(torch.float32).repeat_interleave(16, dim=1) * s2
        out[r:r + 2048] = (table[codes] * scale).to(dtype)
    return out


def test_every_nvfp4_tensor_decodes_as_before():
    """quant.dequantize_nvfp4, which the drafter now loads with, gives the former decoder's bits on the checkpoint."""

    from tensorfold.families.nemotron_bolt.cuda.lilicorr import dense_tensors

    with closing(SafeTensors([DRAFT / "model.safetensors"])) as f:
        keys = set(f.keys())                                     # noqa: SIM118  (no __iter__)
        names = [n for n in keys if n.endswith(".weight") and n + "_scale" in keys]
        assert len(names) >= 16
        for name in names:
            w, s, s2 = (f.get(name + x, "cuda") for x in ("", "_scale", "_scale_2"))
            got = dense_tensors({name: w, name + "_scale": s, name + "_scale_2": s2}, torch.bfloat16)[name]
            want = _former_nvfp4(w, s, float(s2.item()), torch.bfloat16)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), name
