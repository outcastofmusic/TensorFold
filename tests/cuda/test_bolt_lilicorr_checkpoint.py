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

    from tensorfold.families.nemotron_bolt.cuda.lilicorr import dequantize_nvfp4

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
