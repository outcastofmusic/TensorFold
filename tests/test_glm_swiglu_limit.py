"""GLM-5.3 on CUDA: a ``swiglu_limit`` of 0 or null means no clamp, as the reference's ``if limit:`` reads it
(families/glm5_next/mlp.py), and a positive limit clamps as before (CPU only: the Triton kernel in interpret mode)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")          # GLM's CUDA weights import their triton kernels

from tensorfold.families.glm5_next.cuda.weights import Config

TEXT = {"hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 128, "rms_norm_eps": 1e-6,
        "num_attention_heads": 2, "q_lora_rank": 32, "kv_lora_rank": 32, "qk_nope_head_dim": 32, "v_head_dim": 32,
        "layer_types": ["linear_attention", "full_attention"], "n_routed_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64, "intermediate_size": 128, "routed_scaling_factor": 2.5, "eos_token_id": 1}
ABSENT = object()


def _limit(tmp_path: Path, value) -> float:
    text = dict(TEXT) if value is ABSENT else {**TEXT, "swiglu_limit": value}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text}))
    return Config.read(tmp_path).limit


@pytest.mark.parametrize("value, limit", [(10.0, 10.0), (7, 7.0), (0, math.inf), (0.0, math.inf), (None, math.inf),
                                          (-1.0, math.inf), (ABSENT, 10.0)])
def test_the_limit_the_kernels_take(tmp_path, value, limit):
    # absent is transformers' Glm5NextConfig default, 10.0, which both served checkpoints also set
    assert _limit(tmp_path, value) == limit


KERNEL = r"""
import sys
import torch
from tensorfold.families.glm5_next.cuda import glue

torch.manual_seed(0)
rows, w = 3, 64
gu = (torch.randn(rows, 2 * w) * 16).to(torch.bfloat16)        # gates and ups well past +-10
out = torch.empty(rows, w, dtype=torch.bfloat16)
xs = torch.empty(rows, w // 64, dtype=torch.float32)
glue.swiglu(gu, out, xs, float(sys.argv[1]))
g, u = gu[:, :w].float(), gu[:, w:].float()
for name, limit in (("none", float("inf")), ("10", 10.0)):
    cg, cu = g.clamp(max=limit), u.clamp(-limit, limit)
    want = (cg / (1 + torch.exp(-cg))).to(torch.bfloat16).float() * cu
    # the interpreter's bf16 casts truncate, so its two roundings sit within 2 bf16 steps (2 * 2^-7) of torch's
    if torch.allclose(out.float(), want, rtol=0.03, atol=0.05):
        print(name)
"""


@pytest.mark.parametrize("value, clamp", [(0, "none"), (None, "none"), (10.0, "10")])
def test_the_dense_swiglu_clamps_only_under_a_positive_limit(tmp_path, value, clamp):
    limit = _limit(tmp_path, value)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    run = subprocess.run([sys.executable, "-c", KERNEL, repr(limit)], check=False, env=env, capture_output=True,
                         text=True, timeout=240, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == [clamp]
