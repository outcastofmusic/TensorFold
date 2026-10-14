"""GLM's 4-bit draft head: a DFlash2 draft model reads it as MTP does, so it is built and budgeted for either."""

from __future__ import annotations

import math

import pytest

from tensorfold.families.glm5_next.cuda.engine import _weight_transform
from tensorfold.families.glm5_next.cuda.split import rule

pytestmark = pytest.mark.torch

HEAD = {"dtype": "BF16", "shape": [1024, 64]}


@pytest.mark.parametrize("world", [1, 2])
def test_the_head_budget_carries_the_draft_copy_when_only_the_draft_model_drafts(world):
    with_copy = _weight_transform(rule, world, 4, load_mtp=False, draft_head=True)("lm_head.weight", HEAD)[0]
    without = _weight_transform(rule, world, 4, load_mtp=False, draft_head=False)("lm_head.weight", HEAD)[0]
    assert with_copy - without == math.prod(HEAD["shape"]) // world * 9 // 16


def test_admission_counts_the_dense_prompt_workspace(monkeypatch):
    from types import SimpleNamespace

    from tensorfold.families.glm5_next.cuda import packed

    cfg = SimpleNamespace(hidden=256, moe_width=128, experts=8, top_k=2, streams=4, mtp_layers=0, layers=4)
    p = "model.language_model.layers.0."
    headers = {p + "self_attn.o_proj.trellis": {"dtype": "I16", "shape": [32, 16, 48]},    # k 512, n 256
               p + "mlp.down_proj.trellis": {"dtype": "I16", "shape": [64, 16, 48]},       # k 1024, n 256
               "lm_head.trellis": {"dtype": "I16", "shape": [16, 4096, 48]}}               # stays on the decode linear
    monkeypatch.setattr(packed, "DENSE_PROMPT", False)
    plain = packed.workspace_bytes(cfg, headers, prefill_rows=64)
    monkeypatch.setattr(packed, "DENSE_PROMPT", True)
    dense = packed.workspace_bytes(cfg, headers, prefill_rows=64)
    assert dense - plain == 2 * 1024 * 256 + 2 * 64 * 1024 + 2 * 128 * 128     # decoded W, rotated input, Hadamard
