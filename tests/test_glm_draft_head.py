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
