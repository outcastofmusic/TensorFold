"""GLM's DFlash2 rounds with copies (TF_GLM_DFLASH_COPIES): a repeated tail verifies its earlier continuation in place
of a block, the window widens from 6 rows while copies land whole, and the tokens are the ones the rounds give
without copies."""

from __future__ import annotations

import importlib

import pytest

from test_glm_decode_stop import NEVER_ENDS, FakeEngine

pytestmark = pytest.mark.torch

COUNT = 80


@pytest.fixture
def decode(monkeypatch):
    import torch

    module = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    monkeypatch.setattr(module, "commit", lambda *args: None)
    monkeypatch.setattr(module, "_sync", lambda w: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return module


class Windows(FakeEngine):
    """Records each verify window's rows."""

    def __init__(self):
        super().__init__()
        self.widths: list[int] = []

    def forward(self, tokens):
        self.widths.append(len(tokens))
        return super().forward(tokens)


class Blocks:
    """A DFlash2 drafter whose blocks are the fake model's own tokens; counts the blocks it was asked for."""

    def __init__(self):
        self.asked = 0

    def propose(self, last, depth, sampling, confidence):
        self.asked += 1
        return [NEVER_ENDS] * depth

    def add_taps(self, rows):
        pass


def run(decode, copies):
    from tensorfold.families.qwen4_exp.cuda.copy_drafts import CopyIndex

    e, drafter = Windows(), Blocks()
    index = CopyIndex([NEVER_ENDS] * 20, rows=6, most=32) if copies else None
    res = decode.dflash_decode(e, drafter, NEVER_ENDS, COUNT, None, policy=decode.DepthPolicy(5, fixed=True),
                               copies=index)
    return res, e.widths, drafter.asked


def test_copies_widen_while_they_land_and_keep_the_tokens(decode):
    plain, plain_widths, plain_blocks = run(decode, copies=False)
    copied, widths, blocks = run(decode, copies=True)
    assert copied.tokens == plain.tokens
    assert plain_widths[:3] == [6, 6, 6] and plain_blocks == len(plain_widths)
    assert widths[:4] == [6, 12, 24, 32]                   # 5 drafts, then doubling to TENSORFOLD_COPY_ROWS's 32
    assert blocks == 0                                     # every round copied: no block was drafted
