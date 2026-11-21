"""Hy4-flash's CUDA engine on the tiny checkpoint (``tests/hy4_flash_tiny.py``): the kernels against transformers'
goldens (``tests/fixtures/hy4_flash``), and graph replay against eager decode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tests import hy4_flash_tiny as tiny

GOLDENS = Path(__file__).parents[1] / "fixtures" / "hy4_flash" / "tiny_goldens.npz"


@pytest.fixture(scope="module")
def goldens():
    return np.load(GOLDENS)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.hy4_flash.cuda.weights import load

    return load(tiny.write(tmp_path_factory.mktemp("hy4") / "tiny"), "cuda")


def _engine(weights, goldens=None, *, graphs: bool = False, **kw):
    from tensorfold.families.hy4_flash.cuda import model

    model.ENGRAM_EPS = float(torch.finfo(torch.bfloat16).eps)        # transformers' bf16 run
    e = model.Engine(weights, max_len=kw.pop("max_len", 64), prefill_rows=kw.pop("prefill_rows", 64), **kw)
    e.graphs = graphs
    if goldens is not None:
        e.replay = {int(k.rsplit("_", 1)[1]): torch.from_numpy(goldens[k]) for k in goldens.files
                    if k.startswith("long_picks_")}
        e.replay_groups = {int(k.rsplit("_", 1)[1]): torch.from_numpy(goldens[k]) for k in goldens.files
                           if k.startswith("long_groups_")}
    return e


def _close(got: torch.Tensor, want: np.ndarray, bound: float = 0.03) -> None:
    got = got.float().cpu().numpy()
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < bound, f"max error {err:.4f} of the logits' range"


def test_prompt_logits_match_transformers(weights, goldens):
    e = _engine(weights, goldens)
    _close(e.forward(goldens["long_ids"].tolist(), prompt=True, every=True), goldens["long_logits"])


def test_decode_steps_match_transformers(weights, goldens):
    """A 20-row prompt (the batched absorb), then one-row steps on the row kernels, past the window and the limit."""

    e = _engine(weights, goldens)
    ids = goldens["long_ids"].tolist()
    rows = [e.forward(ids[:20], prompt=True, every=True)]
    rows += [e.forward([t], prompt=False) for t in ids[20:]]
    _close(torch.cat(rows), goldens["long_logits"])


def test_graph_replay_equals_eager(weights, goldens):
    ids = goldens["long_ids"].tolist()
    out = []
    for graphs in (False, True):
        e = _engine(weights, graphs=graphs)
        e.forward(ids[:10], prompt=True)
        out.append(torch.cat([e.forward([t], prompt=False) for t in ids[10:]]))
    assert torch.equal(out[0], out[1])
