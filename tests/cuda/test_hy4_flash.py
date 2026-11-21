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


class _Oracle:
    """Proposes serial decoding's own next tokens, one of them wrong every third round: full and partial accepts."""

    def __init__(self, serial: list[int], prompt_len: int):
        self.serial, self.prompt_len, self.rounds = serial, prompt_len, 0

    def propose(self, e, last: int, sampling) -> list[int]:
        at = e.pos - self.prompt_len + 1
        guess = list(self.serial[at:at + 5])
        self.rounds += 1
        if guess and self.rounds % 3 == 0:
            guess[len(guess) // 2] = (guess[len(guess) // 2] + 1) % 512
        return guess

    def commit(self, e, accepted: int) -> None:
        pass


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_drafted_equals_serial(weights, goldens, temperature):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode

    prompt = goldens["long_ids"].tolist()[:20]
    s = Sampling(seed=3, temperature=temperature)
    e = _engine(weights)
    first = prefill(e, prompt, s)
    serial = serial_decode(e, first, 40, s, stop_eos=False).tokens
    e = _engine(weights)
    first = prefill(e, prompt, s)
    oracle = _Oracle(serial, len(prompt))
    got = draft_decode(e, prompt, first, 40, s, stop_eos=False, drafter=oracle, allow_copy=False)
    assert got.tokens == serial
    assert got.stats["accepted"] > got.stats["rounds"] and got.stats["accepted"] < got.stats["drafted"]


@pytest.mark.parametrize("head", ["nvfp4", "slice"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_dfly_drafted_equals_serial(weights, goldens, temperature, head, tmp_path, monkeypatch):
    """DFly2's real rounds (random weights: few drafts land) and copies, keyed drafts, either draft head: the reply is
    serial decoding's."""

    monkeypatch.setenv("TF_HY4_DRAFT_HEAD", head)
    monkeypatch.setenv("TF_HY4_DRAFT_VOCAB", "128")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode

    prompt = goldens["long_ids"].tolist()[:20]
    s = Sampling(seed=5, temperature=temperature)
    e = _engine(weights)
    serial = serial_decode(e, prefill(e, prompt, s), 40, s, stop_eos=False).tokens
    e = _engine(weights)
    drafter = DFly(tiny.write(tmp_path / "t"), e)
    got = draft_decode(e, prompt, prefill(e, prompt, s), 40, s, stop_eos=False, drafter=drafter)
    assert got.tokens == serial
    assert got.stats["drafted"] > 0


@pytest.mark.parametrize("start", [10, 32])
@pytest.mark.parametrize("width", [2, 4, 8, 12, 16])
def test_window_rows_equal_single_rows(weights, goldens, width, start):
    """A verify window's rows are the bits of the same tokens decoded one row a call (playbook 5.1), below and past
    the tiny indexer's 32-key limit."""

    ids = goldens["long_ids"].tolist()
    one = _engine(weights)
    one.forward(ids[:start], prompt=True)
    want = torch.cat([one.forward([t], prompt=False) for t in ids[start:start + width]])
    win = _engine(weights)
    win.forward(ids[:start], prompt=True)
    got = win.forward(ids[start:start + width], prompt=False, every=True)
    assert torch.equal(got, want), f"max diff {float((got - want).abs().max())}"


def test_window_graphs_equal_eager(weights, goldens):
    """Verify windows by graph replay (one graph a width) give the eager windows' bits, at each width twice."""

    ids = goldens["long_ids"].tolist()
    out = []
    for graphs in (False, True):
        e = _engine(weights, graphs=graphs)
        e.forward(ids[:10], prompt=True)
        rows = []
        for width in (6, 3, 6, 16, 3):
            start = e.pos
            rows.append(e.forward(ids[start:start + width], prompt=False, every=True))
            e.rewind(width - 1)                      # keep one row a call, as a rejected draft leaves it
        out.append(torch.cat(rows))
    assert torch.equal(out[0], out[1])


def test_nvfp4_draft_head_gives_its_dense_values(weights):
    """The draft head on the lane matmul (Fp4Linear) against its own codes decoded to fp32: 6 rows, as a round reads."""

    from tensorfold.families.hy4_flash.cuda.dfly import DFly, quantize_nvfp4
    from tensorfold.families.hy4_flash.cuda.weights import _nvfp4_dense

    vocab, head = DFly._head(weights.head, weights.config, True)
    codes, scales, scale2 = quantize_nvfp4(weights.head)
    dense = _nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2, device="cuda"))
    h = torch.randn(6, weights.head.shape[1], device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    h = h.to(torch.bfloat16)
    got, want = head(h).float(), h.float() @ dense.t()
    assert vocab is None and got.shape == want.shape
    assert float((got - want).abs().max() / want.abs().max()) < 0.01
