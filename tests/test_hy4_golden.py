"""tools/hy4_golden.py measures the engine against transformers up to the reply's end token, never past it: past the
end both models are out of distribution, and the tool golden's KL of 21 Nov (0.098 mean, 2.84 max) was mostly those
rows (CPU, the tiny checkpoint of ``hy4_flash_tiny``)."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch

TOOL = Path(__file__).parents[1] / "tools" / "hy4_golden.py"


@pytest.fixture(scope="module")
def golden():
    spec = importlib.util.spec_from_file_location("hy4_golden", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.hy4_flash.cuda.weights import load

    return load(tiny.write(tmp_path_factory.mktemp("hy4") / "tiny"), "cpu")


def test_the_golden_reply_stops_at_its_end_token(golden, weights):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.model import Engine, prefill, serial_decode

    greedy = Sampling(seed=0, temperature=0.0)
    prompt = list(range(10, 22))
    e = Engine(weights, max_len=64, prefill_rows=64)
    free = serial_decode(e, prefill(e, prompt, greedy), 10, greedy, stop_eos=False).tokens
    end = next(i for i in range(1, len(free)) if free[i] not in free[:i])     # a token the reply first writes at i
    ended = dataclasses.replace(weights, config=dataclasses.replace(weights.config, eos=(free[end],)))
    e = Engine(ended, max_len=64, prefill_rows=64)
    res, seq = golden.reply(e, prompt, prefill(e, prompt, greedy), greedy, steps=10)
    assert res.tokens == free[:end + 1]
    assert seq == prompt + free[:end]


@pytest.mark.parametrize("gen, rows", [([5, 7, 1, 1, 1], 3), ([5, 7, 9], 3), ([1], 1)])
def test_the_reference_reads_rows_up_to_the_first_end(golden, gen, rows):
    """An engine golden saved before the fix holds tokens past the end: the reference reads only the rows before it."""

    assert golden.decoded_rows(gen, (1,)) == rows


def test_compare_reads_the_saved_reference_rows(golden, tmp_path):
    """``save_wants`` then ``compare``: a golden equal to the reference scores KL 0 and full agreement; one with a
    perturbed row scores that row's KL, computed by hand."""

    import numpy as np
    import torch

    v, p0, eos = 50, 6, (1,)
    rng = np.random.default_rng(0)
    seq = list(range(10, 10 + p0)) + [7, 8, 9]
    gen = [7, 8, 9, 1, 4]                               # rows up to the end token: 4
    want = torch.from_numpy(rng.normal(size=(len(seq), v)).astype(np.float32))
    want[p0 - 1:p0 + 3, :] -= 20
    for i, t in enumerate(gen[:4]):
        want[p0 - 1 + i, t] += 40
    base = {"a_ids": np.array(seq), "a_prompt": np.array([p0]), "a_gen": np.array(gen)}
    golden.save_wants(base, {"a": want}, eos, tmp_path / "ref.bf16.npz")
    same = dict(base, a_tail=want[p0 - 1:].numpy().astype(np.float16), a_ptail=want[:p0 - 1].numpy().astype(np.float16))
    off = dict(same, a_tail=same["a_tail"].copy())
    off["a_tail"][2, 3] += 5.0
    np.savez(tmp_path / "same.npz", **same)
    np.savez(tmp_path / "off.npz", **off)
    rep = golden.compare_saved(str(tmp_path / "ref.bf16.npz"), [str(tmp_path / "same.npz"), str(tmp_path / "off.npz")])
    s, o = rep[str(tmp_path / "same.npz")], rep[str(tmp_path / "off.npz")]
    assert s["a"]["rows"] == 4 and s["a"]["greedy_steps_equal"] == "4/4"
    assert s["a"]["kl_mean"] < 1e-4 and s["a"]["kl_prompt_mean"] < 1e-4
    lp = want[p0 + 1].log_softmax(-1)
    lq = torch.from_numpy(off["a_tail"][2].astype(np.float32)).log_softmax(-1)
    one = float((lp.exp() * (lp - lq)).sum())
    assert abs(o["a"]["kl_max"] - one) < 2e-3 * max(one, 1e-3)
