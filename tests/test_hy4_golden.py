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
