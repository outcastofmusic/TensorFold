"""TENSORFOLD_PROMPT_TAIL's even last pieces commit the plain cut's bits on the tiny NVFP4 checkpoint.

At 4,135 and 8,233 tokens (sparkDash lengths) the plain cut ends in a 39- or 41-row piece and the even cut in two of
about 1,040. Pieces of 2,048 rows, and of 4,096 while TENSORFOLD_LONG_PIECE_TOKENS (8,192) rows remain as FP4-Local
serves: the last row's logits, the first token, the committed state and every KV cache row, and a greedy reply. With
the expert tile from 256 rows (TENSORFOLD_EXPERT_TILE_ROWS) the 1,040-row pieces run the tile and the 39-row piece the
16-pair kernel, so the bits also hold the tile and the 16-pair kernel equal at those sizes (tf-runs/prefill-gap-1028
M3). Runs wherever a GPU does; queue it, since it compiles the prompt kernels."""

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA state comparisons require an NVIDIA GPU", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402
from test_flashnext_nvfp4_pieces import _assert_same_state, _bits, _last_logits, _prompt  # noqa: E402

from tensorfold.cuda.nvfp4 import experts as nvx  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, chunk_plan, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.prompt_plan import even_tails, long_pieces  # noqa: E402

CAPACITY = 8448
REPLY = 32
# (prompt rows, buffer rows, long-piece threshold): the plain and even cuts of each
CASES = {
    (4135, 2048, 0): ([2048, 2048, 39], [2048, 1040, 1047]),
    (8233, 2048, 0): ([2048] * 4 + [41], [2048] * 3 + [1040, 1049]),
    (4135, 4096, 8192): ([2048, 2048, 39], [2048, 1040, 1047]),
    (8233, 4096, 8192): ([4096, 2048, 2048, 41], [4096, 2048, 1040, 1049]),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(write(tmp_path_factory.mktemp("prompt-tail"), layers=4, experts=16, hidden=512, mxfp8=True,
                   ple_nvfp4=True), mtp=True)
    yield w
    del w
    torch.cuda.empty_cache()


def _fill(w, prompt, rows: int, long_from: int, tail: bool, kv_dtype: str):
    """An engine filled with ``prompt``'s pieces as served: (its cut, engine, last logits, first token)."""

    e = Engine(w, capacity=CAPACITY, max_rows=8, prefill_rows=rows, kv_dtype=kv_dtype)
    with long_pieces(long_from), even_tails(tail):
        cut = [b - a for a, b, _ in chunk_plan(0, len(prompt), rows, ())]
        logits = _bits(_last_logits(e, prompt))
        first = prefill(e, prompt, None)
    return cut, e, logits, first


@pytest.mark.parametrize("tile_rows", [nvx.TILE_ROWS, 256])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
@pytest.mark.parametrize("case", list(CASES), ids=lambda c: f"{c[0]}-rows{c[1]}-long{c[2]}")
def test_the_even_tail_keeps_the_plain_cuts_bits(model, monkeypatch, case, kv_dtype, tile_rows):
    n, rows, long_from = case
    monkeypatch.setattr(nvx, "TILE_ROWS", tile_rows)
    prompt = _prompt(n, model.cfg.vocab)
    cut0, plain, logits0, first0 = _fill(model, prompt, rows, long_from, False, kv_dtype)
    cut1, even, logits1, first1 = _fill(model, prompt, rows, long_from, True, kv_dtype)
    assert (cut0, cut1) == CASES[case]
    assert logits1 == logits0                            # the last row's logits
    assert first1 == first0
    _assert_same_state(even.st, plain.st)                # recurrent state, KV codes and scales
    replies = [serial_decode(e, first0, REPLY, None, stop_eos=False).tokens for e in (plain, even)]
    assert len(replies[0]) == REPLY and replies[1] == replies[0]
