"""Idle graph captures: the next context bucket's decode graphs are recorded between requests, so a request that grows
into that bucket replays them instead of capturing mid-reply, and its tokens stay the serial ones."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402


@pytest.mark.parametrize("sampling", [None, Sampling(seed=4, top_k=20, top_p=0.95)])
def test_idle_captures_cover_the_next_bucket_and_change_no_token(sampling):
    w = _model()
    g = torch.Generator().manual_seed(21)
    short = torch.randint(1, V, (40,), generator=g).tolist()
    long = torch.randint(1, V, (8300,), generator=g).tolist()          # its rounds run in the 16,384 bucket
    eager = Engine(w, capacity=20000, max_rows=8, prefill_rows=512)
    ref = serial_decode(eager, prefill(eager, long, sampling), 24, sampling).tokens

    e = Engine(w, capacity=20000, max_rows=8, prefill_rows=512, graphs=True)
    e.graphs.warm(4)
    e.reset()
    first = prefill(e, short, sampling)
    mtp_decode(e, first, 16, sampling, depth=3, confidence=0.0)
    missing = e.graphs.missing(e.st.pos + 8)
    assert {k[-1] for k in missing} == {16384}                          # the 8,192 bucket was warmed
    for step in e.graphs.idle_steps(e.st.pos + 8):
        step()
    assert e.graphs.missing(e.st.pos + 8) == []
    captured = e.graphs.captures
    first = prefill(e, long, sampling)
    got = mtp_decode(e, first, 24, sampling, depth=3, confidence=0.0)
    assert got.tokens == ref
    assert e.graphs.captures == captured                                # nothing captured mid-reply
