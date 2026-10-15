"""End anchors: the state where a reply ended is kept, so a next turn that sends the reply back resumes after it.

A decode-end state carries decode bits, which differ from a prefill of the same tokens in the last bits, so the checks
are the resume's bookkeeping (where it resumes, the MTP head's rows) and the first token, not every bit."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]


def _fresh(w, prompt):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(e, prompt, None)
    return e, first


@pytest.mark.parametrize("graphs", [False, True])
def test_a_reply_end_resumes_the_next_turn_on_one_engine(graphs):
    w = _model()
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=graphs)
    first = prefill(e, PROMPT, None)
    res = mtp_decode(e, first, 30, None, depth=3, confidence=0.0, keep_end=True)
    ids = PROMPT + res.committed
    assert res.end is not None and res.end["state"]["pos"] == len(ids)
    assert res.end["state"]["mtp_len"] == len(ids) - 1                  # the head lags one row, as a kept prompt
    nxt = ids + [res.tokens[-1], 42, 43]
    fresh_e, fresh_first = _fresh(w, nxt)
    got = prefill(e, nxt, None, resume=res.end)
    assert e.st.pos == len(nxt) and e.st.mtp_len == fresh_e.st.mtp_len
    assert got == fresh_first


@pytest.mark.parametrize("graphs", [False, True])
def test_a_reply_end_resumes_the_next_turn_in_shared_rounds(graphs):
    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.0, graphs=graphs, end_anchor=True)
    s = Stream(PROMPT, 30, None)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    ids = PROMPT + s.committed
    assert any(k[0] == ids for k in dec.kept)
    nxt = ids + [s.out[-1], 42, 43]
    t = Stream(nxt, 12, None)
    dec.admit(t)
    assert t.cached == len(ids)
    while dec.live():
        dec.finish(dec.round())
    _, fresh_first = _fresh(w, nxt)
    assert t.out[0] == fresh_first
