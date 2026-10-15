"""The load-time self-check on the real kernels, and drafts counted by drafter (copy against MTP)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.self_check import check_widths, engine_windows, multi_windows  # noqa: E402

PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]


@pytest.mark.parametrize("graphs", [False, True])
def test_every_verify_width_gives_the_serial_bits_on_one_engine(graphs):
    e = Engine(_model(), capacity=1024, max_rows=8, prefill_rows=16, graphs=graphs)
    got = check_widths(*engine_windows(e, PROMPT), widths=range(2, 8), steps=24)
    assert got["ok"], got


def test_every_verify_width_gives_the_serial_bits_in_shared_rounds():
    dec = MultiDecoder(_model(), slots=2, capacity=1024, depth=6, confidence=0.7, graphs=False)
    start, window, release = multi_windows(dec, PROMPT)
    got = check_widths(start, window, widths=range(2, 8), steps=24)
    release()
    assert got["ok"], got
    assert len(dec.free) == 2 and not dec.live()                # the check gives its slot back


class Oracle:
    """A copy index that proposes the serial reply's next tokens on every other round (the tiny random model's reply
    never repeats itself, so a real index finds nothing); MTP drafts the rounds between."""

    def __init__(self, prompt, ref):
        self.n, self.ref, self.ctx, self.asked = len(prompt), ref, [], 0

    def extend(self, tokens):
        self.ctx.extend(int(t) for t in tokens)

    def chain(self, n):
        self.asked += 1
        at = len(self.ctx) - self.n
        return self.ref[at:at + n] if self.asked % 2 else []


def _reference(w, prompt, count):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    return serial_decode(e, prefill(e, prompt, None), count, None).tokens


def test_drafts_are_counted_by_drafter_in_shared_rounds(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_fill

    w = _model()
    ref = _reference(w, PROMPT, 60)
    monkeypatch.setattr(multi_fill, "CopyIndex", lambda ctx: _seeded(Oracle(PROMPT, ref), ctx))
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.0, copy=True, count_drafters=True)
    s = Stream(PROMPT, 60, None)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert s.out == ref
    d = s.drafters
    assert sum(v[0] for v in d.values()) == s.drafted and sum(v[1] for v in d.values()) == s.accepted
    assert d["copy"][0] > 0 and d["copy"][0] == d["copy"][1] and d["mtp"][0] > 0     # the oracle's drafts all land


def _seeded(oracle, ctx):
    oracle.extend(ctx)
    return oracle


def test_drafts_are_counted_by_drafter_on_one_engine():
    w = _model()
    ref = _reference(w, PROMPT, 60)
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(e, PROMPT, None)
    res = mtp_decode(e, first, 60, None, depth=3, confidence=0.0, copies=_seeded(Oracle(PROMPT, ref), PROMPT + [first]))
    assert res.tokens == ref
    assert sum(v[0] for v in res.drafters.values()) == res.drafted
    assert sum(v[1] for v in res.drafters.values()) == res.accepted
    assert res.drafters["copy"][0] > 0 and res.drafters["copy"][0] == res.drafters["copy"][1]
