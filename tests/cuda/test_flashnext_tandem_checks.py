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

    def __init__(self, prompt, ref, rows=4):
        self.n, self.ref, self.ctx, self.asked, self.rows = len(prompt), ref, [], 0, rows

    def extend(self, tokens):
        self.ctx.extend(int(t) for t in tokens)

    def chain(self, n):
        self.asked += 1
        at = len(self.ctx) - self.n
        return self.ref[at:at + min(n, self.rows - 1)] if self.asked % 2 else []

    def verified(self, drafted, kept):
        pass


def _reference(w, prompt, count):
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    return serial_decode(e, prefill(e, prompt, None), count, None).tokens


def test_drafts_are_counted_by_drafter_in_shared_rounds(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_fill

    w = _model()
    ref = _reference(w, PROMPT, 60)
    monkeypatch.setattr(multi_fill, "CopyIndex", lambda ctx, **_: _seeded(Oracle(PROMPT, ref), ctx))
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


class Wide(Oracle):
    """The oracle with a 16-row copy window, recording the widest copy it proposed."""

    def __init__(self, prompt, ref):
        super().__init__(prompt, ref, rows=16)
        self.widest = 0

    def chain(self, n):
        got = super().chain(n)
        self.widest = max(self.widest, len(got))
        return got


def test_wide_copy_windows_give_the_serial_tokens_in_shared_rounds(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_fill

    w = _model()
    ref = _reference(w, PROMPT, 60)
    made = []

    def index(ctx, **_):
        made.append(_seeded(Wide(PROMPT, ref), ctx))
        return made[-1]

    monkeypatch.setattr(multi_fill, "CopyIndex", index)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.0, copy=True, copy_rows=16)
    streams = [Stream(PROMPT, 60, None), Stream(PROMPT, 60, None)]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == [ref, ref]
    assert max(o.widest for o in made) == 15                     # wider than the MTP chain's 3 drafts


@pytest.mark.parametrize("graphs", [False, True])
def test_wide_copy_windows_give_the_serial_tokens_on_one_engine(graphs):
    w = _model()
    ref = _reference(w, PROMPT, 60)
    e = Engine(w, capacity=1024, max_rows=16, prefill_rows=16, graphs=graphs, graph_rows=8)  # 9-16 rows run eagerly
    first = prefill(e, PROMPT, None)
    oracle = _seeded(Wide(PROMPT, ref), PROMPT + [first])
    res = mtp_decode(e, first, 60, None, depth=3, confidence=0.0, copies=oracle)
    assert res.tokens == ref and oracle.widest == 15 and max(res.widths) == 16


@pytest.mark.parametrize("graphs", [False, True])
def test_every_copy_width_gives_the_serial_bits_on_one_engine(graphs):
    """Copy windows verify up to 32 rows; graphs stop at 8, so 9-32 rows run eagerly (upstream 3e3f895e: a path picked
    by row count changed bits once verify windows grew past 16)."""
    e = Engine(_model(), capacity=1024, max_rows=32, prefill_rows=16, graphs=graphs, graph_rows=8)
    got = check_widths(*engine_windows(e, PROMPT), widths=range(2, 33), steps=64)
    assert got["ok"], got


def test_every_copy_width_gives_the_serial_bits_in_shared_rounds():
    dec = MultiDecoder(_model(), slots=2, capacity=1024, depth=6, confidence=0.7, graphs=False, copy_rows=32)
    start, window, release = multi_windows(dec, PROMPT)
    got = check_widths(start, window, widths=range(2, 33), steps=64)
    release()
    assert got["ok"], got


def _f16_downs(w):
    """``w`` with every hyper-connection's down projection an EXL3 pack's fp16 matrix (``exl3_mm.F16``), whose decode
    call takes at most ``exl3_mm.ROWS`` rows."""

    from test_flashnext_forward import DEV, LOW, D, S

    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    sc = exl3_mm.Scratch(11, False)
    g = torch.Generator().manual_seed(11)
    hcs = [hc for layer in [*w.layers, w.mtp.layer] for hc in (layer.attn_hc, layer.mlp_hc)] + [w.mixer, w.mtp.mixer]
    for hc in hcs:
        n = LOW + (S if hc.inject else 0)
        hc.down = exl3_mm.f16(sc, [(torch.randn((n, S * D), generator=g) * 0.01).half()], DEV)
    sc.part = torch.empty((max(hc.down.sk * hc.down.n for hc in hcs) * exl3_mm.ROWS,), dtype=torch.float32,
                          device=DEV)
    return w


class Always(Oracle):
    """The oracle copying on every round, so streams that all copy share each round."""

    def chain(self, n):
        at = len(self.ctx) - self.n
        return self.ref[at:at + min(n, self.rows - 1)]


def test_six_streams_of_full_copy_windows_give_the_serial_tokens(monkeypatch):
    """Six streams copying 31 drafts each verify 6 x 32 = 192 rows in one round, past the 128 rows an EXL3 pack's HC
    down projection takes in one call (tf-runs/parallel6, 2026-10-21: four streams fit exactly)."""

    from tensorfold.families.qwen4_exp.cuda import exl3_mm, multi, multi_fill

    w = _f16_downs(_model())
    prompts = [PROMPT[i:] + PROMPT[:i] for i in range(6)]
    refs = [_reference(w, p, 100) for p in prompts]

    def index(ctx, **_):
        i = next(i for i, p in enumerate(prompts) if list(ctx[:len(p)]) == p)
        return _seeded(Always(prompts[i], refs[i], rows=32), ctx)

    widest = []

    def stage(w, b, windows):
        segs = real(w, b, windows)
        widest.append(segs[-1][2])
        return segs

    real = multi.stage
    monkeypatch.setattr(multi, "stage", stage)
    monkeypatch.setattr(multi_fill, "CopyIndex", index)
    dec = MultiDecoder(w, slots=6, capacity=1024, depth=3, confidence=0.0, copy=True, copy_rows=32)
    streams = [Stream(p, 100, None) for p in prompts]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.error for s in streams] == [None] * 6
    assert [s.out for s in streams] == refs
    assert max(widest) == 192 > exl3_mm.ROWS
