"""StairCut and the plain gate on Flash Next's CUDA decoders: they change the rows a round verifies, never the text."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("Flash Next CUDA kernels run on sm_12x (GB10, RTX 50, RTX PRO 6000) only",
                allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.staircut import Speculation  # noqa: E402

PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
PRIOR = {0: {r: 2.0 + 0.4 * (r - 1) for r in range(1, 17)}}
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95)]


def _trusting(spec: Speculation) -> Speculation:
    """A calibration that has seen every draft land: the cut verifies wide windows until misses teach it otherwise
    (the tiny random model's head rarely lands one)."""

    for b in spec.cut.cal.bins:
        b[0] = b[1] = 50.0
    return spec


def _losing_gate(spec: Speculation) -> Speculation:
    """A gate that has learned drafting loses at class 0: most rounds are plain, one in 16 probes."""

    spec.gate.spec[0] = [1000.0, 1.0, spec.gate.FIRST]
    spec.gate.plain[0] = 1.0
    return spec


@pytest.mark.parametrize("sampling", SAMPLINGS)
@pytest.mark.parametrize("graphs", [False, True])
def test_staircut_chains_give_serial_tokens(sampling, graphs):
    w = _model()
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=graphs)
    first = prefill(e, PROMPT, sampling)
    ref = serial_decode(e, first, 40, sampling).tokens
    for depth in (2, 6):
        spec = _trusting(Speculation(cut=True, gate=False, confidence=0.7, depth=depth, prior=PRIOR))
        prefill(e, PROMPT, sampling)
        got = mtp_decode(e, first, 40, sampling, depth=depth, spec=spec)
        assert got.tokens == ref, (depth, graphs)
        assert max(got.widths) == depth + 1                              # windows as wide as the depth allows
        assert spec.stats["rounds"] + spec.stats["plain"] == got.rounds
        assert sum(b[1] for b in spec.cut.cal.bins) > 10 * 50.0          # every verified draft taught it


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_plain_rounds_give_serial_tokens_and_keep_the_head_absorbing(sampling):
    w = _model()
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    first = prefill(e, PROMPT, sampling)
    ref = serial_decode(e, first, 48, sampling).tokens
    spec = _losing_gate(Speculation(cut=False, gate=True, confidence=0.7, depth=3, prior=PRIOR))
    prefill(e, PROMPT, sampling)
    got = mtp_decode(e, first, 48, sampling, depth=3, spec=spec)
    assert got.tokens == ref
    assert got.widths.count(1) >= 30 and max(got.widths) > 1            # plain rounds, and a probe that drafted
    assert spec.stats["plain"] == got.widths.count(1)
    assert e.st.mtp_len == e.st.pos                                      # the head followed every committed row


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("gate", [False, True])
def test_shared_rounds_with_staircut_equal_each_stream_alone(graphs, gate):
    w = _model()
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 24, sampling).tokens)
    spec = _trusting(Speculation(cut=True, gate=gate, confidence=0.7, depth=3, prior=PRIOR))
    if gate:
        _losing_gate(spec)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.7, graphs=graphs, spec=spec)
    streams = []
    for prompt, sampling in zip(PROMPTS, samplings):
        s = Stream(prompt, 24, sampling)
        dec.admit(s)
        streams.append(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs
    assert spec.stats["rounds"] + spec.stats["plain"] > 0
    # alone again (the graph slot when ``graphs``): the same tokens
    for prompt, sampling, ref in zip(PROMPTS, samplings, refs):
        s = Stream(prompt, 24, sampling)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        assert s.out == ref


@pytest.mark.parametrize("graphs", [False, True])
def test_warm_up_prices_each_window_and_leaves_decoding_exact(graphs):
    w = _model()
    sampling = Sampling(seed=1234, top_k=20, top_p=0.95)
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
    ref = serial_decode(e, prefill(e, PROMPTS[1], sampling), 24, sampling).tokens
    spec = _trusting(Speculation(cut=True, gate=False, confidence=0.7, depth=3))
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.7, graphs=graphs, spec=spec)
    dec.warm()
    prior = spec.prices.prior[0]
    assert sorted(prior) == [1, 2, 3, 4] and all(ms > 0 for ms in prior.values())
    s = Stream(PROMPTS[1], 24, sampling)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert s.out == ref
