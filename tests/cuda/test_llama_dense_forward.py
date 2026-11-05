"""Dense Llama on CUDA, every weight format: a verify window's rows equal serial steps bit for bit at each width up
to 128, a partial keep commits the serial state, drafted replies equal serial ones, any prompt chunking gives the same
state, a resumed or kept prompt state equals a fresh one, and bf16 logits track the fp32 reference."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import precision
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.llama_dense.cuda.decode import draft_decode, prefill, serial_decode
from tensorfold.families.llama_dense.cuda.forward import State, new_state, tree_forward
from tensorfold.families.llama_dense.cuda.prefill import prefill_state
from tensorfold.families.llama_dense.cuda.weights import load
from tensorfold.families.qwen3_5.cuda.decode import clone_state
from tensorfold.families.qwen3_5.cuda.forward import commit
from tests import llama_tiny

FORMATS = ["bf16", "exl3", "exl3-plain-head", "nvfp4-full", "nvfp4-checkpoint"]
V = llama_tiny.VOCAB
LONG = 2600                     # past 2,048 committed keys: wide windows fold whole key groups, one-row windows not


@pytest.fixture(scope="module", params=FORMATS)
def w(request, tmp_path_factory):
    kind = request.param
    folder = tmp_path_factory.mktemp(kind)
    if kind == "bf16":
        llama_tiny.write(folder)
    elif kind.startswith("exl3"):
        llama_tiny.write_exl3(folder, head=kind == "exl3")
    else:
        llama_tiny.write_nvfp4(folder)
    with precision.using("checkpoint" if kind == "nvfp4-checkpoint" else "full"):
        return load(folder)


def _prompt(n: int, seed: int = 5) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, V, (n,), generator=g).tolist()


def _ids(tokens) -> torch.Tensor:
    return torch.tensor(tokens, dtype=torch.int32, device="cuda")


def _prefilled(w, n: int = LONG) -> State:
    st = new_state(w)
    prefill_state(w, _prompt(n), st)
    return st


def _serial(w, tokens: list[int], n: int = LONG):
    st, rows = _prefilled(w, n), []
    for t in tokens:
        logits, record = tree_forward(w, _ids([t]), [-1], st)
        rows.append(logits[0])
        commit(st, record, [0])
    return rows, st


def _same_kv(a: State, b: State, lo: int, hi: int) -> None:
    for x, y in zip(a.kv, b.kv):
        assert torch.equal(x[0][lo:hi], y[0][lo:hi]) and torch.equal(x[1][lo:hi], y[1][lo:hi])


def test_window_rows_equal_serial_steps(w):
    tokens = _prompt(128, seed=9)
    serial, serial_st = _serial(w, tokens)
    base = _prefilled(w)
    for width in range(1, 129):                                 # a window reads the state and never writes it
        logits, _ = tree_forward(w, _ids(tokens[:width]), list(range(-1, width - 1)), base)
        assert base.pos == LONG
        for r in range(width):
            assert torch.equal(logits[r], serial[r]), f"width {width}, row {r}"
    for width in (2, 3, 16, 17, 33, 64, 127, 128):
        st = _prefilled(w)
        logits, record = tree_forward(w, _ids(tokens[:width]), list(range(-1, width - 1)), st)
        assert st.pos == LONG                                   # nothing committed before ``commit``
        for r in range(width):
            assert torch.equal(logits[r], serial[r]), f"width {width}, row {r}"
        keep = width // 2 + 1                                   # a partial keep: the serial state after ``keep`` steps
        commit(st, record, list(range(keep)))
        assert st.pos == LONG + keep
        _same_kv(st, serial_st, 0, LONG + keep)
        nxt, _ = tree_forward(w, _ids(tokens[keep:keep + 4]), [-1, 0, 1, 2], st)
        for r in range(4):
            assert torch.equal(nxt[r], serial[keep + r]), f"after keeping {keep} of {width}, row {r}"


SAMPLINGS = [None, Sampling(1234, 0.8, 20, 0.95)]


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_drafted_reply_equals_serial(w, sampling, monkeypatch):
    """Copies that are right, then break every third round, verified in windows that double from 16 to 128 rows
    while they land whole and halve when they break: the reply is serial decoding's, token for token."""

    from tensorfold.families.llama_dense.cuda import decode

    prompt = _prompt(300, seed=3)
    st, pending, _ = prefill(w, prompt, sampling)
    ref = serial_decode(w, st, pending, 400, sampling, stop_eos=False).tokens

    class Oracle:
        calls = 0

        def propose(self, context, max_nodes):
            k = len(context) - len(prompt)                      # the reply so far; ref[k:] follows it
            right = ref[k:k + max_nodes]
            self.calls += 1
            if self.calls % 3 == 0 and len(right) > 2:
                cut = len(right) // 2
                right = right[:cut] + [(right[cut] + 1) % V] + right[cut + 1:]
            return right

    monkeypatch.setattr(decode, "CopyIndex", Oracle)
    st, pending, _ = prefill(w, prompt, sampling)
    trace: list[dict] = []
    got = draft_decode(w, st, prompt, pending, 400, sampling, stop_eos=False, trace=trace)
    assert got.tokens == ref
    assert {16, 32, 64, 128} <= {t["rows"] for t in trace}
    assert any(0 < t["accepted"] < t["rows"] - 1 for t in trace)          # partial keeps happened


def test_context_copies_keep_serial_tokens(w):
    """The real copy index on a repetitive prompt: whatever it proposes, the reply is serial's."""

    prompt = _prompt(48, seed=3) * 5
    st, pending, _ = prefill(w, prompt, None)
    ref = serial_decode(w, st, pending, 160, None, stop_eos=False).tokens
    st, pending, _ = prefill(w, prompt, None)
    assert draft_decode(w, st, prompt, pending, 160, None, stop_eos=False).tokens == ref


def test_eos_and_count_stop_both_decodes(w):
    st, _, _ = prefill(w, _prompt(20), None)
    assert serial_decode(w, st, 2, 32, None).tokens == [2]
    assert draft_decode(w, st, _prompt(20), 2, 32, None).tokens == [2] and st.pos == 20
    assert len(draft_decode(w, st, _prompt(20), 7, 5, None, stop_eos=False).tokens) == 5


@pytest.mark.parametrize("size", [1, 7, 64, 512])
def test_every_chunking_gives_the_same_state(w, size):
    prompt = _prompt(700)
    whole, fresh = new_state(w), new_state(w)
    h_whole, _ = prefill_state(w, prompt, whole)
    h_parts, _ = prefill_state(w, prompt, fresh, size=size)
    assert torch.equal(h_whole, h_parts)
    _same_kv(whole, fresh, 0, 700)


def test_resumed_and_kept_states_equal_fresh_ones(w):
    """A state resumed at any row, and a clone kept at any point of a longer prefill, equal fresh prefills there."""

    prompt = _prompt(700)
    fresh = new_state(w)
    h_fresh, kept = prefill_state(w, prompt, fresh, keep=[1, 299, 513])
    resumed = new_state(w)
    prefill_state(w, prompt[:299], resumed, size=64)
    h_resumed, _ = prefill_state(w, prompt, resumed, size=100)
    assert torch.equal(h_fresh, h_resumed)
    _same_kv(fresh, resumed, 0, 700)
    for st, p in zip(kept, (1, 299, 513)):
        alone = new_state(w)
        prefill_state(w, prompt[:p], alone)
        assert st.pos == p
        _same_kv(st, alone, 0, p)
        h_from_kept, _ = prefill_state(w, prompt, clone_state(st))
        assert torch.equal(h_from_kept, h_fresh)


def test_bf16_logits_track_the_fp32_reference(tmp_path):
    """Exactness is against serial; quality is against transformers' math in fp32 (``reference.forward``)."""

    from tensorfold.families.llama_dense.cuda.weights import Config
    from tensorfold.families.llama_dense.reference import forward

    folder = llama_tiny.write(tmp_path / "bf16")
    w = load(folder)
    stored = {k: v.cuda() for k, v in llama_tiny.tensors().items()}
    prompt = _prompt(300)
    want = forward(stored, Config.read(folder), prompt)
    st = new_state(w)
    normed, _ = prefill_state(w, prompt, st, every=True)
    prompt_logits = torch.nn.functional.linear(normed.float(), w.head.weight.float())
    decode_logits = []
    st = new_state(w)
    for t in prompt:
        logits, record = tree_forward(w, _ids([t]), [-1], st)
        decode_logits.append(logits[0].float())
        commit(st, record, [0])
    for got in (prompt_logits, torch.stack(decode_logits)):
        err = (got - want).abs().max() / want.abs().max()
        cos = torch.nn.functional.cosine_similarity(got, want, dim=-1).min()
        agree = (got.argmax(-1) == want.argmax(-1)).float().mean()
        assert err < 0.03 and cos > 0.999 and agree > 0.95, (float(err), float(cos), float(agree))
