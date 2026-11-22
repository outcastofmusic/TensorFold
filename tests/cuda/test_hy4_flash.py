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


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("head", ["nvfp4", "slice"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_dfly_drafted_equals_serial(weights, goldens, temperature, head, graphs, tmp_path, monkeypatch):
    """DFly2's real rounds (random weights: few drafts land) and copies, keyed drafts, either draft head: the reply is
    serial decoding's, also when the verify windows replay graphs and DFly2 reads the rows a graph keeps."""

    monkeypatch.setenv("TF_HY4_DRAFT_HEAD", head)
    monkeypatch.setenv("TF_HY4_DRAFT_VOCAB", "128")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode

    prompt = goldens["long_ids"].tolist()[:20]
    s = Sampling(seed=5, temperature=temperature)
    e = _engine(weights)
    serial = serial_decode(e, prefill(e, prompt, s), 40, s, stop_eos=False).tokens
    e = _engine(weights, graphs=graphs)
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


def test_graphed_calls_keep_the_eager_rows(weights, goldens):
    """After every graphed call, the first (captured) and later (replayed) ones of each width, the normed rows and the
    watched lists a drafter reads are an eager engine's."""

    ids = goldens["long_ids"].tolist()
    engines = []
    for graphs in (False, True):
        e = _engine(weights, graphs=graphs)
        e.watch = {li for li, layer in enumerate(e.w.layers) if layer.attn.indexer is not None}
        e.forward(ids[:26], prompt=True)          # the windows cross the tiny indexer's 32-key limit
        engines.append(e)
    assert engines[0].watch
    for width in (6, 1, 3, 6, 1, 16, 3):
        for e in engines:
            e.forward(ids[e.pos:e.pos + width], prompt=False, every=True)
        want, got = engines
        assert torch.equal(got.normed, want.normed), width
        for li in want.watch:
            (sel, tok, cnt), (gsel, gtok, gcnt) = want.lists[li], got.lists[li]
            assert torch.equal(sel, gsel) and torch.equal(cnt, gcnt), (width, li)
            sparse = sel != 0                        # a graph always ranks groups; dense rows read no token list
            assert torch.equal(tok[sparse], gtok[sparse]), (width, li)
        for e in engines:
            e.rewind(width - 1)


def test_nvfp4_head_is_the_dense_head_bit_for_bit(weights):
    """``Head4`` sums code x block scale (exact in bf16) on the same tile and K order as the bf16 head, then applies the
    global scale: its logits are the dense head's on those values times the scale, at every row count."""

    from tensorfold.families.hy4_flash.cuda import kernels as K
    from tensorfold.families.hy4_flash.cuda.dfly import quantize_nvfp4

    codes, scales, scale2 = quantize_nvfp4(weights.head)
    h4 = K.Head4(codes, scales, scale2)
    w = (h4.dense() / scale2).to(torch.bfloat16)
    x = torch.randn(16, weights.head.shape[1], device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    x = x.to(torch.bfloat16)
    want = K.head(x, w) * scale2
    assert torch.equal(h4(x), want)
    for rows in (1, 6):
        assert torch.equal(h4(x[:rows]), want[:rows])


@pytest.mark.parametrize("rows", [1, 6, 40])
def test_row_fp8_kv_b_matches_its_dense_values(weights, rows):
    """absorb and expand on e4m3 kv_b rows with a scale each, against fp32 math on the decoded values."""

    from tensorfold.families.hy4_flash.cuda import kernels as K

    a = weights.layers[2].attn
    g = torch.Generator("cuda").manual_seed(rows)
    for w in (a.wk, a.wv):
        top = w.float().abs().amax(-1, keepdim=True)
        s = torch.where(top > 0, top / 448.0, torch.ones_like(top))
        f8 = K.RowF8((w.float() / s).to(torch.float8_e4m3fn), s[..., 0])
        dense = f8.dense()
        heads, n, lw = w.shape
        if w is a.wk:
            q = torch.randn(rows, heads, n, device="cuda", generator=g).to(torch.bfloat16)
            got = K.absorb(q, f8, torch.empty((rows, heads, lw), dtype=torch.bfloat16, device="cuda"))
            want = torch.einsum("rhd,hdn->rhn", q.float(), dense)
        else:
            o = torch.randn(rows, heads, lw, device="cuda", generator=g).to(torch.bfloat16)
            got = K.expand(o, f8, torch.empty((rows, heads, n), dtype=torch.bfloat16, device="cuda"))
            want = torch.einsum("rhn,hdn->rhd", o.float(), dense)
        assert float((got.float() - want).abs().max() / want.abs().max()) < 0.01


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_nvfp4_trunk_drafted_equals_serial(goldens, temperature, tmp_path, monkeypatch):
    """With the trunk in NVFP4 and kv_b in row FP8 (``TF_HY4_TRUNK``), DFly2 rounds with graphs give serial
    decoding's reply: the NVFP4 linears and head keep a row's bits in any window."""

    import importlib.util
    import re

    from safetensors import safe_open
    from safetensors.torch import save_file

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode
    from tensorfold.families.hy4_flash.cuda.weights import load

    spec = importlib.util.spec_from_file_location("hy4_trunk", Path(__file__).parents[2] / "tools" / "hy4_trunk.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    ckpt = tiny.write(tmp_path / "tiny")
    pats = [re.compile(p) for p in tool.CLASSES.values()]
    out = {}
    with safe_open(str(ckpt / "model-00001.safetensors"), "pt") as f:
        for name in f.keys():  # noqa: SIM118 (a safetensors handle is not iterable)
            base = name.removesuffix(".weight")
            if base == name or not any(p.search(base) for p in pats) or f.get_slice(name).get_dtype() != "BF16":
                continue                    # the tiny checkpoint's FP8 attention stays as stored
            w = f.get_tensor(name)
            if base.endswith("kv_b_proj"):
                out[name], out[base + ".weight_scale_row"], _ = tool.quantize_rows(w)
            else:
                codes, sc, s2, _ = tool.quantize(w)
                out[name], out[base + ".weight_scale"], out[base + ".weight_scale_2"] = codes, sc, torch.tensor(s2)
    (tmp_path / "trunk").mkdir()
    save_file(out, str(tmp_path / "trunk" / "trunk.safetensors"))
    monkeypatch.setenv("TF_HY4_TRUNK", str(tmp_path / "trunk"))
    w = load(ckpt, "cuda")
    assert not isinstance(w.head, torch.Tensor)
    prompt = goldens["long_ids"].tolist()[:20]
    s = Sampling(seed=5, temperature=temperature)
    e = _engine(w)
    serial = serial_decode(e, prefill(e, prompt, s), 40, s, stop_eos=False).tokens
    e = _engine(w, graphs=True)
    drafter = DFly(ckpt, e)
    got = draft_decode(e, prompt, prefill(e, prompt, s), 40, s, stop_eos=False, drafter=drafter)
    assert got.tokens == serial
    assert got.stats["drafted"] > 0
