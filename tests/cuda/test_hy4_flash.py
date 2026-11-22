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


def test_nvfp4_head_keeps_a_row_s_bits_in_any_window(weights):
    """``Head4`` gives a row the same fp32 logits alone, in a 6-row window and in a 16-row tile (verify windows sample
    them against serial steps), within 1% of its dense values. The mma on codes decoded in registers need not sum in
    the dense head kernel's order, so it is not that kernel's bits (window 2, 22 Nov)."""

    from tensorfold.families.hy4_flash.cuda import kernels as K
    from tensorfold.families.hy4_flash.cuda.dfly import quantize_nvfp4

    codes, scales, scale2 = quantize_nvfp4(weights.head)
    h4 = K.Head4(codes, scales, scale2)
    x = torch.randn(16, weights.head.shape[1], device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    x = x.to(torch.bfloat16)
    full = h4(x)
    for rows in (1, 6):
        assert torch.equal(h4(x[:rows]), full[:rows])
    want = x.float() @ h4.dense().t()
    assert float((full - want).abs().max() / want.abs().max()) < 0.01


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


def test_fused_small_ops_keep_the_torch_bits(monkeypatch):
    """RoPE, SwiGLU, the attention gate and the MoE's sum and norm in one launch each give the torch chains' bits, on
    the shapes and value ranges decode and prompt rows give them (positions to 131,072, strided slices)."""

    from tensorfold.families.hy4_flash.cuda import kernels as K
    from tensorfold.families.hy4_flash.cuda import model

    g = torch.Generator("cuda").manual_seed(7)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

    inv = (1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))).cuda()
    for rows in (1, 6, 2048):
        pos = torch.randint(0, 131072, (rows,), device="cuda", generator=g, dtype=torch.int32)
        wide = rnd(rows, 8, 256, scale=4.0)
        for x in (wide[:, :, 192:], rnd(rows, 576)[:, 512:], rnd(rows, 64)):
            monkeypatch.setattr(model, "FUSED", False)
            want = model.Engine._rope(x, pos, inv)
            monkeypatch.setattr(model, "FUSED", True)
            assert torch.equal(model.Engine._rope(x, pos, inv), want)
        a, b = rnd(rows, 1536, scale=6.0), rnd(rows, 1536, scale=2.0)
        monkeypatch.setattr(model, "FUSED", False)
        want = model._silu_mul(a, b)
        monkeypatch.setattr(model, "FUSED", True)
        assert torch.equal(model._silu_mul(a, b), want)
        v, gate = rnd(rows, 32, 256), rnd(rows, 8192, scale=5.0)
        sig = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
        assert torch.equal(K.gate_mul(v, gate), (v.reshape(rows, -1).float() * sig).to(torch.bfloat16))
        r, s, w = rnd(rows, 3072), rnd(rows, 3072), rnd(3072) + 1
        want = K.rmsnorm((r.float() + s.float()).to(torch.bfloat16), w, 1e-5)
        assert torch.equal(K.add_rmsnorm(r, s, w, 1e-5), want)


def test_fused_engine_gives_the_unfused_logits(weights, goldens, monkeypatch):
    """The whole engine with and without the fusions: prompt and decode logits bit for bit."""

    from tensorfold.families.hy4_flash.cuda import model

    ids = goldens["long_ids"].tolist()[:40]
    out = []
    for fused in (False, True):
        monkeypatch.setattr(model, "FUSED", fused)
        e = _engine(weights)
        rows = [e.forward(ids[:32], prompt=True, every=True)]
        rows += [e.forward(ids[32:38], prompt=False, every=True), e.forward(ids[38:39], prompt=False)]
        out.append(torch.cat([r.float() for r in rows]))
    assert torch.equal(out[0], out[1])


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_dfly_graph_drafts_the_eager_tokens(weights, goldens, temperature, tmp_path, monkeypatch):
    """DFly2's block replayed as a CUDA graph (``TF_HY4_DFLY_GRAPH``) proposes the eager block's drafts, round by
    round, while the target's verify widths (and their key lists) change under it."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill

    prompt = goldens["long_ids"].tolist()[:20]
    s = Sampling(seed=5, temperature=temperature)
    ckpt = tiny.write(tmp_path / "t")
    rounds = []
    for graph in ("0", "1"):
        monkeypatch.setenv("TF_HY4_DFLY_GRAPH", graph)
        e = _engine(weights, graphs=True)
        drafter = DFly(ckpt, e)
        trace: list = []
        got = draft_decode(e, prompt, prefill(e, prompt, s), 48, s, stop_eos=False, drafter=drafter,
                           allow_copy=False, trace=trace)
        rounds.append((got.tokens, [t["tokens"] for t in trace]))
        assert (drafter.graph is not None) == (graph == "1")
    assert rounds[0] == rounds[1]


@pytest.mark.parametrize("rule", ["qwen", "glm"])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_copy_rules_give_serial_replies(weights, temperature, rule):
    """Copies by either rule (``copy_rule``; glm: a 7-row first window and the 8-token back gate) on a prompt that
    repeats itself: the reply is serial decoding's."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode

    prompt = (list(range(10, 40)) * 2)[:56]
    s = Sampling(seed=4, temperature=temperature)
    e = _engine(weights, graphs=True, max_len=192)
    serial = serial_decode(e, prefill(e, prompt, s), 48, s, stop_eos=False).tokens
    e = _engine(weights, graphs=True, max_len=192)
    got = draft_decode(e, prompt, prefill(e, prompt, s), 48, s, stop_eos=False, copy_rule=rule)
    assert got.tokens == serial


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_constrained_drafted_replies_equal_serial(weights, temperature, tmp_path):
    """response_format on the GPU: DFly2 rounds and copies under a JSON schema give the serial constrained reply, and
    every token is the grammar's (tests/test_hy4_response_format.py has the toy vocabulary)."""

    from tests.test_hy4_response_format import _spec

    xgr = pytest.importorskip("xgrammar")
    from tensorfold.engine import grammar
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill, serial_decode

    vocab = ["\x00", "</s>"] + [chr(t) for t in range(2, tiny.V)]
    g = grammar.Grammars(xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=tiny.V, stop_token_ids=[1]))
    prompt = [(t % 60) + 40 for t in range(24)]
    s = Sampling(seed=9, temperature=temperature)
    e = _engine(weights, graphs=True, max_len=128)
    c = g.constraint(g.compile(_spec("json_schema")))
    serial = serial_decode(e, prefill(e, prompt, s, constraint=c), 40, s, constraint=c).tokens
    e = _engine(weights, graphs=True, max_len=128)
    c = g.constraint(g.compile(_spec("json_schema")))
    drafter = DFly(tiny.write(tmp_path / "t"), e)
    got = draft_decode(e, prompt, prefill(e, prompt, s, constraint=c), 40, s, drafter=drafter, constraint=c)
    assert got.tokens == serial
    m = xgr.GrammarMatcher(g.compile(_spec("json_schema")))
    assert all(m.accept_token(t) for t in serial)


def test_side_stream_shared_expert_keeps_the_bits(weights, goldens, monkeypatch):
    """TF_HY4_SIDE: the shared expert on a side stream, eager and graphed, gives the same decode logits."""

    from tensorfold.families.hy4_flash.cuda import model

    ids = goldens["long_ids"].tolist()[:40]
    out = []
    for side in (False, True):
        monkeypatch.setattr(model, "SIDE", side)
        for graphs in (False, True):
            e = _engine(weights, graphs=graphs)
            assert (e.side is not None) == side
            rows = [e.forward(ids[:32], prompt=True)]
            rows += [e.forward(ids[32:38], prompt=False, every=True)]
            rows += [e.forward([t], prompt=False) for t in ids[38:40]]
            rows += [e.forward(ids[34:38], prompt=False, every=True)]
            out.append(torch.cat([r.float() for r in rows]))
    assert all(torch.equal(out[0], o) for o in out[1:])


def test_lazy_engram_staging_keeps_the_logits(weights, goldens, monkeypatch):
    """A prompt chunk that awaits each Engram table before its layer (TF_HY4_LAZY_ENGRAM) gives the logits of one that
    reads all four first, with and without the next chunk's prefetch."""

    from tensorfold.families.hy4_flash.cuda import model

    ids = goldens["long_ids"].tolist()[:48]
    out = []
    for lazy in (False, True):
        monkeypatch.setattr(model, "LAZY_ENGRAM", lazy)
        e = _engine(weights, prefill_rows=16)
        s = model.prefill(e, ids, None)
        out.append((s, e.forward([s], prompt=False).float()))
    assert out[0][0] == out[1][0] and torch.equal(out[0][1], out[1][1])


@pytest.mark.parametrize("rows", [1, 6, 300])
def test_hc_boundary_keeps_the_three_kernels_bits(rows):
    """``hc_boundary``: the write-back, the collapse and the norm in one launch, bit for bit, at Hy4's widths."""

    from tensorfold.families.hy4_flash.cuda import kernels as K

    g = torch.Generator("cuda").manual_seed(rows)
    s, d = 4, 3072
    x = (torch.randn(rows, s, d, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    y = (torch.randn(rows, d, device="cuda", generator=g)).to(torch.bfloat16)
    pp = torch.rand(rows, s, device="cuda", generator=g) * 2
    fn = torch.randn(2 * s, s * d, device="cuda", generator=g) * 0.01
    base, scale = torch.randn(2 * s, device="cuda", generator=g), torch.rand(2, device="cuda", generator=g)
    nw = (torch.rand(d, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
    for pending in (None, (pp, y)):
        a, b = x.clone(), x.clone()
        if pending is not None:
            K.hc_post(a, *pending)
        post, h = K.hc_pre(a, fn, base, scale, eps=1e-5, hc_eps=1e-6, magnitude=2.0)
        want = K.rmsnorm(h, nw, 1e-5)
        got_post, got = K.hc_boundary(b, pending, fn, base, scale, nw, eps=1e-5, hc_eps=1e-6, magnitude=2.0,
                                      rms_eps=1e-5)
        assert torch.equal(a, b) and torch.equal(post, got_post) and torch.equal(want, got)
