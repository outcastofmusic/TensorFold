"""TF_GLM_MOE_IN_PLACE, TF_GLM_SHARED_STREAM, TF_GLM_KDA_WIDE, TF_GLM_TOKEN_LIST and TF_GLM_EARLY_LAUNCH: the switches,
which buffers, kernels and launch order each path takes, and the token list's integers. CPU only: the MoE kernels and
the engine are stand-ins, the token-list kernel runs in Triton's interpreter, and tests/cuda/test_glm_sweep_switches.py
compares the bytes on the GPU."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.torch

from tensorfold.families.glm5_next.cuda import decode, engine, forward, kda, packed, prefill, sparse

SWITCHES = [("TF_GLM_MOE_IN_PLACE", forward.moe_in_place, True), ("TF_GLM_SHARED_STREAM", forward.shared_stream, False),
            ("TF_GLM_KDA_WIDE", kda.decode_wide, False), ("TF_GLM_TOKEN_LIST", sparse.token_list, True),
            ("TF_GLM_EARLY_LAUNCH", engine.early_launch, False)]


@pytest.mark.parametrize("env,read,default", SWITCHES)
def test_each_switch_has_its_default_and_a_typo_raises(monkeypatch, env, read, default):
    monkeypatch.delenv(env, raising=False)
    assert read() is default
    for value, want in (("0", False), ("off", False), ("1", True), ("on", True), ("", default)):
        monkeypatch.setenv(env, value)
        assert read() is want, value
    monkeypatch.setenv(env, "maybe")
    with pytest.raises(ValueError, match=env):
        read()


def test_the_engine_reads_every_switch_into_the_weights_meta(monkeypatch):
    for env, _, _ in SWITCHES:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("TF_GLM_PLAIN_NARROW", raising=False)
    monkeypatch.delenv("TF_GLM_HC_SPLIT", raising=False)
    assert engine.kernel_switches() == {"plain_narrow": True, "hc_split": True, "moe_in_place": True,
                                        "shared_stream": False, "kda_wide": False, "token_list": True,
                                        "early_launch": False}
    for env, _, default in SWITCHES:
        monkeypatch.setenv(env, "0" if default else "1")
    assert engine.kernel_switches() == {"plain_narrow": True, "hc_split": True, "moe_in_place": False,
                                        "shared_stream": True, "kda_wide": True, "token_list": False,
                                        "early_launch": True}


# -- buffers ------------------------------------------------------------------------------------------------------
D, I, E, TOP = 128, 128, 4, 2


def _weights(meta):
    c = NS(hidden=D, streams=4, heads=2, lin_heads=2, kv_lora=128, q_lora=128, qk_dim=128, v_dim=128, index_dim=128,
           index_heads=2, dense_width=256, top_k=TOP, moe_width=I, shared_width=I, experts=E, quant="exl3",
           mtp_layers=0, limit=10.0, routed_scale=1.0, norm_topk=True)
    experts = NS(dims=D, width=I, count=E)
    shared = NS(gu=object(), down=object())
    moe = NS(experts=experts, shared=shared, router=torch.zeros(E, D, dtype=torch.float16), bias=torch.zeros(E))
    layer = NS(kind="moe", index=3, moe=moe, mlp=None)
    head = packed.PlainLinear(torch.ones(256, D, dtype=torch.float16))
    return NS(cfg=c, device="cpu", world=1, meta={"packed_exl3": True, **meta}, layers=[layer], head=head, mtp=None,
              comm=None)


def _buffers(monkeypatch, meta, rows=4, prefill=False):
    monkeypatch.setattr(packed, "projections", lambda w: [w.head])
    monkeypatch.setattr(forward.latent, "ENABLED", False)
    return forward.Buffers(_weights(meta), rows, capacity=16, prefill=prefill)


@pytest.mark.parametrize("meta,on", [({}, True), ({"moe_in_place": True}, True), ({"moe_in_place": False}, False)])
@pytest.mark.parametrize("prefill_rows", [False, True])
def test_the_buffers_take_the_in_place_switch_and_start_ey_at_zero(monkeypatch, meta, on, prefill_rows):
    b = _buffers(monkeypatch, meta, prefill=prefill_rows)
    assert b.moe_in_place is on
    assert b.ey.shape == (4, TOP + 1, D) and b.ey.dtype == torch.float32 and not b.ey.any()


def _fake_routed(calls):
    """``experts.routed``'s writes: every routed slot's row of the scratch's y (its value: the row index + 1); the shared
    slot (pick == E) is left alone."""

    def routed(x, pick, wts, ex, s, out, R, limit, act_mode):
        y = s.y[:R * s.slots].view(R, s.slots, ex.dims)
        calls.append(s.y)
        for r in range(R):
            for k in range(s.slots):
                if pick[r, k] < ex.count:
                    y[r, k] = float(x[r, 0])
        return s.y[:R * s.slots]
    return routed


@pytest.mark.parametrize("rows,window", [(3, 4), (4, 4), (10, 4), (9, 3)])
def test_in_place_windows_write_out_and_give_the_copies_values(monkeypatch, rows, window):
    from tensorfold.cuda.exl3 import experts as gx

    ex = NS(dims=D, width=I, count=E)
    pick = torch.tensor([[r % E, (r + 1) % E, E] for r in range(rows)], dtype=torch.int32)
    x = torch.arange(1, rows + 1, dtype=torch.float32)[:, None].expand(rows, D)
    got = []
    for in_place in (False, True):
        calls = []
        monkeypatch.setattr(gx, "routed", _fake_routed(calls))
        s = gx.Scratch(ex, window, TOP + 1, device="cpu")
        own = s.y
        out = torch.zeros((rows, TOP + 1, D))
        out[:, TOP] = 7.0                                    # the caller's shared slot
        prefill.routed_micro_windows(x, pick, ex, s, out, rows, 10.0, in_place=in_place)
        assert s.y is own                                    # the scratch's own y is back
        assert len(calls) == -(-rows // window)
        if in_place:
            assert all(t.data_ptr() == out[i * window].data_ptr() for i, t in enumerate(calls))
            assert torch.equal(out[:, TOP], torch.full((rows, D), 7.0))
        got.append(out[:, :TOP].clone())
    assert torch.equal(got[0], got[1])


def test_in_place_windows_refuse_rows_they_cannot_write(monkeypatch):
    from tensorfold.cuda.exl3 import experts as gx

    s = gx.Scratch(NS(dims=D, width=I, count=E), 4, TOP + 1, device="cpu")
    out = torch.zeros((4, TOP + 1, D), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous fp32"):
        prefill.routed_micro_windows(torch.zeros(4, D), torch.zeros(4, 3, dtype=torch.int32), NS(dims=D), s, out, 4,
                                     10.0, in_place=True)


def test_the_kda_read_out_rows_are_the_buffers_own_only_for_decode_with_the_switch_on(monkeypatch):
    assert _buffers(monkeypatch, {}).kda_tmp is None
    b = _buffers(monkeypatch, {"kda_wide": True})
    q, y = b.kda_tmp
    assert q.shape == (4, 2, kda.DK) and q.dtype == torch.float32 and y.shape == (4, 2, kda.DV)
    assert y.dtype == torch.bfloat16
    assert _buffers(monkeypatch, {"kda_wide": True}, prefill=True).kda_tmp is None


def test_the_token_list_switch_reaches_the_buffers(monkeypatch):
    assert _buffers(monkeypatch, {}).token_list is True
    assert _buffers(monkeypatch, {"token_list": False}).token_list is False


# -- the MoE block ------------------------------------------------------------------------------------------------
class Log:
    """The MoE block's kernels as stand-ins: each records (stream, name, written tensor)."""

    def __init__(self, monkeypatch, b, stream=None):
        self.events, self.stream = [], "main"
        log = self.events

        def rec(name, t):
            log.append((self.stream, name, t))

        def routed(x, pick, ex, scratch, out, rows, limit, in_place=False):
            rec("routed", out)
            y = out if in_place else scratch.y[:rows * scratch.slots].view(rows, scratch.slots, D)
            y[:, :TOP] = 1.0                            # the routed kernels skip the shared slot
            if not in_place:
                out.copy_(y)
            return out

        def mm(b_, x, q, xs, out, f32=False):
            rec("mm", out)
            out.fill_(2.0 if q is shared.down else 0.5)
            return out

        shared = None

        def bind(m):
            nonlocal shared
            shared = m.shared

        self.bind = bind
        monkeypatch.setattr(forward, "routed_micro_windows", routed)
        monkeypatch.setattr(forward, "mm", mm)
        monkeypatch.setattr(forward.glue, "select", lambda *a: rec("select", None))
        monkeypatch.setattr(forward.glue, "swiglu", lambda *a: rec("swiglu", a[1]))
        monkeypatch.setattr(forward.glue, "combine", lambda y, wts, out: rec("combine", y.clone()))
        monkeypatch.setattr(forward, "gather", lambda w, b_, R: rec("gather", None))
        monkeypatch.setattr(forward.route_log, "note", lambda *a: None)


@pytest.mark.parametrize("in_place", [True, False])
def test_the_shared_expert_lands_in_its_slot_before_the_combine(monkeypatch, in_place):
    w = _weights({"moe_in_place": in_place})
    b = _buffers(monkeypatch, {"moe_in_place": in_place})
    b.ey.fill_(-1.0)
    log = Log(monkeypatch, b)
    m = w.layers[0].moe
    log.bind(m)
    forward.moe_block(w.layers[0], w, b, 3)
    names = [e[1] for e in log.events]
    assert names == ["mm", "select", "routed", "mm", "swiglu", "mm", "combine", "gather"]
    down = log.events[5][2]
    if in_place:            # the shared expert writes the slot itself: no copy through sy
        assert down.data_ptr() == b.ey[0, TOP].data_ptr() and down.stride(0) == (TOP + 1) * D
    else:
        assert down.data_ptr() == b.sy.data_ptr()
    combined = log.events[6][2]
    assert torch.equal(combined[:, :TOP], torch.ones(3, TOP, D)) and torch.equal(combined[:, TOP], torch.full((3, D), 2.0))


class _Stream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.events.append((self.name, f"wait {other.name}", None))


@pytest.mark.parametrize("in_place", [True, False])
def test_the_shared_stream_forks_after_the_router_and_joins_before_the_combine(monkeypatch, in_place):
    w = _weights({"moe_in_place": in_place})
    b = _buffers(monkeypatch, {"moe_in_place": in_place})
    log = Log(monkeypatch, b)
    log.bind(w.layers[0].moe)
    main, side = _Stream(log, "main"), _Stream(log, "side")
    b.side = side

    @contextmanager
    def on(stream):
        before, log.stream = log.stream, stream.name
        yield
        log.stream = before

    monkeypatch.setattr(forward.torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(forward.torch.cuda, "stream", on)
    forward.moe_block(w.layers[0], w, b, 3)
    got = [(s, n) for s, n, _ in log.events]
    assert got == [("main", "mm"), ("main", "select"), ("side", "wait main"), ("side", "mm"), ("side", "swiglu"),
                   ("side", "mm"), ("main", "routed"), ("main", "wait side"), ("main", "combine"), ("main", "gather")]
    combined = log.events[8][2]
    assert torch.equal(combined[:, :TOP], torch.ones(3, TOP, D)) and torch.equal(combined[:, TOP], torch.full((3, D), 2.0))


# -- KDA -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("wide", [False, True])
def test_kda_block_hands_the_chain_the_switch_and_the_buffers_rows(monkeypatch, wide):
    b = _buffers(monkeypatch, {"kda_wide": wide})
    seen = {}

    def chain(*args, wide=None, tmp=None):
        seen.update(wide=wide, tmp=tmp)
        return torch.zeros(1, 256, dtype=torch.bfloat16)

    monkeypatch.setattr(forward.kda_mod, "chain", chain)
    monkeypatch.setattr(forward, "mm", lambda *a, **k: None)
    monkeypatch.setattr(forward, "out_proj", lambda *a: None)
    monkeypatch.setattr(forward, "group_sums", lambda *a: None)
    k = NS(proj=None, fb=None, gb=None, o=None, fa_off=0, ga_off=128, b_off=0, conv=None, a_log=None, dt_bias=None,
           norm=None)
    layer = NS(kda=k, index=0)
    w = NS(cfg=NS(eps=1e-5, lower=-5.0))
    st = NS(kda_index={0: 0}, proj=torch.zeros(1, 4, 512, dtype=torch.bfloat16), cur=[0],
            conv=torch.zeros(1, 3, 8), rec=torch.zeros(2, 1, 1), scratch=[None])
    forward.kda_block(layer, w, st, b, 1)
    assert seen["wide"] is (True if wide else None)
    assert seen["tmp"] is b.kda_tmp and (seen["tmp"] is not None) is wide


def test_the_wide_chain_takes_the_callers_rows_over_the_shared_ones(monkeypatch):
    got = {}

    class Ext:
        def chain_wide(self, *args):
            got["q"], got["y"] = args[-2], args[-1]

    monkeypatch.setattr(kda, "_ext", lambda: Ext())
    q, y = torch.empty(4, 2, kda.DK), torch.empty(4, 2, kda.DV, dtype=torch.bfloat16)
    sc = kda.KDAScratch(4, 2, "cpu")
    a = torch.zeros(1, 256)
    kda.chain(a, 0, a, a, a, a, a, torch.zeros(2), a, a, 1e-5, -5.0, 1, sc, a, wide=True, tmp=(q, y))
    assert got["q"] is q and got["y"] is y
    kda.chain(a, 0, a, a, a, a, a, torch.zeros(2), a, a, 1e-5, -5.0, 1, sc, a, wide=True)
    assert got["q"] is not q and got["q"] is kda._tmp[(2, a.device)][0]


# -- the token list -------------------------------------------------------------------------------------------------
TOKEN_CASES = [(0, 1), (2046, 3), (2047, 1), (2048, 6), (2049, 2), (2050, 4), (5000, 6), (70001, 8), (98299, 5),
               (4000, 2048)]
INTERPRETED = """
import sys, torch
from tensorfold.families.glm5_next.cuda import sparse
for pos, R in %r:
    g = torch.Generator().manual_seed(pos + R)
    pools = torch.sort(torch.randint(0, max(1, (pos + R) // 4 + 1), (R, sparse.TOPK_POOLS), generator=g), 1).values
    pos_dev = torch.tensor([pos], dtype=torch.int32)
    want_t, want_c = sparse.token_rows_ops(pools, pos_dev)
    got_t, got_c = sparse.token_rows(pools, pos_dev)
    assert got_t.dtype == want_t.dtype == torch.int32 and got_c.dtype == want_c.dtype == torch.int32, (pos, R)
    assert torch.equal(got_t, want_t) and torch.equal(got_c, want_c), (pos, R)
    print(pos, R, int(got_c.max()))
"""


def test_the_token_list_kernel_writes_the_torch_ops_integers():
    """The kernel in Triton's interpreter (TRITON_INTERPRET=1 must be set before triton.jit runs, so in a child)."""

    import os
    import subprocess
    import sys

    env = {**os.environ, "TRITON_INTERPRET": "1"}
    run = subprocess.run([sys.executable, "-c", INTERPRETED % (TOKEN_CASES,)], env=env, capture_output=True, text=True,
                         check=False, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    counts = [int(line.split()[2]) for line in run.stdout.splitlines()]
    assert len(counts) == len(TOKEN_CASES) and max(counts) == sparse.TOPK_POOLS * sparse.POOL + 3


@pytest.mark.parametrize("fused", [False, True])
def test_select_tokens_takes_the_kernel_with_the_switch_on(monkeypatch, fused):
    calls = []
    monkeypatch.setattr(sparse, "_scores", type("S", (), {"__getitem__": lambda self, g: lambda *a, **k: None})())
    monkeypatch.setattr(sparse, "top_pools", lambda s, k, at: torch.zeros((s.shape[0], k), dtype=torch.int64))
    monkeypatch.setattr(sparse, "token_rows", lambda p, pos: calls.append("kernel") or (p, p))
    monkeypatch.setattr(sparse, "token_rows_ops", lambda p, pos: calls.append("ops") or (p, p))
    qi, wts = torch.zeros(2, 64), torch.zeros(2, 2)
    sparse.select_tokens(qi, wts, torch.zeros(1), 4096, 2, 4096, torch.zeros(1, dtype=torch.int32), fused=fused)
    assert calls == (["kernel"] if fused else ["ops"])


# -- early launches --------------------------------------------------------------------------------------------------
class FakeEngine:
    """A decode engine whose forward and sampler record the order of GPU launches and host callbacks."""

    def __init__(self, early, log, eos=()):
        self.w = NS(meta={"early_launch": early}, cfg=NS(eos=set(eos)), comm=None)
        self.st = NS(pos=10)
        self.buf = None
        self.log = log
        self.next_token = 100

    def verify_window(self, tokens):
        return tokens

    def forward(self, tokens):
        self.log.append(("forward", tuple(tokens)))
        return torch.zeros(len(tokens), 4)

    def sample(self, logits, positions, sampling):
        self.next_token += 1
        return [self.next_token + r for r in range(logits.shape[0])]

    def follow(self, tokens):
        pass

    def tap_rows(self, n):
        return n


def _quiet(monkeypatch, log):
    monkeypatch.setattr(decode, "_sync", lambda w: None)
    monkeypatch.setattr(decode.torch.cuda, "synchronize", lambda: None)

    def commit(w, st, b, R, keep):
        st.pos += keep
        log.append(("commit", keep))
    monkeypatch.setattr(decode, "commit", commit)


@pytest.mark.parametrize("early", [False, True])
def test_serial_decode_launches_the_next_step_before_the_callback_with_the_same_tokens(monkeypatch, early):
    log = []
    _quiet(monkeypatch, log)
    e = FakeEngine(early, log)
    res = decode.serial_decode(e, 7, 4, None, on_tokens=lambda new: log.append(("fed", tuple(new))) and False)
    assert res.tokens == [7, 101, 102, 103]
    if early:
        assert log == [("forward", (7,)), ("commit", 1), ("forward", (101,)), ("fed", (101,)), ("commit", 1),
                       ("forward", (102,)), ("fed", (102,)), ("commit", 1), ("fed", (103,))]
    else:
        assert log == [("forward", (7,)), ("commit", 1), ("fed", (101,)), ("forward", (101,)), ("commit", 1),
                       ("fed", (102,)), ("forward", (102,)), ("commit", 1), ("fed", (103,))]


def test_a_stop_after_an_early_launch_leaves_its_step_uncommitted(monkeypatch):
    log = []
    _quiet(monkeypatch, log)
    e = FakeEngine(True, log)
    res = decode.serial_decode(e, 7, 10, None, on_tokens=lambda new: True)
    assert res.tokens == [7, 101]
    assert log == [("forward", (7,)), ("commit", 1), ("forward", (101,))] and e.st.pos == 11


def test_no_early_launch_after_the_last_token_or_an_end_token(monkeypatch):
    log = []
    _quiet(monkeypatch, log)
    e = FakeEngine(True, log, eos=(102,))
    decode.serial_decode(e, 7, 10, None, stop_eos=True)
    assert [x for x in log if x[0] == "forward"] == [("forward", (7,)), ("forward", (101,))]


class FakeDrafter:
    def __init__(self, log):
        self.log, self.context_end, self.block, self.ahead = log, 5, 8, None

    def _launch(self, pending):
        self.log.append(("block", pending))

    launch_ahead = None                     # bound below from the real class

    def propose(self, pending, depth, sampling, confidence):
        # Drafter.candidates' gate on the queued pass
        if self.ahead != (pending, self.context_end):
            self._launch(pending)
        self.ahead = None
        self.log.append(("read", pending))
        return [pending + 1, pending + 2][:depth]

    def add_taps(self, n):
        self.context_end += n
        self.log.append(("taps", n))


@pytest.mark.parametrize("early", [False, True])
def test_dflash_rounds_queue_the_block_pass_before_the_callback_and_read_it_once(monkeypatch, early):
    from tensorfold.families.glm5_next.cuda import dflash2

    log = []
    _quiet(monkeypatch, log)
    e = FakeEngine(early, log)
    d = FakeDrafter(log)
    d.launch_ahead = dflash2.Drafter.launch_ahead.__get__(d)
    fed = lambda new: log.append(("fed", tuple(new))) and False
    res = decode.dflash_decode(e, d, 7, 5, None, policy=decode.DepthPolicy(2, fixed=True), on_tokens=fed)
    blocks = [x for x in log if x[0] == "block"]
    reads = [x for x in log if x[0] == "read"]
    assert len(blocks) == len(reads) == res.rounds and d.ahead is None
    if early:                              # each later round's pass was queued before the previous round's callback
        for i, x in enumerate(log):
            if x[0] == "fed" and i + 1 < len(log):
                assert log[i - 1][0] == "block", log
    else:
        assert all(log[i + 1][0] == "read" for i, x in enumerate(log) if x[0] == "block")
