"""TF_FLASHNEXT_SHARED_STREAM's CPU side: the switch, which passes take the side stream, and the order ``nvfp4_moe.moe``
queues its work in (fork before the router, the shared expert under the side stream, join after the routed down)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import nvfp4_moe, shared_stream  # noqa: E402


@pytest.fixture
def mode(monkeypatch):
    """Set the switch for one test; the cached read is cleared before and after."""

    def set_(value: str | None) -> None:
        if value is None:
            monkeypatch.delenv(shared_stream.ENV, raising=False)
        else:
            monkeypatch.setenv(shared_stream.ENV, value)
        shared_stream.mode.cache_clear()

    yield set_
    shared_stream.mode.cache_clear()


def test_the_switch_is_off_unless_set(mode):
    mode(None)
    assert shared_stream.mode() == shared_stream.OFF
    mode("")
    assert shared_stream.mode() == shared_stream.OFF
    mode("1")
    assert shared_stream.mode() == shared_stream.DECODE
    mode(" 2 ")
    assert shared_stream.mode() == shared_stream.ALL


def test_a_typo_raises_instead_of_meaning_off(mode):
    mode("yes")
    with pytest.raises(ValueError, match=shared_stream.ENV):
        shared_stream.mode()


def test_the_switch_is_read_once_a_process(mode):
    mode("1")
    assert shared_stream.mode() == 1
    import os

    os.environ[shared_stream.ENV] = "0"          # captured graphs keep the schedule they were captured with
    assert shared_stream.mode() == 1


@pytest.mark.parametrize("m, decode, prompt", [(0, False, False), (1, True, False), (2, True, True)])
def test_modes_pick_the_passes(m, decode, prompt):
    assert shared_stream.applies(False, m) is decode
    assert shared_stream.applies(True, m) is prompt


def test_a_cpu_tensor_never_makes_a_stream(mode):
    mode("2")
    assert shared_stream.side(torch.device("cpu"), False) is None
    assert shared_stream._SIDES == {}


# ---- the schedule of nvfp4_moe.moe, on fakes that log each step and the stream it was queued on ----------------------

TOP_K, ROWS, D, NI = 2, 3, 8, 4


class _Log:
    def __init__(self) -> None:
        self.steps: list[tuple[str, str]] = []
        self.stream = "main"

    def put(self, name: str) -> None:
        self.steps.append((name, self.stream))


@pytest.fixture
def fakes(monkeypatch):
    from tensorfold.cuda import moe as moe_mod

    log = _Log()

    class Side:
        def fork(self):
            log.put("fork")
            return "side"

        def join(self):
            log.put("join")

    @contextmanager
    def stream(s):
        before, log.stream = log.stream, s
        try:
            yield
        finally:
            log.stream = before

    def shared(lin, x, prompt, f32):
        log.put("shared_down")
        return torch.full((x.shape[0], D), 7.0)

    def shared_act(x, prompt=False):
        log.put("shared_act")
        return torch.full((x.shape[0], NI), 5.0, dtype=torch.bfloat16)

    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(shared_stream, "side", lambda device, prompt_pass:
                        Side() if shared_stream.applies(prompt_pass) else None)
    monkeypatch.setattr(moe_mod, "router", lambda *a: log.put("router"))
    monkeypatch.setattr(moe_mod, "select", lambda *a: log.put("select"))
    monkeypatch.setattr(nvfp4_moe.nvx, "gate_up", lambda *a, **k: log.put("gate_up"))
    monkeypatch.setattr(nvfp4_moe.nvx, "down", lambda *a, **k: log.put("down"))
    monkeypatch.setattr(nvfp4_moe, "_shared", shared)
    ex = SimpleNamespace(routed=3, width=NI, dims=D, routed_experts=None, shared=SimpleNamespace(down=None),
                         shared_act=shared_act)
    buf = SimpleNamespace(y=torch.zeros((ROWS, TOP_K + 1, D)), act=torch.zeros((ROWS, TOP_K + 1, NI),
                          dtype=torch.bfloat16), logits=torch.zeros((ROWS, 4)), plan=None)
    cfg = SimpleNamespace(num_experts_per_tok=TOP_K)
    return log, ex, buf, cfg


def _run(fakes, prompt_pass: bool = False):
    log, ex, buf, cfg = fakes
    nvfp4_moe.moe(torch.zeros((ROWS, D), dtype=torch.bfloat16), None, None, ex, buf, cfg, prompt_pass=prompt_pass)
    assert torch.equal(buf.act[:, TOP_K], torch.full((ROWS, NI), 5.0, dtype=torch.bfloat16))
    assert torch.equal(buf.y[:, TOP_K], torch.full((ROWS, D), 7.0))
    assert not buf.y[:, :TOP_K].any()                      # the routed slots are the (faked) routed kernels' alone
    return log.steps


ONE_STREAM = [("router", "main"), ("select", "main"), ("gate_up", "main"), ("shared_act", "main"), ("down", "main"),
              ("shared_down", "main")]
SIDE = [("fork", "main"), ("shared_act", "side"), ("shared_down", "side"), ("router", "main"), ("select", "main"),
        ("gate_up", "main"), ("down", "main"), ("join", "main")]


def test_off_keeps_the_one_stream_order(mode, fakes):
    mode("0")
    assert _run(fakes) == ONE_STREAM
    assert _run(fakes, prompt_pass=True)[-len(ONE_STREAM):] == ONE_STREAM


def test_decode_passes_fork_before_the_router_and_join_after_the_routed_down(mode, fakes):
    mode("1")
    assert _run(fakes) == SIDE


def test_mode_1_keeps_prompt_chunks_on_one_stream(mode, fakes):
    mode("1")
    assert _run(fakes, prompt_pass=True) == ONE_STREAM


def test_mode_2_forks_prompt_chunks_too(mode, fakes):
    mode("2")
    assert _run(fakes, prompt_pass=True) == SIDE


def test_the_forward_passes_its_buffers_prompt_flag(monkeypatch):
    """moe_block hands the NVFP4 MoE ``b.prefill``: decode windows' bf16 slots (moe_prefill) are not prompt chunks."""

    from tensorfold.families.qwen4_exp.cuda import forward

    seen = []
    monkeypatch.setattr(forward.nvfp4_moe, "moe", lambda *a, prompt_pass: seen.append(prompt_pass))
    layer = SimpleNamespace(moe=SimpleNamespace(experts=SimpleNamespace(kernel="nvfp4"), router=None))
    w = SimpleNamespace(cfg=SimpleNamespace(top_k=TOP_K, experts=3, moe_width=NI, hidden=D), comm=None)
    for prefill in (False, True):
        b = SimpleNamespace(moe=SimpleNamespace(y=torch.zeros((1, 1, 1)), wts=torch.zeros((1, 1))),
                            mixed=torch.zeros((1, 1)), xs_mixed=torch.zeros((1, 1)), prefill=prefill)
        forward.moe_block(layer, w, b, 1)
    assert seen == [False, True]
