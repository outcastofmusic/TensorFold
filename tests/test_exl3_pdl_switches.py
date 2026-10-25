"""TF_EXL3_PDL on the CPU: the flag's values, where the engines read it, and that every launch carries it.
tests/cuda/test_exl3_pdl_kernels.py compares the bytes on the GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, linear, switches  # noqa: E402
from tests.exl3_fakes import FakeExt, _layer

ENV = switches.PDL_ENV


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


@pytest.mark.parametrize("value, on", [(None, False), ("", False), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_the_flag_is_off_unless_turned_on(monkeypatch, value, on):
    for e in switches.ENVS:
        monkeypatch.delenv(e, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert switches.read().pdl is on                               # off: 0.99-1.00x served


@pytest.mark.parametrize("value", ["2", "decode", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(monkeypatch, tmp_path, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        switches.read()
    with pytest.raises(ValueError, match=ENV):
        FlashNextEngine(tmp_path / "missing")                      # no checkpoint there: the flag fails first
    with pytest.raises(ValueError, match=ENV):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


def test_an_engine_installs_what_it_reads(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    for e in switches.ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(ENV, "1")
    switches.use(switches.Switches())
    with pytest.raises(Exception):
        FlashNextEngine(tmp_path / "missing")                      # fails at the checkpoint, after the switches
    assert switches.current() == switches.Switches(pdl=True)


def test_use_returns_the_previous_setting():
    off = switches.Switches(pdl=False)
    first = switches.use(off)
    assert switches.current() is off
    assert switches.use(first) is off


@pytest.mark.parametrize("pdl", [True, False])
@pytest.mark.parametrize("rows", [1, 8, 16, 17])
def test_routed_hands_each_launch_its_switches(monkeypatch, pdl, rows):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(switches.Switches(pdl=pdl))
    ex = _layer()
    s = experts.Scratch(ex, 32, 4, device="cpu")
    x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16)
    pick = torch.zeros((rows, 4), dtype=torch.int32)
    experts.routed(x, pick, None, ex, s, None, rows)
    names = [n for n, _ in fake.calls]
    assert names == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    assert all(args[-1] is pdl for _, args in fake.calls)
    for _, args in (c for c in fake.calls if c[0] == "grouped"):
        assert (args[17], args[19]) == (8, 1)                      # n tiles, tiles in flight: grouped_kernel's cfg
    wts = torch.zeros((rows, 4), dtype=torch.float32)
    fake.calls.clear()
    experts.routed(x, pick, wts, ex, s, torch.zeros((rows, ex.dims)), rows)
    assert fake.calls[-1][0] == "down_combine" and fake.calls[-1][1][-1] is pdl


@pytest.mark.parametrize("pdl", [True, False])
def test_the_linear_hands_its_launches_the_pdl_switch(monkeypatch, pdl):
    fake = FakeExt()
    monkeypatch.setattr(linear, "_ext", lambda: fake)
    switches.use(switches.Switches(pdl=pdl))
    k, n = 256, 128
    lin = linear.Exl3Linear(torch.zeros((n // 128, k // 16, 8, 24), dtype=torch.int32), torch.zeros(k).half(),
                            torch.zeros(n).half(), None, 3.0, "mul1", k, n)
    lin._counters = torch.zeros(8, dtype=torch.int32)
    lin(torch.zeros((2, k), dtype=torch.float16), xh=torch.zeros((2, k), dtype=torch.float16),
        z=torch.zeros(16 * 2 * n))
    assert [c[0] for c in fake.calls] == ["rot_in", "linear"]
    assert all(args[-1] is pdl for _, args in fake.calls)
