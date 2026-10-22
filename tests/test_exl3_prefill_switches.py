"""The prompt-window switches on the CPU: TF_EXL3_GROUP_SCAN, TF_EXL3_DOWN_FUSE, TF_EXL3_GATEUP_FUSE (EXL3 kernels,
Flash Next and GLM-5.3) and TF_FLASHNEXT_MOE_CHUNK (Flash Next). Their values and where the engines read them, which
launches ``routed`` makes with each, a Python model of ``group_scan_kernel``'s rounds against ``group_kernel``'s
outputs (with a planted control that must fail), the 2,048-row routed window and its admission, and (compiled for
sm_121 without a GPU, when nvcc is present) that the new kernels do not spill, the fused prompt instance keeps 2 blocks
an SM, and the fused epilogues compile the epilogue kernels' multiply, FMA, exponent and rounding instructions in the
same counts. tests/cuda/test_exl3_prefill_kernels.py and test_exl3_prefill_replies.py compare the bytes on the GPU."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, switches  # noqa: E402

from tests.test_exl3_decode_switches import NVCC, FakeExt, _layer, _ptxas  # noqa: E402

EXL3_ENVS = {switches.GROUP_SCAN_ENV: "group_scan", switches.DOWN_FUSE_ENV: "down_fuse",
             switches.GATEUP_FUSE_ENV: "gateup_fuse"}
ALL_EXL3_ENVS = tuple(switches.ENVS)


@pytest.fixture(autouse=True)
def restore():
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    before, before_mm = switches.current(), exl3_mm.current()
    yield
    switches.use(before)
    exl3_mm.use(before_mm)


# -- the flags ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env", list(EXL3_ENVS))
@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("yes", True),
                                       ("0", False), ("off", False), (" NO ", False)])
def test_each_exl3_flag_is_on_unless_turned_off(monkeypatch, env, value, on):
    for e in ALL_EXL3_ENVS:
        monkeypatch.delenv(e, raising=False)
    if value is not None:
        monkeypatch.setenv(env, value)
    got = switches.read()
    assert getattr(got, EXL3_ENVS[env]) is on
    assert all(getattr(got, f) is True for e, f in EXL3_ENVS.items() if e != env)     # on since 19 Oct's GPU A/B
    assert (got.expert_decode, got.pdl, got.expert_items) == (False, False, True)     # the others keep defaults


@pytest.mark.parametrize("env", list(EXL3_ENVS))
@pytest.mark.parametrize("value", ["2", "scan", "o", "1.0"])
def test_an_exl3_flag_typo_stops_both_engines_before_any_read(monkeypatch, tmp_path, env, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=env):
        FlashNextEngine(tmp_path / "missing")
    with pytest.raises(ValueError, match=env):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


@pytest.mark.parametrize("value, on", [(None, True), ("1", True), ("off", False)])
def test_the_chunk_window_flag(monkeypatch, value, on):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    monkeypatch.delenv(exl3_mm.MOE_CHUNK_ENV, raising=False)
    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    if value is not None:
        monkeypatch.setenv(exl3_mm.MOE_CHUNK_ENV, value)
    got = exl3_mm.read()
    assert got.moe_chunk is on
    assert (got.stack_direct, got.inner_k, got.moe_direct) == (True, False, True)
    exl3_mm.use(got)
    assert exl3_mm.moe_window() == (exl3_mm.PROMPT_ROWS if on else 1024)


@pytest.mark.parametrize("value", ["2", "chunk", "1.0"])
def test_a_chunk_window_typo_stops_the_engine(monkeypatch, tmp_path, value):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(exl3_mm.MOE_CHUNK_ENV, value)
    with pytest.raises(ValueError, match=exl3_mm.MOE_CHUNK_ENV):
        FlashNextEngine(tmp_path / "missing")


def test_the_engines_install_what_they_read(monkeypatch, tmp_path):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    for e in ALL_EXL3_ENVS:
        monkeypatch.delenv(e, raising=False)
    for e in EXL3_ENVS:
        monkeypatch.setenv(e, "1")
    monkeypatch.setenv(exl3_mm.MOE_CHUNK_ENV, "1")
    switches.use(switches.Switches(group_scan=False, down_fuse=False, gateup_fuse=False, group_scan_decode=False,
                                   linear_pair=False))
    exl3_mm.use(exl3_mm.Switches(moe_chunk=False))
    with pytest.raises(Exception):
        FlashNextEngine(tmp_path / "missing")                      # fails at the checkpoint, after the switches
    assert switches.current() == switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True,
                                                   group_scan_decode=True, linear_pair=True)
    assert exl3_mm.current().moe_chunk is True
    switches.use(switches.Switches(group_scan=False, down_fuse=False, gateup_fuse=False, group_scan_decode=False,
                                   linear_pair=False))
    with pytest.raises(Exception):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)
    assert switches.current() == switches.Switches(group_scan=True, down_fuse=True, gateup_fuse=True,
                                                   group_scan_decode=True, linear_pair=True)


def test_an_unset_environment_reads_each_dataclass_default(monkeypatch):
    """One place holds each served default: a launch before any engine's ``use`` (a tool, a test) takes the
    setting an engine reads, and ENVS names every flag field, so ``read`` misses none (exl3_mm's prompt rows, a count,
    come from TENSORFOLD_PREFILL_ROWS: tests/test_exl3_prompt_rows.py)."""

    from dataclasses import fields

    from tensorfold.families.qwen4_exp.cuda import exl3_mm

    monkeypatch.delenv("TENSORFOLD_PREFILL_ROWS", raising=False)
    for module in (switches, exl3_mm):
        assert sorted(module.ENVS.values()) == sorted(f.name for f in fields(module.Switches) if f.name != "prompt_rows")
        for e in module.ENVS:
            monkeypatch.delenv(e, raising=False)
        assert module.read() == module.Switches()
    assert switches.Switches() == switches.Switches(expert_decode=False, pdl=False, expert_items=True, group_scan=True,
                                                    down_fuse=True, gateup_fuse=True, group_scan_decode=True,
                                                    linear_pair=True)
    assert exl3_mm.Switches() == exl3_mm.Switches(stack_direct=True, inner_k=False, moe_direct=True, moe_chunk=True)


# -- what routed launches -------------------------------------------------------------------------------------------
def _route(monkeypatch, rows, sw, *, wts=False, slots=4, ex=None, y=None):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(sw)
    ex = ex or _layer()                                            # 256 -> 128, 3-bit (rate 6): fusable rates
    s = experts.Scratch(ex, 1024, slots, device="cpu")
    x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16)
    pick = torch.zeros((rows, slots), dtype=torch.int32)
    w = torch.zeros((rows, slots)) if wts else None
    out = torch.zeros((rows, ex.dims)) if wts else None
    got = experts.routed(x, pick, w, ex, s, out, rows, y=y)
    return fake, s, ex, got


ALL_ON = switches.Switches(expert_items=True, group_scan=True, down_fuse=True, gateup_fuse=True,
                           group_scan_decode=False, linear_pair=False, rot_fuse=False)


@pytest.mark.parametrize("rows", [65, 128, 1024])
def test_a_prompt_window_with_every_switch_launches_three_kernels_after_rot_in(monkeypatch, rows):
    fake, s, ex, got = _route(monkeypatch, rows, ALL_ON)
    names = [n for n, _ in fake.calls]
    assert names == ["group", "rot_in", "grouped", "grouped"]
    group_kw, gu_kw, dn_kw = fake.kwargs[0], fake.kwargs[2], fake.kwargs[3]
    assert group_kw["scan"] is True and group_kw["tile"] == 32
    assert gu_kw["items"] is group_kw["items"] and dn_kw["items"] is group_kw["items"]
    assert gu_kw["epi"] == experts.EPI_GATEUP
    assert gu_kw["s0"] is ex.svh_g and gu_kw["s1"] is ex.svh_u and gu_kw["s2"] is ex.suh_d
    assert gu_kw["out"] is s.xd and gu_kw["arrivals"] is s.arrivals
    assert gu_kw["act_mode"] == experts.ACT_F32 and gu_kw["limit"] == float("inf")
    assert dn_kw["epi"] == experts.EPI_DOWN and dn_kw["s0"] is ex.svh_d and dn_kw["out"] is s.y
    assert got.data_ptr() == s.y.data_ptr()


@pytest.mark.parametrize("scan, down, gateup", [(True, False, False), (False, True, False), (False, False, True)])
def test_each_switch_alone_changes_only_its_launch(monkeypatch, scan, down, gateup):
    sw = switches.Switches(expert_items=True, group_scan=scan, down_fuse=down, gateup_fuse=gateup,
                           group_scan_decode=False, linear_pair=False, rot_fuse=False)
    fake, _, _, _ = _route(monkeypatch, 1024, sw)
    names = [n for n, _ in fake.calls]
    want = ["group", "rot_in", "grouped"] + ([] if gateup else ["gateup_epilogue"]) + ["grouped"] + \
        ([] if down else ["down_epilogue"])
    assert names == want
    assert ("scan" in fake.kwargs[0]) is scan
    grouped = [kw for (n, _), kw in zip(fake.calls, fake.kwargs) if n == "grouped"]
    assert ("epi" in grouped[0]) is gateup and ("epi" in grouped[1]) is down


def test_the_down_epilogue_writes_a_given_bf16_buffer(monkeypatch):
    y = torch.zeros((1024 * 4, 256), dtype=torch.bfloat16)
    fake, _, _, got = _route(monkeypatch, 1024, ALL_ON, y=y)
    assert got is y and fake.kwargs[3]["out"] is y


def test_with_weights_the_down_combine_stays(monkeypatch):
    """``wts`` combines the slots in the down epilogue's own launch: down_fuse leaves that path alone."""

    fake, _, _, _ = _route(monkeypatch, 1024, ALL_ON, wts=True)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "grouped", "down_combine"]
    assert "epi" not in fake.kwargs[3]


@pytest.mark.parametrize("rows", [1, 8, 16, 17, 28, 64])
@pytest.mark.parametrize("decode", [True, False])
def test_decode_windows_keep_their_launches(monkeypatch, rows, decode):
    sw = switches.Switches(expert_decode=decode, expert_items=True, group_scan=True, down_fuse=True, gateup_fuse=True,
                           group_scan_decode=False, linear_pair=False)
    fake, _, _, _ = _route(monkeypatch, rows, sw)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    assert all(kw == {} for kw in fake.kwargs)


def test_without_the_item_list_nothing_fuses_and_the_scan_still_groups(monkeypatch):
    sw = switches.Switches(expert_items=False, group_scan=True, down_fuse=True, gateup_fuse=True,
                           group_scan_decode=False, linear_pair=False, expert_staged=False)
    fake, _, _, _ = _route(monkeypatch, 1024, sw)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    assert fake.kwargs[0] == {"scan": True}
    assert all(kw == {} for kw in fake.kwargs[1:])


def test_rates_past_five_bits_keep_the_epilogue_kernels(monkeypatch):
    ex = _layer()
    ex.k2_gu, ex.k2_d = (6, 12), (12, 12)                          # instances of rates 2..16 spilled: none fused
    fake, _, _, _ = _route(monkeypatch, 1024, ALL_ON, ex=ex)
    assert [n for n, _ in fake.calls] == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]


def test_more_experts_than_threads_keep_group_kernel(monkeypatch):
    fake, _, _, _ = _route(monkeypatch, 128, ALL_ON, ex=_layer(E=1025))
    assert "scan" not in fake.kwargs[0]


@pytest.mark.parametrize("launch, k2, ok", [
    (((8, 4, 4, 1, 2), False), (6, 6), True),                     # Flash Next 3-bit experts
    (((8, 4, 1, 1, 2), False), (6, 10), True),
    (((8, 4, 4, 1, 2), False), (118, 118), True),                 # GLM-5.3's 2.25 bits
    (((8, 4, 4, 1, 2), False), (8, 8), True),
    (((8, 4, 4, 1, 2), False), (6, 11), False),                   # a 2..16 instance
    (((8, 4, 4, 1, 2), False), (6, 118), False),
    (((8, 4, 4, 1, 1), False), (6, 6), False),                    # one row tile: not the prompt setting
    (((4, 4, 4, 4, 1), True), (6, 6), False),                     # decode_kernel
    (((4, 4, 2, 2, 2), False), (6, 6), False),
])
def test_which_launches_fuse(launch, k2, ok):
    assert experts.fusable(launch, k2) is ok


def test_scan_limits():
    assert experts.scan_fits(513, 2048 * 11) and experts.scan_fits(1024, 65535)
    assert not experts.scan_fits(1025, 64) and not experts.scan_fits(8, 65536)


@pytest.mark.parametrize("rows, slots, E, I", [(1024, 11, 513, 640), (2048, 11, 513, 640), (1024, 9, 257, 2048),
                                               (64, 4, 8, 128)])
def test_the_arrival_counters_cover_every_item_and_n_block(rows, slots, E, I):
    ex = _layer(D=256, I=I, E=E)
    s = experts.Scratch(ex, rows, slots, device="cpu")
    for R in (65, rows):
        assert s.arrivals.numel() >= s.items(R, 32).numel() * (I // 128)
    assert int(s.arrivals.abs().sum()) == 0


# -- group_scan_kernel's rounds, in Python, against group_kernel's outputs -----------------------------------------
def _group_kernel(pick: list[int], E: int, maxm: int, slots: int, tile: int):
    """group_kernel's outputs: used experts in id order, members (row * 32 + slot, pick order, -1 after), items."""

    ids, members, items = [], [], []
    for e in range(E):
        at = [i for i, v in enumerate(pick) if v == e]
        if not at:
            continue
        place = len(ids)
        ids.append(e)
        row = [(i // slots) * 32 + i % slots for i in at][:maxm]
        members.append(row + [-1] * (maxm - len(row)))
        items += [(place << 16) | t for t in range(-(-min(len(at), maxm) // tile))]
    return ids, members, items


def _group_scan(pick: list[int], E: int, maxm: int, slots: int, tile: int, *, blocks: int = 8, threads: int = 1024,
                keep_zero: bool = True):
    """group_scan_kernel's rounds: lanes of 32, warps in order, a counter per (warp, expert) that holds a round's
    counts, then its offsets, then zero again. ``keep_zero=False`` drops the guard that leaves a warp without the
    expert at zero (the control: stale offsets then pass for counts)."""

    n = len(pick)
    W = threads // 32
    rounds = [list(range(i0, min(i0 + threads, n))) for i0 in range(0, n, threads)]

    def warps(r):                                                  # (warp, [(lane, i)]) of round r
        for w in range(W):
            yield w, [(t - w * 32, r[t]) for t in range(w * 32, min(w * 32 + 32, len(r)))]

    def peers(lanes, key):                                         # match_any: lanes holding the same key
        groups: dict[int, list[int]] = {}
        for lane, i in lanes:
            groups.setdefault(key(pick[i]), []).append(lane)
        return groups

    cnt = [0] * E
    for r in rounds:                                               # the count pass, every block alike
        for _, lanes in warps(r):
            for k, g in peers(lanes, lambda v: v if 0 <= v < E else -1).items():
                if k >= 0:
                    cnt[k] += len(g)                               # the leader's atomicAdd of popc
    place, used = [-1] * E, 0
    item0, nitems = [0] * E, 0
    for e in range(E):                                             # the two block scans
        if cnt[e]:
            place[e], used = used, used + 1
            item0[e] = nitems
            nitems += -(-min(cnt[e], maxm) // tile)
    ids = [None] * used
    members = [[None] * maxm for _ in range(used)]
    items = [None] * nitems
    per = -(-E // blocks)
    for b in range(min(blocks, E)):
        e0, e1 = b * per, min(E, b * per + per)
        for e in range(e0, e1):
            if cnt[e]:
                ids[place[e]] = e
                for t in range(-(-min(cnt[e], maxm) // tile)):
                    items[item0[e] + t] = (place[e] << 16) | t
                for j in range(min(cnt[e], maxm), maxm):
                    members[place[e]][j] = -1
        at = [[0] * E for _ in range(W)]
        run = [0] * E
        for r in rounds:
            groups = {}
            for w, lanes in warps(r):
                groups[w] = peers(lanes, lambda v, e0=e0, e1=e1: v if e0 <= v < e1 else -1)
                for k, g in groups[w].items():
                    if k >= 0:
                        at[w][k] = len(g)                          # the leader writes its group's count
            for e in range(e0, e1):                                # counts -> offsets, warps in order
                acc = run[e]
                for w in range(W):
                    k = at[w][e]
                    if k or not keep_zero:
                        at[w][e] = acc
                    acc += k
                run[e] = acc
            for w, lanes in warps(r):
                rank = {lane: q for g in groups[w].values() for q, lane in enumerate(g)}   # lanes below in the group
                for lane, i in lanes:
                    k = pick[i]
                    if e0 <= k < e1:
                        j = at[w][k] + rank[lane]
                        if j < maxm:
                            members[place[k]][j] = (i // slots) * 32 + i % slots
                for k in groups[w]:
                    if k >= 0:
                        at[w][k] = 0                               # the leader, after its warp read it
    return ids, members, items


def _picks(rows: int, slots: int, E: int, seed: int, *, hot: bool = True) -> list[int]:
    """Distinct routed picks a row (skewed as real routing is), a shared expert every row, and picks of E or -1."""

    g = torch.Generator().manual_seed(seed)
    routed = min(slots - 1, E - 1)
    weights = torch.rand((E - 1,), generator=g) ** (6 if hot else 1)
    pick = torch.stack([torch.multinomial(weights, routed, generator=g) for _ in range(rows)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((rows, 1), E - 1, dtype=torch.int32)], 1)
    if slots - 1 > routed:
        pick = torch.cat([pick, torch.full((rows, slots - 1 - routed), E, dtype=torch.int32)], 1)
    pick[0, 0] = -1
    pick[rows // 2, 1] = E
    return pick.flatten().tolist()


@pytest.mark.parametrize("rows, slots, E, tile, maxm", [
    (1024, 11, 513, 32, 1024),                                     # Flash Next's prompt window
    (2048, 11, 513, 32, 2048),                                     # TF_FLASHNEXT_MOE_CHUNK
    (1024, 9, 257, 32, 1024),                                      # GLM-5.3's shape
    (65, 11, 513, 32, 65), (100, 4, 9, 16, 100), (70, 8, 12, 32, 70), (130, 3, 5, 16, 130),
    (300, 11, 513, 32, 20),                                        # more picks of an expert than members fit
])
@pytest.mark.parametrize("seed", range(2))
def test_the_scan_rounds_write_group_kernels_outputs(rows, slots, E, tile, maxm, seed):
    pick = _picks(rows, slots, E, 11 * seed + rows, hot=seed == 0)
    want = _group_kernel(pick, E, maxm, slots, tile)
    assert _group_scan(pick, E, maxm, slots, tile) == want


@pytest.mark.parametrize("threads, blocks", [(64, 3), (96, 1), (32, 8)])
def test_the_rounds_hold_at_any_round_width(threads, blocks):
    """Rounds of a few warps cross many round boundaries; one block owns every expert, or eight own a few each."""

    pick = _picks(200, 6, 17, 5)
    assert _group_scan(pick, 17, 200, 6, 16, blocks=blocks, threads=threads) == _group_kernel(pick, 17, 200, 6, 16)


def test_the_control_without_the_zero_guard_fails():
    """Planted control (playbook 6.2): a warp without the expert must keep a zero count, or the next round reads the
    last round's offset as its count and members land at the wrong ranks."""

    pick = _picks(300, 11, 33, 3)
    want = _group_kernel(pick, 33, 300, 11, 32)
    assert _group_scan(pick, 33, 300, 11, 32, threads=128) == want
    assert _group_scan(pick, 33, 300, 11, 32, threads=128, keep_zero=False) != want


# -- TF_FLASHNEXT_MOE_CHUNK: a chunk in one routed call --------------------------------------------------------------
@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("rows", [17, 1024, 1500, 2048, 3000])
def test_a_chunk_takes_one_routed_call_of_its_rows(monkeypatch, chunk, rows):
    from tensorfold.families.qwen4_exp.cuda import exl3_mm, forward

    slots, dims = 11, 8
    calls = []

    def routed(x, pick, wts, ex, s, out, n, *a, y=None, **kw):
        calls.append((n, x.data_ptr(), None if y is None else y.data_ptr()))
        y.fill_(0.5)
        return y

    monkeypatch.setattr(experts, "routed", routed)
    monkeypatch.setattr(forward.moe_mod, "router", lambda *a, **k: None)
    monkeypatch.setattr(forward.moe_mod, "select_rows", lambda *a, **k: None)
    exl3_mm.use(exl3_mm.Switches(moe_chunk=chunk, prompt_rows=exl3_mm.MOE_CHUNK_WINDOW))   # 2,048-row chunks
    buf = SimpleNamespace(logits=torch.zeros((rows, 1)), pick=torch.zeros((rows, slots), dtype=torch.int32),
                          wts=torch.zeros((rows, slots)), slots=slots,
                          y=torch.zeros((rows, slots, dims), dtype=torch.bfloat16))
    b = SimpleNamespace(prefill=True, moe=buf, mixed=torch.zeros((rows, dims)))
    w = SimpleNamespace(cfg=SimpleNamespace(top_k=10, experts=512), x3=SimpleNamespace(moe=None))
    forward._exl3_moe(SimpleNamespace(router=None, experts=None), w, b, rows)
    window = 2048 if chunk else 1024
    starts = range(0, rows, window)
    assert [n for n, _, _ in calls] == [min(window, rows - r0) for r0 in starts]
    assert [p for _, p, _ in calls] == [b.mixed[r0].data_ptr() for r0 in starts]
    assert [p for _, _, p in calls] == [buf.y[r0].data_ptr() for r0 in starts]
    assert torch.all(buf.y.float() == 0.5)


@pytest.mark.parametrize("chunk", [True, False])
def test_the_scratch_and_the_admission_take_the_window(monkeypatch, chunk):
    from tensorfold.cuda import geometry
    from tensorfold.families.qwen4_exp.cuda import exl3_mm, exl3_pack

    exl3_mm.use(exl3_mm.Switches(moe_chunk=chunk, prompt_rows=exl3_mm.MOE_CHUNK_WINDOW))   # 2,048-row chunks
    window = 2048 if chunk else 1024
    seen = []
    monkeypatch.setattr(geometry, "exl3_indexed_scratch", lambda text, w, rows: seen.append(w) or 0)
    base = SimpleNamespace(bytes_at=lambda slots: 0, reserve=0, minimum_slots=1)
    exl3_pack.admission(lambda text: base)({})
    assert seen == [window]
    sc = exl3_mm.Scratch(4, True)
    sc.allocate("cpu", experts=_layer(), rows=16, ple_words=0, ple_heads=0, ple_dim=0)
    assert sc.moe.rows == window


def test_the_window_doubles_the_routed_scratch_under_a_gib():
    """Flash Next's shape: 2,560 wide, 640 an expert, 10 routed plus the shared expert."""

    from tensorfold.cuda.geometry import exl3_indexed_scratch

    text = {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts_per_tok": 10, "num_attention_heads": 32,
            "head_dim": 128, "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 32,
            "linear_value_head_dim": 128}
    one, two = (exl3_indexed_scratch(text, w, 2048) for w in (1024, 2048))
    fixed = exl3_indexed_scratch(text, 0, 2048)
    assert (two - fixed) == pytest.approx(2 * (one - fixed), rel=0.01)
    assert two - fixed < 1 << 30


# -- compiled for sm_121 --------------------------------------------------------------------------------------------
HERE = Path(experts.__file__).parent
FLOAT_OPS = re.compile(r"^(?:@!?P\d )?(FFMA[.A-Z]*|FMUL|FMNMX|MUFU\.[A-Z0-9]+|F2F[.A-Z0-9_]*|F2FP[.A-Z0-9_]*|"
                       r"HADD2\.F32)\b")


def _sass_ops(obj: Path) -> dict[str, Counter]:
    out, name = {}, None
    text = subprocess.run([str(Path(NVCC).parent / "cuobjdump"), "-sass", str(obj)], capture_output=True, text=True,
                          check=True).stdout
    for line in text.splitlines():
        m = re.match(r"\s+Function : (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = Counter()
            continue
        m = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;?\s*/\*", line)
        if name and m and (o := FLOAT_OPS.match(m.group(1))):
            out[name][o.group(1)] += 1
    return out


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_fused_instances_do_not_spill_and_keep_two_blocks_an_sm():
    names = [f"grouped_kernel<2, 8, 4, 1, {lo}, {hi}, 2, {epi}>" for epi in (1, 2) for lo, hi in
             ((2, 10), (118, 118), (8, 8))]
    got = {k: v for k, v in _ptxas(names + ["grouped_kernel<2, 8, 4, 1, 2, 10, 2, 0>"]).items()
           if "grouped_kernel" in k}
    assert len(got) == len(names) + 1
    for name, (regs, smem, spills) in got.items():
        assert not spills, name
        assert smem <= 33 * 1024, name
        assert 65536 // (-(-regs // 8) * 8 * 128) >= 2, (name, regs)


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_fused_epilogues_compile_the_epilogue_kernels_rounding_and_fma_instructions(tmp_path):
    """FMA contraction is the compiler's choice: a fused copy could contract a multiply into an add the epilogue
    kernel rounds separately. Each fused instance, less its EPI 0 twin, holds the standalone kernel's multiplies,
    FMAs (expf's and the division's), exponents, reciprocals, min/max and conversions in the same counts (the down
    epilogue twice: two row tiles, and its fp32 and bf16 outputs side by side in one instance)."""

    from torch.utils import cpp_extension

    src = tmp_path / "inst.cu"
    src.write_text((HERE / "experts.cu").read_text() + "\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void grouped_kernel<2, 8, 4, 1, 2, 10, 2, {epi}>(const half*, const half*, "
        "const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, const int*, float*, int, "
        "int, int, int, int, int, uint32_t, const int*, const int*, Epi);\n" for epi in (0, 1, 2)) + "}\n")
    obj = tmp_path / "inst.o"
    cmd = [NVCC, "-O3", "-std=c++17", "-gencode=arch=compute_121,code=sm_121", "-c", str(src), "-o", str(obj),
           f"-I{HERE}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
    import sysconfig
    cmd.append(f"-I{sysconfig.get_paths()['include']}")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    ops = _sass_ops(obj)

    def one(*tags):
        found = [v for k, v in ops.items() if all(t in k for t in tags)]
        assert len(found) == 1, tags
        return found[0]

    gateup = one("gateup_epilogue_kernel")
    down = one("down_epilogue_kernelIf") + one("down_epilogue_kernelI13__nv_bfloat16")
    plain = one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi0E")
    fused_gu = one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi1E") - plain
    fused_dn = one("grouped_kernelILi2ELi8ELi4ELi1ELi2ELi10ELi2ELi2E") - plain
    assert gateup["FFMA"] > 0 and gateup["MUFU.EX2"] > 0
    assert fused_gu == gateup
    assert fused_dn == down + down


# -- the bench tool -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("on", [set(), {"scan"}, {"down"}, {"gateup"}, {"chunk"}, {"scan", "down", "gateup", "chunk"}])
def test_the_bench_sets_each_named_switch_and_keeps_the_rest(on):
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tools.exl3_prefill_bench import _set

    base_sw = switches.Switches(pdl=True, expert_items=False, group_scan=False, down_fuse=False, gateup_fuse=False,
                                group_scan_decode=False, linear_pair=False)
    base_mm = exl3_mm.Switches(stack_direct=False, inner_k=True, moe_chunk=False)
    _set(base_sw, base_mm, on)
    sw, mm = switches.current(), exl3_mm.current()
    assert (sw.group_scan, sw.down_fuse, sw.gateup_fuse, mm.moe_chunk) == \
        ("scan" in on, "down" in on, "gateup" in on, "chunk" in on)
    assert (sw.pdl, sw.expert_items, mm.stack_direct, mm.inner_k, mm.moe_direct) == (True, False, False, True, True)
