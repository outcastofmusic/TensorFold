"""TF_EXL3_EXPERT_ITEMS on the CPU: the flag's values and where the engines read it, which launches of ``routed`` take
group_kernel's item list and at what member-group size, that the list's bound holds every window group_kernel can
list (a Python model of its listing), and (compiled for sm_121 without a GPU, when nvcc is present) that the listed
layout leaves grouped_kernel's registers and shared memory where they were. tests/cuda/test_exl3_sweep2_kernels.py
compares the bytes on the GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts, switches  # noqa: E402

from tests.test_exl3_decode_switches import NVCC, FakeExt, _layer, _ptxas  # noqa: E402

ENV = switches.EXPERT_ITEMS_ENV


@pytest.fixture(autouse=True)
def restore():
    before = switches.current()
    yield
    switches.use(before)


@pytest.mark.parametrize("value, on", [(None, True), ("", True), ("1", True), ("on", True), ("0", False),
                                       ("off", False), ("false", False), (" NO ", False)])
def test_the_flag_is_on_unless_turned_off(monkeypatch, value, on):
    for e in (switches.EXPERT_DECODE_ENV, switches.PDL_ENV, ENV):
        monkeypatch.delenv(e, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert switches.read() == switches.Switches(expert_items=on)


@pytest.mark.parametrize("value", ["2", "items", "o", "1.0"])
def test_a_flag_typo_stops_both_engines_before_any_read(monkeypatch, tmp_path, value):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        FlashNextEngine(tmp_path / "missing")
    with pytest.raises(ValueError, match=ENV):
        GlmEngine(tmp_path / "missing", rank=0, master="", port=0, world=1)


def test_an_engine_installs_the_item_switch(monkeypatch, tmp_path):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    monkeypatch.setenv(ENV, "0")
    switches.use(switches.Switches())
    with pytest.raises(Exception):
        FlashNextEngine(tmp_path / "missing")
    assert switches.current().expert_items is False


def _route(monkeypatch, rows, decode, items, slots=4, group=True):
    fake = FakeExt()
    monkeypatch.setattr(experts, "_ext", lambda: fake)
    switches.use(switches.Switches(expert_decode=decode, pdl=True, expert_items=items))
    ex = _layer()
    s = experts.Scratch(ex, 1024, slots, device="cpu")
    x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16)
    experts.routed(x, torch.zeros((rows, slots), dtype=torch.int32), None, ex, s, None, rows, group=group)
    return fake, s, ex


@pytest.mark.parametrize("rows", [1, 8, 16, 17, 28, 64, 65, 1024])
@pytest.mark.parametrize("decode", [True, False])
@pytest.mark.parametrize("items", [True, False])
def test_grouped_kernels_launches_take_the_list_the_grouping_wrote(monkeypatch, rows, decode, items):
    fake, s, ex = _route(monkeypatch, rows, decode, items)
    names = [n for n, _ in fake.calls]
    assert names == ["group", "rot_in", "grouped", "gateup_epilogue", "grouped", "down_epilogue"]
    on_decode = decode and rows <= experts.DECODE_ROWS
    tile = 16 * experts.PROMPT_TILES if rows > experts.PROMPT_ROWS else 16
    listed = items and not on_decode
    group_kw = fake.kwargs[0]
    if not listed:
        assert all(kw == {} for kw in fake.kwargs)                # the launches of before the switch
        return
    assert group_kw["tile"] == tile and group_kw["icount"] is s.icount
    want = experts.item_bound(rows * 4, min(rows * 4, ex.count), tile)
    assert group_kw["items"].numel() == want
    assert group_kw["items"].data_ptr() == s.items_buf.data_ptr()
    for i in (2, 4):                                              # both grouped launches read the same list
        assert fake.kwargs[i]["items"] is group_kw["items"] and fake.kwargs[i]["icount"] is s.icount
        assert fake.calls[i][1][-2] is False                      # grouped_kernel, not decode_kernel
        assert fake.calls[i][1][22] == tile // 16                 # row tiles a block: the list's member group


def test_without_grouping_no_list_is_read(monkeypatch):
    fake, _, _ = _route(monkeypatch, 64, False, True, group=False)
    assert [n for n, _ in fake.calls][0] == "rot_in"
    assert all(kw == {} for kw in fake.kwargs)


@pytest.mark.parametrize("gu, dn, tile", [
    (((8, 4, 4, 1, 2), False), ((8, 4, 1, 1, 2), False), 32),
    (((8, 4, 4, 1, 1), False), ((8, 4, 1, 1, 1), False), 16),
    (((4, 4, 4, 4, 1), True), ((4, 4, 1, 4, 1), True), 0),           # decode_kernel takes no list
    (((4, 4, 4, 4, 1), True), ((8, 4, 1, 1, 1), False), 16),         # down alone on grouped_kernel
    (((8, 4, 4, 1, 2), False), ((8, 4, 1, 1, 1), False), 0),         # groups of different sizes: no shared list
])
def test_the_listed_tile(gu, dn, tile):
    assert experts.listed_tile(gu, dn) == tile


def _listing(pick: torch.Tensor, E: int, maxm: int, tile: int) -> list[tuple[int, int]]:
    """group_kernel's items as (place, member group): experts below E in id order, ceil(min(c, maxm) / tile) each."""

    counts = torch.bincount(pick[(pick >= 0) & (pick < E)].flatten().long(), minlength=E)
    out, place = [], 0
    for e in range(E):
        c = int(counts[e])
        if c:
            out += [(place, t) for t in range(-(-min(c, maxm) // tile))]
            place += 1
    return out


@pytest.mark.parametrize("rows, slots, E, tile", [(1, 11, 513, 16), (8, 11, 513, 16), (28, 11, 513, 16),
                                                  (1024, 11, 513, 32), (1024, 9, 257, 32), (64, 8, 48, 16),
                                                  (128, 9, 48, 32), (17, 11, 12, 16)])
@pytest.mark.parametrize("seed", range(4))
def test_every_listing_fits_the_bound(rows, slots, E, tile, seed):
    """Routed picks (distinct a row) plus a shared expert every row, and picks of E or -1 that group_kernel skips."""

    g = torch.Generator().manual_seed(seed * 1000 + rows)
    routed = min(slots - 1, E - 1)
    skew = torch.rand((E - 1,), generator=g) ** 4                 # a few hot experts, as real routing has
    pick = torch.stack([torch.multinomial(skew, routed, generator=g) for _ in range(rows)]).to(torch.int32)
    pick = torch.cat([pick, torch.full((rows, 1), E - 1, dtype=torch.int32)], 1)
    if slots - 1 > routed:
        pick = torch.cat([pick, torch.full((rows, slots - 1 - routed), E, dtype=torch.int32)], 1)
    pick[0, 0] = -1
    items = _listing(pick, E, rows, tile)
    places = min(rows * slots, E)
    assert len(items) <= experts.item_bound(rows * slots, places, tile)
    assert experts.item_bound(rows * slots, places, tile) <= 65535
    # the shared expert, in every row, lists ceil(rows / tile) groups; each group's rows are its own
    assert sum(1 for p, _ in items if p == max(p for p, _ in items)) == -(-rows // tile)


def test_the_scratch_holds_the_largest_list_of_its_rows():
    ex = _layer(E=8)
    s = experts.Scratch(ex, 1024, 11, device="cpu")
    for rows in (1, 17, 1024):
        for tile in (16, 32):
            assert s.items(rows, tile).numel() <= s.items_buf.numel()


@pytest.mark.skipif(NVCC is None, reason="nvcc builds the sm_121 instances")
def test_the_listed_layout_keeps_grouped_kernels_footprint():
    """Flash Next's instances (mul1, rates 2..10) at the decode setting (1 row tile) and the prompt one (2): no spill,
    and the prompt instance's 2 blocks an SM (GB10: 64K registers) and 33 KB of shared memory, as before."""

    built = _ptxas(["grouped_kernel<2, 8, 4, 1, 2, 10, 1>", "grouped_kernel<2, 8, 4, 1, 2, 10, 2>"])
    got = {k: v for k, v in built.items() if "grouped_kernel" in k}
    assert len(got) == 2
    for name, (regs, smem, spills) in got.items():
        assert not spills, name
        assert smem <= 33 * 1024, name
        if "Li10ELi2E" in name:
            assert 65536 // (-(-regs // 8) * 8 * 128) == 2, (name, regs)
