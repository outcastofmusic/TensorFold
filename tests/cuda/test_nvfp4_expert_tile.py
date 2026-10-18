"""The shared-memory NVFP4 expert tile (64-pair items) against nvfp4_expert_kernel (16-pair items): byte-equal outputs
for gate/up, fp32 down and bf16 down, on Flash Next's shapes (512 experts, top 10, D 2560, NI 640 or 320 at two ranks),
under uniform, skewed and one-hot-expert routing, special values (NaN, -0, denormal and huge scales, every e4m3 byte),
chunks of 1,024 to 4,096 rows and item tails of every size. Mirrors MiaAI-Lab's experts-tile-check (recipe patch 0014,
MIT); recorded router picks (her --plans replay) are not covered here."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402
from tests.test_nvfp4_expert_tile_plan import counted_picks, plan_items  # noqa: E402

E, D, TOP = 512, 2560, 10
SLOTS = TOP + 1
DEV = "cuda"


def _blocks(shape: tuple, special: bool, g: torch.Generator) -> torch.Tensor:
    """Packed blocks [..., 144] int32: random code words; scale bytes 0x18-0x5F, an eighth negative (``special``: any
    byte, e4m3 NaN included)."""

    raw = torch.randint(0, 256, (*shape, 144 * 4), dtype=torch.uint8, device=DEV, generator=g)
    if not special:
        s = torch.randint(0x18, 0x60, (*shape, 64), dtype=torch.uint8, device=DEV, generator=g)
        neg = torch.randint(0, 8, (*shape, 64), dtype=torch.uint8, device=DEV, generator=g) == 0
        raw[..., 128 * 4:] = s | (neg.to(torch.uint8) << 7)
    return raw.view(torch.int32).contiguous()


def _scales(m: int, special: bool, g: torch.Generator) -> torch.Tensor:
    """[E, m] fp32 2^-10 .. 2^-4; ``special``: every 16th expert-matrix -0, a denormal, 2^100 or NaN."""

    v = (1.0 + torch.rand(E * m, generator=g, device=DEV)) * torch.exp2(
        torch.randint(-10, -3, (E * m,), generator=g, device=DEV).float())
    if special:
        i = torch.arange(E * m, device=DEV) % 16
        v[i == 0] = -0.0
        v[i == 1] = torch.tensor([0x00012345], dtype=torch.int32, device=DEV).view(torch.float32)
        v[i == 2] = 2.0 ** 100
        v[i == 3] = float("nan")
    return v.view(E, m).contiguous()


@pytest.fixture(scope="module", params=[(640, False), (640, True), (320, False)], ids=["ni640", "ni640-special",
                                                                                         "ni320"])
def experts(request):
    ni, special = request.param
    g = torch.Generator(device=DEV).manual_seed(0xE4E4 + ni + special)
    ex = nvx.Experts4(_blocks((E, ni // 32, D // 32, 2), special, g), _blocks((E, D // 32, ni // 32, 1), special, g),
                      _scales(2, special, g), _scales(1, special, g), ni, D, 7.0 if special else 0.0)
    yield ex, special
    del ex
    torch.cuda.empty_cache()


def _picks(rows: int, routing: str, seed: int) -> torch.Tensor:
    """[rows, 11] int32: ten distinct routed experts a row, then the shared one (id E). ``uniform``: Mia's synthetic
    routing (experts 0-31 a tenth hotter); ``skewed``: hers (a quarter of picks on experts 0-7); ``hot``: expert 7 in
    nine rows of ten, the one item that dwarfs the rest."""

    g = torch.Generator().manual_seed(seed)
    w = torch.ones(rows, E)
    if routing == "uniform":
        w[:, :32] = (0.1 / 32) / (0.9 / E)
    elif routing == "skewed":
        w[:, :8] = (0.25 / 8) / (0.75 / E)
    routed = torch.multinomial(w, TOP, replacement=False, generator=g)
    if routing == "hot":
        force = (torch.rand(rows, generator=g) < 0.9) & ~(routed == 7).any(dim=1)
        routed[force, 0] = 7
    return torch.cat([routed, torch.full((rows, 1), E)], dim=1).to(torch.int32).to(DEV).contiguous()


def _inputs(rows: int, ni: int, special: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, D, generator=g, device=DEV).to(torch.bfloat16)
    act = (torch.randn(rows * SLOTS, ni, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    if special:                                     # NaN, -0, a denormal and bf16's largest, scattered
        for t in (x, act):
            bits = t.view(torch.int16).view(-1)
            for k, v in enumerate((0x7FC0, -0x8000, 0x0001, 0x7F7F)):
                bits[k::997] = v
    return x, act


def _run(face: str, ex: nvx.Experts4, x: torch.Tensor, act: torch.Tensor, picks: torch.Tensor, tile: bool):
    """One face on a fresh plan of the kernel's item (64 for the tile, 16 otherwise), into a 0x11-filled buffer."""

    rows, slots = picks.shape
    plan = grouped.Plan(rows, slots, E + 1, DEV, prefill=True)
    grouped.route(picks, plan, nvx.TILE_ITEM if tile else nvx.PREFILL_TILE)
    n = ex.width if face == "gate_up" else ex.dims
    dtype = torch.float32 if face == "down_f32" else torch.bfloat16
    out = torch.empty((rows * slots, n), dtype=dtype, device=DEV)
    out.view(torch.uint8).fill_(0x11)
    if face == "gate_up":
        nvx.gate_up(x, ex, plan, out, rows, skip=E, tile=tile)
    else:
        nvx.down(act, ex, plan, out, rows, skip=E, tile=tile)
    return out, plan


def _same_bytes(face, a: torch.Tensor, b: torch.Tensor) -> None:
    ia = a.view(torch.int32 if a.dtype == torch.float32 else torch.int16)
    ib = b.view(torch.int32 if b.dtype == torch.float32 else torch.int16)
    if not torch.equal(ia, ib):
        differ = int((ia != ib).sum())
        pytest.fail(f"{face}: {differ} of {ia.numel()} values differ between the tile and nvfp4_expert_kernel")


@pytest.mark.parametrize("rows", [129, 974, 1024, 2048, 2049, 4096])
@pytest.mark.parametrize("routing", ["uniform", "skewed", "hot"])
def test_the_tile_writes_the_old_kernels_bytes(experts, routing, rows):
    ex, special = experts
    picks = _picks(rows, routing, rows * 7 + len(routing))
    x, act = _inputs(rows, ex.width, special, rows)
    for face in ("gate_up", "down_f32", "down_b16"):
        old, _ = _run(face, ex, x, act, picks, tile=False)
        new, plan = _run(face, ex, x, act, picks, tile=True)
        assert plan.tile == 64
        _same_bytes(face, new, old)


@pytest.mark.parametrize("counts", [
    [1, 2, 15, 16, 17, 31, 33, 47, 48, 49, 63, 64, 65, 79, 100, 127, 128, 129, 191, 200, 255, 257],
    [1500, 1, 63, 65],
])
def test_item_tails_of_every_size_keep_the_bits(experts, counts):
    """Experts with exactly ``counts`` pairs, one routed slot a row: items end on every row-tile split (4, 3, 2, 1)."""

    ex, special = experts
    picks = counted_picks(counts, E).to(DEV).contiguous()
    rows = picks.shape[0]
    x, act = _inputs(rows, ex.width, special, 99)
    act = act[:rows * 2].contiguous()
    for face in ("gate_up", "down_f32", "down_b16"):
        old, _ = _run(face, ex, x, act, picks, tile=False)
        new, _ = _run(face, ex, x, act, picks, tile=True)
        _same_bytes(face, new, old)


def test_the_planner_cuts_64_pair_items_with_their_tails():
    picks = _picks(2048, "hot", 3)
    plan = grouped.Plan(2048, SLOTS, E + 1, DEV, prefill=True)
    grouped.route(picks, plan, nvx.TILE_ITEM)
    items, members = plan_items(picks.cpu(), E + 1, nvx.TILE_ITEM)
    n = int(plan.counts[0])
    assert n == len(items) <= grouped.max_items(2048 * SLOTS, E + 1, nvx.TILE_ITEM)
    assert [tuple(r) for r in plan.items[:n].tolist()] == items
    assert plan.members[:picks.numel()].tolist() == members


def _layer(e: int, d: int, ni: int):
    def proj(n, k):
        return (torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=DEV),
                torch.randint(40, 60, (n, k // 16), dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn), 0.01)

    return nvfp4_moe.moe4_from_experts([proj(ni, d) for _ in range(e)], [proj(ni, d) for _ in range(e)],
                                       [proj(d, ni) for _ in range(e)],
                                       tuple((torch.randn(o, i) * 0.02).to(torch.bfloat16).to(DEV)
                                             for o, i in ((ni, d), (ni, d), (d, ni))))


@pytest.mark.parametrize("rows", [2048, 1023])
def test_a_prompt_chunk_through_moe_is_the_same_with_the_tile_off_or_on(monkeypatch, rows):
    """moe() on a prompt chunk: TF_FLASHNEXT_PROMPT_EXPERT_TILE=0, 1 and 2 give the same activations and slots; at
    2,048 rows the tile's 64-pair plan runs, at 1,023 the 16-pair one."""

    from types import SimpleNamespace

    from tensorfold.cuda import moe as moe_mod

    torch.manual_seed(17)
    e, d, ni, top_k = 64, 512, 256, 8
    base = _layer(e, d, ni)
    cfg = SimpleNamespace(num_experts_per_tok=top_k, num_experts=e, moe_intermediate_size=ni, hidden_size=d)
    router = (torch.randn(e + 1, d, device=DEV) * 0.1).to(torch.bfloat16)
    x = (torch.randn(rows, d, device=DEV) * 0.5).to(torch.bfloat16)
    tiles = []
    for name in ("gate_up", "down"):
        real = getattr(nvx, name)
        monkeypatch.setattr(nvx, name, lambda *a, _real=real, **kw: (tiles.append(kw["tile"]), _real(*a, **kw)))
    got = {}
    for mode in ("0", "1", "2"):
        monkeypatch.setenv(nvfp4_moe.TILE_ENV, mode)
        ex = nvfp4_moe.MoE4(base.routed_experts, base.shared)      # reads the flag as a loading layer does
        buf = moe_mod.MoEBuffers(rows, cfg, DEV, prefill=True)
        tiles.clear()
        nvfp4_moe.moe(x, None, router, ex, buf, cfg)
        got[mode] = (tuple(tiles), buf.plan.tile, buf.act[:rows].clone(), buf.y[:rows].clone())
    tiled = rows >= nvx.TILE_ROWS
    assert got["0"][:2] == ((False, False), 16)
    assert got["1"][:2] == ((tiled, False), 16)                     # down regrouped in 16-pair items
    assert got["2"][:2] == ((tiled, tiled), 64 if tiled else 16)
    for mode in ("1", "2"):
        assert torch.equal(got[mode][2].view(torch.int16), got["0"][2].view(torch.int16)), mode
        assert torch.equal(got[mode][3].view(torch.int16), got["0"][3].view(torch.int16)), mode


def test_decode_rows_never_take_the_tile(monkeypatch):
    from types import SimpleNamespace

    from tensorfold.cuda import moe as moe_mod

    monkeypatch.setenv(nvfp4_moe.TILE_ENV, "2")
    e, d, ni = 8, 256, 128
    ex = _layer(e, d, ni)
    cfg = SimpleNamespace(num_experts_per_tok=2, num_experts=e, moe_intermediate_size=ni, hidden_size=d)
    router = (torch.randn(e + 1, d, device=DEV) * 0.1).to(torch.bfloat16)
    tiles = []
    for name in ("gate_up", "down"):
        real = getattr(nvx, name)
        monkeypatch.setattr(nvx, name, lambda *a, _real=real, **kw: (tiles.append(kw["tile"]), _real(*a, **kw)))
    buf = moe_mod.MoEBuffers(1024, cfg, DEV)                        # decode buffers: fp32 slots
    nvfp4_moe.moe((torch.randn(1024, d, device=DEV) * 0.5).to(torch.bfloat16), None, router, ex, buf, cfg)
    assert tiles == [False, False] and buf.plan.tile == grouped.TILE
