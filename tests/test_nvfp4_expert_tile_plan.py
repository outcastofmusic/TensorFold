"""The NVFP4 expert tile's CPU side: which calls take the tile, and the 64-pair plan it runs on (items, tails, the
scratch that holds them, the units a launch covers)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import nvfp4_moe  # noqa: E402


def plan_items(picks: torch.Tensor, experts: int, tile: int) -> tuple[list[tuple[int, int, int]], list[int]]:
    """The plan ``grouped.route`` builds: items (expert, first, count) of at most ``tile`` pairs in expert order, and
    members (pair ids grouped by expert, pair order within an expert). The GPU test checks the kernel against it."""

    flat = [int(e) for e in picks.reshape(-1).tolist()]
    by_expert: list[list[int]] = [[] for _ in range(experts)]
    for p, e in enumerate(flat):
        by_expert[e].append(p)
    items, members = [], []
    for e, pairs in enumerate(by_expert):
        first = len(members)
        members.extend(pairs)
        items.extend((e, first + j, min(tile, len(pairs) - j)) for j in range(0, len(pairs), tile))
    return items, members


def counted_picks(counts: list[int], experts: int) -> torch.Tensor:
    """Picks [rows, 2] where routed expert i takes exactly counts[i] rows (slot 1: the shared expert, id ``experts``)."""

    routed = torch.cat([torch.full((c,), e, dtype=torch.int32) for e, c in enumerate(counts)])
    routed = routed[torch.randperm(routed.numel(), generator=torch.Generator().manual_seed(5))]
    return torch.stack([routed, torch.full_like(routed, experts)], dim=1)


@pytest.mark.parametrize("rows, prompt, tiled", [
    (2048, True, True), (1024, True, True), (1023, True, False), (4096, True, True), (2048, False, False),
    (7, False, False),
])
def test_only_prompt_chunks_of_1024_rows_take_the_tile(rows, prompt, tiled):
    assert nvx.TILE_ROWS == 1024
    assert nvfp4_moe.tiled(rows, prompt) is tiled


@pytest.mark.parametrize("counts", [
    [1, 15, 16, 17, 33, 48, 63, 64, 65, 100, 127, 128, 129, 200],     # tails of every size against 16 and 64
    [1900, 3, 0, 61, 1, 64],                                          # one expert takes most rows
    [64] * 12,                                                        # no tails
])
def test_a_64_pair_plan_cuts_each_expert_into_whole_items_and_one_tail(counts):
    experts = len(counts) + 1
    picks = counted_picks(counts, experts - 1)
    items, members = plan_items(picks, experts, nvx.TILE_ITEM)
    routed = [it for it in items if it[0] < experts - 1]
    for e, c in enumerate(counts):
        sizes = [n for x, _, n in routed if x == e]
        assert sizes == [64] * (c // 64) + ([c % 64] if c % 64 else [])
    assert sorted(members) == list(range(picks.numel()))
    rows = picks.shape[0]
    assert len(items) <= grouped.max_items(rows * 2, experts, nvx.TILE_ITEM)
    assert grouped.max_items(rows * 2, experts, nvx.TILE_ITEM) <= grouped.max_items(rows * 2, experts, grouped.TILE)


@pytest.mark.parametrize("rows", [1024, 2048, 4096])
def test_the_prompt_plan_scratch_holds_a_64_pair_plan_of_any_routing(rows):
    slots, experts = 11, 513                                        # Flash Next: top 10 of 512, then the shared one
    plan = grouped.Plan(rows, slots, experts, "cpu", prefill=True)
    worst = grouped.max_items(rows * slots, experts, nvx.TILE_ITEM)
    assert worst <= plan.items.shape[0]
    g = torch.Generator().manual_seed(rows)
    routed = torch.stack([torch.randperm(experts - 1, generator=g)[:slots - 1] for _ in range(rows)])
    picks = torch.cat([routed, torch.full((rows, 1), experts - 1)], dim=1)
    assert len(plan_items(picks, experts, nvx.TILE_ITEM)[0]) <= worst


@pytest.mark.parametrize("nb, groups", [(20, 5), (80, 20), (10, 3), (1, 1)])
def test_a_tile_unit_covers_four_column_blocks(nb, groups):
    plan = grouped.Plan(2048, 11, 513, "cpu", prefill=True)
    plan.tile = nvx.TILE_ITEM
    items = grouped.max_items(2048 * 11, 513, nvx.TILE_ITEM)
    assert nvx.units(2048, plan, nb, tile=True) == items * groups
    assert nvx.units(2048, plan, nb) == items * nb
