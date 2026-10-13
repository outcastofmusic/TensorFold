"""Flash Next's kept prompt states under the entry cap, without a GPU: a slot's chain gives up its middle entries
before another conversation loses its system block's state or its latest turn."""

from __future__ import annotations

from types import SimpleNamespace

from tensorfold.families.qwen4_exp.cuda import prefixes


def _owner(keep=8):
    return SimpleNamespace(kept=[], keep=keep, free=[], _busy=lambda: set())


def test_three_conversations_keep_their_system_block_and_latest_turn():
    owner = _owner()
    wanted = []
    for base in (100, 1000, 2000):                    # each conversation in its own slot, six turns each
        slot = object()
        system = list(range(base, base + 400))
        prefixes.remember(owner, system, slot, {}, None)
        prompt = list(system)
        for turn in range(6):
            prompt = prompt + [1, 2, turn]
            prefixes.remember(owner, list(prompt), slot, {}, None)
        wanted += [system, prompt]
    kept = [k[0] for k in owner.kept]
    assert all(ids in kept for ids in wanted) and len(kept) <= owner.keep
    assert owner.free == []                           # every slot still holds a chain


def test_the_oldest_entry_goes_when_no_chain_has_a_middle():
    owner = _owner(keep=2)
    slots = [object() for _ in range(3)]
    for i, slot in enumerate(slots):
        prefixes.remember(owner, [i, i, i], slot, {}, None)
    assert [k[1] for k in owner.kept] == slots[1:] and owner.free == [slots[0]]
