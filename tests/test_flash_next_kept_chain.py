"""Flash Next's kept prompt states under the entry cap, without a GPU: a slot's chain gives up its middle entries
before another conversation loses its system block's state or its latest turn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


class _Slot:
    def copy_prefix(self, other, n, mtp_len):
        self.copied = (other, n)


def test_a_history_every_continuation_forks_from_is_never_the_entry_given_up():
    """An agent restarting each step from one history (upstream #735): a fork copies the history to a spare slot,
    where it is the slot's shortest entry, so the cap gives up other entries first."""
    a, spares = _Slot(), [_Slot() for _ in range(3)]
    owner = SimpleNamespace(kept=[], keep=4, free=list(spares), depth=3, _busy=lambda: set(),
                            _grow=lambda *a, **k: True, _shrink=lambda *a, **k: None)
    system, history = list(range(100)), list(range(100)) + [1, 2, 3]
    prefixes.remember(owner, system, a, {"mtp_len": 0}, None)
    prefixes.remember(owner, history, a, {"mtp_len": 0}, None)
    prefixes.remember(owner, history + [7, 0], a, {"mtp_len": 0}, None)         # the first step's state
    for step in (1, 2, 3):
        prompt = history + [7, step, 9]
        st, resume, cached = prefixes.slot_for(owner, prompt, True)
        assert cached == len(history) and st is not None
        prefixes.remember(owner, history, st, resume["state"], resume["tail"])     # as multi_plan keeps a resume
        prefixes.remember(owner, history + [7, step], st, {"mtp_len": 0}, None)
        assert history in [k[0] for k in owner.kept], step


def _agents(keep: int) -> list[int]:
    """Eight interleaved six-turn sessions, as the Flare recipe's agent benchmark sends them: each turn keeps its
    message start, its prompt end and its reply end, and the next turn replaces the reply. The rows each later turn
    resumes from beyond the shared system block."""

    owner = _owner(keep)
    system = list(range(400))
    prefixes.remember(owner, system, object(), {}, None)
    slots, prompts, reused = [object() for _ in range(8)], [list(system) for _ in range(8)], []
    for turn in range(6):
        for sid, slot in enumerate(slots):
            if turn:
                best = prefixes.longest(owner.kept, prompts[sid] + [8, 8])
                reused.append(len(best[0]) - len(system) if best else 0)
            start = prompts[sid] + [8, 8] if turn else prompts[sid]
            prompts[sid] = start + [1000 * sid + turn] * 50
            prefixes.remember(owner, list(start), slot, {}, None)
            prefixes.remember(owner, prompts[sid][:-1], slot, {}, None)
            prefixes.remember(owner, prompts[sid] + [5, 5], slot, {}, None)
    return reused


def test_eight_agent_sessions_resume_their_latest_prompt_ends_with_four_states_a_stream():
    from tensorfold.families.qwen4_exp.cuda.engine import KEEP, kept_states

    reused = _agents(kept_states(8))
    assert reused == [52 * (n // 8) + 49 for n in range(40)]   # each turn resumes from the one before
    assert _agents(KEEP).count(0) > 20               # the old cap: most turns resumed from the system block alone


def test_the_kept_state_count_comes_from_the_environment(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.engine import kept_states

    monkeypatch.delenv("TENSORFOLD_KEPT_STATES", raising=False)
    assert [kept_states(n) for n in (1, 2, 4, 8)] == [8, 8, 16, 32]
    monkeypatch.setenv("TENSORFOLD_KEPT_STATES", " 64 ")
    assert kept_states(8) == 64
    for value in ["0", "-1", "8.0", "lots"]:
        monkeypatch.setenv("TENSORFOLD_KEPT_STATES", value)
        with pytest.raises(ValueError, match="TENSORFOLD_KEPT_STATES"):
            kept_states(8)
