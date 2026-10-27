"""PrefixCache eviction: a shared prefix survives its conversations, and states that served one run make way."""

from tensorfold.cuda.streams import PrefixCache


def _turn(cache: PrefixCache, prompt: list[int]) -> int:
    """One request: resume from the longest kept prefix, keep the prompt end; the resumed length."""

    hit = cache.longest(prompt)
    cache.add(prompt[:-1], None, None)
    return len(hit[0]) if hit else 0


def _run(cache: PrefixCache, base: int, sessions: int = 4, turns: int = 4) -> float:
    """``sessions`` interleaved conversations after one system block; the share of prompt tokens resumed."""

    system = list(range(1000, 1010))
    prompts = {s: system + [base + s] for s in range(sessions)}
    resumed = total = 0
    for _ in range(turns):
        for s in range(sessions):
            prompts[s] = prompts[s] + [base + 100 + s] * 3
            resumed += _turn(cache, prompts[s])
            total += len(prompts[s])
    return resumed / total


def test_a_second_run_of_new_sessions_resumes_as_much_as_the_first():
    cache = PrefixCache(6)
    first = _run(cache, 0)
    second = _run(cache, 500)
    assert second >= first - 0.02


def test_the_shared_system_block_outlives_its_conversations_turns():
    cache = PrefixCache(3)
    system = list(range(1000, 1010))
    cache.add(system, None, None)
    for s in range(6):
        assert _turn(cache, system + [s, s]) == len(system)
