"""Flash Next keeps prompt states at marked rows inside a chunk's DeltaNet chain instead of ending the chunk there."""

import random
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.torch


def _old_plan(start, n, rows, points):
    """The released serial planner with every point as a stop: each one ended its chunk."""

    plan, stops = [], sorted(p for p in points if start < p < n)
    while start < n:
        end = min(start + rows, next((p for p in stops if p > start), n))
        plan.append((start, end))
        start = end
    return plan


def test_up_to_four_points_inside_a_chunk_add_no_chunk():
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan

    assert chunk_plan(0, 10_000, 4_096, [], 4) == [(0, 4_096, []), (4_096, 8_192, []), (8_192, 10_000, [])]
    # a system block's end, the last assistant start and the prompt end one token early
    points = [1_500, 9_000, 9_999]
    assert chunk_plan(0, 10_000, 4_096, points, 4) == [(0, 4_096, [1_500]), (4_096, 8_192, []),
                                                       (8_192, 10_000, [9_000, 9_999])]
    assert chunk_plan(0, 10_000, 4_096, points, 1) == [(0, 4_096, [1_500]), (4_096, 8_192, []),
                                                       (8_192, 9_000, [9_000]), (9_000, 10_000, [9_999])]
    assert len(_old_plan(0, 10_000, 4_096, points[:2])) == 4
    assert chunk_plan(100, 300, 4_096, [100, 200, 300], 4) == [(100, 300, [200, 300])]   # the start is resumed


def test_the_point_after_a_chunks_marks_ends_it():
    from tensorfold.cuda.kernels.gdn import MARKS
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan

    assert MARKS == 4
    points = [10, 20, 30, 40, 50, 60]
    assert chunk_plan(0, 100, 64, points, 4) == [(0, 40, [10, 20, 30, 40]), (40, 100, [50, 60])]
    assert chunk_plan(0, 100, 64, points, 2) == [(0, 20, [10, 20]), (20, 40, [30, 40]), (40, 100, [50, 60])]
    for marks in (0, 5):
        with pytest.raises(ValueError, match="states"):
            chunk_plan(0, 100, 64, points, marks)


@pytest.mark.parametrize("seed", range(30))
def test_every_point_is_kept_once_within_the_marks_and_never_in_more_chunks(seed):
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan

    rng = random.Random(seed)
    n, rows, marks = rng.randint(1, 3_000), rng.choice([7, 16, 64, 512]), rng.randint(1, 4)
    start = rng.randint(0, n - 1)
    points = rng.sample(range(1, n + 1), min(n, rng.randint(0, 40)))
    plan = chunk_plan(start, n, rows, points, marks)
    assert plan[0][0] == start and plan[-1][1] == n
    assert all(a < b <= a + rows for a, b, _ in plan)
    assert all(b == plan[i + 1][0] for i, (_, b, _) in enumerate(plan[:-1]))
    assert [p for _, _, here in plan for p in here] == sorted(p for p in set(points) if p > start)
    assert all(a < p <= b and len(here) <= marks for a, b, here in plan for p in here)
    assert len(plan) <= len(_old_plan(start, n, rows, points))


def _decoder(prompts, stops, *, slots, rows=4_096):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    m = object.__new__(MultiDecoder)
    m.prefill_rows, m.share, m.round_s, m.row_s = rows, 0.0, None, None
    m.streams, m.passed, m.slots = {}, {}, [None] * slots
    m.filling = [SimpleNamespace(sid=i, prompt=[0] * n, background=False, draft=True,
                                 st=SimpleNamespace(image_positions=None)) for i, n in enumerate(prompts)]
    m.fills = {s.sid: [SimpleNamespace(stops=list(p)), True, 0, None] for s, p in zip(m.filling, stops)}
    return m


def _segs(pieces):
    segs, at = [], 0
    for _, _, n in pieces:
        segs.append((None, at, at + n))
        at += n
    return segs


def test_a_lone_prompt_keeps_its_message_points_and_end_inside_one_piece():
    m = _decoder([3_000], [[1_200, 2_900]], slots=4)
    pieces = m._pieces()
    assert [(s.sid, a, n) for s, a, n in pieces] == [(0, 0, 3_000)]
    assert [(c.row, c.at) for c in m._cuts(pieces, _segs(pieces))] == [(1_200, 0), (2_900, 0), (2_999, 0)]


def test_with_every_slot_filling_a_piece_keeps_one_state_as_released():
    m = _decoder([3_000], [[1_200, 2_900]], slots=1)
    pieces = m._pieces()
    assert [(s.sid, a, n) for s, a, n in pieces] == [(0, 0, 1_200)]
    assert m._cuts(pieces, _segs(pieces)) == []
    m.fills[0][2] = 1_200
    pieces = m._pieces()
    assert [(s.sid, a, n) for s, a, n in pieces] == [(0, 1_200, 1_700)]


@pytest.mark.parametrize("seed", range(30))
def test_a_pass_keeps_at_most_one_new_state_a_slot(seed):
    from tensorfold.cuda.kernels.gdn import MARKS

    rng = random.Random(seed)
    lengths = [rng.randint(300, 9_000) for _ in range(rng.randint(1, 4))]
    stops = [sorted(rng.sample(range(256, n - 1), rng.randint(0, 12))) for n in lengths]
    m = _decoder(lengths, stops, slots=rng.randint(len(lengths), 6), rows=rng.choice([1_024, 4_096]))
    kept = {s.sid: [] for s in m.filling}
    while m.filling:
        pieces = m._pieces()
        assert pieces
        cuts = m._cuts(pieces, _segs(pieces))
        states = 0
        for (s, a, n), (_, a0, _) in zip(pieces, _segs(pieces)):
            points = m._points(s, a, n)
            mine = [c.row for c in cuts if c.at == a0]
            assert len(mine) <= MARKS and [a + r for r in mine] == [p for p in points if p < a + n]
            kept[s.sid] += points
            states += len(points)
        assert states <= len(m.slots)
        for s, a, n in pieces:
            m.fills[s.sid][2] = a + n
            if a + n == len(s.prompt):
                m.filling.remove(s)
    assert [kept[i] for i in range(len(lengths))] == [st + [n - 1] for st, n in zip(stops, lengths)]


@pytest.mark.parametrize("seed", range(20))
def test_a_lone_prompts_pieces_are_the_serial_engines_chunks(seed):
    """One rule caps a chunk's points (``cap_points``): the decoder's pieces of a lone prompt with every other slot
    free are the serial plan's chunks at MARKS."""

    from tensorfold.cuda.kernels.gdn import MARKS
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan, entry_end

    rng = random.Random(seed)
    n, rows = rng.randint(300, 9_000), rng.choice([512, 1_024, 4_096])
    stops = sorted(rng.sample(range(1, n - 1), rng.randint(1, 20)))
    m = _decoder([n], [stops], slots=MARKS + 1, rows=rows)
    pieces = []
    while m.filling:
        (s, a, k), = m._pieces()
        pieces.append((a, a + k, m._points(s, a, k)))
        m.fills[0][2] = a + k
        if a + k == n:
            m.filling.remove(s)
    assert pieces == chunk_plan(0, n, rows, [*stops, entry_end([0] * n)], MARKS)


def test_forward_refuses_marks_one_chain_cannot_keep():
    from tensorfold.families.qwen4_exp.cuda.forward import Cut, forward

    b = SimpleNamespace(prefill=True)
    for cuts in ([Cut(r) for r in (1, 2, 3, 4, 5)], [Cut(3), Cut(3)], [Cut(0)], [Cut(8)], [Cut(2, at=4)]):
        with pytest.raises(ValueError, match="no kept points"):
            forward(None, None, b, list(range(8)), cuts=cuts)
    with pytest.raises(ValueError, match="no kept points"):
        forward(None, None, SimpleNamespace(prefill=False), list(range(8)), cuts=[Cut(2)])
