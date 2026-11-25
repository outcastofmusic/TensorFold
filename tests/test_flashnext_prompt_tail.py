"""TENSORFOLD_PROMPT_TAIL: a prompt's last two idle pieces split evenly once a full piece would leave a short one.

sparkDash prompts end 39-57 rows past a multiple of 2,048 (4,135, 8,233, 16,425, 32,806, 131,112), so the plain cut
ends every prompt in a pass of a few dozen rows that reads almost every expert (tf-runs/prefill-gap-1028, item 2).
The GPU side, the same bits as the plain cut, is tests/cuda/test_flashnext_prompt_tail.py."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda.prompt_plan import (
    TAIL_ROWS,
    even_tail,
    even_tails,
    long_pieces,
    piece_rows,
    pieces_of,
    prompt_tail,
)

# (buffer rows, TENSORFOLD_LONG_PIECE_TOKENS installed): FP4-Local served, FP4 at 2,048 rows, EXL3 at either width
FP4_LONG, FP4, EXL3, EXL3_4096 = (4096, 8192), (2048, 0), (2048, 0), (4096, 0)


def widths(n: int, rows: int, long_from: int, tail: bool) -> list[int]:
    """The idle pieces ``piece_rows`` cuts over an n-row prompt."""

    out, left = [], n
    with long_pieces(long_from), even_tails(tail):
        while left:
            w = min(piece_rows(rows, left), left)
            out.append(w)
            left -= w
    return out


@pytest.mark.parametrize("n, plain, even", [
    (4135, [2048, 2048, 39], [2048, 1040, 1047]),
    (8233, [4096, 2048, 2048, 41], [4096, 2048, 1040, 1049]),
    (16425, [4096] * 3 + [2048, 2048, 41], [4096] * 3 + [2048, 1040, 1049]),
    (32806, [4096] * 7 + [2048, 2048, 38], [4096] * 7 + [2048, 1040, 1046]),
    (131112, [4096] * 31 + [2048, 2048, 40], [4096] * 31 + [2048, 1040, 1048]),
])
def test_fp4_local_sparkdash_lengths_end_in_two_even_pieces(n, plain, even):
    assert widths(n, *FP4_LONG, tail=False) == plain
    assert widths(n, *FP4_LONG, tail=True) == even


@pytest.mark.parametrize("shape, n, even", [
    (EXL3, 4135, [2048, 1040, 1047]),
    (EXL3, 8233, [2048] * 3 + [1040, 1049]),
    (EXL3, 131112, [2048] * 63 + [1040, 1048]),
    (EXL3_4096, 4135, [2064, 2071]),
    (EXL3_4096, 8233, [4096, 2064, 2073]),
    (FP4, 4135, [2048, 1040, 1047]),
])
def test_every_piece_width_evens_its_tail(shape, n, even):
    assert widths(n, *shape, tail=True) == even


@pytest.mark.parametrize("n, even", [
    (1, [1]), (39, [39]), (2047, [2047]), (2048, [2048]),         # one piece: nothing to even
    (2049, [1024, 1025]),
    (2048 + TAIL_ROWS - 1, [1264, 1295]),                           # the widest tail that still splits
    (2048 + TAIL_ROWS, [2048, 512]),                                # a 512-row tail stays
    (4096, [2048, 2048]), (6144 + 16, [2048, 2048, 1024, 1040]),
])
def test_edges_around_the_threshold(n, even):
    assert widths(n, *EXL3, tail=True) == even


def test_the_rule_cuts_every_length_whole_and_never_ends_short():
    for shape in (FP4_LONG, EXL3, EXL3_4096):
        rows = shape[0]
        for n in range(1, 3 * rows + 700, 7):
            got = widths(n, *shape, tail=True)
            assert sum(got) == n and max(got) <= rows, (shape, n)
            assert got[-1] >= min(n, TAIL_ROWS), (shape, n, got)
            assert all(w % 16 == 0 for w in got[:-1]), (shape, n, got)


def test_a_narrow_piece_is_never_widened():
    assert even_tail(16, 31) == 16                     # half of 31 rounds to 0 rows: keep the piece
    assert even_tail(256, 456) == 224                  # 224 then 232, both inside the 256-row buffers
    assert even_tail(2048, 2048) == 2048 and even_tail(2048, 100) == 2048


def test_off_by_default_and_warm_ups_keep_the_uneven_tail():
    assert widths(4135, *EXL3, tail=False) == [2048, 2048, 39]
    assert piece_rows(2048, 2087) == 2048              # nothing installed
    with even_tails(True):
        assert piece_rows(2048, 2087) == 1040
        with pieces_of(2048, 2048):                    # a warm prompt of width + 19 rows: one full piece first
            assert piece_rows(2048, 2048 + 19) == 2048
        with pieces_of(2048, 4096), long_pieces(8192):
            assert piece_rows(4096, 2048 + 19) == 2048
        assert piece_rows(2048, 2087) == 1040
    assert piece_rows(2048, 2087) == 2048


def test_the_switch_reads_0_or_1(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_PROMPT_TAIL", raising=False)
    assert prompt_tail() is False
    for value, on in [("", False), (" ", False), ("0", False), ("1", True), (" 1 ", True)]:
        monkeypatch.setenv("TENSORFOLD_PROMPT_TAIL", value)
        assert prompt_tail() is on
    for value in ["2", "on", "true", "-1"]:
        monkeypatch.setenv("TENSORFOLD_PROMPT_TAIL", value)
        with pytest.raises(ValueError, match="TENSORFOLD_PROMPT_TAIL"):
            prompt_tail()


def test_the_expert_tile_threshold_reads_a_row_count(monkeypatch):
    pytest.importorskip("torch")
    from tensorfold.cuda.nvfp4.experts import tile_rows

    monkeypatch.delenv("TENSORFOLD_EXPERT_TILE_ROWS", raising=False)
    assert tile_rows() == 1024
    for value, rows in [("", 1024), ("256", 256), (" 512 ", 512)]:
        monkeypatch.setenv("TENSORFOLD_EXPERT_TILE_ROWS", value)
        assert tile_rows() == rows
    for value in ["0", "-1", "1k", "256.0"]:
        monkeypatch.setenv("TENSORFOLD_EXPERT_TILE_ROWS", value)
        with pytest.raises(ValueError, match="TENSORFOLD_EXPERT_TILE_ROWS"):
            tile_rows()


@pytest.mark.torch
def test_the_serial_plan_evens_its_tail_and_keeps_its_points():
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan

    with even_tails(True):
        plan = chunk_plan(0, 4135, 2048, [4000])
        assert [(a, b, here) for a, b, here in plan] == [(0, 2048, []), (2048, 3088, []), (3088, 4135, [4000])]
        plan = chunk_plan(0, 4135, 2048, [2500, 3000], marks=1)     # the second point inside ends the piece early
        assert [(b - a, here) for a, b, here in plan] == [(2048, []), (452, [2500]), (1635, [3000])]
    with long_pieces(8192), even_tails(True):
        assert [b - a for a, b, _ in chunk_plan(0, 8233, 4096, [])] == [4096, 2048, 1040, 1049]


@pytest.mark.torch
def test_an_idle_pass_evens_the_rows_its_filling_prompts_have_left():
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = object.__new__(MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 2048, 0.0, None, None
    dec.streams, dec.passed, dec.slots = {}, {}, [None, None]
    a = SimpleNamespace(sid=1, prompt=[1] * 4135, background=False)
    dec.filling = [a]
    dec.fills = {1: [SimpleNamespace(stops=[]), False, 0, None]}
    with even_tails(True):
        assert dec._pieces() == [(a, 0, 2048)]
        dec.fills[1][2] = 2048
        assert dec._pieces() == [(a, 2048, 1040)]
        dec.fills[1][2] = 3088
        assert dec._pieces() == [(a, 3088, 1047)]
        dec.fills[1][2] = 2048
        dec.streams[3] = SimpleNamespace(done=False)
        assert dec._pass_rows() == 2048                      # live replies keep the released bound
    dec.streams.clear()
    assert dec._pieces() == [(a, 2048, 2048)]                # off: the plain cut, a 39-row pass after it
