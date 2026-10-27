"""The incremental copy index proposes exactly what the full scan proposes, round after round."""

import random

from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, copy_chain, next_copy_rows


def test_index_matches_full_scan_as_context_grows():
    rng = random.Random(4)
    base = [rng.randrange(40) for _ in range(300)]
    context = base[:50]
    index = CopyIndex()
    for step in range(400):
        # grow by 1-6 tokens, sometimes copying an earlier stretch so matches exist
        if rng.random() < 0.5 and len(context) > 40:
            at = rng.randrange(len(context) - 20)
            context += context[at:at + rng.randint(1, 6)]
        else:
            context += [rng.randrange(40) for _ in range(rng.randint(1, 6))]
        assert index.propose(context, 31) == copy_chain(context, 31), step


def test_copy_windows_start_at_the_tree_width_double_while_copies_land_whole_and_halve_after_a_break():
    rows, seen = next_copy_rows(16, False, 16, 128), []
    for landed in (True, True, True, True, False, False, False, False, True):
        seen.append(rows)
        rows = next_copy_rows(rows, landed, 16, 128)
    assert seen == [16, 32, 64, 128, 128, 64, 32, 16, 16]
    assert next_copy_rows(8, False, 8, 128) == 16          # a backed copy needs 8 matching tokens and room past them
    assert next_copy_rows(12, True, 12, 12) == 12          # two ranks and concurrent streams: the old fixed width


def test_a_draft_tree_hangs_beside_a_copy_and_merges_where_it_repeats_the_chain():
    from tensorfold.families.qwen3_5.cuda.decode import hang_tree

    # tree: 7 (root), 8 under 7, 5 (root), 9 under 5; the copy chain is 7 8 6
    guesses, parents, chain = hang_tree([7, 8, 6], [7, 8, 5, 9], [-1, 0, -1, 2])
    assert chain == 3
    assert guesses == [7, 8, 6, 5, 9] and parents == [-1, 0, 1, -1, 3]
    # a tree node off the chain keeps its own subtree even when a later token matches the chain
    guesses, parents, _ = hang_tree([7, 8], [4, 8], [-1, 0])
    assert guesses == [7, 8, 4, 8] and parents == [-1, 0, -1, 2]
    assert hang_tree([], [3, 4], [-1, 0]) == ([3, 4], [-1, 0], 0)
