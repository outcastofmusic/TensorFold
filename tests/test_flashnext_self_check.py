"""The load-time self-check: decode steps replayed through every verify width must give the one-row step's bits."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.self_check import check_widths, enabled  # noqa: E402

V = 50


class Fake:
    """A model whose row for a token depends on the token and every token before it; ``bad`` = (width, row) adds
    one ulp-sized error to that row of every window of that width; ``leak``: a partial commit keeps every row."""

    def __init__(self, bad=None, leak=False):
        self.bad, self.leak, self.seq, self.windows, self.kept = bad, leak, [], [], []

    def start(self) -> int:
        self.seq = [3, 1, 4]
        return 7

    def window(self, tokens, keep=None):
        keep = len(tokens) if keep is None else keep
        self.windows.append(len(tokens))
        self.kept.append(keep)
        rows = []
        for i, t in enumerate(tokens):
            self.seq.append(int(t))
            g = torch.Generator().manual_seed(hash(tuple(self.seq)) % (2**31))
            row = torch.randn(V, generator=g)
            if self.bad == (len(tokens), i):
                row[0] = torch.nextafter(row[0], torch.tensor(float("inf")))
            rows.append(row)
        if not self.leak:
            del self.seq[len(self.seq) - len(tokens) + keep:]
        return torch.stack(rows)


def test_equal_bits_at_every_width_pass():
    fake = Fake()
    got = check_widths(fake.start, fake.window, widths=range(2, 8), steps=24)
    assert got["ok"] and got["first_diff"] is None and got["rows_checked"] > 24 * 3
    assert set(fake.windows) == {1, 2, 3, 4, 5, 6, 7}


def test_one_differing_row_fails_and_names_its_width_and_row():
    fake = Fake(bad=(3, 1))
    got = check_widths(fake.start, fake.window, widths=range(2, 8), steps=24)
    assert not got["ok"] and got["first_diff"]["width"] == 3 and got["first_diff"]["row"] == 1


def test_the_check_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_SELF_CHECK", raising=False)
    assert not enabled()
    monkeypatch.setenv("TENSORFOLD_SELF_CHECK", "1")
    assert enabled()


def test_partial_commits_roll_back_every_rejected_row():
    fake = Fake()
    got = check_widths(fake.start, fake.window, widths=range(2, 8), steps=24)
    assert got["ok"] and {k for w, k in zip(fake.windows, fake.kept) if k < w} == {1, 2, 3, 4, 5, 6}


def test_a_rollback_that_keeps_a_rejected_row_fails():
    fake = Fake(leak=True)
    got = check_widths(fake.start, fake.window, widths=range(2, 8), steps=24)
    assert not got["ok"] and got["first_diff"]["kept"] < got["first_diff"]["width"]
