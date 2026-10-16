"""A gate's cut written into GLM's live stream (``glm5_next.cuda.decode.Splice``): the fix's first token is the
pending one and the rest ride as drafts later rounds write unverified, so the reply is not prefilled again."""

from __future__ import annotations

import importlib
from types import SimpleNamespace as NS

import pytest

from tensorfold.engine.call_gate import ThinkBudget, generate_gated

pytestmark = pytest.mark.torch

END, CLOSE = 99, [7, 99, 8]
AFTER_FIX = 500


def nxt(t):
    """The fake model: the next token follows the last one; after the fix's last token it starts a new run."""
    return AFTER_FIX if t == 8 else t + 1


@pytest.fixture
def loops(monkeypatch):
    import torch

    decode = importlib.import_module("tensorfold.families.glm5_next.cuda.decode")
    choice = importlib.import_module("tensorfold.families.glm5_next.cuda.drafter_choice")
    commits = []

    def commit(w, st, b, R, keep):
        commits.append((R, keep))
        st.pos += keep

    for module in (decode, choice):
        monkeypatch.setattr(module, "commit", commit)
        monkeypatch.setattr(module, "_sync", lambda w: None)
        monkeypatch.setattr(module, "absorb", lambda *a: None)
        monkeypatch.setattr(module, "draft", lambda e, rows, toks, pos, depth, *a: [
            t for t in _chain(toks[-1], depth)])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return decode, choice, commits


def _chain(t, depth):
    out = []
    for _ in range(depth):
        t = nxt(t)
        out.append(t)
    return out


class Engine:
    def __init__(self):
        import torch

        self.w = NS(cfg=NS(eos=(1,), hidden=4), comm=None, device="cpu")
        self.st = NS(pos=10, mtp_drafted=0)
        self.buf = NS(taps=[])
        self.last_hidden = torch.zeros(1, 4, dtype=torch.bfloat16)
        self.windows = []

    def verify_window(self, tokens):
        return list(tokens)

    def forward(self, tokens):
        import torch

        self.windows.append(list(tokens))
        return torch.zeros(len(tokens), 8)

    def sample(self, logits, positions, sampling):
        return [nxt(t) for t in self.windows[-1]]

    def follow(self, tokens):
        pass

    def main_hidden(self, rows):
        import torch

        return torch.zeros(rows.stop - rows.start, 4, dtype=torch.bfloat16)

    def tap_rows(self, keep):
        import torch

        return torch.zeros(keep, 0, dtype=torch.bfloat16)


class Drafter:
    def propose(self, last, depth, sampling, confidence):
        return _chain(last, depth)

    def add_taps(self, rows):
        pass


def run(loops, path, splice, first, count, fed):
    decode, choice, _ = loops
    e = Engine()
    sp, lead = decode.lead_of(splice, first)
    fed.extend(lead)
    feed = lambda new: fed.extend(new) or False                      # noqa: E731
    policy = decode.DepthPolicy(3, fixed=True)
    if path == "serial":
        res = decode.serial_decode(e, lead[0], count, None, on_tokens=feed, splice=sp, lead=lead)
    elif path == "mtp":
        res = decode.mtp_decode(e, lead[0], count, None, policy=policy, on_tokens=feed, splice=sp, lead=lead)
    elif path == "dflash":
        res = decode.dflash_decode(e, Drafter(), lead[0], count, None, policy=policy, on_tokens=feed, splice=sp,
                                   lead=lead)
    else:
        res = choice.auto_decode(e, None, lead[0], count, None, choice=None, m_policy=policy, f_policy=policy,
                                 on_tokens=feed, splice=sp, lead=lead)
    return e, res


PATHS = ["serial", "mtp", "dflash", "auto"]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("budget", [1, 2, 5, 6])
def test_a_budget_close_is_written_into_the_live_stream_once(loops, path, budget):
    _, _, commits = loops
    runs, fed = [], []

    def inline(prompt, count, feed, splice):
        runs.append(list(prompt))
        e, res = run(loops, path, splice, 10, count, fed)
        inline.e, inline.res = e, res
        return {"rounds": res.rounds}

    gate = ThinkBudget(budget, CLOSE, END)
    generate_gated(None, [1, 2, 3], 14, [gate], lambda new: False, inline)
    reply = [10 + i for i in range(budget - 1)] + CLOSE
    while len(reply) < 30:
        reply.append(nxt(reply[-1]))
    assert runs == [[1, 2, 3]]                                       # one run: the reply is not prefilled again
    assert fed[:14] == reply[:14] and inline.res.tokens == reply[:len(inline.res.tokens)]
    # the cache holds the reply's tokens in order, the fix's too: each round's window up to the rows it committed
    written = [t for window, (_, rows) in zip(inline.e.windows, commits) for t in window[:rows]]
    assert written == reply[:len(written)] and len(written) >= len(inline.res.tokens) - 1
    assert inline.e.st.pos - 10 == len(written)


@pytest.mark.parametrize("path", PATHS)
def test_without_a_cut_the_reply_is_the_models(loops, path):
    fed = []
    _, res = run(loops, path, lambda new: None, 10, 12, fed)
    assert fed == [10 + i for i in range(12)] and res.tokens == fed


@pytest.mark.parametrize("path", ["mtp", "dflash", "auto"])
def test_a_fixs_rounds_count_as_neither_drafted_nor_accepted(loops, path):
    fed = []
    cut = {"done": False}

    def splice(new):
        if not cut["done"] and 13 in new:
            cut["done"] = True
            return new.index(13), CLOSE
        return None

    _, res = run(loops, path, splice, 10, 20, fed)
    assert fed[:6] == [10, 11, 12, *CLOSE]
    assert res.accepted <= res.drafted
    if path != "serial":
        assert "s" in res.arms if path == "auto" else True
