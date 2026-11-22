"""response_format on Hy4-flash: the tool-eval audit of 22 Nov saw 20 requests with ``response_format`` refused with
HTTP 400 ("this model's engine does not enforce structured output"), because ``Hy4App.generate`` took no
``constraint``. Now prefill, serial steps and drafted windows mask their rows by the request's grammar, as the qwen3_5
path does (the tiny checkpoint on the CPU, a toy vocabulary where token t spells chr(t))."""

from __future__ import annotations

import inspect
import json

import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch

SCHEMA = {"type": "object", "properties": {"k": {"type": "integer"}, "tag": {"type": "string", "enum": ["a", "bb"]}},
          "required": ["k"], "additionalProperties": False}
EOS = 1                                     # the tiny checkpoint's end token, the grammar's stop token


@pytest.fixture(scope="module")
def grammars():
    xgr = pytest.importorskip("xgrammar")
    from tensorfold.engine import grammar

    vocab = ["\x00", ""] + [chr(t) for t in range(2, tiny.V)]
    vocab[EOS] = "</s>"
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=tiny.V, stop_token_ids=[EOS])
    return grammar.Grammars(info)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    from tensorfold.families.hy4_flash.cuda.weights import load

    return load(tiny.write(tmp_path_factory.mktemp("hy4") / "tiny"), "cpu")


def _spec(kind: str):
    from tensorfold.engine import grammar

    return grammar.Spec("json") if kind == "json_object" else grammar.Spec("json_schema", json.dumps(SCHEMA))


class _Oracle:
    """Drafts the serial reply's next four tokens with the third replaced by "@", which no JSON path takes after a
    digit or a brace: windows must drop it (and what follows) before the verify."""

    def __init__(self, serial: list[int], prompt: int) -> None:
        self.serial, self.prompt = serial, prompt

    def propose(self, e, last: int, sampling) -> list[int]:
        at = e.pos - self.prompt + 1
        guesses = list(self.serial[at:at + 4])
        if len(guesses) > 2:
            guesses[2] = ord("@")
        return guesses

    def commit(self, e, accepted: int) -> None:
        pass


def test_the_app_takes_a_constraint():
    """The server refuses structured output for an engine whose ``generate`` has no ``constraint`` parameter."""

    from tensorfold.families.hy4_flash.cuda.app import Hy4Engine

    assert "constraint" in inspect.signature(Hy4Engine.generate).parameters


@pytest.mark.parametrize("kind", ["json_object", "json_schema"])
def test_serial_and_drafted_replies_keep_the_grammar(weights, grammars, kind):
    """Serial and drafted replies under the grammar: every token the grammar's, and the drafted rounds drop the "@"
    drafts before verifying. The CPU path's windows are not row-invariant (a 2-row window's logits move by 0.03), and
    under a mask the allowed tokens can sit that close, so drafted against serial token equality is
    tests/cuda/test_hy4_flash.py's check."""

    from tensorfold.families.hy4_flash.cuda import model

    prompt = [(t % 60) + 40 for t in range(24)]
    replies, stats = [], None
    for drafted in (False, True):
        e = model.Engine(weights, max_len=128, prefill_rows=64)
        c = grammars.constraint(grammars.compile(_spec(kind)))
        first = model.prefill(e, prompt, None, constraint=c)
        if drafted:
            res = model.draft_decode(e, prompt, first, 40, None, constraint=c, allow_copy=False,
                                     drafter=_Oracle(replies[0], len(prompt)))
            stats = res.stats
        else:
            res = model.serial_decode(e, first, 40, None, constraint=c)
        replies.append(res.tokens)
    for reply in replies:
        m = grammars.xgr.GrammarMatcher(grammars.compile(_spec(kind)))
        assert all(m.accept_token(t) for t in reply)
    assert stats["accepted"] > 0 and stats["drafted"] < 4 * stats["rounds"]     # drafts landed, "@" was dropped
