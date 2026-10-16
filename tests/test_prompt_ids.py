"""A prompt encoded by its pieces between added tokens equals the whole text's encoding (``cuda.prompt_ids``)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.prompt_ids import PromptIds

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
SERVED = ["Qwen3.8-Flash-Next-exl3-3.05bpw", "glm53-flash-sushi-experts-mosaic-dense"]
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


def conversation(turns: int) -> list[dict]:
    """A coding session: tool calls with reasoning, results with code, Unicode, and a combining mark after a tag."""

    messages = [{"role": "system", "content": "You work on a repository.\n\n  Indented, trailing space "},
                {"role": "user", "content": "Fix the tests — naïve café ≠ café; 日本語 🙂 ≯ <think>literal"}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"Reading part {i}.\n", "reasoning_content": f"Step {i}: look.",
                         "tool_calls": [{"id": f"call_{i}", "type": "function",
                                         "function": {"name": "read", "arguments": f'{{"path": "src/m{i}.ts"}}'}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{i}",
                         "content": f"export function f{i}(x: number) {{\n\treturn x * {i};  \n}}\n\n\n" * 3})
    return messages


def served(name: str):
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    folder = MODELS / name
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{folder} is not here")
    return Tokenizer.from_file(str(folder / "tokenizer.json")), ChatTemplate(folder)


@pytest.mark.parametrize("name", SERVED)
@pytest.mark.parametrize("thinking", [True, False])
def test_pieces_equal_the_whole_encoding_on_the_served_checkpoints(name, thinking):
    tok, template = served(name)
    ids = PromptIds(tok)
    assert ids.split is not None                          # these checkpoints' added tokens split cleanly
    for turns in (0, 1, 7, 30):
        text = template.render(conversation(turns), tools=TOOLS, enable_thinking=thinking)
        assert ids.encode(text) == tok.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("name", SERVED)
def test_a_later_turn_encodes_only_its_new_pieces(name):
    tok, template = served(name)
    ids = PromptIds(tok)
    ids.encode(template.render(conversation(10), tools=TOOLS, enable_thinking=True))
    before = ids.misses
    text = template.render(conversation(11), tools=TOOLS, enable_thinking=True)
    assert ids.encode(text) == tok.encode(text, add_special_tokens=False).ids
    assert 0 < ids.misses - before <= 6                   # the new turn's few pieces, not the session's


class Added:
    def __init__(self, content, **flags):
        self.content = content
        self.lstrip = flags.get("lstrip", False)
        self.rstrip = self.single_word = self.normalized = False


class Tok:
    def __init__(self, added):
        self.added, self.calls = added, []

    def get_added_tokens_decoder(self):
        return {i: a for i, a in enumerate(self.added, start=100)}

    def encode(self, text, add_special_tokens=False):
        self.calls.append(text)
        return NS(ids=[ord(c) for c in text])


def test_a_tokenizer_whose_added_tokens_strip_whitespace_encodes_whole():
    tok = Tok([Added("<a>", lstrip=True)])
    ids = PromptIds(tok)
    assert ids.split is None and ids.encode("x <a>y") == [ord(c) for c in "x <a>y"] and tok.calls == ["x <a>y"]


def test_pieces_past_the_limit_go_least_recently_used_first():
    tok = Tok([Added("<s>")])
    ids = PromptIds(tok, limit=6)
    assert ids.encode("abc<s>def") == [97, 98, 99, 100, 100, 101, 102]
    ids.encode("ghi<s>abc")                               # ghi pushes the oldest piece, abc, out; abc comes back
    assert list(ids.pieces) == ["ghi", "abc"] and ids.chars == 6
