"""Request text in a rendered prompt: a control-token spelling the request wrote encodes as plain text, so only the
chat template's own spellings become control tokens (upstream TensorFold #1117).

Before the template renders, each added token's spelling in the request's strings becomes an escape (``ESC`` and
one code character), so the rendered text holds literal spellings only where the template wrote them. The encoder
splits there and gives each escape the ordinary tokenizer's ids for the spelling, from a copy of the tokenizer
without its added vocabulary. A request without such a spelling renders and encodes as it did before.

A spelling can also form across a seam: two text parts the template joins (``x<|im_`` then ``end|>``), or a request
string the template closes (``'<function=' + name + '>'``). A string that starts or ends with a piece of a spelling
gets a boundary escape on that side, and after the render a boundary stays only where a spelling runs across it, so
the encoder spells that text plainly too.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

ESC = "\U0010FFFD"              # the last private-use character; a request's own one is escaped as well
_FIRST = 0xF0000                # code characters after ESC: the request's own ESC, then each spelling in turn
BOUNDARY = ESC + "\U000FFFFD"   # the edge of a request string, which no spelling may cross; it unmarks to nothing


class RequestText:
    def __init__(self, tok: Any) -> None:
        self.tok = tok
        added = tok.get_added_tokens_decoder() if hasattr(tok, "get_added_tokens_decoder") else {}
        self.added_ids = frozenset(int(i) for i in added)
        spellings = sorted({str(t.content) for t in added.values()}, key=len, reverse=True)
        originals = [ESC, *spellings]
        self.escape = {s: ESC + chr(_FIRST + i) for i, s in enumerate(originals)}
        self.original = {e: s for s, e in self.escape.items()}
        self.found = re.compile("|".join(re.escape(s) for s in originals))
        self.escapes = re.compile(re.escape(ESC) + ".", re.DOTALL)
        self.spellings = spellings
        self.heads = {s[:k] for s in spellings for k in range(1, len(s))}   # a string ending with one may open a spelling
        self.tails = {s[k:] for s in spellings for k in range(1, len(s))}   # one starting with one may close a spelling
        self.reach = max((len(s) for s in spellings), default=1) - 1

    @functools.cached_property
    def plain(self) -> Any:
        """The tokenizer without its added vocabulary, made when a request first spells an added token."""

        from tokenizers import Tokenizer

        config = json.loads(self.tok.to_str())
        config["added_tokens"] = []
        return Tokenizer.from_str(json.dumps(config))

    def _opens(self, text: str) -> bool:
        return any(text[-k:] in self.heads for k in range(1, min(len(text), self.reach) + 1))

    def _closes(self, text: str) -> bool:
        return any(text[:k] in self.tails for k in range(1, min(len(text), self.reach) + 1))

    def holds(self, value: Any) -> bool:
        """Whether a string anywhere in ``value`` (JSON-shaped: dicts, lists, strings) holds a spelling or ``ESC``, or
        starts or ends with a piece of a spelling. A message's role is one of five words and is left out."""

        todo = [value]                  # a stack, so a deeply nested request is no deeper here than in json
        while todo:
            item = todo.pop()
            if isinstance(item, str):
                if self.found.search(item) or self._opens(item) or self._closes(item):
                    return True
            elif isinstance(item, dict):
                todo += [k for k in item]
                todo += [v for k, v in item.items() if k != "role"]
            elif isinstance(item, (list, tuple)):
                todo += item
        return False

    def mark(self, value: Any) -> Any:
        """``value`` with every spelling in its strings escaped and a boundary at an edge that holds a piece of one, as
        a new object."""

        if isinstance(value, str):
            text = self.found.sub(lambda m: self.escape[m.group(0)], value)
            return (BOUNDARY if self._closes(text) else "") + text + (BOUNDARY if self._opens(text) else "")
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                out[self.mark(k)] = v if k == "role" else self.mark(v)
            return out
        if isinstance(value, (list, tuple)):
            return [self.mark(v) for v in value]
        return value

    def seal(self, text: str) -> str:
        """A rendered ``text`` with only the boundaries a spelling runs across: the rest go, so text without a seam
        spelling encodes as it did before."""

        parts = text.split(BOUNDARY)
        if len(parts) == 1:
            return text
        joined = "".join(parts)
        cuts, at = [], 0
        for part in parts[:-1]:
            at += len(part)
            cuts.append(at)
        out, last = [], 0
        for cut in cuts:
            near = joined[max(0, cut - self.reach):cut + self.reach]
            edge = cut - max(0, cut - self.reach)
            if any(-1 < near.find(s, max(0, edge - len(s) + 1)) < edge for s in self.spellings):
                out += [joined[last:cut], BOUNDARY]
                last = cut
        out.append(joined[last:])
        return "".join(out)

    def unmark(self, text: str) -> str:
        return self.escapes.sub(lambda m: "" if m.group(0) == BOUNDARY else self.original.get(m.group(0), m.group(0)),
                                text)

    def ids(self, text: str) -> list[int]:
        """The plain ids of ``text`` (escapes and all): no added token, so a spelling is the text it spells."""

        plain = self.unmark(text)
        ids = list(self.plain.encode(plain, add_special_tokens=False).ids)
        if self.added_ids.isdisjoint(ids):
            return ids
        # a model vocabulary that holds a spelling too (Unigram's): spell each one in two pieces
        out: list[int] = []
        at = 0
        for m in self.found.finditer(plain):
            for piece in (plain[at:m.start()], m.group(0)[:1], m.group(0)[1:]):
                if piece:
                    out += self.plain.encode(piece, add_special_tokens=False).ids
            at = m.end()
        if at < len(plain):
            out += self.plain.encode(plain[at:], add_special_tokens=False).ids
        return out
