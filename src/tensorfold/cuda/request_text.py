"""Request text in a rendered prompt: a control-token spelling the request wrote encodes as plain text, so only the
chat template's own spellings become control tokens (upstream TensorFold #1117).

Before the template renders, each added token's spelling in the request's strings becomes an escape (``ESC`` and
one code character), so the rendered text holds literal spellings only where the template wrote them. The encoder
splits there and gives each escape the ordinary tokenizer's ids for the spelling, from a copy of the tokenizer
without its added vocabulary. A request without such a spelling renders and encodes as it did before.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

ESC = "\U0010FFFD"              # the last private-use character; a request's own one is escaped as well
_FIRST = 0xF0000                # code characters after ESC: the request's own ESC, then each spelling in turn


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

    @functools.cached_property
    def plain(self) -> Any:
        """The tokenizer without its added vocabulary, made when a request first spells an added token."""

        from tokenizers import Tokenizer

        config = json.loads(self.tok.to_str())
        config["added_tokens"] = []
        return Tokenizer.from_str(json.dumps(config))

    def holds(self, value: Any) -> bool:
        """Whether a string anywhere in ``value`` (JSON-shaped: dicts, lists, strings) holds a spelling or ``ESC``."""

        if isinstance(value, str):
            return self.found.search(value) is not None
        if isinstance(value, dict):
            return any(self.holds(k) or self.holds(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return any(self.holds(v) for v in value)
        return False

    def mark(self, value: Any) -> Any:
        """``value`` with every spelling in its strings escaped, as a new object."""

        if isinstance(value, str):
            return self.found.sub(lambda m: self.escape[m.group(0)], value)
        if isinstance(value, dict):
            return {self.mark(k): self.mark(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.mark(v) for v in value]
        return value

    def unmark(self, text: str) -> str:
        return self.escapes.sub(lambda m: self.original.get(m.group(0), m.group(0)), text)

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
