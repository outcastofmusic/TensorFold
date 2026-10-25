"""A rendered prompt's ids by the text between added tokens, kept between requests, so a later turn of a conversation
encodes only its new text.

The tokenizer takes added tokens out of the raw text before it normalizes or pre-tokenizes, so the ids of the pieces
between them, joined with the added tokens' own, equal the whole text's (tests/test_prompt_ids.py checks the served
checkpoints). A tokenizer whose added tokens strip their neighbours' whitespace, or are matched after normalization,
breaks that, and its prompts encode whole. Upstream TensorFold 1.0.9 (#767) keeps pieces the same way: 41.5 ms to
2.3 ms before the prompt pass at 35k tokens. On the Spark a whole 32k-token prompt took 35 ms, a 109k one 122 ms.

A text a template rendered from marked request strings (``cuda.request_text``) holds escapes where the request spelt
an added token: a piece with escapes takes the plain ids, so only the template's own spellings are added tokens.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any

from tensorfold.cuda.request_text import ESC, RequestText

LIMIT_CHARS = 16_000_000        # the pieces' text kept, least recently used first out


class PromptIds:
    def __init__(self, tok: Any, limit: int = LIMIT_CHARS) -> None:
        self.tok, self.limit = tok, int(limit)
        self.lock = threading.Lock()
        self.pieces: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self.chars = 0
        self.hits = self.misses = 0
        added = tok.get_added_tokens_decoder() if hasattr(tok, "get_added_tokens_decoder") else {}
        self.ids = {str(t.content): int(i) for i, t in added.items()}
        clean = all(not (t.lstrip or t.rstrip or t.single_word or t.normalized) for t in added.values())
        self.split = (re.compile("|".join(re.escape(t) for t in sorted(self.ids, key=len, reverse=True)))
                      if clean and self.ids else None)
        self.request = RequestText(tok) if hasattr(tok, "to_str") else None

    def encode(self, text: str) -> list[int]:
        if self.split is None:
            if self.request is None or ESC not in text:
                return list(self.tok.encode(text, add_special_tokens=False).ids)
            out: list[int] = []     # a tokenizer that does not split cleanly: the request's escapes split it
            for i, part in enumerate(re.split(f"((?:{re.escape(ESC)}.)+)", text, flags=re.DOTALL)):
                if part:
                    out += self.request.ids(part) if i % 2 else self.tok.encode(part, add_special_tokens=False).ids
            return out
        out: list[int] = []
        at = 0
        for found in self.split.finditer(text):
            if found.start() > at:
                out += self._piece(text[at:found.start()])
            out.append(self.ids[found.group(0)])
            at = found.end()
        if at < len(text):
            out += self._piece(text[at:])
        return out

    def _piece(self, text: str) -> tuple[int, ...]:
        with self.lock:
            ids = self.pieces.get(text)
            if ids is not None:
                self.pieces.move_to_end(text)
                self.hits += 1
                return ids
        if self.request is not None and ESC in text:
            ids = tuple(self.request.ids(text))
        else:
            ids = tuple(self.tok.encode(text, add_special_tokens=False).ids)
        with self.lock:
            self.misses += 1
            if text not in self.pieces and len(text) <= self.limit:
                self.pieces[text] = ids
                self.chars += len(text)
                while self.chars > self.limit:
                    old, _ = self.pieces.popitem(last=False)
                    self.chars -= len(old)
        return ids
