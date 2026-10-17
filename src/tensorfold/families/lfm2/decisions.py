"""The d1 decision format: a state and typed questions in, answers read at the answer slot's option tokens.

Questions follow the Decision Index schema (``type``, ``instructions``, ``criteria``): ``noul`` (yes or no; the answer
is P(yes)), ``choice`` (one of named options) and ``score`` (2 to 10 ordered levels; the answer is the expected
level). The prompt is the one Liquid AI's ``prompt.py`` writes with its defaults (``json_only`` states, no system turn,
the tokenizer's BOS), which llama-server's System One renders too; each option scores its best single-token form, and
the answer is a softmax over the options alone, so the vocabulary's mass never enters it. Ported from
plotarmordev/d1-tensorfold (Apache 2.0).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tensorfold.server.decisions import DecisionError

IM_START, IM_END, BOS = "<|im_start|>", "<|im_end|>", "<|startoftext|>"
YES, NO = ("yes", "Yes", "YES"), ("no", "No", "NO")
# codes tried after an option's own when that one is not a single free token (prompt.py's _FALLBACK_POOL)
POOL = ([chr(c) for c in range(65, 91)] + [f"{i:02d}" for i in range(100)] + [chr(c) for c in range(97, 123)]
        + [f"#{i}" for i in range(200)] + [chr(a) + chr(b) for a in range(65, 91) for b in range(65, 91)])


@dataclass(frozen=True)
class Question:
    kind: str                         # noul | choice | score
    instructions: str
    keys: tuple[str, ...] = ()        # choice: option names; score: level texts; noul: () or ("true", "false")
    texts: tuple[Any, ...] = ()       # choice: descriptions (or None); noul: the true and false descriptions


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def question(raw: Any) -> Question:
    """One question of a System One request, refused (HTTP 400) when the format cannot express it."""

    if not isinstance(raw, dict):
        raise DecisionError("a question is an object with type, instructions and criteria")
    kind = raw.get("type", "choice")
    if raw.get("instructions") is None:
        raise DecisionError("a question needs instructions")
    instructions = _text(raw["instructions"])
    criteria = raw.get("criteria")
    if kind == "noul":
        if criteria is None:
            return Question("noul", instructions)
        if not isinstance(criteria, dict):
            raise DecisionError("noul criteria is an object with true and false")
        return Question("noul", instructions, ("true", "false"), (criteria.get("true"), criteria.get("false")))
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise DecisionError("score criteria is a list of 2 to 10 levels, lowest first")
        return Question("score", instructions, tuple(_text(level) for level in criteria))
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise DecisionError("choice criteria is a non-empty object of option: description")
        return Question("choice", instructions, tuple(str(key) for key in criteria), tuple(criteria.values()))
    raise DecisionError(f"unknown question type {kind!r}: choose noul, choice or score")


class Format:
    """Prompt pieces and readout tokens for one ``tokenizers.Tokenizer``."""

    def __init__(self, tok):
        self.tok = tok
        self._one = lru_cache(maxsize=65536)(self._one_token)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def _one_token(self, text: str) -> int | None:
        ids = self.encode(text)
        return ids[0] if len(ids) == 1 else None

    def codes(self, keys: tuple[str, ...]) -> list[tuple[str, int]]:
        """A distinct single-token code an option: the keys when all are single letters, else A, B, ... (00, 01, ...
        past 26), each replaced by the next free single-token code from ``POOL`` when it is not one token."""

        keys = tuple(key.strip() for key in keys)
        if keys and all(len(key) == 1 and key.isalpha() for key in keys):
            wanted = list(keys)
        elif len(keys) <= 26:
            wanted = [chr(65 + i) for i in range(len(keys))]
        else:
            wanted = [f"{i:02d}" for i in range(len(keys))]
        used: set[int] = set()
        out = []
        for code in wanted:
            for candidate in (code, *POOL):
                tid = self._one(candidate)
                if tid is not None and tid not in used:
                    used.add(tid)
                    out.append((candidate, tid))
                    break
            else:
                raise DecisionError(f"no single-token code left for {len(keys)} options")
        return out

    def readout(self, q: Question) -> list[list[int]]:
        """Token ids an option: an option scores its best form."""

        def forms(texts) -> list[int]:
            seen: list[int] = []
            for text in texts:
                tid = self._one(text)
                if tid is not None and tid not in seen:
                    seen.append(tid)
            return seen

        if q.kind == "noul":
            groups = [forms(YES), forms(NO)]
            if not all(groups):
                raise DecisionError("this tokenizer has no single-token yes or no, which a noul question is read at")
            return groups
        if q.kind == "score":
            groups = [forms((str(i),)) for i in range(len(q.keys))]
            if not all(groups):
                raise DecisionError(f"a score with {len(q.keys)} levels needs single-token digits")
            return groups
        groups = []
        for code, tid in self.codes(q.keys):
            spaced = self._one(" " + code)
            groups.append([tid] + ([spaced] if spaced is not None and spaced != tid else []))
        return groups

    @staticmethod
    def state_block(state: Any) -> str:
        if isinstance(state, str):
            return state + "\n\n"
        return json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"

    def prefix(self, state: Any, images: str = "") -> str:
        """Everything before the question, shared by every question on a state: the pictures' markup, then the state.
        With no state (None) the question follows the pictures directly."""

        body = "" if state is None else self.state_block(state) + "\nQUESTION:\n"
        return f"{BOS}{IM_START}user\n{images}{body}"

    def question_text(self, q: Question) -> str:
        if q.kind == "choice":
            lines = [f"{code} {_text(text) if text else key.replace('_', ' ')}"
                     for (code, _), key, text in zip(self.codes(q.keys), q.keys, q.texts)]
            return f"{q.instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\nReply with the option code only."
        if q.kind == "noul":
            # both sides once either has a description, a missing one as None (llama-server's systemone template)
            extra = "".join(f"\n{name}: {'None' if text is None else _text(text)}" for name, text in
                            zip(("Yes", "No"), q.texts)) if any(text is not None for text in q.texts) else ""
            return f"{q.instructions}{extra}\n\nReply with yes or no only."
        legend = "\n".join(f"{i} {name}" for i, name in enumerate(q.keys))
        return f"{q.instructions}\n\n{legend}\n\nReply with a single digit 0-{len(q.keys) - 1} only."

    def suffix(self, q: Question) -> str:
        """The question and the assistant header, up to the answer slot."""

        return f"{self.question_text(q)}{IM_END}\n{IM_START}assistant\n"


def answer(q: Question, scores: list[float]) -> dict[str, Any]:
    """A noul's P(yes); a choice's pick and its probabilities; a score's expected level (api.py's shapes)."""

    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    total = math.fsum(weights)
    p = [w / total for w in weights]
    if q.kind == "noul":
        return {"type": "noul", "noul": p[0]}
    best = max(range(len(p)), key=p.__getitem__)
    if q.kind == "choice":
        return {"type": "choice", "choice": q.keys[best], "confidence": p[best], "probabilities": dict(zip(q.keys, p))}
    return {"type": "score", "score": math.fsum(i * x for i, x in enumerate(p)), "confidence": p[best],
            "probabilities": {str(i): x for i, x in enumerate(p)}, "legend": {str(i): t for i, t in enumerate(q.keys)}}
