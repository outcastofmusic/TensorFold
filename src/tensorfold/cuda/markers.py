"""Message-start resume points for CUDA engines, found by the Mac planner's rule through the CUDA chat template."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

MIN_GAP = 256            # a snapshot this close to the prompt start or to the previous snapshot saves too little


class TemplateTokens:
    """The tokenizer surface ``message_markers`` reads, over a CUDA server's tokenizer and chat template."""

    def __init__(self, tok: Any, template: Any) -> None:
        self.tok, self.template = tok, template
        self.added_tokens_decoder = tok.get_added_tokens_decoder()
        self.all_special_ids = [int(i) for i, t in self.added_tokens_decoder.items() if t.special]

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, **kwargs: Any):
        text = self.template.template.render(**self.template.specials, messages=messages,
                                             add_generation_prompt=add_generation_prompt, **kwargs)
        return self.tok.encode(text, add_special_tokens=False).ids if tokenize else text


def snapshot_points(openers: Sequence[int], assistant: Sequence[int]) -> Callable[[Sequence[int]], list[int]]:
    """Where a prefill keeps states: the second message's start (a shared system block) and the last assistant start."""

    from tensorfold.engine.prefill_plan import PrefillPlan

    plan = PrefillPlan(openers=openers, assistant=assistant, min_chunk=max(MIN_GAP, len(assistant)))

    def points(ids: Sequence[int]) -> list[int]:
        arr = np.asarray(ids, dtype=np.int64)
        found = plan.points(arr)
        starts = np.flatnonzero(np.isin(arr, plan.openers)) if plan.openers else []
        wanted = ([int(starts[1])] if len(starts) > 1 else []) + ([found[-1]] if found else [])
        out: list[int] = []
        for p in sorted(set(wanted)):
            if MIN_GAP <= p < len(arr) and (not out or p - out[-1] >= MIN_GAP):
                out.append(p)
        return out

    return points


def resume_points(model_dir: str | Path) -> Callable[[Sequence[int]], list[int]] | None:
    """``snapshot_points`` for this checkpoint's chat template, or None when it marks no message starts."""

    from tokenizers import Tokenizer

    from tensorfold.engine.prefill_plan import message_markers

    from .chat_template import ChatTemplate

    model_dir = Path(model_dir)
    if not (model_dir / "tokenizer.json").is_file():
        return None
    try:
        tokens = TemplateTokens(Tokenizer.from_file(str(model_dir / "tokenizer.json")), ChatTemplate(model_dir))
        openers, assistant = message_markers(tokens)
    except (OSError, ValueError, KeyError):
        return None
    return snapshot_points(openers, assistant) if openers or assistant else None


GRID_ENV = "TENSORFOLD_GRID_ANCHORS"


def grid_env() -> int:
    """TENSORFOLD_GRID_ANCHORS=<tokens>: keep prompt states at every multiple of this many tokens (0 or unset: off)."""

    import os

    value = os.environ.get(GRID_ENV, "").strip()
    if not value:
        return 0
    if not value.isdecimal() or int(value) < MIN_GAP:
        raise ValueError(f"{GRID_ENV}: a line spacing of at least {MIN_GAP} tokens, not {value!r}")
    return int(value)


def with_grid(points: Callable[[Sequence[int]], list[int]] | None, step: int) -> Callable[[Sequence[int]], list[int]]:
    """``points`` plus a line every ``step`` tokens from position 0 (below the prompt's end); a line within MIN_GAP
    of a marker gives way to the marker, so a resume finds the nearest state below wherever a prompt departs."""

    def grid(ids: Sequence[int]) -> list[int]:
        marks = sorted(points(ids)) if points is not None else []
        lines = [p for p in range(step, len(ids), step) if all(abs(p - m) >= MIN_GAP for m in marks)]
        return sorted(set(marks) | set(lines))

    return grid
