"""A load-time check that every verify width gives the one-row decode step's bits on this GPU (TENSORFOLD_SELF_CHECK=1).

Drafting is exact only because a verify window's row r holds the same logits, bit for bit, as the one-row step at that
position. The CUDA tests prove it on a tiny random model; this check proves it for the loaded weights and kernels:
decode ``steps`` tokens greedily one row at a time, recording each row, then replay the same tokens from the same
start in windows of each width and compare every row with ``torch.equal``. A difference names the width and the row.
Each width is then replayed again as rounds that reject a draft: a window ends in a wrong token after ``kept`` right
ones, commits only those, and the next window starts where it did, so a rollback that keeps a rejected row's state
shows in the rows after it.
The engine then serves every request without drafts (``draft=False``), which is the serial reference.

The idea comes from TandemLLM's load-time self-check (AGPL-3.0); this is an independent implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from tensorfold.env_flags import flag

ENV = "TENSORFOLD_SELF_CHECK"
PROMPT = [9707, 11, 1879, 13, 1096, 374, 264, 1273, 315, 279, 4479, 1849, 594, 1555, 29136, 13]


def enabled() -> bool:
    return flag(ENV)


@torch.no_grad()
def check_widths(start: Callable[[], int], window: Callable[..., torch.Tensor], *, widths: Iterable[int],
                 steps: int = 32) -> dict:
    """``start()`` empties the state, prefills the check's prompt and returns its first token; ``window(tokens,
    keep)`` verifies ``tokens``, commits the first ``keep`` (all when None) and returns their logit rows. Greedy
    one-row steps set the reference."""

    tok = start()
    fed, ref = [], []
    for _ in range(steps):
        row = window([tok])[0]
        fed.append(tok)
        ref.append(row.clone())
        tok = int(row.argmax())
    checked, bad = 0, None
    for width in widths:
        start()
        for at in range(0, steps - width + 1, width):
            rows = window(fed[at:at + width])
            for r in range(width):
                checked += 1
                if bad is None and not torch.equal(rows[r], ref[at + r]):
                    bad = {"width": int(width), "kept": int(width), "row": r, "position": at + r}
        if bad is not None:
            break
        start()
        at, n = 0, 0
        while at + width <= steps:
            keep = 1 + n % (width - 1) if width > 1 else 1      # every rollback size in turn
            wrong = fed[at + keep] - 1 if fed[at + keep] > 0 else 1
            rows = window([*fed[at:at + keep], wrong, *fed[at + keep + 1:at + width]], keep)
            for r in range(keep):                               # rows past the wrong token follow another text
                checked += 1
                if bad is None and not torch.equal(rows[r], ref[at + r]):
                    bad = {"width": int(width), "kept": keep, "row": r, "position": at + r}
            at, n = at + keep, n + 1
        if bad is not None:
            break
    return {"ok": bad is None, "steps": steps, "rows_checked": checked, "first_diff": bad}


def engine_windows(e, prompt: list[int] = PROMPT):
    """(start, window) over a one-stream ``Engine`` (its graphs when it has them)."""

    from .decode import prefill
    from .forward import commit

    def start() -> int:
        e.reset()
        return prefill(e, prompt, None, mtp=False)

    def window(tokens: list[int], keep: int | None = None) -> torch.Tensor:
        rows = e.forward(tokens)[:len(tokens)].clone()
        commit(e.w, e.st, e.buf, len(tokens), len(tokens) if keep is None else keep)
        return rows

    return start, window


def multi_windows(dec, prompt: list[int] = PROMPT):
    """(start, window, release) over a ``MultiDecoder``'s shared-round path (``compute`` with the multi-stream
    DeltaNet and attention steps, recurrent rows folded in the next round), on a free slot; ``release`` empties it."""

    from . import attn_multi, gdn_multi
    from .decode import prefill
    from .forward import commit, compute, stage
    from .multi import _slot

    st = dec.free[-1]
    held: list[list[int]] = [[]]

    def start() -> int:
        if not dec._grow(st, len(prompt) + 2 * 128, alone=True):      # a check of up to 64 steps and 64-row windows
            raise RuntimeError("no memory for the self-check's slot")
        held[0] = []
        e = _slot(dec.w, st, dec.buf, dec.mbuf, dec.pbuf, dec.capacity, dec.prefill_rows)
        return prefill(e, prompt, None, mtp=False)

    def window(tokens: list[int], keep: int | None = None) -> torch.Tensor:
        keep = len(tokens) if keep is None else keep
        segs = stage(dec.w, dec.buf, [(st, list(tokens))])
        tables = dec.buf.gdn_tables = gdn_multi.Tables(dec.w, dec.gdn, segs, [held[0]])
        dec.buf.attn_step = attn_multi.Step(dec.w, segs, mtp=False)
        try:
            rows = compute(dec.w, segs, dec.buf)[:len(tokens)].clone()
        finally:
            dec.buf.gdn_tables = dec.buf.attn_step = None
        held[0] = gdn_multi.keep(tables, [keep])[0]
        commit(dec.w, st, dec.buf, len(tokens), keep, at=0, states=False)
        return rows

    def release() -> None:
        dec._shrink(st)

    return start, window, release


__all__ = ["ENV", "PROMPT", "check_widths", "enabled", "engine_windows", "multi_windows"]
