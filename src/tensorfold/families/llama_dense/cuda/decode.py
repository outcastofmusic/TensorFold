"""One stream's decode: serial steps, or chains copied from the context and DFlash2 trees verified in one window, each
drafted token the one serial decoding gives (a row's bits are the same alone and in every window)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen3_5.cuda.decode import CopyIndex, DecodeResult, clone_state, next_copy_rows
from tensorfold.families.qwen3_5.cuda.forward import State, _mm, _paths, commit

from .forward import new_state, tree_forward
from .prefill import prefill_state
from .weights import Weights

FIRST_COPY_ROWS = 16          # a copy's first window (room for a backed copy: 8 matching tokens); doubles while whole
TREE_ROWS = 16                # a DFlash2 tree's rows, as the 27B verifies on one GB10


def _ids(ids: Sequence[int], device) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device=device)


@torch.no_grad()
def prefill(w: Weights, prompt: Sequence[int], sampling: Sampling | None, *, state: State | None = None,
            limit: int = 0, room=None, keep: Sequence[int] = (), draft=None) -> tuple[State, int, list]:
    """Commit the prompt (from a kept ``state``: a fresh prefill's bits), sample the first reply token, and clone the
    state after prompt[:p] for each ``p`` in ``keep`` (with a ``draft``: (state, drafter snapshot) pairs)."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    st = clone_state(state) if state is not None else new_state(w)
    if state is None:
        st.limit = limit                    # a fresh state's caches stop growing here; a resumed one keeps its own
    if room is not None:
        st.room = room                      # before a grow, the engine frees other conversations' kept buffers
    normed, kept = prefill_state(w, prompt, st, keep=keep, draft=draft)
    pending = sample_rows(_mm(normed, w.head), [len(prompt)], sampling)[0]
    return st, pending, kept


@torch.no_grad()
def serial_decode(w: Weights, st: State, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = True) -> DecodeResult:
    """One-row steps through the verify path: the reference every drafted reply must equal."""

    if count < 1:
        raise ValueError("count must include the first sampled token")
    st = clone_state(st)
    out = [pending]
    start = time.perf_counter()
    while len(out) < count and (not stop_eos or out[-1] not in w.config.eos):
        logits, record = tree_forward(w, _ids([out[-1]], w.norm.device), [-1], st)
        token = sample_rows(logits, [st.pos + 1], sampling)[0]
        commit(st, record, [0])
        out.append(token)
    torch.cuda.synchronize()
    rounds = len(out) - 1
    return DecodeResult(out, time.perf_counter() - start, rounds, 0, 0, widths=[1] * rounds)


@torch.no_grad()
def draft_decode(w: Weights, st: State, prompt: Sequence[int], pending: int, count: int,
                 sampling: Sampling | None, *, max_rows: int = 128, allow_copy: bool = True, stop_eos: bool = True,
                 on_tokens: Callable[[list[int]], bool | None] | None = None, inplace: bool = False,
                 trace: list | None = None, draft=None, tree_rows: int = TREE_ROWS) -> DecodeResult:
    """Copy chains from the context verified as one window, else a ``draft`` (DFlash2, its context at st.pos) tree of
    up to ``tree_rows``; a round with neither decodes one row."""

    if count < 1 or not 1 <= max_rows <= 128:
        raise ValueError("count >= 1 and 1 <= max_rows <= 128 required")
    st = st if inplace else clone_state(st)
    out = [pending]
    context = list(prompt) + out
    copies = CopyIndex() if allow_copy and max_rows > 1 else None
    first = min(max_rows, FIRST_COPY_ROWS)
    rows = first
    tree_rows = min(tree_rows, max_rows)
    taps = draft.tap_layers if draft is not None else ()
    spent = {"draft": 0.0, "verify": 0.0, "commit": 0.0}
    rounds = drafted = accepted = 0
    widths: list[int] = []
    start = time.perf_counter()
    while len(out) < count and (not stop_eos or out[-1] not in w.config.eos):
        t0 = time.perf_counter()
        copied = copies.propose(context, rows - 1) if copies is not None else []
        if copied:
            guesses, parents = copied, list(range(-1, len(copied) - 1))
        elif draft is not None and tree_rows > 1:
            guesses, parents = draft.propose_tree(out[-1], len(context), tree_rows - 1, sampling)
        else:
            guesses, parents = [], []
        tokens = [out[-1]] + guesses
        tree = [-1] + [0 if p < 0 else p + 1 for p in parents]
        t1 = time.perf_counter()
        logits, record, *tapped = tree_forward(w, _ids(tokens, w.norm.device), tree, st, taps=taps)
        depths, _ = _paths(tree)
        sampled = sample_rows(logits, [st.pos + 1 + d for d in depths], sampling)
        t2 = time.perf_counter()
        children = {}
        for r in range(1, len(tokens)):
            children.setdefault((tree[r], tokens[r]), r)
        path = [0]                           # row 0 always holds; a child holds while its parent sampled its token
        while len(out) + len(path) < count and not (stop_eos and sampled[path[-1]] in w.config.eos):
            child = children.get((path[-1], sampled[path[-1]]))
            if child is None:
                break
            path.append(child)
        commit(st, record, path)
        if draft is not None:
            draft.add_taps(tapped[0][path])
        new = [tokens[r] for r in path[1:]] + [sampled[path[-1]]]
        out.extend(new)
        context.extend(new)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        spent["draft"] += t1 - t0
        spent["verify"] += t2 - t1
        spent["commit"] += t3 - t2
        if copied:
            rows = next_copy_rows(rows, len(path) == len(tokens), first, max_rows)
        if trace is not None:
            trace.append({"source": "copy" if copied else "tree" if guesses else "one", "rows": len(tokens),
                          "accepted": len(path) - 1, "verify_ms": round(1000 * (t2 - t1), 3)})
        rounds += 1
        drafted += len(tokens) - 1
        accepted += len(path) - 1
        widths.append(len(tokens))
        if on_tokens is not None and on_tokens(new):
            break
    return DecodeResult(out, time.perf_counter() - start, rounds, drafted, accepted, spent["draft"],
                        spent["verify"], 0.0, spent["commit"], widths)
