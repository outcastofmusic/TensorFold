"""The Llama CUDA engine: one GPU, one stream, startup memory admission, prompt-end and message-start prefix reuse,
copy drafts and, with a drafter, DFlash2 trees."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from tensorfold.families.qwen3_5.cuda.engine import KEEP_ONE, entry_end

MAX_ROWS = 128                # the widest verify window a copy may earn


def geometry(text: dict, rows: int, chunk: int):
    """Bytes at a cache capacity: every layer's keys and values (plus one layer mid-grow), the verify window's
    activations, logits and attention partials, or a prompt chunk's arrays, whichever is larger."""

    from tensorfold.cuda.capacity import Geometry
    from tensorfold.cuda.geometry import prompt_row_bytes
    from tensorfold.cuda.kernels.attention import slots

    d, i, vocab = int(text["hidden_size"]), int(text["intermediate_size"]), int(text["vocab_size"])
    layers, heads = int(text["num_hidden_layers"]), int(text["num_attention_heads"])
    kv = int(text.get("num_key_value_heads") or heads)
    hd = int(text.get("head_dim") or d // heads)
    fixed = max(16 * rows * (d + i + vocab + heads * hd) * 4, chunk * prompt_row_bytes(text))

    def bytes_at(capacity: int) -> int:
        cache = (layers + 1) * max(1024, capacity) * kv * hd * 4
        partials = rows * heads * (hd + 2) * slots(capacity, rows) * 4
        return fixed + cache + partials

    return Geometry(bytes_at, rows)


def weight_bytes(name: str, info: dict) -> tuple[int, int]:
    """A bf16 tensor as loaded: its stored bytes, nothing staged."""

    from tensorfold.cuda.geometry import size

    return (0 if name.endswith("rotary_emb.inv_freq") else size(info, name)), 0


class LlamaEngine:
    """A dense Llama on one GPU: copies from the context and DFlash2 trees (``draft_dir``) drafted, verified in one
    window, bit-equal to serial decode."""

    tp = 1
    concurrent = False

    def __init__(self, model_dir: Path, *, draft_dir: Path | None = None, allow_copy: bool = True,
                 context: int | None = None, context_explicit: bool | None = None, max_rows: int = MAX_ROWS,
                 tree_rows: int | None = None):
        import torch

        from tensorfold.cuda.capacity import admit, config, total_bytes
        from tensorfold.cuda.geometry import draft_geometry, prompt_row_bytes, prompt_rows
        from tensorfold.cuda.markers import resume_points
        from tensorfold.cuda.nvfp4.format import is_quantized
        from tensorfold.cuda.streams import KVRoom, PrefixCache
        from tensorfold.families.qwen3_5.cuda.affine_memory import draft_weights
        from tensorfold.families.qwen3_5.cuda.exl3_load import admission as exl3_admission
        from tensorfold.families.qwen3_5.cuda.exl3_load import quant_config
        from tensorfold.families.qwen3_5.cuda.nvfp4_load import admission as nvfp4_admission

        from .decode import TREE_ROWS
        from .weights import load

        self.torch = torch
        self.allow_copy, self.max_rows = allow_copy, int(max_rows)
        self.tree_rows = min(self.max_rows, TREE_ROWS if tree_rows is None else int(tree_rows))
        torch.cuda.set_device(0)
        chunk = prompt_rows(total_bytes(torch), prompt_row_bytes(config(model_dir)))
        shape = lambda text: geometry(text, self.max_rows, chunk)
        transform = weight_bytes
        if quant_config(Path(model_dir)) is not None:
            shape, transform = exl3_admission(shape)
        elif is_quantized(Path(model_dir)):
            shape, transform = nvfp4_admission(shape)
        # the drafter's context and block pass, plus a snapshot for each kept prompt state
        self.capacity_plan = admit(model_dir, context, context_explicit, torch, shape, transform, draft_dir=draft_dir,
                                   draft_weights=draft_weights,
                                   draft_geometry=lambda text: draft_geometry(text, 1, self.max_rows, bounded=True,
                                                                              kept=KEEP_ONE))
        self.context_window = self.capacity_plan["context_window"]
        self.w = load(model_dir)
        self.w.prompt_rows = chunk
        self.draft = None if draft_dir is None else _drafter(Path(draft_dir), self.w)
        torch.cuda.empty_cache()
        self.eos = tuple(self.w.config.eos)
        self.model_dir = Path(model_dir)
        self.points = resume_points(model_dir)              # message starts a prefill keeps states at
        self.cache = PrefixCache(KEEP_ONE)                  # (ids, state, drafter snapshot): states share buffers
        plan = self.capacity_plan
        spare = plan["budget_bytes"] - plan["weight_bytes_estimate"] - plan["cache_workspace_bytes_estimate"]
        c = self.w.config
        live = (c.layers + 1) * max(1024, self.context_window) * c.kv_heads * c.head_dim * 4
        self.room = KVRoom(self.cache, spare + live)

    def _resume(self, prompt: list[int]):
        best = self.cache.longest(prompt)
        if best is not None:                # resumed writes overwrite the rows of longer entries on these buffers
            n = len(best[0])
            self.cache.entries = [e for e in self.cache.entries if len(e[0]) <= n or e[0][:n] != best[0]]
        return best

    def _keep_points(self, prompt: Sequence[int], base: int) -> list[int]:
        """Message starts past the resumed prefix, then the prompt's entry end unless a kept start sits just before."""

        from tensorfold.cuda.markers import MIN_GAP

        stops = [p for p in (self.points(prompt) if self.points is not None else []) if p >= base + MIN_GAP]
        end = entry_end(prompt)
        if end > base and not (stops and end - stops[-1] < MIN_GAP):
            stops.append(end)
        return sorted(set(stops))

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True):
        """``draft=False``: one-row rounds and no prefix reuse, the serial reference."""

        from .decode import draft_decode, prefill

        if len(prompt) >= self.context_window:
            raise ValueError(f"prompt of {len(prompt)} tokens exceeds the {self.context_window}-token safe capacity; "
                             "shorten the prompt or reserve fewer reply tokens")
        max_tokens = max(1, min(int(max_tokens), self.context_window - len(prompt)))
        t0 = time.perf_counter()
        hit = self._resume(prompt) if draft else None
        base = len(hit[0]) if hit else 0
        keep = self._keep_points(prompt, base) if draft else []
        drafter = self.draft if draft else None
        if drafter is not None:
            drafter.restore(hit[2] if hit else ([None] * drafter.layers, [None] * drafter.layers, 0, 0))
        st, pending, kept = prefill(self.w, prompt, sampling, state=hit[1] if hit else None, limit=self.context_window,
                                    room=self.room, keep=keep, draft=drafter)
        for p, item in zip(keep, kept):
            state, snap = item if drafter is not None else (item, None)
            self.cache.add(list(prompt[:p]), state, snap)
        prefill_s = time.perf_counter() - t0
        cached = hit[1].pos if hit else 0
        if on_tokens([pending]):
            return {"prefill_s": prefill_s, "cached": cached}
        result = draft_decode(self.w, st, prompt, pending, max_tokens, sampling,
                              max_rows=self.max_rows if draft else 1, allow_copy=self.allow_copy and draft,
                              stop_eos=stop_eos, on_tokens=on_tokens, inplace=True, draft=drafter,
                              tree_rows=self.tree_rows)
        return {"prefill_s": prefill_s, "decode_s": result.seconds, "rounds": result.rounds, "cached": cached,
                "drafts": draft, "min_rows": min(result.widths, default=0), "drafted": result.drafted_rows,
                "accepted": result.accepted_drafts}


def _drafter(draft_dir: Path, w):
    """DFlash2 over the target's whole vocabulary, refused unless it reads this target's width and layers."""

    from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2

    c = w.config
    draft = DFlash2(draft_dir, w, vocab_spans=((0, c.vocab),))
    if draft.hidden != c.hidden or not all(0 <= i < c.layers for i in draft.tap_layers):
        raise ValueError(f"the drafter reads {draft.hidden}-wide taps at layers {list(draft.tap_layers)}; this target "
                         f"is {c.hidden} wide with {c.layers} layers")
    return draft
