"""The LFM2-MoE CUDA engine behind the server (``eos``, ``generate``, ``context_window``) and its thinking app."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tensorfold.cuda.server import App

from . import CONTEXT, PREFILL_ROWS, WORKSPACE


def cache_bytes(c, max_len: int, prefill_rows: int = PREFILL_ROWS) -> int:
    """KV caches, conv states and a prompt chunk's widest activations, in bytes."""

    kv = len(c.attention_layers) * 2 * max_len * c.kv_heads * c.head_dim * 2
    conv = len(c.conv_layers) * (c.conv_taps - 1) * c.hidden * 2
    chunk = prefill_rows * (max(2 * c.intermediate, 2 * c.top_k * c.moe_width, 3 * c.hidden) * 4 + 8 * c.hidden * 2)
    return kv + conv + chunk + c.vocab * 4


class Lfm2MoeEngine:
    """One GPU, one stream: requests take turns (the server serializes them)."""

    def __init__(self, model_dir: Path, *, context: int | None = None, context_explicit: bool = False,
                 layers: int | None = None) -> None:
        import torch

        from tensorfold.cuda.capacity import available_bytes

        from ..config import Config
        from .engine import CHUNK, Engine
        from .weights import load, weight_bytes

        torch.cuda.set_device(0)
        self.model_dir = Path(model_dir)
        c = Config.read(self.model_dir)
        want = int(context) if context_explicit and context else min(CONTEXT, c.context or CONTEXT)
        if c.context and want > c.context:
            raise ValueError(f"--context {want} exceeds LFM2-MoE's {c.context}-token window")
        self.max_len = -(-want // CHUNK) * CHUNK
        need = weight_bytes(self.model_dir) + cache_bytes(c, self.max_len) + WORKSPACE
        free = available_bytes(torch)
        gib = 1 << 30
        if need > free:
            raise ValueError(f"LFM2-MoE needs {need / gib:.1f} GiB on the GPU for its weights and a {self.max_len}-"
                             f"token context; {free / gib:.1f} GiB is free above the memory floor. Free memory or "
                             f"pass a smaller --context")
        from tensorfold.cuda.markers import resume_points
        from tensorfold.families.nemotron_bolt.cuda.reuse import Store

        self.marks = resume_points(self.model_dir)          # message starts a prefill keeps states at, or None
        started = time.perf_counter()
        self.w = load(self.model_dir, layers=layers)
        self.e = Engine(self.w, max_len=self.max_len)
        # kept prompt states get what the weights and caches leave beside a prompt chunk's activations and the
        # workspace, at most TF_LFM2_CACHE_GIB
        torch.cuda.empty_cache()
        room = available_bytes(torch) - cache_bytes(c, 0) - WORKSPACE
        budget = max(0, min(int(float(os.environ.get("TF_LFM2_CACHE_GIB", "4")) * gib), room))
        self.store = Store(int(os.environ.get("TF_LFM2_CACHE_ENTRIES", "8")), budget)
        self.eos = tuple(c.eos)
        self.tp, self.rank = 1, 0
        kind = "block FP8" if c.fp8 else "bf16"
        kept = (f"up to {self.store.entries} prompt states kept in {self.store.budget / gib:.1f} GiB"
                if self.store.on else "no prompt states kept")
        print(f"[tensorfold] LFM2-MoE on CUDA: {len(self.w.blocks)} layers, {kind} weights, one stream; "
              f"{self.max_len}-token cache; {kept}; loaded in {time.perf_counter() - started:.1f}s", flush=True)

    @property
    def context_window(self) -> int:
        return self.max_len

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, constraint=None, stop_eos: bool = True) -> dict[str, Any]:
        """Prefill, then one-row decode steps until an end token, ``max_tokens`` or ``on_tokens``' True.

        A request resumes from the longest kept prefix of its prompt and keeps states as its prefill passes
        ``keep_points`` (message starts and the prompt end); ``draft: false``, the reference, prefills afresh and
        keeps nothing. One stream: requests take turns, so the store needs no lock.
        """

        from tensorfold.families.nemotron_bolt.cuda.reuse import keep_points

        from .engine import prefill, serial_decode

        prompt = [int(t) for t in prompt]
        room = self.max_len - len(prompt)
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        max_tokens = max(1, min(int(max_tokens), room))
        reuse = bool(draft) and self.store.on
        hit = self.store.longest(prompt, False) if reuse else None
        begin = len(hit.ids) if hit is not None else 0

        def keep(n: int, state: dict) -> None:
            self.store.add(prompt[:n], state, None)

        t0 = time.perf_counter()
        pending = prefill(self.e, prompt, sampling, constraint=constraint, resume=hit.state if hit is not None else None,
                          keeps=keep_points(prompt, begin, self.marks) if reuse else (), keep=keep)
        prefill_s = time.perf_counter() - t0
        stats: dict[str, Any] = {"prefill_s": round(prefill_s, 4), "cached": begin, "drafts": False,
                                 "prefill_tps": round((len(prompt) - begin) / prefill_s, 2) if prefill_s else 0.0}
        if (on_tokens is not None and on_tokens([pending])) or (stop_eos and pending in self.eos) or max_tokens <= 1:
            return stats
        res = serial_decode(self.e, pending, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                            constraint=constraint)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def follow(self) -> None:
        raise RuntimeError("LFM2-MoE runs on one GPU: there is no rank 1 to follow")

    def shutdown(self) -> None:
        """One GPU: nothing to tell another rank."""


def always_thinking(body: dict[str, Any]) -> dict[str, Any]:
    """The request with thinking on: LFM2.5's template opens ``<think>`` on every reply and reads no switch, so a
    request's ``enable_thinking: false`` or ``reasoning_effort: "none"`` would put the reasoning into ``content``."""

    out = dict(body)
    if out.get("reasoning_effort") == "none":
        out.pop("reasoning_effort")
    kwargs = out.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        kwargs = {k: v for k, v in kwargs.items() if k not in ("enable_thinking", "thinking")}
        if kwargs.get("reasoning_effort") == "none":
            kwargs.pop("reasoning_effort")
        out["chat_template_kwargs"] = kwargs
    return out


class Lfm2MoeApp(App):
    """The CUDA server's app with thinking always on (``always_thinking``)."""

    def __init__(self, engine, model_dir: Path, served: str, **options: Any):
        options["default_thinking"] = True
        super().__init__(engine, model_dir, served, **options)

    def _prepare(self, body: dict[str, Any], chat: bool):
        return super()._prepare(always_thinking(body), chat)
