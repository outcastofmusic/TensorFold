"""The Bolt CUDA engine behind the server: ``eos``, ``generate`` and ``shutdown`` on one GPU; LiLiCorr drafts when given."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import CONTEXT, DRAFTS, WORKSPACE


def draft_taps(drafter_dir: str | Path) -> tuple[int, ...]:
    """The target layers a LiLiCorr / DFlash draft reads, from its config.json (``dflash_config.target_layer_ids``)."""

    config = json.loads((Path(drafter_dir) / "config.json").read_text())
    ids = (config.get("dflash_config") or {}).get("target_layer_ids") or config.get("target_layer_ids")
    if not ids:
        raise ValueError(f"{drafter_dir} names no target layers (dflash_config.target_layer_ids): not a LiLiCorr draft")
    return tuple(int(i) for i in ids)


def cache_bytes(c, max_len: int, rows: int, prefill_rows: int) -> int:
    """The engine's KV caches, Mamba states and largest per-chunk activations, in bytes."""

    kv = sum(2 * max_len * kv * c.head_dim * 2 for kv in c.kv_heads)
    nm = c.pattern.count("M")
    mamba = nm * (c.m_heads * c.m_head_dim * c.m_state * 4 + 3 * c.conv_dim * 2 + 2 * rows * (2 * c.conv_dim * 2
                                                                                                 + c.m_heads * 4))
    widest = max([*c.moe_width, c.shared_width, c.proj_dim])
    chunk = prefill_rows * (max(c.top_k, default=1) * widest * 2 * 2 + c.proj_dim * 2 + 8 * c.hidden * 2)
    return kv + mamba + chunk + rows * c.vocab * 2


class BoltEngine:
    """``eos``, ``generate``, ``context_window`` and ``shutdown``, as the CUDA server expects (one GPU)."""

    def __init__(self, model_dir: Path, *, drafter: str | Path = "", drafts: int = DRAFTS, context: int | None = None,
                 context_explicit: bool = False, layers: int | None = None) -> None:
        """``drafter``: a LiLiCorr draft folder ("" decodes serially); ``layers``: only the first layers (tests)."""
        import torch

        from tensorfold.cuda.capacity import available_bytes

        from .engine import PREFILL_ROWS, ROWS, A, Engine
        from .weights import Config, load, weight_bytes

        torch.cuda.set_device(0)
        self.model_dir = Path(model_dir)
        c = Config.read(self.model_dir)
        taps = draft_taps(drafter) if drafter else ()
        # the CLI passes the model's own window (or 0, which it reads from no llm_config) when --context is absent
        want = int(context) if context_explicit and context else CONTEXT
        self.max_len = -(-(want + ROWS) // A.CHUNK) * A.CHUNK
        need = weight_bytes(self.model_dir, layers) + cache_bytes(c, self.max_len, ROWS, PREFILL_ROWS) + WORKSPACE
        free = available_bytes(torch)                 # a GB10's page cache counts as free; the pool's floor is kept
        gib = 1 << 30
        if need > free:
            raise ValueError(f"Nemotron Bolt needs {need / gib:.1f} GiB on the GPU for its weights and a "
                             f"{self.max_len}-token context; {free / gib:.1f} GiB is free above the memory floor. "
                             f"Free memory or pass a smaller --context")
        from tensorfold.cuda.markers import resume_points

        from .reuse import Store

        # kept prompt states get what the weights and caches leave, at most TF_BOLT_CACHE_GIB
        budget = min(int(float(os.environ.get("TF_BOLT_CACHE_GIB", "4")) * gib), free - need)
        self.store = Store(int(os.environ.get("TF_BOLT_CACHE_ENTRIES", "8")), budget)
        self.marks = resume_points(self.model_dir)          # message starts a prefill keeps states at, or None
        started = time.perf_counter()
        self.w = load(self.model_dir, layers=layers)
        self.e = Engine(self.w, max_len=self.max_len, taps=taps)
        self.drafter = None
        self.drafts = int(drafts)
        # TF_BOLT_DEPTH=fixed drafts ``drafts`` a round; otherwise a round's depth follows acceptance and measured costs
        self.adaptive = os.environ.get("TF_BOLT_DEPTH", "auto") != "fixed"
        if drafter and self.drafts > 0:
            from .depth import Costs
            from .lilicorr import Drafter

            self.drafter = Drafter(drafter, embed=self.e.embed_rows, lm_head=self.e.head_logits)
            self.drafts = min(self.drafts, self.drafter.c.block - 1)
            self.costs = Costs(self.drafts + 1)
        self.eos = tuple(c.eos)
        self.tp, self.rank = 1, 0
        from .engine import EXPERTS

        rule = (f"LiLiCorr drafts ({drafter}), {'up to ' if self.adaptive else ''}{self.drafts} a round"
                if self.drafter is not None else "serial decoding (no drafter)")
        kept = (f"up to {self.store.entries} prompt states kept in {self.store.budget / gib:.1f} GiB"
                if self.store.on else "no prompt states kept")
        print(f"[tensorfold] Nemotron Bolt on CUDA: {len(self.w.blocks)} layers, experts on the {EXPERTS}; {rule}; "
              f"{self.max_len}-token cache; {kept}; loaded in {time.perf_counter() - started:.1f}s", flush=True)

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity: the cache rows less one verify window."""

        return max(0, self.max_len - self.e.max_rows)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.max_len - len(prompt) - self.e.max_rows
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.max_len}-token context")
        return max(1, min(int(max_tokens), room))

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, constraint=None, stop_eos: bool = True) -> dict[str, Any]:
        """Prefill, then LiLiCorr rounds (``draft`` with a drafter and no grammar) or serial ones; same tokens.

        A ``draft`` request resumes from the longest kept prefix of its prompt and keeps states as its prefill passes
        ``keep_points``; ``draft: false``, the reference, prefills afresh and keeps nothing.
        """

        from .engine import prefill, serial_decode
        from .reuse import keep_points

        prompt = [int(t) for t in prompt]
        max_tokens = self._limit(prompt, max_tokens)
        drafting = bool(draft) and self.drafter is not None and constraint is None
        reuse = bool(draft) and self.store.on
        hit = self.store.longest(prompt, drafting) if reuse else None
        if drafting and hit is not None:
            self.drafter.restore(hit.drafter)
        elif drafting:
            self.drafter.reset()
        begin = len(hit.ids) if hit is not None else 0

        def keep(n: int, state: dict) -> None:
            self.store.add(prompt[:n], state, self.drafter.snapshot() if drafting else None)

        t0 = time.perf_counter()
        pending = prefill(self.e, prompt, sampling, on_taps=self.drafter.add_taps if drafting else None,
                          constraint=constraint, resume=hit.state if hit is not None else None,
                          keeps=keep_points(prompt, begin, self.marks) if reuse else (), keep=keep)
        prefill_s = time.perf_counter() - t0
        stats: dict[str, Any] = {"prefill_s": round(prefill_s, 4), "cached": begin, "drafts": drafting,
                                 "prefill_tps": round((len(prompt) - begin) / prefill_s, 2) if prefill_s else 0.0}
        if (on_tokens is not None and on_tokens([pending])) or (stop_eos and pending in self.eos) or max_tokens <= 1:
            return stats
        if drafting:
            from .depth import DepthPolicy
            from .lilicorr import lilicorr_decode

            policy = DepthPolicy(self.drafts, self.costs) if self.adaptive else None
            res = lilicorr_decode(self.e, self.drafter, pending, max_tokens, sampling, stop_eos=stop_eos,
                                  on_tokens=on_tokens, depth=self.drafts, policy=policy)
            stats.update(drafted=res.drafted, accepted=res.accepted, serial_rounds=res.widths.count(1))
        else:
            res = serial_decode(self.e, pending, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                                constraint=constraint)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def follow(self) -> None:
        raise RuntimeError("Nemotron Bolt runs on one GPU: there is no rank 1 to follow")

    def shutdown(self) -> None:
        """One GPU: nothing to tell another rank."""
