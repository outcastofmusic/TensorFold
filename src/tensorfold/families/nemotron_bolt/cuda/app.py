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


def chunk_bytes(c, prefill_rows: int) -> int:
    """A prompt chunk's largest activations, in bytes: a MoE layer's widest expert rows (top_k routed rows a token,
    or the shared expert's one) twice over, Mamba's in_proj rows and a few hidden-wide rows."""

    widest = max([k * w for k, w in zip(c.top_k, c.moe_width)] + [c.shared_width])
    return prefill_rows * (widest * 2 * 2 + c.proj_dim * 2 + 8 * c.hidden * 2)


def cache_bytes(c, max_len: int, rows: int, prefill_rows: int) -> int:
    """The engine's KV caches, Mamba states and largest per-chunk activations, in bytes."""

    kv = sum(2 * max_len * kv * c.head_dim * 2 for kv in c.kv_heads)
    nm = c.pattern.count("M")
    mamba = nm * (c.m_heads * c.m_head_dim * c.m_state * 4 + 3 * c.conv_dim * 2 + 2 * rows * (2 * c.conv_dim * 2
                                                                                                 + c.m_heads * 4))
    return kv + mamba + chunk_bytes(c, prefill_rows) + rows * c.vocab * 2


def need_bytes(model_dir: str | Path, c, max_len: int, *, drafter: str | Path = "", layers: int | None = None,
               graphs: bool = False, draft_q8: bool = False) -> int:
    """What Bolt holds on the GPU with a ``max_len``-token cache: its weights, caches and chunk activations, the
    workspace, with a ``drafter`` folder the drafter's weights and context caches (and with ``draft_q8`` its packed
    Q8_0 head), and with ``graphs`` the window graphs (their pool, the drafter's tap rows they keep, their nodes)."""

    from .engine import PREFILL_ROWS, ROWS
    from .weights import weight_bytes

    need = weight_bytes(model_dir, layers) + cache_bytes(c, max_len, ROWS, PREFILL_ROWS) + WORKSPACE
    if drafter:
        from .lilicorr import device_bytes

        need += device_bytes(drafter)
        if draft_q8 and head_is_q8(model_dir):
            from .q8_head import head_bytes

            need += head_bytes(c)
    if graphs:
        from .graphs import graph_bytes

        need += graph_bytes(c, max_len, ROWS, len(draft_taps(drafter)) if drafter else 0)
    return need


def head_is_q8(model_dir: str | Path) -> bool:
    """Whether the checkpoint stores its lm_head as Q8_0 (Bolt's 24 GB export does)."""

    from .weights import quant_layers

    return any(name.rsplit(".", 1)[-1] == "lm_head" and kind == "Q8_0"
               for name, kind in quant_layers(Path(model_dir)).items())


class BoltEngine:
    """``eos``, ``generate``, ``context_window`` and ``shutdown``, as the CUDA server expects (one GPU)."""

    def __init__(self, model_dir: Path, *, drafter: str | Path = "", drafts: int = DRAFTS, context: int | None = None,
                 context_explicit: bool = False, layers: int | None = None) -> None:
        """``drafter``: a LiLiCorr draft folder ("" decodes serially); ``layers``: only the first layers (tests)."""
        import torch

        from tensorfold.cuda.capacity import available_bytes
        from tensorfold.env_flags import flag

        from .engine import PREFILL_ROWS, ROWS, A, Engine
        from .weights import Config, load

        torch.cuda.set_device(0)
        self.model_dir = Path(model_dir)
        c = Config.read(self.model_dir)
        taps = draft_taps(drafter) if drafter else ()
        self.drafts = int(drafts)
        drafting = bool(drafter) and self.drafts > 0
        graphs = flag("TF_BOLT_GRAPHS", default=True)     # 0 runs every verify window eagerly
        # the drafter's candidates from the checkpoint's Q8_0 head (0.54 GiB a block pass reads, not 1 GiB); 0 shares
        # the verify's bf16 head. Drafts and acceptance only: a reply token is always the verify's sample
        draft_q8 = drafting and flag("TF_BOLT_DRAFT_Q8", default=True)
        # the CLI passes the model's own window (or 0, which it reads from no llm_config) when --context is absent
        want = int(context) if context_explicit and context else CONTEXT
        self.max_len = -(-(want + ROWS) // A.CHUNK) * A.CHUNK
        need = need_bytes(self.model_dir, c, self.max_len, drafter=drafter if drafting else "", layers=layers,
                          graphs=graphs, draft_q8=draft_q8)
        free = available_bytes(torch)                 # a GB10's page cache counts as free; the pool's floor is kept
        gib = 1 << 30
        if need > free:
            raise ValueError(f"Nemotron Bolt needs {need / gib:.1f} GiB on the GPU for its weights"
                             f"{' and drafter' if drafting else ''} and a {self.max_len}-token context; "
                             f"{free / gib:.1f} GiB is free above the memory floor. Free memory or pass a smaller "
                             f"--context")
        from tensorfold.cuda.kept_states import Store
        from tensorfold.cuda.markers import resume_points

        self.marks = resume_points(self.model_dir)          # message starts a prefill keeps states at, or None
        started = time.perf_counter()
        self.w = load(self.model_dir, layers=layers, q8_head=draft_q8)
        self.e = Engine(self.w, max_len=self.max_len, taps=taps, graphs=graphs)
        self.drafter = None
        # TF_BOLT_DEPTH=fixed drafts ``drafts`` a round; otherwise a round's depth follows acceptance and measured costs
        self.adaptive = os.environ.get("TF_BOLT_DEPTH", "auto") != "fixed"
        if drafting:
            from functools import partial

            from .depth import Costs
            from .lilicorr import Drafter
            from .q8_head import logits

            # the drafter's head: the packed Q8_0 one where the checkpoint has it and TF_BOLT_DRAFT_Q8 is on
            self.draft_head = self.e.head_logits if self.w.head_q8 is None else partial(logits, packed=self.w.head_q8)
            self.drafter = Drafter(drafter, embed=self.e.embed_rows, lm_head=self.draft_head)
            self.drafts = min(self.drafts, self.drafter.c.block - 1)
            self.costs = Costs(self.drafts + 1)
        # greedy windows of every size a round can verify are captured now; other sampling modes on first use
        warmed = self.e.warm(range(1, self.drafts + 2 if self.drafter is not None else 2))
        # kept prompt states get what the loaded weights, drafter and caches leave beside a prompt chunk's
        # activations, the workspace and the window graphs other sampling modes capture later, at most
        # TF_BOLT_CACHE_GIB
        torch.cuda.empty_cache()
        room = available_bytes(torch) - chunk_bytes(c, PREFILL_ROWS) - WORKSPACE
        if graphs:
            from .graphs import graph_bytes

            room -= graph_bytes(c, self.max_len, ROWS, len(taps))
        budget = max(0, min(int(float(os.environ.get("TF_BOLT_CACHE_GIB", "4")) * gib), room))
        self.store = Store(int(os.environ.get("TF_BOLT_CACHE_ENTRIES", "8")), budget)
        self.eos = tuple(c.eos)
        self.tp, self.rank = 1, 0
        rule = (f"LiLiCorr drafts ({drafter}), {'up to ' if self.adaptive else ''}{self.drafts} a round, "
                f"{'a Q8_0' if self.w.head_q8 is not None else 'the bf16'} draft head"
                if self.drafter is not None else "serial decoding (no drafter)")
        kept = (f"up to {self.store.entries} prompt states kept in {self.store.budget / gib:.1f} GiB"
                if self.store.on else "no prompt states kept")
        shown = f"{warmed} window graphs captured" if graphs else "eager windows (TF_BOLT_GRAPHS=0)"
        print(f"[tensorfold] Nemotron Bolt on CUDA: {len(self.w.blocks)} layers, IQ2_XXS experts; {rule}; "
              f"{self.max_len}-token cache; {kept}; {shown}; loaded in {time.perf_counter() - started:.1f}s",
              flush=True)

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

        from tensorfold.cuda.kept_states import keep_points

        from .engine import prefill, serial_decode

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

        # a kept state: the Mamba ssm and conv states and the attention rows (Engine.state), with the LiLiCorr
        # drafter's context (Drafter.snapshot: its ring of tap rows, anchor and context_end) when the prefill fed one
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
