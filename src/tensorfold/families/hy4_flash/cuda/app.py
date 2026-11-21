"""The Hy4-flash CUDA engine behind the server (``eos``, ``generate``, ``context_window``) and its app.

Hy4's tokenizer spells its control tokens with a ``:opensource`` suffix (``<think:opensource>``,
``<tool_call:opensource>``, ``<arg_key:opensource>``) and its template takes ``reasoning_effort`` ``high`` or
``no_think`` only. The app serves a normalized view of the checkpoint's text files: the same ids spelt as the
server's parsers read them (``<think>``, ``</think>``, GLM's ``<tool_call>NAME<arg_key>..``) and the template
rendering those spellings. Its template wrapper maps the server's thinking switch onto the template's two efforts, so
a request's ``enable_thinking: false`` renders ``no_think`` and any effort with thinking on renders ``high``. The two
ids that wrap a reply's calls (``<tool_calls>``, ``</tool_calls>``) never reach the server: the model reads them, the
reply's text holds GLM's form.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tensorfold.cuda.server import App

from .. import CONTEXT

SUFFIX = ":opensource"
WRAPPERS = ("<tool_calls>", "</tool_calls>")      # a reply's call block opener and closer, dropped from the reply
WORKSPACE = 6 << 30                                # a prompt chunk's activations, graphs and allocator slack


def _view_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensorfold" / "hy4-text"


def text_view(model_dir: Path) -> Path:
    """A folder with the checkpoint's text files, control tokens spelt without ``:opensource`` (same ids)."""

    model_dir = Path(model_dir)
    tok_raw = (model_dir / "tokenizer.json").read_bytes()
    tmpl_raw = (model_dir / "chat_template.jinja").read_bytes()
    cfg_raw = (model_dir / "tokenizer_config.json").read_bytes()
    key = hashlib.sha256(tok_raw + tmpl_raw + cfg_raw).hexdigest()[:16]
    view = _view_root() / key
    if (view / "done").exists():
        return view
    view.mkdir(parents=True, exist_ok=True)
    tok = json.loads(tok_raw)
    for added in tok.get("added_tokens", []):
        added["content"] = added["content"].replace(SUFFIX, "")
    vocab = tok.get("model", {}).get("vocab")
    if isinstance(vocab, dict):
        for name in [n for n in vocab if SUFFIX in n]:
            vocab[name.replace(SUFFIX, "")] = vocab.pop(name)
    (view / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
    template = tmpl_raw.decode()
    if f"set HYTK = '{SUFFIX}'" not in template:
        raise ValueError("Hy4-flash's chat template no longer sets HYTK: its control-token spelling is unknown")
    (view / "chat_template.jinja").write_text(template.replace(f"set HYTK = '{SUFFIX}'", "set HYTK = ''"))
    cfg = json.loads(cfg_raw)
    cfg.pop("chat_template", None)

    def strip(v):
        if isinstance(v, str):
            return v.replace(SUFFIX, "")
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items()}
        return v

    (view / "tokenizer_config.json").write_text(json.dumps(strip(cfg), ensure_ascii=False))
    for name in ("config.json", "generation_config.json"):
        target = view / name
        if (model_dir / name).exists() and not target.exists():
            target.symlink_to((model_dir / name).resolve())
    (view / "done").write_text("")
    return view


class EffortTemplate:
    """The checkpoint's template with the server's thinking switch heard as Hy4's ``reasoning_effort``."""

    efforts = frozenset({"high"})          # every requested effort coerces to the one thinking level
    thinking_switch = True

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def render(self, messages, *, tools, enable_thinking: bool, extra: dict[str, Any] | None = None, **kw) -> str:
        extra = {**(extra or {}), "reasoning_effort": "high" if enable_thinking else "no_think"}
        return self.inner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra, **kw)


class Hy4App(App):
    """The CUDA server's app over the normalized text view (``text_view``) and the effort mapping."""

    def __init__(self, engine, model_dir: Path, served: str, **options: Any):
        super().__init__(engine, text_view(Path(model_dir)), served, **options)
        self.template = EffortTemplate(self.template)
        engine.hide = {self.tok.token_to_id(t) for t in WRAPPERS} - {None}


def cache_bytes(c, max_len: int, prefill_rows: int) -> int:
    """Latent rows of every layer, the sparse layers' token and group keys, and a prompt chunk's widest arrays."""

    per_token = c.layers * 576 * 2 + len(c.sparse_layers) * (c.index_dim * 2 + c.index_dim * 2 // c.index_pool)
    groups = max_len // c.index_pool
    chunk = prefill_rows * (c.top_k * (c.moe_width * 2 + c.latent * 4) + 64 * 4 * 512 * 4 + groups * 4 * 2
                            + c.hc * c.hidden * 8)
    return per_token * max_len + chunk + c.vocab * 4 * 2


def weight_bytes(model_dir: Path) -> int:
    """The text model's bytes as loaded: every stored tensor but the vision tower, the drafter and Engram tables (their
    row scales stay on the host, 660 MB a table), plus the lane matmul's fp32 scale per (64 inputs, column) of each
    block-FP8 projection."""

    from tensorfold.cuda.capacity import headers
    from tensorfold.cuda.geometry import size

    total = 0
    for name, info in headers(model_dir).items():
        if name.startswith(("vit.", "model.dfly_layers.")) or name.endswith("multi_head_embedding.embedding.weight"):
            continue
        total += size(info, name)
        if info["dtype"] == "F8_E4M3" and name.endswith(".weight") and len(info["shape"]) == 2:
            n, k = info["shape"]
            total += -(-n // 128) * 128 * (k // 64) * 4
    return total


class Hy4Engine:
    """One GPU, one stream: requests take turns (the server serializes them)."""

    def __init__(self, model_dir: Path, *, context: int | None = None, context_explicit: bool = False,
                 drafts: bool = True, layers: int | None = None) -> None:
        import torch

        from tensorfold.cuda.capacity import available_bytes
        from tensorfold.cuda.kept_states import Store

        from ..config import Config
        from .model import PREFILL_ROWS, Engine
        from .weights import load

        torch.cuda.set_device(0)
        self.model_dir = Path(model_dir)
        c = Config.read(self.model_dir)
        gib = 1 << 30
        want = int(context) if context_explicit and context else min(CONTEXT, c.context)
        if want > c.context:
            raise ValueError(f"--context {want} exceeds Hy4-flash's {c.context}-token window")
        weights = weight_bytes(self.model_dir)
        free = available_bytes(torch)
        room = free - weights - WORKSPACE
        fit = want
        while fit > 4096 and cache_bytes(c, fit, PREFILL_ROWS) > room:
            fit //= 2
        if fit < want and context_explicit:
            raise ValueError(f"Hy4-flash needs {(weights + cache_bytes(c, want, PREFILL_ROWS) + WORKSPACE) / gib:.1f}"
                             f" GiB for its weights and a {want}-token context; {free / gib:.1f} GiB is free above "
                             f"the memory floor. Pass --context {fit} or free memory")
        if cache_bytes(c, fit, PREFILL_ROWS) > room:
            raise ValueError(f"Hy4-flash needs {weights / gib:.1f} GiB for its weights; {free / gib:.1f} GiB is free "
                             "above the memory floor")
        self.max_len = fit
        self.drafts = drafts
        started = time.perf_counter()
        self.w = load(self.model_dir, layers=layers)
        self.e = Engine(self.w, max_len=self.max_len)
        self.drafter = None
        raw = json.loads((self.model_dir / "config.json").read_text())
        if drafts and raw.get("dfly_target_kv_layers") and os.environ.get("TF_HY4_DFLY", "1") != "0":
            from .dfly import DFly

            self.drafter = DFly(self.model_dir, self.e)
        torch.cuda.empty_cache()
        budget = max(0, min(int(float(os.environ.get("TF_HY4_CACHE_GIB", "4")) * gib),
                            available_bytes(torch) - WORKSPACE))
        self.store = Store(int(os.environ.get("TF_HY4_CACHE_ENTRIES", "8")), budget)
        from tensorfold.cuda.markers import resume_points

        self.marks = resume_points(text_view(self.model_dir))
        self.eos = tuple(c.eos)
        self.tp, self.rank = 1, 0
        self.hide: set[int] = set()
        kept = (f"up to {self.store.entries} prompt states kept in {self.store.budget / gib:.1f} GiB"
                if self.store.on else "no prompt states kept")
        kept += ", DFly2 drafts" if self.drafter is not None else ", no DFly2 drafts"
        print(f"[tensorfold] Hy4-flash on CUDA: {len(self.w.layers)} layers, {weights / gib:.1f} GiB of weights, "
              f"one stream; {self.max_len}-token cache; {kept}; loaded in {time.perf_counter() - started:.1f}s",
              flush=True)

    @property
    def context_window(self) -> int:
        return self.max_len

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], bool | None],
                 draft: bool = True, stop_eos: bool = True) -> dict[str, Any]:
        """Prefill, then one-row decode steps until an end token, ``max_tokens`` or ``on_tokens``' True.

        A request resumes from the longest kept prefix of its prompt and keeps states at its message starts and
        its end; ``draft: false``, the reference, prefills afresh and keeps nothing."""

        from tensorfold.cuda.kept_states import keep_points

        from .model import draft_decode, prefill, serial_decode

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

        def report(tokens: list[int]) -> bool | None:
            shown = [t for t in tokens if t not in self.hide]
            return on_tokens(shown) if shown or not tokens else None

        t0 = time.perf_counter()
        pending = prefill(self.e, prompt, sampling, resume=hit.state if hit is not None else None,
                          keeps=keep_points(prompt, begin, self.marks) if reuse else (), keep=keep)
        prefill_s = time.perf_counter() - t0
        drafting = bool(draft) and self.drafts
        stats: dict[str, Any] = {"prefill_s": round(prefill_s, 4), "cached": begin, "drafts": drafting,
                                 "prefill_tps": round((len(prompt) - begin) / prefill_s, 2) if prefill_s else 0.0}
        if report([pending]) or (stop_eos and pending in self.eos) or max_tokens <= 1:
            return stats
        if drafting:
            res = draft_decode(self.e, prompt, pending, max_tokens, sampling, stop_eos=stop_eos, on_tokens=report,
                               drafter=self.drafter, allow_copy=os.environ.get("TF_HY4_COPIES", "1") != "0")
            stats.update(drafted=res.stats["drafted"], accepted=res.stats["accepted"])
        else:
            res = serial_decode(self.e, pending, max_tokens, sampling, stop_eos=stop_eos, on_tokens=report)
        rounds = res.stats["rounds"] if drafting else res.rounds       # a drafted round commits one or more tokens
        stats.update(decode_s=round(res.seconds, 4), rounds=rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def follow(self) -> None:
        raise RuntimeError("Hy4-flash runs on one GPU: there is no rank 1 to follow")

    def shutdown(self) -> None:
        """One GPU: nothing to tell another rank."""
