"""The LFM2 decision engine: a state and its questions in, typed answers out, one forward a request.

A request reads its state once (the trunk) and each question as a branch off it, packed on one axis, so several
questions cost about one forward. Requests are serialized: a decision takes tens of milliseconds, and a request read
alone gives the same bits whether others wait or not. A request whose questions do not fit one forward's token budget
continues in the next forward over the same trunk.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tensorfold.server.decisions import DecisionError

from . import TOKEN_BUDGET, read_text_config
from .decisions import Format, Question, answer, question
from .model import Config, Model, Tree


def geometry(cfg: Config):
    """The startup admission's activation geometry: bytes a forward holds a packed token."""

    from tensorfold.cuda.capacity import Geometry

    per_token = cfg.token_bytes()
    return Geometry(bytes_at=lambda slots: slots * per_token, reserve=0, minimum_slots=1)


def weight_bytes(name: str, info: dict) -> tuple[int, int]:
    """Every checkpoint tensor is read as stored: the bf16 language model, the vision tower and its projector."""

    from tensorfold.cuda.geometry import size

    return size(info, name), 0


# How `_scores` reads the option logits, chosen at startup with TF_D1_HEAD while the choice is measured:
# union        one head matmul of every question's row over the union of their option tokens, one read-back;
# per_question one matmul a question at its own shape (the bits per_question_sync gives), one read-back;
# per_question_sync  one matmul and one read-back a question.
HEADS = ("union", "per_question", "per_question_sync")


def head_choice() -> str:
    head = os.environ.get("TF_D1_HEAD", "").strip() or "union"
    if head not in HEADS:
        raise ValueError(f"TF_D1_HEAD={head!r}: one of {', '.join(HEADS)}")
    return head


class Engine:
    eos: tuple[int, ...] = ()
    supports_logprobs = False

    def __init__(self, model_dir: str | Path, *, context: int | None = None, context_explicit: bool | None = None,
                 token_budget: int = TOKEN_BUDGET, device: str = "cuda", pictures: bool = True,
                 processor: Any = None):
        from tokenizers import Tokenizer

        self.head = head_choice()
        self.model_dir = Path(model_dir)
        raw, text = read_text_config(self.model_dir)
        self.cfg = Config.of(text)
        native = int(text.get("max_position_embeddings") or 0)
        if device == "cuda":
            from tensorfold.cuda.capacity import admit

            torch.cuda.set_device(0)
            # weights, the vision tower and a forward's activations, admitted before any weight loads
            self.capacity_plan = admit(self.model_dir, context, context_explicit, torch, lambda t: geometry(Config.of(t)),
                                       weight_bytes)
            self.context = int(self.capacity_plan["context_window"])
        else:
            self.capacity_plan = None
            self.context = min(int(context or native), native) if native else int(context or 0)
        if self.context < 1:
            raise ValueError("LFM2 d1 needs a positive --context")
        self.context_window = self.context
        self.token_budget = max(1, min(int(token_budget), self.context))
        started = time.perf_counter()
        self.model = Model.load(self.model_dir, self.cfg, device)
        self.pictures = None
        if pictures and isinstance(raw.get("vision_config"), dict):
            from .vision import Vision

            self.pictures = Vision.load(self.model_dir, device, processor=processor)
        self.fmt = Format(Tokenizer.from_file(str(self.model_dir / "tokenizer.json")))
        self.load_s = time.perf_counter() - started
        self._lock = threading.Lock()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- the API

    def decide(self, state: Any, questions: Any, images: list | None = None) -> dict[str, Any]:
        """A System One request's answers; the state, its pictures and the questions are all it reads."""

        if not isinstance(questions, dict) or not questions:
            raise DecisionError("questions is a non-empty object of id: question")
        named = [(str(name), question(raw)) for name, raw in questions.items()]
        if images and self.pictures is None:
            raise DecisionError("this checkpoint has no vision tower, so it reads no pictures")
        with self._lock:
            answers, read = self._answer(state, named, list(images or []))
        return {"answers": answers, "usage": {"input_tokens": read, "output_tokens": 0}}

    def generate(self, *_, **__):
        raise DecisionError("d1 is a decision model and writes no text: use POST /v1/systemone or /v1/decisions")

    def score_labels(self, prompt_ids, label_ids) -> tuple[list[float], float]:
        """``/v1/decisions``: the label logits at a prompt's last token and the vocabulary's log-sum-exp."""

        return self.score_labels_many([(prompt_ids, label_ids)])[0]

    def score_labels_many(self, items: list[tuple[list[int], list[int]]]) -> list[tuple[list[float], float]]:
        """Several prompts, their shared head read once as the trunk and each rest as a branch."""

        rows = [[int(t) for t in ids] for ids, _ in items]
        if any(not row for row in rows):
            raise DecisionError("a decision prompt cannot be empty")
        shared = 0                                     # every row keeps at least its last token as its branch
        while shared < min(map(len, rows)) - 1 and len({row[shared] for row in rows}) == 1:
            shared += 1
        with self._lock:
            hidden = self._rows(rows[0][:shared], [row[shared:] for row in rows])
            out = []
            for row, (_, labels) in zip(hidden, items):
                logits = self.model.logits(row[None])[0]
                out.append(([float(logits[int(t)]) for t in labels], float(logits.logsumexp(0))))
        return out

    # ---------------------------------------------------------------- forwards

    def _answer(self, state: Any, named: list[tuple[str, Question]], images: list) -> tuple[dict, int]:
        f = self.fmt
        embeds = None
        if images:
            from .vision import cap

            pictures = [cap(image) for image in images]
            trunk, embeds = self.pictures.trunk(self.model, f.prefix(state, self.pictures.markup(len(pictures))),
                                                pictures)
        else:
            trunk = f.encode(f.prefix(state))
        branches = [f.encode(f.suffix(q)) for _, q in named]
        groups = [f.readout(q) for _, q in named]
        hidden = self._rows(trunk, branches, embeds)
        scores = self._scores(hidden, groups)
        answers = {name: answer(q, row) for (name, q), row in zip(named, scores)}
        return answers, len(trunk) + sum(map(len, branches))

    def _scores(self, hidden: torch.Tensor, groups: list[list[list[int]]]) -> list[list[float]]:
        """Each question's options' best logits among their forms, read from the head as `self.head` says."""

        if self.head == "union":
            ids = sorted({t for group in groups for forms in group for t in forms})
            logits = self.model.logits(hidden, ids).tolist()
            where = {t: i for i, t in enumerate(ids)}
            return [[max(row[where[t]] for t in forms) for forms in group] for row, group in zip(logits, groups)]
        own = [sorted({t for forms in group for t in forms}) for group in groups]
        if self.head == "per_question_sync":
            logits = [self.model.logits(row[None], ids)[0].tolist() for row, ids in zip(hidden, own)]
        else:
            # Model.logits uploads its ids, a blocking copy, so the head rows are gathered with one upload. Each
            # question then multiplies its own contiguous slice of them at the shape per_question_sync uses.
            heads = self.model.embed[torch.tensor([t for ids in own for t in ids], device=hidden.device)]
            parts, start = [], 0
            for row, ids in zip(hidden, own):
                parts.append(F.linear(row[None], heads[start:start + len(ids)]).float()[0])
                start += len(ids)
            flat = torch.cat(parts).tolist()
            logits, start = [], 0
            for ids in own:
                logits.append(flat[start:start + len(ids)])
                start += len(ids)
        return [[max(row[ids.index(t)] for t in forms) for forms in group]
                for row, ids, group in zip(logits, own, groups)]

    def _rows(self, trunk: list[int], branches: list[list[int]], embeds: torch.Tensor | None = None) -> torch.Tensor:
        """The final hidden state at each branch's last token, as many branches a forward as the budget holds."""

        longest = len(trunk) + max(map(len, branches))
        if longest > self.context:
            raise DecisionError(f"a question with its state is {longest} tokens, past this server's {self.context}-token "
                                "context")
        if longest > self.token_budget:
            raise DecisionError(f"a question with its state is {longest} tokens, past the {self.token_budget}-token "
                                "forward budget")
        model = self.model
        base = embeds if embeds is not None else model.embed_ids(trunk) if trunk else model.embed[:0]
        out, start = [], 0
        while start < len(branches):
            end, used = start + 1, len(trunk) + len(branches[start])
            while end < len(branches) and used + len(branches[end]) <= self.token_budget:
                used += len(branches[end])
                end += 1
            part = branches[start:end]
            x = torch.cat((base, model.embed_ids([t for row in part for t in row])))
            out.append(model.forward(x, Tree(len(trunk), [len(row) for row in part], model.device, model.cfg.taps)))
            start = end
        return torch.cat(out)
