"""Verify MTP chains against the same keyed samples as serial decoding, committing only rows before the first mismatched draft so drafts never change emitted tokens."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.kernels.gdn import MARKS
from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.sampling import comm_gather, nucleus_rows, sample_rows
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import CONFIDENCE, DEPTH
from .forward import Cut, commit, cut_snapshot, forward
from . import image_rows
from .state import CAND, Buffers, State
from .mtp import mtp_forward
from .weights import Weights


def sample_mapped(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  id_map: torch.Tensor) -> list[int]:
    """Rows of logits over a token subset (column j is token id_map[j]) -> tokens, with the keyed rule on the real ids (the draft head over the draft vocabulary)."""

    if sampling is None or sampling.temperature <= 0:
        return [int(t) for t in id_map[logits.argmax(dim=-1)].cpu().tolist()]
    k = min(logits.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else logits.shape[1]
    vals, idx = torch.topk(logits.float(), k, dim=-1, sorted=False)
    return choose_rows(vals.cpu().numpy(), id_map[idx].cpu().numpy().astype(np.int64), positions, sampling)


def tp_sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                   offset: int = 0, id_map: torch.Tensor | None = None, with_prob: bool = False):
    """Gather each rank's global-id candidates and apply the same keyed draw on every rank; ``with_prob`` also returns temperature-1 probabilities from gathered log-sum-exps."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    if not greedy and not sampling.top_k:           # top_k off: the shared nucleus rule over every rank's shard
        probs: list[float] | None = [] if with_prob else None
        chosen = nucleus_rows(logits, positions, sampling, offset=offset, id_map=id_map, gather=comm_gather(w.comm),
                              probs=probs)
        return (chosen, probs) if with_prob else chosen
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if greedy:
        # argmax takes the first (lowest-id) maximum whatever the row count; topk promises no order among ties
        ids = logits.argmax(dim=-1, keepdim=True)
        vals = torch.gather(logits, 1, ids).float()
    else:
        vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (id_map[ids] if id_map is not None else ids + int(offset)).to(torch.int32)
    parts = [vals, ids.view(torch.float32)]
    if with_prob:
        parts.append(torch.logsumexp(logits.float(), dim=-1, keepdim=True))
    packed = torch.cat(parts, dim=1).contiguous()
    width = packed.shape[1]
    world = int(w.meta["world"])
    got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=logits.device)
    w.comm.all_gather(packed.view(-1), got)
    g = got.view(world, R, width).cpu()
    values = torch.cat([g[r, :, :k] for r in range(world)], dim=1).numpy().astype(np.float32)
    tokens = torch.cat([g[r, :, k:2 * k].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
    tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * k].numpy().astype(np.float64)                       # [world, R]
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def choose_gathered(w: Weights, cand_all: torch.Tensor, R: int, positions: Sequence[int], sampling: Sampling | None,
                    with_prob: bool = False):
    """Apply keyed sampling to candidates gathered inside the step graph; ``with_prob`` also returns each selected token's probability."""

    world, width = int(w.meta["world"]), 2 * CAND + 1
    g = cand_all[:world * R * width].view(world, R, width).cpu().numpy()
    values = np.concatenate([g[r, :, :CAND] for r in range(world)], axis=1).astype(np.float32)
    tokens = np.concatenate([np.ascontiguousarray(g[r, :, CAND:2 * CAND]).view(np.int32) for r in range(world)],
                            axis=1).astype(np.int64)
    if sampling is None or sampling.temperature <= 0:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = g[:, :, 2 * CAND].astype(np.float64)
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for i, t in enumerate(chosen):
        hit = np.nonzero(tokens[i] == t)[0]
        probs.append(float(np.exp(float(values[i, hit[0]]) - total[i])) if len(hit) else 0.0)
    return chosen, probs


def choose_gathered_streams(w: Weights, cand_all: torch.Tensor, R: int, starts: Sequence[int],
                            positions: Sequence[Sequence[int]], samplings: Sequence[Sampling | None],
                            with_prob: bool = False):
    """One read-back of gathered rows; each stream samples its own rows with the same keyed rule it uses alone."""

    world, width = int(w.meta["world"]), 2 * CAND + 1
    g = cand_all[:world * R * width].view(world, R, width).cpu().numpy()
    values = np.concatenate([g[r, :, :CAND] for r in range(world)], axis=1).astype(np.float32)
    tokens = np.concatenate([np.ascontiguousarray(g[r, :, CAND:2 * CAND]).view(np.int32) for r in range(world)],
                            axis=1).astype(np.int64)
    if with_prob:
        lse = g[:, :, 2 * CAND].astype(np.float64)
        top = lse.max(axis=0)
        total = top + np.log(np.exp(lse - top).sum(axis=0))
    chosen_all, probs_all = [], []
    for k, (pos, smp) in enumerate(zip(positions, samplings)):
        a0, a1 = starts[k], starts[k + 1]
        v, t = values[a0:a1], tokens[a0:a1]
        if smp is None or smp.temperature <= 0:
            order = np.lexsort((t, -v), axis=-1)
            chosen = [int(t[i, order[i, 0]]) for i in range(a1 - a0)]
        else:
            chosen = choose_rows(v, t, pos, smp)
        chosen_all.append(chosen)
        if with_prob:
            probs = []
            for i, tok in enumerate(chosen):
                hit = np.nonzero(t[i] == tok)[0]
                probs.append(float(np.exp(float(v[i, hit[0]]) - total[a0 + i])) if len(hit) else 0.0)
            probs_all.append(probs)
    return (chosen_all, probs_all) if with_prob else chosen_all


def _gathered_fits(sampling: Sampling | None) -> bool:
    """Whether a step's gathered candidates (CAND a rank) cover the sampler's top-k plus its margin."""

    return sampling is None or sampling.temperature <= 0 or (bool(sampling.top_k) and sampling.top_k + MARGIN <= CAND)


PREFILL_ROWS = 2048      # rows of a prompt chunk


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's kept state ends: one token early, since a next turn sent back without its reasoning renders ``<think>`` and two newlines there."""

    return max(1, len(prompt) - 1)


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 4096, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, kv_dtype: str = "bf16") -> None:
        self.w = w
        self.capacity = capacity
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.kv_dtype = kv_dtype
        self.buf = Buffers(w, max_rows, capacity, moe_prefill=True)       # the experts' arithmetic MultiDecoder's use
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        self.st = State(w, capacity, max_rows, kv_dtype)
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            # experts that read their plan on the host (NVFP4) can't be captured: decline graphs before a capture fails
            if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
                print("[tensorfold] CUDA graphs off: this MoE reads its plan's item list on the host, which a "
                      "capture rejects; decode runs eagerly (correct, slower)")
            else:
                self.graphs = Graphs(self, max_rows=max_rows)

    def reset(self) -> None:
        self.st.reset(self.w)

    def twin(self) -> "Engine":
        """Share weights and scratch with an independent serial state, without MTP or graphs; requests run sequentially so scratch reuse leaves this engine's state and prefix cache intact."""

        other = object.__new__(Engine)
        other.w, other.capacity, other.rows, other.prefill_rows = self.w, self.capacity, self.rows, self.prefill_rows
        other.buf, other.mbuf, other.pbuf, other.graphs = self.buf, None, self.pbuf, None
        other.st = State(self.w, self.capacity, self.rows, self.st.kv_dtype)
        return other

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A decode step's forward (a CUDA graph when enabled): logits [R, V]."""

        if self.graphs is not None and self.st.image_positions is None:    # graphs are captured for text rows
            return self.graphs.forward(tokens)
        return forward(self.w, self.st, self.buf, tokens)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, gathered: bool = True) -> list[int]:
        """Rows of logits at their positions -> tokens (``draft``: the MTP head's; ``gathered``: own candidates)."""

        mapped = draft and self.w.draft_ids is not None
        if self.w.comm is not None:
            b = self.mbuf if draft else self.buf
            if gathered and logits.data_ptr() == b.logits.data_ptr() and _gathered_fits(sampling):
                return choose_gathered(self.w, b.cand_all, logits.shape[0], positions, sampling)
            return tp_sample_rows(self.w, logits, positions, sampling, offset=self.w.meta["vocab_offset"],
                                  id_map=self.w.draft_ids if mapped else None)
        if mapped:
            return sample_mapped(logits, positions, sampling, self.w.draft_ids)
        return sample_rows(logits, positions, sampling)

    def sample_draft(self, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
        """Return the position's keyed draft and its temperature-1 probability; this confidence ends draft chains early without changing output."""

        w = self.w
        mapped = w.draft_ids is not None
        if w.comm is not None:
            if logits.data_ptr() == self.mbuf.logits.data_ptr() and _gathered_fits(sampling):
                toks, probs = choose_gathered(w, self.mbuf.cand_all, 1, [position], sampling, with_prob=True)
            else:
                toks, probs = tp_sample_rows(w, logits[:1], [position], sampling, offset=w.meta["vocab_offset"],
                                             id_map=w.draft_ids if mapped else None, with_prob=True)
            return toks[0], probs[0]
        if mapped and getattr(self, "_draft_host", None) is None:
            self._draft_host = w.draft_ids.cpu().numpy()
        row = logits[:1].float()
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        if sampling is None or sampling.temperature <= 0:
            top, col = row.max(dim=-1, keepdim=True)            # the first maximum: argmax's (and serial's) choice
            got = torch.cat([top, lse, col.float()], dim=1).cpu().numpy()[0]       # one sync
            c = int(got[2])
            tok = int(self._draft_host[c]) if mapped else c
            return tok, float(np.exp(float(got[0]) - float(got[1])))
        k = min(row.shape[1], int(sampling.top_k) + MARGIN) if sampling.top_k else row.shape[1]
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        got = torch.cat([vals, lse, idx.float()], dim=1).cpu().numpy()[0]          # one sync
        cols = got[k + 1:].astype(np.int64)
        ids = self._draft_host[cols] if mapped else cols
        tok = choose_rows(got[None, :k].astype(np.float32), ids[None, :], [position], sampling)[0]
        hit = np.nonzero(ids == tok)[0]
        return int(tok), float(np.exp(float(got[hit[0]]) - float(got[k]))) if len(hit) else 0.0

    def mtp_forward(self, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
        if self.graphs is not None and self.st.image_positions is None:
            return self.graphs.mtp_forward(next_tokens, streams)
        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, streams)


def absorb(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions with main-model streams [n, S*D] and next tokens; logits of the last."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = e.mtp_forward(next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept rows and chain drafts, always retaining the first even below ``confidence``, then stopping before later drafts below it or after a low-confidence first draft."""

    st = e.st
    logits = absorb(e, streams, next_tokens)
    drafts: list[int] = []
    for j in range(count):
        low = False
        if confidence > 0:
            d, p = e.sample_draft(logits, position + j, sampling)
            low = p < confidence
            if low and j > 0:
                break
        else:
            d = e.sample(logits[:1], [position + j], sampling, draft=True)[0]
        drafts.append(d)
        if low:
            break
        if j + 1 < count:
            prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
            logits = e.mtp_forward([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
            next_tokens = [d]
    return drafts


def draft_cut(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
              sampling: Sampling | None, spec) -> tuple[list[int], list[float], int]:
    """``draft`` under a ``Speculation`` (staircut.py): absorb the kept rows, then (unless the round is plain) chain
    while ``spec.more`` and keep ``spec.keep`` drafts. Returns (drafts, their probabilities, MTP steps run)."""

    st = e.st
    logits = absorb(e, streams, next_tokens)
    if count < 1 or spec.plain(st.pos):
        return [], [], 1
    drafts: list[int] = []
    probs: list[float] = []
    for j in range(count):
        d, p = e.sample_draft(logits, position + j, sampling)
        drafts.append(d)
        probs.append(p)
        if j + 1 >= count or not spec.more(st.pos, probs):
            break
        prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
        logits = e.mtp_forward([d], prev)
        st.set_mtp_len(st.mtp_len + 1)
        st.mtp_drafted += 1
        next_tokens = [d]
    n = spec.keep(st.pos, probs)
    return drafts[:n], probs[:n], len(drafts)


def _absorbs(e: Engine, mtp: bool) -> bool:
    return mtp and e.w.mtp is not None and e.mbuf is not None


@torch.no_grad()
def prefill_begin(e: Engine, prompt: Sequence[int], *, mtp: bool = True, resume: dict | None = None) -> int:
    """Empty the state, or restore a kept prompt end and absorb its tail; returns the first prompt row to commit."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    if resume is None:
        e.reset()
        return 0
    st = e.st
    st.restore(resume["state"])
    if not 0 < st.pos < len(prompt):
        raise ValueError("a resumed prompt must extend the cached tokens")
    if _absorbs(e, mtp) and resume.get("tail") is not None:
        mtp_forward(e.w, st, e.pbuf, [prompt[st.pos]], resume["tail"])
        st.set_mtp_len(st.mtp_len + 1)
    return st.pos


def chunk_plan(start: int, n: int, rows: int, points, marks: int = 1) -> list[tuple[int, int, list[int]]]:
    """(start, end, kept points in (start, end]) for chunks of up to ``rows`` over prompt rows [start, n). A chunk
    keeps up to ``marks`` (at most MARKS) new states, its chain leaving those inside it as it passes, so a further
    point ends it there. Admission counts one new state a chunk on the serial engine: ``marks`` 1."""

    if not 1 <= marks <= MARKS:
        raise ValueError(f"a chunk keeps 1 to {MARKS} states, not {marks}")
    points, plan = sorted(p for p in set(points) if start < p <= n), []
    while start < n:
        here = [p for p in points if start < p <= min(start + rows, n)]
        end = here[marks - 1] if len(here) > marks else min(start + rows, n)
        plan.append((start, end, here[:marks]))
        start = end
    return plan


@torch.no_grad()
def prefill_chunk(e: Engine, prompt: Sequence[int], start: int, *, mtp: bool = True, points: Sequence[int] = (),
                  end: int | None = None) -> tuple[torch.Tensor | None, dict[int, dict]]:
    """Commit up to ``e.prefill_rows`` rows from ``start`` (the last chunk returns its logits); each of ``points`` in
    (start, end], up to MARKS inside, maps to what a fresh prefill of prompt[:point] leaves: {"state", "tail"}."""

    w, st, pb = e.w, e.st, e.pbuf
    end = min(start + e.prefill_rows, len(prompt) if end is None else end)
    chunk = list(prompt[start:end])
    R = len(chunk)
    final = end == len(prompt)
    rows = sorted({p - start for p in points})
    if any(not 0 < r <= R for r in rows):
        raise ValueError(f"kept points {sorted(points)} lie outside the chunk ({start}, {end}]")
    cuts = [Cut(r) for r in rows if r < R]                # inside the chunk, not at its end
    # only the prompt's last row is sampled: the head runs on the final chunk alone
    logits = forward(w, st, pb, chunk, logits=final, cuts=cuts)
    last = logits.clone() if final else None
    e.last_streams = pb.streams[R - 1:R].clone()
    use_mtp = _absorbs(e, mtp)
    at, found = {cut.row: cut for cut in cuts}, {}
    for r in rows:               # before the MTP head writes the streams: a point's tail, its state inside the chunk
        mtp_len = st.mtp_len + r - 1 if use_mtp else st.mtp_len           # every row but the point's last
        tail = pb.streams[r - 1:r].clone() if use_mtp else None
        found[start + r] = (cut_snapshot(w, st, pb, at[r], mtp_len) if r in at else None, mtp_len, tail)
    nxt = list(prompt[start + 1:end + 1])
    if use_mtp and nxt:
        mtp_forward(w, st, pb, nxt, pb.streams[:len(nxt)])
        st.set_mtp_len(st.mtp_len + len(nxt))
    commit(w, st, pb, R, R)
    # as a fresh prefill of prompt[:point] leaves it; a point that ends the chunk is the committed state
    return last, {p: {"state": snap if snap is not None else {**st.snapshot(), "mtp_len": mtp_len}, "tail": tail}
                  for p, (snap, mtp_len, tail) in found.items()}


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
            resume: dict | None = None, constraint=None, probabilities=None, keep_at: int | None = None,
            stops: Sequence[int] = (), keep=None, vision=None, marks: int = 1) -> int:
    """Commit the prompt in chunks and sample the first token (``resume`` equals a fresh run); ``e.kept`` resumes prompt[:keep_at].

    A chunk keeps up to ``marks`` of the states at ``stops`` and ``keep_at`` (chunk_plan)."""

    if vision is not None and (resume is not None or keep_at is not None or stops or keep is not None):
        raise ValueError("an image prompt prefills from its start and keeps no token-only snapshot")
    start, last = prefill_begin(e, prompt, mtp=mtp, resume=resume), None
    if vision is not None:
        image_rows.attach(e.st, vision, len(prompt))
    if keep_at is not None and not start <= keep_at <= len(prompt):
        raise ValueError(f"keep_at {keep_at} is outside the prefilled range [{start}, {len(prompt)}]")
    saved = e.kept = resume if keep_at == start else None
    points = {p for p in stops if start < p < len(prompt)} | ({keep_at} if keep_at is not None else set())
    for start, end, here in chunk_plan(start, len(prompt), e.prefill_rows, points, marks):
        last, kept = prefill_chunk(e, prompt, start, mtp=mtp, points=here, end=end)
        for point, k in kept.items():
            e.kept = k
            if point == keep_at:
                saved = k
            if keep is not None:
                keep(point, k["state"], k["tail"])
    if keep_at is not None:
        e.kept = saved
    image_rows.finish(e.st)
    if constraint is not None:                           # a reply's grammar: this rank's vocabulary columns
        last = constraint.mask(last, None, e.w.meta.get("vocab_offset", 0))
    first = e.sample(last, [len(prompt)], sampling)[0]
    if probabilities is not None:
        capture(last, [first], [len(prompt)], probabilities)
    if constraint is not None:
        constraint.advance([first])
    e.first = first
    return first


def time_windows(run, widths, reps: int = 5) -> dict[int, float]:
    """ms of ``run(rows)`` (a verify window's forward) for each width: the median of ``reps`` after one untimed."""

    out: dict[int, float] = {}
    for rows in widths:
        run(rows)
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t = time.perf_counter()
            run(rows)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        out[int(rows)] = round(sorted(times)[reps // 2], 4)
    return out


WARM_TAIL = 18      # a partial chunk after a full one: neither its rows nor the MTP head's 17 divide by 16


@torch.no_grad()
def warm(e: Engine) -> None:
    """Prefill a synthetic prompt (a full chunk, then a partial one cut at the kept point a row before its end) and empty the state, so no request compiles or loads a prompt kernel."""

    prompt = [0] * min(e.prefill_rows + WARM_TAIL + 1, e.capacity)
    prefill(e, prompt, None, keep_at=entry_end(prompt))
    e.kept = None
    e.reset()


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)
    widths: list[int] = field(default_factory=list)     # rows each round verified
    drafters: dict = field(default_factory=dict)        # {"mtp": [drafted, accepted], "copy": [...]}
    end: dict | None = None                             # ``keep_end``: the state where the reply ended, to resume

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens=None, constraint=None, probabilities=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``on_tokens(new)`` hears each step's token; it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        logits = e.forward([out[-1]])
        if constraint is not None:
            constraint.mask(logits[:1], None, w.meta.get("vocab_offset", 0))
        tok = e.sample(logits[:1], [st.pos + 1], sampling, gathered=constraint is None)[0]
        if probabilities is not None:
            capture(logits[:1], [tok], [st.pos + 1], probabilities)
        commit(w, st, b, 1, 1)
        out.append(tok)
        if constraint is not None:
            constraint.advance([tok])
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, committed=out[:-1], widths=[1] * (len(out) - 1))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, depth: int = DEPTH,
               confidence: float = CONFIDENCE, stop_eos: bool = False, on_tokens=None, constraint=None,
               probabilities=None, copies=None, spec=None,
               keep_end: bool = False) -> DecodeResult:  # copies: a CopyIndex over the prompt and pending
    """Verify pending and drafted tokens from the prefill state, commit rows before the first mismatched draft, and call ``on_tokens(new)`` with kept tokens after pending, stopping on True.

    ``spec`` (staircut.py's ``Speculation``, or None for the fixed confidence cut): each round's chain depth and
    whether it drafts at all, priced by this loop's own clock. ``keep_end``: the result's ``end`` resumes a prompt
    that extends the committed reply, as a kept prompt state does (the MTP head one row behind, the last row's
    stream as its tail); its bits are the decode's, which a prefill of the same tokens does not reproduce."""

    trace = {"probs": [], "steps": 1, "ms": 0.0, "kind": "mtp"}     # what wrote the next round's drafts

    def propose(streams, next_tokens, n):
        """Copy drafts when the index has a chain (the MTP cache still absorbs the kept rows), else the MTP chain."""
        t = time.perf_counter()
        copied = copies.chain(n) if copies is not None else []
        trace["kind"] = "copy" if copied else "mtp"
        if copied:
            absorb(e, streams, next_tokens)
            if spec is not None:
                trace.update(probs=[], steps=1, ms=(time.perf_counter() - t) * 1e3)
            return copied
        if spec is None:
            return draft(e, streams, next_tokens, st.pos + 1, n, sampling, confidence)
        drafts, probs, steps = draft_cut(e, streams, next_tokens, st.pos + 1, n, sampling, spec)
        trace.update(probs=probs, steps=steps, ms=(time.perf_counter() - t) * 1e3)
        return drafts

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    widths: list[int] = []
    by_drafter: dict[str, list[int]] = {}
    pos0 = st.pos
    unabsorbed = None                                  # the last round's kept rows, not yet in the MTP cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    drafts = propose(e.last_streams, [pending], min(depth, count - len(out)))
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        tokens = [out[-1]] + drafts
        window = constraint.window(tokens, list(range(-1, len(tokens) - 1))) if constraint is not None else None
        if window is not None:                           # the drafts no accepted path can hold are cut first
            tokens, drafts = window.tokens, window.tokens[1:]
        R = len(tokens)
        t0 = time.perf_counter() if spec is not None else 0.0
        context = st.pos
        logits = e.forward(tokens)
        if window is not None:
            constraint.mask(logits[:R], window, w.meta.get("vocab_offset", 0))
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling, gathered=window is None)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        if probabilities is not None:
            n = min(keep, count - len(out))
            capture(logits[:n], sampled[:n], list(range(st.pos + 1, st.pos + 1 + n)), probabilities)
        commit(w, st, b, R, keep)
        if spec is not None:                 # the sample's read-back synchronised: the clock covers the verify
            ms = (time.perf_counter() - t0) * 1e3
            if R == 1:
                spec.plain_round(context, ms + trace["ms"], timed=True)
            else:
                spec.verified(context, trace["probs"], R, keep - 1, timed=True, verify_ms=ms,
                              draft_ms=trace["ms"], steps=trace["steps"])
        unabsorbed = (keep, sampled[:keep])
        if R > 1:
            by = by_drafter.setdefault(trace["kind"], [0, 0])
            by[0] += R - 1
            by[1] += keep - 1
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        widths.append(R)
        new = sampled[:keep][:max(0, count - len(out))]
        if constraint is not None:
            constraint.advance(sampled[:keep])
        out.extend(sampled[:keep])
        if copies is not None:
            copies.extend(sampled[:keep])
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        n = min(depth, count - len(out))
        drafts = []
        if n > 0:
            drafts = propose(b.streams[:keep], sampled[:keep], n)
            unabsorbed = None
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    end = None
    if unabsorbed is not None and keep_end and st.image_positions is None:
        n = unabsorbed[0]                    # every kept row but the last; the last row's stream is the tail
        if n > 1:
            absorb(e, b.streams[:n - 1], unabsorbed[1][:n - 1])
        elif st.mtp_drafted:
            st.set_mtp_len(st.mtp_len - st.mtp_drafted)
            st.mtp_drafted = 0
        end = {"state": st.snapshot(), "tail": b.streams[n - 1:n].clone()}
    elif unabsorbed is not None:            # the MTP cache takes the last kept rows: it then covers the sequence
        absorb(e, b.streams[:unabsorbed[0]], unabsorbed[1])
    committed = out[:st.pos - pos0]
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, keeps, committed, widths, by_drafter, end)
