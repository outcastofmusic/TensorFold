"""CUDA decode accepts drafts only when they match the serial keyed sample; all ranks sample identical gathered candidates without a broadcast."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.sampling import comm_gather, nucleus_rows, one_rank
from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import glue, prof, qmm
from .forward import Buffers, Cut, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_forward, mtp_stage
from .sparse import pool_bucket
from .weights import Weights


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    if not greedy and not sampling.top_k:           # top_k off: the shared nucleus rule over every rank's shard
        return nucleus_rows(logits, positions, sampling, offset=w.vocab_offset if offset is None else offset,
                            gather=one_rank if w.comm is None else comm_gather(w.comm), probs=probs)
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
        w.comm.all_gather(packed.view(-1), got)
        g = got.view(w.world, R, 2 * k).cpu()
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Return each chosen token's top-k/top-p probability, using temperature 1 for greedy draft confidence."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


PREFILL_ROWS = 2048      # rows of a prompt chunk


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = ()) -> None:
        self.w = w
        w.meta["long_context"] = long_context
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.buf = Buffers(w, max_rows, capacity)
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        if taps:
            self.buf.set_taps(tuple(taps), w.cfg.hidden)         # before any graph capture
            self.pbuf.set_taps(tuple(taps), w.cfg.hidden)
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.st = State(w, capacity, max_rows)
        self.last_hidden: torch.Tensor | None = None
        self.constraint = self.window = None            # a request's grammar, and the next sample's rows under it
        self.draft_n = w.head.n
        self.graphs = None
        self.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}   # steps by path
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, graph_rows, graph_rows)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g, kind, bucket = None, "main", None
        if self.graphs is not None and dense:
            g = self.graphs.main.get((R, self.st.parity))
        elif self.graphs is not None and self.st.pos >= self.w.cfg.dense_limit and self.st.index is not None:
            # every row past the dense limit: the sparse graph for this pool bucket (same kernels as eager)
            bucket = pool_bucket(self.st.pos, R, self.st.index[0][2].shape[0] - 2)
            g, kind = self.graphs.sparse.get((R, self.st.parity, bucket)), "sparse"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.buf.logits[:R]
        self.replays["eager"] += 1
        out = compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)
        if bucket is not None:
            self.graphs.capture_sparse(self, R, bucket)
        return out

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world]."""

        if self.mbuf is None:
            raise ValueError("MTP weights/work were explicitly omitted")
        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        g, kind = None, "mtp"
        if self.graphs is not None and not self.mbuf.zero_first and dense:
            g = self.graphs.mtp.get(n)
        elif (self.graphs is not None and not self.mbuf.zero_first and self.st.index is not None
              and self.st.mtp_len >= self.w.cfg.dense_limit):
            bucket = pool_bucket(self.st.mtp_len, n, self.st.index[-1][2].shape[0] - 2)
            g, kind = self.graphs.sparse_mtp.get((n, bucket)), "sparse_mtp"
        if g is not None:
            self.replays[kind] += 1
            g.replay()
            return self.mbuf.logits[:1, :self.draft_n]
        from .latent import CHUNK

        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // CHUNK),
                           host_pos=self.st.mtp_len)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        if not draft and self.constraint is not None and self.window is not None:
            logits = logits.float().clone()  # grammar masks must not alter graph-owned buffers either
            self.constraint.mask(logits, self.window, self.w.vocab_offset)   # this rank's vocabulary columns
            self.window = None
        return sample_rows(self.w, logits, positions, sampling, None, probs)

    def verify_window(self, tokens: list[int]) -> list[int]:
        """The window a reply's grammar keeps (a chain cut at its first rejected draft), masked at the next sample."""

        if self.constraint is None:
            return tokens
        self.window = self.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
        return self.window.tokens

    def follow(self, tokens: Sequence[int]) -> None:
        if self.constraint is not None:
            self.constraint.advance(tokens)

    def tap_rows(self, n: int, b: Buffers | None = None) -> torch.Tensor:
        """The last forward's first n rows of DFlash2 taps, concatenated in layer order: [n, taps * D]."""

        return torch.cat([t[:n] for t in (b or self.buf).taps], dim=1)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """Return the final-normed main-model rows that the MTP head reads after a forward with logits."""

        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """Absorb hidden rows and their next tokens into the MTP cache in independent chunks; return the last logits."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept positions and chain drafts until cumulative confidence fails, always keeping the first draft."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass(eq=False)          # the cache's lookups and removals mean this one state: identity
class Snapshot:
    """Copy committed KDA state and retain attention caches whose valid prefixes survive resume; pending MTP rows await next tokens from the new prompt."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    drafter_end: int
    rows: list | None = None      # the attention rows of ids, saved when another conversation took the live caches
    nbytes: int = 0
    drafter_rows: list | None = None  # a ring drafter's window rows before drafter_end, copied when taken
    born: int = -1                # a background pass's state: the length of the turn it was kept for (-1: its own)
    branched: bool = False        # resumed again after a later turn extended it: freed only as least recently used
    blocks: list | None = None    # its rows' full blocks on disk, once written or read back (disk_states.blocks)


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool,
                  drafter=None) -> Snapshot:
    """A ring drafter's window rows are copied now (``_ring_window``): its next rows overwrite them in the ring."""
    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    snap = Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    st.mtp_len - st.mtp_drafted if mtp and pending is not None else -1,
                    drafter.context_end if drafter is not None else -1)
    if drafter is not None and getattr(drafter, "ring", 0) and snap.drafter_end == len(snap.ids):
        snap.drafter_rows = _ring_window(drafter, len(snap.ids))
    return snap


def _ring_slots(drafter, n: int) -> torch.Tensor:
    """The ring slots of the window rows a block pass at context end n reads (n - window - 1 .. n - 1)."""
    lo = max(0, n - drafter.window - 1)
    return torch.arange(lo, n, device=drafter.kc[0].device) % drafter.ring


def _ring_window(drafter, n: int) -> list[torch.Tensor]:
    """A copy of a ring drafter's window rows before n, in position order."""
    idx = _ring_slots(drafter, n)
    return [c.index_select(1, idx) for c in drafter.kc] + [c.index_select(1, idx) for c in drafter.vc]


def _put_ring_window(drafter, n: int, rows: list[torch.Tensor]) -> None:
    idx = _ring_slots(drafter, n)
    caches = list(drafter.kc) + list(drafter.vc)
    if len(rows) != len(caches) or any(r.shape[1] != idx.numel() for r in rows):
        raise ValueError("a kept state's DFlash2 window does not match the drafter's ring")
    for c, r in zip(caches, rows):
        c.index_copy_(1, idx, r)


def _row_views(st, n: int, m: int) -> list[torch.Tensor]:
    """Views of the attention rows a snapshot of n tokens (m in the MTP head) depends on: latents, indexer keys, gates and pools."""
    views = [kc[:n] for kc in st.kc]
    idx = st.index or []
    main_idx = idx[:len(st.kc)]
    for ik, ig, pk in main_idx:
        views += [ik[:n], ig[:n], pk[:n // 4 + 1]]
    if m > 0 and hasattr(st, "mtp_kc"):
        views.append(st.mtp_kc[:m])
        if len(idx) > len(st.kc):
            ik, ig, pk = idx[-1]
            views += [ik[:m], ig[:m], pk[:m // 4 + 1]]
    return views


def complete_rows(st, n: int, m: int) -> list[int]:
    """The rows of each view of ``_row_views(st, n, m)`` that no later token changes: all but a pool's last, which
    pools the tokens past the state's end as they come."""
    pools = {pk.data_ptr() for _, _, pk in st.index or []}
    return [v.shape[0] - (v.data_ptr() in pools) for v in _row_views(st, n, m)]


def save_rows(e: Engine, snap: Snapshot) -> None:
    """Copy a snapshot's attention rows out of the live caches before another conversation overwrites them.

    A ring drafter's window was copied when the snapshot was taken, so the state still resumes with DFlash2 drafts;
    a flat drafter's rows are not copied, and such a state resumes only without them.
    """
    views = _row_views(e.st, len(snap.ids), max(snap.mtp_len, 0))
    snap.rows = [v.clone() for v in views]
    snap.nbytes = sum(r.numel() * r.element_size() for r in snap.rows)
    if snap.drafter_rows is None:
        snap.drafter_end = -1


def row_bytes(e: Engine, snap: Snapshot) -> int:
    """What ``save_rows`` would copy for this snapshot."""
    return sum(v.numel() * v.element_size() for v in _row_views(e.st, len(snap.ids), max(snap.mtp_len, 0)))


def snapshot_bytes(snap: Snapshot) -> int:
    """A kept snapshot's device bytes: KDA states, conv windows, pending MTP rows, a ring window, saved rows."""
    held = [snap.rec, snap.conv] + ([snap.pending] if snap.pending is not None else []) + (snap.drafter_rows or [])
    return sum(t.numel() * t.element_size() for t in held) + (snap.nbytes if snap.rows is not None else 0)


def load_rows(e: Engine, snap: Snapshot) -> None:
    """Put a saved snapshot's attention rows back into the live caches."""
    for dst, src in zip(_row_views(e.st, len(snap.ids), max(snap.mtp_len, 0)), snap.rows):
        dst.copy_(src)


def restore(e: Engine, snap: Snapshot, drafter=None) -> None:
    st = e.st
    if st.cur:
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0
    if drafter is not None:
        if getattr(drafter, "ring", 0):
            if snap.drafter_rows is None or snap.drafter_end != len(snap.ids):
                raise ValueError("this snapshot kept no DFlash2 window for the drafter's ring")
            _put_ring_window(drafter, snap.drafter_end, snap.drafter_rows)
        drafter.context_end = snap.drafter_end
        drafter.pos_dev.fill_(snap.drafter_end)


# -- prefill ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
            resume: Snapshot | None = None, keep_at: int | None = None, stops: Sequence[int] = (), keep=None,
            stop=None, rows: int | None = None) -> int:
    """Commit the prompt in chunks and sample its first token; a resumed prompt ends in a fresh prefill's state.

    ``keep`` gets the state at ``keep_at`` after the last chunk, and the state at each of ``stops`` as its chunk
    ends. A kept point splits no chunk: one inside a chunk is a cut (``forward.Cut``), which runs the KDA chain in
    pieces within the same pass, so a prompt costs its passes however many states it keeps.

    ``stop()`` is asked before each chunk: once it holds, the prompt is abandoned with ``RequestCancelled`` and
    the caller must reset the engine before its next prompt.

    ``rows`` caps a chunk below ``e.prefill_rows``: a background pass asked to stop waits out a shorter chunk.
    """

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.pbuf
    use_mtp = mtp and w.mtp is not None and w.cfg.mtp_layers > 0
    begin = 0
    if resume is None:
        e.reset()
        if drafter is not None:
            drafter.reset()
    else:
        begin = len(resume.ids)
        if begin >= len(prompt) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        restore(e, resume, drafter)
        if use_mtp:
            k = resume.pending.shape[0]
            _absorb_rows(e, resume.pending, list(prompt[begin - k + 1:begin + 1]))
    if keep_at is not None and (keep is None or not max(1, begin) <= keep_at <= len(prompt)):
        raise ValueError("a kept prefix needs a callback and a point in the prompt's prefill")
    stops = sorted({p for p in stops if begin < p < len(prompt) and p != keep_at})
    if stops and keep is None:
        raise ValueError("kept prefixes need a callback")
    points = sorted(stops + ([keep_at] if keep_at is not None and keep_at > begin else []))
    kept = resume if keep_at == begin else None
    last = None
    prof.active = True
    start = begin
    while start < len(prompt):
        if stop is not None and stop():
            from tensorfold.server.cancellation import RequestCancelled

            prof.active = False
            raise RequestCancelled("the client left while its prompt was filling")
        end = min(start + min(rows or e.prefill_rows, e.prefill_rows), len(prompt))
        chunk = list(prompt[start:end])
        R = len(chunk)
        here = [p - start for p in points if start < p <= end]
        cuts = [Cut(point, torch.empty_like(st.rec[0]), st.conv.clone()) for point in here if point < R]
        last = compute(w, st, b, stage(w, st, b, chunk), nch=chunks_for(st, R), host_pos=st.pos, cuts=cuts).clone()
        e.last_hidden = b.fnormed[R - 1:R].clone()
        snaps = []
        for point, cut in zip(here, cuts + [None]):       # the cuts in order, then a point at the chunk's end
            rec = cut.rec if cut is not None else st.rec[st.cur[0] if st.cur else 0].clone()
            conv = cut.conv if cut is not None else st.conv.clone()
            at = start + point
            snaps.append(Snapshot(list(prompt[:at]), rec, conv,
                                  b.fnormed[point - 1:point].clone() if use_mtp else None,
                                  at - 1 if use_mtp else -1, at if drafter is not None else -1))
        if drafter is not None:
            taps, done = e.tap_rows(R, b), 0
            # a ring's rows past a kept point overwrite its window once they pass the ring's slack (129 rows for
            # GLM-5.3-Flash-DFlash2), so the taps go in up to each point and its window is copied there
            for snap in snaps if getattr(drafter, "ring", 0) else ():
                point = len(snap.ids) - start
                drafter.add_taps(taps[done:point])
                snap.drafter_rows, done = _ring_window(drafter, len(snap.ids)), point
            drafter.add_taps(taps[done:])
        if use_mtp:
            nxt = list(prompt[start + 1:start + R + 1])
            if nxt:
                with prof.timed("mtp absorb"):
                    _absorb_rows(e, b.fnormed[:len(nxt)], nxt)
        with prof.timed("commit"):
            commit(w, st, b, R, R)
        for snap in snaps:
            if len(snap.ids) == keep_at:
                kept = snap
            else:
                keep(snap)
        start = end
    if kept is not None:
        keep(kept)
    prof.active = False
    prof.report(len(prompt) - begin)
    if e.constraint is not None:                         # the first token's row, under the reply's grammar
        e.window = e.constraint.window([0], [-1])
    first = e.sample(last, [len(prompt)], sampling)[0]
    e.follow([first])
    return first


def _absorb_rows(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> None:
    """A prompt's rows into the MTP cache through the prefill buffers (the prefill arithmetic, like the prompt)."""

    st = e.st
    mtp_forward(e.w, st, e.pbuf, next_tokens, hidden)
    st.set_mtp_len(st.mtp_len + len(next_tokens))


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    arms: str = ""                          # auto_decode: the drafter of each round, "m" (MTP) or "f" (DFlash2)
    pending: torch.Tensor | None = None     # auto_decode: MTP input rows of committed positions not absorbed yet

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


FORCED_ROWS = 5                          # a fix's tokens a round writes, within the widest verify window


class Splice:
    """A gate's cut written into the live stream (``call_gate.generate_gated``'s inline path, as Flash Next's
    ``MultiDecoder._spliced``), so the reply is not prefilled again after a thinking budget's close or a call's fix.

    The gates see each round's tokens before it commits. At a cut the round commits only the rows before the cut's
    token, the fix's first token becomes the pending one and the rest wait as drafts that later rounds write without
    verifying them. The fix's rows come from the verify forward, as a draft's do: text after a cut is the model's
    continuation of the same tokens, not bit for bit the continuation a fresh prefill of them would give. A decode
    loop's ``lead`` is the tokens the reply already has (the prompt's first token, or the fix it was cut into)."""

    def __init__(self, splice) -> None:
        self.splice = splice
        self.forced: list[int] = []

    def drafts(self, room: int) -> list[int] | None:
        """The fix's tokens this round writes, or None when none wait."""

        if not self.forced or room <= 0:
            return None
        take = min(room, FORCED_ROWS)
        drafts, self.forced = self.forced[:take], self.forced[take:]
        return drafts

    def round(self, window: list[int], sampled: list[int], keep: int, forced: int) -> tuple[int, list[int], list[int]]:
        """(rows to commit, tokens the reply gains, each committed row's next token) for a round whose window held
        ``forced`` of a fix's tokens after its pending one: those rows are kept whatever was sampled."""

        if forced:
            keep = forced + 1
            new = [int(sampled[forced])]
        else:
            new = [int(t) for t in sampled[:keep]]
        hit = self.splice(new)
        if hit is None or not hit[1]:
            return keep, new, [*window[1:keep], new[-1]]
        at, fix = hit
        rows = keep - len(new) + 1 + int(at)             # the pending row, then every row before the cut's token
        self.forced = [int(t) for t in fix[1:]]
        return rows, [*new[:at], *map(int, fix)], [*window[1:rows], int(fix[0])]


def lead_of(splice, first: int) -> tuple[Splice | None, list[int]]:
    """The decode loops' ``Splice`` for an inline gate's ``splice`` (None without one) and the reply's first tokens:
    the prompt's sampled token, or the fix a gate cut it into, whose first token is then the pending one."""

    if splice is None:
        return None, [int(first)]
    sp = Splice(splice)
    hit = sp.splice([int(first)])
    if hit is None or not hit[1]:
        return sp, [int(first)]
    lead = [int(t) for t in hit[1]]
    sp.forced = lead[1:]
    return sp, lead


def _fed(w: Weights, on_tokens, tokens: list[int]) -> bool:
    """Hand a round's tokens to ``on_tokens``; True when it asks to stop and this rank decodes alone.

    A thinking budget's cut or a client that left asks it. Two ranks decode on to the count or an end token: rank 1
    runs without a feed, so it cannot hear rank 0's stop and would wait in the next round's collective."""

    return on_tokens is not None and bool(on_tokens(tokens)) and w.comm is None


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None, splice: Splice | None = None,
                  lead: Sequence[int] = ()) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``splice``: a
    gate's cut written into the live stream (``Splice``), the fix's tokens a step each without sampling."""

    w, st, b = e.w, e.st, e.buf
    sp = splice
    out, head = list(lead) if lead else [pending], pending
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    early = bool(getattr(w, "meta", {}).get("early_launch", False))      # TF_GLM_EARLY_LAUNCH (engine.early_launch)
    ahead = None                                          # the next step's logits, launched before the host's callback
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = ahead if ahead is not None else e.forward(e.verify_window([head]))
        ahead = None
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if sp is not None and sp.forced:
            tok, new = sp.forced.pop(0), []               # the fix's next token: the reply has it already
        else:
            tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
            new = [tok]
            hit = sp.splice(new) if sp is not None else None
            if hit is not None and hit[1]:
                new, tok = [int(t) for t in hit[1]], int(hit[1][0])
                sp.forced = new[1:]
        e.follow(new)
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.extend(new)
        head = tok
        if early and len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
            # the next step needs only head and the commit; a stop below leaves its rows uncommitted, as a verify
            # window leaves its refused drafts
            ahead = e.forward(e.verify_window([head]))
        if new and _fed(w, on_tokens, new):
            break
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Choose a fixed draft count or adapt it to running acceptance."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.rate = 0.8

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None, splice: Splice | None = None,
                  lead: Sequence[int] = ()) -> DecodeResult:
    """Verify pending and MTP draft rows through the first mismatch, starting with every prompt position except the
    last in the MTP cache; ``splice``: a gate's cut written into the live stream (``Splice``)."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    sp = splice
    out, head = list(lead) if lead else [pending], pending
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    drafts = draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence) if depth > 0 else []
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        forced = sp.drafts(count - len(out)) if sp is not None else None
        tokens = e.verify_window([head] + (forced if forced is not None else drafts))
        drafts = tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts if forced is None else ()):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        rows, new, follows = (sp.round(tokens, sampled, keep, len(forced or ())) if sp is not None
                              else (keep, list(sampled[:keep]), list(sampled[:keep])))
        commit(w, st, b, R, rows)
        t3 = time.perf_counter()
        rounds += 1
        if forced is None:
            drafted += len(drafts)
            accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(rows)
        e.follow(new)
        out.extend(new)
        head = follows[-1]
        stop = _fed(w, on_tokens, new[:max(0, count - (len(out) - len(new)))])
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if stop or len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        # the MTP head absorbs this round's rows even when the next round writes a fix's tokens instead of drafts
        drafts = (draft(e, e.main_hidden(slice(0, rows)), follows, st.pos + 1, depth, sampling, policy.confidence)
                  if depth > 0 else [])
        stages["draft"] += time.perf_counter() - t4
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


@torch.no_grad()
def dflash_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                  policy: DepthPolicy | None = None, stop_eos: bool = False, on_tokens=None,
                  splice: Splice | None = None, copies=None,
                  lead: Sequence[int] = ()) -> DecodeResult:
    """Verify pending and DFlash2 draft rows through the first mismatch and absorb kept taps, starting from prefill's
    drafter state; ``splice``: a gate's cut written into the live stream (``Splice``). ``copies`` (a ``CopyIndex``
    of the prompt and reply, TF_GLM_DFLASH_COPIES): where the reply repeats earlier text, the round verifies the
    earlier continuation in place of a block, and the drafter absorbs its kept rows' taps as any round's."""

    w, st, b = e.w, e.st, e.buf
    from . import dflash2

    policy = policy or DepthPolicy(3, fixed=True)
    sp = splice
    out, head = list(lead) if lead else [pending], pending
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    depths: list[int] = []
    keeps: list[int] = []
    early = bool(getattr(w, "meta", {}).get("early_launch", False)) and sp is None    # TF_GLM_EARLY_LAUNCH (engine.early_launch)
    _sync(w)
    start = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    try:
        return _dflash_rounds(e, drafter, w, st, b, dflash2, policy, sp, out, head, depth, count, sampling, stop_eos,
                              on_tokens, early, stages, depths, keeps, start, copies)
    finally:
        drafter.ahead = None                       # a block pass launched for a round that never came


def _dflash_rounds(e, drafter, w, st, b, dflash2, policy, sp, out, head, depth, count, sampling, stop_eos, on_tokens,
                   early, stages, depths, keeps, start, copies=None) -> DecodeResult:
    rounds = drafted = accepted = 0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        forced = sp.drafts(count - len(out)) if sp is not None else None
        copied = copies.chain(count - len(out)) if copies is not None and forced is None else []
        drafts = forced if forced is not None else copied if copied else (
            drafter.propose(head, depth, sampling, policy.confidence) if depth > 0 else [])
        t1 = time.perf_counter()
        tokens = e.verify_window([head] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts if forced is None else ()):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        if copied:
            copies.verified(len(drafts), keep - 1)
        elif dflash2.TREE_PROBE and forced is None:
            dflash2.tree_probe(drafter.probe, drafts, sampled, keep)
        t3 = time.perf_counter()
        rows, new, follows = (sp.round(tokens, sampled, keep, len(forced or ())) if sp is not None
                              else (keep, list(sampled[:keep]), list(sampled[:keep])))
        commit(w, st, b, R, rows)
        t4 = time.perf_counter()
        drafter.add_taps(e.tap_rows(rows))
        t5 = time.perf_counter()
        rounds += 1
        if forced is None:
            drafted += len(drafts)
            accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(rows)
        e.follow(new)
        out.extend(new)
        if copies is not None:
            copies.extend(new)
        head = follows[-1]
        if early and len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
            # the next round's block pass needs only head and the taps just added
            drafter.launch_ahead(head, policy.most)
        stop = _fed(w, on_tokens, new[:max(0, count - (len(out) - len(new)))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        if stop:
            break
        # a copy round leaves the block policy where its last block put it
        depth = min(depth if copied else policy.next(len(drafts), keep - 1), count - len(out))
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)
