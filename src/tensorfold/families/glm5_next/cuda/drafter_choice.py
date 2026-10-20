"""Which drafter each GLM round uses, MTP chains or DFlash2 blocks, and the drafted decoding that follows it."""

from __future__ import annotations

import time
from typing import Sequence

import torch

from tensorfold.engine.exact_sampling import Sampling

from . import dflash2
from .decode import DecodeResult, DepthPolicy, Engine, Splice, _fed, _sync, absorb, draft
from .forward import commit


COPY_DRAFTS = 5         # a copy's first window's drafts: with the pending token, the widest window captured as a graph


def _other(arm: str) -> str:
    return "f" if arm == "m" else "m"


class DrafterChoice:
    """MTP chains ("m") or DFlash2 blocks ("f") a round, by tokens committed a model-ms; both ranks agree."""

    def __init__(self, costs: dict, *, first: str, explore: int = 2, every: int = 8, margin: float = 0.03,
                 window: int = 6, recheck: int = 3) -> None:
        self.costs = costs
        self.first = first
        self.explore, self.every, self.margin, self.window, self.recheck = explore, every, margin, window, recheck
        self.rounds: list[tuple[str, int, float]] = []       # (drafter, tokens committed, model ms)
        self.choice = first
        self.run = 0
        self.since = {"m": 0, "f": 0}      # a drafter's rate counts its rounds from here: left, it is judged afresh

    def cost(self, arm: str, rows: int, steps: int, backlog: int) -> float:
        """Model ms of a round: the verify window, then MTP steps or a DFlash2 block, plus the backlog's catch-up."""

        c = self.costs
        verify = c["verify"][min(rows, len(c["verify"])) - 1]
        if arm == "m":
            return verify + c["mtp"] + c["mtp_step"] * max(steps - 1, 0) + c["mtp_row"] * max(backlog - 1, 0)
        return verify + c["block"] + c["taps_row"] * backlog

    def rate(self, arm: str) -> float | None:
        rs = [r for r in self.rounds[self.since[arm]:] if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self) -> str:
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(_other(cur))
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin      # no bias at the first choice
        if ro is not None and (rc is None or ro > rc * need):
            self.since[cur] = n            # a bad stretch that forced the switch no longer counts against it
            self.choice = cur = _other(cur)
            self.run = max(0, self.every - self.recheck) if self.recheck else 0     # and it is probed again soon
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(cur)
        self.run += 1
        return cur

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        self.rounds.append((arm, keep, self.cost(arm, rows, steps, backlog)))


@torch.no_grad()
def auto_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                choice: DrafterChoice | None,
                m_policy: DepthPolicy, f_policy: DepthPolicy, stop_eos: bool = False, on_tokens=None,
                copies=None, splice: Splice | None = None,
                  lead: Sequence[int] = ()) -> DecodeResult:
    """Drafted decoding with ``choice`` picking each round's drafter; drafts only propose, so it equals serial.

    ``copies`` (a ``CopyIndex`` of the prompt and reply): where the reply repeats earlier text, the round drafts the
    earlier continuation ("c") and no drafter runs; the MTP head absorbs those rows later, as any backlog.
    ``splice``: a gate's cut written into the live stream (``decode.Splice``); a round that writes the fix's tokens
    ("s") runs no drafter either, and its rows join the backlogs the same way."""

    w, st, b = e.w, e.st, e.buf
    cap = 256                               # backlog rows a drafter may owe before it absorbs them anyway
    m_rows = torch.empty((cap, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    m_rows[:1].copy_(e.last_hidden)
    m_next: list[int] = [pending]
    f_taps = None
    n_f = 0
    if drafter is not None:
        f_taps = torch.empty((cap, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    sp = splice
    out, head = list(lead) if lead else [pending], pending
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    arms: list[str] = []
    last = {"m": (0, 0), "f": (0, 0), "c": (0, 0)}
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        room = count - len(out)
        t0 = time.perf_counter()
        forced = sp.drafts(room) if sp is not None else None
        copied = copies.chain(room) if copies is not None and forced is None else []
        arm = "s" if forced is not None else "c" if copied else choice.pick() if choice is not None else "m"
        if arm == "s":
            backlog, steps, drafts = 0, 0, forced
        elif arm == "c":
            backlog, steps, drafts = 0, 0, copied
        elif arm == "m":
            backlog = len(m_next)
            depth = max(1, min(m_policy.next(*last["m"]), room))
            drafts = draft(e, m_rows[:backlog], m_next, st.pos + 1, depth, sampling, m_policy.confidence)
            steps = 1 + st.mtp_drafted
            m_next = []
        else:
            backlog = n_f
            if n_f:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            depth = max(1, min(f_policy.next(*last["f"]), room))
            drafts = drafter.propose(head, depth, sampling, f_policy.confidence)
            steps = 0
        t1 = time.perf_counter()
        tokens = e.verify_window([head] + drafts)
        drafts = tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts if arm != "s" else ()):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        if arm == "f" and dflash2.TREE_PROBE:
            dflash2.tree_probe(drafter.probe, drafts, sampled, keep)
        t3 = time.perf_counter()
        rows, new, follows = (sp.round(tokens, sampled, keep, len(forced or ())) if sp is not None
                              else (keep, list(sampled[:keep]), list(sampled[:keep])))
        commit(w, st, b, R, rows)
        t4 = time.perf_counter()
        # the kept rows join both backlogs (a full backlog is taken first), each with the token that follows it
        if len(m_next) + rows > cap:
            absorb(e, m_rows[:len(m_next)], m_next)
            m_next = []
        m_rows[len(m_next):len(m_next) + rows].copy_(e.main_hidden(slice(0, rows)))
        m_next.extend(follows)
        if drafter is not None:
            if n_f + rows > cap:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            f_taps[n_f:n_f + rows].copy_(e.tap_rows(rows))
            n_f += rows
        t5 = time.perf_counter()
        if choice is not None and arm not in ("c", "s"):
            choice.record(arm, R, steps, backlog, keep)
        if arm == "c":
            copies.verified(len(drafts), keep - 1)
        if copies is not None:
            copies.extend(new)
        if arm != "s":
            last[arm] = (len(drafts), keep - 1)
            drafted += len(drafts)
            accepted += keep - 1
        rounds += 1
        depths.append(len(drafts))
        keeps.append(rows)
        arms.append(arm)
        e.follow(new)
        out.extend(new)
        head = follows[-1]
        stop = _fed(w, on_tokens, new[:max(0, count - (len(out) - len(new)))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        if stop:
            break
    _sync(w)
    seconds = time.perf_counter() - start
    if drafter is not None and n_f:
        drafter.add_taps(f_taps[:n_f])          # DFlash2's context ends where the committed rows end
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, stages, depths, keeps, "".join(arms),
                        m_rows[:len(m_next)])
