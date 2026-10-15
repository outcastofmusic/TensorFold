"""Which MTP drafts a Flash Next round verifies, by expected tokens a millisecond, and when a round drafts at all.

``StairCut`` (TENSORFOLD_STAIRCUT=1) replaces the fixed confidence cut (stop before a draft the head gives less than
``CONFIDENCE``). A round that verifies the first ``n`` drafts commits ``1 + q_1 + ... + q_n`` tokens on average, where
``q_k`` is the chance the first ``k`` drafts all land: the product of each draft's calibrated probability
(``Calibration``: the head's probability mapped to how often drafts of that probability landed). It costs the verify
of ``n + 1`` rows (``Prices``) plus ``n`` MTP steps. The cut keeps the ``n`` that maximises tokens a millisecond, and
the chain stops drafting once deeper drafts are not expected to raise it.

``PlainGate`` (TENSORFOLD_PLAIN_GATE=1) keeps decoding from running slower than the plain step: per context class it
follows a drafted round's ms a committed token against a plain round's ms, and takes plain rounds (the anchor alone,
the MTP head still absorbing the kept rows) while drafting loses by more than 10%, drafting one probe round every 16
rounds, an interval that doubles to 256 while drafting keeps losing. A class measures its plain step with one plain
round after its first 16 drafted rounds, and again every 256.

Both change which rows a round verifies, never what it commits: every verified row is sampled as the serial step
samples it, so the text is the serial reference's either way. Prices and costs come from this process's clocks, so
two ranks of a split model (which must choose identical windows) run neither.

The idea and its thresholds come from TandemLLM's StairCut (AGPL-3.0); this is an independent implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tensorfold.env_flags import flag

ENV_CUT = "TENSORFOLD_STAIRCUT"
ENV_GATE = "TENSORFOLD_PLAIN_GATE"
FIRST_CLASS = 8192        # contexts up to here are class 0; above it, the next power of two (the graphs' buckets)


def context_class(n: int) -> int:
    """The price class of a context of ``n`` tokens: 0 up to 8,192, else the power of two at or above ``n``."""

    return 0 if n <= FIRST_CLASS else 1 << (int(n) - 1).bit_length()


class Prices:
    """The ms of a verified round of ``rows`` rows (verify, sample and commit), per context class.

    ``prior``: {class: {rows: ms}}, the table measured at warm-up; a class missing from it takes the nearest lower one.
    Each observed round moves its row count's price by EMA 0.1 (from where the prior put it) and the class's shift of
    the prior by the same weight, so counts not seen yet at a long context follow the prior plus that shift. A round
    over 4x its price is a stall and teaches nothing. Prices never fall with more rows: a cut must not buy a row
    because one count was measured at a quieter moment. Without a prior, a count not measured yet costs what the
    nearest measured count costs."""

    EMA = 0.1
    OUTLIER = 4.0

    def __init__(self, prior: dict[int, dict[int, float]] | None = None) -> None:
        self.prior = {int(c): {int(r): float(ms) for r, ms in rows.items()} for c, rows in (prior or {}).items()}
        self.learned: dict[int, dict[int, float]] = {}
        self.shift: dict[int, float] = {}

    def _prior(self, rows: int, cls: int) -> float | None:
        if not self.prior:
            return None
        below = [c for c in self.prior if c <= cls]
        table = self.prior[max(below) if below else min(self.prior)]
        if not table:
            return None
        if rows in table:
            return table[rows]
        known = sorted(table)
        if rows < known[0]:
            return table[known[0]]
        if rows > known[-1]:
            if len(known) == 1:
                return table[known[0]]
            a, b = known[-2], known[-1]
            return table[b] + (table[b] - table[a]) / (b - a) * (rows - b)
        lo = max(r for r in known if r < rows)
        hi = min(r for r in known if r > rows)
        return table[lo] + (table[hi] - table[lo]) * (rows - lo) / (hi - lo)

    def _raw(self, rows: int, cls: int) -> float | None:
        learned = self.learned.get(cls, {})
        if rows in learned:
            return learned[rows]
        prior = self._prior(rows, cls)
        if prior is not None:
            return prior + self.shift.get(cls, 0.0)
        above = [r for r in learned if r > rows]          # no prior: the nearest count measured, never a free row
        if above:
            return learned[min(above)]
        return learned[max(learned)] if learned else None

    def table(self, n: int, cls: int) -> list[float]:
        """[0, price of 1 row, ..., price of ``n`` rows], never falling with rows (0 while nothing is known)."""

        out, top = [0.0], 0.0
        for rows in range(1, n + 1):
            v = self._raw(rows, cls)
            top = max(top, v if v is not None else 0.0)
            out.append(top)
        return out

    def known(self, cls: int) -> bool:
        return bool(self.prior) or bool(self.learned.get(cls))

    def observe(self, rows: int, ms: float, cls: int) -> None:
        if rows < 1 or ms <= 0.0:
            return
        ref = self.table(rows, cls)[rows]
        if ref > 0.0 and ms > self.OUTLIER * ref:
            return
        a = self.EMA
        learned = self.learned.setdefault(cls, {})
        start = self._raw(rows, cls)
        learned[rows] = ms if start is None else (1 - a) * start + a * ms
        prior = self._prior(rows, cls)
        if prior is not None:
            self.shift[cls] = (1 - a) * self.shift.get(cls, 0.0) + a * (ms - prior)


class Calibration:
    """The head's probability mapped to how often its drafts landed, by tenth of probability: (hits + 4p) / (n + 4),
    the evidence halved past 400 a bin so it follows the text."""

    PRIOR = 4.0
    CAP = 400.0

    def __init__(self) -> None:
        self.bins = [[0.0, 0.0] for _ in range(10)]

    def q(self, p: float) -> float:
        hits, n = self.bins[min(max(int(p * 10), 0), 9)]
        return (hits + self.PRIOR * p) / (n + self.PRIOR)

    def learn(self, p: float, hit: bool) -> None:
        b = self.bins[min(max(int(p * 10), 0), 9)]
        b[0] += float(hit)
        b[1] += 1.0
        if b[1] > self.CAP:
            b[0] *= 0.5
            b[1] *= 0.5


class StairCut:
    """The prefix of a drafted chain that maximises (1 + sum of q) / (verify price of its rows + its MTP steps)."""

    EMA = 0.1
    OUTLIER = 4.0

    def __init__(self, prices: Prices, calibration: Calibration | None = None) -> None:
        self.prices = prices
        self.cal = calibration or Calibration()
        self.step: dict[int, float] = {}           # class -> ms of one MTP step (the absorb counts as one)

    def _reach(self, probs: Sequence[float]) -> list[float]:
        out, run = [], 1.0
        for p in probs:
            run *= self.cal.q(p)
            out.append(run)
        return out

    def _cost(self, price: list[float], n: int, cls: int, paid: int) -> float:
        """A round verifying ``n`` drafts, ``paid`` MTP steps already run: a shorter cut gets none of them back."""

        return price[n + 1] + self.step.get(cls, 0.0) * max(n, paid, 1)

    def _best(self, reach: list[float], price: list[float], cls: int) -> tuple[float, int]:
        paid = len(reach)
        best, gain = (1.0 / max(self._cost(price, 0, cls, paid), 1e-9), 0), 1.0
        for n in range(1, len(reach) + 1):
            gain += reach[n - 1]
            score = gain / max(self._cost(price, n, cls, paid), 1e-9)
            if score > best[0]:
                best = (score, n)
        return best

    def cut(self, probs: Sequence[float], cls: int) -> int:
        """How many of the drafted ``probs`` to verify (0: the anchor alone, a plain round)."""

        if not self.prices.known(cls):
            return len(probs)                     # nothing priced yet: verify all, which prices the rows
        price = self.prices.table(len(probs) + 1, cls)
        return self._best(self._reach(probs), price, cls)[1]

    def more(self, probs: Sequence[float], cls: int, most: int) -> bool:
        """Draft one more level after ``probs``? Only if the deeper drafts are expected to raise the best score within
        ``most`` drafts, each landing as often as the last one's calibrated chance.

        An optimistic bound (every deeper draft lands as often as the deepest so far) drafted levels the cut then
        dropped, and lost 2.6% to the fixed cut at 3 ms an MTP step; this estimate gained 2.0% there
        (tools/staircut_sim.py, simulated)."""

        k = len(probs)
        if k >= most:
            return False
        if not self.prices.known(cls):
            return True
        reach = self._reach(probs)
        price = self.prices.table(most + 1, cls)
        best = self._best(reach, price, cls)[0]
        gain, run, rate = 1.0 + sum(reach), reach[-1], self.cal.q(probs[-1])
        for m in range(1, most - k + 1):
            run *= rate
            gain += run
            if gain / max(self._cost(price, k + m, cls, k), 1e-9) > best:
                return True
        return False

    def learn(self, probs: Sequence[float], accepted: int) -> None:
        """A verified chain's drafts teach the calibration up to the first one that missed (later ones were not
        reached)."""

        for i, p in enumerate(probs):
            if i > accepted:
                break
            self.cal.learn(p, i < accepted)

    def observe(self, rows: int, verify_ms: float, steps: int, draft_ms: float, cls: int) -> None:
        """A round alone on the GPU: its verify of ``rows`` rows and the ``steps`` MTP steps of its draft."""

        self.prices.observe(rows, verify_ms, cls)
        if steps > 0 and draft_ms > 0.0:
            ms = draft_ms / steps
            old = self.step.get(cls)
            if old is None or ms <= self.OUTLIER * old:
                self.step[cls] = ms if old is None else (1 - self.EMA) * old + self.EMA * ms


class PlainGate:
    """Plain rounds in a context class while its drafted rounds cost more a committed token than its plain ones."""

    PROBE = 16
    PROBE_MAX = 256
    SPEC_EMA = 0.05       # a drafted round's weight: ms and tokens are separate EMAs, and their ratio is the cost
    PLAIN_EMA = 0.1
    OUTLIER = 4.0
    FIRST = 16            # a class's first drafted rounds are summed, not averaged by EMA, and the gate waits for them:
                          # one slow first round (a 1-token round, a cold kernel) would otherwise send the class plain,
                          # where 5% a probe takes hundreds of rounds to bring it back
    MARGIN = 0.1          # drafting must cost 10% more a token than the plain step before a class goes plain: the
                          # EMA of rounds that commit 1 to 7 tokens wanders, and a class that drafted 13% cheaper than
                          # plain went plain at the first excursion (tools/staircut_sim.py, hard text at 32k)

    def __init__(self, plain_prior: Callable[[int], float | None] | None = None) -> None:
        self.plain_prior = plain_prior or (lambda cls: None)
        self.spec: dict[int, list[float]] = {}     # class -> [EMA of a drafted round's ms, EMA of its tokens, rounds]
        self.plain: dict[int, float] = {}          # class -> EMA of a plain round's ms
        self.skipped: dict[int, int] = {}
        self.probe: dict[int, int] = {}
        self.since: dict[int, int] = {}            # class -> rounds since its plain step was last measured

    def spec_ms(self, cls: int) -> float | None:
        m = self.spec.get(cls)
        return None if m is None else m[0] / m[1]

    def plain_ms(self, cls: int) -> float | None:
        got = self.plain.get(cls)
        return got if got is not None else self.plain_prior(cls)

    def skip(self, cls: int, measure: bool = True) -> bool:
        """True: this round takes the plain step. A class whose drafted rounds lose still drafts one round in its
        probe interval; every probe that loses doubles the interval, and a win starts it over. ``measure``: the
        round will be timed, so it may be the one that measures the plain step (a shared round never is, and forcing
        one would leave the class plain until a stream runs alone)."""

        spec, plain = self.spec_ms(cls), self.plain_ms(cls)
        if spec is not None and self.spec[cls][2] >= self.FIRST:
            self.since[cls] = self.since.get(cls, 0) + 1
            if measure and (cls not in self.plain or self.since[cls] > self.PROBE_MAX):
                self.since[cls] = 0                # measure the plain step: once, then every PROBE_MAX rounds
                return True
        if spec is None or plain is None or spec <= plain * (1 + self.MARGIN) or self.spec[cls][2] < self.FIRST:
            self.probe[cls], self.skipped[cls] = self.PROBE, 0
            return False
        self.skipped[cls] = self.skipped.get(cls, 0) + 1
        interval = self.probe.get(cls, self.PROBE)
        if self.skipped[cls] < interval:
            return True
        self.skipped[cls], self.probe[cls] = 0, min(2 * interval, self.PROBE_MAX)
        return False

    def drafted(self, ms: float, tokens: int, cls: int) -> None:
        m = self.spec.get(cls)
        if m is None:
            self.spec[cls] = [ms, float(tokens), 1]
            return
        if m[2] < self.FIRST:
            m[0], m[1], m[2] = m[0] + ms, m[1] + tokens, m[2] + 1
            if m[2] == self.FIRST:                 # the sums become a round's means, which the EMA follows
                m[0], m[1] = m[0] / self.FIRST, m[1] / self.FIRST
            return
        if ms / max(tokens, 1) > self.OUTLIER * m[0] / m[1]:
            return
        a = self.SPEC_EMA
        m[0], m[1], m[2] = (1 - a) * m[0] + a * ms, (1 - a) * m[1] + a * tokens, m[2] + 1

    def plain_round(self, ms: float, cls: int) -> None:
        old = self.plain.get(cls)                  # the first measurement replaces the prior outright
        if old is not None and ms > self.OUTLIER * old:
            return
        self.plain[cls] = ms if old is None else (1 - self.PLAIN_EMA) * old + self.PLAIN_EMA * ms


class Speculation:
    """The decoders' one handle: ``plain`` (skip drafting), ``more`` and ``keep`` (the chain's depth), and the
    ``verified`` and ``plain_round`` reports. Without StairCut the chain keeps the fixed confidence rule."""

    def __init__(self, *, cut: bool, gate: bool, confidence: float, depth: int,
                 prior: dict[int, dict[int, float]] | None = None) -> None:
        self.prices = Prices(prior)
        self.cut = StairCut(self.prices) if cut else None
        self.gate = PlainGate(self._plain_prior) if gate else None
        self.confidence, self.depth = float(confidence), int(depth)
        self.stats = {"rounds": 0, "plain": 0}

    @classmethod
    def from_env(cls, *, confidence: float, depth: int, world: int = 1,
                 prior: dict[int, dict[int, float]] | None = None, log=print) -> Speculation | None:
        """None unless TENSORFOLD_STAIRCUT or TENSORFOLD_PLAIN_GATE is on (and the model drafts on one rank)."""

        want_cut, want_gate = flag(ENV_CUT), flag(ENV_GATE)
        if not (want_cut or want_gate) or depth < 1:
            return None
        if world != 1:
            log(f"[tensorfold] {ENV_CUT}/{ENV_GATE} ignored: two ranks must choose the same windows, and these "
                "rules follow this process's clocks")
            return None
        return cls(cut=want_cut, gate=want_gate, confidence=confidence, depth=depth, prior=prior)

    def _plain_prior(self, cls: int) -> float | None:
        """A plain round before any is measured: a one-row verify plus one MTP step (the absorb)."""

        if not self.prices.known(cls):
            return None
        step = self.cut.step.get(cls, 0.0) if self.cut is not None else 0.0
        return self.prices.table(1, cls)[1] + step

    def plain(self, context: int, *, measure: bool = True) -> bool:
        return self.gate is not None and self.gate.skip(context_class(context), measure)

    def more(self, context: int, probs: Sequence[float]) -> bool:
        if self.cut is not None:
            return self.cut.more(probs, context_class(context), self.depth)
        return len(probs) < self.depth and probs[-1] >= self.confidence

    def keep(self, context: int, probs: Sequence[float]) -> int:
        """Drafts to verify. The fixed rule keeps the first draft, then each next one at ``confidence`` or above."""

        if self.cut is not None:
            return self.cut.cut(probs, context_class(context))
        n = 1
        while n < len(probs) and probs[n] >= self.confidence:
            n += 1
        return min(n, len(probs))

    def verified(self, context: int, probs: Sequence[float], rows: int, accepted: int, *, timed: bool,
                 verify_ms: float = 0.0, draft_ms: float = 0.0, steps: int = 0) -> None:
        """A drafted round's outcome: ``probs`` of its MTP drafts (empty for copied ones), ``rows`` verified,
        ``accepted`` drafts kept. ``timed``: it ran alone, so its clock prices its rows."""

        cls = context_class(context)
        self.stats["rounds"] += 1
        if self.cut is not None:
            self.cut.learn(probs[:max(rows - 1, 0)], accepted)
        if not timed:
            return
        if self.cut is not None:
            self.cut.observe(rows, verify_ms, steps, draft_ms, cls)
        else:
            self.prices.observe(rows, verify_ms, cls)        # the gate's plain prior reads the one-row price
        if self.gate is not None:
            self.gate.drafted(verify_ms + draft_ms, accepted + 1, cls)

    def plain_round(self, context: int, ms: float, *, timed: bool) -> None:
        self.stats["plain"] += 1
        if timed and self.gate is not None:
            self.gate.plain_round(ms, context_class(context))
