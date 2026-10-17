"""How many LiLiCorr drafts a Bolt round verifies: the depth with the most expected tokens a millisecond.

The model. A round's drafts are accepted in order while they match the target's samples, each with one probability a
given the ones before it, so a round of d drafts yields E(d) = 1 + a + ... + a^d tokens. It costs
T(d) = V(1 + d) + B [d > 0] + O: the verify window of 1 + d rows, the drafter's block pass (with the tap rows it still
owes), and the round's other host work. A round takes argmax E(d) / T(d) over d = 0 .. most, the smallest on a tie;
d = 0 is a serial round, which runs no drafter. Depth changes which rows a window verifies, never a token: a window
row gets the serial step's bits.

The measurements. ``propose`` returns on the host (its candidates come back to the CPU) and ``tokens()`` waits for the
verify window, so the clock around them times B and V(rows) with no extra synchronization; O is the rest of the
round. V, B and O belong to the GPU and the model, so they persist across requests (``Costs``). The acceptance rate
is the request's own: a = hits / (hits + misses) over exponential averages of a round's accepted drafts and of
whether it rejected one, the geometric model's maximum-likelihood rate when a round may end at its depth. A serial
stretch runs the best positive depth every ``probe`` rounds, so a reply that turns predictable drafts again.
"""

from __future__ import annotations


class Costs:
    """Exponential averages of a round's measured ms: verify windows by rows, the drafter's block pass, the rest."""

    def __init__(self, rows: int, weight: float = 0.2) -> None:
        self.verify: list[float | None] = [None] * (rows + 1)       # index = window rows; 0 unused
        self.block: float | None = None
        self.rest: float | None = None
        self.weight = float(weight)

    def _avg(self, old: float | None, new: float) -> float:
        return new if old is None else old + self.weight * (new - old)

    def record(self, rows: int, verify_ms: float, block_ms: float | None, rest_ms: float) -> None:
        self.verify[rows] = self._avg(self.verify[rows], verify_ms)
        if block_ms is not None:
            self.block = self._avg(self.block, block_ms)
        self.rest = self._avg(self.rest, rest_ms)

    def window(self, rows: int) -> float | None:
        """V(rows): its average once measured, else the least-squares line through the measured rows' averages."""

        if self.verify[rows] is not None:
            return self.verify[rows]
        seen = [(r, v) for r, v in enumerate(self.verify) if v is not None]
        if len(seen) < 2:
            return None
        mr = sum(r for r, _ in seen) / len(seen)
        mv = sum(v for _, v in seen) / len(seen)
        slope = sum((r - mr) * (v - mv) for r, v in seen) / sum((r - mr) ** 2 for r, _ in seen)
        return max(mv + slope * (rows - mr), 0.0)


class DepthPolicy:
    """One request's drafts a round, from its acceptance and the shared ``Costs``."""

    def __init__(self, most: int, costs: Costs, *, prior: float = 0.8, weight: float = 0.25, probe: int = 16) -> None:
        if len(costs.verify) < most + 2:
            raise ValueError(f"{most} drafts need window costs to {most + 1} rows")
        self.most, self.costs, self.weight, self.probe = int(most), costs, float(weight), int(probe)
        self.hits, self.misses = prior / (1.0 - prior), 1.0
        self.serial = 0                        # serial rounds since the last drafted one

    @property
    def rate(self) -> float:
        return self.hits / (self.hits + self.misses)

    def expected(self, d: int) -> float:
        a = self.rate
        return sum(a ** j for j in range(d + 1))

    def ms(self, d: int) -> float | None:
        """T(d) by the measured costs, None while one of its parts is unmeasured."""

        c = self.costs
        v = c.window(1 + d)
        if v is None or (d and c.block is None):
            return None
        return v + (c.block if d else 0.0) + (c.rest or 0.0)

    def next(self, room: int) -> int:
        """Drafts for the next round, at most ``room``."""

        top = min(self.most, int(room))
        if top <= 0:
            return 0
        c = self.costs
        if c.block is None:                    # nothing drafted measured yet: draft as deep as allowed
            return top
        if c.verify[1] is None:                # one serial round measures what drafting is judged against
            return 0
        speed = {d: self.expected(d) / t for d in range(top + 1) if (t := self.ms(d))}
        best = max(speed, key=speed.get)
        if best:
            self.serial = 0
            return best
        self.serial += 1
        if self.serial < self.probe or len(speed) < 2:
            return 0
        self.serial = 0
        return max((d for d in speed if d), key=speed.get)

    def done(self, drafts: int, accepted: int, verify_ms: float, block_ms: float, rest_ms: float) -> None:
        """Follow a round: ``drafts`` verified, ``accepted`` of them kept, and its measured parts."""

        self.costs.record(1 + drafts, verify_ms, block_ms if drafts else None, rest_ms)
        if drafts:
            w = self.weight
            self.hits += w * (accepted - self.hits)
            self.misses += w * (float(accepted < drafts) - self.misses)
