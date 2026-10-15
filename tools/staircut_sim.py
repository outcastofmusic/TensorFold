"""Simulate Flash Next decoding under the fixed 0.7 confidence cut, StairCut and the plain gate, on synthetic traces.

No GPU, no model: every number this prints is simulated. The trace gives each reply position an MTP chain (the head's
probability of each of 6 drafts and whether that draft lands); every policy decodes the same trace, so the policies
differ only in the rows they verify. A round costs the verify price of its rows plus one MTP step per draft (the
absorb counts as one), with +-5% noise.

Inputs, all documented elsewhere:
- verify prices: TandemLLM's ``ops/prices/qwen3.8-flash-next/flashnext-3.05.dgx-spark.json`` (2026-10-12, PROVISIONAL),
  ms of a round past its draft by verified rows and context class, and the plain step's ms per class. Copied here as
  numbers (facts, not code).
- acceptance: the live tensorfold-serve counters on 2026-10-14 (EXL3 3.05bpw, the 0.7 cut, depth 6): 198,270 of
  278,764 drafts accepted (71.1%), 294,492 tokens, so about 3.06 tokens a round. The trace's kinds, depth decay and
  head calibration (a draft of probability p lands with chance p ** 0.7: an underconfident head) were fitted by grid
  search so the fixed 0.7 rule reproduces them: 0.705 accepted, 3.12 tokens and 3.01 drafts a round
  (``--check`` prints the fit).
- MTP step ms: not measured for Flash Next; ``--step`` sweeps it (default 1.5 ms).

Usage: python3 tools/staircut_sim.py [--tokens 40000] [--step 1.5] [--seed 1] [--check]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfold.families.qwen4_exp.cuda.staircut import Speculation, context_class  # noqa: E402

DEPTH = 6
# ms by verified rows (TandemLLM price file); rows past the last listed add 2.8 ms a row (its step_ms)
CHAIN = {0: {1: 24.0, 2: 27.8, 3: 31.1, 4: 34.7, 5: 38.6, 6: 42.3, 7: 43.8},
         32768: {2: 33.1, 3: 38.6, 4: 45.7, 5: 46.4},
         131072: {2: 44.0, 3: 49.4, 4: 55.8, 5: 59.7},
         262144: {2: 48.0, 3: 53.0, 4: 58.0, 5: 63.0}}
PLAIN = {0: 22.0, 32768: 23.5, 131072: 24.5, 262144: 24.7}
STEP_ROW = 2.8

# text kinds: (share of the reply, Beta(a, b) of a position's predictability)
KINDS = {"prose": (0.45, (3.3, 1.6)), "code": (0.30, (5.25, 1.0)), "json": (0.15, (10.5, 0.8)),
         "hard": (0.10, (1.35, 2.0))}
SEGMENT = 160          # tokens of one kind before the next kind starts
DECAY = 0.96           # a deeper draft's probability: the level times U(DECAY, 1) per depth
GAMMA = 0.7            # a draft of head probability p lands with chance p ** GAMMA


def true_price(rows: int, cls: int) -> float:
    table = CHAIN[cls]
    if rows in table:
        return table[rows]
    known = sorted(table)
    if rows < known[0]:                       # a class without a 1-row price: the plain step plus the 2-to-3 slope
        return PLAIN[cls] + (table[known[1]] - table[known[0]]) * 0.5
    lo = max(r for r in known if r < rows) if rows < known[-1] else known[-1]
    if rows > known[-1]:
        return table[lo] + STEP_ROW * (rows - lo)
    hi = min(r for r in known if r > rows)
    return table[lo] + (table[hi] - table[lo]) * (rows - lo) / (hi - lo)


class Trace:
    """Per position: the head's probability for each of the next DEPTH drafts and whether each lands."""

    def __init__(self, n: int, seed: int, only: str | None = None):
        rng = random.Random(seed)
        names = list(KINDS)
        weights = [KINDS[k][0] for k in names]
        self.kind, self.chain = [], []
        while len(self.kind) < n + DEPTH + 2:
            kind = only or rng.choices(names, weights)[0]
            a, b = KINDS[kind][1]
            for _ in range(SEGMENT):
                self.kind.append(kind)
                level = rng.betavariate(a, b)
                probs, lands = [], []
                for k in range(DEPTH):
                    p = min(0.999, max(0.001, level * rng.uniform(DECAY, 1.0) ** (k + 1) + rng.gauss(0, 0.05)))
                    probs.append(p)
                    lands.append(rng.random() < p ** GAMMA)    # GAMMA < 1: the head is underconfident
                self.chain.append((probs, lands))


def simulate(trace: Trace, tokens: int, policy: str, cls: int, step: float, seed: int) -> dict:
    rng = random.Random(seed)
    spec = None
    if policy != "fixed":
        spec = Speculation(cut="cut" in policy, gate="gate" in policy, confidence=0.7, depth=DEPTH,
                           # the warm-up prior times the forward alone: 80% of a round
                           prior={0: {r: 0.8 * true_price(r, 0) for r in range(1, DEPTH + 2)}})
    context = {0: 2000, 32768: 30000, 131072: 140000, 262144: 260800}[cls]
    assert context_class(context) >= cls
    pos, ms = 0, 0.0
    rounds = drafted = accepted = 0
    plain_rounds = 0
    while pos < tokens:
        probs, lands = trace.chain[pos]
        if spec is None:                     # decode.draft: the first draft always, then each next one at 0.7
            n, steps = 1, 1                  # or more; the draft under 0.7 that ends the chain was drafted too
            if probs[0] >= 0.7:
                while n < DEPTH and probs[n] >= 0.7:
                    n += 1
                steps = n if n == DEPTH else n + 1
        elif spec.plain(context):
            n, steps = 0, 1
        else:
            seen = []
            for p in probs:
                seen.append(p)
                if not spec.more(context, seen):
                    break
            n, steps = spec.keep(context, seen), len(seen)
        acc = 0
        while acc < n and lands[acc]:
            acc += 1
        rows = n + 1
        noise = rng.uniform(0.95, 1.05)
        verify = (PLAIN[cls] if rows == 1 else true_price(rows, cls)) * noise
        draft = step * steps * rng.uniform(0.95, 1.05)
        ms += verify + draft
        if spec is not None:
            if rows == 1:
                spec.plain_round(context, verify + draft, timed=True)
            else:
                spec.verified(context, seen[:n], rows, acc, timed=True, verify_ms=verify, draft_ms=draft, steps=steps)
        rounds += 1
        drafted += n
        accepted += acc
        plain_rounds += rows == 1
        pos += acc + 1
    return {"tok_s": 1000.0 * pos / ms, "tokens_round": pos / rounds, "accept": accepted / max(drafted, 1),
            "drafts_round": drafted / rounds, "plain": plain_rounds / rounds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", type=int, default=40000)
    ap.add_argument("--step", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=GAMMA, help="a draft of probability p lands with chance p**gamma")
    ap.add_argument("--check", action="store_true", help="print the fixed rule's fit to the live counters")
    args = ap.parse_args()
    globals()["GAMMA"] = args.gamma
    policies = ["fixed", "cut", "gate", "cut+gate"]
    print(f"SIMULATED, not measured. {args.tokens} tokens, MTP step {args.step} ms, gamma {args.gamma}, "
          f"seed {args.seed}")
    if args.check:
        r = simulate(Trace(args.tokens, args.seed), args.tokens, "fixed", 0, args.step, args.seed)
        print(f"fixed 0.7 rule on the mixed trace: accept {r['accept']:.3f} (live 0.711), "
              f"tokens/round {r['tokens_round']:.2f} (live 3.06), drafts/round {r['drafts_round']:.2f} (live 2.90)")
    print(f"{'trace':8} {'class':>7} " + " ".join(f"{p:>16}" for p in policies))
    for kind in [None, *KINDS]:
        trace = Trace(args.tokens, args.seed, only=kind)
        for cls in CHAIN:
            got = {p: simulate(trace, args.tokens, p, cls, args.step, args.seed) for p in policies}
            base = got["fixed"]["tok_s"]
            cells = [f"{got['fixed']['tok_s']:7.1f} tok/s   "]
            for p in policies[1:]:
                r = got[p]
                cells.append(f"{r['tok_s']:6.1f} {100 * (r['tok_s'] / base - 1):+5.1f}%  ")
            print(f"{kind or 'mixed':8} {cls:>7} " + " ".join(f"{c:>16}" for c in cells))
    print("plain-step ceiling by class (tok/s): "
          + ", ".join(f"{c}: {1000.0 / (PLAIN[c] + args.step):.1f}" for c in PLAIN))


if __name__ == "__main__":
    main()
