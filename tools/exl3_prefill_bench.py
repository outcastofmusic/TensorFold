"""Time Flash Next EXL3's prompt chunks and decode windows with the prompt-window switches off, each alone on, and all
on, on one load, so a served A/B's difference can be split by switch.

Switches: TF_EXL3_GROUP_SCAN ("scan"), TF_EXL3_DOWN_FUSE ("down"), TF_EXL3_GATEUP_FUSE ("gateup"),
TF_FLASHNEXT_MOE_CHUNK ("chunk"), TF_EXL3_EXPERT_STAGED ("staged") and TF_EXL3_ROT_FUSE ("rot"); ``--names`` picks the
ones swept (the first four unless given). Every other switch keeps its environment's setting (today's defaults when
unset). The pack loads with the chunk window on, so its routed scratch holds 2,048 rows and every setting runs on the
one load. Per setting: a fresh prefill of each ``--prompts`` length in 2,048-row chunks (median of ``--reps`` after one
untimed), and decode windows of 1, 4 and 7 rows from CUDA graphs and 17 and 28 rows eager (median of ``--reps``), which
no switch should move. The settings run off, each alone, all, off again; a setting's last-row prompt logits are compared
with the first off run's bytes. Prints one JSON line a (setting, measure).

  python tools/exl3_prefill_bench.py /models/Qwen3.8-Flash-Next-exl3-3.05bpw [--prompts 8192,32768] [--names staged]
      [--output out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import torch

from tensorfold.cuda.exl3 import switches
from tensorfold.families.qwen4_exp.cuda import exl3_mm
from tools.exl3_sweep2_bench import _prefill_ms, _windows_ms

NAMES = ("scan", "down", "gateup", "chunk")
FIELDS = {"scan": "group_scan", "down": "down_fuse", "gateup": "gateup_fuse", "staged": "expert_staged",
          "rot": "rot_fuse"}
EXTRA = ("staged", "rot")      # sweepable by --names, not swept by default


def _set(base_sw: switches.Switches, base_mm: exl3_mm.Switches, on: set[str], names=NAMES) -> None:
    """The swept ``names`` on when in ``on`` and off otherwise; every other switch keeps the base setting."""

    switches.use(dataclasses.replace(base_sw, **{FIELDS[n]: n in on for n in names if n in FIELDS}))
    exl3_mm.use(dataclasses.replace(base_mm, **({"moe_chunk": "chunk" in on} if "chunk" in names else {})))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prompts", default="8192")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--no-windows", action="store_true", help="skip the decode windows")
    p.add_argument("--names", default=",".join(NAMES), help="the switches swept, of " + ", ".join(NAMES + EXTRA))
    p.add_argument("--output")
    a = p.parse_args()
    from tensorfold.families.qwen4_exp.cuda.weights import load

    base_sw, base_mm = switches.read(), exl3_mm.read()
    exl3_mm.use(dataclasses.replace(base_mm, moe_chunk=True))       # the routed scratch for 2,048-row calls
    w = load(a.model, mtp=True, draft_vocab=None, hc_fuse=True)
    lengths = [int(n) for n in a.prompts.split(",")]
    names = tuple(a.names.split(","))
    if not set(names) <= set(NAMES) | set(EXTRA):
        raise SystemExit(f"--names: unknown switch in {a.names!r}")
    prompts = {n: torch.randint(10, 150_000, (n,), generator=torch.Generator().manual_seed(9)).tolist()
               for n in lengths}
    settings = [("off", set())] + [(n, {n}) for n in names] + [("all", set(names)), ("off again", set())]
    rows, first = [], {}
    for label, on in settings:
        _set(base_sw, base_mm, on, names)
        got = []
        for n, prompt in prompts.items():
            ms, last = _prefill_ms(w, prompt, a.reps)
            first.setdefault(n, last)
            same = bool(torch.equal(last.view(torch.int16), first[n].view(torch.int16)))
            got.append({"setting": label, "measure": f"prefill {n}", "ms": round(ms, 2),
                        "tok_s": round(n / ms * 1e3, 1), "same_logits": same})
        if not a.no_windows:
            got += [{"setting": label, "measure": k, "ms": v} for k, v in _windows_ms(w, a.reps).items()]
        for r in got:
            print(json.dumps(r), flush=True)
        rows += got
        torch.cuda.empty_cache()
    if a.output:
        with open(a.output, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
