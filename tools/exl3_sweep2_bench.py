"""Time Flash Next EXL3's prompt chunks and decode windows with the sweep's switches off, each alone on, and all on, on
one load, so a served A/B's difference can be split by switch.

Switches: TF_EXL3_EXPERT_ITEMS ("items"), TF_FLASHNEXT_STACK_DIRECT ("stack"), TF_FLASHNEXT_F16_INNER_K ("inner_k"),
TF_FLASHNEXT_MOE_DIRECT ("moe"), and TF_FLASHNEXT_HC_FUSE on the pack's fp16 hyper-connections ("hc"). The other
switches (TF_EXL3_PDL) keep their environment's setting. Per setting: a fresh prefill of a ``--prompt``-token prompt in 2,048-row chunks
(median of ``--reps`` after one untimed), and the forward of decode windows of 1, 4 and 7 rows from CUDA graphs
(serial and MTP verify) and 17 and 28 rows eager (four streams' windows past one 16-row tile), each the median of
``--reps`` after one untimed (``decode.time_windows``). The settings run off, each alone, all, off again; a setting's
last-row prompt logits are compared with the first off run's bytes. Prints one JSON line a (setting, measure).

  python tools/exl3_sweep2_bench.py /models/Qwen3.8-Flash-Next-exl3-3.05bpw [--prompt 8192] [--output out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch

from tensorfold.cuda.exl3 import switches
from tensorfold.families.qwen4_exp.cuda import exl3_mm

NAMES = ("items", "stack", "inner_k", "moe", "hc")


def _set(w, on: set[str]) -> None:
    switches.use(dataclasses.replace(switches.read(), expert_items="items" in on))
    exl3_mm.use(exl3_mm.Switches(stack_direct="stack" in on, inner_k="inner_k" in on, moe_direct="moe" in on,
                                 moe_chunk=False))
    hcs = [w.mixer] + [hc for layer in w.layers for hc in (layer.attn_hc, layer.mlp_hc)]
    if w.mtp is not None:
        hcs += [w.mtp.mixer, w.mtp.layer.attn_hc, w.mtp.layer.mlp_hc]
    for hc in hcs:
        hc.fuse = "hc" in on


def _prefill_ms(w, prompt: list[int], reps: int) -> tuple[float, torch.Tensor]:
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill_begin, prefill_chunk

    times, last = [], None
    for i in range(reps + 1):
        e = Engine(w, capacity=len(prompt) + 64, max_rows=8, prefill_rows=2048, graphs=False)
        torch.cuda.synchronize()
        t = time.perf_counter()
        start = prefill_begin(e, prompt)
        while start < len(prompt):
            last, _ = prefill_chunk(e, prompt, start)
            start = min(start + e.prefill_rows, len(prompt))
        torch.cuda.synchronize()
        if i:
            times.append((time.perf_counter() - t) * 1e3)
        last = last.clone()
        del e
    return sorted(times)[len(times) // 2], last


def _windows_ms(w, reps: int) -> dict[str, float]:
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, time_windows

    def timed(rows: int, graphs: bool, widths: tuple[int, ...]) -> dict[int, float]:
        e = Engine(w, capacity=4096, max_rows=rows, prefill_rows=2048, graphs=graphs)
        prefill(e, list(range(1000, 1600)), None)          # 600 tokens of context
        return time_windows(lambda r: e.forward([7] * r), widths, reps)

    out = {f"window {r} graph": ms for r, ms in timed(8, True, (1, 4, 7)).items()}
    out.update({f"window {r} eager": ms for r, ms in timed(28, False, (17, 28)).items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prompt", type=int, default=8192)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--output")
    a = p.parse_args()
    from tensorfold.families.qwen4_exp.cuda.weights import load

    w = load(a.model, mtp=True, draft_vocab=None, hc_fuse=True)
    prompt = torch.randint(10, 150_000, (a.prompt,), generator=torch.Generator().manual_seed(9)).tolist()
    settings = [("off", set())] + [(n, {n}) for n in NAMES] + [("all", set(NAMES)), ("off again", set())]
    rows, first = [], None
    for label, on in settings:
        _set(w, on)
        ms, last = _prefill_ms(w, prompt, a.reps)
        first = last if first is None else first
        same = bool(torch.equal(last.view(torch.int16), first.view(torch.int16)))
        got = [{"setting": label, "measure": f"prefill {a.prompt}", "ms": round(ms, 2),
                "tok_s": round(a.prompt / ms * 1e3, 1), "same_logits": same}]
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
