"""Time TF_EXL3_GROUP_SCAN_DECODE ("scan") and TF_EXL3_LINEAR_PAIR ("pair") on Flash Next EXL3, each alone, so a
served A/B's difference can be split by switch. The other EXL3 switches and the pack's (TF_FLASHNEXT_MOE_CHUNK and
the rest) keep their environment's setting, which unset is fba05ef2's defaults.

Kernels (random weights at the pack's shapes, CUDA events, the median of ``--reps`` timed launches after warm-up):
the grouping of 1, 5, 7, 16, 28 and 1,024 rows (11 slots, 513 experts) on group_kernel and group_scan_kernel (the
1,024-row window is a prompt's, for scale), and the
EXL3 linear at the dense shapes (DeltaNet qkv, z, out; attention q, k/v, o; the indexer; lm_head) at 1, 5, 7, 16,
17 and 28 rows with one and two row tiles a pass. Each pair of launches is compared byte for byte.

Model (one load): per setting (off, scan, pair, all, off again), the forward of verify windows of 1, 5 and 7 rows
from CUDA graphs (one stream's serial and MTP rounds) and 16 and 28 rows eager (four streams' windows: the
concurrent decoder runs without graphs), each the median of ``--reps`` after one untimed (``decode.time_windows``),
then a fresh prefill of a ``--prompt``-token prompt in 2,048-row chunks whose last-row logits are compared with the
first off run's bytes. Prints one JSON line a measurement.

  python tools/exl3_decode2_bench.py /models/Qwen3.8-Flash-Next-exl3-3.05bpw [--prompt 8192] [--output out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import torch

from tensorfold.cuda.exl3 import experts, linear, switches

SETTINGS = [("off", set()), ("scan", {"scan"}), ("pair", {"pair"}), ("all", {"scan", "pair"}),
            ("off again", set())]
GROUP_ROWS = (1, 5, 7, 16, 28, 1024)
WINDOW_ROWS = (1, 5, 7, 16, 28)
LINEAR_ROWS = (1, 5, 7, 16, 17, 28)
# (name, K, N, bits): Flash Next's dense trellis projections and the head
DENSE = [("deltanet qkv", 2560, 10240, 5), ("deltanet z", 2560, 6144, 5), ("out/o", 6144, 2560, 5),
         ("attention q", 2560, 12288, 5), ("attention k/v", 2560, 512, 5), ("indexer", 2560, 640, 3),
         ("lm_head", 2560, 248320, 5)]


def _set(on: set[str]) -> None:
    switches.use(dataclasses.replace(switches.read(), group_scan_decode="scan" in on, linear_pair="pair" in on))


def _event_us(fn, reps: int) -> float:
    """Median microseconds of ``fn`` over ``reps`` launches, each between two events, after three untimed."""

    for _ in range(3):
        fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return round(sorted(times)[len(times) // 2], 2)


def _groups(reps: int) -> list[dict]:
    slots, E = 11, 513
    ext, rows = experts._ext(), []
    g = torch.Generator().manual_seed(5)
    for R in GROUP_ROWS:
        routed = torch.stack([torch.randperm(E - 1, generator=g)[:slots - 1] for _ in range(R)])
        pick = torch.cat([routed, torch.full((R, 1), E - 1)], 1).to(torch.int32).cuda()   # the shared expert last
        maxu = min(R * slots, E)
        out = {}
        for scan in (False, True):
            ids = torch.zeros((maxu,), dtype=torch.int32, device="cuda")
            count = torch.zeros((1,), dtype=torch.int32, device="cuda")
            members = torch.full((maxu, R), -1, dtype=torch.int32, device="cuda")

            def launch(pick=pick, ids=ids, count=count, members=members, R=R, scan=scan):
                ext.group(pick, ids, count, members, R, slots, E, False, scan=scan)

            us = _event_us(launch, reps)
            out[scan] = (us, ids, count, members)
        same = all(torch.equal(a, b) for a, b in zip(out[False][1:], out[True][1:]))
        rows.append({"kernel": "group", "rows": R, "group_kernel_us": out[False][0],
                     "group_scan_kernel_us": out[True][0], "speedup": round(out[False][0] / out[True][0], 3),
                     "same_bytes": same})
    return rows


def _linears(reps: int) -> list[dict]:
    rows = []
    g = torch.Generator().manual_seed(9)
    for name, k, n, bits in DENSE:
        tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int32, generator=g).to(torch.int16)
        su = ((torch.rand(k, generator=g) + 0.5) / math.sqrt(k)).half()
        sv = (torch.rand(n, generator=g) + 0.5).half()
        lin = linear.Exl3Linear.from_tensors(tr, su, sv, "mul1")
        del tr
        mb = lin.nbytes() / 1e6
        for m in LINEAR_ROWS:
            x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
            out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            xh = torch.empty((m, k), dtype=torch.float16, device="cuda")
            z = torch.empty((lin.split[0] * m * n,), dtype=torch.float32, device="cuda")
            got = {}
            for pair in (False, True):
                _set({"pair"} if pair else set())
                us = _event_us(lambda lin=lin, x=x, out=out, xh=xh, z=z: lin(x, out=out, xh=xh, z=z), reps)
                got[pair] = (us, out.clone())
            same = torch.equal(got[False][1].view(torch.int16), got[True][1].view(torch.int16))
            rows.append({"kernel": f"linear {name}", "rows": m, "mb": round(mb, 1), "one_tile_us": got[False][0],
                         "two_tiles_us": got[True][0], "speedup": round(got[False][0] / got[True][0], 3),
                         "gbs_one_tile": round(mb / got[False][0] * 1e3, 1), "same_bytes": same})
        del lin
        torch.cuda.empty_cache()
    _set(set())
    return rows


def _windows_ms(w, reps: int) -> dict[str, float]:
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, time_windows

    def timed(rows: int, graphs: bool, widths: tuple[int, ...]) -> dict[int, float]:
        e = Engine(w, capacity=4096, max_rows=rows, prefill_rows=2048, graphs=graphs)
        prefill(e, list(range(1000, 1600)), None)          # 600 tokens of context
        return time_windows(lambda r: e.forward([7] * r), widths, reps)

    graphed = tuple(r for r in WINDOW_ROWS if r <= 8)
    out = {f"window {r} graph": ms for r, ms in timed(8, True, graphed).items()}
    eager = tuple(r for r in WINDOW_ROWS if r > 8)
    out.update({f"window {r} eager": ms for r, ms in timed(max(eager), False, eager).items()})
    return out


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prompt", type=int, default=8192)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--kernel-reps", type=int, default=50)
    p.add_argument("--output")
    a = p.parse_args()
    rows = []

    def emit(got: list[dict]) -> None:
        for r in got:
            print(json.dumps(r), flush=True)
        rows.extend(got)

    emit(_groups(a.kernel_reps))
    emit(_linears(a.kernel_reps))
    torch.cuda.empty_cache()

    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.read())                     # the pack's switches as the server reads them, before the load
    w = load(a.model, mtp=True, draft_vocab=None)
    prompt = torch.randint(10, 150_000, (a.prompt,), generator=torch.Generator().manual_seed(9)).tolist()
    first = None
    for label, on in SETTINGS:
        _set(on)
        got = [{"setting": label, "measure": k, "ms": v} for k, v in _windows_ms(w, a.reps).items()]
        ms, last = _prefill_ms(w, prompt, max(1, a.reps // 2))
        first = last if first is None else first
        got.append({"setting": label, "measure": f"prefill {a.prompt}", "ms": round(ms, 2),
                    "tok_s": round(a.prompt / ms * 1e3, 1),
                    "same_logits": bool(torch.equal(last.view(torch.int16), first.view(torch.int16)))})
        emit(got)
        torch.cuda.empty_cache()
    if a.output:
        with open(a.output, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
