"""Time LFM2.5-24B-A2B's expert kernels on prompt chunks and print the crossover the engine's PAIRS_ROWS should take.

  python tools/lfm2_moe_prompt_kernels.py --model /home/agis/models/LFM2.5-24B-A2B-FP8 [--prefill]

The first part loads the checkpoint's layers up to its first MoE layer (about 1 GB) and times one MoE layer's
experts at each chunk height on four paths: ``pairs`` (the decode kernel), ``dot`` at tile heights 16, 32 and 64, and
``grouped`` (the fp32 reference, TF_LFM2_PROMPT_KERNEL=fp32). Inputs are embedding rows of random tokens, routed by
the layer's own router. Each cell is the median of CUDA-event timings after warm-up; a prompt pays the cell once per
MoE layer, so the last columns scale the best path to the whole model. It prints PAIRS_ROWS (the tallest chunk at which
``pairs`` beats every ``dot`` height) and the fastest tile height per chunk, for ``tile_rows``.

``--prefill`` then loads the whole engine (about 30 GB) and times ``prefill`` end to end at 64, 256, 1k and 4k tokens
under each TF_LFM2_PROMPT_KERNEL value.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

ROWS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048)


def time_ms(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def layer_table(model: Path, reps: int) -> None:
    from triton.runtime.errors import OutOfResources

    from tensorfold.families.lfm2_moe.config import Config
    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tensorfold.families.lfm2_moe.cuda.weights import load

    full = Config.read(model)
    layers = len(full.layers) - full.dense_layers
    w = load(model, "cuda", layers=full.dense_layers + 1)
    c = w.config
    moe = w.blocks[c.dense_layers].moe
    ex = moe.experts
    work = torch.empty(ex.w13[0].numel(), dtype=torch.float32, device="cuda")
    g = torch.Generator().manual_seed(0)
    paths = {"pairs": lambda x, ids: X.pairs(x, ids, ex),
             "dot16": lambda x, ids: X.dot(x, ids, ex, 16),
             "dot32": lambda x, ids: X.dot(x, ids, ex, 32),
             "dot64": lambda x, ids: X.dot(x, ids, ex, 64),
             "fp32": lambda x, ids: X.grouped(x, ids, ex, work)}
    print(f"one MoE layer's experts ({ex.count} experts, top {c.top_k}, width {ex.width}), ms; {layers} MoE layers "
          f"a prompt")
    print(f"{'rows':>6} " + " ".join(f"{p:>8}" for p in paths) + f" {'best':>6} {'model ms':>9} {'tok/s':>8}")
    crossover, heights = 0, {}
    for rows in ROWS:
        tokens = torch.randint(1000, c.vocab, (rows,), generator=g).cuda()
        x = w.embed[tokens].contiguous()
        ids, _ = X.route(x, moe, c.top_k, c.scaling)
        cells = {}
        for name, fn in paths.items():
            try:
                cells[name] = time_ms(lambda fn=fn, x=x, ids=ids: fn(x, ids), reps)
            except OutOfResources as err:     # a tile height past the GPU's shared memory: report it, time the rest
                print(f"  {name} at {rows} rows failed: {type(err).__name__}: {str(err)[:120]}")
                cells[name] = float("inf")
        best = min(cells, key=cells.get)
        dots = {k: v for k, v in cells.items() if k.startswith("dot")}
        top = min(dots, key=dots.get)
        heights[rows] = int(top[3:])
        if cells["pairs"] <= dots[top] and crossover == rows_before(rows):
            crossover = rows
        model_ms = cells[best] * layers
        print(f"{rows:>6} " + " ".join(f"{cells[p]:8.3f}" for p in paths) + f" {best:>6} {model_ms:9.1f} "
              f"{rows / model_ms * 1000:8.0f}")
    print(f"\nPAIRS_ROWS = {crossover}   (pairs is at least as fast as every dot height up to this chunk height)")
    print("fastest dot tile height by chunk rows:", heights)


def rows_before(rows: int) -> int:
    """The previous row count in ROWS (0 before the first), so the crossover stops at the first chunk dot wins."""

    i = ROWS.index(rows)
    return ROWS[i - 1] if i else 0


def prefill_table(model: Path) -> None:
    import time

    from tensorfold.families.lfm2_moe.cuda.app import Lfm2MoeEngine
    from tensorfold.families.lfm2_moe.cuda.engine import prefill

    be = Lfm2MoeEngine(model, context=8192, context_explicit=True)
    e = be.e
    g = torch.Generator().manual_seed(0)
    print(f"\n{'tokens':>6} " + " ".join(f"{k:>14}" for k in ("fp32", "pairs", "dot", "auto")))
    for n in (64, 256, 1024, 4096):
        ids = torch.randint(1000, 120000, (n,), generator=g).tolist()
        cells = []
        for kernel in ("fp32", "pairs", "dot", "auto"):
            e.prompt_kernel = kernel
            best = float("inf")
            for _ in range(3):
                torch.cuda.synchronize()
                t = time.perf_counter()
                prefill(e, ids, None)
                torch.cuda.synchronize()
                best = min(best, time.perf_counter() - t)
            cells.append(f"{best:6.3f}s {n / best:6.0f}/s")
        print(f"{n:>6} " + " ".join(f"{c:>14}" for c in cells), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("/home/agis/models/LFM2.5-24B-A2B-FP8"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--prefill", action="store_true", help="also time whole prefills under each kernel choice")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    with torch.no_grad():
        layer_table(args.model, args.reps)
        torch.cuda.empty_cache()
        if args.prefill:
            prefill_table(args.model)


if __name__ == "__main__":
    main()
