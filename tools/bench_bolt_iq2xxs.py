"""Time Bolt's IQ2_XXS kernels on decode-shaped and prefill-shaped calls, with GB/s over the weight bytes they read.

Decode: 9 rows over 9 distinct experts of 496 (one token's routed experts), up (N 2560, K 1024) and down (N 1024,
K 2560). Every timed call picks a fresh set of experts, so the weights come from DRAM, not from the previous call's L2.
Verify: 4 and 16 tokens of 9 rows in one call. Prefill: 4096 tokens x 9 experts = 36,864 rows over 496 experts.
Run inside the serving image on the GPU:

    python tools/bench_bolt_iq2xxs.py [--experts 496] [--tokens 4096]
"""

from __future__ import annotations

import argparse

import torch

from tensorfold.families.nemotron_bolt.cuda import iq2xxs


def packed(experts: int, n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(0, 256, (experts, n, k // 256, 66), generator=g, device="cuda", dtype=torch.uint8)
    d = (torch.rand((experts, n, k // 256), generator=g, device="cuda") * 0.02 + 0.001).half()
    w[..., :2] = d.unsqueeze(-1).view(torch.uint8)
    return w.reshape(experts, n, k // 256 * 66)


def timed(fn, calls: int, graph: bool = False) -> float:
    """Mean seconds a call over ``calls`` after a warm-up, on CUDA events; ``graph`` replays the calls as one CUDA
    graph, so the time is the GPU's and not Python's launch rate (the engine replays decode steps as graphs too)."""

    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    def run() -> None:
        for i in range(calls):
            fn(i)

    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        g.replay()
        torch.cuda.synchronize()
        run = g.replay
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / calls


def bench(name: str, n: int, k: int, args: argparse.Namespace) -> float:
    """Print one matrix's decode and prefill lines; return its decode seconds a call."""

    e, topk = args.experts, args.topk
    w = packed(e, n, k, seed=n)
    row_bytes = k // 256 * 66
    g = torch.Generator(device="cuda").manual_seed(1)
    picks = torch.stack([torch.randperm(e, generator=g, device="cuda")[:topk] for _ in range(args.calls + 3)])
    picks = picks.to(torch.int32)
    x = torch.randn((topk, k), device="cuda").to(torch.bfloat16)
    s = timed(lambda i: iq2xxs.iq2xxs_grouped(x, w, picks[i]), args.calls, graph=True)
    eager = timed(lambda i: iq2xxs.iq2xxs_grouped(x, w, picks[i]), args.calls)
    read = topk * n * row_bytes
    print(f"decode {name:4s} R={topk} N={n} K={k}: {s * 1e6:7.1f} us, {read / s / 1e9:6.1f} GB/s "
          f"({read / 1e6:.2f} MB of weights; {eager * 1e6:.1f} us a call launched eagerly from Python)")
    for tokens in (4, 16):                                         # a drafter's verify window, one call
        vids = torch.cat([torch.randperm(e, generator=g, device="cuda")[:topk] for _ in range(tokens)]).to(torch.int32)
        vx = torch.randn((tokens * topk, k), device="cuda").to(torch.bfloat16)
        v = timed(lambda i, vx=vx, vids=vids: iq2xxs.iq2xxs_grouped(vx, w, vids), 50, graph=True)
        vread = int(torch.unique(vids).numel()) * n * row_bytes
        print(f"verify {name:4s} {tokens:2d} tokens, R={tokens * topk}: {v * 1e6:7.1f} us, {vread / v / 1e9:6.1f} GB/s "
              f"({vread / 1e6:.1f} MB of weights)")
    xs = torch.randn((args.tokens * topk, k), device="cuda").to(torch.bfloat16)
    ids = torch.stack([torch.randperm(e, generator=g, device="cuda")[:topk] for _ in range(args.tokens)])
    ids = ids.reshape(-1).to(torch.int32)
    used = int(torch.unique(ids).numel())
    p = timed(lambda i: iq2xxs.iq2xxs_grouped(xs, w, ids), 10)
    flops = 2 * xs.shape[0] * n * k
    print(f"prefill {name:4s} R={xs.shape[0]} N={n} K={k}: {p * 1e3:7.2f} ms, {flops / p / 1e12:5.1f} TFLOP/s, "
          f"{used * n * row_bytes / p / 1e9:6.1f} GB/s over {used} experts' weights read once, "
          f"{args.tokens / p:8.0f} tokens/s for this matrix alone")
    dense = torch.randn((n, k), device="cuda").to(torch.bfloat16)
    d = timed(lambda i: xs @ dense.T, 10)
    print(f"  (cuBLAS bf16 dense {xs.shape[0]} x {n} x {k}: {d * 1e3:.2f} ms, {flops / d / 1e12:.1f} TFLOP/s)")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=496)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=9)
    ap.add_argument("--calls", type=int, default=200)
    args = ap.parse_args()
    print(f"GPU {torch.cuda.get_device_name()}, {args.experts} experts, top-{args.topk}")
    up = bench("up", 2560, 1024, args)
    torch.cuda.empty_cache()
    down = bench("down", 1024, 2560, args)
    both = up + down
    print(f"decode: up + down {both * 1e6:.1f} us a layer; 40 MoE layers of these alone: {1 / (40 * both):.0f} tok/s")


if __name__ == "__main__":
    main()
