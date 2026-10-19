"""Time the routed-expert projections of a decode window on grouped_kernel and on each decode_kernel tile setting.

Synthetic layers at Flash Next EXL3's shapes (2560 <-> 640, 3-bit mul1 experts and a 5-bit shared expert, 11 slots)
and GLM-5.3's (4096 <-> 2048, 2.25-bit mcg, 8 slots). Each launch is captured 50 times in one CUDA graph and timed
over 20 replays; the GB/s counts the distinct experts' trellis bytes (a projection's share taken as 2/3 and
1/3 of an expert's, exact when its three matrices share a width). Every setting's partial sums are checked equal
to grouped_kernel's first. Prints one JSON line a (model, rows, projection, setting).

  python tools/exl3_expert_sweep.py [--rows 1,2,4,8] [--pdl 0|1] [--output sweep.json]
"""

from __future__ import annotations

import argparse
import json

import torch

from tensorfold.cuda.exl3 import experts

SETTINGS = [("grouped", None), ("decode 4x4", (4, 4)), ("decode 2x4", (2, 4)), ("decode 4x2", (4, 2)),
            ("decode 2x8", (2, 8))]


def _trellis(E, k, n, words16, g):
    return torch.randint(-32768, 32768, (E, k // 16, n // 16, words16), dtype=torch.int32,
                         generator=g).to(torch.int16).cuda()


def _layer(model, E=96):
    g = torch.Generator().manual_seed(3)
    if model == "flash_next":
        D, I, slots, cb = 2560, 640, 11, "mul1"
        mats = {p: [] for p in ("gate", "up", "down")}
        for e in range(E + 1):
            w16 = 80 if e == E else 48
            for p, (k, n) in (("gate", (D, I)), ("up", (D, I)), ("down", (I, D))):
                mats[p].append((_trellis(1, k, n, w16, g)[0].contiguous(), torch.ones(k).half().cuda(),
                                torch.ones(n).half().cuda()))
        return experts.prepare(mats["gate"], mats["up"], mats["down"], cb), slots, E, True
    D, I, slots = 4096, 2048, 8
    tr = [_trellis(E, k, n, 36, g) for k, n in ((D, I), (D, I), (I, D))]
    one = [torch.ones((E, n)).half().cuda() for n in (D, D, I, I, I, D)]
    return experts.prepare_stacked(*tr, *one, "mcg", window=14), slots, E, False


def _graph_time(fn, launches=50, replays=20):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        graph.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (launches * replays)        # microseconds a launch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,2,3,4,5,6,7,8,16")
    p.add_argument("--pdl", type=int, default=1)
    p.add_argument("--output")
    args = p.parse_args()
    ext = experts._ext()
    rows_out = []
    for model in ("flash_next", "glm53"):
        ex, slots, E, shared = _layer(model)
        D, I = ex.dims, ex.width
        for R in [int(r) for r in args.rows.split(",")]:
            g = torch.Generator().manual_seed(R)
            routed = slots - 1 if shared else slots
            sel = torch.stack([torch.randperm(E, generator=g)[:routed] for _ in range(R)]).to(torch.int32)
            if shared:
                sel = torch.cat([sel, torch.full((R, 1), E, dtype=torch.int32)], 1)
            sel = sel.cuda().contiguous()
            s = experts.Scratch(ex, R, slots)
            P = R * slots
            ids, members = s.window(R)
            ext.group(sel, ids, s.count, members, R, slots, ex.count)
            s.xg.normal_(0, 0.1), s.xu.normal_(0, 0.1), s.xd.normal_(0, 0.1)
            distinct = sorted(set(sel.view(-1).tolist()))
            for proj, mats, (k, n), cfg, xs, ptr, k2, rng in (
                    ("gate+up", 2, (D, I), s.cfg_gu, (s.xg, s.xu), (ex.gate_ptr, ex.up_ptr),
                     (ex.gate_k2, ex.up_k2), ex.k2_gu),
                    ("down", 1, (I, D), s.cfg_d, (s.xd, s.xd), (ex.down_ptr, ex.down_ptr),
                     (ex.down_k2, ex.down_k2), ex.k2_d)):
                nbytes = sum(int(ex.trellis_bytes[e]) for e in distinct) * (2 if mats == 2 else 1) // 3
                ref = None
                for name, tile in SETTINGS:
                    nt, pf = tile or (cfg[0], cfg[3])
                    if n % (16 * nt):
                        continue

                    def launch(nt=nt, pf=pf, dec=tile is not None):
                        ext.grouped(*xs, *ptr, *k2, ids, s.count, members, s.z, mats, k, n, P, cfg[2], slots, ex.cb,
                                    nt, cfg[1], pf, rng[0], rng[1], 1, ex.window, dec, bool(args.pdl))

                    s.z.fill_(float("nan"))
                    launch()
                    torch.cuda.synchronize()
                    z = s.z[:mats * cfg[2] * P * n].clone()
                    same = True if ref is None else bool(torch.equal(z.view(torch.int32), ref.view(torch.int32)))
                    ref = z if ref is None else ref
                    us = _graph_time(launch)
                    row = {"model": model, "rows": R, "proj": proj, "setting": name, "us": round(us, 2),
                           "gbs": round(nbytes / us / 1e3, 1), "same_bytes": same}
                    print(json.dumps(row), flush=True)
                    rows_out.append(row)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows_out, f, indent=1)
    bad = [r for r in rows_out if not r["same_bytes"]]
    if bad:
        raise SystemExit(f"{len(bad)} settings wrote other bytes than grouped_kernel: {bad[:3]}")


if __name__ == "__main__":
    main()
