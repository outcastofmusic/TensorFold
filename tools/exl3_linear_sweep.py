"""Launch-shape sweep of the EXL3 dense linear (linear.cu's linear_kernel) at Flash Next's decode shapes.

Measurement only: the variants live in tools/linear_sweep/ and nothing in src/ calls them. Each variant keeps the
shipped reduction (plan(K, N)'s K range for every warp, warps added in order, then splits in order), and every
variant's output is compared with the shipped kernel's by torch.equal on every real layer of the shape.

  orig mb<B> [pf2] [gw]  (gw: with the shipped griddepcontrol.wait)
  orig mb<B> [pf2]   linear_kernel under __launch_bounds__(WK * 32, B): registers capped so B blocks fit an SM;
                     pf2 adds an L2 prefetch two k steps past the one being loaded
  wsplit w<L> mb<B>  the same WK warps spread over WK / L blocks of L warps, per-warp sums through Z, the last block
                     of a column block adding them in the shipped order

Timing: the pack's real layers (cloned until a shape's copies hold at least --cold-mb, so no launch finds its
weights in L2), one launch a copy captured in one CUDA graph, the median over --iters replays of replay time / copies.
The read reference streams the same bytes with a plain load kernel (__ldcs, uint4) under the same rotation.

  python tools/exl3_linear_sweep.py /models/Qwen3.8-Flash-Next-exl3-3.05bpw --output sweep.json
  python tools/exl3_linear_sweep.py --build-only        # compile both extensions without a GPU (sm_121)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

HERE = Path(__file__).parent / "linear_sweep"
EXL3 = Path(__file__).resolve().parent.parent / "src" / "tensorfold" / "cuda" / "exl3"
ROWS = (1, 5, 7, 16)

# (name, prefixes' layer kind, K2, verify launches and draft launches a 7-row round: ana/budget.py on mtp.json)
SHAPES = [
    ("deltanet qkv", ["linear_attn.in_proj_qkv"], 36.0, 0.0),
    ("deltanet z", ["linear_attn.in_proj_z"], 36.0, 0.0),
    ("out/o", ["linear_attn.out_proj", "self_attn.o_proj"], 48.0, 4.4),
    ("attention q", ["self_attn.q_proj"], 12.0, 4.4),
    ("attention k/v", ["self_attn.k_proj", "self_attn.v_proj"], 24.0, 8.8),
    ("indexer", ["self_attn.indexer.index_qk_proj"], 12.0, 4.4),
    ("mtp fc", ["fc"], 0.0, 8.8),
    ("lm_head", ["lm_head"], 1.0, 0.0),
]


def _ext():
    from torch.utils import cpp_extension

    srcs = [HERE / n for n in ("sweep.cpp", "sweep_k6.cu", "sweep_k8.cu", "sweep_k10.cu", "read.cu")]
    return cpp_extension.load(name="tf_linear_sweep_v1", sources=[str(s) for s in srcs],
                              extra_include_paths=[str(EXL3), str(HERE)],
                              extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr",
                                                 "-gencode=arch=compute_121,code=sm_121"], verbose=False)


def _shipped():
    from tensorfold.cuda.exl3 import linear

    return linear._ext()


def variants(wk: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    """(label, (kind, wkl, minb, pfd)) for a plan with ``wk`` warps a block."""

    out = [("orig mb1 (copy)", (0, 0, 1, 0)), ("orig mb1 gw", (2, 0, 1, 0)), ("orig mb1 pf2 gw", (2, 0, 1, 2))]
    out += [(f"orig mb{b}", (0, 0, b, 0)) for b in ((2,) if wk == 8 else (3, 4))]
    out += [("orig mb1 pf2", (0, 0, 1, 2))]
    out += [(f"orig mb{b} pf2", (0, 0, b, 2)) for b in ((2,) if wk == 8 else (3, 4))]
    if wk == 8:
        out += [(f"wsplit w4 mb{b}", (1, 4, b, 0)) for b in (1, 2, 3, 4)]
    out += [(f"wsplit w2 mb{b}", (1, 2, b, 0)) for b in (1, 4, 6, 8)]
    return out


def graph_us(launches, iters: int) -> float:
    """Median microseconds a launch: ``launches`` captured in one graph, replayed ``iters`` times after three."""

    for f in launches:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in launches:
            f()
    for _ in range(3):
        g.replay()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(launches))
    del g
    return statistics.median(times)


def prefixes(model: str, kinds: list[str]) -> list[str]:
    weight_map = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted({k[: -len(".trellis")] for k in weight_map if k.endswith(".trellis")})
    out = []
    for kind in kinds:
        if kind == "lm_head":
            out.append("lm_head")
        elif kind == "fc":
            out += ["mtp.fc_embedding", "mtp.fc_hidden"]
        else:
            got = [n for n in names if n.startswith("model.language_model.layers.") and n.endswith("." + kind)]
            out += sorted(got, key=lambda n: int(n.split(".")[3]))
    return out


def read_reference(ext, sm: int, iters: int, cold_mb: int) -> tuple[list[dict], dict]:
    """The streaming read's GB/s on 2 GiB (the attainable peak) and the per-config table."""

    nbytes = 2 << 30
    buf = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda").random_()
    out = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
    rows = []
    for mult in (2, 4, 8, 16, 32):
        for threads in (256, 512, 1024):
            for unroll in (4, 8):
                blocks = sm * mult
                if blocks * threads > out.numel():
                    continue

                def f(blocks=blocks, threads=threads, unroll=unroll):
                    ext.read_bytes(buf, nbytes, out, blocks, threads, unroll)

                us = graph_us([f], max(20, iters // 10))
                rows.append({"measure": "read 2 GiB", "blocks": blocks, "threads": threads, "unroll": unroll,
                             "us": round(us, 1), "gbs": round(nbytes / us / 1e3, 1)})
    best = max(rows, key=lambda r: r["gbs"])
    del buf
    torch.cuda.empty_cache()
    return rows, best


def read_shape(ext, nbytes: int, sm: int, grid: tuple[int, int], iters: int, cold_mb: int) -> dict:
    """A streaming read of ``nbytes`` a launch over rotating regions (cold L2): best config, and the linear's grid."""

    copies = max(1, min(400, math.ceil(cold_mb * 1e6 / nbytes)))
    stride = (nbytes + 4095) // 4096 * 4096
    buf = torch.empty(stride * copies // 4, dtype=torch.int32, device="cuda").random_()
    views = [buf[i * stride // 4: i * stride // 4 + nbytes // 4] for i in range(copies)]
    out = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")
    res = {}
    for blocks, threads in [(sm * m, 512) for m in (2, 4, 8, 16)] + [(sm * 4, 1024), grid]:
        launches = [lambda v=v, blocks=blocks, threads=threads: ext.read_bytes(v, nbytes, out, blocks, threads, 4)
                    for v in views]
        res[(blocks, threads)] = graph_us(launches, iters)
    best = min(res, key=res.get)
    del buf, views
    torch.cuda.empty_cache()
    return {"read_best_us": round(res[best], 2), "read_best_cfg": list(best),
            "read_best_gbs": round(nbytes / res[best] / 1e3, 1),
            "read_linear_grid_us": round(res[grid], 2), "copies": copies}


def sweep_shape(ext, shipped, model: str, name: str, kinds: list[str], iters: int, cold_mb: int,
                rows_list: tuple[int, ...]) -> list[dict]:
    from tensorfold.cuda.exl3 import linear

    layers = [linear.Exl3Linear.load(model, p) for p in prefixes(model, kinds)]
    lin0 = layers[0]
    k, n, k2 = lin0.k, lin0.n, lin0.k2
    sk, wk = lin0.split
    assert all((x.k, x.n, x.k2, x.split) == (k, n, k2, (sk, wk)) for x in layers), name
    nbytes = lin0.nbytes()
    sk_stride, nb_stride = lin0.strides
    copies = max(len(layers), min(400, math.ceil(cold_mb * 1e6 / nbytes)))
    words = [layers[i % len(layers)].words if i < len(layers) else layers[i % len(layers)].words.clone()
             for i in range(copies)]
    svh = [layers[i % len(layers)].svh for i in range(copies)]
    counters = torch.zeros((8 * (n // 128),), dtype=torch.int32, device="cuda")
    vlist = variants(wk)
    out = []
    g = torch.Generator().manual_seed(11)
    for m in rows_list:
        x = torch.randn((m, k), generator=g).to(torch.bfloat16).cuda()
        y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        z = torch.empty((max(sk * wk, 1) * m * n,), dtype=torch.float32, device="cuda")
        xh = torch.empty((m, k), dtype=torch.float16, device="cuda")
        shipped.rot_in(x, lin0.suh, xh, False)

        def base_launch(i, y=y, xh=xh, z=z):
            shipped.linear(xh, words[i], sk_stride, nb_stride, svh[i], None, y, z if sk > 1 else None, counters,
                           k2, 2, sk, wk, False, 1)

        def var_launch(i, v, y=y, xh=xh, z=z):
            kind, wkl, minb, pfd = v
            need_z = sk > 1 or kind == 1
            ext.linear(xh, words[i], sk_stride, nb_stride, svh[i], None, y, z if need_z else None, counters,
                       k2, sk, wk, kind, wkl, minb, pfd)

        # bytes: every real layer, its own rot_in, shipped vs each variant
        same = {label: True for label, _ in vlist}
        for lay in layers:
            xl = torch.empty_like(xh)
            shipped.rot_in(x, lay.suh, xl, False)
            ref = torch.empty_like(y)
            shipped.linear(xl, lay.words, sk_stride, nb_stride, lay.svh, None, ref, z if sk > 1 else None, counters,
                           k2, 2, sk, wk, False, 1)
            for label, v in vlist:
                got = torch.full_like(y, float("nan"))
                kind, wkl, minb, pfd = v
                ext.linear(xl, lay.words, sk_stride, nb_stride, lay.svh, None, got,
                           z if (sk > 1 or kind == 1) else None, counters, k2, sk, wk, kind, wkl, minb, pfd)
                same[label] &= bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
        torch.cuda.synchronize()

        base_us = graph_us([lambda i=i: base_launch(i) for i in range(copies)], iters)
        row = {"shape": name, "rows": m, "K": k, "N": n, "bits": k2 / 2, "SK": sk, "WK": wk, "mb": round(nbytes / 1e6, 2), "bytes": nbytes,
               "copies": copies, "layers": len(layers), "shipped_us": round(base_us, 2),
               "shipped_gbs": round(nbytes / base_us / 1e3, 1), "variants": {}}
        for label, v in vlist:
            us = graph_us([lambda i=i, v=v: var_launch(i, v) for i in range(copies)], iters)
            row["variants"][label] = {"us": round(us, 2), "gbs": round(nbytes / us / 1e3, 1), "same_bytes": same[label]}
        print(json.dumps(row), flush=True)
        out.append(row)
    del layers, words, svh
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--cold-mb", type=int, default=256)
    p.add_argument("--shapes", default="")
    p.add_argument("--rows", default=",".join(map(str, ROWS)))
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--output")
    a = p.parse_args()
    if a.build_only:                          # no GPU in the build container: build.py asks the device for its arch
        torch.cuda.get_device_capability = lambda *_: (12, 1)
        _shipped()
        _ext()
        print("built")
        return
    ext, shipped = _ext(), _shipped()
    props = torch.cuda.get_device_properties(0)
    sm = props.multi_processor_count
    l2 = getattr(props, "L2_cache_size", 0)
    print(json.dumps({"gpu": props.name, "sms": sm, "l2_bytes": l2}), flush=True)
    results = {"gpu": props.name, "sms": sm, "l2_bytes": l2}
    read_rows, best = read_reference(ext, sm, a.iters, a.cold_mb)
    print(json.dumps({"read_peak": best}), flush=True)
    results["read_2gib"] = read_rows
    results["read_peak"] = best
    rows_list = tuple(int(r) for r in a.rows.split(","))
    want = set(a.shapes.split(",")) if a.shapes else None
    results["shapes"] = []
    for name, kinds, nv, nd in SHAPES:
        if want and name not in want:
            continue
        got = sweep_shape(ext, shipped, a.model, name, kinds, a.iters, a.cold_mb,
                          (1,) if name == "mtp fc" else rows_list)
        r0 = got[0]
        ref = read_shape(ext, r0["bytes"], sm, (r0["N"] // 128 * r0["SK"], r0["WK"] * 32),
                         a.iters, a.cold_mb)
        print(json.dumps({"shape": name, **ref}), flush=True)
        for r in got:
            r.update(ref)
            r["verify_launches"], r["draft_launches"] = nv, nd
        results["shapes"].extend(got)
        if a.output:
            Path(a.output).write_text(json.dumps(results, indent=1))
    if a.output:
        Path(a.output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
