"""How long a Flash Next decode graph's launch keeps the GPU idle, measured without nsys node tracing.

One load of the pack (draft head on, as the server loads it), an Engine with graphs and int8 KV (the lone graph
slot's path: same Graphs, buffers and compute), 600 tokens of context. For each window (1 row: a serial token;
5 and 7 rows: MTP verifies) the engine captures its graph, then ``--reps`` replays are timed three ways:

- host: wall time of the ``graph.replay()`` call (cudaGraphLaunch) on the host, perf_counter;
- gpu: event to event around the replay while a ``torch.cuda._sleep`` ahead of it keeps the stream busy, so the
  whole graph is submitted before the GPU reaches it: the graph's execution alone;
- idle: event to event around the replay on an idle stream: launch delay plus execution.
idle - gpu (medians) was meant as the delay from the call to the graph's first kernel. On the GB10 (20 Oct) the
graph ran 0.7-1.0 ms slower right after the sleep, so gpu overstates execution and the difference came out negative:
read the delay from an nsys graph-level trace instead (-t cuda-sw --cuda-graph-trace=graph, no node trace).

``--pieces 2,3,4``: the same window captured as consecutive layer ranges (embed with the first, the head with the
last), replayed in order on the same stream, timed the same way, and its logits and streams compared byte for byte
with the single graph's. Prints one JSON line a measurement."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


def _sleep_cycles(ms: float) -> int:
    """Cycles of ``torch.cuda._sleep`` that keep the stream busy about ``ms``."""

    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    a.record()
    torch.cuda._sleep(10_000_000)
    b.record()
    torch.cuda.synchronize()
    return int(10_000_000 * ms / a.elapsed_time(b))


def _time(launch, reps: int, cycles: int) -> dict:
    """Medians (ms) of host call wall, GPU execution behind a busy stream, and event span on an idle stream."""

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    host, gpu, idle = [], [], []
    for busy in (True, False):
        for i in range(reps + 3):
            torch.cuda.synchronize()
            if busy:
                torch.cuda._sleep(cycles)
            ev[0].record()
            t = time.perf_counter()
            launch()
            dt = (time.perf_counter() - t) * 1e3
            ev[1].record()
            torch.cuda.synchronize()
            if i < 3:
                continue
            (gpu if busy else idle).append(ev[0].elapsed_time(ev[1]))
            if not busy:
                host.append(dt)
    med = statistics.median
    return {"host_call_ms": round(med(host), 4), "gpu_ms": round(med(gpu), 4), "idle_span_ms": round(med(idle), 4),
            "launch_delay_ms": round(med(idle) - med(gpu), 4),
            "idle_span_p10_p90": [round(sorted(idle)[len(idle) // 10], 4), round(sorted(idle)[9 * len(idle) // 10], 4)]}


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))


def _pieces(e, R: int, n: int) -> list:
    """The window's forward captured as ``n`` graphs over consecutive layer ranges, in the engine's graph pool."""

    from tensorfold.families.qwen4_exp.cuda.forward import _embed, finish, layer_forward, stage

    w, st, b = e.w, e.st, e.buf
    segs = stage(w, b, [(st, [7] * R)])
    context = e.graphs._bucket(st.pos + R)
    L = len(w.layers)
    bounds = [round(i * L / n) for i in range(n + 1)]
    held: dict = {}

    def piece(i: int):
        def run() -> None:
            if i == 0:
                _embed(w, b.ids[:R], w.cfg.streams, b.h[:R])
                held["p"] = None
            for layer in w.layers[bounds[i]:bounds[i + 1]]:
                held["p"] = layer_forward(layer, w, segs, b, R, held["p"], context=context)
            if i == n - 1:
                finish(w, w.mixer, b, R, held["p"], logits=True)
        return run

    for i in range(n):                    # eager once: compiles nothing new, leaves the same buffers
        piece(i)()
    graphs = [e.graphs._capture(piece(i)) for i in range(n)]
    graphs.append(held)                   # the pending tensors between pieces stay referenced
    return graphs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--rows", default="1,5,7")
    p.add_argument("--pieces", default="2,3,4")
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--output")
    a = p.parse_args()

    from tensorfold.families.qwen4_exp.cuda import exl3_mm
    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.forward import stage
    from tensorfold.families.qwen4_exp.cuda.weights import load

    exl3_mm.use(exl3_mm.read())
    w = load(a.model, mtp=True, draft_vocab="default")
    e = Engine(w, capacity=8192, max_rows=8, prefill_rows=2048, graphs=True, kv_dtype="int8")
    prefill(e, list(range(1000, 1600)), None)
    cycles = _sleep_cycles(15.0)
    rows_out = []

    def emit(r: dict) -> None:
        print(json.dumps(r), flush=True)
        rows_out.append(r)

    for R in [int(x) for x in a.rows.split(",")]:
        st, b = e.st, e.buf
        e.forward([7] * R)                                  # captures (R, parity, context) on first use
        key = (R, st.cur[0] if st.cur else 0, e.graphs._bucket(st.pos + R))
        g = e.graphs.main[key]
        stage(w, b, [(st, [7] * R)])
        g.replay()
        torch.cuda.synchronize()
        ref = (b.logits[:R].clone(), b.streams[:R].clone())
        emit({"rows": R, "pieces": 1, **_time(g.replay, a.reps, cycles)})
        for n in [int(x) for x in a.pieces.split(",") if x]:
            parts = _pieces(e, R, n)
            gs = parts[:-1]

            def launch(gs=gs) -> None:
                for x in gs:
                    x.replay()

            launch()
            torch.cuda.synchronize()
            same = bool(torch.equal(b.logits[:R].view(torch.int16), ref[0].view(torch.int16))
                        and torch.equal(b.streams[:R].view(torch.int16), ref[1].view(torch.int16)))
            emit({"rows": R, "pieces": n, "same_bytes": same, **_time(launch, a.reps, cycles)})
            del parts, gs
        # the single graph again: drift check
        emit({"rows": R, "pieces": 1, "again": True, **_time(g.replay, a.reps, cycles)})
    if a.output:
        with open(a.output, "w") as f:
            json.dump(rows_out, f, indent=1)


if __name__ == "__main__":
    main()
