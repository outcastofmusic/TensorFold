"""Per-round kernel statistics of a decode trace from nsys (SQLite export of a software CUDA trace).

    nsys profile -t cuda-sw --cuda-graph-trace=node ...      (or nsys launch / start / stop around one request)
    nsys export -t sqlite -o REPORT.sqlite REPORT.nsys-rep
    python3 tools/nsys_rounds.py REPORT.sqlite [--big 500] [--tokens 160] [--bytes GB_PER_ROUND] [--top 15]

In the tensorfold-run image (CUDA 13.3 on the 580 kernel driver, forward-compat mode) nsys's default hardware trace
records no kernels on the GB10; ``-t cuda-sw`` does. A round runs from one split point to the next: each
cudaGraphLaunch by default (serial decode: one graph a token), only graph launches of at least ``--big`` kernels
(drafted decode: the verify graph starts a round and the draft graphs stay inside it), or any runtime call named by
``--split``. Per round it reports kernels, the union of kernel busy time, the wall time between round starts and
their difference as idle share, then the kernels that take the most time over the steady rounds (the middle 80% by
wall time). Under the software trace a kernel's duration includes its own launch cost (an empty kernel reads 0.77 us
against 0.70 us a dependent launch untraced, tools/launch_bench.cu), so idle share counts host and dependency gaps,
graph memcpy nodes included, but not launch overhead.
"""
import argparse
import bisect
import collections
import re
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--split", default="cudaGraphLaunch", help="runtime call name prefix that starts a round (cudaGraphLaunch, cudaEventSynchronize, cudaStreamSynchronize)")
ap.add_argument("--top", type=int, default=15); ap.add_argument("--bytes", type=float, default=None)
ap.add_argument("--tokens", type=float, default=None, help="tokens emitted over the traced request, for tokens a round")
ap.add_argument("--big", type=int, default=0, help="with --split cudaGraphLaunch: only graph launches of at least "
                "this many kernels start a round (the verify graph; draft graphs stay inside the round)")
ap.add_argument("--min-kernels", type=int, default=50, help="drop rounds with fewer kernels (host-only blips)")
a = ap.parse_args()
c = sqlite3.connect(a.db)
S = dict(c.execute("select id, value from StringIds"))

def short(name: str) -> str:
    n = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    n = re.sub(r"\((const|void|int|float|long|unsigned|__|signed|char|bool|short|double|\)).*$", "", n)  # args
    base = re.sub(r"<.*", "", n).split("::")[-1]
    tmpl = re.search(r"<(.*)>", n)
    t = re.sub(r"\(\w+\)", "", tmpl.group(1)) if tmpl else ""
    return base + (f"<{t[:60]}>" if t else "")

ids = [i for i, v in S.items() if v.startswith(a.split)]
# a launch call starts its round where it begins; a synchronization ends the previous round where it returns
col = "start" if "Launch" in a.split else "end"
rows_ = list(c.execute(
    f"select {col}, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where nameId in ({','.join(map(str, ids))})"))
big_corr = set()
if a.big:
    kc = collections.Counter(r[0] for r in c.execute("select correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))
    big_corr = {corr for _, corr in rows_ if kc[corr] >= a.big}
    rows_ = [x for x in rows_ if x[1] in big_corr]
marks = sorted(t for t, _ in rows_)
# kernels in GPU order, attributed to the round whose launch (host correlation) followed the mark
launch_t = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
ks = list(c.execute("select start, end, demangledName, correlationId, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
rounds = collections.defaultdict(list)
for s, e, n, corr, node in ks:
    t = launch_t.get(corr, s)
    i = bisect.bisect_right(marks, t) - 1
    if i >= 0:
        rounds[i].append((s, e, n, node is not None, corr in big_corr))
stats = []
idx = sorted(rounds)
for j, i in enumerate(idx):
    r = rounds[i]
    if len(r) < a.min_kernels:
        continue
    r.sort()
    busy, cur_s, cur_e = 0, r[0][0], r[0][1]
    for s, e, *_ in r[1:]:
        if s > cur_e:
            busy += cur_e - cur_s; cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    first = r[0][0]; last = max(e for _, e, *_ in r)
    wall = (min(x[0] for x in rounds[idx[j + 1]]) - first) if j + 1 < len(idx) else last - first
    vb = [x for x in r if x[4]]
    vspan = (max(x[1] for x in vb) - vb[0][0]) if vb else 0
    stats.append({"i": i, "n": len(r), "graph": sum(1 for x in r if x[3]), "busy": busy, "span": last - first,
                  "wall": wall, "vn": len(vb), "vspan": vspan, "vsum": sum(x[1] - x[0] for x in vb),
                  "ksum": sum(e - s for s, e, *_ in r), "kern": r})
stats.sort(key=lambda d: d["wall"])
k = len(stats); steady = stats[k // 10: k - k // 10] or stats
med = lambda key: statistics.median(d[key] for d in steady)
print(f"rounds {len(stats)} (steady {len(steady)}), kernels/round median {med('n'):.0f} (in graph {med('graph'):.0f})")
print(f"wall/round median {med('wall')/1e6:.3f} ms; GPU busy (union) {med('busy')/1e6:.3f} ms; kernel sum "
      f"{med('ksum')/1e6:.3f} ms; first-to-last kernel {med('span')/1e6:.3f} ms")
print(f"idle share of wall {1 - med('busy')/med('wall'):.1%} (inside the round's kernel span "
      f"{1 - med('busy')/med('span'):.1%}, host time between rounds {1 - med('span')/med('wall'):.1%})")
if a.bytes:
    print(f"bytes/round {a.bytes} GB -> {a.bytes / (med('wall') / 1e9):.1f} GB/s on wall, "
          f"{a.bytes / (med('busy') / 1e9):.1f} GB/s on GPU busy (273 GB/s peak)")
if a.tokens:
    span = (max(d["kern"][-1][1] for d in stats) - min(d["kern"][0][0] for d in stats)) / 1e9
    print(f"{a.tokens:.0f} tokens over {len(stats)} rounds = {a.tokens/len(stats):.2f} a round; "
          f"traced span {span:.3f} s = {a.tokens/span:.1f} tok/s")
tot = collections.Counter(); cnt = collections.Counter()
if a.big:
    print(f"verify graph: {med('vn'):.0f} kernels, span {med('vspan')/1e6:.3f} ms, kernel sum {med('vsum')/1e6:.3f} ms "
          f"(idle in span {1 - med('vsum')/med('vspan'):.1%}); rest of round (sampling, drafts, host) "
          f"{(med('wall') - med('vspan'))/1e6:.3f} ms with {med('n') - med('vn'):.0f} kernels")
for d in steady:
    for s, e, n, *_ in d["kern"]:
        key = short(S.get(n, str(n)))
        tot[key] += e - s; cnt[key] += 1
allk = sum(tot.values()); R = len(steady)
print("\n| kernel | launches/round | µs/round | share of kernel time |\n|---|---:|---:|---:|")
for key, v in tot.most_common(a.top):
    print(f"| `{key}` | {cnt[key]/R:.0f} | {v/R/1e3:.0f} | {v/allk:.1%} |")
