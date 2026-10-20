"""A-B-A decode-speed A/B of TensorFold env flags on Flash Next, with an exactness check between arms.

Each arm is one server boot of this checkout with its env flags (the others off). The order runs A, B, A by default,
so drift between boots shows up as A1 against A2. Every arm decodes the same prompts (prose, code, json, edit, copy,
and optionally a long-context one) greedy and at T=0.7, with seed 1234 + rep in every arm. TensorFold samples with
keyed draws, so a repeat with the same seed replays the same text: each rep gets its own seed, and a speed flag must
leave the text identical at both temperatures. The script compares every arm's text with A1's and lists each
difference.

The warm-up skips the long prompt, so each boot's first long request is the cold one: it crosses into a decode
graph bucket nothing captured yet, which is what TENSORFOLD_IDLE_GRAPHS (T3) removes. The summary prints that
request's TTFT and decode speed per boot; a cold stall is one sample a boot, so measure T3 with more boots
(``--order A,B,A,B,A,B``).

``--concurrency N`` sends each rep's requests N at a time, so streams share rounds as they do under ``--parallel 4``
in production (where TENSORFOLD_PLAIN_GATE's shared-round path runs); the summary adds each boot's batch throughput.

``--turns`` measures a follow-up turn instead (TENSORFOLD_END_ANCHOR, TENSORFOLD_GRID_ANCHORS: T5): each prompt is
answered, then asked a follow-up that sends the answer back, with thinking off and on. The summary prints the
follow-up's TTFT and the prompt tokens the server found cached.

Run it only when the box is free: it boots its own server (container ``tftest-ab-<arm>``, port 30010 by default,
the checkout mounted over the image's install) and refuses to start below ``--min-avail-gib`` of MemAvailable. Each
boot's server log is saved beside ``--out`` (the self-check and loop-close lines land there).

  python3 tools/tandem_ab.py --model /home/agis/models/Qwen3.8-Flash-Next-exl3-3.05bpw \\
      --arms 'A=;B=TENSORFOLD_STAIRCUT=1' --order A,B,A --reps 3 --out ab-staircut.json

Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGE = "tensorfold-run:ffdaafceb2f0"

PROMPTS = {
    "prose": "Write a 400-word short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a complete Python module implementing an LRU cache class with get, put, delete and resize "
            "methods, type hints, docstrings and a small unittest suite.",
    "json": "Return a JSON array of 12 objects describing fictional employees, each with id, name, email, "
            "department, salary, start_date and a skills array of 3 strings. Output only the JSON.",
    "edit": "Rewrite the following function so it uses pathlib instead of os.path, changing nothing else, and "
            "output the whole function:\n\n```python\nimport os\n\ndef collect(root, ext):\n    \"\"\"Every file "
            "under root with the extension ext, sorted.\"\"\"\n    found = []\n    for dirpath, dirnames, filenames "
            "in os.walk(root):\n        for name in filenames:\n            if os.path.splitext(name)[1] == ext:\n"
            "                found.append(os.path.join(dirpath, name))\n    found.sort()\n    return found\n\n\n"
            "def sizes(root, ext):\n    \"\"\"Each matching file's size in bytes.\"\"\"\n    out = {}\n    for path in "
            "collect(root, ext):\n        out[path] = os.path.getsize(path)\n    return out\n```",
    "copy": "Repeat the following list exactly, one item per line, then stop:\n" + "\n".join(
        f"{i}. item-{i:03d} costs {i * 7 % 97}.{i % 10}0 EUR and ships in {i % 5 + 1} days" for i in range(1, 41)),
    "rewrite": "Rename the function `load_orders` to `read_orders` everywhere in this module, change nothing else, "
               "and output the whole module:\n\n```python\n" + "\n".join([
        '"""Orders: read them from CSV, total them by customer, and write a report."""',
        "", "import csv", "from collections import defaultdict", "from dataclasses import dataclass",
        "from decimal import Decimal", "from pathlib import Path", "", "",
        "@dataclass(frozen=True)", "class Order:", "    order_id: str", "    customer: str", "    amount: Decimal",
        "    currency: str", "", "",
        "def load_orders(path: Path) -> list[Order]:", '    """Every row of the CSV at path as an Order."""',
        "    orders = []", "    with path.open(newline=\"\") as handle:",
        "        for row in csv.DictReader(handle):",
        "            orders.append(Order(row[\"id\"], row[\"customer\"], Decimal(row[\"amount\"]), row[\"currency\"]))",
        "    return orders", "", "",
        "def totals(orders: list[Order], currency: str) -> dict[str, Decimal]:",
        '    """Each customer\'s total in one currency; orders in other currencies are skipped."""',
        "    out: dict[str, Decimal] = defaultdict(Decimal)", "    for order in orders:",
        "        if order.currency != currency:", "            continue",
        "        out[order.customer] += order.amount", "    return dict(out)", "", "",
        "def top(totals_by_customer: dict[str, Decimal], n: int) -> list[tuple[str, Decimal]]:",
        '    """The n customers with the largest totals, largest first, ties by name."""',
        "    return sorted(totals_by_customer.items(), key=lambda kv: (-kv[1], kv[0]))[:n]", "", "",
        "def report(path: Path, currency: str, n: int = 10) -> str:",
        '    """A plain-text report of the top n customers for the orders at path."""',
        "    orders = load_orders(path)", "    ranked = top(totals(orders, currency), n)",
        "    lines = [f\"Top {len(ranked)} customers by {currency} spend ({len(orders)} orders read)\"]",
        "    for rank, (customer, amount) in enumerate(ranked, start=1):",
        "        lines.append(f\"{rank:>3}. {customer:<30} {amount:>12.2f} {currency}\")",
        "    return \"\\n\".join(lines)", "", "",
        "def write_report(source: Path, target: Path, currency: str, n: int = 10) -> int:",
        '    """Write the report for source to target; return the number of orders read."""',
        "    orders = load_orders(source)", "    target.write_text(report(source, currency, n) + \"\\n\")",
        "    return len(orders)", "", "",
        "if __name__ == \"__main__\":", "    import sys", "",
        "    print(report(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else \"EUR\"))",
    ]) + "\n```",
}


def long_prompt(tokens: int, seed: int = 7) -> str:
    """A filler report of about ``tokens`` tokens (4 characters a token), then a question about it."""

    rng = random.Random(seed)
    words = ("system", "latency", "budget", "review", "quarter", "network", "storage", "policy", "release",
             "customer", "incident", "capacity", "forecast", "vendor", "migration", "audit", "schema", "region")
    lines, size = [], 0
    while size < tokens * 4:
        line = f"Section {len(lines) + 1}: " + " ".join(rng.choice(words) for _ in range(14)) + "."
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines) + "\n\nSummarise the report above in 300 words, naming its most frequent topics."


def avail_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def sh(*cmd: str, check: bool = True) -> str:
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout


def boot(name: str, env: dict, args) -> None:
    flags = []
    for k, v in env.items():
        flags += ["-e", f"{k}={v}"]
    sh("docker", "rm", "-f", name, check=False)
    sh("docker", "run", "-d", "--init", "--name", name, "--gpus", "all", "--ipc=host", "--network", "host",
       "-v", f"{args.model}:{args.model}:ro", "-v", f"{REPO}:/tf:ro", "-e", "PYTHONPATH=/tf/src",
       "-v", "tensorfold-ext:/ext", "-e", "TORCH_EXTENSIONS_DIR=/ext", "-e", "TRITON_CACHE_DIR=/ext/triton",
       "-e", "HF_HUB_OFFLINE=1", *flags, args.image,
       "tensorfold", "serve", args.model, "--host", "127.0.0.1", "--port", str(args.port),
       "--name", "ab", "--context", str(args.context), "--parallel", str(args.parallel),
       "--kv-dtype", args.kv_dtype, "--no-update-check", *args.serve_arg)
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(3600):
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=3).read()
            return
        except OSError:
            if name not in sh("docker", "ps", "--format", "{{.Names}}"):
                sys.exit(f"{name} exited while loading:\n" + sh("docker", "logs", "--tail", "20", name, check=False))
            time.sleep(0.5)
    sys.exit(f"{name} did not answer within 30 minutes")


def stop(name: str) -> None:
    sh("docker", "rm", "-f", name, check=False)
    for _ in range(120):                       # the next boot needs the memory back
        if name not in sh("docker", "ps", "-a", "--format", "{{.Names}}"):
            return
        time.sleep(1)


def metrics(base: str) -> dict:
    out = {}
    try:
        for line in urllib.request.urlopen(base + "/metrics", timeout=10).read().decode().splitlines():
            if line.startswith("tensorfold:") and not line.startswith("#"):
                key, _, value = line.rpartition(" ")
                out[key] = float(value)
    except OSError:
        pass
    return out


def run_one(base: str, prompt: str | list, temperature: float, tokens: int, seed: int = 1234,
            thinking: bool = False) -> dict:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    body = {"model": "ab", "max_tokens": tokens, "temperature": temperature, "stream": True, "seed": seed,
            "stream_options": {"include_usage": True}, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start, first, last, text, usage = time.perf_counter(), None, None, [], None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    if piece:
                        text.append(piece)
    n = (usage or {}).get("completion_tokens") or 0
    tps = (n - 1) / (last - first) if first is not None and last is not None and last > first and n > 1 else 0.0
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"text": "".join(text), "tokens": n, "decode_tps": round(tps, 2),
            "ttft_s": round((first or start) - start, 3), "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": cached}


FOLLOW_UP = "Now give the three most important points of your answer as a short bulleted list."


def two_turns(base: str, prompt: str, tokens: int, seed: int, thinking: bool) -> dict:
    """The prompt's answer, then a follow-up that sends it back: the follow-up's TTFT and cached prompt tokens."""

    first = run_one(base, prompt, 0.0, tokens, seed=seed, thinking=thinking)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": first["text"]},
                {"role": "user", "content": FOLLOW_UP}]
    second = run_one(base, messages, 0.0, 64, seed=seed, thinking=thinking)
    return {"text": first["text"] + "\n---\n" + second["text"], "tokens": second["tokens"],
            "decode_tps": second["decode_tps"], "ttft_s": second["ttft_s"], "prompt_tokens": second["prompt_tokens"],
            "cached_tokens": second["cached_tokens"], "first_ttft_s": first["ttft_s"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="A=;B=TENSORFOLD_STAIRCUT=1",
                    help="NAME=ENV=VALUE,ENV=VALUE;NAME=... (an empty list: every flag off)")
    ap.add_argument("--order", default="A,B,A")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--long-tokens", type=int, default=0, help="add a long-context prompt of about this many tokens")
    ap.add_argument("--kinds", default=",".join(PROMPTS))
    ap.add_argument("--port", type=int, default=30010)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kv-dtype", default="int8")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--serve-arg", action="append", default=[], help="an extra tensorfold serve argument")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight at once within a rep")
    ap.add_argument("--turns", action="store_true", help="time a follow-up turn (T5) instead of decode speed")
    ap.add_argument("--min-avail-gib", type=float, default=90.0)
    ap.add_argument("--out", default="tandem-ab.json")
    args = ap.parse_args()

    if args.port in (30000, 30001, 4000):
        sys.exit("ports 30000, 30001 and 4000 belong to the live servers and the gateway")
    if avail_gib() < args.min_avail_gib:
        sys.exit(f"MemAvailable is {avail_gib():.0f} GiB, below --min-avail-gib {args.min_avail_gib:g}: stop the "
                 "serving engines first (this boots a full model)")
    arms = {}
    for part in args.arms.split(";"):
        name, _, envs = part.partition("=")
        arms[name.strip()] = dict(e.split("=", 1) for e in envs.split(",") if e.strip())
    prompts = {k: PROMPTS[k] for k in args.kinds.split(",") if k}
    if args.long_tokens:
        prompts["long"] = long_prompt(args.long_tokens)
    base = f"http://127.0.0.1:{args.port}"
    results = []
    for i, arm in enumerate(args.order.split(",")):
        name = f"tftest-ab-{arm.lower()}{i}"
        print(f"[ab] boot {i + 1}: arm {arm} {arms[arm] or '(flags off)'}", flush=True)
        boot(name, arms[arm], args)
        try:
            for kind, prompt in prompts.items():          # one untimed request a kind: kernels and graphs warm
                if kind != "long":                        # the long prompt's first request stays cold (T3)
                    run_one(base, prompt, 0.0, 32)
            before = metrics(base)
            # --turns: "temperature" 0.0 is thinking off and 1.0 thinking on (both greedy), so the summary's cells hold
            for rep in range(args.reps):
                jobs = [(kind, prompt, temp) for kind, prompt in prompts.items()
                        for temp in ((0.0, 1.0) if args.turns else (0.0, 0.7))]

                def go(job, rep=rep):
                    _, prompt, temp = job
                    if args.turns:
                        return two_turns(base, prompt, args.tokens, 1234 + rep, thinking=temp == 1.0)
                    return run_one(base, prompt, temp, args.tokens, seed=1234 + rep)

                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                    got = list(pool.map(go, jobs))
                wall = time.perf_counter() - t0
                for (kind, _, temp), r in zip(jobs, got):
                    results.append({"boot": i, "arm": arm, "kind": kind, "temperature": temp, "rep": rep, **r})
                    print(f"[ab] {arm}{i} {kind:6} T={temp} rep {rep}: {r['decode_tps']} tok/s "
                          f"({r['tokens']} tokens), TTFT {r['ttft_s']} s, cached {r['cached_tokens']}", flush=True)
                results.append({"boot": i, "arm": arm, "rep": rep, "batch_tps": sum(r["tokens"] for r in got) / wall})
            after = metrics(base)
            results.append({"boot": i, "arm": arm, "metrics": {k: after[k] - before.get(k, 0.0) for k in after
                                                              if "total" in k and not k.endswith("_bucket")}})
        finally:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
            Path(f"{args.out}.{arm}{i}.log").write_text(logs.stdout + logs.stderr)
            stop(name)
    Path(args.out).write_text(json.dumps({"args": vars(args), "arms": arms, "results": results}, indent=1))

    rows = [r for r in results if "kind" in r]
    ref = {(r["kind"], r["temperature"], r["rep"]): r["text"] for r in rows if r["boot"] == 0}
    diffs = [(r["boot"], r["arm"], r["kind"], r["temperature"], r["rep"]) for r in rows
             if r["boot"] > 0 and r["text"] != ref[(r["kind"], r["temperature"], r["rep"])]]
    print("\nmedian decode tok/s by boot (arm):")
    boots = sorted({(r["boot"], r["arm"]) for r in rows})
    print(f"{'kind':8} {'T':>4} " + " ".join(f"{a}{b:<6}" for b, a in boots))
    temps = (0.0, 1.0) if args.turns else (0.0, 0.7)
    for kind in prompts:
        for temp in temps:
            cells = [statistics.median(r["decode_tps"] for r in rows
                                       if r["boot"] == b and r["kind"] == kind and r["temperature"] == temp)
                     for b, _ in boots]
            print(f"{kind:8} {temp:>4} " + " ".join(f"{c:<7.1f}" for c in cells))
    for b, a in boots[1:]:                              # boot against boot 0: A2/A1 is the drift between boots
        ratios = [statistics.median(r["decode_tps"] for r in rows if r["boot"] == b and r["kind"] == kind
                                    and r["temperature"] == temp)
                  / statistics.median(r["decode_tps"] for r in rows if r["boot"] == 0 and r["kind"] == kind
                                      and r["temperature"] == temp)
                  for kind in prompts for temp in temps]
        print(f"{a}{b}/{boots[0][1]}0: geometric mean of median ratios {statistics.geometric_mean(ratios):.3f}")
    batches = [r for r in results if "batch_tps" in r]
    print("\nbatch throughput by boot (generated tokens / wall s, median of reps):")
    for b, a in boots:
        print(f"  {a}{b}: {statistics.median(r['batch_tps'] for r in batches if r['boot'] == b):.1f}")
    if args.turns:
        print("\nfollow-up turn, median TTFT s / median cached prompt tokens (T=0 thinking off, T=1 thinking on):")
        for kind in prompts:
            for temp in temps:
                cells = []
                for b, _ in boots:
                    got = [r for r in rows if r["boot"] == b and r["kind"] == kind and r["temperature"] == temp]
                    cached = [r["cached_tokens"] for r in got if r["cached_tokens"] is not None]
                    cells.append(f"{statistics.median(r['ttft_s'] for r in got):.3f}/"
                                 f"{statistics.median(cached) if cached else '-'}")
                print(f"  {kind:8} {temp:>4} " + " ".join(f"{c:<14}" for c in cells))
    if "long" in prompts:
        print("\nfirst long request a boot (cold graphs):")
        for b, a in boots:
            cold = next(r for r in rows if r["boot"] == b and r["kind"] == "long")
            print(f"  {a}{b}: TTFT {cold['ttft_s']} s, {cold['decode_tps']} tok/s")
    first_arm = boots[0][1]
    for arm in sorted({a for _, a in boots} - {first_arm}):
        ratios = []
        for kind in prompts:
            for temp in temps:
                ctrl = [r["decode_tps"] for r in rows if r["arm"] == first_arm and r["kind"] == kind
                        and r["temperature"] == temp]
                test = [r["decode_tps"] for r in rows if r["arm"] == arm and r["kind"] == kind
                        and r["temperature"] == temp]
                ratios.append(statistics.median(test) / statistics.median(ctrl))
        print(f"{arm}/{first_arm}: geometric mean of median ratios {statistics.geometric_mean(ratios):.3f} "
              f"over {len(ratios)} cells")
    print(f"\ntext identical to boot 0 in {len(rows) - len(diffs) - len(ref)}/{len(rows) - len(ref)} requests")
    for d in diffs:
        print(f"  DIFFERS: boot {d[0]} arm {d[1]} {d[2]} T={d[3]} rep {d[4]}")
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
