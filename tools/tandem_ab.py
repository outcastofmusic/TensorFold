"""A-B-A decode-speed A/B of TensorFold env flags on Flash Next, with an exactness check between arms.

Each arm is one server boot of this checkout with its env flags (the others off). The order runs A, B, A by default,
so drift between boots shows up as A1 against A2. Every arm decodes the same prompts (prose, code, json, edit, copy,
and optionally a long-context one) greedy and at T=0.7 with a fixed seed. TensorFold samples with keyed draws, so a
speed flag must leave the text identical at both temperatures: the script compares every arm's text with A1's and
lists each difference.

Run it only when the box is free: it boots its own server (container ``tftest-ab-<arm>``, port 30010 by default,
the checkout mounted over the image's install) and refuses to start below ``--min-avail-gib`` of MemAvailable. Each
boot's server log is saved beside ``--out`` (the self-check and loop-close lines land there).

  python3 tools/tandem_ab.py --model /home/agis/models/Qwen3.8-Flash-Next-exl3-3.05bpw \\
      --arms 'A=;B=TENSORFOLD_STAIRCUT=1' --order A,B,A --reps 3 --out ab-staircut.json

Standard library only.
"""

from __future__ import annotations

import argparse
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


def run_one(base: str, prompt: str, temperature: float, tokens: int) -> dict:
    body = {"model": "ab", "max_tokens": tokens, "temperature": temperature, "stream": True, "seed": 1234,
            "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
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
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(piece)
    n = (usage or {}).get("completion_tokens") or 0
    tps = (n - 1) / (last - first) if first is not None and last is not None and last > first and n > 1 else 0.0
    return {"text": "".join(text), "tokens": n, "decode_tps": round(tps, 2),
            "ttft_s": round((first or start) - start, 3)}


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
                run_one(base, prompt, 0.0, 32)
            before = metrics(base)
            for rep in range(args.reps):
                for kind, prompt in prompts.items():
                    for temp in (0.0, 0.7):
                        r = run_one(base, prompt, temp, args.tokens)
                        results.append({"boot": i, "arm": arm, "kind": kind, "temperature": temp, "rep": rep, **r})
                        print(f"[ab] {arm}{i} {kind:6} T={temp} rep {rep}: {r['decode_tps']} tok/s "
                              f"({r['tokens']} tokens)", flush=True)
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
    for kind in prompts:
        for temp in (0.0, 0.7):
            cells = [statistics.median(r["decode_tps"] for r in rows
                                       if r["boot"] == b and r["kind"] == kind and r["temperature"] == temp)
                     for b, _ in boots]
            print(f"{kind:8} {temp:>4} " + " ".join(f"{c:<7.1f}" for c in cells))
    first_arm = boots[0][1]
    for arm in sorted({a for _, a in boots} - {first_arm}):
        ratios = []
        for kind in prompts:
            for temp in (0.0, 0.7):
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
