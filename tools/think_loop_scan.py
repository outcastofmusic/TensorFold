"""Run the think-loop detector (``tensorfold.engine.think_loop``) over saved reasoning, offline.

Every JSON file under the given roots is searched for strings at keys named ``reasoning`` or ``reasoning_content``.
Each block of at least ``--min-chars`` is tokenized with the served tokenizer and fed to ``LoopDetector`` as the
engine would feed it. The report counts blocks, tokens, and fires by arm and by length, and prints each fire's text
around the firing token, for a person to label as a loop or a false fire.

  python tools/think_loop_scan.py --tokenizer MODEL/tokenizer.json ~/work/tfeval ~/work/eval --out scan.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfold.engine.think_loop import LoopDetector  # noqa: E402

KEYS = ("reasoning", "reasoning_content")


def blocks(obj, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in KEYS and isinstance(v, str):
                out.append(v)
            else:
                blocks(v, out)
    elif isinstance(obj, list):
        for v in obj:
            blocks(v, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--min-chars", type=int, default=2000)
    ap.add_argument("--out")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    seen: set[str] = set()
    texts: list[tuple[str, str]] = []
    for root in args.roots:
        for path in Path(root).rglob("*.json"):
            try:
                doc = json.loads(path.read_text(errors="ignore"))
            except (ValueError, OSError):
                continue
            found: list[str] = []
            blocks(doc, found)
            for text in found:
                key = hashlib.sha1(text.encode()).hexdigest()
                if len(text) >= args.min_chars and key not in seen:     # each distinct block once
                    seen.add(key)
                    texts.append((str(path), text))
    fires, tokens_total, buckets = [], 0, {}
    for path, text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        tokens_total += len(ids)
        bucket = "<2k" if len(ids) < 2048 else "2k-8k" if len(ids) < 8192 else ">=8k"
        b = buckets.setdefault(bucket, [0, 0])
        b[0] += 1
        d = LoopDetector()
        for i, t in enumerate(ids):
            why = d.observe(t)
            if why:
                b[1] += 1
                around = tok.decode(ids[max(0, i - 120):i + 1])
                fires.append({"path": path, "tokens": len(ids), "at": i, "why": why, "before": around})
                break
    print(f"{len(texts)} distinct reasoning blocks of {args.min_chars}+ characters, {tokens_total} tokens")
    for name in ("<2k", "2k-8k", ">=8k"):
        n, f = buckets.get(name, [0, 0])
        print(f"  {name:6} {n:5} blocks, {f} fired")
    by_arm: dict[str, int] = {}
    for f in fires:
        arm = f["why"].split("(")[0]
        by_arm[arm] = by_arm.get(arm, 0) + 1
    print(f"fired: {len(fires)} of {len(texts)} by arm {by_arm}")
    for f in fires:
        print(f"\n--- {f['path']} ({f['tokens']} tokens) fired at {f['at']}: {f['why']}\n{f['before'][-600:]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"blocks": len(texts), "tokens": tokens_total, "buckets": buckets,
                                              "fires": fires}, indent=1))


if __name__ == "__main__":
    main()
