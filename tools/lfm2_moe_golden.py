"""Write golden logits for LFM2.5-24B-A2B from the fp32 reference forward on the CPU, for the CUDA engine to match.

Three prompts, rendered with the checkpoint's chat_template.jinja as the CUDA server renders them (thinking always
on) and encoded without special tokens: a short chat, a tool-call prompt with 2 tools, and a prompt of about 2,000
tokens (the checkpoint's LICENSE, then a question). For each, the file holds the prompt ids, the top-20 logits at the
prompt's last 64 positions, 32 greedy tokens, the top-20 logits each greedy token was chosen from, and the smallest gap
between a step's top two logits (an engine whose logits err by less than half of it repeats the greedy tokens).

    python tools/lfm2_moe_golden.py --model /models/LFM2.5-24B-A2B-FP8 --out /models/LFM2.5-24B-A2B-FP8-golden
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.lfm2_moe.reference import Reference, State

STEPS, LAST, TOP, CHUNK = 32, 64, 20, 512
LONG_DOC_TOKENS = 1970

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather and the next days' forecast for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights", "description": "Search one-way flights between two airports on a date.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "IATA code of the departure airport"},
            "destination": {"type": "string", "description": "IATA code of the arrival airport"},
            "date": {"type": "string", "description": "Departure date, YYYY-MM-DD"}},
            "required": ["origin", "destination", "date"]}}},
]


def prompts(model_dir: Path, tok) -> dict[str, dict]:
    license_text = (model_dir / "LICENSE").read_text()
    doc = tok.decode(tok.encode(license_text, add_special_tokens=False).ids[:LONG_DOC_TOKENS])
    return {
        "chat": {"messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                 "tools": None},
        "tools": {"messages": [
            {"role": "system", "content": "You are a travel assistant. Use the tools when they help."},
            {"role": "user", "content": "I fly from Athens to Paris on 2026-10-20. What will the weather be "
                                        "there, and which flights can I take?"}],
                  "tools": TOOLS},
        "long": {"messages": [{"role": "user", "content": doc + "\n\nSummarize the license above in three "
                                                                "bullet points."}],
                 "tools": None},
    }


def top(rows: torch.Tensor) -> list[dict]:
    """Each row's top-20 ids and logits (descending; ties by lower id) and its logsumexp."""

    out = []
    for r in rows:
        order = torch.sort(r, descending=True, stable=True).indices[:TOP]
        out.append({"ids": order.tolist(), "logits": r[order].tolist(), "logsumexp": torch.logsumexp(r, 0).item()})
    return out


def greedy(ref: Reference, ids: list[int]) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """32 greedy tokens, the prompt's last 64 rows' logits and each step's logits (the prompt fed in chunks)."""

    state, rows = State(), []
    for s in range(0, len(ids), CHUNK):
        rows.append(ref.forward(ids[s:s + CHUNK], state)[-LAST:])
    prompt_rows = torch.cat(rows)[-LAST:]
    steps, out = [prompt_rows[-1]], []
    while len(out) < STEPS:
        out.append(int(torch.argmax(steps[-1])))
        if len(out) < STEPS:
            steps.append(ref.forward([out[-1]], state)[-1])
    return out, prompt_rows, torch.stack(steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="prompt names to write (default: all three)")
    ap.add_argument("--commit", default="unknown", help="the reference's git commit, recorded in each file")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    template = ChatTemplate(args.model)
    ref = Reference(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, p in prompts(args.model, tok).items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        text = template.render(p["messages"], tools=p["tools"], enable_thinking=True)
        ids = tok.encode(text, add_special_tokens=False).ids
        with torch.no_grad():
            tokens, prompt_rows, step_rows = greedy(ref, ids)
        top2 = step_rows.topk(2, dim=-1).values
        record = {
            "name": name, "model": args.model.name, "reference_commit": args.commit,
            "reference": "tensorfold.families.lfm2_moe.reference, fp32, FP8 blocks decoded exactly",
            **p, "text": text, "prompt_ids": ids,
            "prompt_positions": list(range(len(ids) - prompt_rows.shape[0], len(ids))),
            "prompt_top": top(prompt_rows),
            "greedy": tokens, "greedy_text": tok.decode(tokens, skip_special_tokens=False),
            "step_top": top(step_rows),
            "min_top2_gap": float((top2[:, 0] - top2[:, 1]).min()),
            "seconds": round(time.time() - t0, 1),
        }
        (args.out / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(f"{name}: {len(ids)} prompt tokens, {record['seconds']} s, smallest top-2 gap "
              f"{record['min_top2_gap']:.3f}: {record['greedy_text']!r}", flush=True)


if __name__ == "__main__":
    main()
