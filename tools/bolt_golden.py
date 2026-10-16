"""Write golden logits for Nemotron 3.5 Bolt from the fp32 reference forward, for the CUDA engine and drafter to match.

Three prompts, rendered with the checkpoint's chat_template.jinja the way the CUDA server renders them and encoded
without special tokens: a short chat, a tool-call prompt with 2 tools, and a 2,000-token prompt (the model card's
first 1,970 tokens from "## Description", then a question). For each, the file holds the prompt ids, the top-20
logits at the prompt's last 64 positions, 32 greedy tokens and the top-20 logits each greedy token was chosen from.
README.md in the output folder describes the format.

    python tools/bolt_golden.py --model /models/Nemotron-3.5-Bolt-74B-A9B-24GB --out golden/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.nemotron_bolt import reference as ref

STEPS, LAST, TOP = 32, 64, 20
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


def long_document(model_dir: Path, tok) -> str:
    card = (model_dir / "README.md").read_text()
    body = card[card.index("## Description"):]
    return tok.decode(tok.encode(body, add_special_tokens=False).ids[:LONG_DOC_TOKENS])


def prompts(model_dir: Path, tok) -> dict[str, dict]:
    doc = long_document(model_dir, tok)
    return {
        "chat": {"messages": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
                 "tools": None, "enable_thinking": False},
        "tools": {"messages": [
            {"role": "system", "content": "You are a travel assistant. Use the tools when they help."},
            {"role": "user", "content": "I fly from Athens to Paris on 2026-10-20. What will the weather be "
                                        "there, and which flights can I take?"}],
                  "tools": TOOLS, "enable_thinking": False},
        "long": {"messages": [{"role": "user", "content": doc + "\n\nSummarize the model card above in three "
                                                                "bullet points."}],
                 "tools": None, "enable_thinking": True},
    }


def top(rows: torch.Tensor) -> list[dict]:
    """Each row's top-20 ids and logits (descending; ties by lower id) and its logsumexp."""

    out = []
    for r in rows:
        order = torch.sort(r, descending=True, stable=True).indices[:TOP]
        out.append({"ids": order.tolist(), "logits": r[order].tolist(), "logsumexp": torch.logsumexp(r, 0).item()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", help="prompt names to write (default: all three)")
    ap.add_argument("--commit", default="unknown", help="the reference's git commit, recorded in each file")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    template = ChatTemplate(args.model)
    ck = ref.Checkpoint(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, p in prompts(args.model, tok).items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        text = template.render(p["messages"], tools=p["tools"], enable_thinking=p["enable_thinking"])
        ids = tok.encode(text, add_special_tokens=False).ids
        greedy, prompt_rows, step_rows = ref.greedy(ck, torch.tensor(ids), STEPS, last=LAST)
        record = {
            "name": name, "model": args.model.name, "reference_commit": args.commit,
            "reference": "tensorfold.families.nemotron_bolt.reference, fp32, exact decoded weights, dt floor 0",
            **p, "text": text, "prompt_ids": ids,
            "prompt_positions": list(range(len(ids) - prompt_rows.shape[0], len(ids))),
            "prompt_top": top(prompt_rows),
            "greedy": greedy.tolist(), "greedy_text": tok.decode(greedy.tolist(), skip_special_tokens=False),
            "step_top": top(step_rows),
            "seconds": round(time.time() - t0, 1),
        }
        (args.out / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(f"{name}: {len(ids)} prompt tokens, {record['seconds']} s: {record['greedy_text']!r}", flush=True)


if __name__ == "__main__":
    main()
