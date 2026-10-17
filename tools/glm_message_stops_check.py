"""Five different last questions after one long history, against a running GLM server: the first prefills it all, the
next four resume where the last user message starts, and every reply equals the same prompt's cache-off reply.

    python3 tools/glm_message_stops_check.py [--url http://127.0.0.1:31005] [--history-tokens 6000]

The cache-off reference is each prompt again with ``"draft": false`` (a fresh prefill and serial decoding, keeping no
state). ``--save`` writes the run, and ``--against`` compares with a saved one instead: run with ``--save`` on a server
started with TF_GLM_CACHE_ENTRIES=0, then with ``--against`` and the same ``--nonce`` on the cached server. The
expected resume point is the last ``<|user|>`` of the prompt ``/tokenize`` renders. Greedy replies; standard library
only. Exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request

QUESTIONS = ["Which warehouse shipped the most items?", "Summarize the notes in two sentences.",
             "List every day on which item 4821 moved.", "What is the largest quantity mentioned?",
             "Name one note that contradicts another."]


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def history(nonce: str, turns: int) -> list[dict]:
    rng = random.Random(nonce)
    messages = [{"role": "system", "content": f"Session {nonce}. You keep a logistics notebook and answer briefly."}]
    for turn in range(turns):
        notes = " ".join(f"Note {turn}.{k}: item {rng.randint(1000, 9999)} left warehouse {rng.choice('ABCDEFG')} on "
                         f"day {rng.randint(1, 365)} with quantity {rng.randint(1, 500)}." for k in range(12))
        messages += [{"role": "user", "content": f"Record these notes. {notes}"},
                     {"role": "assistant", "content": f"Recorded the 12 notes of batch {turn}."}]
    return messages


def build(url: str, model: str, nonce: str, target: int) -> list[dict]:
    """The history grown a turn at a time until /tokenize counts ``target`` tokens."""

    turns = 1
    while post(url, "/tokenize", {"model": model, "messages": history(nonce, turns),
                                  "add_generation_prompt": False})["count"] < target:
        turns += 1
    return history(nonce, turns)


def last_user(url: str, model: str, messages: list[dict]) -> tuple[int, int]:
    """(the prompt's token count, the position of its last <|user|>) as /tokenize renders it."""

    got = post(url, "/tokenize", {"model": model, "messages": messages, "return_token_strs": True})
    strs = got["token_strs"]
    if "<|user|>" not in strs:
        raise SystemExit("/tokenize shows no <|user|> token: is this a GLM server?")
    return got["count"], len(strs) - 1 - strs[::-1].index("<|user|>")


def ask(url: str, model: str, messages: list[dict], tokens: int, draft: bool) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0, "stream": False,
            "return_token_ids": True}
    if not draft:
        body["draft"] = False
    reply = post(url, "/v1/chat/completions", body)
    usage, extra = reply["usage"], reply.get("tensorfold") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {"prompt": usage["prompt_tokens"], "cached": cached, "ids": extra.get("token_ids"),
            "sha": extra.get("token_sha")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:31005")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--history-tokens", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--nonce", default=f"{random.getrandbits(48):012x}", help="the history's seed (a fresh one: no "
                    "earlier request shares it)")
    ap.add_argument("--save", help="write this run's replies here")
    ap.add_argument("--against", help="compare with a run --save wrote instead of sending draft:false references")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    base = build(url, args.model, args.nonce, args.history_tokens)
    runs = []
    for q in QUESTIONS:
        messages = base + [{"role": "user", "content": q}]
        count, point = last_user(url, args.model, messages)
        runs.append({"question": q, "count": count, "point": point,
                     **ask(url, args.model, messages, args.max_tokens, True)})
    if args.against:
        with open(args.against) as f:
            refs = [r["ids"] for r in json.load(f)["runs"]]
    elif args.save:
        refs = [r["ids"] for r in runs]
    else:
        refs = [ask(url, args.model, base + [{"role": "user", "content": q}], args.max_tokens, False)["ids"]
                for q in QUESTIONS]
    bad = 0
    print(f"nonce {args.nonce}: a history of {runs[0]['point']} tokens before the last user message")
    for i, (run, ref) in enumerate(zip(runs, refs)):
        want = 0 if i == 0 else run["point"]
        if run["count"] != run["prompt"]:
            print(f"  /tokenize counts {run['count']} tokens, the reply {run['prompt']}: the resume point is a guess")
        same = run["ids"] == ref
        ok = same and (run["cached"] == want or args.save is not None)
        bad += not ok
        print(f"  {i + 1}: prompt {run['prompt']} cached {run['cached']} (expected {want}) sha {run['sha']} "
              f"{'same' if same else 'DIFFERENT'} reply {'ok' if ok else 'FAIL'}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump({"nonce": args.nonce, "runs": runs}, f)
    print("PASS" if not bad else f"FAIL: {bad} of {len(runs)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
