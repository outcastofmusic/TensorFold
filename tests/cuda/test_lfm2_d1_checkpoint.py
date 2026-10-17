"""The real d1-3B checkpoint on the GPU: answers against transformers' own bf16 Lfm2-VL on each question's full prompt,
a question alone against beside others, a picture, latency, memory, and the HTTP routes.

    TF_D1_MODEL=/home/agis/models/d1-3B (the default)
"""

from __future__ import annotations

import http.client
import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from tensorfold.families.lfm2.decisions import answer, question

D1 = Path(os.environ.get("TF_D1_MODEL", "/home/agis/models/d1-3B"))
GIB = 1024 ** 3

pytestmark = pytest.mark.skipif(not (D1 / "model.safetensors").exists(), reason="needs the d1-3B checkpoint")

TEAM = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                     "fraud": "Suspected unauthorised use"}}
MODELS = {"type": "choice", "instructions": "Which model should answer this request?",
          "criteria": {"qwen-flash-next": "fast general model, good at code and tools",
                       "qwen-flash-next-fp4": "the same model, more accurate, slower",
                       "glm-5.3-flash": "strongest reasoning, long careful answers",
                       "nemotron-bolt": "small fast model for simple questions"}}
CASES = [
    ("I was charged twice this month, please refund one of them.", {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": TEAM,
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "Today", "Blocking the customer now"]}}),
    ("The app crashes when I open settings.", {"team": TEAM}),
    ({"request": "Prove that there are infinitely many primes of the form 4k+3, then formalize it in Lean 4."}, {
        "budget": {"type": "score", "instructions": "How much thinking does this request need before answering?",
                   "criteria": ["none", "a few hundred tokens", "a few thousand tokens", "as much as allowed"]},
        "model": MODELS}),
    ({"request": "What is the capital of France?"}, {"model": MODELS}),
    ({"request": "Summarize this article in two sentences.", "reply": ""}, {
        "bad": {"type": "noul", "instructions": "Is the reply empty, cut off, or repeating itself?"}}),
    ({"request": "List three fruits.", "reply": "Apple, banana, apple, banana, apple, banana, apple, banana"}, {
        "bad": {"type": "noul", "instructions": "Is the reply empty, cut off, or repeating itself?"}}),
]


def top(a: dict):
    return a.get("choice") or (a["noul"] > 0.5 if a["type"] == "noul" else max(a["probabilities"],
                                                                                   key=a["probabilities"].get))


def probabilities(a: dict) -> list[float]:
    return [a["noul"]] if a["type"] == "noul" else list(a["probabilities"].values())


def reference_answers(model_dir: Path, device: str, fmt, cases=CASES) -> list[dict]:
    """transformers' own Lfm2VlForConditionalGeneration in bf16, each question's full prompt alone (no remote code)."""

    from transformers import Lfm2VlForConditionalGeneration

    model = Lfm2VlForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch.bfloat16,
                                                           attn_implementation="sdpa").to(device).eval()
    out = []
    with torch.inference_mode():
        for state, questions in cases:
            got = {}
            for name, raw in questions.items():
                q = question(raw)
                ids = fmt.encode(fmt.prefix(state) + fmt.suffix(q))
                logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=1).logits[0, -1].float()
                got[name] = answer(q, [max(float(logits[t]) for t in forms) for forms in fmt.readout(q)])
            out.append(got)
    del model
    return out


def compare(ours: list[dict], theirs: list[dict]) -> tuple[int, int, float]:
    """(questions whose top answer agrees, questions, the largest probability difference)."""

    same = total = 0
    worst = 0.0
    for a, b in zip(ours, theirs):
        for name in b:
            total += 1
            same += top(a[name]) == top(b[name])
            worst = max(worst, max(abs(x - y) for x, y in zip(probabilities(a[name]), probabilities(b[name]))))
    return same, total, worst


@pytest.fixture(scope="module")
def engine():
    from tensorfold.families.lfm2.engine import Engine

    torch.cuda.reset_peak_memory_stats()
    return Engine(D1, context=4096, context_explicit=True)


def test_memory_after_load(engine):
    held = torch.cuda.memory_allocated() / GIB
    print(f"\nd1-3B loaded: {held:.2f} GiB allocated, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB, "
          f"{engine.load_s:.1f} s")
    assert held < 7.0


def test_answers_match_transformers_bf16(engine):
    ours = [engine.decide(state, questions)["answers"] for state, questions in CASES]
    theirs = reference_answers(D1, "cuda", engine.fmt)
    same, total, worst = compare(ours, theirs)
    print(f"\ntop answers equal on {same}/{total} questions, largest probability difference {worst:.4f}")
    for state, a, b in zip(CASES, ours, theirs):
        print(json.dumps(state[0])[:60], {k: (top(a[k]), top(b[k])) for k in b})
    assert same >= total - 1 and worst < 0.06


def test_a_question_alone_reads_as_beside_others(engine):
    state, questions = CASES[0]
    together = engine.decide(state, questions)["answers"]
    for name, raw in questions.items():
        alone = engine.decide(state, {name: raw})["answers"][name]
        assert top(alone) == top(together[name])
        assert max(abs(x - y) for x, y in zip(probabilities(alone), probabilities(together[name]))) < 0.02


def test_a_picture(engine):
    from PIL import Image

    red = Image.new("RGB", (384, 288), (220, 20, 20))
    colour = {"type": "choice", "instructions": "What colour fills the picture?",
              "criteria": {"red": "Red", "green": "Green", "blue": "Blue"}}
    out = engine.decide(None, {"colour": colour}, images=[red])["answers"]["colour"]
    assert out["choice"] == "red", out


def _median_ms(fn, runs: int = 20) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def test_latency(engine):
    state, questions = CASES[0]
    one = _median_ms(lambda: engine.decide(state, {"refund": questions["refund"]}))
    three = _median_ms(lambda: engine.decide(state, questions))
    long_state = " ".join(["The customer wrote again about the double charge and the refund."] * 250)
    long = _median_ms(lambda: engine.decide(long_state, {"refund": questions["refund"]}), runs=10)
    print(f"\none question {one:.1f} ms, three questions {three:.1f} ms, a "
          f"{len(engine.fmt.encode(long_state))}-token state {long:.1f} ms (medians); "
          f"peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
    assert one < 80 and three < 120


@contextmanager
def _serving(app):
    from tensorfold.cuda.http import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def test_http_routes_on_one_kept_connection(engine):
    from tensorfold.families import lfm2

    app = lfm2.CUDA_APP(engine, D1, "d1-3b")
    state, questions = CASES[0]
    with _serving(app) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        times = []
        for _ in range(12):
            start = time.perf_counter()
            connection.request("POST", "/v1/systemone", json.dumps({"state": state, "questions": {
                "refund": questions["refund"]}}), {"Content-Type": "application/json"})
            reply = connection.getresponse()
            body = json.loads(reply.read())
            times.append((time.perf_counter() - start) * 1000)
            assert reply.status == 200 and body["answers"]["refund"]["noul"] > 0.5
        connection.request("POST", "/v1/chat/completions", json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
                           {"Content-Type": "application/json"})
        refused = connection.getresponse()
        assert refused.status == 400 and b"writes no text" in refused.read()
        connection.request("POST", "/v1/decisions", json.dumps({"input": state, "questions": [
            {"id": "refund", "type": "yes_no", "question": "The customer asks for a refund."}]}),
            {"Content-Type": "application/json"})
        decided = connection.getresponse()
        assert decided.status == 200 and json.loads(decided.read())["answers"]["refund"]["probabilities"]["yes"] > 0.5
        connection.close()
    print(f"\n/v1/systemone over one kept connection: median {statistics.median(times[2:]):.1f} ms")
    assert statistics.median(times[2:]) < 100
