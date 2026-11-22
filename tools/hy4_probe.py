"""Hy4-flash speed probe in one process (no server): serial against drafted decode by text kind, and a prompt chunk's
kernel profile.

    python tools/hy4_probe.py MODEL OUT.json [--tokens 256] [--profile] [--temperature 0.9]
                              [--drafters nvfp4,slice,nvfp4+adaptive@0.15] [--trace TRACE.json]

Each kind decodes serial (one-row graph steps), then drafted (DFly2 and copies, DFly2 alone for each ``--drafters``
draft head, copies alone), from the same prompt and key; the drafted tokens must equal the serial ones. A drafter is
a draft head (``nvfp4`` or ``slice``, TF_HY4_DRAFT_HEAD), and ``+adaptive@FLOOR`` turns on confidence-head depth
(TF_HY4_DFLY_ADAPTIVE, TF_HY4_DFLY_FLOOR), ``+greedy`` drafts greedily under sampling (TF_HY4_DFLY_KEYED=0), and
``+copy`` lets copies draft too, and ``+depth@N`` caps a round at N drafts (TF_HY4_DFLY_DEPTH). The drafted arms run in a rotated order a kind, so no arm is always the
first to read rejected drafts' Engram rows cold (plan B9). ``--trace`` keeps every DFly2 round (its confidences and
accepted count) for the confidence head's calibration. ``--profile`` times one 2,048-row prompt chunk by kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROMPTS = {
    "chat": "Explain the difference between TCP and UDP to a new engineer, with an everyday analogy.",
    "code": "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and tests.",
    "prose": "Write a 300-word short story about a lighthouse keeper who finds a message in a bottle.",
    "json": "Return a JSON array of 12 objects describing GPUs, each with name, vendor, memory_gb, tdp_w and year.",
    "rewrite": "Rewrite the following text, changing only British spellings to American ones:\n\n" + (
        "The colour of the harbour changed as the theatre lights came on. We organised the catalogue, analysed the "
        "programme and travelled to the centre to honour the neighbour who had realised the favour. ") * 6,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0, help="the keyed sampler's temperature (seed 0, top_k off, top_p 1)")
    p.add_argument("--drafters", default="nvfp4", help="comma-separated draft heads, each optionally +adaptive@FLOOR")
    p.add_argument("--trace", help="write every DFly2 round's confidences and accepted drafts here")
    p.add_argument("--profile-decode", action="store_true", help="one eager decode row by kernel, and its host staging")
    a = p.parse_args()

    import torch
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.app import EffortTemplate, text_view
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import Engine, draft_decode, prefill, serial_decode
    from tensorfold.families.hy4_flash.cuda.weights import load

    view = text_view(Path(a.model))
    tok = Tokenizer.from_file(str(view / "tokenizer.json"))
    template = EffortTemplate(ChatTemplate(view))
    w = load(a.model, "cuda")
    e = Engine(w, max_len=16384)
    drafters = {}
    for spec in a.drafters.split(","):
        head, *opts = spec.split("+")
        adaptive = next((o for o in opts if o.startswith("adaptive")), "")
        env = {"TF_HY4_DRAFT_HEAD": head, "TF_HY4_DFLY_ADAPTIVE": "1" if adaptive else "0",
               "TF_HY4_DFLY_KEYED": "0" if "greedy" in opts else "1"}
        if "@" in adaptive:
            env["TF_HY4_DFLY_FLOOR"] = adaptive.split("@", 1)[1]
        depth = next((o for o in opts if o.startswith("depth@")), "")
        env["TF_HY4_DFLY_DEPTH"] = depth.split("@", 1)[1] if depth else "5"
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            drafters[spec] = DFly(Path(a.model), e)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    first_spec = next(iter(drafters))
    key = Sampling(seed=0, temperature=a.temperature, top_k=0, top_p=1.0)     # run.sh's hy4: top_k off, top_p 1
    report: dict = {"tokens": a.tokens, "temperature": a.temperature, "drafters": list(drafters)}
    traces: dict = {}
    drafted = [("dfly+copy", {"drafter": drafters[first_spec], "allow_copy": True})]
    drafted += [(f"dfly:{spec}", {"drafter": d, "allow_copy": "copy" in spec.split("+")}) for spec, d in drafters.items()]
    drafted += [("copy", {"drafter": None, "allow_copy": True})]
    for i, (name, text) in enumerate(PROMPTS.items()):
        ids = tok.encode(template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False),
                         add_special_tokens=False).ids
        row: dict = {"prompt": len(ids)}
        serial = None
        turn = i % len(drafted)
        for arm, kw in [("serial", None), *drafted[turn:], *drafted[:turn]]:
            e.reset()
            first = prefill(e, ids, key)
            if kw is None:
                res = serial_decode(e, first, a.tokens, key, stop_eos=False)
                serial = res.tokens
                row[arm] = {"tps": round(res.tokens_per_second, 2)}
            else:
                trace: list = []
                res = draft_decode(e, ids, first, a.tokens, key, stop_eos=False, trace=trace, **kw)
                st = res.stats
                row[arm] = {"tps": round(res.tokens_per_second, 2), "equal": res.tokens == serial,
                            "rounds": st["rounds"], "drafted": st["drafted"], "accepted": st["accepted"],
                            "per_round": round((st["accepted"] + st["rounds"]) / max(1, st["rounds"]), 2),
                            "draft_ms_round": round(1000 * st["draft_s"] / max(1, st["rounds"]), 2)}
                traces[f"{name}/{arm}"] = [{"drafts": len(t["tokens"]) - 1, "accepted": t["accepted"],
                                            "confidence": t.get("confidence")} for t in trace if not t["copied"]]
        report[name] = row
        print(name, json.dumps(row), flush=True)
    if a.trace:
        Path(a.trace).write_text(json.dumps(traces))
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        ids = list(range(1000, 1000 + 2048))
        e.reset()
        e.forward(ids, prompt=True)
        torch.cuda.synchronize()
        e.reset()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            e.forward(ids, prompt=True)
            torch.cuda.synchronize()
        report["chunk_s"] = round(time.perf_counter() - t0, 3)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        report["profile"] = table
        print(table, flush=True)
    if a.profile_decode:
        from torch.profiler import ProfilerActivity, profile

        e.reset()
        e.graphs = False
        prefill(e, list(range(1000, 1600)), key)
        for t in range(5):
            e.forward([2000 + t], prompt=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(20):
            e._stage_engram([3000 + t])
        report["engram_stage_ms"] = round(1000 * (time.perf_counter() - t0) / 20, 3)
        steps = []
        for t in range(10):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e.forward([2100 + t], prompt=False)
            torch.cuda.synchronize()
            steps.append(time.perf_counter() - t1)
        report["eager_step_ms"] = round(1000 * sorted(steps)[5], 2)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            e.forward([2200], prompt=False)
            torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
        report["decode_profile"] = table
        print(f"engram staging {report['engram_stage_ms']} ms a row, eager step {report['eager_step_ms']} ms", flush=True)
        print(table, flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    os.sys.stdout.flush()


if __name__ == "__main__":
    main()
