"""Hy4-flash speed probe in one process (no server): serial against drafted decode by text kind, and a prompt chunk's
kernel profile.

    python tools/hy4_probe.py MODEL OUT.json [--tokens 256] [--profile]

Each kind decodes greedily serial (one-row graph steps), then drafted (DFly2 and copies, then DFly2 alone), from the
same prompt; the drafted tokens must equal the serial ones. ``--profile`` times one 2,048-row prompt chunk by kernel.
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
    dfly = DFly(Path(a.model), e)
    greedy = Sampling(seed=0, temperature=0.0)
    report: dict = {"tokens": a.tokens}
    for name, text in PROMPTS.items():
        ids = tok.encode(template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False),
                         add_special_tokens=False).ids
        row: dict = {"prompt": len(ids)}
        serial = None
        for arm, kw in (("serial", None), ("dfly+copy", {"drafter": dfly, "allow_copy": True}),
                        ("dfly", {"drafter": dfly, "allow_copy": False}), ("copy", {"drafter": None,
                                                                                    "allow_copy": True})):
            e.reset()
            first = prefill(e, ids, greedy)
            if kw is None:
                res = serial_decode(e, first, a.tokens, greedy, stop_eos=False)
                serial = res.tokens
                row[arm] = {"tps": round(res.tokens_per_second, 2)}
            else:
                res = draft_decode(e, ids, first, a.tokens, greedy, stop_eos=False, **kw)
                st = res.stats
                row[arm] = {"tps": round(res.tokens_per_second, 2), "equal": res.tokens == serial,
                            "rounds": st["rounds"], "drafted": st["drafted"], "accepted": st["accepted"],
                            "per_round": round((st["accepted"] + st["rounds"]) / max(1, st["rounds"]), 2),
                            "draft_ms_round": round(1000 * st["draft_s"] / max(1, st["rounds"]), 2)}
        report[name] = row
        print(name, json.dumps(row), flush=True)
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
    Path(a.out).write_text(json.dumps(report, indent=1))
    os.sys.stdout.flush()


if __name__ == "__main__":
    main()
