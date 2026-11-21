"""Where a drafted Hy4-flash reply leaves serial decoding (playbook 5.2): the first differing token, and that
position's logits from one-row steps against the window it was verified in.

    python tools/hy4_exact.py MODEL [--kind json] [--tokens 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--kind", default="json")
    p.add_argument("--tokens", type=int, default=256)
    a = p.parse_args()

    import torch
    from tokenizers import Tokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from hy4_probe import PROMPTS

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.app import EffortTemplate, text_view
    from tensorfold.families.hy4_flash.cuda.model import Engine, draft_decode, prefill, serial_decode
    from tensorfold.families.hy4_flash.cuda.weights import load

    view = text_view(Path(a.model))
    tok = Tokenizer.from_file(str(view / "tokenizer.json"))
    ids = tok.encode(EffortTemplate(ChatTemplate(view)).render([{"role": "user", "content": PROMPTS[a.kind]}],
                                                               tools=None, enable_thinking=False),
                     add_special_tokens=False).ids
    e = Engine(load(a.model, "cuda"), max_len=8192)
    greedy = Sampling(seed=0, temperature=0.0)
    serial = serial_decode(e, prefill(e, ids, greedy), a.tokens, greedy, stop_eos=False).tokens
    e.reset()
    trace: list = []
    drafted = draft_decode(e, ids, prefill(e, ids, greedy), a.tokens, greedy, stop_eos=False, drafter=None,
                           allow_copy=True, trace=trace).tokens
    if drafted == serial:
        print("equal: drafted copies give serial decoding's reply")
        return
    d = next(i for i, (x, y) in enumerate(zip(serial, drafted)) if x != y)
    at = len(ids) + d - 1                      # the row whose logits chose reply token d
    rnd = next(r for r in trace if r["base"] <= at < r["base"] + len(r["tokens"]))
    row = at - rnd["base"]
    print(f"first difference at reply token {d}: serial {serial[d]} drafted {drafted[d]}; verified in a "
          f"{len(rnd['tokens'])}-row window at {rnd['base']} (row {row}, copied {rnd['copied']})")
    out = {}
    for name, graphs in (("one-row eager", False), ("one-row graph", True)):
        e.reset()
        e.graphs, e.graph = graphs, None
        prefill(e, ids, greedy)
        logits = None
        for t in serial[:d]:                   # each reply token at its position, one row a call
            logits = e.forward([t], prompt=False)
        out[name] = logits[-1].float()
    e.reset()
    e.graphs = False
    prefill(e, ids, greedy)
    for t in serial[:rnd["base"] - len(ids)]:
        e.forward([t], prompt=False)
    out["window"] = e.forward(rnd["tokens"], prompt=False, every=True)[row].float()
    for name, v in out.items():
        top = torch.topk(v, 2)
        print(f"{name:14s} argmax {int(top.indices[0])} top2 {top.values.tolist()} max|diff vs one-row eager| "
              f"{float((v - out['one-row eager']).abs().max()):.6f}")


if __name__ == "__main__":
    main()
