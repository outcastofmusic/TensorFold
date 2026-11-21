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
    rec: dict = {}
    from tensorfold.families.hy4_flash.cuda import kernels as K
    from tensorfold.families.hy4_flash.cuda import model as MM

    inner_attention = K.attention
    seen: dict = {}

    def attention(qa, cache, lo, cnt, sel, tok, sinks, scale, out, scratch=None):
        if cache.data_ptr() == e.latent[0].data_ptr() and seen.get("on"):      # layer 0's call
            seen[seen["on"]] = {"qa": qa.clone(), "lo": lo.clone(), "cnt": cnt.clone(), "sel": sel.clone(),
                                "cache": cache[:at + 1].clone()}
        return inner_attention(qa, cache, lo, cnt, sel, tok, sinks, scale, out, scratch)

    MM.K.attention = attention

    def watch(tag):                            # each call's per-layer streams and sublayer outputs, by layer
        layer_fn, attn, mlp, eg = e._layer, e._attention, e._mlp, e._engram
        def layer(i, lay, x, prompt):
            y = layer_fn(i, lay, x, prompt)
            rec.setdefault(tag, {})[("layer", i)] = y.clone()
            return y
        def attention(i, a, x, prompt):
            y = attn(i, a, x, prompt)
            rec.setdefault(tag, {})[("attn", i)] = y.clone()
            return y
        def moe(i, m, x, prompt):
            y = mlp(i, m, x, prompt)
            rec.setdefault(tag, {})[("mlp", i)] = y.clone()
            return y
        def engram(j, g, x):
            y = eg(j, g, x)
            rec.setdefault(tag, {})[("engram", e.c.engram_layers[j])] = y.clone()
            return y
        e._layer, e._attention, e._mlp, e._engram = layer, attention, moe, engram
        return lambda: setattr(e, "_layer", layer_fn) or setattr(e, "_attention", attn) or \
            setattr(e, "_mlp", mlp) or setattr(e, "_engram", eg)

    out = {}
    for name, graphs in (("one-row eager", False), ("one-row graph", True)):
        e.reset()
        e.graphs, e.graph = graphs, None
        prefill(e, ids, greedy)
        logits = None
        for i, t in enumerate(serial[:d]):     # each reply token at its position, one row a call
            undo = watch(name) if not graphs and i == d - 1 else None
            seen["on"] = "one" if undo else None
            logits = e.forward([t], prompt=False)
            seen["on"] = None
            if undo:
                undo()
        out[name] = logits[-1].float()
    e.reset()
    e.graphs = False
    prefill(e, ids, greedy)
    for t in serial[:rnd["base"] - len(ids)]:
        e.forward([t], prompt=False)
    undo = watch("window")
    seen["on"] = "win"
    out["window"] = e.forward(rnd["tokens"], prompt=False, every=True)[row].float()
    seen["on"] = None
    undo()
    a, b = seen["one"], seen["win"]
    print(f"layer 0 attention inputs: qa row equal {torch.equal(a['qa'][-1], b['qa'][row])}, "
          f"cnt {int(a['cnt'][-1])} vs {int(b['cnt'][row])}, lo {int(a['lo'][-1])} vs {int(b['lo'][row])}, "
          f"sel {int(a['sel'][-1])} vs {int(b['sel'][row])}")
    rows_differ = (a["cache"] != b["cache"]).any(-1).nonzero().flatten().tolist()
    print(f"layer 0 latent rows that differ (positions): {rows_differ[:20]} of {at + 1}")
    one, win = rec["one-row eager"], rec["window"]
    for key in sorted(one, key=lambda k: (k[1], ["engram", "attn", "mlp", "layer"].index(k[0]))):
        a, b = one[key].float().reshape(1, -1) if one[key].shape[0] == 1 else one[key][-1:].float().reshape(1, -1), \
            win[key][row:row + 1].float().reshape(1, -1)
        diff = float((a - b).abs().max())
        if diff > 0:
            print(f"first difference: {key[0]} of layer {key[1]}, max {diff:.6f} of {float(a.abs().max()):.4f}")
            break
    else:
        print("every recorded output is equal for the row")
    for name, v in out.items():
        top = torch.topk(v, 2)
        print(f"{name:14s} argmax {int(top.indices[0])} top2 {top.values.tolist()} max|diff vs one-row eager| "
              f"{float((v - out['one-row eager']).abs().max()):.6f}")


if __name__ == "__main__":
    main()
