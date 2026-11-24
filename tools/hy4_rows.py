"""Row invariance on the real checkpoint (playbook 5.1): a verify window's rows against the same tokens one row a call,
every layer and sublayer bit for bit; the first part that differs, by width.

    python tools/hy4_rows.py MODEL [--kind json] [--widths 2,6,16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--kind", default="json")
    p.add_argument("--widths", default="2,6,16")
    p.add_argument("--start", type=int, default=200, help="reply tokens decoded before the windows")
    p.add_argument("--repeat", type=int, default=1, help="the prompt this many times (past the indexer's limit)")
    a = p.parse_args()

    import torch
    from tokenizers import Tokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from hy4_probe import PROMPTS

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.app import EffortTemplate, text_view
    from tensorfold.families.hy4_flash.cuda.model import Engine, prefill, serial_decode
    from tensorfold.families.hy4_flash.cuda.weights import load

    view = text_view(Path(a.model))
    tok = Tokenizer.from_file(str(view / "tokenizer.json"))
    ids = tok.encode(EffortTemplate(ChatTemplate(view)).render([{"role": "user", "content": PROMPTS[a.kind] * a.repeat}],
                                                               tools=None, enable_thinking=False),
                     add_special_tokens=False).ids
    e = Engine(load(a.model, "cuda"), max_len=8192)
    e.graphs = False
    greedy = Sampling(seed=0, temperature=0.0)
    widths = [int(w) for w in a.widths.split(",")]
    serial = serial_decode(e, prefill(e, ids, greedy), a.start + max(widths) + 1, greedy, stop_eos=False).tokens

    rec: list = []

    def watch():
        layer_fn, attn, mlp, eg = e._layer, e._attention, e._mlp, e._engram

        def layer(i, lay, x, prompt):
            y = layer_fn(i, lay, x, prompt)
            rec.append((i, 3, "layer", y.clone()))
            return y

        def attention(i, at, x, prompt):
            rec.append((i, 1, "attn_in", x.clone()))
            y = attn(i, at, x, prompt)
            rec.append((i, 2, "attn", y.clone()))
            return y

        def moe(i, m, x, prompt):
            y = mlp(i, m, x, prompt)
            rec.append((i, 2.5, "mlp", y.clone()))
            return y

        def engram(j, g, x):
            y = eg(j, g, x)
            rec.append((e.c.engram_layers[j], 0, "engram", y.clone()))
            return y

        e._layer, e._attention, e._mlp, e._engram = layer, attention, moe, engram

    watch()
    for w in widths:
        window_tokens = serial[a.start:a.start + w]
        rows: list = []
        e.reset()
        prefill(e, ids, greedy)
        for t in serial[:a.start]:
            e.forward([t], prompt=False)
        for t in window_tokens:                       # one row a call
            rec.clear()
            want = e.forward([t], prompt=False)
            rows.append(({(i, o, n): v for i, o, n, v in rec}, want))
        e.rewind(w)
        rec.clear()
        got = e.forward(window_tokens, prompt=False, every=True)
        win = {(i, o, n): v for i, o, n, v in rec}
        first = None
        for r, (one, logit) in enumerate(rows):
            for key in sorted(one):
                x, y = one[key].reshape(1, -1) if key[2] != "engram" else one[key], win[key][r:r + 1]
                y = y.reshape(x.shape)
                if not torch.equal(x, y):
                    diff = float((x.float() - y.float()).abs().max())
                    if first is None or (key, r) < first[:2]:
                        first = (key, r, diff)
                    break
        logits_equal = torch.equal(got, torch.cat([lg for _, lg in rows]))
        print(f"{a.kind} x{a.repeat} at {len(ids) + a.start}: width {w}: logits equal {logits_equal}; first differing part "
              f"{'none' if first is None else f'{first[0][2]} of layer {first[0][0]} at row {first[1]}, max {first[2]:.6g}'}",
              flush=True)


if __name__ == "__main__":
    main()
