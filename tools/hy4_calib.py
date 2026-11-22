"""Hy4-flash's projection inputs on calibration text: each input site's second moments for ``hy4_trunk.py --stats``.

    python tools/hy4_calib.py MODEL OUT_DIR [--tokens 32768] [--full 3072]      # in the serving image, on the GPU

The engine prefills calibration text (the fork's own source and prose, rendered as user turns; never the KL goldens'
``docs/recipes``) in 2,048-row chunks, and a tap on every trunk projection adds its input rows to its site's sums:
X^T X [K, K] in fp32 when K <= ``--full``, else its diagonal. Projections that read the same rows share a site (q_a,
kv_a, linear_gate and the indexer's key and weight projections read the attention input; the shared expert, the
latent down projection and the router read the MLP input). kv_b's key rows see the query's no-rope part a head at a
time and its value rows the attention output's latent, so their sites keep a diagonal a head.

OUT_DIR holds ``sites-L07.safetensors`` (``<site>.h`` or ``<site>.diag``, and ``<site>.n`` rows) a layer,
``sites-head.safetensors``, and ``sites.json``: {projection name: site}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SOURCES = ("src/tensorfold/**/*.py", "README.md", "RUNBOOK.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs/*.md")


class Sites:
    """Running sums of each site's input rows; a site counts a tensor of rows once however many projections read it."""

    def __init__(self, full: int) -> None:
        self.full = full
        self.h: dict[str, torch.Tensor] = {}
        self.n: dict[str, int] = {}
        self.seen: dict[str, tuple[int, int, int]] = {}

    def add(self, site: str, x: torch.Tensor) -> None:
        key = (x.data_ptr(), x._version, x.shape[0])
        if self.seen.get(site) == key:
            return
        self.seen[site] = key
        f = x.reshape(-1, x.shape[-1]).float()
        if f.shape[-1] <= self.full:
            acc = self.h.get(site)
            if acc is None:
                acc = self.h[site] = torch.zeros((f.shape[-1], f.shape[-1]), dtype=torch.float32, device=f.device)
            acc.addmm_(f.t(), f)
        else:
            acc = self.h.get(site)
            if acc is None:
                acc = self.h[site] = torch.zeros(f.shape[-1], dtype=torch.float64, device=f.device)
            acc += (f.double() ** 2).sum(0)
        self.n[site] = self.n.get(site, 0) + f.shape[0]

    def add_heads(self, site: str, x: torch.Tensor) -> None:
        """Rows [R, H, K] read a head at a time: a diagonal [H, K]."""

        key = (x.data_ptr(), x._version, x.shape[0])
        if self.seen.get(site) == key:
            return
        self.seen[site] = key
        acc = self.h.get(site)
        if acc is None:
            acc = self.h[site] = torch.zeros(x.shape[1:], dtype=torch.float64, device=x.device)
        acc += (x.double() ** 2).sum(0)
        self.n[site] = self.n.get(site, 0) + x.shape[0]


class Tap:
    """A projection that adds its input rows to a site, then runs the projection."""

    def __init__(self, lin, site: str, sites: Sites) -> None:
        self.lin, self.site, self.sites = lin, site, sites
        self.n, self.k = lin.n, lin.k

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        self.sites.add(self.site, x)
        return self.lin(x, prompt)


def tap_model(w, sites: Sites) -> dict[str, str]:
    """Wrap every trunk projection of ``w`` in a ``Tap``: {projection name: site}."""

    names: dict[str, str] = {}

    def wrap(obj, attr: str, name: str, site: str) -> None:
        setattr(obj, attr, Tap(getattr(obj, attr), site, sites))
        names[name] = site

    for i, layer in enumerate(w.layers):
        p = f"model.layers.{i}."
        a, at = layer.attn, p + "self_attn."
        for attr, name in (("q_a", "q_a_proj"), ("kv_a", "kv_a_proj_with_mqa"), ("gate", "linear_gate")):
            wrap(a, attr, at + name, at + "in")
        wrap(a, "q_b", at + "q_b_proj", at + "qr")
        wrap(a, "o", at + "o_proj", at + "o_in")
        names[at + "kv_b_proj.key"] = at + "kv_b.key"
        names[at + "kv_b_proj.value"] = at + "kv_b.value"
        if a.indexer is not None:
            ci = at + "compressed_indexer."
            wrap(a.indexer, "wk", ci + "token_key_projection", at + "in")
            wrap(a.indexer, "wh", ci + "head_weight_projection", at + "in")
            wrap(a.indexer, "wq", ci + "query_projection", at + "qr")
        m, mt = layer.mlp, p + "mlp."
        if hasattr(m, "s_gate"):
            wrap(m, "pre", mt + "routed_expert_down_proj", mt + "in")
            wrap(m, "s_gate", mt + "shared_experts.gate_proj", mt + "in")
            wrap(m, "s_up", mt + "shared_experts.up_proj", mt + "in")
            wrap(m, "s_down", mt + "shared_experts.down_proj", mt + "s_act")
            wrap(m, "post", mt + "routed_expert_up_proj", mt + "routed")
        else:
            wrap(m, "gate", mt + "gate_proj", mt + "in")
            wrap(m, "up", mt + "up_proj", mt + "in")
            wrap(m, "down", mt + "down_proj", mt + "act")
        if layer.engram is not None:
            e, et = layer.engram, p + "engram_layer."
            wrap(e, "value", et + "value_proj", et + "flat")
            for s in range(len(e.keys)):
                e.keys[s] = Tap(e.keys[s], et + "flat", sites)
                names[et + f"key_projs.{s}"] = et + "flat"
    names["lm_head"] = "head.in"
    return names


def tap_kv_b(w, sites: Sites):
    """kv_b's absorb and expand inputs, a diagonal a head, by the weight each call reads. Returns the undo."""

    from tensorfold.families.hy4_flash.cuda import kernels as K
    from tensorfold.families.hy4_flash.cuda import model

    site_of = {}
    for i, layer in enumerate(w.layers):
        at = f"model.layers.{i}.self_attn."
        site_of[id(layer.attn.wk)] = at + "kv_b.key"
        site_of[id(layer.attn.wv)] = at + "kv_b.value"
    absorb, expand = K.absorb, K.expand

    def tapped_absorb(q, wk, out):
        sites.add_heads(site_of[id(wk)], q)
        return absorb(q, wk, out)

    def tapped_expand(o_lat, wv, out):
        sites.add_heads(site_of[id(wv)], o_lat)
        return expand(o_lat, wv, out)

    model.K.absorb, model.K.expand = tapped_absorb, tapped_expand

    def undo() -> None:
        model.K.absorb, model.K.expand = absorb, expand

    return undo


def run(e, texts: list[list[int]], log=print) -> None:
    """Each text as prompt chunks with every row's logits, so the head's site sees every row."""

    t0 = time.perf_counter()
    for j, ids in enumerate(texts):
        e.reset()
        for s in range(0, len(ids), e.prefill_rows):
            e.forward(ids[s:s + e.prefill_rows], prompt=True, every=True)
        log(f"text {j + 1}/{len(texts)}: {len(ids)} tokens, {time.perf_counter() - t0:.0f}s")


def calibration_texts(root: Path, tokens: int, encode) -> list[list[int]]:
    """Chunks of about 2,000 tokens of the fork's own code and prose as user turns, alternating kinds, ``tokens`` in
    all."""

    files = sorted({f for pat in SOURCES for f in root.glob(pat) if "recipes" not in f.parts})
    code = [f for f in files if f.suffix == ".py"]
    prose = [f for f in files if f.suffix == ".md"]
    out, total, k = [], 0, 0
    pools = [code[::7], prose]
    while total < tokens and any(pools):
        pool = pools[k % 2] or pools[(k + 1) % 2]
        f = pool.pop(0)
        k += 1
        text = f.read_text(errors="replace")[:9000]
        ask = "Review this file and summarize what it does." if f.suffix == ".py" else "Summarize this document."
        ids = encode(f"{ask}\n\n{text}")[:2040]
        if len(ids) < 256:
            continue
        out.append(ids)
        total += len(ids)
    return out


def main() -> None:
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.app import EffortTemplate, text_view
    from tensorfold.families.hy4_flash.cuda.model import Engine
    from tensorfold.families.hy4_flash.cuda.weights import load

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--full", type=int, default=3072, help="widest input whose full X^T X is kept")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    view = text_view(Path(a.model))
    tok = Tokenizer.from_file(str(view / "tokenizer.json"))
    template = EffortTemplate(ChatTemplate(view))

    def encode(text: str) -> list[int]:
        rendered = template.render([{"role": "user", "content": text}], tools=None, enable_thinking=False)
        return tok.encode(rendered, add_special_tokens=False).ids

    texts = calibration_texts(Path(__file__).parents[1], a.tokens, encode)
    t0 = time.perf_counter()
    w = load(a.model, "cuda")
    sites = Sites(a.full)
    names = tap_model(w, sites)
    tap_kv_b(w, sites)
    head = w.head
    w.head = Tap(_Head(head), "head.in", sites)
    e = Engine(w, max_len=4096)
    e.graphs = False
    e.set_sampling(Sampling(seed=0, temperature=0.0))
    run(e, texts, lambda m: print(m, flush=True))
    by_file: dict[str, dict[str, torch.Tensor]] = {}
    for site, acc in sites.h.items():
        tag = f"L{int(site.split('.')[2]):02d}" if site.startswith("model.layers.") else "head"
        kind = "h" if acc.dim() == 2 and not site.endswith(("kv_b.key", "kv_b.value")) else "diag"
        by_file.setdefault(tag, {})[f"{site}.{kind}"] = acc.float().cpu().contiguous()
        by_file[tag][f"{site}.n"] = torch.tensor(sites.n[site], dtype=torch.int64)
    for tag, tensors in by_file.items():
        save_file(tensors, str(out / f"sites-{tag}.safetensors"))
    (out / "sites.json").write_text(json.dumps({"names": names, "tokens": sum(map(len, texts)),
                                                "texts": len(texts), "full": a.full}, indent=1))
    print(f"{len(sites.h)} sites, {sum(map(len, texts))} tokens, {time.perf_counter() - t0:.0f}s", flush=True)


class _Head:
    """The head as a projection (``Engine.head`` calls a non-tensor head with the normed rows)."""

    def __init__(self, head) -> None:
        self.head = head
        shape = head.shape
        self.n, self.k = int(shape[0]), int(shape[1])

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        from tensorfold.families.hy4_flash.cuda import kernels as K

        return K.head(x, self.head) if isinstance(self.head, torch.Tensor) else self.head(x)


if __name__ == "__main__":
    main()
