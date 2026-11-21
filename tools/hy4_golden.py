"""Hy4-flash's engine against transformers' model on the real checkpoint, teacher-forced (playbook 4.5-4.7).

    python tools/hy4_golden.py engine MODEL OUT.npz          # in the serving image, on the GPU
    python tools/hy4_golden.py reference MODEL ENGINE.npz    # where transformers PR #50903 is installed

``engine`` renders four prompts (chat, a tool call, code, and a document of about 3,000 tokens that takes the sparse
layers past 2,048 keys), decodes up to 32 greedy tokens each (to the end token), then runs each whole sequence through one prompt pass and
saves every position's top-20 logits and the full logits of the decoded region. ``reference`` replays the same ids
through transformers' decoder layers one at a time, on the GPU, each holding the checkpoint's weights decoded to
bf16 (block FP8 and NVFP4) and reading its Engram rows from the tables' files. It reports top-1 agreement, the greedy
tokens where transformers' top two stand apart, and KL(transformers || engine) over the decoded region up to the
reply's end token: past it both models are out of distribution (a golden saved before 21 Nov holds those rows).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

STEPS = 32
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                           "parameters": {"type": "object", "properties": {
                                               "city": {"type": "string"}, "unit": {"type": "string",
                                                                                    "enum": ["c", "f"]}},
                                               "required": ["city"]}}}]


def prompts(root: Path) -> dict[str, tuple[list[dict], list | None]]:
    """The four built-in prompts, or ``HY4_GOLDEN_PROMPTS``' JSON file: {name: [messages, tools or null]}."""

    own = os.environ.get("HY4_GOLDEN_PROMPTS")
    if own:
        return {k: (v[0], v[1]) for k, v in json.loads(Path(own).read_text()).items()}
    doc = "\n\n".join(p.read_text() for p in sorted((root / "docs" / "recipes").glob("*.md"))[:6])[:13000]
    return {
        "chat": ([{"role": "user", "content": "Explain in three sentences why the sky is blue."}], None),
        "tool": ([{"role": "user", "content": "What's the weather in Thessaloniki right now, in celsius?"}], TOOLS),
        "code": ([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                              "iteratively, with a docstring."}], None),
        "long": ([{"role": "user", "content": doc + "\n\nIn two sentences, what are these documents about?"}], None),
    }


def reply(e, ids: list[int], first: int, sampling, steps: int = STEPS):
    """The reply after a prefilled prompt up to and including its end token (``steps`` tokens at most), and the
    sequence the teacher-forced pass reads: the prompt, then every reply token whose next one the engine chose."""

    from tensorfold.families.hy4_flash.cuda.model import serial_decode

    res = serial_decode(e, first, steps, sampling, stop_eos=True)
    return res, list(ids) + res.tokens[:-1]


def decoded_rows(gen, eos) -> int:
    """The reply's rows up to and including its first end token: the ones a comparison reads."""

    ends = {int(t) for t in eos}
    return next((i + 1 for i, t in enumerate(gen) if int(t) in ends), len(gen))


def engine(model: str, out: str) -> None:
    import torch
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.app import EffortTemplate, text_view
    from tensorfold.families.hy4_flash.cuda.model import Engine, prefill
    from tensorfold.families.hy4_flash.cuda.weights import load

    view = text_view(Path(model))
    tok = Tokenizer.from_file(str(view / "tokenizer.json"))
    template = EffortTemplate(ChatTemplate(view))
    t0 = time.perf_counter()
    w = load(model, "cuda")
    print(f"loaded in {time.perf_counter() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    e = Engine(w, max_len=8192)
    greedy = Sampling(seed=0, temperature=0.0)
    saved: dict[str, np.ndarray] = {}
    for name, (messages, tools) in prompts(Path(__file__).parents[1]).items():
        text = template.render(messages, tools=tools, enable_thinking=False)
        ids = tok.encode(text, add_special_tokens=False).ids
        t0 = time.perf_counter()
        first = prefill(e, ids, greedy)
        pre = time.perf_counter() - t0
        res, seq = reply(e, ids, first, greedy)
        gen = res.tokens
        print(f"{name}: {len(ids)} prompt tokens, prefill {pre:.2f}s, decode {res.tokens_per_second:.1f} tok/s: "
              f"{tok.decode(gen, skip_special_tokens=False)!r}", flush=True)
        e.reset()
        rows = []
        for s in range(0, len(seq), e.prefill_rows):
            rows.append(e.forward(seq[s:s + e.prefill_rows], prompt=True, every=True).float())
        logits = torch.cat(rows)
        vals, top = logits.topk(20, dim=-1)
        saved[f"{name}_ids"] = np.array(seq, np.int64)
        saved[f"{name}_prompt"] = np.array([len(ids)])
        saved[f"{name}_gen"] = np.array(gen, np.int64)
        saved[f"{name}_top_ids"] = top.cpu().numpy().astype(np.int32)
        saved[f"{name}_top_vals"] = vals.cpu().numpy()
        saved[f"{name}_tail"] = logits[len(ids) - 1:].cpu().numpy().astype(np.float16)
        saved[f"{name}_decode_tps"] = np.array([res.tokens_per_second])
        saved[f"{name}_prefill_s"] = np.array([pre])
        e.reset()
    np.savez(out, **saved)


# ------------------------------------------------------------------------------------------------ reference ---

SUB_LAYER = 1                # the layer whose sublayer outputs a dump also keeps
KEEP_FP32 = ("attn_hc.", "ffn_hc.", "sinks", "key_value_projection", "key_gate_projection", "intra_group_ape",
             "e_score_correction_bias", "hc_head.")


def reference(model: str, engine_npz: str) -> None:
    import torch
    from safetensors import safe_open
    from torch import nn
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig

    sys.path.insert(0, str(Path(__file__).parent))
    from hy4_reference import hf_name

    from tensorfold.families.hy4_flash.config import Config
    from tensorfold.families.hy4_flash.engram import RowReader, decode_rows

    root = Path(model)
    eos = Config.read(root).eos
    raw = json.loads((root / "config.json").read_text())
    text = {k: v for k, v in raw["text_config"].items() if k not in ("quantization_config", "architectures",
                                                                       "model_type")}
    cfg = HYV4FlashTextConfig(**text)
    cfg._attn_implementation = "eager"
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    dev = torch.device("cuda")

    def get(name: str) -> torch.Tensor:
        """One tensor, its file opened for this read only: a kept handle keeps every page it touched mapped (RSS
        grew 2 GiB a layer and took available memory beside production from 47 to 13 GiB, 21 Nov)."""

        with safe_open(str(root / index[name]), "pt", device="cpu") as f:
            return f.get_tensor(name)

    e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)

    def value(name: str) -> torch.Tensor:
        """A weight's values: block FP8 and NVFP4 decoded, others as stored."""

        w = get(name).to(dev)
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if w.dtype == torch.float8_e4m3fn and base and base + ".weight_scale_inv" in index:
            s = get(base + ".weight_scale_inv").to(dev).float()
            n, k = w.shape
            return w.float() * s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        if w.dtype == torch.uint8 and base and base + ".weight_scale_2" in index:
            codes = torch.stack([w & 0xF, w >> 4], -1).flatten(-2).long()
            sc = get(base + ".weight_scale").to(dev).view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
            s2 = get(base + ".weight_scale_2").to(dev).float()
            return e2m1[codes] * sc * s2.reshape(*s2.shape, 1, 1)
        return w

    @torch.no_grad()
    def fill(dst: torch.Tensor, name: str, chunk: int = 16) -> None:
        """NVFP4 experts decoded ``chunk`` at a time straight into ``dst`` (bf16, exact): no whole-tensor temporaries."""

        base = name[:-len(".weight")]
        codes, sc, s2 = get(name), get(base + ".weight_scale"), get(base + ".weight_scale_2").float()
        for e0 in range(0, codes.shape[0], chunk):
            c = codes[e0:e0 + chunk].to(dev).to(torch.int32)
            v = e2m1[torch.stack([c & 0xF, c >> 4], -1).flatten(-2)]
            v *= sc[e0:e0 + chunk].to(dev).view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
            v *= s2[e0:e0 + chunk].to(dev).reshape(-1, 1, 1)
            dst[e0:e0 + chunk].copy_(v)
            del c, v

    class Rows(nn.Module):                     # HYV4FlashEngramEmbedding over the table's file
        def __init__(self, head_vocab_sizes, embedding_dim):
            super().__init__()
            self.head_vocab_sizes = head_vocab_sizes
            self.offsets = nn.Buffer(torch.tensor([0] + head_vocab_sizes[:-1]).cumsum(0), persistent=False)
            self.reader = None

        def forward(self, hash_ids):
            rows = (hash_ids.clamp_min(0) + self.offsets.to(hash_ids.device)).cpu().numpy()
            codes, scales = self.reader.gather(rows)
            out = torch.from_numpy(decode_rows(codes, scales, self.reader.dim)).view(*rows.shape, -1)
            out = out.to(hash_ids.device).to(torch.bfloat16)
            return out.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)

    M.HYV4FlashEngramEmbedding = Rows
    if os.environ.get("HY4_REF_FP32_ATTN") == "1":     # the reference's attention in fp32, to size its bf16 rounding
        eager = M.eager_attention_forward

        def fp32_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            out, w = eager(module, query.float(), key.float(), value.float(),
                           None if attention_mask is None else attention_mask.float(), scaling, dropout, **kw)
            return out.to(query.dtype), w

        M.eager_attention_forward = fp32_attention
    torch.set_default_dtype(torch.bfloat16)
    rotary = M.HYV4FlashRotaryEmbedding(cfg).to(dev)
    z = np.load(engine_npz)
    names = sorted({k.rsplit("_", 1)[0] for k in z.files if k.endswith("_ids") and not k.endswith("top_ids")})
    seqs = {n: z[f"{n}_ids"].tolist() for n in names}
    hidden: dict[str, torch.Tensor] = {}
    pos, masks, pes, grams = {}, {}, {}, {}
    emb = get("model.embed_tokens.weight").to(dev)
    for n, ids in seqs.items():
        T = len(ids)
        x = torch.tensor([ids], device=dev)
        hidden[n] = emb[x].unsqueeze(2).expand(1, T, cfg.hc_mult, -1).contiguous()
        p = torch.arange(T, device=dev).view(1, 1, -1).expand(3, 1, -1)
        pos[n] = p
        q, k = torch.arange(T, device=dev)[:, None], torch.arange(T, device=dev)[None, :]
        lo = torch.finfo(torch.bfloat16).min
        causal = torch.where(k <= q, 0.0, lo).to(torch.bfloat16)[None, None]
        sliding = torch.where((k <= q) & (q - k < cfg.sliding_window), 0.0, lo).to(torch.bfloat16)[None, None]
        masks[n] = {"indexed_attention": causal, "sliding_attention": sliding}
        pes[n] = {kind: rotary(hidden[n], position_ids=p, layer_type=kind) for kind in ("indexed_attention",
                                                                                         "sliding_attention")}
        grams[n] = M.HYV4FlashTextModel.build_ngram_tokens(type("C", (), {"config": cfg})(), x, None)
    del emb
    t0 = time.perf_counter()
    import resource

    count = int(os.environ.get("HY4_REF_LAYERS", cfg.num_hidden_layers))
    dump_name = os.environ.get("HY4_REF_DUMP", "")
    dump: dict[str, np.ndarray] | None = {"ids": np.array(seqs[dump_name])} if dump_name else None
    for i in range(count):
        with torch.device(dev):
            layer = M.HYV4FlashDecoderLayer(cfg, i)
        state = layer.state_dict()
        prefix = f"model.layers.{i}."
        done = set()
        experts: dict[str, torch.Tensor] = {}
        for name in [n for n in index if n.startswith(prefix)]:
            if name.endswith(("weight_scale_inv", "weight_scale", "weight_scale_2")) or "multi_head_embedding" in name:
                continue
            m = re.match(r".*\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$", name)
            if m:
                experts[m.group(1)] = name
                continue
            key = hf_name(name)[len(f"layers.{i}."):]
            if key not in state:
                raise KeyError(f"{name} -> {key}")
            v = value(name)
            target = dict(layer.named_parameters()).get(key)
            if target is None:
                target = dict(layer.named_buffers())[key]
            if not v.is_floating_point():
                target.data = v                                       # the Engram lookup table
            elif any(s in key for s in KEEP_FP32):                    # transformers' _keep_in_fp32_modules_strict
                target.data = v.float()
            else:
                target.data = v.to(target.dtype)
            done.add(key)
        if experts:
            mlp = layer.mlp.experts
            width = mlp.gate_up_proj.shape[1] // 2
            fill(mlp.gate_up_proj[:, :width], experts["gate_proj"])
            fill(mlp.gate_up_proj[:, width:], experts["up_proj"])
            fill(mlp.down_proj, experts["down_proj"])
            done |= {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        if layer.engram_layer is not None:
            e = layer.engram_layer
            e.lookup_table.data = e.lookup_table.data.to(torch.int32)
            e.hash_multipliers.copy_(e.draw_hash_multipliers(int(e.lookup_table.max()) + 1))
            table = f"{prefix}engram_layer.multi_head_embedding.embedding"
            e.multi_head_embedding.reader = RowReader(root / index[table + ".weight"], table,
                                                      int(sum(e.config_head_vocab_sizes)), 96)
            done.add("engram_layer.lookup_table")
        missing = [k for k in state if k not in done and not k.endswith(("inv_freq", "offsets", "head_vocab_sizes",
                                                                         "hash_multipliers"))]
        if missing:
            raise KeyError(f"layer {i}: not loaded {missing[:6]}")
        kind = cfg.layer_types[i]
        hooks = []
        current = [None]
        if dump is not None and i == SUB_LAYER:      # the first MoE / Engram / FP8 layer's sublayer outputs
            def keep(tag, now=current, at=i):
                def hook(m, args, out):
                    if now[0] == dump_name:
                        dump[f"{tag}_{at}"] = (out[0] if isinstance(out, tuple) else out)[0].to(
                            torch.bfloat16).view(torch.int16).cpu().numpy()
                return hook
            hooks = [layer.self_attn.register_forward_hook(keep("attn")), layer.mlp.register_forward_hook(keep("mlp"))]
            if layer.engram_layer is not None:
                hooks.append(layer.engram_layer.register_forward_hook(keep("engram")))
        with torch.no_grad():
            for n in seqs:
                current[0] = n
                hidden[n] = layer(hidden[n], attention_mask=masks[n][kind], position_embeddings=pes[n][kind],
                                  position_ids=pos[n], cache=None, use_cache=False, indexer_position_ids=pos[n],
                                  ngram_tokens=grams[n], engram_mask=torch.ones_like(grams[n][..., 0], dtype=torch.bool))
        for h in hooks:
            h.remove()
        if dump is not None:                         # each layer's streams for one sequence (``HY4_REF_DUMP``)
            dump[f"layer_{i}"] = hidden[dump_name][0].to(torch.bfloat16).view(torch.int16).cpu().numpy()
        del layer, state, experts
        torch.cuda.empty_cache()
        print(f"layer {i} done, {time.perf_counter() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f}"
              f" GiB, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
    if count < cfg.num_hidden_layers:
        if dump is not None:
            np.savez(Path(engine_npz).with_suffix(f".layers-{dump_name}.npz"), **dump)
        return
    with torch.device(dev):
        head = M.HYV4FlashHyperHead(cfg)
        norm = M.HYV4FlashRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    head.hc_fn.data = value("model.hc_head.hc_head_fn").float()
    head.hc_base.data = value("model.hc_head.hc_head_base").float()
    head.hc_scale.data = value("model.hc_head.hc_head_scale").float()
    norm.weight.data = value("model.norm.weight")
    lm = value("lm_head.weight").float()
    report = {}
    with torch.no_grad():
        for n, seq in seqs.items():
            want = norm(head(hidden[n]))[0].float() @ lm.t()                       # [T, V] fp32
            p0 = int(z[f"{n}_prompt"][0])
            top = torch.from_numpy(z[f"{n}_top_ids"][:, 0]).to(dev)
            agree = (want.argmax(-1) == top).float()
            two = want.topk(2, -1).values
            gap = two[:, 0] - two[:, 1]
            sure = gap > 0.5
            rows = decoded_rows(z[f"{n}_gen"].tolist(), eos)
            tail = torch.from_numpy(z[f"{n}_tail"][:rows].astype(np.float32)).to(dev)
            lp, lq = want[p0 - 1:p0 - 1 + rows].log_softmax(-1), tail.log_softmax(-1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            gen = torch.tensor(z[f"{n}_gen"].tolist()[:rows], device=dev)
            greedy_ok = (want[p0 - 1:p0 - 1 + rows].argmax(-1) == gen)
            at = torch.arange(len(seq), device=dev)
            buckets = {f"top1_{lo}_{hi}": round(float(agree[(at >= lo) & (at < hi)].mean()), 4)
                       for lo, hi in ((0, 512), (512, 2048), (2048, 1 << 30)) if bool(((at >= lo) & (at < hi)).any())}
            report[n] = {"tokens": len(seq), "prompt": p0, **buckets,
                         "top1_all": round(float(agree.mean()), 4),
                         "top1_sure": round(float(agree[sure].mean()), 4), "sure_rows": int(sure.sum()),
                         "greedy_steps_equal": f"{int(greedy_ok.sum())}/{greedy_ok.numel()}",
                         "greedy_misses_sure": int((~greedy_ok & sure[p0 - 1:p0 - 1 + rows]).sum()),
                         # 3 significant digits: a confident reply's KL sits far below 1e-5 (the tool reply's
                         # rows lead by 9.4 logits at least, and 5 decimals printed its KL as 0.0)
                         "kl_mean": float(f"{float(kl.mean()):.3g}"), "kl_max": float(f"{float(kl.max()):.3g}"),
                         "smallest_gap_decoded": round(float(gap[p0 - 1:p0 - 1 + rows].min()), 4)}
            print(n, json.dumps(report[n]), flush=True)
    Path(engine_npz).with_suffix(".reference.json").write_text(json.dumps(report, indent=1))
    if dump is not None:
        dump["normed"] = norm(head(hidden[dump_name]))[0].to(torch.bfloat16).view(torch.int16).cpu().numpy()
        np.savez(Path(engine_npz).with_suffix(f".layers-{dump_name}.npz"), **dump)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("engine")
    a.add_argument("model")
    a.add_argument("out")
    b = sub.add_parser("reference")
    b.add_argument("model")
    b.add_argument("engine_npz")
    args = p.parse_args()
    if args.cmd == "engine":
        engine(args.model, args.out)
    else:
        reference(args.model, args.engine_npz)


if __name__ == "__main__":
    main()
