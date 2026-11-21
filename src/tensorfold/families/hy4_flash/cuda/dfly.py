"""Hy4-flash's built-in drafter, DFly2 (``model.dfly_layers``): a block of five tokens from one pass over six rows.

A round's block is the target's pending token at the next position, then five mask tokens (``dfly_mask_token_id``).
The first row's embedding is fused with the target's last normed row (``mtp_fusion``: ``e + gate * eh_proj(enorm(e),
hnorm(h))``). Five MLA layers with dense SwiGLU MLPs and a plain residual run over the block. Draft layer j reads the
latent cache of target sparse layer ``dfly_target_kv_layers[j]``: the keys the target's indexer listed for its last
committed row (every earlier key below 2,048 tokens), then the block's own rows, causal, which it writes into that
cache at the block's positions (the target's verify window overwrites them). After the final norm, each of rows 1-5
is corrected by an MLP over it and the previous token's embedding (``hidden_correction``) and read through the
target's head, in turn. Draft i is drawn by the request's keyed sampler at position pos + 1 + i, where the verify
window samples it (playbook 5.2): a sampled reply then accepts the drafts serial decoding would write, where greedy
drafts matched only its argmax. This follows vLLM's ``HYV4DFlyQueryModel`` and ``build_dfly_reuse_indices``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..config import SPARSE
from . import kernels as K
from .weights import Attention, B16Lin, CpuLin, F8Lin, _fp8_block_dense, _nvfp4_dense

PREFIX = "model.dfly_layers."


@dataclass
class DraftLayer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    attn: Attention
    gate: Any
    up: Any
    down: Any


class Fp4Lin:
    """A 2-D NVFP4 projection on the lane matmul (``Fp4Linear``)."""

    def __init__(self, lin) -> None:
        self.lin = lin
        self.n, self.k = lin.n, lin.k

    def __call__(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        return self.lin(x)


class DFly:
    """The drafter's weights and one round's ``propose``; the target engine keeps what it reads (``Engine.watch``)."""

    def __init__(self, model_dir: Path, engine, device: str = "cuda") -> None:
        from tensorfold.families.qwen3_5.cuda.weights import _Tensors

        raw = json.loads((Path(model_dir) / "config.json").read_text())
        self.kv_layers = [int(i) for i in raw["dfly_target_kv_layers"]]
        index_layers = [int(i) for i in raw["dfly_target_index_layer"]]
        if index_layers != self.kv_layers:
            raise ValueError(f"DFly2 reads its keys and its index from the same target layers, not {index_layers} "
                             f"and {self.kv_layers}")
        if raw.get("dfly_context_local_window"):
            raise ValueError("DFly2 with a local context window is not read by this engine")
        if raw.get("dfly_mtp_fusion", "slot0") != "slot0" or not raw.get("dfly_enable_hidden_correction", True):
            raise ValueError("DFly2 runs the slot-0 fusion and the hidden correction")
        self.mask = int(raw["dfly_mask_token_id"])
        c = engine.c
        for li in self.kv_layers:
            if c.kinds[li] != SPARSE:
                raise ValueError(f"DFly2 target layer {li} is not a sparse layer")
        self.engine = engine
        self.width = 6                                        # the pending token and five drafts
        cuda = torch.device(device).type == "cuda"
        t = _Tensors(Path(model_dir), device, skip=lambda n: not n.startswith(PREFIX))
        try:
            self._load(t, c, cuda, int(raw["dfly_num_hidden_layers"]))
            left = list(t)
        finally:
            t.close()
        unused = [n for n in left if "confidence_head" not in n]
        if unused:
            raise ValueError(f"unused DFly2 tensors: {unused[:5]}")
        engine.watch |= set(self.kv_layers)
        # drafts read a slice of the head: the first ids (byte-level BPE ids run roughly by merge frequency) and every
        # special token; the verify window reads the whole head, so replies stay serial decoding's
        rows = int(os.environ.get("TF_HY4_DRAFT_VOCAB", "32768"))
        head = engine.w.head
        if 0 < rows < head.shape[0]:
            self.vocab = torch.cat([torch.arange(rows), torch.arange(c.bos, head.shape[0])]).to(head.device)
            self.head = head[self.vocab].contiguous()
        else:
            self.vocab, self.head = None, head
        self.inv = (1.0 / (c.sparse.theta ** (torch.arange(0, c.sparse.rope, 2, dtype=torch.float32)
                                              / c.sparse.rope))).to(device)
        self.offsets = torch.arange(self.width, device=device)

    def _load(self, t, c, cuda: bool, count: int) -> None:
        from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8BlockLinear

        def bf16(name: str) -> torch.Tensor:
            return t.pop(PREFIX + name).to(torch.bfloat16).contiguous()

        def lin(name: str):
            w = t.pop(PREFIX + name + ".weight")
            if w.dtype == torch.float8_e4m3fn:
                s = t.pop(PREFIX + name + ".weight_scale_inv")
                return F8Lin(Fp8BlockLinear.from_checkpoint(w, s)) if cuda else CpuLin(_fp8_block_dense(w, s))
            if w.dtype == torch.uint8:
                sc, s2 = t.pop(PREFIX + name + ".weight_scale"), t.pop(PREFIX + name + ".weight_scale_2")
                if cuda:
                    return Fp4Lin(Fp4Linear.from_checkpoint(w, sc.view(torch.uint8), float(s2.float().reshape(-1)[0])))
                return CpuLin(_nvfp4_dense(w, sc.view(torch.uint8), s2.float().reshape(())))
            return B16Lin(w) if cuda else CpuLin(w)

        heads = c.sparse
        self.layers: list[DraftLayer] = []
        for j in range(count):
            a = f"{j}.self_attn."
            kv_b = bf16(a + "kv_b_proj.weight").view(heads.heads, heads.nope + heads.v, c.kv_rank)
            attn = Attention(kind=SPARSE, heads=heads.heads, q_a=lin(a + "q_a_proj"),
                             q_norm=bf16(a + "q_a_layernorm.weight"), q_b=lin(a + "q_b_proj"),
                             kv_a=lin(a + "kv_a_proj_with_mqa"), kv_norm=bf16(a + "kv_a_layernorm.weight"),
                             wk=kv_b[:, :heads.nope].contiguous(), wv=kv_b[:, heads.nope:].contiguous(),
                             gate=lin(a + "linear_gate"), o=lin(a + "o_proj"),
                             post_norm=bf16(a + "post_proj_norm.weight"),
                             sinks=t.pop(PREFIX + a + "learnable_sink_param").float().contiguous())
            self.layers.append(DraftLayer(bf16(f"{j}.input_layernorm.weight"),
                                          bf16(f"{j}.post_attention_layernorm.weight"), attn,
                                          lin(f"{j}.mlp.gate_proj"), lin(f"{j}.mlp.up_proj"), lin(f"{j}.mlp.down_proj")))
        self.final_norm = bf16("final_norm.weight")
        self.enorm, self.hnorm = bf16("mtp_fusion.enorm.weight"), bf16("mtp_fusion.hnorm.weight")
        self.eh_proj, self.fuse_gate = lin("mtp_fusion.eh_proj"), bf16("mtp_fusion.gate")
        hc = "hidden_correction."
        self.hidden_norm, self.embed_norm = bf16(hc + "hidden_norm.weight"), bf16(hc + "embed_norm.weight")
        self.hc_gate, self.hc_up, self.hc_down = lin(hc + "gate_proj"), lin(hc + "up_proj"), lin(hc + "down_proj")
        for name in [n for n in t if "confidence_head" in n]:
            t.pop(name)

    def nbytes(self) -> int:
        return sum(x.nbytes() for layer in self.layers for x in (layer.gate, layer.up, layer.down))

    # -- a round ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def propose(self, e, last: int, sampling) -> list[int]:
        """Five draft tokens after ``last`` (the pending token, at position e.pos), each by ``sampling``'s keyed rule
        at its position (greedy at T = 0)."""

        from .model import _silu_mul, keyed_rows

        c = e.c
        anchor = e.pos
        if anchor + self.width > e.max_len or e.normed is None:
            return []
        dev = e.device
        ids = torch.full((self.width,), self.mask, dtype=torch.int64, device=dev)
        ids[0] = int(last)
        x = e.w.embed[ids]
        target = e.normed[e.src_row:e.src_row + 1]
        fused = self.eh_proj(torch.cat([K.rmsnorm(x[:1], self.enorm, c.eps), K.rmsnorm(target, self.hnorm, c.eps)],
                                       -1))
        x[0] = (x[0].float() + self.fuse_gate.float() * fused[0].float()).to(torch.bfloat16)
        pos = anchor + self.offsets
        hidden, residual = x, None
        for j, layer in enumerate(self.layers):
            if residual is not None:
                hidden = (hidden.float() + residual.float()).to(torch.bfloat16)
            residual = hidden
            a = self._attention(e, j, layer.attn, K.rmsnorm(hidden, layer.input_norm, c.eps), pos, anchor)
            hidden = (a.float() + residual.float()).to(torch.bfloat16)
            residual = hidden
            h = K.rmsnorm(hidden, layer.post_norm, c.eps)
            hidden = layer.down(_silu_mul(layer.gate(h), layer.up(h)))
        out = K.rmsnorm((hidden.float() + residual.float()).to(torch.bfloat16), self.final_norm, c.eps)
        greedy = sampling is None or sampling.temperature <= 0
        prev = ids[:1]
        drafts = []
        for step in range(self.width - 1):
            h = out[step + 1:step + 2]
            both = torch.cat([K.rmsnorm(h, self.hidden_norm, c.eps), K.rmsnorm(e.w.embed[prev], self.embed_norm, c.eps)],
                             -1)
            h = (h.float() + self.hc_down(_silu_mul(self.hc_gate(both), self.hc_up(both))).float()).to(torch.bfloat16)
            logits = K.head(h, self.head)
            if greedy:                    # stays on the device: one read-back a round
                prev = logits.argmax(-1)
                prev = prev if self.vocab is None else self.vocab[prev]
            else:                         # the next step embeds it: one read-back a draft
                prev = torch.tensor(keyed_rows(logits, [anchor + 1 + step], sampling, self.vocab), device=dev)
            drafts.append(prev)
        return [int(t) for t in torch.cat(drafts).tolist()]

    def _attention(self, e, j: int, a: Attention, x: torch.Tensor, pos: torch.Tensor, anchor: int) -> torch.Tensor:
        """The block's MLA over target layer ``kv_layers[j]``: the target's list for its last row, then the block."""

        c = e.c
        rows, heads = x.shape[0], c.sparse
        li = self.kv_layers[j]
        gate = a.gate(x)
        qr = K.rmsnorm(a.q_a(x), a.q_norm, c.eps)
        q = a.q_b(qr).view(rows, heads.heads, heads.qk)
        ckv = a.kv_a(x)
        lat = torch.empty((rows, K.LW + K.RW), dtype=torch.bfloat16, device=x.device)
        lat[:, :K.LW] = K.rmsnorm(ckv[:, :c.kv_rank], a.kv_norm, c.eps)
        lat[:, K.LW:] = e._rope(ckv[:, c.kv_rank:], pos, self.inv)
        e.latent[li][anchor:anchor + rows] = lat
        qa = torch.empty((rows, heads.heads, K.LW + K.RW), dtype=torch.bfloat16, device=x.device)
        qa[:, :, :K.LW] = K.absorb(q[:, :, :heads.nope], a.wk, torch.empty(
            (rows, heads.heads, K.LW), dtype=torch.bfloat16, device=x.device))
        qa[:, :, K.LW:] = e._rope(q[:, :, heads.nope:], pos, self.inv)
        tok, cnt = self._lists(e, li, anchor, rows)
        o_lat = torch.empty((rows, heads.heads, K.LW), dtype=torch.bfloat16, device=x.device)
        ones = torch.ones(rows, dtype=torch.int32, device=x.device)
        K.attention(qa, e.latent[li], torch.zeros_like(ones), cnt, ones, tok, a.sinks, heads.qk ** -0.5, o_lat,
                    e.scratch)
        v = K.expand(o_lat, a.wv, torch.empty((rows, heads.heads, heads.v), dtype=torch.bfloat16, device=x.device))
        sig = torch.sigmoid(gate.float()).to(torch.bfloat16).float()
        out = (v.reshape(rows, -1).float() * sig).to(torch.bfloat16)
        return K.rmsnorm(a.o(out), a.post_norm, c.eps)

    def _lists(self, e, li: int, anchor: int, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Each block row's keys: the target's list for its last committed row (its picked tokens past 2,048 keys,
        else every earlier position), then block rows 0 .. i. Returns (tokens [rows, 2048 + rows], counts)."""

        sel, tok, cnt = e.lists[li]
        r, width = e.src_row, e.c.index_topk
        dev = tok.device
        k = torch.arange(rows, device=dev, dtype=torch.int32)
        out = torch.empty((rows, width + rows), dtype=torch.int32, device=dev)
        out[:, :width] = torch.where(sel[r] != 0, tok[r, :width], torch.arange(width, device=dev, dtype=torch.int32))
        n = cnt[r]
        out[:, (n + k).long()] = (anchor + k)[None, :].expand(rows, rows)
        return out, n + 1 + k

    def commit(self, e, accepted: int) -> None:
        """Nothing to keep: the next round reads the target's state after its verify window."""
