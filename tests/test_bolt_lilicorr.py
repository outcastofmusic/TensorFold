"""Bolt's LiLiCorr drafter: its forward against a transcription of vLLM's math, and drafted replies equal serial ones."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")

import numpy as np
import torch.nn.functional as F
from bolt_lilicorr_fakes import FakeTarget, nvfp4_pack, random_tensors, serial, tiny_config
from torch import nn

from tensorfold.cuda.nvfp4.format import E2M1, e4m3
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.nemotron_bolt.cuda import lilicorr
from tensorfold.families.nemotron_bolt.cuda.lilicorr import (
    DraftConfig,
    Drafter,
    dense_tensors,
    expected_shapes,
    lilicorr_decode,
)


def make(*, window: int = 6, dtype=torch.float32, seed: int = 0, target: FakeTarget | None = None, **kw):
    cfg = DraftConfig.from_json(tiny_config(window=window, **kw))
    tensors = random_tensors(expected_shapes(cfg), seed)
    target = target or FakeTarget([1, 2, 3])
    return Drafter((cfg, tensors), embed=target.embed, lm_head=target.lm_head, device="cpu", dtype=dtype), tensors


# -- vLLM's math, transcribed in fp32 (qwen3_dflash.py, qwen3_dflash2.py, lilicorr.py) ----------------------------
def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope(x, pos, theta):
    hd = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    rot = torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    return x * cos + rot * sin


def _grouped_conv(x, delta, base, block, groups, gs, taps):
    """qwen3_dflash2._grouped_conv (its CPU path), verbatim but for names."""

    blocks = x.unflatten(-1, (groups, gs))
    coefficients = base.view(1, taps, groups, gs) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(x.shape[0]) % block
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def reference_block(c: DraftConfig, P: dict, taps: torch.Tensor, pending: int, depth: int, embed):
    """Context K/V for every committed position, then the query block with FA's symmetric window and sinks."""

    s, n = taps.shape[0], depth + 1
    G = c.hidden // c.conv_group
    ctx = _rms(taps @ P["fc.weight"].T, P["hidden_norm.weight"], c.eps)
    cpos = torch.arange(s)
    qpos = torch.arange(s, s + n)
    ids = torch.tensor([pending] + [c.mask_id] * depth)
    h, residual = embed(ids).float(), None
    for i in range(c.layers):
        p = f"layers.{i}."
        A = {name: P[p + "self_attn." + name] for name in
             ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
              "attention_sink_bias") if p + "self_attn." + name in P}

        def kv(x, pos, A=A):
            k = _rms((x @ A["k_proj.weight"].T).view(-1, c.kv_heads, c.head_dim), A["k_norm.weight"], c.eps)
            v = (x @ A["v_proj.weight"].T).view(-1, c.kv_heads, c.head_dim)
            return _rope(k.transpose(0, 1), pos, c.theta), v.transpose(0, 1)
        ck, cv = kv(ctx, cpos)
        if residual is None:
            residual = h
        else:
            residual = h + residual
        h = _rms(residual, P[p + "input_layernorm.weight"], c.eps)
        coef = (h @ P[p + "attention_conv.kernel_projection.weight"].T).reshape(n, 2, c.conv_taps, G)
        h = _grouped_conv(h, coef[:, 0], P[p + "attention_conv.base_kernel"][0], n, G, c.conv_group, c.conv_taps)
        q = _rms((h @ A["q_proj.weight"].T).view(n, c.heads, c.head_dim), A["q_norm.weight"], c.eps)
        q = _rope(q.transpose(0, 1), qpos, c.theta)
        qk, qv = kv(h, qpos)
        K, V = torch.cat([ck, qk], 1), torch.cat([cv, qv], 1)
        kpos = torch.cat([cpos, qpos])
        rep = c.heads // c.kv_heads
        K, V = K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0)
        sc = q @ K.transpose(-1, -2) / c.head_dim ** 0.5
        allowed = (qpos[:, None] - kpos[None, :]).abs() <= c.window - 1
        sc = sc.masked_fill(~allowed, float("-inf"))
        if c.sinks:
            sink = A["attention_sink_bias"].view(-1, 1, 1).expand(-1, n, 1)
            pr = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]
        else:
            pr = torch.softmax(sc, -1)
        h = (pr @ V).transpose(0, 1).reshape(n, -1) @ A["o_proj.weight"].T
        h = _grouped_conv(h, coef[:, 1], P[p + "attention_conv.base_kernel"][1], n, G, c.conv_group, c.conv_taps)
        residual = h + residual
        h = _rms(residual, P[p + "post_attention_layernorm.weight"], c.eps)
        coef = (h @ P[p + "mlp_conv.kernel_projection.weight"].T).reshape(n, 2, c.conv_taps, G)
        h = _grouped_conv(h, coef[:, 0], P[p + "mlp_conv.base_kernel"][0], n, G, c.conv_group, c.conv_taps)
        h = (F.silu(h @ P[p + "mlp.gate_proj.weight"].T) * (h @ P[p + "mlp.up_proj.weight"].T)) \
            @ P[p + "mlp.down_proj.weight"].T
        h = _grouped_conv(h, coef[:, 1], P[p + "mlp_conv.base_kernel"][1], n, G, c.conv_group, c.conv_taps)
    return _rms(h + residual, P["norm.weight"], c.eps)[1:], ctx[-1]


def reference_lattice(c: DraftConfig, P: dict, emb, log_probs, hidden, anchor, valid):
    """vLLM's test_lilicorr._reference_scores for one sequence: nn.MultiheadAttention, concatenated factors."""

    q = "lilicorr."

    def L(x, name):
        return x @ P[q + name + ".weight"].T + P[q + name + ".bias"]

    N, k = log_probs.shape
    rank = torch.arange(k) / max(k - 1, 1)
    top1 = (torch.arange(k) == 0).float()
    feats = torch.stack((log_probs, log_probs.exp(), log_probs - log_probs.amax(-1, keepdim=True),
                         rank.expand_as(log_probs), top1.expand_as(log_probs)), -1)
    x = L(emb, "token_proj") + L(hidden, "pass_hidden_proj").unsqueeze(-2)
    feats = F.layer_norm(feats, (5,), P[q + "feature_norm.weight"], P[q + "feature_norm.bias"])
    x = x + L(F.silu(L(feats, "feature_mlp.up_proj")), "feature_mlp.down_proj")
    x = x + P[q + "slot_embedding"][0, 0, :N] + P[q + "rank_embedding"][0, 0]
    x = x.reshape(1, N * k, -1)
    slots = torch.arange(N).repeat_interleave(k)
    rel = slots[:, None] - slots[None, :]
    bias = P[q + "relative_slot_bias"][:, rel + c.block - 1] + (rel == 0) * P[q + "same_slot_bias"][:, None, None]
    for i in range(c.lili_layers):
        p = f"{q}layers.{i}."
        mha = nn.MultiheadAttention(c.lili_hidden, c.lili_heads, batch_first=True)
        mha.load_state_dict({"in_proj_weight": P[p + "attn.in_proj_weight"], "in_proj_bias": P[p + "attn.in_proj_bias"],
                             "out_proj.weight": P[p + "attn.out_proj.weight"],
                             "out_proj.bias": P[p + "attn.out_proj.bias"]})
        nx = _rms(x, P[p + "attn_norm.weight"], c.eps)
        x = x + mha(nx, nx, nx, attn_mask=bias, need_weights=False)[0]
        nx = _rms(x, P[p + "mlp_norm.weight"], c.eps)
        x = x + L(F.silu(L(nx, f"layers.{i}.mlp.up_proj")), f"layers.{i}.mlp.down_proj")
    x = _rms(x, P[q + "output_norm.weight"], c.eps).view(N, k, -1)
    a = _rms(L(anchor, "context_proj") * valid, P[q + "anchor_norm.weight"], c.eps)
    ex = a.expand_as(x)
    f = F.silu(L(torch.cat((x, ex, x * ex), -1), "factor_input_proj"))
    inc = F.normalize(L(f, "in_head"), dim=-1, eps=c.vector_eps)
    outg = F.normalize(L(f, "out_head"), dim=-1, eps=c.vector_eps)
    a = F.normalize(L(a, "anchor_out_head"), dim=-1, eps=c.vector_eps)
    start = (a * inc[0]).sum(-1)
    pairs = outg[:-1] @ inc[1:].transpose(-1, -2)
    return c.logit_scale * torch.cat((start[None, None].expand(1, k, k), pairs), 0)


# -- tests ----------------------------------------------------------------------------------------------------------
def test_config_reads_the_real_layout():
    c = DraftConfig.from_json(tiny_config())
    assert (c.window, c.block, c.top_k, c.tap_layers, c.lili_inter) == (6, 8, 4, (0, 2, 3), 64)
    assert expected_shapes(c)["fc.weight"] == (64, 192)
    with pytest.raises(ValueError, match="power of two"):
        DraftConfig.from_json(tiny_config(top_k=6))


def test_nvfp4_matches_the_format_tables():
    """Low nibble first, e2m1 x e4m3 x tensor scale: against the fork's numpy tables."""

    g = torch.Generator().manual_seed(1)
    codes, scale, s2 = nvfp4_pack(8, 64, g)
    got = lilicorr.dequantize_nvfp4(codes, scale, s2, torch.float32).numpy()
    c = codes.numpy().astype(np.int64)
    nib = np.stack([c & 0xF, c >> 4], -1).reshape(8, 64)
    want = E2M1[nib] * np.repeat(e4m3(scale.view(torch.uint8).numpy()), 16, axis=1) * np.float32(0.01)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_packed_checkpoint_tensors_dequantize_by_name():
    g = torch.Generator().manual_seed(2)
    codes, scale, s2 = nvfp4_pack(16, 32, g)
    raw = {"layers.0.mlp.up_proj.weight": codes, "layers.0.mlp.up_proj.weight_scale": scale,
           "layers.0.mlp.up_proj.weight_scale_2": s2, "norm.weight": torch.ones(4)}
    out = dense_tensors(raw, torch.bfloat16)
    assert set(out) == {"layers.0.mlp.up_proj.weight", "norm.weight"}
    assert out["layers.0.mlp.up_proj.weight"].shape == (16, 32)
    with pytest.raises(ValueError, match="without their scales"):
        dense_tensors({"a.weight": codes}, torch.bfloat16)


@pytest.mark.parametrize("chunks,depth", [((5,), 7), ((3, 1, 9, 2), 7), ((2, 2), 3), ((11,), 1)])
def test_block_matches_vllm_math(chunks, depth):
    """Context added in chunks (one longer than the 6-row window ring), then a block: rows, anchor, lattice."""

    target = FakeTarget([1, 2, 3])
    d, P = make(target=target)
    c = d.c
    g = torch.Generator().manual_seed(5)
    taps = torch.randn((sum(chunks), 3 * c.hidden), generator=g)
    at = 0
    for n in chunks:
        d.add_taps(taps[at:at + n])
        at += n
    assert d.context_end == at
    pending = 11
    hidden = d.block_hidden(pending, depth)
    want, anchor = reference_block(c, P, taps, pending, depth, target.embed)
    torch.testing.assert_close(hidden, want, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(d.anchor, anchor, atol=2e-5, rtol=2e-5)
    ids, log_probs, scores = d.candidates(pending, depth)
    logits = target.lm_head(want)
    vals, want_ids = logits.topk(c.top_k, -1)
    assert torch.equal(ids, want_ids)
    torch.testing.assert_close(log_probs, vals - logits.logsumexp(-1, keepdim=True))
    lattice = reference_lattice(c, P, target.embed(ids.reshape(-1)).view(depth, c.top_k, -1), log_probs, want,
                                anchor, 1.0)
    torch.testing.assert_close(scores, lattice, atol=1e-4, rtol=1e-4)


def test_bf16_block_stays_near_fp32():
    d32, _ = make()
    d16, _ = make(dtype=torch.bfloat16)
    taps = torch.randn((9, 3 * 64), generator=torch.Generator().manual_seed(4))
    for d in (d32, d16):
        d.add_taps(taps)
    a, b = d32.block_hidden(5, 7), d16.block_hidden(5, 7).float()
    assert (a - b).norm() / a.norm() < 0.03


def test_walk_follows_the_lattice():
    d, _ = make()
    d.add_taps(torch.zeros((2, 192)))
    cand = np.array([[10, 11], [20, 21], [30, 31]])
    scores = np.zeros((3, 2, 2))
    scores[0, :, 1] = 1.0          # anchor -> 11
    scores[1, 1, 0] = 1.0          # 11 -> 20
    scores[2, 0, 1] = 1.0          # 20 -> 31
    scores[2, 1, 0] = 5.0          # (21 -> 30, not on the path)
    assert d.walk(cand, scores, None) == [11, 20, 31]
    drawn = d.walk(cand, scores * 50, Sampling(seed=9, temperature=1.0))
    assert drawn == [11, 20, 31]   # noise is O(1), the path's margin is 50


def test_snapshot_restores_the_context():
    d, _ = make()
    taps = torch.randn((8, 192), generator=torch.Generator().manual_seed(6))
    d.add_taps(taps[:5])
    snap = d.snapshot()
    first = d.block_hidden(3, 4)
    d.add_taps(taps[5:])
    d.restore(snap)
    torch.testing.assert_close(d.block_hidden(3, 4), first)
    d.reset()
    assert d.context_end == 0 and d.propose(3, 4, None) == []


class Oracle:
    """A drafter that knows the serial reply: right for ``good`` drafts a round, then one wrong token."""

    def __init__(self, reply: list[int], prompt_len: int, good: int, block: int = 8) -> None:
        self.reply, self.base, self.good = reply, prompt_len, good
        self.context_end = prompt_len
        self.c = type("C", (), {"block": block})()
        self.calls: list[int] = []

    def propose(self, pending, depth, sampling):
        self.calls.append(depth)
        at = self.context_end - self.base + 1          # reply index of the first draft
        right = self.reply[at:at + depth]
        drafts = right[:self.good] + [(t + 1) % 300 for t in right[self.good:self.good + 1]]
        return drafts[:depth]

    def add_taps(self, taps):
        self.context_end += taps.shape[0]


def drafted(prompt, count, sampling, drafter_factory, **kw):
    t = FakeTarget(prompt, eos=kw.pop("eos", ()))
    first = t.sample(t.prompt_logits, [t.pos], sampling)[0]
    d = drafter_factory(t)
    if isinstance(d, Drafter):
        d.add_taps(t.taps_of(0, list(prompt)))
    return lilicorr_decode(t, d, first, count, sampling, **kw), t, d


SAMPLINGS = [None, Sampling(seed=77, temperature=1.0, top_k=20, top_p=0.95),
             Sampling(seed=5, temperature=0.6, top_k=0, top_p=0.9)]


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk", "nucleus"])
@pytest.mark.parametrize("good", [0, 2, 7])
def test_oracle_drafts_equal_serial_and_count_acceptance(sampling, good):
    prompt = [4, 9, 2, 6]
    want = serial(prompt, 40, sampling)
    res, t, d = drafted(prompt, 40, sampling, lambda t: Oracle(want, len(prompt), good))
    assert res.tokens == want
    assert t.pos == len(prompt) + len(want) - 1 and d.context_end == t.pos
    assert sum(res.keeps) == len(want) - 1 and res.rounds == len(res.keeps) == t.verifies
    assert res.accepted == sum(k - 1 for k in res.keeps)
    assert res.drafted == sum(w - 1 for w in res.widths)
    # each round drafts min(good + 1, 7, what count leaves) and keeps the good ones plus the target's own token
    widths, keeps, have = [], [], 1
    while have < 40:
        n = min(good + 1, 7, 40 - have - 1)
        widths.append(1 + n)
        keeps.append(1 + min(good, n))
        have += keeps[-1]
    assert (res.widths, res.keeps) == (widths, keeps)
    assert res.acceptance_length == pytest.approx((len(want) - 1) / res.rounds)


@pytest.mark.parametrize("sampling", SAMPLINGS[:2], ids=["greedy", "topk"])
def test_lilicorr_drafter_replies_equal_serial(sampling):
    """The real (random, tiny) drafter end to end on the fake target: same tokens, context tracked."""

    prompt = [4, 9, 2, 6, 8, 1, 3]
    want = serial(prompt, 30, sampling)
    res, t, d = drafted(prompt, 30, sampling, lambda t: make(target=t)[0])
    assert res.tokens == want
    assert d.context_end == t.pos == len(prompt) + len(want) - 1
    assert res.drafted == sum(w - 1 for w in res.widths) and res.drafted > 0


def test_count_caps_the_last_round():
    prompt = [4, 9, 2]
    want = serial(prompt, 10, None)
    res, t, _ = drafted(prompt, 10, None, lambda t: Oracle(want, 3, 7))
    assert res.tokens == want and res.widths == [8, 1] and res.keeps == [8, 1]
    assert t.pos == 3 + 9                                # nothing committed past the reply


def test_end_token_stops_inside_a_round():
    prompt = [4, 9, 2]
    probe = serial(prompt, 20, None)
    eos = (probe[5],)
    want = serial(prompt, 20, None, stop_eos=True, eos=eos)
    assert want[-1] == eos[0] and len(want) <= 6
    res, t, _ = drafted(prompt, 20, None, lambda t: Oracle(probe, 3, 7), stop_eos=True, eos=eos)
    assert res.tokens == want
    assert t.pos == 3 + len(want) - 1


def test_on_tokens_true_stops():
    prompt = [4, 9, 2]
    want = serial(prompt, 30, None)
    seen: list[list[int]] = []

    def feed(new):
        seen.append(new)
        return len(seen) == 2

    res, _, _ = drafted(prompt, 30, None, lambda t: Oracle(want, 3, 3), on_tokens=feed)
    assert res.rounds == 2 and [x for s in seen for x in s] == res.tokens[1:]
    assert res.tokens == want[:len(res.tokens)] and len(res.tokens) == 1 + 4 + 4


def test_depth_zero_is_serial_rounds():
    prompt = [4, 9, 2]
    want = serial(prompt, 12, SAMPLINGS[1])
    res, _, d = drafted(prompt, 12, SAMPLINGS[1], lambda t: Oracle(want, 3, 7), depth=0)
    assert res.tokens == want and res.widths == [1] * 11 and d.calls == []


def test_drafter_must_hold_the_prompt():
    t = FakeTarget([1, 2])
    d, _ = make(target=t)
    with pytest.raises(ValueError, match="holds 0 positions"):
        lilicorr_decode(t, d, 5, 4, None)


class SamplingTarget(FakeTarget):
    """An engine that samples inside ``verify`` (``tokens()``), like BoltEngine and Nemotron-H's engine."""

    sample = None

    def set_sampling(self, sampling):
        self.rule = sampling

    def verify(self, tokens):
        logits = super().verify(tokens)
        self._tokens = FakeTarget.sample(self, logits, [self.pos + 1 + r for r in range(len(tokens))], self.rule)
        return logits

    def tokens(self):
        return list(self._tokens)


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "topk", "nucleus"])
def test_engine_sampler_in_verify(sampling):
    prompt = [4, 9, 2, 6]
    want = serial(prompt, 25, sampling)
    t = SamplingTarget(prompt)
    first = FakeTarget.sample(t, t.prompt_logits, [t.pos], sampling)
    res = lilicorr_decode(t, Oracle(want, len(prompt), 2), first[0], 25, sampling)
    assert res.tokens == want and t.rule is sampling and res.accepted > 0
