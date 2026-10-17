"""LFM2-MoE's CUDA engine: the kernels against their fp32 definitions on a tiny checkpoint in the real layout, and the
24B-A2B FP8 checkpoint against the fp32 reference's golden files (tools/lfm2_moe_golden.py) and for decode speed.

The real-checkpoint tests need the folder (TF_LFM2_MOE_MODEL), its golden files (TF_LFM2_MOE_GOLDEN) and 40 GiB of
free memory; they are skipped without them. Bytes a decoded token reads, from the headers: 1.43 GB of routed experts
(4 of 64, 38 layers), 1.01 GB of bf16 conv projections, 0.52 GB of tied head, 0.25 GB of attention and dense FFN:
3.2 GB, so about 85 tok/s at the GB10's 273 GB/s and 60-70 at what a serial decode reaches in practice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling

REAL = Path(os.environ.get("TF_LFM2_MOE_MODEL", "/home/agis/models/LFM2.5-24B-A2B-FP8"))
GOLDEN = Path(os.environ.get("TF_LFM2_MOE_GOLDEN", "/home/agis/models/LFM2.5-24B-A2B-FP8-golden"))
IDS = [1, 5, 9, 300, 22, 17, 401, 3, 3, 88, 120, 260, 7, 64, 511, 2]
NEAR_TIE = 0.02
# The engine's code on the CPU against the golden files (2026-10-17): teacher-forced top-20 logit errors of at most
# 2.31 (median 0.57) on chat and 0.91 (0.19) on tools, nearly all from near-tie expert picks (20 picks within 0.0025
# of the next expert's score on chat); with the reference run on the engine's own picks, at most 0.23 (median 0.11).
# Its greedy token differed only where the reference's top two logits were within 0.2.
GOLDEN_MAX, GOLDEN_MEDIAN, PICKS_MAX = 3.0, 0.8, 0.5
SURE_GAP = 1.0            # a step whose top two reference logits are this far apart must keep its greedy token
DECODE_FLOOR = 25.0       # tok/s a serial greedy decode must reach (bytes bound about 85; launch-bound below that)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.lfm2_moe_tiny import write

    return write(tmp_path_factory.mktemp("lfm2_moe") / "fp8")


@pytest.fixture(scope="module")
def weights(tiny):
    from tensorfold.families.lfm2_moe.cuda.weights import load

    return load(tiny, "cuda")


def _engine(weights, max_len: int = 1024):
    from tensorfold.families.lfm2_moe.cuda.engine import Engine

    return Engine(weights, max_len=max_len)


def _close(got, want, rel: float) -> bool:
    return float((got.float().cpu() - want.float().cpu()).abs().max()) <= rel * float(want.abs().max())


# -- kernels on the tiny checkpoint ---------------------------------------------------------------------------------

def test_block_fp8_linears_equal_their_fp32_decode(tiny, weights):
    from tensorfold.families.lfm2_moe.checkpoint import Tensors

    t = Tensors(tiny)
    p = "model.layers.2.self_attn."
    want_w = torch.cat([t.linear(p + n) for n in ("q_proj", "k_proj", "v_proj")])
    x = torch.randn(5, want_w.shape[1], generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    want = x.float() @ want_w.t()
    lin = weights.blocks[2].attn.qkv
    for prompt in (False, True):
        assert _close(lin(x.cuda(), prompt), want, 0.01)


def test_the_expert_kernels_equal_the_grouped_path(weights):
    from tensorfold.families.lfm2_moe.cuda import experts as X

    ex = weights.blocks[2].moe.experts
    x = torch.randn(4, weights.config.hidden, device="cuda").to(torch.bfloat16)
    ids = torch.tensor([[0, 3], [7, 1], [3, 0], [5, 4]], device="cuda")
    assert _close(X.pairs(x, ids, ex), X.grouped(x, ids, ex), 0.01)


def test_the_block_decode_kernel_gives_dequantize_s_bits():
    from tensorfold.families.lfm2_moe.checkpoint import dequantize
    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tests.lfm2_moe_tiny import e4m3_codes

    w = e4m3_codes(3072, 2048, 1)
    s = torch.rand(24, 16, generator=torch.Generator().manual_seed(2)) * 1e-2
    want = dequantize(w, s)
    work = torch.full((w.numel(),), float("nan"), device="cuda")
    got = X.dense(w.cuda(), s.cuda(), work)
    assert got.data_ptr() == work.data_ptr() and torch.equal(got.cpu(), want)


def test_the_grouped_path_keeps_its_bits(weights):
    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tests.lfm2_moe_tiny import grouped_by_dequantize

    ex = weights.blocks[2].moe.experts
    g = torch.Generator().manual_seed(3)
    x = torch.randn(300, weights.config.hidden, generator=g).to(torch.bfloat16).cuda()
    ids = torch.randint(0, ex.count, (300, 2), generator=g).cuda()
    assert torch.equal(X.grouped(x, ids, ex), grouped_by_dequantize(x, ids, ex))


def test_the_engine_computes_the_reference_on_its_own_picks(tiny, weights, monkeypatch):
    from tests.lfm2_moe_tiny import Picks, reference_logits

    picks = Picks(monkeypatch, len(weights.config.layers) - weights.config.dense_layers)
    got = _engine(weights).forward(IDS, prompt=True, every=True)
    picks.replay()
    want = reference_logits(tiny, IDS)
    assert _close(got, want, 0.02) and all(gap < NEAR_TIE for gap in picks.gaps)
    assert (got.float().cpu().argmax(-1) == want.argmax(-1)).all()


def test_chunks_and_decode_steps_continue_the_prompt(weights):
    e = _engine(weights)
    whole = e.forward(IDS, prompt=True, every=True)
    e.reset()
    parts = torch.cat([e.forward(IDS[:7], prompt=True, every=True), e.forward(IDS[7:], prompt=True, every=True)])
    e.reset()
    e.forward(IDS[:9], prompt=True)
    steps = torch.cat([e.forward([t], prompt=False) for t in IDS[9:]])
    assert _close(parts, whole, 0.02) and _close(steps, whole[9:], 0.02)
    assert (steps.argmax(-1) == whole[9:].argmax(-1)).all()


def test_keyed_sampling_repeats_a_seed_and_equals_the_host_rule(weights):
    from tensorfold.engine.exact_sampling import choose
    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    e = _engine(weights)

    def reply(seed: int) -> list[int]:
        s = Sampling(seed, temperature=50.0, top_k=50, top_p=1.0)
        return serial_decode(e, prefill(e, IDS, s), 12, s, stop_eos=False).tokens

    assert reply(3) == reply(3) and len({tuple(reply(s)) for s in range(4)}) > 1
    s = Sampling(11, temperature=0.7, top_k=5, top_p=0.9)
    e.reset()
    e.set_sampling(s)
    logits = e.forward(IDS, prompt=True)
    vals, ids = torch.topk(logits[0].float(), 5 + 8)
    assert e.sample(logits) == choose(vals.cpu().numpy(), ids.cpu().numpy(), len(IDS), s)


# -- the 24B-A2B ------------------------------------------------------------------------------------------------------

def _room() -> bool:
    from tensorfold.cuda.capacity import available_bytes

    return available_bytes(torch) >= 40 << 30


real = pytest.mark.skipif(not (REAL / "config.json").is_file(), reason="needs the LFM2.5-24B-A2B-FP8 checkpoint")


@pytest.fixture(scope="module")
def big():
    if not _room():
        pytest.skip("needs 40 GiB of free memory")
    from tensorfold.families.lfm2_moe.cuda.app import Lfm2MoeEngine

    engine = Lfm2MoeEngine(REAL, context=16384, context_explicit=True)
    yield engine
    del engine
    torch.cuda.empty_cache()


def _golden(name: str) -> dict:
    path = GOLDEN / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"no golden file {path} (tools/lfm2_moe_golden.py writes it)")
    return json.loads(path.read_text())


def _teacher_forced(e, g: dict) -> list[torch.Tensor]:
    """The engine's logits [V] for the prompt's next token and after each of the reference's greedy tokens."""

    e.reset()
    e.set_sampling(None)
    ids = g["prompt_ids"]
    for s in range(0, len(ids), e.prefill_rows):
        logits = e.forward(ids[s:s + e.prefill_rows], prompt=True)
    rows = [logits[0]]
    for tok in g["greedy"][:-1]:
        rows.append(e.forward([tok], prompt=False)[0])
    return [r.float().cpu() for r in rows]


@real
@pytest.mark.parametrize("name", ["chat", "tools", "long"])
def test_logits_follow_the_fp32_reference_and_greedy_tokens_match(big, name):
    """Teacher-forced on the reference's greedy tokens: top-20 logit errors within the CPU-measured bounds, the
    reference's token wherever its top two logits are SURE_GAP apart, and the engine's own greedy reply equal to the
    reference's up to its first step closer than that."""

    from tensorfold.families.lfm2_moe.cuda.engine import prefill, serial_decode

    g = _golden(name)
    rows = _teacher_forced(big.e, g)
    errors = [float((row[t["ids"]] - torch.tensor(t["logits"])).abs().max()) for row, t in zip(rows, g["step_top"])]
    gaps = [t["logits"][0] - t["logits"][1] for t in g["step_top"]]
    wrong = [i for i, (row, gap) in enumerate(zip(rows, gaps))
             if gap >= SURE_GAP and int(row.argmax()) != g["greedy"][i]]
    median = sorted(errors)[len(errors) // 2]
    print(f"{name}: top-20 logit error largest {max(errors):.3f}, median {median:.3f}; wrong sure steps {wrong}")
    assert max(errors) <= GOLDEN_MAX and median <= GOLDEN_MEDIAN and not wrong
    out = serial_decode(big.e, prefill(big.e, g["prompt_ids"], None), len(g["greedy"]), None, stop_eos=False).tokens
    sure = next((i for i, gap in enumerate(gaps) if gap < SURE_GAP), len(gaps))
    assert out[:sure] == g["greedy"][:sure], (out, g["greedy"], sure)


@real
def test_logits_on_the_engine_s_own_picks_follow_the_reference(big, monkeypatch):
    """The chat golden's prompt and greedy tokens as one prompt; the fp32 reference (CPU, about a minute) takes the
    engine's expert picks, so what remains is the kernels' rounding."""

    from tensorfold.families.lfm2_moe.reference import Reference, State
    from tests.lfm2_moe_tiny import Picks

    g = _golden("chat")
    ids = g["prompt_ids"] + g["greedy"][:-1]
    picks = Picks(monkeypatch, len(big.w.config.layers) - big.w.config.dense_layers)
    big.e.reset()
    got = big.e.forward(ids, prompt=True, every=True).float().cpu()
    picks.replay()
    want = Reference(REAL).forward(ids, State())
    n = len(g["prompt_ids"]) - 1
    top = want[n:].topk(20, dim=-1).indices
    worst = float((got[n:] - want[n:]).abs().gather(1, top).max())
    print(f"chat on the engine's picks: largest top-20 logit error {worst:.3f}; {len(picks.gaps)} picks differ, "
          f"the widest within {max(picks.gaps, default=0.0):.4f}")
    assert worst <= PICKS_MAX and all(gap < NEAR_TIE for gap in picks.gaps)


@real
def test_the_real_experts_kernels_equal_the_grouped_path(big):
    from tensorfold.families.lfm2_moe.cuda import experts as X

    ex = big.w.blocks[2].moe.experts
    x = torch.randn(3, big.w.config.hidden, device="cuda").to(torch.bfloat16)
    ids = torch.tensor([[0, 17, 33, 63], [5, 6, 7, 8], [63, 0, 1, 2]], device="cuda")
    assert _close(X.pairs(x, ids, ex), X.grouped(x, ids, ex), 0.01)


@real
def test_the_real_grouped_path_keeps_its_bits(big):
    from tensorfold.families.lfm2_moe.cuda import experts as X
    from tests.lfm2_moe_tiny import grouped_by_dequantize

    ex = big.w.blocks[2].moe.experts
    g = torch.Generator().manual_seed(4)
    x = torch.randn(512, big.w.config.hidden, generator=g).to(torch.bfloat16).cuda()
    ids = torch.randint(0, ex.count, (512, 4), generator=g).cuda()
    assert torch.equal(X.grouped(x, ids, ex, big.e.decoded), grouped_by_dequantize(x, ids, ex))


@real
def test_decode_speed(big):
    """A 256-token greedy reply to the chat prompt; prints tok/s next to the 85 tok/s bytes bound."""

    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(REAL / "tokenizer.json"))
    text = ChatTemplate(REAL).render([{"role": "user", "content": "Write a short story about a lighthouse."}],
                                     tools=None, enable_thinking=True)
    prompt = tok.encode(text, add_special_tokens=False).ids
    big.generate(prompt, 16, None, None, stop_eos=False)                     # warm the kernels
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats = big.generate(prompt, 256, None, None, stop_eos=False)
    wall = time.perf_counter() - started
    print(f"decode {stats['decode_tps']} tok/s, prefill {stats['prefill_tps']} tok/s, {wall:.1f} s for 256 tokens")
    assert stats["decode_tps"] >= DECODE_FLOOR
