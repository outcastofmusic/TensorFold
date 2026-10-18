"""Nemotron 3.5 Bolt's drafter-only Q8_0 lm_head on the GPU: the row kernel against the bf16 head's values, its speed
beside the bf16 head's, and drafted replies that equal serial ones with either head, on the tiny checkpoint and the
real model (TF_BOLT_TARGET, TF_BOLT_DRAFT), with acceptance and tok/s printed for both heads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling

TARGET = Path(os.environ.get("TF_BOLT_TARGET", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB"))
DRAFT = Path(os.environ.get("TF_BOLT_DRAFT", "/home/agis/models/Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"))
full_model = pytest.mark.skipif(not (TARGET / "model.safetensors.index.json").exists()
                                or not (DRAFT / "config.json").exists(),
                                reason="needs the Bolt checkpoint and its LiLiCorr drafter")
SAMPLINGS = [None, Sampling(seed=11, temperature=0.6, top_k=20, top_p=0.95)]
PROMPT = [5, 77, 300, 2, 999, 14, 15, 600, 41, 42, 43, 7, 8, 9, 512, 513, 100, 101, 102, 103, 104]


def _packed(v: int, d: int, seed: int = 0) -> torch.Tensor:
    """A random Q8_0 head on the GPU: uint8 [v, d // 32 * 34], fp16 scales near the real head's."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    scales = (torch.rand((v, d // 32, 1), generator=g, device="cuda") * 2e-3 + 1e-4).to(torch.float16)
    codes = torch.randint(-127, 128, (v, d // 32, 32), generator=g, device="cuda", dtype=torch.int8)
    return torch.cat([scales.view(torch.uint8), codes.view(torch.uint8)], dim=-1).reshape(v, -1).contiguous()


@pytest.mark.parametrize("rows,vocab,width", [(1, 1000, 512), (7, 4099, 4096), (16, 64, 128), (17, 1000, 512)])
def test_the_row_kernel_sums_the_bf16_heads_values(rows, vocab, width):
    from tensorfold.families.nemotron_bolt.cuda.q8_head import logits
    from tensorfold.families.nemotron_bolt.cuda.weights import dequantize_q8_0_rows

    packed = _packed(vocab, width)
    head = dequantize_q8_0_rows(packed)                               # the bf16 head the verify reads
    h = torch.randn((rows, width), device="cuda").to(torch.bfloat16)
    got = logits(h, packed)
    want = h.float() @ head.float().T                                 # bf16 products are exact in fp32
    assert got.shape == (rows, vocab) and got.dtype == torch.float32
    assert (got - want).abs().max() <= 1e-4 * want.abs().max()       # only the summation order differs
    k = min(8, vocab)
    agree = (got.topk(k).indices.sort().values == want.topk(k).indices.sort().values).all(dim=1).float().mean()
    assert agree >= 0.9


def test_the_q8_0_head_reads_half_the_bytes_of_the_bf16_one():
    from tensorfold.families.nemotron_bolt.cuda.engine import dense
    from tensorfold.families.nemotron_bolt.cuda.q8_head import logits
    from tensorfold.families.nemotron_bolt.cuda.weights import dequantize_q8_0_rows

    packed = _packed(131072, 4096)                                   # Bolt's head: 0.53 GiB packed, 1 GiB bf16
    head = dequantize_q8_0_rows(packed)
    h = torch.randn((7, 4096), device="cuda").to(torch.bfloat16)     # a block pass's 7 mask rows

    def ms(fn) -> float:
        for _ in range(3):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 20

    q8, bf16 = ms(lambda: logits(h, packed)), ms(lambda: dense(h, head, False))
    print(f"\nlm_head over 7 rows: Q8_0 {q8:.2f} ms, bf16 {bf16:.2f} ms", flush=True)
    assert q8 < bf16


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_the_tiny_engine_drafts_with_either_head_and_replies_the_same(tmp_path, monkeypatch, sampling):
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine
    from tests.bolt_lilicorr_fakes import write_draft
    from tests.bolt_tiny import write

    tiny = write(tmp_path / "bolt")
    draft = write_draft(tmp_path / "draft", hidden=512, vocab=1024, inter=128)    # the tiny target's width
    replies = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("TF_BOLT_DRAFT_Q8", flag)
        eng = BoltEngine(tiny, drafter=draft, context=512, context_explicit=True)
        assert (eng.w.head_q8 is None) == (flag == "0")
        for draft_on in (True, False):
            got: list[int] = []
            stats = eng.generate(PROMPT, 40, sampling, lambda new, got=got: got.extend(new) and False,
                                 draft=draft_on, stop_eos=False)
            assert stats["drafts"] == draft_on
            replies[flag, draft_on] = got
        del eng
    assert len({tuple(r) for r in replies.values()}) == 1


# -- the real model -------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eng():
    from tensorfold.families.nemotron_bolt.cuda.app import BoltEngine

    return BoltEngine(TARGET, drafter=DRAFT, context=16384, context_explicit=True)


def _prompt(text: str) -> list[int]:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(TARGET / "tokenizer.json"))
    t = ChatTemplate(TARGET)
    rendered = t.template.render(**t.specials, messages=[{"role": "user", "content": text}],
                                 add_generation_prompt=True)
    return tok.encode(rendered, add_special_tokens=False).ids


PROSE = "Explain how a Mamba-2 state space layer differs from attention, in four paragraphs."
CODE = "Write a Python module with a binary search tree class: insert, delete, search, in-order traversal and tests."


@full_model
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
@pytest.mark.parametrize("text", [PROSE, CODE], ids=["prose", "code"])
def test_full_model_acceptance_and_speed_with_either_draft_head(eng, text, sampling):
    assert eng.w.head_q8 is not None                                 # TF_BOLT_DRAFT_Q8 is on by default
    prompt = _prompt(text)
    heads = {"q8": eng.draft_head, "bf16": eng.e.head_logits}

    def run(draft: bool, adaptive: bool) -> tuple[list[int], dict]:
        eng.adaptive = adaptive
        eng.store.kept.clear()
        out: list[int] = []
        stats = eng.generate(prompt, 300, sampling, lambda new: out.extend(new) and False, draft=draft,
                             stop_eos=False)
        return out, stats

    try:
        eng.drafter.lm_head = heads["q8"]
        run(True, True)                                               # kernels warm, costs measured
        serial, s0 = run(False, True)
        stats = {}
        for name, head in heads.items():
            eng.drafter.lm_head = head
            for adaptive in (False, True):
                out, st = run(True, adaptive)
                assert out == serial, (name, adaptive)
                stats[name, adaptive] = st
    finally:
        eng.drafter.lm_head = heads["q8"]
        eng.adaptive = True
    lines = [f"{name} head: fixed 7 {stats[name, False]['accepted']}/{stats[name, False]['drafted']} accepted, "
             f"{stats[name, False]['decode_tps']} tok/s; adaptive {stats[name, True]['decode_tps']} tok/s"
             for name in heads]
    print(f"\n{'prose' if text is PROSE else 'code'} {'greedy' if sampling is None else 'sampled'}: serial "
          f"{s0['decode_tps']} tok/s; " + "; ".join(lines), flush=True)
