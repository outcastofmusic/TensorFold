"""Nemotron 3.5 Bolt's adaptive draft depth on the GPU: the same tokens as fixed-depth and serial decoding, and prose
that no longer decodes below serial speed. The real model and drafter only (TF_BOLT_TARGET, TF_BOLT_DRAFT)."""

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

pytestmark = pytest.mark.skipif(not (TARGET / "model.safetensors.index.json").exists()
                                or not (DRAFT / "config.json").exists(),
                                reason="needs the Bolt checkpoint and its LiLiCorr drafter")

PROSE = ("Write a long, quiet short story about a lighthouse keeper who keeps a weather log for thirty years, and "
         "about the daughter who reads it after he dies. Use plain, unhurried sentences.")
CODE = "Write a Python module with a binary search tree class: insert, delete, search, in-order traversal and tests."


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


def _run(eng, prompt, sampling, *, draft: bool, adaptive: bool) -> tuple[list[int], dict]:
    eng.adaptive = adaptive
    eng.store.kept.clear()                        # every run prefills afresh, so the decode alone differs
    out: list[int] = []
    stats = eng.generate(prompt, 400, sampling, lambda new: out.extend(new) and False, draft=draft, stop_eos=False)
    return out, stats


@pytest.mark.parametrize("sampling", [None, Sampling(seed=11, temperature=0.7, top_k=0, top_p=0.95)],
                         ids=["greedy", "sampled"])
@pytest.mark.parametrize("text", [PROSE, CODE], ids=["prose", "code"])
def test_adaptive_depth_keeps_the_tokens_and_the_speed(eng, text, sampling):
    prompt = _prompt(text)
    _run(eng, prompt, sampling, draft=True, adaptive=True)            # costs measured, kernels warm
    serial, s0 = _run(eng, prompt, sampling, draft=False, adaptive=True)
    fixed, s1 = _run(eng, prompt, sampling, draft=True, adaptive=False)
    adaptive, s2 = _run(eng, prompt, sampling, draft=True, adaptive=True)
    assert adaptive == fixed == serial
    print(f"\n{'prose' if text is PROSE else 'code'} {'greedy' if sampling is None else 'sampled'}: serial "
          f"{s0['decode_tps']} tok/s; fixed 7: {s1['decode_tps']} tok/s, {s1['accepted']}/{s1['drafted']} accepted; "
          f"adaptive: {s2['decode_tps']} tok/s, {s2['accepted']}/{s2['drafted']} accepted, {s2['serial_rounds']}/"
          f"{s2['rounds']} serial rounds", flush=True)
    assert s2["decode_tps"] >= 0.95 * max(s0["decode_tps"], s1["decode_tps"])     # the better of serial and 7
