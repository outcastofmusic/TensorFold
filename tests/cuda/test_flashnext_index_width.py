"""An indexer head wider than the attention heads (the tiny NVFP4 checkpoint: head_dim 64, indexer 128) is written
whole: attn_prep fills every column of the indexer queries and keys, so a prefill no longer reads a cache column it
never wrote. Before 25 Oct 2026 the columns past head_dim kept what the memory held, and a prompt's bits followed
whatever earlier engine or request had used it."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the attention prep kernel needs an NVIDIA GPU", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write

from tensorfold.families.qwen4_exp.cuda import glue
from tensorfold.families.qwen4_exp.cuda.decode import Engine, chunk_plan, prefill_begin, prefill_chunk
from tensorfold.families.qwen4_exp.cuda.weights import load


@pytest.mark.parametrize("bits", [0, 8])
def test_every_indexer_column_is_written_when_the_index_is_wider_than_a_head(bits):
    nq, nkv, hd, ni, ihd, half, rows, cap = 2, 1, 64, 4, 128, 8, 5, 16
    g = torch.Generator(device="cuda").manual_seed(3)
    p = torch.randn((rows, nq * 2 * hd + 2 * nkv * hd + ni * ihd + ihd), generator=g, device="cuda").to(torch.bfloat16)
    nan = {"dtype": torch.bfloat16, "device": "cuda"}
    q, iq, ikc = (torch.full(shape, float("nan"), **nan) for shape in ((rows, nq, hd), (rows, ni, ihd), (cap, ihd)))
    if bits:
        kc = torch.zeros((cap * nkv, hd), dtype=torch.int8, device="cuda")
        vc, ks = torch.zeros_like(kc), torch.zeros((cap * nkv, hd // 32), **nan)
        vs = torch.zeros_like(ks)
    else:
        kc, ks, vs = torch.zeros((cap * nkv, hd), **nan), None, None
        vc = torch.zeros_like(kc)
    glue.attn_prep(p, torch.tensor([3], dtype=torch.int32, device="cuda"), torch.ones(hd, device="cuda"),
                   torch.ones(hd, device="cuda"), torch.ones(ihd, device="cuda"), torch.rand(half, device="cuda"),
                   q, kc, vc, iq, ikc, 1e-6, q_heads=nq, kv_heads=nkv, head_dim=hd, index_heads=ni, index_dim=ihd,
                   ks=ks, vs=vs, bits=bits)
    assert not iq.isnan().any()
    assert not ikc[3:3 + rows].isnan().any() and ikc[:3].isnan().all() and ikc[3 + rows:].isnan().all()
    raw = p[:, nq * 2 * hd + 2 * nkv * hd + ni * ihd:]
    assert torch.equal(ikc[3:3 + rows].view(torch.int16), raw.contiguous().view(torch.int16))   # the raw keys


def test_a_prefill_does_not_read_what_the_index_cache_held():
    w = load(write(Path(tempfile.mkdtemp()), layers=4, experts=16, hidden=512, mxfp8=True, ple_nvfp4=True), mtp=True)
    prompt = torch.randint(1, w.cfg.vocab, (3073,), generator=torch.Generator().manual_seed(11)).tolist()

    def logits(fill: int) -> bytes:
        e = Engine(w, capacity=3200, max_rows=8, prefill_rows=2048, kv_dtype="bf16")
        for cache in e.st.ikc:
            cache.view(torch.uint8).fill_(fill)
        start, last = prefill_begin(e, prompt), None
        for begin, end, _ in chunk_plan(start, len(prompt), e.prefill_rows, ()):
            last, _ = prefill_chunk(e, prompt, begin, end=end)
        return last.view(torch.uint8).cpu().numpy().tobytes()

    assert logits(0x00) == logits(0x3F) == logits(0x7F)
