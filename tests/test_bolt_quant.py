"""Bolt's Q8_0, IQ2_XXS and NVFP4 decoders equal a line-by-line transcription of llama.cpp's and vLLM's loops."""

import hashlib
import json
import os
import random
import struct
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch

MODEL = Path(os.environ.get("TF_BOLT_MODEL", "/models/Nemotron-3.5-Bolt-74B-A9B-24GB"))


def _quant():
    from tensorfold.families.nemotron_bolt import quant

    return quant


def _fp16_bytes(rng: random.Random) -> bytes:
    """A finite fp16 scale of either sign, from subnormal to large."""

    return struct.pack("<e", rng.choice([-1, 1]) * rng.uniform(2.0 ** -20, 4.0) * 2.0 ** rng.randint(-8, 4))


def _q8_0_scalar(block: bytes) -> list[float]:
    """ggml-quants.c dequantize_row_q8_0, one block: y[j] = qs[j] * d."""

    d = struct.unpack("<e", block[:2])[0]
    return [q * d for q in struct.unpack("<32b", block[2:])]


def _iq2xxs_scalar(block: bytes, grid, ksigns) -> list[float]:
    """ggml-quants.c dequantize_row_iq2_xxs, one block, written as the C loop is."""

    d = struct.unpack("<e", block[:2])[0]
    qs = block[2:]
    y = []
    for ib32 in range(8):
        aux0, aux1 = struct.unpack("<II", qs[8 * ib32:8 * ib32 + 8])
        aux8 = struct.pack("<I", aux0)
        db = d * (0.5 + (aux1 >> 28)) * 0.25
        for l in range(4):
            g = struct.pack("<Q", grid[aux8[l]])
            signs = ksigns[(aux1 >> 7 * l) & 127]
            for j in range(8):
                y.append(db * g[j] * (-1.0 if signs & (1 << j) else 1.0))
    return y


def _e4m3(b: int) -> float:
    sign = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 0x7
    return sign * (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _packed(blocks: list[bytes], rows: int):
    import torch

    flat = b"".join(blocks)
    return torch.frombuffer(bytearray(flat), dtype=torch.uint8).reshape(rows, -1)


def test_iq2xxs_tables_match_llama_cpp_checksums():
    q = _quant()
    assert len(q.IQ2XXS_GRID) == 256 and len(q.KSIGNS_IQ2XS) == 128
    assert hashlib.sha256(struct.pack("<256Q", *q.IQ2XXS_GRID)).hexdigest() == q.IQ2XXS_GRID_SHA256
    assert hashlib.sha256(bytes(q.KSIGNS_IQ2XS)).hexdigest() == q.KSIGNS_IQ2XS_SHA256
    assert q.IQ2XXS_GRID_SHA256 == "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
    assert q.KSIGNS_IQ2XS_SHA256 == "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31"


def test_iq2xxs_tables_have_their_documented_structure():
    q = _quant()
    # every grid byte is one of IQ2_XXS's three magnitudes, and the 256 rows are distinct
    assert {(v >> (8 * j)) & 0xFF for v in q.IQ2XXS_GRID for j in range(8)} == {0x08, 0x19, 0x2B}
    assert len(set(q.IQ2XXS_GRID)) == 256
    # a sign index keeps its 7 bits and gains bit 7 when its parity is odd, so every byte has even parity
    assert list(q.KSIGNS_IQ2XS) == [i | ((i.bit_count() & 1) << 7) for i in range(128)]


def test_q8_0_hand_built_blocks_equal_the_scalar_loop():
    import torch

    q = _quant()
    rng = random.Random(1)
    blocks = [_fp16_bytes(rng) + bytes(rng.randrange(256) for _ in range(32)) for _ in range(12)]
    blocks[0] = struct.pack("<e", 0.5) + struct.pack("<32b", *range(-128, 128, 8))      # qs -128..120 step 8
    got = q.dequantize_q8_0(_packed(blocks, 3), dtype=torch.float32)
    assert got.shape == (3, 4 * 32)
    want = torch.tensor([v for b in blocks for v in _q8_0_scalar(b)], dtype=torch.float32).reshape(3, -1)
    assert torch.equal(got, want)
    assert got[0, :4].tolist() == [-64.0, -60.0, -56.0, -52.0]


def test_iq2xxs_hand_built_blocks_equal_the_scalar_loop():
    import torch

    q = _quant()
    rng = random.Random(2)
    blocks = [_fp16_bytes(rng) + bytes(rng.randrange(256) for _ in range(64)) for _ in range(8)]
    # block 0: run 0 picks grid rows 0, 1, 2, 255, sign indices 0, 1, 127, 64 and scale 15; other runs all zero
    signs = 0 | (1 << 7) | (127 << 14) | (64 << 21) | (15 << 28)
    blocks[0] = struct.pack("<e", 1.0) + bytes([0, 1, 2, 255]) + struct.pack("<I", signs) + bytes(56)
    got = q.dequantize_iq2xxs(_packed(blocks, 2), dtype=torch.float32)
    assert got.shape == (2, 4 * 256)
    want = [v for b in blocks for v in _iq2xxs_scalar(b, q.IQ2XXS_GRID, q.KSIGNS_IQ2XS)]
    assert torch.equal(got, torch.tensor(want, dtype=torch.float32).reshape(2, -1))
    db = 1.0 * 15.5 * 0.25
    assert got[0, :8].tolist() == [8 * db] * 8                                    # row 0, no signs
    assert got[0, 8:16].tolist() == [-0x2B * db] + [8 * db] * 6 + [-8 * db]       # row 1 0x...082b, signs 129
    assert got[0, 16:24].tolist() == [-v * db for v in (0x19, 0x19, 8, 8, 8, 8, 8, 8)]   # 127 -> every bit set


def test_nvfp4_hand_built_matrix_equals_the_scalar_definition():
    import torch

    q = _quant()
    rng = random.Random(3)
    n, k = 4, 64
    words = [rng.randrange(256) for _ in range(n * k // 2)]
    scales = [rng.choice([b for b in range(256) if b & 0x7F != 0x7F]) for _ in range(n * k // 16)]
    scale2 = 0.0123
    w = torch.tensor(words, dtype=torch.uint8).reshape(n, k // 2)
    s = torch.tensor(scales, dtype=torch.uint8).reshape(n, k // 16).view(torch.float8_e4m3fn)
    got = q.dequantize_nvfp4(w, s, torch.tensor(scale2, dtype=torch.float32), dtype=torch.float32)
    s2 = struct.unpack("<f", struct.pack("<f", scale2))[0]
    want = []
    for r in range(n):
        for c in range(k):
            byte = words[r * k // 2 + c // 2]
            code = byte & 0xF if c % 2 == 0 else byte >> 4
            val = _E2M1[code & 7] * (-1.0 if code & 8 else 1.0)
            blk = struct.unpack("<f", struct.pack("<f", _e4m3(scales[r * k // 16 + c // 16]) * s2))[0]
            want.append(struct.unpack("<f", struct.pack("<f", val * blk))[0])
    assert torch.equal(got, torch.tensor(want, dtype=torch.float32).reshape(n, k))


def test_bf16_output_is_the_fp32_value_rounded_once():
    import torch

    q = _quant()
    rng = random.Random(4)
    p8 = _packed([_fp16_bytes(rng) + bytes(rng.randrange(256) for _ in range(32)) for _ in range(8)], 2)
    p2 = _packed([_fp16_bytes(rng) + bytes(rng.randrange(256) for _ in range(64)) for _ in range(4)], 2)
    assert q.dequantize_q8_0(p8).dtype == torch.bfloat16
    assert torch.equal(q.dequantize_q8_0(p8), q.dequantize_q8_0(p8, dtype=torch.float32).bfloat16())
    assert torch.equal(q.dequantize_iq2xxs(p2), q.dequantize_iq2xxs(p2, dtype=torch.float32).bfloat16())


def test_stacked_leading_dims_decode_row_by_row():
    import torch

    q = _quant()
    gen = torch.Generator().manual_seed(5)
    raw = torch.randint(0, 256, (3, 2, 2 * 66), dtype=torch.uint8, generator=gen)
    raw[..., 0::66] = 0x00
    raw[..., 1::66] = 0x3C                                   # d = 1.0 in every block
    whole = q.dequantize_iq2xxs(raw, dtype=torch.float32)
    assert whole.shape == (3, 2, 512)
    assert torch.equal(whole[2, 1], q.dequantize_iq2xxs(raw[2, 1], dtype=torch.float32))


def test_partial_blocks_are_refused():
    import torch

    q = _quant()
    with pytest.raises(ValueError):
        q.dequantize_q8_0(torch.zeros((2, 35), dtype=torch.uint8))
    with pytest.raises(TypeError):
        q.dequantize_iq2xxs(torch.zeros((2, 66), dtype=torch.int8))


def test_cuda_decodes_equal_cpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    q = _quant()
    gen = torch.Generator().manual_seed(6)
    raw = torch.randint(0, 256, (64, 4 * 66), dtype=torch.uint8, generator=gen)
    raw[:, 1::66] &= 0x3F                                    # keep every fp16 d finite
    raw8 = torch.randint(0, 256, (64, 4 * 34), dtype=torch.uint8, generator=gen)
    raw8[:, 1::34] &= 0x3F
    assert torch.equal(q.dequantize_iq2xxs(raw.cuda()).cpu(), q.dequantize_iq2xxs(raw))
    assert torch.equal(q.dequantize_q8_0(raw8.cuda()).cpu(), q.dequantize_q8_0(raw8))


def _tensor(name: str):
    from safetensors import safe_open

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(MODEL / index[name]), framework="pt") as f:
        return f.get_tensor(name)


@pytest.mark.skipif(not (MODEL / "model.safetensors.index.json").is_file(), reason="set TF_BOLT_MODEL to the checkpoint")
def test_real_tensors_decode_to_sane_weights():
    import torch

    q = _quant()
    p = "language_model.backbone.layers."
    checks = {
        "q8_0 lm_head": (q.dequantize_q8_0(_tensor("language_model.lm_head.weight")[:4096].flatten(-2)), (4096, 4096)),
        "q8_0 q_proj": (q.dequantize_q8_0(_tensor(p + "7.mixer.q_proj.weight").flatten(-2)), (4096, 4096)),
        "q8_0 fc1_latent": (q.dequantize_q8_0(_tensor(p + "1.mixer.fc1_latent_proj.weight").flatten(-2)), (1024, 4096)),
        "iq2 up": (q.dequantize_iq2xxs(_tensor(p + "1.mixer.experts.0.up_proj.weight").flatten(-2)), (2048, 1024)),
        "iq2 down": (q.dequantize_iq2xxs(_tensor(p + "1.mixer.experts.0.down_proj.weight").flatten(-2)), (1024, 2048)),
        "iq2 shared": (q.dequantize_iq2xxs(_tensor(p + "1.mixer.shared_experts.up_proj.weight").flatten(-2)),
                       (5376, 4096)),
        "nvfp4 in_proj": (q.dequantize_nvfp4(_tensor(p + "0.mixer.in_proj.weight"), _tensor(p + "0.mixer.in_proj.weight_scale"),
                                             _tensor(p + "0.mixer.in_proj.weight_scale_2")), (18560, 4096)),
    }
    for name, (w, shape) in checks.items():
        w = w.float()
        assert tuple(w.shape) == shape, name
        assert torch.isfinite(w).all(), name
        std = w.std().item()
        # trained projections of width 1k to 5k: per-weight spread 1e-3 .. 0.2 and no outliers past 64 std. Pruned
        # units decode to all-zero rows: 8 of expert 0's 2,048 and 347 of the shared expert's 5,376 (layer 1).
        assert 1e-3 < std < 0.2, (name, std)
        assert (w.abs().amax(1) == 0).float().mean().item() <= 0.1, name
        assert w.abs().max().item() < 64 * std, (name, w.abs().max().item(), std)
