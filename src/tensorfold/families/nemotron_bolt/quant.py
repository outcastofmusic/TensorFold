"""Bolt's three weight codecs as pure torch: ggml Q8_0 and IQ2_XXS blocks and ModelOpt NVFP4, decoded to bf16.

Each decoder computes in fp32 exactly as its reference does, then rounds once to the output dtype (bf16 by default;
dtype=torch.float32 keeps the exact values, which the reference forward uses). Packed ggml weights arrive flat:
[..., K // 32 * 34] for Q8_0 and [..., K // 256 * 66] for IQ2_XXS. The checkpoint stores them as [N, K // 32, 34]
and [N, K // 256, 66], so a caller passes w.flatten(-2).
"""

from __future__ import annotations

import torch

from tensorfold.families.qwen4_exp.cuda import nvfp4

Q8_0_BLOCK = 32            # weights a Q8_0 block
Q8_0_BYTES = 34            # fp16 d, then 32 int8 quants (ggml-common.h:251-255, block_q8_0)
IQ2XXS_BLOCK = 256         # weights an IQ2_XXS block (QK_K)
IQ2XXS_BYTES = 66          # fp16 d, then uint16 qs[32] (ggml-common.h:381-385, block_iq2_xxs)

# iq2xxs_grid, copied from llama.cpp ggml/src/ggml-common.h:560-625 (commit 23f7aabc). Each entry packs 8 magnitudes,
# byte j (little-endian) the j-th weight, every byte one of 0x08, 0x19, 0x2b.
IQ2XXS_GRID = (
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
)

# ksigns_iq2xs, copied from ggml-common.h:513-522: a 7-bit index -> 8 sign bits, bit 7 set when the index's parity is odd.
KSIGNS_IQ2XS = (
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
)

# sha256 of IQ2XXS_GRID as 256 little-endian uint64 and of KSIGNS_IQ2XS as 128 bytes, from ggml-common.h itself
IQ2XXS_GRID_SHA256 = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
KSIGNS_IQ2XS_SHA256 = "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31"


def _blocks(packed: torch.Tensor, nbytes: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed ggml weights are uint8, got {packed.dtype}")
    if packed.shape[-1] % nbytes:
        raise ValueError(f"last dim {packed.shape[-1]} is not a whole number of {nbytes}-byte blocks")
    return packed.reshape(*packed.shape[:-1], packed.shape[-1] // nbytes, nbytes)


def _fp16(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Two uint8 tensors (little-endian halves) -> their fp16 value widened to fp32."""

    bits = lo.to(torch.int32) | (hi.to(torch.int32) << 8)
    return (bits - ((bits >> 15) << 16)).to(torch.int16).view(torch.float16).float()


def dequantize_q8_0(packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """uint8 [..., K // 32 * 34] -> [..., K]: y = qs[j] * d in fp32 (ggml-quants.c:553-567, dequantize_row_q8_0)."""

    b = _blocks(packed, Q8_0_BYTES)
    d = _fp16(b[..., 0], b[..., 1])[..., None]
    q = b[..., 2:].view(torch.int8).float()
    return (q * d).reshape(*packed.shape[:-1], -1).to(dtype)


_tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _iq2xxs_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The grid as [256, 8] fp32 magnitudes and the signs as [128, 8] fp32 of +-1, on device."""

    if device not in _tables:
        grid = torch.tensor([[(v >> (8 * j)) & 0xFF for j in range(8)] for v in IQ2XXS_GRID], dtype=torch.float32)
        signs = torch.tensor([[-1.0 if s & (1 << j) else 1.0 for j in range(8)] for s in KSIGNS_IQ2XS])
        _tables[device] = (grid.to(device), signs.to(device))
    return _tables[device]


def dequantize_iq2xxs(packed: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """uint8 [..., K // 256 * 66] -> [..., K], following ggml-quants.c:2489-2513 (dequantize_row_iq2_xxs).

    A block's 64 qs bytes are 8 runs of 32 weights. A run's first 4 bytes pick 4 grid rows of 8 magnitudes; its last
    4, a little-endian uint32, hold four 7-bit sign indices (bits 0-27) and a 4-bit scale s (bits 28-31), and the run
    is d * (0.5 + s) * 0.25 * grid * sign. Every product is exact in fp32: d has 11 significant bits, (0.5 + s) 5,
    the grid 6.
    """

    b = _blocks(packed, IQ2XXS_BYTES)
    grid, signs = _iq2xxs_tables(packed.device)
    d = _fp16(b[..., 0], b[..., 1])                                     # [..., nb]
    runs = b[..., 2:].reshape(*b.shape[:-1], 8, 8).to(torch.int64)      # [..., nb, 8 runs, 8 bytes]
    aux = runs[..., 4] | (runs[..., 5] << 8) | (runs[..., 6] << 16) | (runs[..., 7] << 24)
    db = d[..., None] * (0.5 + (aux >> 28).float()) * 0.25             # [..., nb, 8]
    shifts = torch.tensor([0, 7, 14, 21], device=packed.device)
    sign = signs[(aux[..., None] >> shifts) & 127]                     # [..., nb, 8, 4, 8]
    mag = grid[runs[..., :4]]                                          # [..., nb, 8, 4, 8]
    y = db[..., None, None] * mag * sign
    return y.reshape(*packed.shape[:-1], -1).to(dtype)


def dequantize_nvfp4(weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor,
                     dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """ModelOpt W4A16 NVFP4: uint8 [N, K // 2] codes (low nibble the even input), e4m3 [N, K // 16] scales and the
    fp32 tensor scale -> [N, K] = e2m1 * (fp32(scale) * weight_scale_2), as vLLM's emulation decodes it
    (vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:546-629, unswizzled scales)."""

    return nvfp4.dequantize(weight, weight_scale, weight_scale_2).to(dtype)
