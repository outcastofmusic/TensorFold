"""The routed-expert kernels at a rate between ExLlamaV3's widths and a narrowed codeword window (sushi's packs).

- The lane decode equals sushi's converter on its own fixtures, and the numpy reference on random bitstreams at
  2.25 bits (36 words a tile) and 2.5 bits, for windows of 16, 14 and 12 bits and every codebook.
- A 2.25-bit, 14-bit-window mcg layer: a row gives the same bits alone and in windows up to 128 rows, the prompt
  path gives the same bits, and the output agrees with a float64 reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tests.cuda.test_exl3_experts import _hadamard, _picks, _rot, _scale
from tests.test_exl3_rates_and_windows import _read

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _words(k: int, n: int, words: int, gen: torch.Generator) -> torch.Tensor:
    v = torch.randint(-32768, 32768, (k // 16, n // 16, words), dtype=torch.int32, generator=gen)
    return v.to(torch.int16).cuda().contiguous()


@pytest.mark.parametrize("name", ["exl3_k2p5_mcg_linear", "exl3_k3_mcg_linear", "exl3_k2p5_mcg_w12_linear"])
def test_the_lane_decode_equals_sushis_converter(name):
    from tensorfold.cuda.exl3 import experts

    meta, t = _read(name + ".safetensors")
    trellis = torch.from_numpy(t["trellis"].view(np.int16).copy()).cuda()
    got = experts.dequant(trellis, meta["codebook"], window=int(meta["window"]))
    assert np.array_equal(got.cpu().numpy().view(np.uint16), t["inner"].view(np.uint16))


@pytest.mark.parametrize("bits", (2.25, 2.5))
@pytest.mark.parametrize("window", (16, 14, 12))
@pytest.mark.parametrize("codebook", fmt.CODEBOOKS)
def test_the_lane_decode_equals_the_numpy_reference(bits, window, codebook):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(int(100 * bits) + window)
    t = _words(64, 96, int(16 * bits), g)
    got = experts.dequant(t, codebook, window=window).cpu().numpy()
    want = fmt.unpack(t.cpu().numpy(), bits, codebook, window=window)
    assert np.array_equal(got.view(np.uint16), want.view(np.uint16))


def test_a_rate_or_window_the_kernels_do_not_read_is_refused():
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="rate"):
        experts.dequant(_words(32, 32, 38, g), "mcg")          # 2.375 bits: no kernel instance
    with pytest.raises(ValueError, match="window"):
        experts.dequant(_words(32, 32, 36, g), "mcg", window=7)


def test_a_quarter_bit_layer_with_a_14_bit_window_keeps_rows_independent_and_matches_float64():
    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK, WORDS, WINDOW, ROWS = 24, 512, 256, 6, 36, 14, 128
    g = torch.Generator().manual_seed(21)

    def mats(k, n, in_scale, out_scale):
        return [(_words(k, n, WORDS, g), _scale(k, in_scale, g), _scale(n, out_scale, g)) for _ in range(E)]

    gate, up = mats(D, I, 1 / math.sqrt(D), 1.0), mats(D, I, 1 / math.sqrt(D), 1.0)
    down = mats(I, D, 1 / math.sqrt(I), 0.25)
    ex = experts.prepare(gate, up, down, "mcg", window=WINDOW)
    x = torch.randn((ROWS, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, ROWS, TOPK, g, shared=True)
    scratch = experts.Scratch(ex, ROWS, TOPK + 1)

    def run(rows, prompt=None):
        return experts.routed(x[rows], sel[rows].contiguous(), w[rows].contiguous(), ex, scratch, None, len(rows),
                              prompt=prompt).clone()

    full = run(list(range(ROWS)))
    assert torch.isfinite(full).all()
    for n in (1, 2, 3, 16, 17, 64, 128):
        for start in (0, ROWS - n):
            rows = list(range(start, start + n))
            assert torch.equal(run(rows), full[rows]), (n, start)
    assert torch.equal(run(list(range(ROWS)), prompt=True), run(list(range(ROWS)), prompt=False))

    H = _hadamard(x.device)

    def lin(v, m):
        t, suh, svh = m
        wq = torch.from_numpy(fmt.unpack(t.cpu().numpy(), WORDS / 16, "mcg", window=WINDOW).astype(np.float64)).cuda()
        return _rot(_rot(v * suh.double(), H) @ wq, H) * svh.double()

    ref = torch.zeros((4, D), dtype=torch.float64, device=x.device)
    for r in range(4):
        for j in range(TOPK):
            e = int(sel[r, j])
            gg, uu = lin(x[r].double(), gate[e]), lin(x[r].double(), up[e])
            ref[r] += float(w[r, j]) * lin(gg * torch.sigmoid(gg) * uu, down[e])
    err = (full[:4].double() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-2, err
    wide = experts.prepare(gate, up, down, "mcg")          # the same bits read at 16: another layer
    assert not torch.equal(experts.routed(x[:4], sel[:4].contiguous(), w[:4].contiguous(), wide, scratch, None, 4),
                           full[:4])
