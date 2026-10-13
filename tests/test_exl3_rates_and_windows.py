"""The numpy EXL3 reference at the rates and codeword windows sushi's packs use, without a GPU.

The fixtures are sushi's own (github.com/beamivalice/sushi, src/exl3/fixtures, MIT, Copyright (c) 2026 Theinruj
Toranavikrai and David Dalcu): a 128 x 128 layer's trellis, the weights its converter decoded (``inner``) and the
layer after both rotations and scales (``public``). One was searched and decoded at a 12-bit window.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tensorfold.cuda.exl3 import format as fmt

FIXTURES = Path(__file__).parent / "fixtures" / "sushi_exl3"
DTYPES = {"U16": np.uint16, "F16": np.float16}


def _read(name: str) -> tuple[dict, dict[str, np.ndarray]]:
    raw = (FIXTURES / name).read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    meta = header.pop("__metadata__")
    data = raw[8 + size:]
    return meta, {k: np.frombuffer(data[v["data_offsets"][0]:v["data_offsets"][1]], DTYPES[v["dtype"]])
                  .reshape(v["shape"]) for k, v in header.items()}


@pytest.mark.parametrize("name", ["exl3_k2p5_mcg_linear", "exl3_k3_mcg_linear", "exl3_k2p5_mcg_w12_linear"])
def test_a_sushi_fixture_decodes_to_its_converters_weights(name):
    meta, t = _read(name + ".safetensors")
    bits, window = float(meta["k"]), int(meta["window"])
    inner = fmt.unpack(t["trellis"], bits, meta["codebook"], window=window)
    assert np.array_equal(inner.view(np.uint16), t["inner"].view(np.uint16))
    public = fmt.dequantize(t["trellis"], t["suh"], t["svh"], bits, meta["codebook"], window=window)
    want = t["public"].astype(np.float64)
    assert np.abs(public - want).max() <= 2e-3 * np.abs(want).max()     # theirs is fp32 arithmetic rounded to fp16


def test_a_window_is_part_of_the_format_and_not_a_preference():
    _, t = _read("exl3_k2p5_mcg_w12_linear.safetensors")
    wide = fmt.unpack(t["trellis"], 2.5, "mcg")
    assert not np.array_equal(wide.view(np.uint16), t["inner"].view(np.uint16))
    for bad in (7, 17):
        with pytest.raises(ValueError, match="window"):
            fmt.unpack(t["trellis"], 2.5, "mcg", window=bad)


def test_every_rate_ends_value_t_at_floor_of_t_plus_one_times_the_rate():
    for bits in fmt.BITS:                                    # the widths ExLlamaV3 writes keep their bit layout
        k2 = int(2 * bits)
        p1 = np.arange(1, 257)
        assert np.array_equal(fmt.stream_ends(bits), (p1 * k2 - (p1 % 2 if k2 % 2 else 0)) // 2)
    ends = fmt.stream_ends(2.25)                             # 36 halfwords a tile: 2, 4, 6, 9, ...
    assert ends[:5].tolist() == [2, 4, 6, 9, 11] and ends[-1] == 576 and fmt.tile_words(2.25) == 36
    for bad in (0.5, 2.3, 8.125, 2.0625):                    # below K1, not an eighth, above K8, an odd halfword count
        with pytest.raises(ValueError, match="EXL3 rate"):
            fmt.stream_ends(bad)


def test_a_quarter_bit_tile_round_trips_through_the_bitstream():
    rng = np.random.default_rng(0)
    ends = fmt.stream_ends(2.25)
    bits = rng.integers(0, 2, 576, dtype=np.uint64)          # the tile's ring of bits, most significant bit first
    words = np.array([int("".join(map(str, bits[i:i + 32])), 2) for i in range(0, 576, 32)], dtype=np.uint64)
    halves = np.stack([words & 0xFFFF, words >> 16], 1).reshape(-1).astype(np.uint16)
    got = fmt.states(halves[None], 2.25)[0]
    ring = np.concatenate([bits, bits])
    want = [int("".join(map(str, ring[576 + e - 16:576 + e])), 2) for e in ends]
    assert got.tolist() == want
    assert fmt.states(halves[None], 2.25, window=14)[0].tolist() == [w & 0x3FFF for w in want]
