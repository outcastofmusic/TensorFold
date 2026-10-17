"""CSF-coded expert scales (``weight_scale_encoding: csf``) decode to the e4m3 bytes they encode, and load as plain ones."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.nvfp4 import csf_scales  # noqa: E402


def encode(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An independent encoder: each row's base is its smallest byte (capped at 240), bytes past base + 15 are exceptions."""

    b = scale.view(torch.uint8).to(torch.int64)
    rows, cols = b.shape
    base = b.min(dim=1, keepdim=True).values.clamp(max=240)
    code = b - base
    far = (code < 0) | (code > 15)
    nib = torch.where(far, torch.zeros_like(code), code)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    fixed = torch.cat([base.to(torch.uint8).reshape(rows // 16, 16), packed.reshape(rows // 16, 16 * cols // 2)], 1)
    pos = far.reshape(-1).nonzero().reshape(-1)
    words = pos | (b.reshape(-1)[pos] << 24)
    return fixed, ((words + 2**31) % 2**32 - 2**31).to(torch.int32)    # uint32 words as the reader hands them: int32


def scales(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    near = torch.randint(40, 56, (rows, cols), generator=g)          # most bytes sit within 15 of the row's least
    wild = torch.randint(0, 127, (rows, cols), generator=g)          # a few do not: exceptions
    pick = torch.rand((rows, cols), generator=g) < 0.02
    return torch.where(pick, wild, near).to(torch.uint8).view(torch.float8_e4m3fn)


def decode(pairs):
    return csf_scales(torch.stack([f for f, _ in pairs]), [x for _, x in pairs])


@pytest.mark.parametrize("rows,cols", [(640, 160), (2560, 40), (16, 2)])
def test_decode_returns_the_encoded_bytes(rows, cols):
    want = scales(rows, cols, rows + cols)
    fixed, exceptions = encode(want)
    assert fixed.shape == (rows // 16, 16 * (1 + cols // 2))
    got = decode([(fixed, exceptions)])
    assert got.dtype == torch.float8_e4m3fn and got.shape == (1, rows, cols)
    assert torch.equal(got[0].view(torch.uint8), want.view(torch.uint8))


def test_stacked_experts_keep_their_own_exceptions():
    want = [scales(64, 20, seed) for seed in range(5)]
    want[2] = torch.full((64, 20), 50, dtype=torch.uint8).view(torch.float8_e4m3fn)      # an expert with none
    pairs = [encode(w) for w in want]
    assert pairs[2][1].numel() == 0 and len({x.numel() for _, x in pairs}) > 2
    got = decode(pairs)
    assert got.shape == (5, 64, 20)
    for e, w in enumerate(want):
        assert torch.equal(got[e].view(torch.uint8), w.view(torch.uint8)), e


def test_bad_streams_are_refused():
    fixed, exceptions = encode(scales(16, 4, 1))
    with pytest.raises(ValueError, match="uint8"):
        decode([(fixed[:, :17], exceptions)])
    with pytest.raises(ValueError, match="exception lists"):
        csf_scales(fixed[None], [exceptions, exceptions])
    high = fixed.clone()
    high[0, 0] = 241
    with pytest.raises(ValueError, match="0..240"):
        decode([(high, exceptions)])
    with pytest.raises(ValueError, match="inside"):
        decode([(fixed, torch.tensor([16 * 4], dtype=torch.int32))])
