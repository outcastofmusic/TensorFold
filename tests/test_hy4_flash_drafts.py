"""Hy4-flash's drafted decoding on the CPU (the tiny checkpoint of ``hy4_flash_tiny``): drafted replies are serial
decoding's at every temperature, and DFly2 draws its drafts by the request's keyed sampler at the positions the
verify window samples them (playbook 5.2), so a sampled reply accepts what serial decoding would write."""

from __future__ import annotations

import numpy as np
import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.write(tmp_path_factory.mktemp("hy4") / "tiny")


@pytest.fixture(scope="module")
def weights(checkpoint):
    from tensorfold.families.hy4_flash.cuda.weights import load

    return load(checkpoint, "cpu")


PROMPT = list(range(10, 30))


def _engine(weights):
    from tensorfold.families.hy4_flash.cuda.model import Engine

    return Engine(weights, max_len=64, prefill_rows=64)


class _Fake:
    """A target whose logits at a row depend on the token, its position and the one before it only, so a verify
    window's rows are one-row steps' bits by construction (the CPU path's fp32 torch ops are not: their reductions
    follow the call's shape, which the GPU path's per-row kernels fix, 41676052). The engine's own ``sample``."""

    from tensorfold.families.hy4_flash.cuda.model import Engine

    sample = Engine.sample
    cuda, prefill_rows, src_row = False, 64, 0

    def __init__(self, eos: int = 1):
        import types

        self.c, self.ids, self.pos, self.sampling = types.SimpleNamespace(eos=(eos,)), [], 0, None

    def reset(self):
        self.ids, self.pos = [], 0

    def set_sampling(self, sampling):
        self.sampling = sampling

    def prefetch(self, *a):
        pass

    def forward(self, tokens, prompt=False, every=False):
        import torch

        rows = []
        for t in tokens:
            before = self.ids[-1] if self.ids else -1
            g = torch.Generator().manual_seed(hash((int(t), self.pos, before)) & 0x7FFFFFFF)
            rows.append(torch.randn(tiny.V, generator=g) * 3)
            self.ids.append(int(t))
            self.pos += 1
        out = torch.stack(rows)
        return out if every else out[-1:]

    def rewind(self, n):
        if n:
            self.pos -= n
            del self.ids[-n:]


def _serial(engine, sampling, count=30):
    from tensorfold.families.hy4_flash.cuda.model import prefill, serial_decode

    return serial_decode(engine, prefill(engine, PROMPT, sampling), count, sampling, stop_eos=False).tokens


class _Oracle:
    """Proposes serial decoding's own next tokens, one of them wrong every third round: full and partial accepts."""

    def __init__(self, serial: list[int]):
        self.serial, self.rounds = serial, 0

    def propose(self, e, last: int, sampling) -> list[int]:
        at = e.pos - len(PROMPT) + 1
        guess = list(self.serial[at:at + 5])
        self.rounds += 1
        if guess and self.rounds % 3 == 0:
            guess[len(guess) // 2] = (guess[len(guess) // 2] + 1) % tiny.V
        return guess

    def commit(self, e, accepted: int) -> None:
        pass


SAMPLINGS = [{"temperature": 0.0}, {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
             {"temperature": 0.9, "top_k": 0, "top_p": 1.0}]          # the last is run.sh's hy4 default


@pytest.mark.parametrize("kw", SAMPLINGS, ids=["greedy", "top_k", "top_k_off"])
def test_oracle_drafted_equals_serial(kw):
    """draft_decode samples each window row by the serial rule at its position, on the CPU as on CUDA."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill

    s = Sampling(seed=3, **kw)
    serial = _serial(_Fake(), s)
    e = _Fake()
    got = draft_decode(e, PROMPT, prefill(e, PROMPT, s), 30, s, stop_eos=False, drafter=_Oracle(serial),
                       allow_copy=False)
    assert got.tokens == serial
    assert got.stats["rounds"] < got.stats["accepted"] < got.stats["drafted"]


def test_dfly_drafted_equals_serial(weights, checkpoint):
    """DFly2's real rounds on the tiny checkpoint, greedy (the CPU path's window rows are not one-row steps' bits, so
    a sampled tie may fall the other way here; tests/cuda checks sampled rounds on the GPU path)."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill

    s = Sampling(seed=5, temperature=0.0)
    serial = _serial(_engine(weights), s)
    e = _engine(weights)
    drafter = DFly(checkpoint, e, device="cpu")
    got = draft_decode(e, PROMPT, prefill(e, PROMPT, s), 30, s, stop_eos=False, drafter=drafter)
    assert got.tokens == serial
    assert got.stats["drafted"] > 0


@pytest.mark.parametrize("kw", SAMPLINGS[1:], ids=["top_k", "top_k_off"])
def test_dfly_draws_each_draft_by_the_keyed_rule_at_its_position(weights, checkpoint, monkeypatch, kw):
    """The drafter's head gives the same logits at every step: draft i is the keyed draw at position pos + 1 + i over
    the head's ids, as serial decoding draws that position (argmax would repeat one token)."""

    import torch

    from tensorfold.engine.exact_sampling import Sampling, choose
    from tensorfold.families.hy4_flash.cuda import dfly
    from tensorfold.families.hy4_flash.cuda.model import prefill

    e = _engine(weights)
    drafter = dfly.DFly(checkpoint, e, device="cpu")
    s = Sampling(seed=11, **kw)
    last = prefill(e, PROMPT, s)                     # a round's first propose reads the prompt's last row
    ids = drafter.vocab if drafter.vocab is not None else torch.arange(weights.head.shape[0])
    flat = torch.linspace(4.0, 0.0, ids.numel())[torch.randperm(ids.numel(), generator=torch.Generator().manual_seed(0))]
    monkeypatch.setattr(drafter, "head", lambda h: flat[None, :].to(h.device))
    got = drafter.propose(e, last, s)
    want = [choose(flat.numpy().astype(np.float32), ids.numpy().astype(np.int64), e.pos + 1 + i, s) for i in range(5)]
    assert got == want and len(set(got)) > 1


# -- the draft head -------------------------------------------------------------------------------------------------

def test_nvfp4_quantizer_round_trips_within_its_grid():
    """bf16 [N, K] -> e2m1 codes (low nibble first), e4m3 scales a 16 inputs, one fp32 global scale; decoded with the
    loader's own NVFP4 decoder, each value is its block's nearest grid point."""

    import torch

    from tensorfold.families.hy4_flash.cuda.dfly import quantize_nvfp4
    from tensorfold.families.hy4_flash.cuda.weights import _nvfp4_dense

    w = (torch.randn(192, 256, generator=torch.Generator().manual_seed(0)) * 0.02).to(torch.bfloat16)
    codes, scales, scale2 = quantize_nvfp4(w, rows=64)
    assert codes.dtype == torch.uint8 and codes.shape == (192, 128)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (192, 16)
    back = _nvfp4_dense(codes, scales.view(torch.uint8), torch.tensor(scale2))
    err = ((back - w.float()) ** 2).sum() / (w.float() ** 2).sum()
    assert err < 0.015                     # NVFP4 with amax scales: about 1% of the weight's energy
    step = (scales.float() * scale2).repeat_interleave(16, -1)
    assert bool(((back - w.float()).abs() <= step * 1.0 + 1e-8).all())       # within one grid step (6 -> 4 is 2)


def test_dfly_reads_the_whole_vocabulary_through_an_nvfp4_head(weights, checkpoint, monkeypatch):
    """The default draft head covers every id; TF_HY4_DRAFT_HEAD=slice keeps the A/B arm (the first ids and the
    specials, bf16)."""

    import torch

    from tensorfold.families.hy4_flash.cuda.dfly import DFly

    e = _engine(weights)
    full = DFly(checkpoint, e, device="cpu")
    assert full.vocab is None
    h = (torch.randn(64, weights.head.shape[1], generator=torch.Generator().manual_seed(1))).to(torch.bfloat16)
    got, want = full.head(h).float(), h.float() @ weights.head.float().t()
    assert got.shape == want.shape
    assert float((got.argmax(-1) == want.argmax(-1)).float().mean()) > 0.9
    monkeypatch.setenv("TF_HY4_DRAFT_HEAD", "slice")
    monkeypatch.setenv("TF_HY4_DRAFT_VOCAB", "128")
    sliced = DFly(checkpoint, e, device="cpu")
    assert sliced.vocab.numel() == 128 + weights.head.shape[0] - weights.config.bos
    assert sliced.head(h).shape == (64, sliced.vocab.numel())


# -- adaptive depth -------------------------------------------------------------------------------------------------

def _confident(drafter, p: float):
    """The confidence head set to give every slot probability ``p`` (a zero weight and the bias logit(p))."""

    import math

    import torch

    drafter.conf_w = torch.zeros_like(drafter.conf_w)
    drafter.conf_b = torch.full_like(drafter.conf_b, math.log(p / (1 - p)))


@pytest.mark.parametrize("p, depth", [(0.6, 3), (0.95, 5), (0.05, 1)])
def test_adaptive_depth_keeps_the_drafts_whose_chance_of_landing_clears_the_floor(weights, checkpoint, monkeypatch,
                                                                                  p, depth):
    """TF_HY4_DFLY_ADAPTIVE=1: slot i's chance of landing is the product of the confidence head's first i + 1
    probabilities; a round keeps the slots at or above TF_HY4_DFLY_FLOOR, and always the first."""

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import prefill

    monkeypatch.setenv("TF_HY4_DFLY_ADAPTIVE", "1")
    monkeypatch.setenv("TF_HY4_DFLY_FLOOR", "0.2")          # 0.6: 0.6, 0.36, 0.216 stay, 0.13 goes
    e = _engine(weights)
    drafter = DFly(checkpoint, e, device="cpu")
    _confident(drafter, p)
    s = Sampling(seed=0, temperature=0.0)
    got = drafter.propose(e, prefill(e, PROMPT, s), s)
    assert len(got) == depth
    assert drafter.confidence == pytest.approx([p] * 5, abs=1e-6)


def test_adaptive_depth_is_off_by_default(weights, checkpoint):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import prefill

    e = _engine(weights)
    drafter = DFly(checkpoint, e, device="cpu")
    _confident(drafter, 0.05)
    s = Sampling(seed=0, temperature=0.0)
    assert len(drafter.propose(e, prefill(e, PROMPT, s), s)) == 5


def test_the_confidence_head_reads_each_slot_s_normed_row(weights, checkpoint, monkeypatch):
    """sigmoid(proj(final_norm(row))) a slot, from the checkpoint's own weights."""

    import torch

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda import dfly
    from tensorfold.families.hy4_flash.cuda.model import prefill

    monkeypatch.setenv("TF_HY4_DFLY_ADAPTIVE", "1")
    monkeypatch.setenv("TF_HY4_DFLY_FLOOR", "0")
    _, values = tiny.tensors()
    w = torch.from_numpy(values["model.dfly_layers.confidence_head.proj.weight"])
    b = torch.from_numpy(values["model.dfly_layers.confidence_head.proj.bias"])
    e = _engine(weights)
    drafter = dfly.DFly(checkpoint, e, device="cpu")
    seen = []
    norm = dfly.K.rmsnorm

    def keep(x, weight, eps, out=None):
        y = norm(x, weight, eps)
        if weight is drafter.final_norm:
            seen.append(y)
        return y

    monkeypatch.setattr(dfly.K, "rmsnorm", keep)
    s = Sampling(seed=0, temperature=0.0)
    drafter.propose(e, prefill(e, PROMPT, s), s)
    want = torch.sigmoid(seen[0][1:].float() @ w.t() + b)[:, 0]
    assert drafter.confidence == pytest.approx(want.tolist(), abs=1e-5)


def test_adaptive_drafted_equals_serial(weights, checkpoint, monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda.dfly import DFly
    from tensorfold.families.hy4_flash.cuda.model import draft_decode, prefill

    monkeypatch.setenv("TF_HY4_DFLY_ADAPTIVE", "1")
    s = Sampling(seed=5, temperature=0.0)
    serial = _serial(_engine(weights), s)
    e = _engine(weights)
    drafter = DFly(checkpoint, e, device="cpu")
    got = draft_decode(e, PROMPT, prefill(e, PROMPT, s), 30, s, stop_eos=False, drafter=drafter)
    assert got.tokens == serial
