"""A tiny random LiLiCorr draft checkpoint and a fake Bolt target that implements the drafter's side of the engine."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling, choose_rows

VOCAB = 300
HIDDEN = 64


def tiny_config(*, window: int = 6, block: int = 8, top_k: int = 4, sinks: bool = True, hidden: int = HIDDEN,
                vocab: int = VOCAB, inter: int = 96) -> dict:
    """config.json of a draft at the real checkpoint's layout, shrunk: 2 layers, 3 taps, a 6-position window."""

    return {
        "architectures": ["LiLiCorrDraftModel"], "block_size": block, "hidden_size": hidden, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": inter,
        "rms_norm_eps": 1e-6, "vocab_size": vocab, "rope_theta": 10000.0, "sliding_window": window,
        "dflash_config": {
            "mask_token_id": 7, "target_layer_ids": [0, 2, 3], "use_swa": True, "swa_window_size": window,
            "causal": False, "attention_sink_bias": sinks, "projector_type": "lilicorr", "lilicorr_enabled": True,
            "lilicorr_candidate_topk": top_k, "lilicorr_hidden_size": 32, "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 2, "lilicorr_mlp_ratio": 2.0, "lilicorr_factor_dim": 16,
            "lilicorr_vector_eps": 1e-4, "lilicorr_logit_scale": 8.0, "conv_kernel_size": 2, "conv_group_size": 16}}


def random_tensors(shapes: dict[str, tuple[int, ...]], seed: int = 0) -> dict[str, torch.Tensor]:
    """fp32 tensors at the given shapes: norms near 1, everything else small and random."""

    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, shape in shapes.items():
        t = torch.randn(shape, generator=g)
        if name.endswith(("norm.weight", "layernorm.weight")) and "feature_norm" not in name:
            t = 1.0 + 0.1 * t
        elif name.endswith("base_kernel"):
            t = 0.5 + 0.2 * t
        elif name.endswith("attention_sink_bias"):
            t = 0.5 * t
        else:
            t = t * (1.0 / max(shape[-1], 1) ** 0.5 if len(shape) >= 2 else 0.1)
        out[name] = t
    return out


def nvfp4_pack(n: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random ModelOpt NVFP4 bytes for an [n, k] weight: codes [n, k/2], e4m3 scales [n, k/16], an fp32 scale."""

    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    scale = (torch.rand((n, k // 16), generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    return codes, scale, torch.tensor(0.01, dtype=torch.float32)


def write_draft(path, *, seed: int = 0, **config):
    """A tiny draft checkpoint folder: config.json, and model.safetensors with fc and the MLPs packed as NVFP4."""

    import json

    from safetensors.torch import save_file

    from tensorfold.families.nemotron_bolt.cuda.lilicorr import DraftConfig, expected_shapes, fp4_kept

    path.mkdir(parents=True, exist_ok=True)
    raw = tiny_config(**config)
    (path / "config.json").write_text(json.dumps(raw))
    g = torch.Generator().manual_seed(seed + 1)
    tensors = {}
    for name, t in random_tensors(expected_shapes(DraftConfig.from_json(raw)), seed).items():
        if not fp4_kept(name):
            tensors[name] = t.to(torch.bfloat16)
            continue
        codes, scale, scale2 = nvfp4_pack(t.shape[0], t.shape[1], g)
        base = name.removesuffix(".weight")
        tensors.update({name: codes, base + ".weight_scale": scale, base + ".weight_scale_2": scale2})
    save_file(tensors, str(path / "model.safetensors"))
    return path


class FakeTarget:
    """A deterministic toy target: logits from the last two tokens, taps from (position, token).

    It implements what ``lilicorr_decode`` reads: ``pos``, ``eos``, ``set_sampling``, ``verify``, ``sample``,
    ``commit``, ``tap_rows``, and ``embed`` / ``lm_head`` for the drafter.
    """

    def __init__(self, prompt: Sequence[int], *, taps: int = 3, eos: tuple[int, ...] = (), seed: int = 3) -> None:
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn((VOCAB, VOCAB), generator=g) * 3.0
        self.table2 = torch.randn((VOCAB, VOCAB), generator=g) * 1.5
        self.embedding = torch.randn((VOCAB, HIDDEN), generator=g) * 0.5
        self.head = torch.randn((VOCAB, HIDDEN), generator=g) * 0.3
        self.tap_proj = torch.randn((VOCAB, taps * HIDDEN), generator=g)
        self.eos = eos
        self.ids: list[int] = []            # committed tokens (positions 0 .. pos - 1)
        self.window: list[int] = []         # the last verify's tokens
        self.verifies = 0
        self.prompt_logits = self._logits(list(prompt))[-1:]
        self.ids = list(prompt)
        self.window = list(prompt)

    @property
    def pos(self) -> int:
        return len(self.ids)

    def _logits(self, tokens: list[int]) -> torch.Tensor:
        hist = self.ids + tokens
        out = []
        for j in range(len(tokens)):
            t = len(self.ids) + j
            prev = hist[t - 1] if t > 0 else 0
            out.append(self.table[hist[t]] + self.table2[prev])
        return torch.stack(out)

    def set_sampling(self, sampling: Sampling | None) -> None:
        self.sampling = sampling

    def verify(self, tokens: list[int]) -> torch.Tensor:
        self.verifies += 1
        self.window = list(tokens)
        return self._logits(list(tokens))

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None) -> list[int]:
        if sampling is None or sampling.temperature <= 0:
            return [int(i) for i in logits.argmax(dim=-1)]
        ids = np.broadcast_to(np.arange(VOCAB, dtype=np.int64), logits.shape)
        return choose_rows(logits.double().numpy(), ids, positions, sampling)

    def commit(self, keep: int) -> None:
        self._kept = self.window[:keep]
        self.ids.extend(self._kept)

    def taps_of(self, start: int, tokens: list[int]) -> torch.Tensor:
        pos = torch.arange(start, start + len(tokens), dtype=torch.float32)[:, None]
        return torch.tanh(self.tap_proj[torch.tensor(tokens)] + 0.01 * pos)

    def tap_rows(self, n: int) -> torch.Tensor:
        return self.taps_of(self.pos - n, self.ids[self.pos - n:])

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[ids.cpu()]

    def lm_head(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() @ self.head.T


def serial(prompt: Sequence[int], count: int, sampling: Sampling | None, *, stop_eos: bool = False,
           eos: tuple[int, ...] = ()) -> list[int]:
    """The reference: one verify row a step through the same sampler."""

    t = FakeTarget(prompt, eos=eos)
    out = [t.sample(t.prompt_logits, [t.pos], sampling)[0]]
    while len(out) < count and not (stop_eos and out[-1] in eos):
        logits = t.verify([out[-1]])
        tok = t.sample(logits, [t.pos + 1], sampling)[0]
        t.commit(1)
        out.append(tok)
    return out


class FakeEngine(FakeTarget):
    """``FakeTarget`` with ``Engine``'s prefill side: prompt chunks, a state that exists only at the committed end (or
    at a chunk's cuts), ``state`` / ``restore``, and logits that also read the whole history, as a recurrent state."""

    max_rows = 16

    def __init__(self, *, prefill_rows: int = 64, eos: tuple[int, ...] = (), seed: int = 3) -> None:
        self.table3 = torch.randn((VOCAB, VOCAB), generator=torch.Generator().manual_seed(seed + 1))
        super().__init__([0], eos=eos, seed=seed)
        self.prefill_rows = prefill_rows
        self.sampling = None
        self.chunks: list[int] = []         # every prompt chunk's rows, in order
        self.reset()

    def _logits(self, tokens: list[int]) -> torch.Tensor:
        hist = self.ids + list(tokens)
        sums = np.cumsum(hist)[len(self.ids):] % VOCAB
        return super()._logits(tokens) + self.table3[torch.as_tensor(sums)]

    def reset(self) -> None:
        self.ids, self.window, self._pending = [], [], None

    def restore(self, st: dict) -> None:
        self.reset()
        self.ids = list(st["ids"])

    def prefill_chunk(self, tokens: Sequence[int], cuts: Sequence[int] = ()) -> dict[int, dict]:
        tokens = [int(t) for t in tokens]
        assert 1 <= len(tokens) <= self.prefill_rows and all(0 < c < len(tokens) for c in cuts)
        logits = self._logits(tokens)
        parts = {int(c): {"ids": self.ids + tokens[:c]} for c in cuts}
        self.ids = self.ids + tokens
        self.window = tokens
        self._pending = self.sample(logits[-1:], [self.pos], self.sampling)[0]
        self.chunks.append(len(tokens))
        return parts

    def prefill_token(self) -> int:
        return self._pending

    def state(self, n: int, part: dict | None = None) -> dict:
        if part is None and n != self.pos:
            raise ValueError(f"no state at {n}: the engine is at {self.pos}")
        return {"ids": list(part["ids"] if part is not None else self.ids), "pos": n}
