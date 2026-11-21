"""Hy4-flash's CUDA engine code run on the CPU (fp32 torch ops in place of the kernels) against transformers' model.

The goldens (``fixtures/hy4_flash/tiny_goldens.npz``) are transformers PR #50903's ``HYV4FlashTextModel`` in bf16 on
the tiny checkpoint of ``hy4_flash_tiny`` (``tools/hy4_reference.py fixture``): its Engram hashes, and its logits at
every position of a 48-token sequence, which passes the tiny indexer's 32-key limit and the 16-key window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch

FIXTURES = Path(__file__).parent / "fixtures" / "hy4_flash"
GOLDENS = FIXTURES / "tiny_goldens.npz"


@pytest.fixture(scope="module")
def goldens():
    return np.load(GOLDENS)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny.write(tmp_path_factory.mktemp("hy4") / "tiny")


@pytest.fixture(scope="module")
def weights(checkpoint):
    from tensorfold.families.hy4_flash.cuda.weights import load

    return load(checkpoint, "cpu")


NEAR_TIE = 0.01        # a pick's 4th-to-5th score margin within bf16 noise: the engine may take the other expert


def _engine(weights, goldens=None, name: str = "long", **kw):
    """The engine on the CPU; with ``goldens``, it takes transformers' expert picks and indexer groups for ``name`` (a
    near tie, which bf16 rows can turn the other way, would otherwise send the two apart: ``test_own_picks_agree``
    and ``test_own_groups_agree`` check the engine's own)."""

    import torch

    from tensorfold.families.hy4_flash.cuda import model

    e = model.Engine(weights, max_len=kw.pop("max_len", 64), prefill_rows=kw.pop("prefill_rows", 64), **kw)
    if goldens is not None:
        e.replay = {int(k.rsplit("_", 1)[1]): torch.from_numpy(goldens[k]) for k in goldens.files
                    if k.startswith(f"{name}_picks_")}
        e.replay_groups = {int(k.rsplit("_", 1)[1]): torch.from_numpy(goldens[k]) for k in goldens.files
                           if k.startswith(f"{name}_groups_")}
    return e


def test_release_config_parses():
    from tensorfold.families.hy4_flash.config import SLIDING, SPARSE, Config, refusal

    raw = json.loads((FIXTURES / "release_config.json").read_text())
    assert refusal(raw) is None
    c = Config.parse(raw)
    assert c.layers == 42 and c.hidden == 3072 and c.vocab == 120832
    assert c.sparse_layers == (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41)
    assert c.kinds[2] == SLIDING and c.kinds[41] == SPARSE
    assert (c.sparse.heads, c.sliding.heads, c.sparse.theta, c.sliding.theta) == (32, 64, 1e7, 1e4)
    assert c.engram_layers == (1, 11, 21, 31)
    assert [sum(s) for s in c.engram_sizes] == [164994380, 164999008, 165004086, 165008942]
    assert (c.experts, c.top_k, c.latent, c.moe_width, c.scaling) == (512, 16, 1536, 1536, 3.998)
    assert (c.index_heads, c.index_dim, c.index_topk, c.index_pool, c.window) == (16, 128, 2048, 4, 512)
    assert c.eos == (120025,) and c.dense_layers == (0,)


@pytest.mark.parametrize("field,value", [("hc_mult", 2), ("kv_lora_rank", 256), ("norm_topk_prob", False),
                                         ("indexer_types", ["full", "shared"])])
def test_refuses_what_it_does_not_compute(field, value):
    from tensorfold.families.hy4_flash.config import refusal

    raw = json.loads((FIXTURES / "release_config.json").read_text())
    raw["text_config"][field] = value
    assert field in refusal(raw)


def test_engram_hashes_match_transformers(goldens):
    from tensorfold.families.hy4_flash.config import Config
    from tensorfold.families.hy4_flash.engram import Hasher, ngram_tokens

    c = Config.parse(tiny.config())
    stored, _ = tiny.tensors()
    ids = goldens["long_ids"]
    grams = ngram_tokens(ids, 0, np.zeros(0, np.int64), c.engram_ngram, c.pad)
    for j, layer in enumerate(c.engram_layers):
        h = Hasher(stored[f"model.layers.{layer}.engram_layer.lookup_table"][0], c.engram_sizes[j],
                   seed=c.engram_seed, layer_id=layer + 1, ngram=c.engram_ngram, heads=c.engram_heads, pad=c.pad)
        assert np.array_equal(h.hashes(grams), goldens[f"hashes_{layer}"])


def test_ngrams_carry_across_calls():
    from tensorfold.families.hy4_flash.engram import ngram_tokens

    ids = np.arange(10, 20)
    whole = ngram_tokens(ids, 0, np.zeros(0, np.int64), 3, 2)
    assert whole[0].tolist() == [10, 2, 2] and whole[1].tolist() == [11, 10, 2] and whole[5].tolist() == [15, 14, 13]
    parts = np.concatenate([ngram_tokens(ids[:4], 0, np.zeros(0, np.int64), 3, 2),
                            ngram_tokens(ids[4:], 4, ids[:4], 3, 2)])
    assert np.array_equal(parts, whole)


def _close(got, want, name: str) -> None:
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < 0.02, f"{name}: max error {err:.4f} of the logits' range"
    # greedy agrees wherever transformers' top two stand apart by more than the bf16 noise
    top2 = np.sort(want, -1)[..., -2:]
    clear = (top2[..., 1] - top2[..., 0]) > 0.05 * np.abs(want).max()
    assert np.array_equal(got.argmax(-1)[clear], want.argmax(-1)[clear]), name


def test_prompt_logits_match_transformers(weights, goldens):
    for name in ("short", "long"):
        e = _engine(weights, goldens, name)
        ids = goldens[f"{name}_ids"].tolist()
        logits = e.forward(ids, prompt=True, every=True)
        _close(logits.numpy(), goldens[f"{name}_logits"], name)


def test_chunks_and_decode_steps_match_one_pass(weights, goldens):
    """Prompt chunks cut across indexer groups and the window, then one-row steps: each row's logits as one pass."""

    e = _engine(weights, goldens, prefill_rows=7)
    ids = goldens["long_ids"].tolist()
    want = goldens["long_logits"]
    rows = []
    for s in range(0, 30, 7):
        rows.append(e.forward(ids[s:min(s + 7, 30)], prompt=True, every=True).numpy())
    for t in ids[30:]:
        rows.append(e.forward([t], prompt=False).numpy())
    _close(np.concatenate(rows), want, "chunks")


def test_kept_state_resumes(weights, goldens):
    import torch

    e = _engine(weights)
    ids = goldens["long_ids"].tolist()
    e.forward(ids[:40], prompt=True)
    st = e.state(40)
    a = e.forward(ids[40:], prompt=True)
    e.reset()
    e.restore(st)
    b = e.forward(ids[40:], prompt=True)
    assert torch.equal(a, b)


def test_own_picks_agree(weights, goldens):
    """The engine's own expert picks are transformers' wherever the 4th-to-5th margin stands above bf16 noise."""

    import torch

    e = _engine(weights, goldens)
    e.seen = {}
    ids = goldens["long_ids"].tolist()
    e.forward(ids, prompt=True)
    for layer, got in e.seen.items():
        mine = torch.cat(got).sort(-1).values.numpy()
        want = np.sort(goldens[f"long_picks_{layer}"], -1)
        clear = goldens[f"long_margin_{layer}"] > NEAR_TIE
        assert clear.mean() > 0.6            # the tiny router leaves a quarter of its picks within noise
        assert np.array_equal(mine[clear], want[clear]), f"layer {layer}"


def test_own_groups_agree(weights, goldens):
    """The indexer keeps transformers' groups wherever its last kept group stands clear of the first it drops."""

    import torch

    e = _engine(weights, goldens)
    e.replay_groups = None
    e.seen_groups = {}
    ids = goldens["long_ids"].tolist()
    e.forward(ids, prompt=True)
    checked = 0
    for layer, got in e.seen_groups.items():
        top, margin = (torch.cat(t) for t in zip(*got))
        want = goldens[f"long_groups_{layer}"]
        for r in range(len(ids)):
            if want[r, 0] >= 0 and float(margin[r]) > 0.05:
                assert top[r].tolist() == want[r].tolist(), f"layer {layer} row {r}"
                checked += 1
    assert checked >= 20
