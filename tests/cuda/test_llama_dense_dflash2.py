"""DFlash2 on dense Llama: the drafter reads the released Nanbeige config's shape (3 taps, ``sliding_window: null``, 7
query heads a KV head, 32 candidates), prefill hands it the rows its window keeps, and drafted replies equal serial
ones token for token."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.geometry import DRAFT_CONTEXT
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.llama_dense.cuda.decode import draft_decode, prefill, serial_decode
from tensorfold.families.llama_dense.cuda.engine import _drafter
from tensorfold.families.llama_dense.cuda.weights import load
from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2
from tests import llama_tiny

FIXTURES = Path(__file__).parents[1] / "fixtures" / "llama_dense"
LAYERS = 3
HEADS, KV, INTER = 35, 5, 4096          # 7 query heads a KV head as Nanbeige's 28 / 4; big enough to pack to 4 bits
H, V = llama_tiny.HIDDEN, llama_tiny.VOCAB
SAMPLINGS = [None, Sampling(77, 0.7, 20, 0.95)]


def write_drafter(folder: Path, taps=(0, 1, 2), fc_taps: int | None = None, seed: int = 5, **change) -> Path:
    """A random drafter under the released config.json, its sizes cut to the tiny target and everything else kept."""

    from safetensors.torch import save_file

    cfg = json.loads((FIXTURES / "dflash2_config.json").read_text())
    cfg |= {"hidden_size": H, "intermediate_size": INTER, "num_attention_heads": HEADS, "num_key_value_heads": KV,
            "num_hidden_layers": 2, "layer_types": ["full_attention"] * 2, "vocab_size": V,
            "num_target_layers": LAYERS, **change}
    cfg["dflash_config"] = {**cfg["dflash_config"], "target_layer_ids": list(taps)}
    rank = cfg["dflash_config"]["selector_rank"]
    g = torch.Generator().manual_seed(seed)

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=g) * s).bfloat16()

    def ones(n):
        return torch.ones(n).bfloat16()

    t = {"candidate_selector.hidden_projection.weight": r(rank, H),
         "candidate_selector.predecessor_codebook": r(V, rank, s=0.1),
         "candidate_selector.successor_codebook": r(V, rank, s=0.1),
         "fc.weight": r(H, (fc_taps or len(taps)) * H), "hidden_norm.weight": ones(H), "norm.weight": ones(H)}
    for i in range(2):
        p = f"layers.{i}."
        t |= {p + "self_attn.q_proj.weight": r(HEADS * 128, H), p + "self_attn.k_proj.weight": r(KV * 128, H),
              p + "self_attn.v_proj.weight": r(KV * 128, H), p + "self_attn.o_proj.weight": r(H, HEADS * 128),
              p + "self_attn.q_norm.weight": ones(128), p + "self_attn.k_norm.weight": ones(128),
              p + "input_layernorm.weight": ones(H), p + "post_attention_layernorm.weight": ones(H),
              p + "mlp.gate_proj.weight": r(INTER, H), p + "mlp.up_proj.weight": r(INTER, H),
              p + "mlp.down_proj.weight": r(H, INTER)}
        for conv in ("attention_conv", "mlp_conv"):
            t[p + conv + ".base_kernel"] = r(2, 2, H, s=0.5)
            t[p + conv + ".kernel_projection.weight"] = r(4 * H // 16, H)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(t, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(json.dumps(cfg))
    return folder


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    folder = llama_tiny.write(tmp_path_factory.mktemp("target"), layers=LAYERS)
    return folder, load(folder)


@pytest.fixture(scope="module")
def drafter_dir(tmp_path_factory):
    return write_drafter(tmp_path_factory.mktemp("drafter"))


def _empty(d):
    d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
    return d


def _prompt(n: int, seed: int = 3) -> list[int]:
    return torch.randint(4, V, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_the_released_config_shape_loads_and_drafts(target, tmp_path):
    _, w = target
    d = DFlash2(write_drafter(tmp_path / "real", taps=(1, 25, 49)), w, vocab_spans=((0, V),))
    assert d.tap_layers == (1, 25, 49) and d.window == DRAFT_CONTEXT - 1
    assert (d.top_k, d.mask_id, d.is_causal, d.trained) == (32, 0, False, 8)
    assert d.sub_parts is not None and d.head_ids.tolist() == list(range(V))   # the bf16 head as stored, whole
    _empty(d).add_taps(torch.randn(40, 3 * H, device="cuda").bfloat16())
    nodes, parents = d.propose_tree(7, 41, 15)
    assert 1 <= len(nodes) <= 15 and all(0 <= t < V for t in nodes) and d.last_candidates.shape[1] == 32
    assert all(-1 <= p < i for i, p in enumerate(parents))
    with pytest.raises(ValueError, match="3 target layer taps"):
        d.add_taps(torch.zeros(2, 5 * H, device="cuda").bfloat16())
    with pytest.raises(ValueError, match="layers"):          # a 3-layer target has no layer 25
        _drafter(tmp_path / "real", w)
    with pytest.raises(ValueError, match="fc.weight"):
        DFlash2(write_drafter(tmp_path / "bad", taps=(0, 1), fc_taps=3), w, vocab_spans=((0, V),))


def test_prefill_hands_the_drafter_its_window_at_each_kept_point(target, tmp_path):
    """Kept points' snapshots and the end's context equal a drafter fed each prefix alone (rows outside a 64-row
    window are skipped, so only their count reaches it)."""

    _, w = target
    d = _drafter(write_drafter(tmp_path / "w64", sliding_window=64), w)
    prompt = _prompt(500)
    keep = [30, 200, 211, 499]
    _empty(d)
    _, _, kept = prefill(w, prompt, None, keep=keep, draft=d)
    end = d.snapshot()
    for p, (st, snap) in zip(keep + [len(prompt)], kept + [(None, end)]):
        alone = _drafter(tmp_path / "w64", w)
        _empty(alone)
        prefill(w, prompt[:p], None, draft=alone)
        want = alone.snapshot()
        assert snap[2:] == want[2:] == (min(p, 63), p)
        for a, b in zip(snap[0] + snap[1], want[0] + want[1]):
            torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)     # the drafter's rounding moves drafts only
        assert st is None or st.pos == p


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
@pytest.mark.parametrize("copies", [True, False], ids=["copies", "trees-only"])
def test_dflash2_replies_equal_serial(target, drafter_dir, sampling, copies):
    _, w = target
    d = _empty(_drafter(drafter_dir, w))
    prompt = _prompt(300)
    st, pending, _ = prefill(w, prompt, sampling, draft=d)
    want = serial_decode(w, st, pending, 200, sampling, stop_eos=False).tokens
    trace: list[dict] = []
    got = draft_decode(w, st, prompt, pending, 200, sampling, draft=d, allow_copy=copies, stop_eos=False, trace=trace)
    assert got.tokens == want
    assert sum(t["source"] == "tree" for t in trace) > 0 and max(t["rows"] for t in trace) > 1
    assert d.context_end == len(prompt) + len(want) - 1          # every committed row's taps reached the drafter


class Oracle:
    """A drafter stand-in whose trees hold the serial reply under a wrong sibling at every depth."""

    tap_layers = (0, 1, 2)

    def __init__(self, reply: list[int], start: int):
        self.reply, self.start, self.rows = reply, start, 0

    def propose_tree(self, pending, context_length, max_nodes, sampling=None):
        at = context_length - self.start             # the reply index the next drafted token holds
        nodes, parents, right = [], [], -1
        for t in self.reply[at:at + max_nodes // 2]:
            nodes += [(t + 1) % V, t]
            parents += [right, right]
            right = len(nodes) - 1
        return nodes, parents

    def add_taps(self, taps):
        assert taps.shape[1] == 3 * H
        self.rows += taps.shape[0]


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_accepted_tree_branches_commit_the_serial_tokens(target, sampling):
    """Trees whose right path sits behind wrong siblings: accepted depths commit, the taps follow the path."""

    _, w = target
    prompt = _prompt(120, seed=8)
    st, pending, _ = prefill(w, prompt, sampling)
    want = serial_decode(w, st, pending, 120, sampling, stop_eos=False).tokens
    oracle = Oracle(want, len(prompt))
    trace: list[dict] = []
    got = draft_decode(w, st, prompt, pending, 120, sampling, draft=oracle, allow_copy=False, stop_eos=False,
                       trace=trace)
    assert got.tokens == want and got.accepted_drafts > 60
    assert oracle.rows == len(want) - 1
    assert max(t["accepted"] for t in trace) == 7                # 15 rows: 7 depths of (wrong, right) pairs


@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "sampled"])
def test_engine_with_a_drafter_replies_as_serial_with_prefix_reuse(target, drafter_dir, sampling):
    from tensorfold.families.llama_dense import cuda_engine

    folder, _ = target
    eng = cuda_engine(folder, drafter=str(drafter_dir), context=4096, context_explicit=True)

    def run(prompt, draft=True):
        out: list[int] = []
        stats = eng.generate(prompt, 64, sampling, lambda tokens: out.extend(tokens), draft=draft, stop_eos=False)
        return out, stats

    first = _prompt(700, seed=11)
    reply, _ = run(first)
    assert reply == run(first, draft=False)[0]
    assert all(e[2] is not None for e in eng.cache.entries)
    second = first + reply + _prompt(30, seed=12)
    again, stats = run(second)
    assert stats["cached"] == len(first) - 1 and stats["drafted"] > 0
    assert again == run(second, draft=False)[0]
