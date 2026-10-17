"""Bolt's reference forward on the tiny checkpoint in the 24GB layout (tests/bolt_tiny.py): a prompt fed whole or a
token at a time agrees."""

import json

import pytest

pytestmark = pytest.mark.torch

from tests.bolt_tiny import PATTERN, write

VOCAB = 1024


def test_whole_prompt_equals_token_by_token(tmp_path):
    import torch

    from tensorfold.families.nemotron_bolt import reference as ref

    ck = ref.Checkpoint(write(tmp_path))
    assert [b.kind for b in ck.config.blocks] == list(PATTERN)
    tokens = torch.tensor([5, 17, 3, 88, 41, 41, 9, 60])
    whole = ref.forward(ck, tokens)
    st = ref.State()
    steps = torch.cat([ref.forward(ck, tokens[i:i + 1], st) for i in range(len(tokens))])
    assert whole.shape == (8, VOCAB) and torch.isfinite(whole).all()
    assert st.length == 8 and st.keys[2].shape == (8, 2, 64)
    assert (whole - steps).abs().max().item() < 1e-4 * whole.abs().max().item()
    # a split prompt resumes from the state as well: 5 rows, then 3
    st = ref.State()
    ref.forward(ck, tokens[:5], st)
    rest = ref.forward(ck, tokens[5:], st)
    assert (whole[5:] - rest).abs().max().item() < 1e-4 * whole.abs().max().item()
    # logits differ between positions: the layers really mix the context
    assert (whole[4] - whole[5]).abs().max().item() > 1e-3


def test_greedy_returns_the_rows_each_token_came_from(tmp_path):
    import torch

    from tensorfold.families.nemotron_bolt import reference as ref

    ck = ref.Checkpoint(write(tmp_path))
    prompt = torch.tensor([5, 17, 3])
    tokens, prompt_rows, rows = ref.greedy(ck, prompt, 4, last=2)
    assert tokens.shape == (4,) and prompt_rows.shape == (2, VOCAB) and rows.shape == (4, VOCAB)
    assert torch.equal(rows[0], prompt_rows[-1])
    assert torch.equal(rows.argmax(-1), tokens)
    whole = ref.forward(ck, torch.cat([prompt, tokens[:-1]]))
    assert (whole[-4:] - rows).abs().max().item() < 1e-4 * whole.abs().max().item()


def test_route_ranks_by_biased_score_and_weights_by_plain_score():
    import torch

    from tensorfold.families.nemotron_bolt.reference import route

    logits = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    bias = torch.tensor([0.0, 0.0, 2.0, 0.0])                # lifts expert 2 above experts 3 and 1
    idx, w = route(logits, bias, 2, 5.0, True)
    assert sorted(idx[0].tolist()) == [0, 2]
    s = torch.sigmoid(logits[0])
    want = {0: s[0] / (s[0] + s[2]) * 5.0, 2: s[2] / (s[0] + s[2]) * 5.0}
    for e, wt in zip(idx[0].tolist(), w[0].tolist()):
        assert wt == pytest.approx(want[e].item(), rel=1e-6)


def test_route_breaks_ties_toward_the_lower_expert():
    import torch

    from tensorfold.families.nemotron_bolt.cuda import reference as cuda_reference
    from tensorfold.families.nemotron_bolt.reference import route

    idx, _ = route(torch.zeros((1, 64)), torch.zeros(64), 9, 1.0, True)
    assert idx[0].tolist() == list(range(9))                     # as the engine's router kernel picks
    assert cuda_reference.route is route


def test_config_reads_the_bolt_layout(tmp_path):
    from tensorfold.families.nemotron_bolt.reference import Config

    root = tmp_path / "m"
    root.mkdir()
    raw = {"llm_config": {"hidden_size": 4096, "vocab_size": 131072, "num_attention_heads": 32,
                          "num_key_value_heads": 2, "head_dim": 128, "mamba_num_heads": 128, "mamba_head_dim": 64,
                          "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "moe_latent_size": 1024,
                          "moe_shared_expert_intermediate_size": 5376, "n_shared_experts": 1,
                          "routed_scaling_factor": 5.0, "norm_topk_prob": True, "layer_norm_epsilon": 1e-5,
                          "eos_token_id": 2, "block_configs": [{"block_type": "mamba"}, {
                              "block_type": "moe", "moe_intermediate_size": 2048, "n_routed_experts": 240,
                              "num_experts_per_tok": 9}]}}
    (root / "config.json").write_text(json.dumps(raw))
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 11]}))
    c = Config.from_dir(root)
    assert c.conv_dim == 128 * 64 + 2 * 8 * 128 == 10240
    assert c.xd + c.conv_dim + c.m_heads == 18560                          # in_proj's rows in the checkpoint
    assert c.blocks[1].experts == 240 and c.blocks[1].top_k == 9 and c.eos == (2, 11)
    assert (c.dt_floor, c.dt_ceil) == (0.0, float("inf"))                 # no time_step_limit: unclamped
    raw["llm_config"]["time_step_limit"] = [0.001, {"__float__": "Infinity"}]
    (root / "config.json").write_text(json.dumps(raw))
    c = Config.from_dir(root)
    assert (c.dt_floor, c.dt_ceil) == (0.001, float("inf"))
