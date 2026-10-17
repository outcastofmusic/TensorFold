"""Bolt's reference forward on a tiny checkpoint in the 24GB layout: a prompt fed whole or a token at a time agrees."""

import json

import pytest

pytestmark = pytest.mark.torch

HIDDEN, VOCAB, LATENT, WIDTH, SHARED = 256, 96, 256, 256, 256
HEADS, KV, HD = 4, 2, 16
MH, MP, MG, MS, CONV = 8, 8, 2, 16, 4
EXPERTS, TOP_K = 4, 2
KINDS = ("mamba", "moe", "attention", "moe", "mamba")


def _q8_0(n: int, k: int, gen):
    import torch

    q = torch.randint(-127, 128, (n, k // 32, 32), dtype=torch.int8, generator=gen).view(torch.uint8)
    d = torch.full((n, k // 32, 1), 0.002, dtype=torch.float16).view(torch.uint8).reshape(n, k // 32, 2)
    return torch.cat([d, q], -1)                                          # [n, k/32, 34] as the checkpoint stores it


def _iq2xxs(n: int, k: int, gen):
    import torch

    qs = torch.randint(0, 256, (n, k // 256, 64), dtype=torch.uint8, generator=gen)
    d = torch.full((n, k // 256, 1), 0.002, dtype=torch.float16).view(torch.uint8).reshape(n, k // 256, 2)
    return torch.cat([d, qs], -1)


def _nvfp4(n: int, k: int, gen):
    import torch

    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    s = torch.full((n, k // 16), 0.02, dtype=torch.float32).to(torch.float8_e4m3fn)
    return w, s, torch.tensor(0.5, dtype=torch.float32)


def _checkpoint(root):
    import torch
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(7)
    rnd = lambda *shape, s=0.05: (torch.randn(shape, generator=gen) * s).bfloat16()
    t, recipes = {}, {}
    q8 = {"quant_algo": "Q8_0", "group_size": 32, "block_payload_bytes": 34, "packing": "ggml"}
    iq2 = {"quant_algo": "IQ2_XXS", "group_size": 256, "block_payload_bytes": 66, "packing": "ggml"}
    p = "language_model."
    t[p + "backbone.embeddings.weight"] = rnd(VOCAB, HIDDEN, s=1.0)
    t[p + "backbone.norm_f.weight"] = (1 + rnd(HIDDEN, s=0.1).float()).bfloat16()
    t[p + "lm_head.weight"] = _q8_0(VOCAB, HIDDEN, gen)
    recipes[p + "lm_head"] = q8
    xd, conv_dim = MH * MP, MH * MP + 2 * MG * MS
    blocks = []
    for i, kind in enumerate(KINDS):
        b = f"{p}backbone.layers.{i}."
        t[b + "norm.weight"] = (1 + rnd(HIDDEN, s=0.1).float()).bfloat16()
        if kind == "mamba":
            blocks.append({"block_type": "mamba"})
            for name, n, k in (("in_proj", xd + conv_dim + MH, HIDDEN), ("out_proj", HIDDEN, xd)):
                w, s, s2 = _nvfp4(n, k, gen)
                t[b + f"mixer.{name}.weight"], t[b + f"mixer.{name}.weight_scale"] = w, s
                t[b + f"mixer.{name}.weight_scale_2"] = s2
                recipes[b + f"mixer.{name}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            t[b + "mixer.conv1d.weight"] = rnd(conv_dim, 1, CONV, s=0.3)
            t[b + "mixer.conv1d.bias"] = rnd(conv_dim, s=0.1)
            t[b + "mixer.A_log"] = rnd(MH, s=1.0)
            t[b + "mixer.D"] = rnd(MH, s=1.0)
            t[b + "mixer.dt_bias"] = rnd(MH, s=1.0)
            t[b + "mixer.norm.weight"] = (1 + rnd(xd, s=0.1).float()).bfloat16()
        elif kind == "attention":
            blocks.append({"block_type": "attention"})
            for name, n, k in (("q_proj", HEADS * HD, HIDDEN), ("k_proj", KV * HD, HIDDEN), ("v_proj", KV * HD, HIDDEN),
                               ("o_proj", HIDDEN, HEADS * HD)):
                t[b + f"mixer.{name}.weight"] = _q8_0(n, k, gen)
                recipes[b + f"mixer.{name}"] = q8
        else:
            blocks.append({"block_type": "moe", "moe_intermediate_size": WIDTH, "n_routed_experts": EXPERTS,
                           "num_experts_per_tok": TOP_K})
            t[b + "mixer.gate.weight"] = rnd(EXPERTS, HIDDEN, s=0.2)
            t[b + "mixer.gate.e_score_correction_bias"] = torch.randn(EXPERTS, generator=gen) * 0.1
            for name, n, k in (("fc1_latent_proj", LATENT, HIDDEN), ("fc2_latent_proj", HIDDEN, LATENT)):
                t[b + f"mixer.{name}.weight"] = _q8_0(n, k, gen)
                recipes[b + f"mixer.{name}"] = q8
            for name, n, k in [(f"experts.{e}.up_proj", WIDTH, LATENT) for e in range(EXPERTS)] + \
                    [(f"experts.{e}.down_proj", LATENT, WIDTH) for e in range(EXPERTS)] + \
                    [("shared_experts.up_proj", SHARED, HIDDEN), ("shared_experts.down_proj", HIDDEN, SHARED)]:
                t[b + f"mixer.{name}.weight"] = _iq2xxs(n, k, gen)
                recipes[b + f"mixer.{name}"] = iq2
    names = sorted(t)
    half = len(names) // 2
    files = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    for f, keys in files.items():
        save_file({k: t[k].contiguous() for k in keys}, str(root / f))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: f for f, keys in files.items() for k in keys}}))
    (root / "hf_quant_config.json").write_text(json.dumps(
        {"quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": recipes}}))
    (root / "config.json").write_text(json.dumps({"llm_config": {
        "hidden_size": HIDDEN, "vocab_size": VOCAB, "block_configs": blocks, "num_attention_heads": HEADS,
        "num_key_value_heads": KV, "head_dim": HD, "mamba_num_heads": MH, "mamba_head_dim": MP, "n_groups": MG,
        "ssm_state_size": MS, "conv_kernel": CONV, "moe_latent_size": LATENT,
        "moe_shared_expert_intermediate_size": SHARED, "n_shared_experts": 1, "routed_scaling_factor": 5.0,
        "norm_topk_prob": True, "layer_norm_epsilon": 1e-5, "eos_token_id": 2}}))
    return root


def test_whole_prompt_equals_token_by_token(tmp_path):
    import torch

    from tensorfold.families.nemotron_bolt import reference as ref

    ck = ref.Checkpoint(_checkpoint(tmp_path))
    assert [b.kind for b in ck.config.blocks] == list(KINDS)
    tokens = torch.tensor([5, 17, 3, 88, 41, 41, 9, 60])
    whole = ref.forward(ck, tokens)
    st = ref.State()
    steps = torch.cat([ref.forward(ck, tokens[i:i + 1], st) for i in range(len(tokens))])
    assert whole.shape == (8, VOCAB) and torch.isfinite(whole).all()
    assert st.length == 8 and st.keys[2].shape == (8, KV, HD)
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

    ck = ref.Checkpoint(_checkpoint(tmp_path))
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
