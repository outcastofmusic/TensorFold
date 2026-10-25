"""Idle prompt pieces use spare budget without changing the admitted window or delaying live replies."""

import copy
import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda.geometry import indexed_prompt_bytes
from tensorfold.families.qwen4_exp.cuda.prompt_plan import choose, pass_limit, storage
from tests.test_cuda_capacity import small_config
from tests.test_cuda_geometry import allocations, bytes_in  # noqa: F401

FLASH = {"hidden_size": 2560, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
         "hc_count": 4, "hc_lowrank": 320, "shared_expert_intermediate_size": 640}
# FP4-Local's text config, the scalars the prompt buffers and the plan read
FP4_LOCAL = {**FLASH, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
             "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
             "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "indexer_n_heads": 4,
             "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4, "ple_embed_dim": 2560,
             "heads_per_ngram": 8, "vocab_size": 248320}


def modelopt(dense: str | None = "MXFP8", experts: str = "NVFP4", prefix: str = "model.language_model.") -> dict:
    """FP4-Local's quantization block in miniature: NVFP4 experts, ``dense`` linears (None: bf16, unlisted), an
    NVFP4 n-gram table, and the vision tower's and MTP head's own formats."""

    layers = {f"{prefix}layers.{i}.mlp.experts": {"quant_algo": experts, "group_size": 16} for i in range(4)}
    layers[f"{prefix}layers.1.ple.ple_embedding.ngram_embedding"] = {"quant_algo": "NVFP4"}
    if dense is not None:
        for name in ("linear_attn.in_proj_qkv", "linear_attn.out_proj", "mlp.shared_expert.gate_proj"):
            layers[f"{prefix}layers.0.{name}"] = {"quant_algo": dense}
        layers[f"{prefix}layers.3.self_attn.q_proj"] = {"quant_algo": dense}
    layers["model.visual.blocks.0.mlp.linear_fc1"] = {"quant_algo": "W4A16_NVFP4"}
    layers["model.visual.blocks.0.attn.qkv"] = {"quant_algo": "MXFP8"}
    layers["mtp.layers.0.mlp.experts"] = {"quant_algo": "W4A16_NVFP4"}
    return {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers}


MLX = {"bits": 4, "group_size": 32, "mode": "affine"}
LONG = 16384        # a TENSORFOLD_LONG_PIECE_TOKENS threshold: NVFP4 checkpoints plan wide pieces only with one


def text(quant=None):
    return {**small_config(), **FLASH, "_quantization": MLX if quant is None else quant}


def temporary(t, mtp):
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import nvfp4_temporary_row

    return nvfp4_temporary_row(t, mtp) if storage(t["_quantization"]) == "nvfp4" else 0


def receipt(t, spare, *, loading=0):
    fixed = indexed_prompt_bytes(t, 2048)
    total = max(fixed, loading) + 1000
    return {"context_window": 65536, "cache_slots": 65543, "largest_window": 131072,
            "cache_workspace_bytes_estimate": fixed, "serving_peak_bytes_estimate": fixed + 1000,
            "startup_peak_bytes_estimate": loading + 1000, "total_bytes_estimate": total,
            "mapped_table_bytes": 500, "full_mapped_working_set_bytes_estimate": total + 500,
            "budget_bytes": total + spare}


KINDS = [pytest.param(MLX, False, id="mlx"),
         pytest.param(modelopt(), False, id="nvfp4", marks=pytest.mark.torch),
         pytest.param(modelopt(), True, id="nvfp4-mtp", marks=pytest.mark.torch)]


@pytest.mark.parametrize("quant,mtp", KINDS)
@pytest.mark.parametrize("loading", [0, 1 << 40])
def test_larger_pieces_count_every_extra_byte_without_reducing_the_window(quant, mtp, loading):
    t = text(quant)
    extra = indexed_prompt_bytes(t, 2048) + 2048 * temporary(t, mtp)
    plan = receipt(t, extra, loading=loading)
    old = copy.deepcopy(plan)
    rows, workspace = choose(plan, t, (12, 1), mtp=mtp, long_from=LONG)
    assert rows == 4096 and workspace == indexed_prompt_bytes(t, rows) + rows * temporary(t, mtp)
    assert plan["prefill_rows"] == rows and plan["prompt_workspace_bytes_estimate"] == workspace
    for key in ("context_window", "cache_slots", "largest_window", "budget_bytes"):
        assert plan[key] == old[key]
    assert plan["cache_workspace_bytes_estimate"] == old["cache_workspace_bytes_estimate"] + extra
    assert plan["serving_peak_bytes_estimate"] == old["serving_peak_bytes_estimate"] + extra
    peak = max(plan["serving_peak_bytes_estimate"], old["startup_peak_bytes_estimate"])
    assert plan["total_bytes_estimate"] == peak <= plan["budget_bytes"]
    assert plan["full_mapped_working_set_bytes_estimate"] == peak + plan["mapped_table_bytes"]


@pytest.mark.parametrize("quant,mtp", KINDS)
def test_insufficient_headroom_keeps_the_original_window_and_estimates(quant, mtp):
    t = text(quant)
    plan = receipt(t, indexed_prompt_bytes(t, 2048) + 2048 * temporary(t, mtp) - 1)
    old = copy.deepcopy(plan)
    assert choose(plan, t, (12, 1), mtp=mtp, long_from=LONG) == (2048, 0)
    assert plan == old


@pytest.mark.torch
def test_nvfp4_pieces_also_need_room_for_the_bf16_partials():
    """The headroom the MLX checkpoint's wider pieces take is not enough for the NVFP4 checkpoint's."""

    mlx, nvfp4 = text(), text(modelopt())
    spare = indexed_prompt_bytes(mlx, 2048)
    assert choose(receipt(mlx, spare), mlx, (12, 1), long_from=LONG) == (4096, indexed_prompt_bytes(mlx, 4096))
    assert choose(receipt(nvfp4, spare), nvfp4, (12, 1), long_from=LONG) == (2048, 0)


@pytest.mark.torch
def test_the_nvfp4_temporary_is_the_mtp_heads_fc_hidden_partials_or_the_hyper_connection_downs():
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import nvfp4_temporary_row

    # fc_hidden: 4 K slices of [4 streams x rows, 2,560] fp32; the down and inject: 32 slices and the sums of 324
    assert nvfp4_temporary_row(FP4_LOCAL, True) == 4 * 4 * 2560 * 4 == 163840
    assert nvfp4_temporary_row(FP4_LOCAL, False) == 33 * 324 * 4 == 42768


@pytest.mark.parametrize("quant", [MLX, pytest.param(modelopt(), marks=pytest.mark.torch)], ids=["mlx", "nvfp4"])
def test_opt_in_fp8_keeps_the_released_piece_size(quant):
    plan = {"untouched": True}
    assert choose(plan, text(quant), (12, 1), fp8=True, long_from=LONG) == (2048, 0)
    assert plan == {"untouched": True}


@pytest.mark.parametrize("quant,kind", [
    (MLX, "mlx"), (modelopt(), "nvfp4"), (modelopt(prefix="model."), "nvfp4"),
    ({"bits": 4, "group_size": 64}, None), ({}, None),
    (modelopt(dense="FP8_PB_WO"), None),             # block FP8 linears run the lane matmul, with its own partials
    (modelopt(dense=None), None),                    # bf16 linears run the bf16 matmul's K slices
    (modelopt(experts="W4A16_NVFP4"), None),
    ({**modelopt(), "quant_method": "compressed-tensors"}, None),
])
def test_storage_names_the_checkpoints_the_wide_pieces_serve(quant, kind):
    assert storage(quant) == kind


@pytest.mark.parametrize("quant", [MLX, modelopt()], ids=["mlx", "nvfp4"])
@pytest.mark.parametrize("change,capability,world,vision", [
    ({}, (9, 0), 1, False), ({}, (12, 0), 1, False), ({}, (12, 1), 2, False),
    ({}, (12, 1), 1, True), ({"hidden_size": 512}, (12, 1), 1, False),
    ({"num_experts": 128}, (12, 1), 1, False),
    ({"_quantization": {"bits": 4, "group_size": 64}}, (12, 1), 1, False),
    ({"_quantization": {}}, (12, 1), 1, False),
    ({"_quantization": modelopt(dense="FP8_PB_WO")}, (12, 1), 1, False),
])
def test_unqualified_models_and_devices_keep_released_pieces(quant, change, capability, world, vision):
    t = {**text(quant), **change}
    plan = {"untouched": True}
    assert choose(plan, t, capability, world=world, vision=vision, long_from=LONG) == (2048, 0)
    assert plan == {"untouched": True}


@pytest.mark.torch
@pytest.mark.parametrize("mtp", [False, True])
def test_the_extra_nvfp4_rows_fit_the_bytes_the_plan_adds(monkeypatch, allocations, mtp):  # noqa: F811
    """At FP4-Local's shapes, 2,048 more prompt rows allocate fewer bytes than ``choose`` adds: the buffers'
    growth, and the widest temporary's (its row count doubles with the piece)."""

    arrays, fake = allocations
    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    for mod in (state, state.attn_mod, state.moe_mod, state.gdn_mod, importlib.import_module("tensorfold.cuda.experts")):
        monkeypatch.setattr(mod, "torch", fake)
    t = {**FP4_LOCAL, "_quantization": modelopt()}
    cfg = SimpleNamespace(hidden=2560, streams=4, low=320, heads=24, head_dim=256, kv_heads=2, index_heads=4,
                          index_dim=128, index_budget=2048, index_ratio=4, top_k=10, experts=512, moe_width=640,
                          nk=16, nv=48, conv_kernel=4, heads_per_ngram=8, ple_dim=2560)
    weights = SimpleNamespace(cfg=cfg, device="cpu", layers=[], meta={"world": 1}, head=SimpleNamespace(n=t["vocab_size"]))
    sizes = {}
    for rows in (2048, 4096):
        arrays.clear()
        state.Buffers(weights, rows, 262144, prefill=True)
        sizes[rows] = bytes_in(arrays)
    plan = receipt(t, 1 << 40)
    old = plan["serving_peak_bytes_estimate"]
    assert choose(plan, t, (12, 1), mtp=mtp, long_from=LONG)[0] == 4096
    temp = temporary(t, mtp)
    assert sizes[4096] - sizes[2048] + 2048 * temp <= plan["serving_peak_bytes_estimate"] - old


@pytest.mark.parametrize("share,round_s,row_s", [(0, None, None), (0.5, None, None), (0.5, 0.1, 0.0001)])
def test_idle_pieces_are_wide_and_live_pieces_never_exceed_the_released_limit(share, round_s, row_s):
    assert pass_limit(4096, False, share, round_s, row_s) == 4096
    assert 128 <= pass_limit(4096, True, share, round_s, row_s) <= 2048


def test_decode_share_and_small_test_buffers_keep_their_bounds():
    assert pass_limit(4096, True, 0.5, 0.05, 0.0001) == 960
    assert pass_limit(4096, True, 0.5, 0.0001, 0.0001) == 128
    assert pass_limit(16, True, 0.5, 0.0001, 0.0001) == 16


@pytest.mark.torch
def test_both_planner_paths_bound_live_pieces_and_restore_idle_width():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    dec = object.__new__(MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 4096, 0.0, None, None
    dec.streams, dec.passed, dec.slots = {}, {}, [None]
    prompt = SimpleNamespace(sid=1, prompt=[1] * 9000, background=False)
    dec.filling = [prompt]
    dec.fills = {1: [SimpleNamespace(stops=[]), False, 0, None]}
    assert dec._pieces() == [(prompt, 0, 4096)]
    reply = SimpleNamespace(done=False)
    dec.streams[2] = reply
    assert dec._pieces() == dec._pieces(dec._pass_rows()) == [(prompt, 0, 2048)]
    dec.share, dec.round_s, dec.row_s = 0.5, 0.05, 0.0001
    assert dec._pieces() == [(prompt, 0, 960)]
    reply.done = True
    assert dec._pieces() == [(prompt, 0, 4096)]


@pytest.mark.torch
@pytest.mark.parametrize("mtp", [False, True])
def test_without_a_threshold_nvfp4_keeps_the_released_pieces_and_mlx_its_wide_ones(mtp):
    t = text(modelopt())
    plan = receipt(t, 1 << 40)
    old = copy.deepcopy(plan)
    assert choose(plan, t, (12, 1), mtp=mtp) == (2048, 0)
    assert plan == old
    mlx = text()
    assert choose(receipt(mlx, 1 << 40), mlx, (12, 1))[0] == 4096


@pytest.mark.torch
@pytest.mark.parametrize("long_from", [1, 4096, 8192, 131072])
def test_any_threshold_budgets_the_whole_wide_piece(long_from):
    """A start that fits at one threshold fits a prompt that takes 4,096-row pieces at any other."""

    t = text(modelopt())
    plans = {k: receipt(t, 1 << 40) for k in (long_from, LONG)}
    picked = {k: choose(plan, t, (12, 1), mtp=True, long_from=k) for k, plan in plans.items()}
    assert picked[long_from] == picked[LONG] == (4096, indexed_prompt_bytes(t, 4096) + 4096 * temporary(t, True))
    assert plans[long_from] == plans[LONG]


def test_the_threshold_comes_from_the_environment(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import long_piece_tokens

    monkeypatch.delenv("TENSORFOLD_LONG_PIECE_TOKENS", raising=False)
    assert long_piece_tokens() == 0
    for value, tokens in [(" ", 0), ("0", 0), ("8192", 8192), (" 32768 ", 32768)]:
        monkeypatch.setenv("TENSORFOLD_LONG_PIECE_TOKENS", value)
        assert long_piece_tokens() == tokens
    for value in ["-1", "8k", "8192.0", "off"]:
        monkeypatch.setenv("TENSORFOLD_LONG_PIECE_TOKENS", value)
        with pytest.raises(ValueError, match="TENSORFOLD_LONG_PIECE_TOKENS"):
            long_piece_tokens()


def test_pieces_are_wide_while_the_threshold_remains_and_narrow_after():
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import long_pieces, piece_rows, pieces_of, warm_widths

    assert piece_rows(4096, 10) == 4096 and warm_widths(4096) == [4096]        # nothing installed: one width
    with long_pieces(8192):
        assert [piece_rows(4096, left) for left in (131072, 8192, 8191, 100)] == [4096, 4096, 2048, 2048]
        assert piece_rows(2048, 131072) == 2048 and piece_rows(1024, 100) == 1024
        assert warm_widths(4096) == [4096, 2048] and warm_widths(2048) == [2048]
        with pieces_of(4096, 4096):
            assert piece_rows(4096, 100) == 4096
        with pieces_of(2048, 4096):
            assert piece_rows(4096, 131072) == 2048
        assert piece_rows(4096, 100) == 2048
    assert piece_rows(4096, 100) == 4096


@pytest.mark.torch
@pytest.mark.parametrize("long_from,widths", [
    (0, [4096] * 4 + [3616]),
    (8192, [4096] * 3 + [2048] * 3 + [1568]),         # 7,712 rows left after three wide pieces
    (20000, [4096] + [2048] * 7 + [1568]),
    (20001, [2048] * 9 + [1568]),
])
def test_the_serial_plan_switches_width_once_the_rows_left_fall_under_the_threshold(long_from, widths):
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.decode import chunk_plan
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import long_pieces

    with long_pieces(long_from):
        plan = chunk_plan(0, 20000, 4096, [10000])
    assert [end - start for start, end, _ in plan] == widths
    assert plan[-1][1] == 20000 and [p for *_, here in plan for p in here] == [10000]


@pytest.mark.torch
def test_an_idle_pass_narrows_by_the_rows_every_filling_prompt_has_left():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import long_pieces

    dec = object.__new__(MultiDecoder)
    dec.prefill_rows, dec.share, dec.round_s, dec.row_s = 4096, 0.0, None, None
    dec.streams, dec.passed, dec.slots = {}, {}, [None, None]
    a = SimpleNamespace(sid=1, prompt=[1] * 9000, background=False)
    b = SimpleNamespace(sid=2, prompt=[1] * 3000, background=False)
    dec.filling = [a]
    dec.fills = {1: [SimpleNamespace(stops=[]), False, 0, None], 2: [SimpleNamespace(stops=[]), False, 0, None]}
    with long_pieces(8192):
        assert dec._pieces() == [(a, 0, 4096)]               # 9,000 rows left
        dec.fills[1][2] = 4096
        assert dec._pieces() == [(a, 4096, 2048)]            # 4,904 left
        dec.filling = [a, b]
        assert dec._pieces() == [(b, 0, 2048)]               # 7,904 left between them: fewest rows first
        dec.fills[1][2] = 0
        assert dec._pieces() == [(b, 0, 3000), (a, 0, 1096)]     # 12,000 left
        dec.streams[3] = SimpleNamespace(done=False)
        assert dec._pass_rows() == 2048                      # live replies keep the released bound
