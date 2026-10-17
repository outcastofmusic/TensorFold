"""Nemotron 3.5 Bolt on CPU: the config from block_configs, the ggml dequantizers against llama.cpp's loops, the
loader's name mapping on a tiny checkpoint in the real layout, the fp32 reference, and the family's registration."""

from __future__ import annotations

import json
import struct

import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.bolt_tiny import write

    return write(tmp_path_factory.mktemp("bolt"))


def test_config_reads_each_block():
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_tiny import EXPERTS, WIDTHS, llm_config

    c = Config.parse({"model_type": "nemotron_h_omni", "llm_config": llm_config()})
    assert c.pattern == "ME*EME"
    assert c.experts == EXPERTS and c.moe_width == WIDTHS and c.top_k == (3, 3, 3)
    assert c.kv_heads == (2,) and c.latent == 256 and c.shared_width == 512
    assert c.dt_min == 0.0 and c.dt_max == float("inf")
    assert (c.xd, c.conv_dim, c.proj_dim) == (256, 768, 1032)


def test_config_eos_lists_generation_config_first(tiny):
    from tensorfold.families.nemotron_bolt.cuda.weights import Config

    assert Config.read(tiny).eos == (2, 11)


def test_a_per_block_kv_head_count_overrides_the_global_one():
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_tiny import llm_config

    t = llm_config()
    t["block_configs"][2]["num_key_value_heads"] = 4
    assert Config.parse(t).kv_heads == (4,)


def test_q8_0_slices_equal_the_whole_decode():
    import torch

    from tensorfold.families.nemotron_bolt import quant
    from tensorfold.families.nemotron_bolt.cuda.weights import dequantize_q8_0_rows
    from tests.bolt_tiny import q8_0

    gen = torch.Generator().manual_seed(4)
    packed = q8_0(torch.randn((301, 128), generator=gen), gen).flatten(-2)
    whole = quant.dequantize_q8_0(packed)
    for rows in (1, 7, 300, 301, 4096):
        assert torch.equal(dequantize_q8_0_rows(packed, rows), whole), rows


def test_admission_counts_the_drafter_and_the_widest_activation(tiny, tmp_path):
    import torch

    from tensorfold.families.nemotron_bolt.cuda.app import cache_bytes, chunk_bytes, need_bytes
    from tensorfold.families.nemotron_bolt.cuda.lilicorr import Drafter, device_bytes
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_lilicorr_fakes import write_draft

    c = Config.read(tiny)
    # the tiny model's widest prompt activation is a routed MoE's 3 x 512, wider than the shared expert's 512; no
    # tensor is top_k x Mamba's in_proj wide
    assert chunk_bytes(c, 64) == 64 * (3 * 512 * 2 * 2 + c.proj_dim * 2 + 8 * c.hidden * 2)
    assert cache_bytes(c, 1024, 16, 64) - chunk_bytes(c, 64) == cache_bytes(c, 1024, 16, 0)
    draft = write_draft(tmp_path / "draft")
    loaded = Drafter(draft, embed=lambda ids: torch.zeros((ids.shape[0], 64)), lm_head=lambda h: h, device="cpu")
    assert device_bytes(draft, fp4=False) == loaded.nbytes()                  # what a CPU load holds, all dense
    packed = device_bytes(draft)                                              # CUDA: fc and the MLPs stay NVFP4
    assert 0 < packed < loaded.nbytes()
    assert need_bytes(tiny, c, 1024, drafter=draft) - need_bytes(tiny, c, 1024) == packed


def _f16(b: bytes) -> float:
    return struct.unpack("<e", b)[0]


def test_q8_0_is_code_times_scale():
    import torch

    from tensorfold.families.nemotron_bolt.quant import dequantize_q8_0

    gen = torch.Generator().manual_seed(1)
    packed = torch.randint(0, 256, (3, 2 * 34), generator=gen, dtype=torch.int32).to(torch.uint8)
    packed.view(3, 2, 34)[..., :2] = torch.tensor([0.0123], dtype=torch.float16).view(torch.uint8)
    got = dequantize_q8_0(packed)
    raw = bytes(packed.flatten().tolist())
    want = []
    for row in range(3):                                  # llama.cpp dequantize_row_q8_0: y = qs[j] * d
        for blk in range(2):
            off = row * 68 + blk * 34
            d = torch.tensor(_f16(raw[off:off + 2]), dtype=torch.float32)
            want += [(torch.tensor(float(struct.unpack("b", raw[off + 2 + j:off + 3 + j])[0])) * d).item()
                     for j in range(32)]
    assert torch.equal(got, torch.tensor(want, dtype=torch.float32).to(torch.bfloat16).view(3, 64))


def _iq2xxs_loop(raw: bytes, nb: int) -> list[float]:
    """llama.cpp dequantize_row_iq2_xxs, transliterated (fp32 at each step, as the C code rounds)."""

    import numpy as np

    from tensorfold.families.nemotron_bolt.cuda.fallback import GRID

    signs = [i | ((i.bit_count() & 1) << 7) for i in range(128)]
    f = np.float32
    out = []
    for i in range(nb):
        blk = raw[i * 66:(i + 1) * 66]
        d = f(_f16(blk[:2]))
        for ib in range(8):
            a0, a1 = struct.unpack("<II", blk[2 + 8 * ib:2 + 8 * ib + 8])
            db = f(f(d * f(f(0.5) + f(a1 >> 28))) * f(0.25))
            for l in range(4):
                g = GRID[(a0 >> (8 * l)) & 0xFF]
                s = signs[(a1 >> (7 * l)) & 127]
                for j in range(8):
                    out.append(float(f(f(db * f((g >> (8 * j)) & 0xFF)) * f(-1.0 if s & (1 << j) else 1.0))))
    return out


def test_iq2xxs_matches_llama_cpp():
    import torch

    from tensorfold.families.nemotron_bolt.cuda.fallback import dequantize_iq2xxs, dequantize_iq2xxs_torch

    gen = torch.Generator().manual_seed(2)
    packed = torch.randint(0, 256, (2, 3 * 66), generator=gen, dtype=torch.int32).to(torch.uint8)
    packed.view(2, 3, 66)[..., :2] = torch.tensor([0.00731], dtype=torch.float16).view(torch.uint8)
    got = dequantize_iq2xxs_torch(packed)
    want = []
    for row in range(2):
        want += _iq2xxs_loop(bytes(packed[row].tolist()), 3)
    assert got.shape == (2, 768)
    assert torch.equal(got, torch.tensor(want, dtype=torch.float32).to(torch.bfloat16).view(2, 768))
    assert torch.equal(dequantize_iq2xxs(packed), got)          # a CPU tensor takes the torch definition


def test_grouped_rows_take_their_own_expert():
    import torch

    from tensorfold.families.nemotron_bolt.cuda.fallback import dequantize_iq2xxs_torch, iq2xxs_grouped, iq2xxs_linear
    from tests.bolt_tiny import iq2xxs

    gen = torch.Generator().manual_seed(3)
    table = torch.stack([iq2xxs(64, 512, gen).reshape(64, -1) for _ in range(5)])
    x = torch.randn((7, 512), generator=gen).to(torch.bfloat16)
    ids = torch.tensor([4, 0, 4, 2, 2, 2, 1], dtype=torch.int32)
    got = iq2xxs_grouped(x, table, ids)
    for r in range(7):
        w = dequantize_iq2xxs_torch(table[ids[r]])
        assert torch.equal(got[r], iq2xxs_linear(x[r:r + 1], table[ids[r]])[0])
        assert torch.allclose(got[r].float(), (x[r].float() @ w.float().T), rtol=2e-2, atol=1e-3)


def test_loader_maps_the_checkpoint(tiny):
    import numpy as np
    import torch
    from safetensors import safe_open

    from tensorfold.cuda.nvfp4 import format as nvfp4_format
    from tensorfold.families.nemotron_bolt.cuda.fallback import dequantize_iq2xxs
    from tensorfold.families.nemotron_bolt.cuda.weights import load
    from tensorfold.families.nemotron_bolt.quant import dequantize_q8_0
    from tests.bolt_tiny import EXPERTS

    w = load(tiny, "cpu")
    assert [b.kind for b in w.blocks] == list("ME*EME")
    stored = {}
    for f in sorted(tiny.glob("model-*.safetensors")):
        with safe_open(str(f), "pt") as h:
            stored.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118  (safe_open has no __iter__)
    p = "language_model.backbone.layers."
    moe = w.blocks[3].moe
    assert moe.up.shape == (EXPERTS[1], 512, 66) and moe.down.shape == (EXPERTS[1], 256, 2 * 66)
    assert torch.equal(moe.up[5], stored[p + "3.mixer.experts.5.up_proj.weight"].reshape(512, -1))
    assert torch.equal(moe.down[11], stored[p + "3.mixer.experts.11.down_proj.weight"].reshape(256, -1))
    assert moe.shared_up.shape == (512, 2 * 66) and moe.top_k == 3
    assert dequantize_iq2xxs(moe.shared_down).shape == (512, 512)
    attn = w.blocks[2].attn
    q = dequantize_q8_0(stored[p + "2.mixer.q_proj.weight"].reshape(256, -1))
    assert attn.qkv.shape == (256 + 2 * 128, 512) and torch.equal(attn.qkv[:256], q)
    assert w.head.shape == (1024, 512) and w.embed.dtype == torch.bfloat16
    m = w.blocks[0].mamba
    name = p + "0.mixer.in_proj"
    want = nvfp4_format.dequant("nvfp4", stored[name + ".weight"].numpy(),
                                stored[name + ".weight_scale"].view(torch.uint8).numpy(),
                                float(stored[name + ".weight_scale_2"]))
    assert np.array_equal(m.in_proj.dequantized().float().numpy(),
                          torch.from_numpy(np.asarray(want, dtype=np.float32)).to(torch.bfloat16).float().numpy())
    conv = stored[p + "0.mixer.conv1d.weight"]
    assert torch.equal(m.conv_w, conv[:, 0, :].float().t())          # tap k: the input 3 - k rows back
    assert torch.equal(m.a, -torch.exp(stored[p + "0.mixer.A_log"].float()))


def test_loader_names_a_tensor_it_did_not_read(tmp_path):
    import torch

    from tensorfold.families.nemotron_bolt.cuda.weights import load
    from tests.bolt_tiny import write

    path = write(tmp_path / "odd", extra={"language_model.backbone.layers.0.mixer.surprise": torch.zeros(4)})
    with pytest.raises(ValueError, match="surprise"):
        load(path, "cpu")


def test_a_truncated_load_reads_only_its_layers(tiny):
    from tensorfold.families.nemotron_bolt.cuda.weights import load, weight_bytes

    w = load(tiny, "cpu", layers=2)
    assert w.config.pattern == "ME" and w.config.experts == (8,) and len(w.blocks) == 2
    assert weight_bytes(tiny, 2) < weight_bytes(tiny)


def test_reference_is_causal_and_taps_the_stream(tiny):
    import torch

    from tensorfold.families.nemotron_bolt.cuda import reference
    from tensorfold.families.nemotron_bolt.cuda.weights import load

    w = load(tiny, "cpu")
    tokens = torch.tensor([5, 77, 300, 2, 999, 14, 15, 600])
    full, taps = reference.forward(w, tokens, taps=(1, 5))
    head, _ = reference.forward(w, tokens[:5])
    assert full.shape == (8, 1024) and torch.isfinite(full).all()
    assert torch.allclose(full[:5], head, rtol=1e-4, atol=1e-4)
    assert [t.shape for t in taps] == [(8, 512), (8, 512)]
    final = reference._rms(taps[1], w.norm_f, w.config.eps) @ w.head.float().T
    assert torch.allclose(final, full, rtol=1e-5, atol=1e-5)      # the last layer's stream is the head's input


def test_family_is_detected_and_its_weights_readable(tiny):
    from tensorfold import families

    family = families.detect(tiny)
    assert family.module == "tensorfold.families.nemotron_bolt"
    families.require_readable(family, families.read_config(tiny), "cuda")
    family.package.check(tiny)
    assert families.backends_of(family) == ("cuda",)


def test_check_refuses_another_language_model(tmp_path):
    from tensorfold.families.nemotron_bolt import check
    from tests.bolt_tiny import write

    path = write(tmp_path / "plain")
    config = json.loads((path / "config.json").read_text())
    config["llm_config"]["model_type"] = "nemotron_h"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nemotron_h_puzzle"):
        check(path)


def test_check_refuses_an_unknown_layer_format(tmp_path):
    from tensorfold.families.nemotron_bolt import check
    from tests.bolt_tiny import write

    path = write(tmp_path / "q4k")
    quant = json.loads((path / "hf_quant_config.json").read_text())
    quant["quantization"]["quantized_layers"]["language_model.lm_head"]["quant_algo"] = "Q4_K"
    (path / "hf_quant_config.json").write_text(json.dumps(quant))
    with pytest.raises(ValueError, match="Q4_K"):
        check(path)


def test_draft_taps_read_the_drafters_target_layers(tmp_path):
    from tensorfold.families.nemotron_bolt.cuda.app import draft_taps

    (tmp_path / "config.json").write_text(json.dumps({"dflash_config": {"target_layer_ids": [1, 18, 35]}}))
    assert draft_taps(tmp_path) == (1, 18, 35)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ValueError, match="target layers"):
        draft_taps(tmp_path)
