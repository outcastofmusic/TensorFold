"""A GLM folder whose routed experts are stacked a layer (sushi's layout) beside packed dense tensors, without a GPU:
what admission counts for it, what its config says about the experts, and the tool that assembles one."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = "model.language_model."


def _write(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    entries, blob = {}, b""
    for name, (dtype, shape, data) in tensors.items():
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(data)]}
        blob += data
    raw = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def _read(path: Path) -> dict[str, bytes]:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    entries = json.loads(raw[8:8 + size])
    return {k: raw[8 + size + v["data_offsets"][0]:8 + size + v["data_offsets"][1]] for k, v in entries.items()}


def test_admission_counts_stacked_experts_as_stored_and_keeps_them_out_of_the_projection_scratch():
    pytest.importorskip("torch")
    from tensorfold.families.glm5_next.cuda import packed

    cfg = SimpleNamespace(layers=45, hidden=4096, moe_width=2048, experts=288, top_k=8, mtp_layers=0, streams=4)
    count = packed.resident_transform(cfg, load_mtp=False)
    gate = {"dtype": "U16", "shape": [288, 256, 128, 36]}
    down = {"dtype": "U16", "shape": [288, 128, 256, 36]}
    stored = 288 * 256 * 128 * 36 * 2
    assert count(P + "layers.3.mlp.switch_mlp.gate_proj.trellis", gate)[0] == stored + 288 * 12
    assert count(P + "layers.3.mlp.switch_mlp.down_proj.trellis", down)[0] == stored + 288 * 20
    assert count(P + "layers.3.mlp.switch_mlp.gate_proj.suh", {"dtype": "F16", "shape": [288, 4096]})[0] == 288 * 4096 * 2
    head = {"lm_head.trellis": {"dtype": "I16", "shape": [256, 9680, 80]}}
    with_experts = {**head, P + "layers.3.mlp.switch_mlp.gate_proj.trellis": gate}
    assert packed.workspace_bytes(cfg, with_experts, load_mtp=False) == packed.workspace_bytes(cfg, head, load_mtp=False)


def _tool():
    spec = importlib.util.spec_from_file_location("stacked_pack", ROOT / "tools" / "glm5_stacked_experts_pack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packs(tmp_path: Path) -> tuple[Path, Path]:
    dense, experts = tmp_path / "dense", tmp_path / "experts"
    dense.mkdir(), experts.mkdir()
    text = {"num_hidden_layers": 4, "first_k_dense_replace": 3, "num_nextn_predict_layers": 1}
    (dense / "config.json").write_text(json.dumps({"text_config": text, "quantization_config": {"quant_method": "exl3"}}))
    (dense / "tokenizer.json").write_text("{}")
    rng = np.random.default_rng(0)

    def blob(n):
        return rng.integers(0, 256, n, dtype=np.uint8).tobytes()

    _write(dense / "model-00001-of-00002.safetensors", {
        P + "layers.3.mlp.experts.0.gate_proj.trellis": ("I16", [1, 1, 32], blob(64)),
        P + "layers.0.self_attn.qkv_proj.trellis": ("I16", [2, 2, 64], blob(512)),
        P + "layers.0.self_attn.qkv_proj.mul1": ("I32", [], blob(4)),
        "model.visual.blocks.0.weight": ("F16", [4], blob(8))})
    _write(dense / "model-00002-of-00002.safetensors", {
        P + "layers.4.eh_proj.trellis": ("I16", [1, 1, 32], blob(64)),          # the MTP layer
        P + "layers.3.mlp.gate.weight": ("F16", [2, 4], blob(16)),
        "lm_head.trellis": ("I16", [1, 2, 80], blob(320))})
    (experts / "config.json").write_text(json.dumps({"expert_quant": {"format": "exl3", "k": 2.5, "codebook": "mcg",
                                                                       "window": 14}}))
    _write(experts / "model-exl3-L03.safetensors", {
        P + f"layers.3.mlp.switch_mlp.{proj}.{part}": ("U16" if part == "trellis" else "F16", [2, 4], blob(16))
        for proj in ("gate_proj", "up_proj", "down_proj") for part in ("trellis", "suh", "svh")})
    return dense, experts


def test_the_tool_copies_the_dense_tensors_links_the_experts_and_drops_the_mtp_layer(tmp_path, monkeypatch, capsys):
    tool = _tool()
    dense, experts = _packs(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["tool", "--dense", str(dense), "--experts", str(experts), "--out", str(out)])
    tool.main()
    got = _read(out / "model-dense.safetensors")
    source = {**_read(dense / "model-00001-of-00002.safetensors"), **_read(dense / "model-00002-of-00002.safetensors")}
    kept = {P + "layers.0.self_attn.qkv_proj.trellis", P + "layers.0.self_attn.qkv_proj.mul1",
            P + "layers.3.mlp.gate.weight", "lm_head.trellis"}
    assert set(got) == kept and all(got[k] == source[k] for k in kept)
    assert (out / "model-exl3-L03.safetensors").resolve() == (experts / "model-exl3-L03.safetensors").resolve()
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert {index["weight_map"][k] for k in kept} == {"model-dense.safetensors"}
    assert index["weight_map"][P + "layers.3.mlp.switch_mlp.down_proj.trellis"] == "model-exl3-L03.safetensors"
    assert len(index["weight_map"]) == len(kept) + 9
    assert index["metadata"]["total_size"] == sum(len(got[k]) for k in kept) + 9 * 16
    config = json.loads((out / "config.json").read_text())
    assert config["expert_quant"] == {"format": "exl3", "k": 2.5, "codebook": "mcg", "window": 14}
    assert config["text_config"]["num_nextn_predict_layers"] == 0
    assert (out / "tokenizer.json").is_file()


def test_the_tool_refuses_a_pack_with_a_missing_expert_layer_before_writing(tmp_path, monkeypatch):
    tool = _tool()
    dense, experts = _packs(tmp_path)
    (experts / "model-exl3-L03.safetensors").unlink()
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["tool", "--dense", str(dense), "--experts", str(experts), "--out", str(out)])
    with pytest.raises(SystemExit, match="lacks 1 expert files"):
        tool.main()
    assert not out.exists()


def test_a_rerun_after_an_interrupted_assembly_finishes_it(tmp_path, monkeypatch):
    tool = _tool()
    dense, experts = _packs(tmp_path)
    out = tmp_path / "out"
    argv = ["tool", "--dense", str(dense), "--experts", str(experts), "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    real = tool.write_dense

    def interrupted(*args):
        real(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(tool, "write_dense", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tool.main()
    assert not out.exists()                          # prepare.sh reads config.json: a half-built folder is never there
    monkeypatch.setattr(tool, "write_dense", real)
    tool.main()
    assert (out / "config.json").is_file() and (out / "model-exl3-L03.safetensors").is_symlink()
