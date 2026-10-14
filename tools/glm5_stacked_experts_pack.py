"""Assemble a GLM-5.3-Flash folder from one pack's dense tensors and another's stacked routed experts.

    python tools/glm5_stacked_experts_pack.py --dense ~/models/glm53-flash-mosaic-12l \\
        --experts ~/models/glm53-flash-sushi-2.4bpw --out ~/models/glm53-flash-sushi-experts-mosaic-dense

Both packs quantize zai-org/GLM-5.3-Flash, so their tensors mix the way a mosaic's do. ``--dense`` is a packed EXL3
folder the CUDA engine already serves (turboderp's layout: trellis attention and head, per-expert routed tensors);
its routed experts, MTP layer and vision tower are left out. ``--experts`` is a sushi pack (github.com/beamivalice/sushi):
one ``model-exl3-L<layer>.safetensors`` a MoE layer, each projection's experts stacked in one trellis. Nothing is
decoded or requantized: the dense tensors are copied byte for byte into ``model-dense.safetensors`` and the expert
files are linked. The folder's config is the dense pack's with the expert pack's ``expert_quant`` and no MTP layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from contextlib import ExitStack
from pathlib import Path

LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "processor_config.json", "LICENSE")
BLOCK = 64 << 20


def header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        found = json.loads(f.read(size))
    found.pop("__metadata__", None)
    return found, 8 + size


def dense_tensors(folder: Path, layers: int) -> list[tuple[str, Path, int, dict]]:
    """(name, file, data start, entry) of every tensor the engine reads outside the routed experts, in file order."""

    kept = []
    for path in sorted(folder.glob("*.safetensors")):
        entries, base = header(path)
        for name, info in sorted(entries.items(), key=lambda item: item[1]["data_offsets"][0]):
            layer = LAYER.match(name)
            if name.startswith("model.visual.") or ".mlp.experts." in name or (layer and int(layer[1]) >= layers):
                continue
            kept.append((name, path, base, info))
    return kept


def write_dense(out: Path, tensors: list[tuple[str, Path, int, dict]]) -> None:
    entries, offset = {}, 0
    for name, _, _, info in tensors:
        size = info["data_offsets"][1] - info["data_offsets"][0]
        entries[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(entries, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(out, "wb") as dst, ExitStack() as stack:
        dst.write(struct.pack("<Q", len(raw)) + raw)
        opened: dict[Path, object] = {}
        for _, path, base, info in tensors:
            if path not in opened:
                opened[path] = stack.enter_context(open(path, "rb"))
            src = opened[path]
            src.seek(base + info["data_offsets"][0])
            left = info["data_offsets"][1] - info["data_offsets"][0]
            while left:
                block = src.read(min(BLOCK, left))
                if not block:
                    raise OSError(f"{path} ends inside a tensor")
                dst.write(block)
                left -= len(block)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dense, experts, out = (p.expanduser().resolve() for p in (args.dense, args.experts, args.out))

    config = json.loads((dense / "config.json").read_text())
    stacked = json.loads((experts / "config.json").read_text()).get("expert_quant")
    if not stacked or stacked.get("format") != "exl3":
        raise SystemExit(f"{experts}/config.json names no exl3 expert_quant: not a stacked-experts pack")
    text = config["text_config"]
    layers = int(text["num_hidden_layers"])
    moe = [i for i, kind in enumerate(text.get("mlp_layer_types") or
                                      ["dense"] * int(text.get("first_k_dense_replace", 3)) +
                                      ["sparse"] * (layers - int(text.get("first_k_dense_replace", 3))))
           if kind == "sparse"]
    files = {i: experts / f"model-exl3-L{i:02d}.safetensors" for i in moe}
    missing = [p.name for p in files.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"{experts} lacks {len(missing)} expert files: {', '.join(missing[:4])} ...")

    if out.exists():
        raise SystemExit(f"{out} exists: remove it to assemble again")
    final, out = out, out.with_name(out.name + ".partial")   # renamed once whole, so a rerun finds no half-built folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tensors = dense_tensors(dense, layers)
    write_dense(out / "model-dense.safetensors", tensors)
    weight_map = {name: "model-dense.safetensors" for name, _, _, _ in tensors}
    total = sum(info["data_offsets"][1] - info["data_offsets"][0] for _, _, _, info in tensors)
    for i, path in files.items():
        entries, _ = header(path)
        wanted = {f"model.language_model.layers.{i}.mlp.switch_mlp.{proj}.{part}"
                  for proj in ("gate_proj", "up_proj", "down_proj") for part in ("trellis", "suh", "svh")}
        if set(entries) != wanted:
            raise SystemExit(f"{path.name} does not hold exactly layer {i}'s nine stacked expert tensors")
        os.symlink(path, out / path.name)
        weight_map.update(dict.fromkeys(entries, path.name))
        total += sum(e["data_offsets"][1] - e["data_offsets"][0] for e in entries.values())

    text["num_nextn_predict_layers"] = 0
    config["expert_quant"] = stacked
    config["tensorfold_assembled_from"] = {"dense": dense.name, "experts": experts.name}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1) + "\n")
    for name in COPIED:
        if (dense / name).is_file():
            shutil.copy2(dense / name, out / name)
    out.rename(final)
    print(f"{final}: {len(tensors)} dense tensors copied, {len(files)} expert files linked, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
