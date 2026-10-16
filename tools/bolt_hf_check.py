"""Compare Bolt's reference forward with the checkpoint's own modeling_nemotron_h.py on its first layers.

Each Hugging Face block is built from the checkpoint's NemotronHPuzzleConfig for its layer, loaded with the same exact
fp32 weights the reference decodes, run in fp32 on its torch path, and freed before the next layer. A layer is checked
two ways: on the reference's own input (the layer's error alone) and along each side's chained residual stream.

The torch path clamps Mamba's dt at time_step_min (1e-3) where the checkpoint's CUDA path and vLLM pass dt_limit
(0, inf), so a Mamba layer is also run with the reference's dt floor at 1e-3: that column shows the rest of the gap.
modeling_nemotron_h.py imports mamba_ssm's Triton rmsnorm_fn, which this image lacks; the tool supplies the reference
definition (vLLM's flash_linear_attention/ops/layernorm_guard.py rms_norm_ref) under that name.

    python tools/bolt_hf_check.py --model /models/Nemotron-3.5-Bolt-74B-A9B-24GB --layers 9 --out check.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from tensorfold.families.nemotron_bolt import reference as ref

TEXT = ("The river ran high that spring, and the ferry stopped for the first time anyone could remember. "
        "def merge(intervals):\n    intervals = sorted(intervals)\n    out = [intervals[0]]\n    return out\n")
HF_FILES = ("modeling_nemotron_h.py", "modeling_nemotron_h_puzzle.py", "configuration_nemotron_h.py",
            "configuration_nemotron_h_puzzle.py")
RMSNORM_FN = '''
import torch
import torch.nn.functional as F


def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        g = x.reshape(*x.shape[:-1], -1, group_size)
        rstd = 1 / torch.sqrt((g.square()).mean(dim=-1, keepdim=True) + eps)
        out = (g * rstd).reshape(x.shape) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)
'''


def import_hf(model_dir: Path):
    """The checkpoint's modeling files as package ``bolt_hf``, with the gated RMSNorm they import from mamba_ssm."""

    root = Path(tempfile.mkdtemp(prefix="bolt-hf-"))
    pkg = root / "bolt_hf"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name in HF_FILES:
        shutil.copy(model_dir / name, pkg / name)
    triton = root / "mamba_ssm" / "ops" / "triton"
    triton.mkdir(parents=True)
    for d in (root / "mamba_ssm" / "ops", triton):
        (d / "__init__.py").write_text("")
    # transformers reads the version of whatever mamba_ssm imports; below 2.0.4 it reports no kernels
    (root / "mamba_ssm" / "__init__.py").write_text('__version__ = "0.0.1"\n')
    (triton / "layernorm_gated.py").write_text(RMSNORM_FN)
    sys.path.insert(0, str(root))
    import bolt_hf.modeling_nemotron_h as modeling
    from bolt_hf.configuration_nemotron_h_puzzle import NemotronHPuzzleConfig

    if modeling.is_fast_path_available:
        raise RuntimeError("mamba_ssm's kernels are importable here; this check compares against the torch path")
    return modeling, NemotronHPuzzleConfig


def hf_block(modeling, config, ck: ref.Checkpoint, i: int):
    layer_config = config.get_nemotron_h_config_for_layer(i)
    layer_config._attn_implementation = "eager"
    with torch.device("meta"):
        block = modeling.NemotronHBlock(layer_config, layer_idx=i)
    state = {}
    for key in block.state_dict():
        full = f"{ref.LAYERS}{i}.{key}"
        module = full.removesuffix(".weight")
        quantized = key.endswith(".weight") and ck.recipe(module) is not None
        state[key] = ck.linear(module) if quantized else ck.vector(full)
    block.load_state_dict(state, assign=True, strict=True)
    return block.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    modeling, Puzzle = import_hf(args.model)
    raw = json.loads((args.model / "config.json").read_text())["llm_config"]
    config = Puzzle(**raw)
    ck = ref.Checkpoint(args.model, args.device)
    clamped = dataclasses.replace(ck.config, dt_floor=float(raw["time_step_min"]))
    ids = torch.tensor(Tokenizer.from_file(str(args.model / "tokenizer.json")).encode(TEXT).ids)
    print(f"{len(ids)} tokens; HF torch path, fp32, exact decoded weights", flush=True)
    rows = []
    with torch.no_grad():
        x_ref = ref.embed(ck, ids)
        x_hf = x_ref.clone()
        chain = ref.State()
        for i in range(args.layers):
            t0 = time.time()
            kind = ck.config.blocks[i].kind
            block = hf_block(modeling, config, ck, i)
            hf_on_ref = block(x_ref[None])[0]
            x_hf = block(x_hf[None])[0]
            del block
            mine = ref.layer(ck, i, x_ref, ref.State())
            row = {"layer": i, "kind": kind, "scale": x_ref.abs().max().item(),
                   "isolated": (mine - hf_on_ref).abs().max().item()}
            if kind == "mamba":
                plain, ck.config = ck.config, clamped
                row["isolated_dt_floor_1e-3"] = (ref.layer(ck, i, x_ref, ref.State()) - hf_on_ref).abs().max().item()
                ck.config = plain
            x_ref = ref.layer(ck, i, x_ref, chain)
            row["chained"] = (x_ref - x_hf).abs().max().item()
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps({"tokens": len(ids), "layers": rows}, indent=1))


if __name__ == "__main__":
    main()
