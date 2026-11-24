"""tools/hy4_calib.py on the tiny checkpoint (CPU): every trunk projection has a site, each site's sums are its input
rows' X^T X, and tapping leaves the logits' bits alone."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tests import hy4_flash_tiny as tiny

pytestmark = pytest.mark.torch


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sites_cover_the_trunk_and_hold_their_rows_sums(tmp_path):
    import torch

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.hy4_flash.cuda import model
    from tensorfold.families.hy4_flash.cuda.weights import load

    calib, trunk = _tool("hy4_calib"), _tool("hy4_trunk")
    ckpt = tiny.write(tmp_path / "tiny")
    ids = list(range(5, 37))
    plain = model.Engine(load(ckpt, "cpu"), max_len=64, prefill_rows=64)
    plain.set_sampling(Sampling(seed=0, temperature=0.0))
    want = plain.forward(ids, prompt=True, every=True)

    w = load(ckpt, "cpu")
    sites = calib.Sites(full=4096)
    names = calib.tap_model(w, sites)
    w.head = calib.Tap(calib._Head(w.head), "head.in", sites)
    undo = calib.tap_kv_b(w, sites)
    seen = []
    orig = sites.add

    def spy(site, x):
        before = sites.n.get(site, 0)
        orig(site, x)
        if site == "model.layers.2.mlp.in" and sites.n[site] != before:
            seen.append(x.reshape(-1, x.shape[-1]).float().clone())

    sites.add = spy
    try:
        e = model.Engine(w, max_len=64, prefill_rows=64)
        e.set_sampling(Sampling(seed=0, temperature=0.0))
        calib.run(e, [ids], log=lambda m: None)
    finally:
        undo()
    counts, sums = dict(sites.n), {k: v.clone() for k, v in sites.h.items()}
    x = torch.cat(seen)
    e2 = model.Engine(w, max_len=64, prefill_rows=64)
    e2.set_sampling(Sampling(seed=0, temperature=0.0))
    assert torch.equal(e2.forward(ids, prompt=True, every=True), want)

    trunk_names = []
    with open(ckpt / "model.safetensors.index.json") as f:
        import json
        index = json.load(f)["weight_map"]
    for n in index:
        base = n[:-len(".weight")] if n.endswith(".weight") else None
        if base and ".dfly" not in base and any(re.search(pat, base) for k, pat in trunk.CLASSES.items() if k != "kvb"):
            trunk_names.append(base)
    assert trunk_names and all(n in names for n in trunk_names), set(trunk_names) - set(names)
    assert counts["model.layers.2.mlp.in"] == x.shape[0] == len(ids)
    assert torch.allclose(sums["model.layers.2.mlp.in"], x.t() @ x, rtol=1e-5, atol=1e-5)
    assert counts["head.in"] == len(ids)
    kb = sums["model.layers.2.self_attn.kv_b.key"]
    assert kb.shape == (tiny.SLIDING_HEADS, tiny.NOPE) and bool((kb > 0).all())
