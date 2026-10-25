"""TENSORFOLD_PREFILL_ROWS on a Flash Next EXL3 pack (CPU): the prompt rows ``exl3_mm.read`` takes, the routed window
they give, the admission and scratch sized for them, and the routed grouping's pick limit at the largest window.
tests/cuda/test_exl3_prompt_rows.py compares the bytes of 2,048- and 4,096-row chunks on the GPU."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm, exl3_pack  # noqa: E402
from tests.exl3_fakes import _layer  # noqa: E402
from tests.test_cuda_capacity import HEAD, Loaded, checkpoint, fake_runtime, small_config  # noqa: E402, F401

ENV = "TENSORFOLD_PREFILL_ROWS"
TEXT = {"hidden_size": 2560, "moe_intermediate_size": 640, "num_experts_per_tok": 10, "num_attention_heads": 24,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 48,
        "linear_value_head_dim": 128, "hc_count": 4, "heads_per_ngram": 8, "ple_embed_dim": 2560}


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(exl3_mm.MOE_CHUNK_ENV, raising=False)
    before = exl3_mm.current()
    yield
    exl3_mm.use(before)


@pytest.mark.parametrize("value, rows, window", [(None, 4096, 4096), ("2048", 2048, 2048), ("4096", 4096, 4096),
                                                 ("1024", 1024, 1024), ("8192", 8192, 4096)])
def test_the_prompt_rows_and_their_routed_window(monkeypatch, value, rows, window):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    got = exl3_mm.read()
    assert got.prompt_rows == rows
    assert dataclasses.replace(got, prompt_rows=4096) == exl3_mm.Switches()    # the other switches keep their defaults
    exl3_mm.use(got)
    assert exl3_mm.moe_window() == window
    exl3_mm.use(dataclasses.replace(got, moe_chunk=False))
    assert exl3_mm.moe_window() == 1024                                        # MOE_WINDOW, whatever the chunk


def test_unset_is_the_served_chunk():
    assert exl3_mm.Switches().prompt_rows == exl3_mm.PROMPT_ROWS == exl3_mm.MOE_MOST_ROWS == 4096


@pytest.mark.parametrize("value", ["255", "16385", "4k"])
def test_a_bad_value_stops_the_read(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        exl3_mm.read()


def test_the_largest_window_fits_the_scan_grouping():
    slots = TEXT["num_experts_per_tok"] + 1
    assert experts.scan_fits(512, exl3_mm.MOE_MOST_ROWS * slots)
    assert not experts.scan_fits(512, 2 * exl3_mm.MOE_MOST_ROWS * slots)


@pytest.mark.parametrize("rows", [2048, 4096])
def test_the_admission_and_the_scratch_take_the_prompt_rows(monkeypatch, rows):
    from tensorfold.cuda import geometry

    exl3_mm.use(dataclasses.replace(exl3_mm.Switches(), prompt_rows=rows))
    seen = []
    monkeypatch.setattr(geometry, "exl3_indexed_scratch", lambda text, w, r: seen.append((w, r)) or 0)
    base = SimpleNamespace(bytes_at=lambda slots: 0, reserve=0, minimum_slots=1)
    exl3_pack.admission(lambda text: base)({})
    assert seen == [(rows, rows)]
    sc = exl3_mm.Scratch(4, True)
    sc.allocate("cpu", experts=_layer(), rows=rows, ple_words=3, ple_heads=2, ple_dim=8)
    assert sc.moe.rows == rows
    assert sc.ple_dev.shape == (rows * 2, 3) and sc.ple_emb.shape == (rows, 8)


def test_4096_rows_admit_under_three_gib_more():
    """Flash Next's shape: the routed scratch doubles, the prompt workspace and n-gram staging double."""

    from tensorfold.cuda.geometry import exl3_indexed_scratch

    two, four = exl3_indexed_scratch(TEXT, 2048, 2048), exl3_indexed_scratch(TEXT, 4096, 4096)
    assert 1.9 < four / two < 2.1
    assert four - two < 3 << 30


@pytest.mark.torch
@pytest.mark.parametrize("value, rows", [(None, 4096), ("2048", 2048)])
def test_an_exl3_engine_cuts_prompts_in_the_switch_s_rows(tmp_path, monkeypatch, fake_runtime, value, rows):  # noqa: F811
    """The pieces the engine cuts, and the rows it admits, follow ``prompt_rows`` unset as well as set."""

    from tensorfold.families.qwen4_exp.cuda import engine

    if value is not None:
        monkeypatch.setenv(ENV, value)
    checkpoint(tmp_path, small_config(), HEAD)
    monkeypatch.setattr(exl3_pack, "is_exl3", lambda path: True)
    monkeypatch.setattr(exl3_pack, "extra_files", lambda path: ())
    obj = engine.FlashNextEngine.__new__(engine.FlashNextEngine)
    with pytest.raises(Loaded):
        obj.__init__(tmp_path, depth=0)
    assert obj.prefill_rows == rows
