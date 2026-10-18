"""NVFP4 prompt pieces of 2,048 and 4,096 rows commit the same bits: the last row's logits, a greedy reply, and
states kept inside a piece that resume to what a fresh prefill leaves.

The tiny ModelOpt checkpoint (NVFP4 experts, MXFP8 linears, an NVFP4 n-gram table) runs wherever a GPU does.
NVIDIA's Qwen3.8 Flash Next FP4-Local runs too when TF_FP4_LOCAL names its folder (its server plans 78 GiB), and
then a 4,096-row piece's measured memory is held to the workspace ``prompt_plan.choose`` plans. Pieces of 1,024+ rows
take the shared-memory expert tile, so both widths run it, and the 1-row last piece runs the 16-pair kernel."""

import os
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA state comparisons require an NVIDIA GPU", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write

from tensorfold.families.qwen4_exp.cuda.decode import (
    Engine,
    chunk_plan,
    prefill,
    prefill_begin,
    prefill_chunk,
    serial_decode,
)
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.weights import load

LENGTH = 8193              # four 2,048-row pieces or two 4,096-row ones, then a 1-row piece
CUTS = (3073, 4097)        # kept points inside a piece at both widths
REPLY = 48
CAPACITY = 8448
WIDTHS = (2048, 4096)


@pytest.fixture(scope="module", params=["tiny", "fp4-local"])
def model(request, tmp_path_factory):
    if request.param == "tiny":
        root = write(tmp_path_factory.mktemp("nvfp4-pieces"), layers=4, experts=16, hidden=512, mxfp8=True,
                     ple_nvfp4=True)
    else:
        root = os.environ.get("TF_FP4_LOCAL")
        if not root:
            pytest.skip("TF_FP4_LOCAL: the FP4-Local checkpoint's folder")
    w = load(root, mtp=True)
    w.root = Path(root)
    yield w
    del w
    torch.cuda.empty_cache()


def _prompt(n: int, vocab: int, seed: int = 11) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, min(vocab, 200_000), (n,), generator=gen).tolist()


def _bits(t: torch.Tensor) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def _snap_diff(a: dict, b: dict) -> list[str]:
    """The snapshot fields whose bytes differ, [] when none: rec and conv name their DeltaNet layers, ple_tail its
    window rows (the last ``row`` rows came from the piece, the rest from the committed tail)."""

    def rows(x: torch.Tensor, y: torch.Tensor) -> list[int]:
        return [i for i in range(x.shape[0]) if _bits(x[i]) != _bits(y[i])]

    diff = [f"{k} {a[k]} != {b[k]}" for k in ("pos", "mtp_len") if a[k] != b[k]]
    diff += [f"{k} {rows(a[k], b[k])}" for k in ("rec", "conv", "ple_tail") if _bits(a[k]) != _bits(b[k])]
    ha, hb = a["ple_history"], b["ple_history"]
    if (ha is None) != (hb is None) or (ha is not None and ha.tolist() != hb.tolist()):
        diff.append(f"ple_history {None if ha is None else ha.tolist()} != {None if hb is None else hb.tolist()}")
    return diff


def _assert_same_state(a: State, b: State) -> None:
    """Committed rows, recurrent states and every cache row up to the position, codes and scales alike."""

    assert a.pos == b.pos and a.mtp_len == b.mtp_len
    assert _snap_diff(a.snapshot(), b.snapshot()) == []
    for la, lb in zip(a.kc, b.kc):
        for name in ("k", "v", "ks", "vs"):
            x, y = getattr(la, name), getattr(lb, name)
            rows = min(a.pos, x.shape[0])           # bf16 caches hold one dummy scale
            assert _bits(x[:rows]) == _bits(y[:rows]), name
    for ia, ib in zip(a.ikc, b.ikc):
        assert _bits(ia[:a.pos]) == _bits(ib[:b.pos])
    assert _bits(a.mtp_ikc[:a.mtp_len]) == _bits(b.mtp_ikc[:b.mtp_len])


def _last_logits(e: Engine, prompt: list[int], resume: dict | None = None) -> torch.Tensor:
    """The prompt's last-row logits, its pieces planned as ``prefill`` plans them."""

    start, last = prefill_begin(e, prompt, resume=resume), None
    for begin, end, _ in chunk_plan(start, len(prompt), e.prefill_rows, ()):
        last, _ = prefill_chunk(e, prompt, begin, end=end)
    return last


def _engines(w, kv_dtype):
    return {rows: Engine(w, capacity=CAPACITY, max_rows=8, prefill_rows=rows, kv_dtype=kv_dtype) for rows in WIDTHS}


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_both_widths_give_the_same_logits_state_and_greedy_reply(model, kv_dtype):
    engines = _engines(model, kv_dtype)
    prompt = _prompt(LENGTH, model.cfg.vocab)
    logits = {rows: _bits(_last_logits(e, prompt)) for rows, e in engines.items()}
    assert logits[2048] == logits[4096]
    _assert_same_state(engines[2048].st, engines[4096].st)
    replies = {}
    for rows, e in engines.items():
        first = prefill(e, prompt, None)
        replies[rows] = serial_decode(e, first, REPLY, None, stop_eos=False).tokens
    assert len(replies[2048]) == REPLY and replies[2048] == replies[4096]


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
@pytest.mark.parametrize("cut", CUTS)
def test_a_state_kept_inside_a_piece_resumes_to_the_fresh_prefill_at_both_widths(model, kv_dtype, cut):
    engines = _engines(model, kv_dtype)
    fresh = Engine(model, capacity=CAPACITY, max_rows=8, prefill_rows=2048, kv_dtype=kv_dtype)
    prompt = _prompt(LENGTH, model.cfg.vocab)
    kept = {}
    for rows, e in engines.items():
        prefill(e, prompt, None, keep_at=cut)
        kept[rows] = e.kept
    prefill(fresh, prompt[:cut], None)                           # what a fresh prefill of the kept rows leaves
    snap = fresh.st.snapshot()
    # each width against the fresh prefill, so a failure says whether 2,048-row pieces alone already differ
    assert {rows: _snap_diff(k["state"], snap) for rows, k in kept.items()} == {rows: [] for rows in WIDTHS}
    assert all(_bits(k["tail"]) == _bits(fresh.last_streams) for k in kept.values())
    want = _bits(_last_logits(fresh, prompt))
    reference = fresh.st
    for rows, e in engines.items():
        assert _bits(_last_logits(e, prompt, resume=kept[rows])) == want, rows
        _assert_same_state(e.st, reference)
    first = prefill(fresh, prompt, None)
    reply = serial_decode(fresh, first, REPLY, None, stop_eos=False).tokens
    for rows, e in engines.items():
        assert prefill(e, prompt, None, resume=kept[rows]) == first
        assert serial_decode(e, first, REPLY, None, stop_eos=False).tokens == reply, rows


def _device_bytes(obj, seen: set) -> int:
    """Device bytes of the tensors ``obj`` holds (its attributes, lists and dicts), each storage once."""

    if isinstance(obj, torch.Tensor):
        if obj.device.type != "cuda" or obj.untyped_storage().data_ptr() in seen:
            return 0
        seen.add(obj.untyped_storage().data_ptr())
        return obj.untyped_storage().nbytes()
    if isinstance(obj, (list, tuple)):
        return sum(_device_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(_device_bytes(x, seen) for x in obj.values())
    if hasattr(obj, "__dict__") and type(obj).__module__.startswith("tensorfold"):
        return sum(_device_bytes(x, seen) for x in vars(obj).values())
    return 0


def test_a_4096_row_piece_stays_inside_the_planned_workspace(model):
    """The piece's buffers (less the attention scratch, which the cache geometry counts) and the peak of the
    temporaries a 4,097-row prefill allocates beside them fit ``choose``'s workspace with MTP drafts."""

    from tensorfold.cuda.capacity import config
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import IDLE_ROWS, choose

    if getattr(model.cfg, "hidden", 0) != 2560:
        pytest.skip("the plan serves FP4-Local's shape only")
    text = config(model.root)
    plan = {"serving_peak_bytes_estimate": 0, "startup_peak_bytes_estimate": 0, "budget_bytes": 1 << 50,
            "cache_workspace_bytes_estimate": 0, "mapped_table_bytes": 0}
    rows, workspace = choose(plan, text, (12, 1), mtp=True)
    assert rows == IDLE_ROWS == 4096
    e = Engine(model, capacity=CAPACITY, max_rows=8, prefill_rows=rows, kv_dtype="int8")
    buffers = _device_bytes(e.pbuf, set()) - _device_bytes(e.pbuf.attn, set())
    prefill(e, _prompt(rows + 1, model.cfg.vocab), None)                 # compile and load every prompt kernel
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    prefill(e, _prompt(rows + 1, model.cfg.vocab, seed=12), None)
    torch.cuda.synchronize()
    temporaries = torch.cuda.max_memory_allocated() - held
    print(f"4,096-row piece: buffers {buffers / 2**30:.3f} GiB, temporaries {temporaries / 2**30:.3f} GiB, "
          f"planned {workspace / 2**30:.3f} GiB")
    assert buffers + temporaries <= workspace
