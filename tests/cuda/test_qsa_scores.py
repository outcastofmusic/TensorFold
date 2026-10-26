"""qsa_scores.cu against the Triton kernels: qsa_scores.cu writes `_scores`' bytes at the same positions, the tiled
select from 8,192 blocks lists `_select`'s blocks (lower ids first among equal scores), and prompts prefill to the same
bits as on the Triton kernels (``qsa.fast`` patched off, as TF_FLASHNEXT_QSA=torch served until 25 Oct).

Shapes are the FP4-Local and EXL3 3.05bpw indexer's (4 heads of 128 dims, 4-key blocks, a 2,048-key budget) at the
262,144-key window `--context 262144` admits; prompt blocks are ATT_ROWS (256) rows, as forward.py launches them."""

import pytest
import torch
import triton

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402
from test_flashnext_prompt_cache import _assert_same_state  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import attention as att  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qsa  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import ATT_ROWS  # noqa: E402

DEV = "cuda"
CAP, HI, DI, RATIO, BUDGET = 262151, 4, 128, 4, 2048
TOP = BUDGET // RATIO
ENDS = [2_048 + 100, 8_192, 32_768, 32_768 + 300, 131_072 + 7, CAP]     # a block's last row's end: 2k .. 256k keys
SENTINEL = 0xA7


def _bf16(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int16).view(torch.bfloat16)


def _inputs(kind: str, blocks: int, rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """pooled [blocks, DI] and iq [rows, HI, DI] bf16: ``normal`` values, ``ties`` (blocks repeat 7 distinct keys, so
    hundreds of blocks share a score at the cut), or ``special`` (inf, -inf, NaN, -0 and subnormal codes)."""

    g = torch.Generator(device=DEV).manual_seed(seed)
    pooled = (torch.randn((blocks, DI), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    iq = (torch.randn((rows, HI, DI), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    if kind == "ties":
        pooled = pooled[torch.arange(blocks, device=DEV) % 7].contiguous()
        iq = torch.round(iq.float() * 4).div(4).to(torch.bfloat16)
    elif kind == "special":
        codes = torch.tensor([0x7F80, 0xFF80, 0x7FC0, 0x8000, 0x0001, 0x8003], device=DEV)
        flat = pooled.view(-1)
        at = torch.randint(0, flat.numel(), (max(1, flat.numel() // 997),), generator=g, device=DEV)
        flat[at] = _bf16(codes[torch.arange(at.numel(), device=DEV) % codes.numel()])
        qf = iq.view(-1)
        qat = torch.randint(0, qf.numel(), (max(1, qf.numel() // 89),), generator=g, device=DEV)
        qf[qat] = _bf16(codes[3:][torch.arange(qat.numel(), device=DEV) % 3])       # -0 and subnormals in queries
    return pooled, iq


def _scratch(rows: int, fast: bool) -> att.AttnScratch:
    sc = att.AttnScratch(rows, 24, 256, CAP, DEV, budget=BUDGET, ratio=RATIO, fast=fast)
    for t in (sc.scores, sc.ids, sc.nk, sc.sparse):
        t.view(torch.uint8).fill_(SENTINEL)             # every position either path leaves alone stays this
    return sc


def _bytes(sc: att.AttnScratch) -> list[torch.Tensor]:
    return [t.view(torch.uint8).clone() for t in (sc.scores, sc.ids, sc.nk, sc.sparse)]


def _both(iq, pooled, pos, rows, end):
    """(torch path, cuda path) scratch bytes after ``qsa_rows`` of rows P0 .. P0 + rows - 1 with context ``end``."""

    out = []
    for fast in (False, True):
        sc = _scratch(rows, fast)
        att.qsa_rows(iq, pooled, pos, sc, rows, context=end)
        out.append(_bytes(sc))
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("kind", ["normal", "ties", "special"])
@pytest.mark.parametrize("end", ENDS)
@pytest.mark.parametrize("rows", [ATT_ROWS, 33, 1])
def test_scores_and_lists_match_the_torch_path_byte_for_byte(kind, end, rows):
    """Every score word, key list, length and flag, with positions neither path writes (rows at or under the budget,
    blocks past a row's end) still the sentinel in both. At 2k the block straddles the budget: dense rows first."""

    nb = -(-CAP // RATIO)
    pooled, iq_all = _inputs(kind, nb, rows + 16, seed=end + rows)
    iq = iq_all[16:16 + rows]                              # a row view inside a larger buffer, as forward.py passes
    pos = torch.tensor([end - rows], dtype=torch.int32, device=DEV)
    plain, fast = _both(iq, pooled, pos, rows, end)
    names = ("scores", "ids", "nk", "sparse")
    for name, a, b in zip(names, plain, fast):
        assert torch.equal(a, b), (name, kind, end, rows)
    if end == ENDS[0] and rows == ATT_ROWS:
        sparse = plain[3].view(torch.int32)
        assert sparse[0] == 0 and sparse[-1] == 1           # the block holds both dense and sparse rows


@pytest.mark.parametrize("end", [32_768 + 4, 65_536, 131_072])
@pytest.mark.parametrize("pattern", ["tied", "flat", "neg_inf", "nan"])
def test_the_tiled_select_lists_select_s_blocks_between_8192_blocks_and_the_registers(end, pattern):
    """Rows of 8,192 to 32,768 blocks, where the fast scratch runs `_select_tiles` and the torch one `_select`: the same
    lists on tied, all-equal, -inf-masked and NaN scores, and the reference's (TOP best, lower ids first, block
    order)."""

    rows = ATT_ROWS
    blocks = -(-end // RATIO)
    width = triton.next_power_of_2(blocks)
    assert qsa.tiled(blocks, width, True, att.SELECT_REGS) and not qsa.tiled(blocks, width, False, att.SELECT_REGS)
    g = torch.Generator(device=DEV).manual_seed(end)
    base = torch.randn((rows, -(-CAP // RATIO)), generator=g, device=DEV)
    if pattern == "tied":
        scores = torch.floor(base * 2) / 2
    elif pattern == "flat":
        scores = torch.zeros_like(base)
    elif pattern == "neg_inf":                             # most blocks masked off: under TOP finite ones in some
        scores = torch.where(base > 1.5, base, torch.full_like(base, float("-inf")))
    else:
        scores = torch.where(base > 2.0, torch.full_like(base, float("nan")), base)
    pos = torch.tensor([end - rows], dtype=torch.int32, device=DEV)
    lists = []
    for fast in (False, True):
        sc = _scratch(rows, fast)
        sc.scores.copy_(scores)
        att._launch_select(sc, pos, rows, blocks)
        lists.append(_bytes(sc)[1:])
    for a, b in zip(*lists):
        assert torch.equal(a, b), (pattern, end)
    if pattern in ("tied", "neg_inf"):
        ids = lists[0][0].view(torch.int32).reshape(rows, -1)
        for r in (0, rows - 1):
            c = (end - rows + r + 1) // RATIO
            s = scores[r, :c].tolist()
            order = sorted(range(c), key=lambda b: (-s[b], b))[:TOP]
            assert ids[r, :TOP * RATIO:RATIO].tolist() == [RATIO * b for b in sorted(order)], (pattern, r)


def _prompt(n: int, seed: int) -> list[int]:
    return torch.randint(1, V, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@pytest.fixture(scope="module")
def w():
    return _model()


SERVED_FAST = qsa.fast


def _engine(w, monkeypatch, flag: str, capacity: int) -> Engine:
    """``flag`` torch: the prompt buffers on the Triton kernels, else as served."""

    monkeypatch.setattr(qsa, "fast", (lambda *a: False) if flag == "torch" else SERVED_FAST)
    e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=2048)
    assert e.pbuf.attn.fast is (flag == "cuda") and not e.buf.attn.fast
    return e


def test_a_32k_prompt_prefills_and_decodes_the_same_on_either_kernel(w, monkeypatch):
    """33,000 tokens (8,250 blocks at the end: the fast scratch's tiled select runs) and 24 greedy tokens: the same
    first token, state bits and replies on the Triton kernels and qsa_scores.cu."""

    prompt = _prompt(33_000, seed=32)
    runs = {}
    for flag in ("torch", "cuda"):
        e = _engine(w, monkeypatch, flag, 34_816)
        first = prefill(e, prompt, None)
        runs[flag] = (e, first, serial_decode(e, first, 24, None, stop_eos=False).tokens)
    (a, fa, ta), (b, fb, tb) = runs["torch"], runs["cuda"]
    assert fa == fb and ta == tb
    _assert_same_state(a.st, b.st)


@pytest.mark.parametrize("base, extra", [(9_000, 700), (33_000, 1_500)])
def test_a_resumed_prompt_equals_a_fresh_one_on_the_cuda_kernels(w, monkeypatch, base, extra):
    """A kept prompt end resumed with more tokens leaves the state and the first token a fresh prefill leaves."""

    head = _prompt(base, seed=base)
    full = head + _prompt(extra, seed=extra)
    e = _engine(w, monkeypatch, "cuda", 36_864)
    keeps = {}
    prefill(e, head + [7], None, stops=[base], keep=lambda p, snap, tail: keeps.__setitem__(p, (snap, tail)))
    snap, tail = keeps[base]
    got = prefill(e, full, None, resume={"state": snap, "tail": tail})
    fresh = _engine(w, monkeypatch, "cuda", 36_864)
    want = prefill(fresh, full, None)
    assert got == want
    _assert_same_state(e.st, fresh.st)
    assert serial_decode(e, got, 12, None, stop_eos=False).tokens == \
           serial_decode(fresh, want, 12, None, stop_eos=False).tokens
