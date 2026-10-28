"""Reuse expert weights in wider idle prompt pieces only when the admitted window leaves enough room."""

import os
from contextlib import contextmanager

from tensorfold.cuda.geometry import PREFILL_ROWS, indexed_prompt_bytes

IDLE_ROWS = 4096
LONG_PIECE_ENV = "TENSORFOLD_LONG_PIECE_TOKENS"
LONG_PIECE_TOKENS = 8192   # the threshold the 25 Oct sweep picked
_long_from = 0          # the installed threshold (use): 0 keeps every idle piece at its buffers' rows
TAIL_ENV = "TENSORFOLD_PROMPT_TAIL"
TAIL_ROWS = 512         # a last piece shorter than this splits the last two evenly (MiaAI-Lab's recipe 0013)
TAIL_ALIGN = 16
_even_tail = False      # the installed rule (use_tail)


def long_piece_tokens() -> int:
    """TENSORFOLD_LONG_PIECE_TOKENS (unset or empty: ``LONG_PIECE_TOKENS``): an NVFP4 checkpoint's idle pieces take
    4,096 rows while at least this many prompt rows remain to fill, else 2,048. 0 keeps 2,048 rows, the width before
    the sweep. On 25 Oct 2026 8,192 ran cold prefill 1.05-1.13x as fast at 32k-131k tokens and 0.99x at 4k; 16,384
    and 32,768 gained less (tf-runs/nvfp4-long-pieces). A value that is not a whole number stops the start."""

    value = os.environ.get(LONG_PIECE_ENV, "").strip()
    if not value:
        return LONG_PIECE_TOKENS
    if not value.isdecimal():
        raise ValueError(f"{LONG_PIECE_ENV}: a token count of 0 or more, not {value!r}")
    return int(value)


def prompt_tail() -> bool:
    """TENSORFOLD_PROMPT_TAIL=1 evens out a prompt's last two idle pieces (``even_tail``); unset, empty or 0 keeps
    the plain cut. Off until the prefill-gap-1028 M3 A/B measures it. Any other value stops the start."""

    value = os.environ.get(TAIL_ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{TAIL_ENV}: 0 or 1, not {value!r}")
    return value == "1"


def use_tail(on: bool) -> bool:
    """Install the even-tail rule ``piece_rows`` applies; returns the previous setting."""

    global _even_tail
    previous, _even_tail = _even_tail, bool(on)
    return previous


@contextmanager
def even_tails(on: bool):
    """``use_tail(on)`` for the block, then the previous setting again."""

    previous = use_tail(on)
    try:
        yield
    finally:
        use_tail(previous)


def even_tail(width: int, left: int) -> int:
    """The next piece's rows when ``left`` rows remain and pieces take ``width``: once a full piece would leave fewer
    than TAIL_ROWS, the last two pieces split ``left`` evenly, the first rounded down to a multiple of TAIL_ALIGN.
    sparkDash prompts end 39-57 rows past a multiple of 2,048, and that short pass reads almost every expert's
    weights for a few rows (about 150 ms a prompt on EXL3, tf-runs/prefill-gap-1028). 4,135 rows at 2,048 a piece:
    2,048, then 1,040 and 1,047."""

    if not 0 < left - width < TAIL_ROWS:
        return width
    half = left // 2 // TAIL_ALIGN * TAIL_ALIGN
    return half if half and left - half <= width else width


def use(tokens: int) -> int:
    """Install ``tokens`` as the threshold ``piece_rows`` reads; returns the previous one."""

    global _long_from
    previous, _long_from = _long_from, int(tokens)
    return previous


@contextmanager
def long_pieces(tokens: int):
    """``use(tokens)`` for the block, then the previous threshold again."""

    previous = use(tokens)
    try:
        yield
    finally:
        use(previous)


def piece_rows(rows: int, left: int) -> int:
    """An idle piece's rows when ``left`` prompt rows remain to fill: the buffers' ``rows``, or PREFILL_ROWS while
    fewer than the installed threshold remain, then evened by ``even_tail`` near the end while TENSORFOLD_PROMPT_TAIL
    is installed. A row's bits do not depend on its piece, so neither switch changes them."""

    width = rows if not _long_from or rows <= PREFILL_ROWS or left >= _long_from else PREFILL_ROWS
    return even_tail(width, left) if _even_tail else width


def warm_widths(rows: int) -> list[int]:
    """The piece widths a warm-up prefills so no request compiles a prompt kernel: ``rows``, and PREFILL_ROWS too
    while a threshold switches between them."""

    return [rows, PREFILL_ROWS] if _long_from and rows > PREFILL_ROWS else [rows]


@contextmanager
def pieces_of(width: int, rows: int):
    """Every idle piece ``width`` rows for the block (``rows``: the buffers'): a warm-up's fixed widths, its tail
    left uneven so the warm prompt still runs one full piece."""

    with long_pieces(0 if width == rows else 1 << 62), even_tails(False):   # no prompt reaches 2^62 rows
        yield


def storage(quant: dict) -> str | None:
    """The checkpoint storage wide idle pieces serve: "mlx" (affine 4-bit), "nvfp4" (ModelOpt with NVFP4 routed
    experts and every other quantized language-model linear MXFP8, as FP4-Local), or None."""

    if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) == (4, 32, "affine"):
        return "mlx"
    if str(quant.get("quant_method") or "").lower() != "modelopt":
        return None
    experts, dense = set(), set()
    for name, layer in (quant.get("quantized_layers") or {}).items():
        if not name.startswith(("model.language_model.layers.", "model.layers.")) or ".ngram_embedding" in name:
            continue                                  # the vision tower, the MTP head and the n-gram tables
        algo = str(layer.get("quant_algo") or "").upper()
        (experts if name.endswith(".mlp.experts") or ".mlp.experts." in name else dense).add(algo)
    return "nvfp4" if experts == {"NVFP4"} and dense == {"MXFP8"} else None


def nvfp4_temporary_row(text: dict, mtp: bool) -> int:
    """Bytes a prompt row adds to the NVFP4 path's widest temporary: the fp32 K-slice partials of its bf16 linears
    (``bf16.split_k``; FP4 tables: ``nvfp4.split_for``). One matmul's partials are freed before the next allocates.
    The MLX path has none: its prompt matmuls take K whole. The MTP head's fc_hidden reads each row's streams, so
    with drafts it is the widest: 4 slices x 4 streams x 2,560 x 4 bytes = 160 KiB a row."""

    from .bf16 import split_k
    from .nvfp4 import split_for

    d, s, low = int(text["hidden_size"]), int(text["hc_count"]), int(text["hc_lowrank"])
    ple, shared = int(text.get("ple_embed_dim") or d), int(text["shared_expert_intermediate_size"])
    attn = int(text["num_attention_heads"]) * int(text.get("head_dim") or d // int(text["num_attention_heads"]))
    # a hyper-connection's down and inject: their K slices and the fp32 sums the activation reads
    widest = (low + s) * 4 * (split_k(low + s, s * d) + 1)
    faces = [(1, s * d, ple, split_k), (1, d, ple, split_k)]           # (rows a prompt row feeds, N, K): n-gram
    if mtp:      # the head's fc_embedding, fc_hidden, attention output and bf16 shared expert (on FP4 tables)
        faces += [(1, d, d, split_k), (s, d, d, split_k), (1, d, attn, split_k),
                  (1, 2 * shared, d, split_for), (1, d, shared, split_for)]
    for rows, n, k, split in faces:
        slices = split(n, k)
        widest = max(widest, rows * slices * n * 4 if slices > 1 else 0)
    return widest


def choose(receipt: dict, text: dict, capability: tuple[int, int], *, world: int = 1,
           vision: bool = False, fp8: bool = False, mtp: bool = False, long_from: int = 0) -> tuple[int, int]:
    """Keep the baseline window; spend spare budget on the measured GB10 shape's prompt workspace (``mtp``: the
    engine loads the MTP head, whose prompt rows an NVFP4 checkpoint runs on bf16 linears). An NVFP4 checkpoint
    widens only with ``long_from`` (TENSORFOLD_LONG_PIECE_TOKENS) above 0, and then budgets the 4,096-row piece at
    any threshold: a prompt long enough to take one must fit the window admitted at the start."""

    shape = tuple(text.get(k) for k in ("hidden_size", "num_experts", "num_experts_per_tok",
                                       "moe_intermediate_size", "hc_count", "hc_lowrank"))
    kind = storage(text.get("_quantization") or {})
    if (capability != (12, 1) or world != 1 or vision or fp8 or shape != (2560, 512, 10, 640, 4, 320) or not kind
            or (kind == "nvfp4" and not long_from)):
        return PREFILL_ROWS, 0
    temporary = nvfp4_temporary_row(text, mtp) if kind == "nvfp4" else 0
    extra = indexed_prompt_bytes(text, IDLE_ROWS - PREFILL_ROWS) + (IDLE_ROWS - PREFILL_ROWS) * temporary
    serving = receipt["serving_peak_bytes_estimate"] + extra
    total = max(receipt["startup_peak_bytes_estimate"], serving)
    if total > receipt["budget_bytes"]:
        return PREFILL_ROWS, 0
    workspace = indexed_prompt_bytes(text, IDLE_ROWS) + IDLE_ROWS * temporary
    receipt.update(prefill_rows=IDLE_ROWS, prompt_workspace_bytes_estimate=workspace,
                   cache_workspace_bytes_estimate=receipt["cache_workspace_bytes_estimate"] + extra,
                   serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                   full_mapped_working_set_bytes_estimate=total + receipt["mapped_table_bytes"])
    return IDLE_ROWS, workspace


def pass_limit(rows: int, live: bool, share: float, round_s: float | None, row_s: float | None,
               minimum: int = 128) -> int:
    """Live replies retain the released pass bound; an idle decoder can use its full prompt workspace."""

    limit = min(rows, PREFILL_ROWS) if live else rows
    if not live or share <= 0 or not round_s or not row_s:
        return limit
    chosen = int(round_s / (share * row_s)) // 64 * 64
    return min(limit, max(minimum, chosen))
