"""Reuse expert weights in wider idle prompt pieces only when the admitted window leaves enough room."""

from tensorfold.cuda.geometry import PREFILL_ROWS, indexed_prompt_bytes

IDLE_ROWS = 4096


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
           vision: bool = False, fp8: bool = False, mtp: bool = False) -> tuple[int, int]:
    """Keep the baseline window; spend spare budget on the measured GB10 shape's prompt workspace (``mtp``: the
    engine loads the MTP head, whose prompt rows an NVFP4 checkpoint runs on bf16 linears)."""

    shape = tuple(text.get(k) for k in ("hidden_size", "num_experts", "num_experts_per_tok",
                                       "moe_intermediate_size", "hc_count", "hc_lowrank"))
    kind = storage(text.get("_quantization") or {})
    if capability != (12, 1) or world != 1 or vision or fp8 or shape != (2560, 512, 10, 640, 4, 320) or not kind:
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
