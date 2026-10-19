# LFM2.5-24B-A2B

The CUDA family `src/tensorfold/families/lfm2_moe/` serves Liquid AI's LFM2.5-24B-A2B (model_type `lfm2_moe`, 23.97B
parameters, 2.45B active a token) on one GPU, from `LiquidAI/LFM2.5-24B-A2B-FP8` or the bf16 export. The fork runs it
as a tool-routing and sub-agent model beside the larger ones in `swap.py`'s set.

## License

The weights are under the LFM Open License v1.0 (the checkpoint's `LICENSE`). Its section 5 licenses commercial use
only to an entity whose annual revenue is under US$10,000,000; a qualified non-profit's research use is exempt.
Redistribution must carry the license. `tests/fixtures/lfm2_moe/` holds the checkpoint's chat template, tokenizer
config and config files for the CPU tests, with the license beside them.

## Run

```bash
./prepare.sh lfm2-24b          # LiquidAI/LFM2.5-24B-A2B-FP8 at f54bf395 into ~/models/LFM2.5-24B-A2B-FP8 (24.75 GB)
./run.sh lfm2-24b              # served as lfm2.5-24b-a2b on 127.0.0.1:30000, or through ./run.sh swap
tensorfold serve ~/models/LFM2.5-24B-A2B-FP8 --name lfm2.5-24b-a2b --context 131072 --top-p 1.0
```

The server reads Liquid's sampling from `generation_config.json`: temperature 0.1 and top_k 50. `run.sh` adds
`--top-p 1.0`, where the server's default is 0.95. The card's repetition penalty of 1.05 is not applied: the CUDA
engines' keyed sampler has no penalty term. Seeds draw as on the fork's other CUDA families.

## The checkpoint

| | |
| --- | --- |
| Layers | 40: 30 gated short convs, 10 GQA attention layers (2, 6, ..., 38) |
| Widths | hidden 2048; 32 query heads and 8 KV heads of 64; vocabulary 128,000 (125,016 tokens used) |
| FFN | layers 0-1 a dense SwiGLU of 11,776; layers 2-39 64 experts of 1,536, top 4 |
| Routing | sigmoid scores plus `expert_bias` pick; the picks' sigmoids, normalized to sum 1 (+1e-6), weight them |
| Conv | in_proj to B, C, x; a causal depthwise conv of 3 taps over B * x (no bias); times C; out_proj |
| Attention | per-head RMSNorm of q and k, then RoPE (theta 1e7, rotate-half); no sliding window |
| Head | tied to the embedding; final norm `embedding_norm`; no embedding or logit scale |
| Context | 131,072 |
| FP8 export | compressed-tensors 0.19.0, `FP8_BLOCK`: e4m3 weights with a bf16 `weight_scale` per 128x128 block |
| bf16 in the FP8 export | embedding, routers, `expert_bias` (fp32), norms, and every conv layer's in_proj, out_proj and conv |

The FP8 export quantizes 7,342 matrices: attention, the dense FFN and 7,296 expert matrices. Every one of their
1,416,064 blocks has its largest code at 448, as llm-compressor's `absmax / 448` scale makes it, with no NaN code
(read on the CPU, 2026-10-17). compressed-tensors' `weight_scale` is the same multiplier as DeepSeek's
`weight_scale_inv`, so the fork's `Fp8BlockLinear` reads it unchanged.

## CUDA engine

One stream: requests take turns, and there are no drafts. A prefill keeps the conversation's state at the second
message's start, at the last assistant turn's start and at the prompt end less one token (`tensorfold.cuda.markers`):
the conv states (240 KiB) and the attention rows so far (20 KiB a token). A request resumes from the longest kept
prefix of its prompt and prefills only the rest. Up to `TF_LFM2_CACHE_ENTRIES` states (default 8) are kept in at
most `TF_LFM2_CACHE_GIB` (default 4, capped by free memory); past either cap a conversation's earlier turn goes
first, then the oldest state. A resumed prompt starts its chunks at the kept length, so its bf16 rows can round apart
from a fresh prefill's, within the golden bounds below. `draft: false` prefills afresh and keeps nothing.

| Part | Kernel |
| --- | --- |
| Attention q/k/v (stacked), out, dense FFN | `Fp8BlockLinear`, the lane matmul over the stored e4m3 bytes |
| Conv in/out projections, tied head | `b16` (bf16) |
| Routed experts, decode (up to 16 rows) | `cuda/experts.py`'s Triton kernels: a program a (row, pick) and 32 outputs, fp32 sums a 128-input block, times the block's scale |
| Routed experts, prompt chunks | the decode kernels up to 16 rows (`PAIRS_ROWS`), then `dot`: pairs sorted by expert into 16-, 32- or 64-row tiles on the device, each tile reading its expert's e4m3 bytes once as bf16 on `tl.dot`, fp32 sums a 128-input block, times the block's scale. `TF_LFM2_PROMPT_KERNEL=fp32` keeps the previous path (each used expert decoded to fp32, its rows multiplied in fp32) as the reference; `pairs` and `dot` force one path |
| Norms | Qwen's row-invariant `add_rmsnorm` |
| Attention | `prefill_attention` for prompts; Nemotron-H's 512-key chunked kernel for decode |
| Sampling | Nemotron-H's keyed sampler: (seed, position, logits) decides a token |

Activations round as transformers' bf16 forward does. The conv's state is the last two products B * x, bf16.

A one-row decode step replays a CUDA graph captured on the first step. The step reads its position (RoPE, the KV row
it writes, attention's length) from a device tensor and its token from a static buffer, so one graph serves every
position, and a replay gives the eager step's bits. `TF_LFM2_GRAPHS=0` runs every step eagerly.

A decode call runs the gated conv after its in_proj and the MoE combine as one Triton kernel each (`cuda/fused.py`),
computes RoPE's angles and the KV rows once a call, and skips route's int32 id copy and its multiply by a scaling of
1.0. Counted on the CPU dispatch (2026-10-19), a one-row step drops from about 2,040 launches to about 1,020. The
kernels round where torch's kernels round and launch with Triton's fma contraction off, so the bits stay the same.
`TF_LFM2_DECODE_FUSE=0` restores the torch chains; the GPU byte comparison has not run yet.

## Memory and speed

| Per stream | |
| --- | ---: |
| KV cache a token | 20 KiB (10 layers x 2 x 8 heads x 64 x bf16) |
| KV at 131,072 tokens | 2.50 GiB |
| Conv state | 240 KiB (30 layers x 2 x 2048 x bf16) |
| Weights on the GPU | 24.75 GB, plus 6% scale overhead on the 0.25 GB of stacked FP8 linears |

| Read a decoded token | GB |
| --- | ---: |
| Routed experts (4 of 64, 38 layers) | 1.435 |
| Conv projections (bf16) | 1.007 |
| Tied head | 0.524 |
| Attention and dense FFN | 0.250 |
| Total | 3.226 |

At the GB10's 273 GB/s that bounds one stream at 85 tok/s, and at 220 GB/s, which serial decodes reach in practice,
at 68. The bf16 conv projections read as many bytes as the routed experts; quantizing them to FP8 at load would cut
0.5 GB a token but changes the model's arithmetic.

## Thinking and tools

The template opens `<think>` on every reply and has no switch. The family's app keeps thinking on whatever a request
sends, so `reasoning_content` always holds the reasoning and `content` the answer. Earlier turns' reasoning is
dropped from the prompt unless `chat_template_kwargs` sets `preserve_thinking`.

Tools render as a JSON list in the system turn. Calls are Pythonic, `<|tool_call_start|>[get_weather(city='Paris')]
<|tool_call_end|>`, parsed by `tensorfold.server.pythonic_calls` (Python's parser, JSON's true/false/null, names
with hyphens, positional arguments by schema order). Results go back under the `tool` role. A call the model
leaves unclosed at its end token is closed when the closed block parses.

## Exactness against the fp32 reference

`reference.py` is an fp32 forward from the checkpoint with every FP8 block decoded exactly. On the real checkpoint it
continues the three golden prompts coherently (`tools/lfm2_moe_golden.py`, 2026-10-17: 52 s for the chat prompt on
the CPU). The engine's own code, run on the CPU with the full checkpoint, differs from it by at most 2.31 in a top-20
logit over the chat golden's 32 teacher-forced steps (median 0.57) and 0.91 on the tool prompt (median 0.19). Nearly
all of that comes from expert picks that bf16 rounding flips: 20 picks on the chat prompt sat within 0.0025 of the
next expert's score. Run on the engine's own picks, the reference differs by at most 0.23 (median 0.11). The engine's
greedy token differed only at steps whose top two reference logits were within 0.2.

## Tests

| Where | What |
| --- | --- |
| `tests/test_lfm2_moe_reference.py` | config refusals; the fp32 reference equals transformers' `modeling_lfm2_moe.py` to 6.1e-5 of 164 on a tiny FP8 checkpoint; chunked and stepped prompts |
| `tests/test_lfm2_moe_engine_cpu.py` | the CUDA engine's code on the CPU against the reference on the engine's own expert picks; generation stops; keyed sampling; the expert kernels in Triton's interpreter |
| `tests/test_lfm2_moe_tools.py` | the template against transformers' renderer; Pythonic calls round trip |
| `tests/cuda/test_lfm2_moe.py` | kernels against their fp32 definitions; the 24B-A2B against `tools/lfm2_moe_golden.py`'s files; decode speed |
| `tools/lfm2_moe_smoke.sh` | the served model on :31005: chat, tool call, tool result, stream |
