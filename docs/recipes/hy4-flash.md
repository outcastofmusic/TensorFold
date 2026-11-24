# Hy4-flash

The CUDA family `src/tensorfold/families/hy4_flash/` serves Tencent's Hy4-flash (model_type `hy_v4_flash`, 220B
parameters of which 63.5B are Engram tables, 11.3B active a token) on one GB10, from Avifenesh's NVFP4 pack. Tencent's
own FP8 release (287.1 GB, 148.6 GB of FP8 routed experts) does not fit one GB10.

## License

The weights are under the Tencent Hy License 1.0 (the checkpoint's `LICENSE`): no territory exclusion; extra
commercial terms from 100 million monthly active users. The NVFP4 pack is a third-party quantization of
`tencent/Hy4-flash` under the same license.

## Run

```bash
./prepare.sh hy4-flash         # Avifenesh/Hy4-flash-NVFP4 at f9eac06c into ~/models/Hy4-flash-NVFP4 (161.1 GB)
./run.sh hy4-flash             # served as hy4-flash on 127.0.0.1:30000, or through ./run.sh swap
```

run.sh serves a 65,536-token context at temperature 0.9, top_p 1.0. DFly2 drafts with the NVFP4 draft head and
adaptive depth at floor 0.15, keyed by the request's sampler, and copies draft too (`HY4_DRAFT_HEAD`, `HY4_ADAPTIVE`,
`HY4_FLOOR`, `HY4_KEYED`, `CONTEXT`). It runs alone: 89.7 GiB of weights, with 20 GiB available after load and
2.4 GiB of kept prompt states. At 131,072 tokens 17 GiB stays available but no prompt state fits, so no turn
reuses its prefix.

## Speed and quality on one GB10 (21 Nov, served, thinking off)

| | chat | code | prose | json | rewrite |
| --- | ---: | ---: | ---: | ---: | ---: |
| greedy tok/s | 21.7 | 26.4 | 17.4 | 35.7 | 64.3 |
| T 0.9, top_p 1.0 tok/s | 19.1-19.9 | 29.0-29.9 | 15.8-17.2 | 39.3-40.6 | 64.9-67.1 |

The decode and prefill rows were measured at 131,072 tokens. Prefill runs at 601-638 tok/s on novel text and
764-806 on repeated filler (4k-32k tokens). TrueScore 76x2 scores 91.7 against GLM-5.3 Flash's 93.1 and Qwen3.8
Flash Next's 88.3. tool-eval-bench 69x2 scores 86-88 against 93 and 92. The measurements are in
`notes/tf-runs/hy4-flash-1121`.

## The checkpoint

| | |
| --- | --- |
| Layers | 42: 12 sparse MLA layers (0, 1, 5, 9, ..., 41; 32 heads) and 30 sliding MLA layers (64 heads, the last 512 keys) |
| Attention | MLA: q_lora 1,024, a 512-wide normed latent plus 64 rotated channels cached a token, nope 192, v 256, score scale 1/16, a learned sink a head, values gated by sigmoid(linear_gate) |
| Indexer | sparse layers past 2,048 keys: groups of 4 token keys pooled by softmax gates, 16 heads of 128; a row keeps its 511 best closed groups and its own |
| Streams | identity hyper-connections over 4 streams (sigmoid pre and post gates, no mixing matrix) |
| Engram | before layers 1, 11, 21, 31: hashed 2- and 3-grams, 8 heads an order, about 165 million 96-wide rows a table; FP8 with a scale a row, read from disk |
| FFN | layer 0 dense SwiGLU of 13,824; layers 1-41 a latent MoE (3,072 to 1,536), 512 experts of 1,536, top 16, sigmoid with a correction bias, weights normalized and scaled by 3.998, plus a shared expert |
| Head | 4-gate stream collapse, RMSNorm, an fp32 head over 120,832 tokens; end token `<｜hy_end:opensource｜>` (120025) |
| Drafter | DFly2: 5 MLA layers reading the target's caches at sparse layers 25-41, a block of 6 rows a round |

## Formats on the GPU

| Tensors | Stored | Kernel |
| --- | --- | --- |
| Routed experts | NVFP4 (e4m3 scale per 16, fp32 per expert) | the grouped NVFP4 kernel |
| q_b, o_proj, linear_gate (layers 1-41, drafter) | FP8 e4m3, fp32 scale per 128x128 | the lane matmul (`Fp8BlockLinear`) |
| Everything else in the trunk | bf16 | `b16` |
| Engram tables | FP8 rows on disk, scales in host memory | 64 reads in flight; a prompt chunk's rows read during the previous chunk |

## Text layer

The tokenizer spells its control tokens with `:opensource` (`<think:opensource>`, `<tool_call:opensource>`). The app
serves a view of the text files with those spellings stripped (same ids), so the server's `</think>` split and GLM's
`<tool_call>NAME<arg_key>..` parser apply; the reply's `<tool_calls>` wrapper ids are dropped. The template takes
`reasoning_effort` `high` or `no_think` only: the app renders `no_think` when a request turns thinking off and `high`
for any effort with thinking on.

## Correctness

Checked against transformers PR #50903 (`modeling_hy_v4_flash`, head b865850d): a tiny checkpoint in the release's
formats (`tests/hy4_flash_tiny.py`, goldens in `tests/fixtures/hy4_flash`), and the real pack layer by layer
(`tools/hy4_golden.py`, `tools/hy4_trace.py`). On the real pack, teacher-forced: KL 0.0014 (chat), 0.0025 (code),
0.0078 (a 3,113-token document) against transformers' bf16 run, greedy steps 32/32, 32/32, 30/32.

Two bugs the reference found (21 Nov): Engram's key and query norms need fp32's epsilon (torch's nn.RMSNorm uses it on
bf16 rows; bf16's 2^-7 sat above the streams' mean square and put layer 1 69% off), and every reduction a decode row
needs runs on per-row Triton kernels, so a verify window's rows equal one-row steps (`tools/hy4_rows.py`, widths 2-16,
below and past 2,048 keys).

## Gates

- Drafted replies equal serial decoding's, greedy and sampled (DFly2 and copies; `tools/hy4_probe.py`).
- Graph replay equals eager.
- Engram epsilon: `TF_HY4_ENGRAM_EPS` (fp32's). Draft head slice: `TF_HY4_DRAFT_VOCAB` (32,768 ids plus the specials;
  0: the whole head); verification reads the whole head.
