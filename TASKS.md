# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

## In progress

| Task | State |
|---|---|
| TrueScore 76x2 and planning x4 on Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt | FP4-Local 91.0 / planning 89.8, EXL3 87.8 / 89.0, GLM 93.0 / planning running, Bolt next (`tf-runs/truescore-queue.sh`, 17 Oct) |
| Research Liquid AI's LFM2.5 series for agentic work | Background agent: lineup, Spark fit, upstream TensorFold support, agentic benchmarks |
| Qwen disk states, and a persist endpoint `swap.py` calls before it stops a model | Background agent on `feat/qwen-disk-states`; its GPU tests run after the benchmarks |
| GLM regenerate check of the branch-point rule (#735), base against ports | Chained after the benchmarks (`tf-runs/glm-regen.sh`) |

## To do

| Task | Notes |
|---|---|
| Restore production Qwen | `POST :30000/swap/qwen3.8-flash-next` after the GLM regenerate check. Production has been down since 17 Oct 07:24 UTC |
| Decide the swap set and update `swap.py` / `run.sh` | Swap between Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt, from the benchmark results |
| Merge the branches into `fork/dev` | `feat/upstream-ports`, `feat/qwen-scan-marks`, `feat/qwen-fp4-csf`, `feat/bolt`, then the full CPU suite and the GPU tests |
| Bolt: upstream Nemotron ideas | Prompt reuse (1.0.5), concurrent rounds and batched drafting (1.0.9), 4,096-row prompt passes. Bolt runs one sequence and prefills every request in full |
| Bolt: adaptive draft depth | LiLiCorr at a fixed 7 drafts runs prose at 20 to 22 tok/s against 29 serial |
| Nemotron 3.5 Super VL (120B-A12B) as a swap candidate | Upstream 1.0.12 serves it on one or two GB10s from NVIDIA's NVFP4. Try the upstream native binary behind `swap.py` first, then decide on a port |
| GLM keeps states at message starts | A shared 6,059-token history resumed 0 of 5 times (17 Oct) |
| A long-context eval | TrueScore has 1 LC and 1 LCH scenario. Recall with distractors and multi-step questions at 16k, 64k and 128k, paired across models |
| GLM's window | 98,304 of a native 1,048,576 tokens, start-up estimate 101.1 of 105.4 GiB, no int8 KV. Measure KV bytes per token, then try int8 KV with a quality check |
| Behaviour past the window | Refusal, truncation and client compaction, per engine and through the gateway |
| Thinking budget | Only `canon-budget/2048-s1` ran (51 requests). Run 4,096 and 8,000, then pick from the gateway data |
| Thinking-on arms | TrueScore and planning with thinking on, per model |
| A budget per task shape | Choose by tools present and turn count once the budget data shows where thinking pays |
| Empty replies | Qwen closes 4% of draws inside an open think block. A gateway retry with a fresh seed first |
| Check local GLM's planning transcripts | Its first 31 planning scenarios averaged 53.1 against Qwen's 92 to 94 |

## Done

| Task | Result |
|---|---|
| Empty-reply investigation | `~/empty-replies/empty-replies-findings.md` |
| Release 0.8.0 | Background prefill, disk states for GLM, GLM stop fix and the upstream fixes, 16 Oct |
| GLM splice of gate cuts into the live stream | Merged into `fork/dev`, `8c405e56` |
| Bolt: reference, IQ2_XXS kernels, engine, LiLiCorr, merged into `feat/bolt` | Full 88-layer model: 95 of 96 greedy golden tokens, drafted equals serial 10 of 10, 29 tok/s serial |
| Qwen FP4-Local (CSF scales) | Decoder equals vLLM's on 73,728 matrices. Decode 0.71 to 0.82x EXL3 (`feat/qwen-fp4-csf`) |
| Upstream ports | Owner-only request log, linear-time GLM call parse (#883), branch points (#735), DeltaNet scan marks (#851) |
| `playbook.md` | Guidance for bringing up a new model |
| Z.ai GLM-5.3-Flash reference runs | TrueScore 91.1, planning 72.5. Its thinking-off switch did not hold (reasoning in 56/152 and 159/160 replies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
