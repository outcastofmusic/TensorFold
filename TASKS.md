# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

## In progress

| Task | State |
|---|---|
| Integration branch `feat/integration` (worktree `~/workspace/tf-int`) | Merged upstream-ports, qwen-fp4-csf, qwen-scan-marks, qwen-disk-states and bolt without conflicts; CPU suite 4,591 passed, 0 failed. GPU tests after the benchmarks, then `fork/dev` |
| GLM keeps states at message starts | Background agent, `feat/glm-message-stops` |
| Bolt prompt reuse and adaptive draft depth | Background agent, `feat/bolt-reuse` |
| Long-context eval set | Background agent, new files in `spark-bench-ext` (`lc_ext.py`, `run_lc.py`) |
| d1-3B decision model as a sidecar beside Clef | Background agent, `feat/d1-sidecar`: an LFM2 (d1) family, `/v1/systemone` and `/v1/decisions`, a `d1` sidecar in `run.sh` on port 30003 |
| Gateway empty-reply retry | Background agent, `~/workspace/gateway-empty-retry` (`feat/empty-retry`), built and tested only, not deployed |
| TrueScore 76x2 and planning x4 on Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt | FP4-Local 91.0 / planning 89.8, EXL3 87.8 / 89.0, GLM 93.0 / planning running, Bolt next (`tf-runs/truescore-queue.sh`, 17 Oct) |
| Qwen disk states, and a persist endpoint `swap.py` calls before it stops a model | Built on `feat/qwen-disk-states` (`da1dc25d`, `e99b729c`): CPU suite 4,453 passed, 0 failed. GPU tests and a real-model restart check run after the benchmarks. A state takes 298.7 MB at 10k tokens and 1,030.9 MB at 50k |
| GLM regenerate check of the branch-point rule (#735), base against ports | Chained after the benchmarks (`tf-runs/glm-regen.sh`) |

## To do

| Task | Notes |
|---|---|
| Restore production Qwen | `POST :30000/swap/qwen3.8-flash-next` after the GLM regenerate check. Production has been down since 17 Oct 07:24 UTC |
| Decide the swap set and update `swap.py` / `run.sh` | Swap between Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt, from the benchmark results |
| Nemotron 3.5 Super VL (120B-A12B) as a swap candidate | Upstream 1.0.12 serves it on one or two GB10s from NVIDIA's NVFP4. Try the upstream native binary behind `swap.py` first, then decide on a port |
| GLM's window | 98,304 of a native 1,048,576 tokens, start-up estimate 101.1 of 105.4 GiB, no int8 KV. Measure KV bytes per token, then try int8 KV with a quality check |
| Behaviour past the window | Refusal, truncation and client compaction, per engine and through the gateway |
| Thinking budget | Only `canon-budget/2048-s1` ran (51 requests). Run 4,096 and 8,000, then pick from the gateway data |
| Thinking-on arms | TrueScore and planning with thinking on, per model |
| A budget per task shape | Choose by tools present and turn count once the budget data shows where thinking pays |
| Trial LFM2.5-24B-A2B as a resident tool-routing and sub-agent model | 24B total, 2.45B active, 13.6 GB at 4-bit or 24.8 GB FP8. Vendor-reported BFCLv4 65 and IFBench 75, no independent agentic score. First run it outside TensorFold (llama.cpp Q4_K_M or vLLM FP8) on our own tool-call suite. Upstream serves it on Metal only, so a TensorFold port means `lfm2_moe` on CUDA. License caps commercial use at under US$10M annual revenue |
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
| Prune kept states on disk | 72 h TTL on unused states and old builds' folders, one budget over every build's folder, oldest first (`feat/qwen-disk-states`). CPU suite 4,459 passed |
| Decide persist on a swap that cuts requests off | Skip it (17 Oct): those requests hold the engine, so their latest states are lost unless an idle write caught them. `swap.py` line 131 |
| Swap targets for all four models | `qwen-flash-next-fp4` and `nemotron-bolt` in `run.sh`, `swap.py` and `prepare.sh`, default unchanged (`ceb12076`). Not yet run on the GPU |
| LFM2.5 research | Only the 24B-A2B suits agentic work. 1.2B-Instruct and 350M skip thinking, for titles and guards. LFM2.5 cannot draft for our models (different vocabularies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
