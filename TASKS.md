# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

## In progress

| Task | State |
|---|---|
| Integration branch `feat/integration` (worktree `~/workspace/tf-int`) | All merged; CPU 4,723 passed. GPU 17 Oct 20:25-20:46: Qwen disk states 154 passed, Qwen marks/concurrency 243, GLM 63, Bolt 82, d1 6 (after a tolerance fix), LFM2.5 11. Bolt golden 94/96 (was 95/96; the chat prompt's closest gap 0.25 against a 0.77 error), drafted == serial 10/10; adaptive depth prose 30.9 tok/s against 29.1 serial and 23.7 fixed. d1 25.9 ms a question. Served checks running (`tf-runs/served`) |
| GLM keeps states at message starts | Built on `feat/glm-message-stops` (`44781bd0`, `14200b09`) and merged into `feat/integration` (CPU 4,596 passed). Every kept point is now a cut inside its prefill pass, so keeping states adds no passes; a new stop at the last user message. GPU tests and `tools/glm_message_stops_check.py` run after the benchmarks; then update playbook section 7 |
| Bolt prompt reuse and adaptive draft depth | Built on `feat/bolt-reuse` (`269ce0b1`, `45e0dafd`), merged into `feat/integration`. States kept at message starts (257 MB at 10k tokens, 585 MB at 50k, computed); draft depth chosen per round, modelled to bring prose from 20-22 to about 29 tok/s. Prompt-mode expert kernels now run at every row count, so short-prompt bits changed: rerun the golden check with the GPU tests after the benchmarks |
| d1-3B decision model as a sidecar beside Clef | Built on `feat/d1-sidecar` and merged into `feat/integration` (CPU 4,675 passed): an `lfm2` family in plain torch, `/v1/systemone` and `/v1/decisions`, pictures included; 9/9 top answers equal Transformers bf16 on the CPU with real weights. `D1=1` starts it on :30003 after the big model (7.3 GiB); opt-in until the GPU test (`tests/cuda/test_lfm2_d1_checkpoint.py`, parity and latency) passes. Nothing calls it yet |
| Import LFM2.5-24B-A2B (FP8) into the fork | Built on `feat/lfm2-moe`, merged into `feat/integration` (CPU 4,723 passed). The reference matches transformers to 6.1e-5; the engine on the CPU with real weights follows it except where bf16 flips near-tied expert picks. One stream, no CUDA graphs or prompt reuse yet; about 3.2 GB read a token (bound 68-85 tok/s). `./run.sh lfm2-24b`. GPU tests and `tools/lfm2_moe_smoke.sh` after the benchmarks, then TrueScore |
| Qwen disk states, and a persist endpoint `swap.py` calls before it stops a model | Built on `feat/qwen-disk-states` (`da1dc25d`, `e99b729c`): CPU suite 4,453 passed, 0 failed. GPU tests and a real-model restart check run after the benchmarks. A state takes 298.7 MB at 10k tokens and 1,030.9 MB at 50k |

## To do

| Task | Notes |
|---|---|
| Deploy the gateway empty-reply retry | Built `2a9ea81` on `feat/empty-retry` (`~/workspace/gateway-empty-retry`), go tests pass. Deploy: merge into `~/mini-llm-gateway`, add `"reseed": {"qwen3.8-flash-next": {"retries": 1, "hold_reasoning_bytes": 4096}}` to gateway.json, Ctrl-C and `just serve` in its tmux pane. Waiting for your go |
| Evaluate the gateway's shadow checks: real-model run | Case set built (`feat/check-eval`, `70b666d`, `~/workspace/gateway-check-eval`): 158 labelled cases, runner, fake-engine run done. Provisional: side_requests and arg_loop act; empty_retry (fire only on finish stop), retry (only with Retry-After) and repeated_call (exempt command and wait tools) tweak; tool_repair and reasoning_loop stay shadow. Run the real-model commands in CHECK-EVAL.md when a model is up, then label results and decide |
| Run the long-context set | Built in `spark-bench-ext` (`lc_ext.py`, `run_lc.py`): 72 measurement scenarios (recall with near-duplicate distractors, aggregation over 2-4 facts, superseded values in long tool transcripts) at 16k/64k/128k, 468 unit tests pass. Pilot the 27-scenario dev set (`--scenario-ids lc`) on one model first, then `lcm` x3: about 4.5 h for Qwen and 10.5 h for GLM without prefix reuse. GLM's 98,304 window leaves 8,400 tokens at 128k |
| Labelled sets for d1's uses | d1's recipe measured 61.7% accuracy and 12.3% calibration error on its Fast Decisions run. Each use (thinking budget, model routing, empty or looping reply) needs its own labelled set before a threshold is trusted |
| Decide the swap set and update `swap.py` / `run.sh` | Swap between Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt, from the benchmark results |
| Nemotron 3.5 Super VL (120B-A12B) as a swap candidate | Upstream 1.0.12 serves it on one or two GB10s from NVIDIA's NVFP4. Try the upstream native binary behind `swap.py` first, then decide on a port |
| GLM's window | 98,304 of a native 1,048,576 tokens, start-up estimate 101.1 of 105.4 GiB, no int8 KV. Measure KV bytes per token, then try int8 KV with a quality check |
| Behaviour past the window | Refusal, truncation and client compaction, per engine and through the gateway |
| Thinking budget | Only `canon-budget/2048-s1` ran (51 requests). Run 4,096 and 8,000, then pick from the gateway data |
| Thinking-on arms | TrueScore and planning with thinking on, per model |
| A budget per task shape | Choose by tools present and turn count once the budget data shows where thinking pays |
| Thinking-on planning arms for GLM and Bolt | Both trail Qwen by 35+ points on planning with thinking off; Z.ai's GLM with thinking scored Q77 |
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
| Four-model TrueScore and planning (thinking off, 17 Oct) | TrueScore: GLM 93.0, Bolt 92.0, FP4-Local 91.0, EXL3 87.8; no pair significant. Planning: FP4-Local 89.8 (Q94), EXL3 89.0 (Q92), GLM 60.5 (Q58), Bolt 38.3 (Q27). GLM and Bolt trail Qwen by 35+ points on planning with thinking off. Results in `tf-runs/truescore`, paired script `paired.py` |
| GLM regenerate check (#735) | Three served attempts never hit the eviction window; replies identical across arms each time. The CPU tests cover the eviction choice |
| Faster LFM2.5 prompts | Prefill 747-756 tok/s from 4k tokens, but a fixed ~1.5 s a prompt (42 tok/s at 64 tokens) because each chunk decodes its used experts to fp32. Decode experts to bf16 once a chunk, or run short chunks on the grouped FP8 kernel; it hurts the short tool-routing turns this model is for |
| Restore production Qwen | Back on 17 Oct 20:21 from `fork/dev` (`8c405e56`, image `tensorfold-run:590ffa112f24`), down from 07:24 UTC |
| LFM2.5 research | Only the 24B-A2B suits agentic work. 1.2B-Instruct and 350M skip thinking, for titles and guards. LFM2.5 cannot draft for our models (different vocabularies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
