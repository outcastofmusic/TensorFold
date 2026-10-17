# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

Standing rule: after every few merges into `fork/dev`, review what landed since the last review (bugs, poor engineering, over-complicated, dead or duplicated code, optimizations) without changing replies or speed, and record the reviewed range here. Last reviewed: 0.8.0 (`8c405e56`).

## In progress

| Task | State |
|---|---|
| Benchmark round 2 (`tf-runs/bench2.sh`) | LFM2.5-24B and Super VL TrueScore + planning; GLM and Bolt planning with thinking on; long-context pilot on FP4-Local. Then production restarts on the new `fork/dev` |

## To do

| Task | Notes |
|---|---|
| Code review of everything since 0.8.0 | Review `8c405e56..fork/dev` (and the gateway branches `feat/empty-retry`, `feat/check-eval`) for bugs, poor engineering, over-complicated code, dead or duplicated code (e.g. the two LFM2 conv/attention paths, `lfm2` and `lfm2_moe`), and optimizations. Constraints: no change to business logic or replies (drafted == serial, resumed == fresh, golden checks hold) and no speed regression (re-run the speed checks that cover each touched path: bench_openai for Qwen, the Bolt depth test, d1 latency, LFM2.5 decode and prefill). Review first, then fixes in separate commits, each with the CPU suite and the matching GPU tests |
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
| Merge into `fork/dev` | `5d90dc01` pushed 17 Oct: upstream ports, FP4-Local CSF, scan marks, Qwen disk states with pruning and swap persist, GLM message stops, Bolt with reuse and adaptive depth, d1 (opt-in), LFM2.5-24B, swap targets for all. GPU tests 559/560 then all pass; served checks pass (Qwen disk resume 12,071 tokens in 1.6 s, same reply as cold; GLM message stops; LFM2.5 and Super VL smoke). Clef now starts before FP4-Local, and d1 stays off beside GLM |
| LFM2.5 research | Only the 24B-A2B suits agentic work. 1.2B-Instruct and 350M skip thinking, for titles and guards. LFM2.5 cannot draft for our models (different vocabularies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
