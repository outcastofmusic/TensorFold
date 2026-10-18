# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

Standing rule: after every few merges into `fork/dev`, review what landed since the last review (bugs, poor engineering, over-complicated, dead or duplicated code, optimizations) without changing replies or speed, and record the reviewed range here. Last reviewed: 0.8.0 (`8c405e56`).

## In progress

| Task | State |
|---|---|
| Follow-up 2 and d1 part of 4: LFM2.5 fast short prefill, prompt reuse, decode graphs; d1 one head matmul | Built on `perf/lfm2-prompts` (`d6319a1e`, `e8c805e2`, `a0b41220`, `b585b8fb`), CPU 4,803 passed. Not merged until `tf-runs/gpu-lfm2/run.sh` passes after bench2, then `PAIRS_ROWS` from its timing tool and a paired TrueScore |
| Follow-up 4: Qwen persists only the longest state per chain; block reuse; scheduler liveness | Built on `perf/disk-states` (`d1c880fa`, `c5f31a9f`, `61f71117`), CPU 4,808 passed. Block names live on the snapshot (ids alone are unsafe: decoded rows differ from a cold prefill). Not merged until `tf-runs/gpu-disk/run.sh` passes after bench2. Deploying it changes the fingerprints, so states on disk start over once |
| Follow-ups 1 and 3: Bolt decode CUDA graphs and a drafter-only Q8_0 head | Built on `perf/bolt-graphs` (`20201633`, `1b52dad4`), CPU 4,745 passed; expected serial 40-60 tok/s against 29. Not merged until its GPU run passes: `tf-runs/gpu-graphs/run.sh` after bench2 |
| Code review since 0.8.0 (`8c405e56..7891345f`) | All fixes merged into `feat/integration` (`1bd20c09`): ops, GLM, Bolt, LFM2, Qwen and disk states; CPU 4,794 passed. Gateway branches fixed separately. Waiting for the GPU tests after bench2, then `fork/dev` and "Last reviewed" |
| Qwen variants: Mia's Zig recipe on FP4-Local | Analysis done (`tf-runs/mia-qwen/ANALYSIS.md`): port patch 0014's shared-memory expert tile (`fn_experts_tile.cu`, same bits as our `nvfp4_expert_kernel`, +13-15% prefill on Zig), then 4,096-row prompt chunks for NVFP4 and n-gram read-ahead. Skip fp8 KV (under 3% capacity), fair fill and the repeat cache (we have both). Our FP4-Local prefill 1,857 tok/s at 8k against Mia's 2,583 (different tools, 1.4x). Next: A/B both engines with one tool on :31005, nsys on our 16k prefill, then port the tile |
| Definitive playbook audit | Four read-only audits running (upstream bring-ups, our patches and reviews, per-model results, serving layer and external engines' practice). Then rewrite `playbook.md` with a step-by-step checklist |

## To do

Order (18 Oct): the pipeline code changes above, then the Qwen3.8 variants (Mia's recipe, the FP4-Local vs EXL3 tie-breaker), then the rest.

| Task | Notes |
|---|---|
| Qwen FP4-Local vs EXL3 tie-breaker | Two more TrueScore repeats each (76x2 more, about 1.5 h each) to halve the CI on today's +2.3 [-1.1, +6.4]. Lead holds clear of zero: FP4-Local for planning and hard reasoning; near zero: EXL3 stays the default |
| Tool-call markup with parallel_tool_calls false | Gemma 4's `<|tool_call>` and DeepSeek-V4's DSML envelopes still stream as content; `server/tool_policy.py`'s `_FIXED` table can take them |
| Next review: shared prompt-state store | LFM2.5's reuse imports Bolt's `Store`/`keep_points` from `nemotron_bolt/cuda/reuse.py`; move them to a shared module (e.g. `tensorfold/cuda/kept_states.py`) once both branches are merged |
| Thinking-on TrueScore arms | TrueScore with thinking on per model Also the rest of round 2, stopped 18 Oct for the GPU window: GLM planning with thinking on (52 requests answered, rerun), Bolt planning with thinking on, and the long-context pilot on FP4-Local |
| Nemotron 3.5 Super VL (120B-A12B) as a swap candidate | Upstream 1.0.12 serves it on one or two GB10s from NVIDIA's NVFP4. Native 1.0.12 smoke passes. TrueScore 94.1 (highest so far), planning 62.6 (round 2, 18 Oct). Then a native target in `run.sh`/`swap.py` (`serve_native`) |
| Model split decision (planner vs executor) | After the GPU window. (1) Implementation suite (`cx-signal` + code + agentic, 44 x4, `run_ext.py`) and the long-context set on Qwen FP4-Local, Bolt with graphs and LFM2.5 with its prompt fixes. (2) Co-residence: does Qwen EXL3 + Clef + Bolt or LFM2.5 fit, and with how much KV room? A split only pays if the executor is clearly faster at equal quality and no swap sits between planner and executor |
| Deploy the gateway empty-reply retry | Reviewed and fixed: `feat/empty-retry` `19d4145` (hold opt-in at 0, retry keeps the last body sent, shadow mode, headers flushed at once), go test -race passes. Deploy in shadow first: `"reseed": {"qwen3.8-flash-next": {"retries": 1, "mode": "shadow"}}`, then act. Merge into `~/mini-llm-gateway`, Ctrl-C and `just serve` in its tmux pane. Waiting for your go |
| Evaluate the gateway's shadow checks: real-model run | Case set built (`feat/check-eval`, rebased with fixes `19ee308`, `e1e1f8d`: localhost-only upstream, stops on a swap, recorded traffic gitignored, `~/workspace/gateway-check-eval`): 158 labelled cases, runner, fake-engine run done. Provisional: side_requests and arg_loop act; empty_retry (fire only on finish stop), retry (only with Retry-After) and repeated_call (exempt command and wait tools) tweak; tool_repair and reasoning_loop stay shadow. Run the real-model commands in CHECK-EVAL.md when a model is up, then label results and decide |
| Run the long-context set | Built in `spark-bench-ext` (`lc_ext.py`, `run_lc.py`): 72 measurement scenarios (recall with near-duplicate distractors, aggregation over 2-4 facts, superseded values in long tool transcripts) at 16k/64k/128k, 468 unit tests pass. Pilot the 27-scenario dev set (`--scenario-ids lc`) on one model first, then `lcm` x3: about 4.5 h for Qwen and 10.5 h for GLM without prefix reuse. GLM's 98,304 window leaves 8,400 tokens at 128k |
| Labelled sets for d1's uses | d1's recipe measured 61.7% accuracy and 12.3% calibration error on its Fast Decisions run. Each use (thinking budget, model routing, empty or looping reply) needs its own labelled set before a threshold is trusted |
| Decide the swap set and update `swap.py` / `run.sh` | Swap between Qwen FP4-Local, Qwen EXL3, GLM-5.3 and Nemotron Bolt, from the benchmark results |
| GLM's window | 98,304 of a native 1,048,576 tokens, start-up estimate 101.1 of 105.4 GiB, no int8 KV. Measure KV bytes per token, then try int8 KV with a quality check |
| Behaviour past the window | Refusal, truncation and client compaction, per engine and through the gateway |
| Thinking budget | Only `canon-budget/2048-s1` ran (51 requests). Run 4,096 and 8,000, then pick from the gateway data |
| A budget per task shape | Choose by tools present and turn count once the budget data shows where thinking pays |
| Check local GLM's planning transcripts | Its first 31 planning scenarios averaged 53.1 against Qwen's 92 to 94 |

## Done

| Task | Result |
|---|---|
| Benchmark round 2 (18 Oct) | LFM2.5-24B TrueScore 91.7, planning 58.6 (443 min). Super VL TrueScore 94.1, planning 62.6. Stopped during GLM thinking-on planning to free the GPU; the remaining arms moved to Thinking-on TrueScore arms. `tf-runs/bench2/RESULTS.md` |
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
| Restore production Qwen | Back on 17 Oct 20:21 from `fork/dev` (`8c405e56`, image `tensorfold-run:590ffa112f24`), down from 07:24 UTC |
| Merge into `fork/dev` | `5d90dc01` pushed 17 Oct: upstream ports, FP4-Local CSF, scan marks, Qwen disk states with pruning and swap persist, GLM message stops, Bolt with reuse and adaptive depth, d1 (opt-in), LFM2.5-24B, swap targets for all. GPU tests 559/560 then all pass; served checks pass (Qwen disk resume 12,071 tokens in 1.6 s, same reply as cold; GLM message stops; LFM2.5 and Super VL smoke). Clef now starts before FP4-Local, and d1 stays off beside GLM |
| LFM2.5 research | Only the 24B-A2B suits agentic work. 1.2B-Instruct and 350M skip thinking, for titles and guards. LFM2.5 cannot draft for our models (different vocabularies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
