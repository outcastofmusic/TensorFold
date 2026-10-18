# Tasks

The running list of TensorFold work on this box: what is done, in progress, to do or cancelled. `playbook.md` holds the
guidance for bringing up a new model; this file holds the work. Dates are 2026.

Standing rule: after every few merges into `fork/dev`, review what landed since the last review (bugs, poor engineering, over-complicated, dead or duplicated code, optimizations) without changing replies or speed, and record the reviewed range here. Last reviewed: `3acff4dd` (review of `7891345f..3acff4dd`, docs/reviews/2026-10-18.md, fixes pushed 18 Oct). Next review starts at `3acff4dd`.

Standing rule (18 Oct): keep every code change that gives a clear, measured win and passes the gates, however small. Small gains accumulate.

## In progress

| Task | State |
|---|---|
| Qwen variants: close the prefill gap to Mia | HC fusions merged (ef6834b4, 18 Oct): FP4-Local cold prefill 12k 2,228 tok/s (Mia 2,145), 32k 2,076 (Mia 2,159), bytes and served replies identical. Remaining at 32k: gate/up expert tile 5,369 against Mia's 4,690 us with identical code (about 33 ms a chunk, cause unknown: plan order, X layout or compile flags); shared expert on a second stream (about 22 ms). Both bit-identical. 131k not re-profiled |

## To do

Order (18 Oct): the pipeline code changes above, then the Qwen3.8 variants (Mia's recipe, the FP4-Local vs EXL3 tie-breaker), then the rest.

| Task | Notes |
|---|---|
| Upstream 1.1.0 ports | Merged (18 Oct): grammar vocab from llm_config (Bolt structured output), linear-time parse for all four call envelopes (7.5-10 s to under 5 ms at 2,000 unclosed openers), GLM SwiGLU limit 0 as no limit, WebP note. Super VL native 1.1.0 pin merged, GPU check running (`tf-runs/gpu-super110`). Later: tool_choice required/named as a full grammar (LFM2.5 and d1 refuse today), Bolt on native 1.1.0 (A/B), lone-stream decode at --parallel 4 against 1 |
| Tool parser leftovers | `parse_tool_calls_from_content` raises AttributeError when a `<tool_call>` payload starts with `<｜DSML｜tool_calls>` and has no closer (tools.py:472, only ValueError/TypeError caught); 2,000 unclosed `<parameter=` tags inside one call still rescan the call (9.7 s at 1 MB, `_TOOL_PARAMETER_BLOCK_RE`); MLX applies no SwiGLU clamp when the key is missing, CUDA uses 10.0 |
| 4,096-row NVFP4 prompt pieces, long prompts only | Four A/B rounds (18 Oct, last on an idle box): 131k faster in all four (1.09, 1.08, 1.03, 1.04x), 4k slower in three of four (0.72-0.97x), 32k mixed; replies identical. Next: take 4,096-row pieces only when the prompt left to fill is long (threshold from a sweep), then A/B again |
| Tiny NVFP4 checkpoint: kept state inside a prompt piece | DeltaNet layer 1's `rec` and `conv` at a kept point inside a piece differ from a fresh prefill ending there, deterministically, on the base too and at both piece sizes (`tests/cuda/test_flashnext_nvfp4_pieces.py`, 18 Oct). FP4-Local passes. Find what differs about layer 1 of the tiny checkpoint. Also: two loads of the tiny checkpoint give different replies with every switch off (18 Oct, `tests/cuda/test_hc_fusions.py` scratch run): look for nondeterminism or uninitialised memory in the tiny ModelOpt load path first |
| Bolt's dropped closing braces | 35 of 160 Bolt planning replies end `...}]` + fence without the object's `}` and score 0 (other models 0 of 160). Replay with `return_token_ids` to tell the model from the detokenizer |
| Tool-call markup with parallel_tool_calls false | Gemma 4's `<|tool_call>` and DeepSeek-V4's DSML envelopes still stream as content; `server/tool_policy.py`'s `_FIXED` table can take them |
| Shared keep-point filter for the Qwen families | The keep-point rule is repeated about 8 times across qwen3_5, qwen3_5_moe and qwen4_exp (engine.py, multi.py, multi_plan.py); Qwen ends at `entry_end(prompt)` where `cuda/kept_states.keep_points` uses `len - 1`, so it needs its own shared helper and tests. Store, keep_points moved for Bolt and LFM2.5 (18 Oct) |
| Launch-gap ideas from NVIDIA's Kimi K3 decode work | `tf-runs/megakernel/REPORT.md`: no full megakernel (GB300-only code, gain capped at 5-12%). First nsys on one graph replay plus a PDL microbench on GB10; then GLM's two scratch copies in `packed.py:134,158` and dead `group_sums` (hours, no bit change); PDL on EXL3 kernels only if gaps are 1 µs or more. GLM's bigger lever is its kernels at 53% of bandwidth |
| Playbook gaps (checks we do not run yet) | From `playbook.md` rows marked (gap): tokenizer round-trip and HF id diff on adversarial text; single-BOS test; one shared golden-logit gate with a stated tolerance per family; streamed markup-leak test per family; `test_prompt_parity.py` entries for Bolt, LFM2 and FP4-Local; RoPE/YaRN check against HF at 32k and 128k; warn on `generation_config.json` keys the CLI drops (LFM2.5's repetition_penalty); logprobs on tool or thinking requests; Bolt goldens into the repo; recipes for Bolt and FP4-Local |
| Thinking-on TrueScore arms | TrueScore with thinking on per model Also the rest of round 2, stopped 18 Oct for the GPU window: GLM planning with thinking on (52 requests answered, rerun), Bolt planning with thinking on, and the long-context pilot on FP4-Local |
| Nemotron 3.5 Super VL (120B-A12B) as a swap candidate | Upstream 1.0.12 serves it on one or two GB10s from NVIDIA's NVFP4. Native 1.0.12 smoke passes. TrueScore 94.1 (highest so far), planning 62.6 (round 2, 18 Oct). Then a native target in `run.sh`/`swap.py` (`serve_native`) |
| Model split decision (planner vs executor) | After the GPU window. (1) Implementation suite (`cx-signal` + code + agentic, 44 x4, `run_ext.py`) and the long-context set on Qwen FP4-Local, Bolt with graphs and LFM2.5 with its prompt fixes. (2) Co-residence: does Qwen EXL3 + Clef + Bolt or LFM2.5 fit, and with how much KV room? A split only pays if the executor is clearly faster at equal quality and no swap sits between planner and executor |
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
| d1 head read (18 Oct) | `per_question` is the default (user's choice): the original bits at the same speed; `union` (b585b8fb) changed 1/200 answers and gained nothing (all three within 1.7%). `TF_D1_HEAD` keeps the others. `tf-runs/gpu-d1/RESULTS.md` |
| Gateway empty-reply reseed (18 Oct) | Deployed in shadow: `main` 19d4145, `reseed.qwen3.8-flash-next` retries 1, hold 0. Old config at `~/.local/state/gateway.json.bak-20261018`. Decide act from its shadow data once traffic returns |
| Qwen FP4-Local vs EXL3 tie-breaker (18 Oct) | EXL3 stays the default. Pooled 76x4 a arm: FP4-Local - EXL3 +2.3 [-0.9, +6.4]; same session +2.3 [-1.2, +6.6]. Each arm repeated within 0.3 across days (FP4 90.3 / 90.6, EXL3 88.1 / 88.3). The interval is set by scenario spread, so more repeats will not narrow it. FP4-Local stays a swap target for planning (Q94 against 92) at 0.71-0.82x the decode speed and 23 GiB more memory |
| Code review since 0.8.0 (`8c405e56..7891345f`) | `docs/reviews/2026-10-17.md`. Fixes merged; GPU tests found two defects in them, both fixed on 18 Oct: the IQ2_XXS shared-memory ceiling cache (688bc309, fixed by `6f19c352`) and a stale GLM ring test (`a2dedd3d`) |
| Bolt follow-ups: decode CUDA graphs and a drafter-only Q8_0 head | Merged (`e6bf4fb9`). Graphs add under 1% (serial 30.9 to 31.3 tok/s): decode was not launch-bound. The Q8_0 head lengthens accepted runs: drafted greedy code 36.5 to 45.1, chat 34.5 to 46.1, tools 49.1 to 60.4, long 37.2 to 46.4, prose 30.9 to 33.2 tok/s. Golden 94/96, drafted == serial 10/10 |
| LFM2.5 follow-ups: tensor-core prompt kernel, prompt reuse, decode graphs; d1 one head matmul | Merged (`0d5afcab`). Prefill 1k 1,228 to 4,802 tok/s, 16k 1,792 to 3,982. `PAIRS_ROWS` 64 from the timing tool (`1985f992`). LFM2 24/24, d1 6/6. Paired TrueScore against the old build still to run |
| Qwen disk-state follow-ups: longest state per chain, block reuse, scheduler liveness | Merged (`5ea1a0ef`). 336 GPU tests pass. Fingerprints change, so states on disk start over once |
| Definitive playbook (18 Oct) | `playbook.md` rewritten as a 15-phase checklist with pass criteria and evidence, from four audits in `tf-runs/playbook-audit/` (upstream bring-ups, fork patches, per-model results, serving layer and other engines) |
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
| Four-model TrueScore and planning (thinking off, 17 Oct) | TrueScore: GLM 93.0, Bolt 92.0, FP4-Local 91.0, EXL3 87.8; no pair significant. Planning: FP4-Local 89.8 (Q94), EXL3 89.0 (Q92), GLM 60.5 (Q58), Bolt 38.3 (Q27). With thinking off, planning measures visible reasoning (Qwen writes a median 2,614 tokens, GLM 188, Bolt 121): GLM with thinking on scored 99.0 on the 13 PA scenarios it finished, so rank planners only with thinking on. Results in `tf-runs/truescore`, paired script `paired.py` |
| GLM regenerate check (#735) | Three served attempts never hit the eviction window; replies identical across arms each time. The CPU tests cover the eviction choice |
| Restore production Qwen | Back on 17 Oct 20:21 from `fork/dev` (`8c405e56`, image `tensorfold-run:590ffa112f24`), down from 07:24 UTC |
| Merge into `fork/dev` | `5d90dc01` pushed 17 Oct: upstream ports, FP4-Local CSF, scan marks, Qwen disk states with pruning and swap persist, GLM message stops, Bolt with reuse and adaptive depth, d1 (opt-in), LFM2.5-24B, swap targets for all. GPU tests 559/560 then all pass; served checks pass (Qwen disk resume 12,071 tokens in 1.6 s, same reply as cold; GLM message stops; LFM2.5 and Super VL smoke). Clef now starts before FP4-Local, and d1 stays off beside GLM |
| LFM2.5 research | Only the 24B-A2B suits agentic work. 1.2B-Instruct and 350M skip thinking, for titles and guards. LFM2.5 cannot draft for our models (different vocabularies) |
| Muse Spark Open research | 822B-A44B, smallest checkpoint 175.6 GB: does not fit one Spark. AA index 39 against Qwen 40 and GLM 42 |

## Cancelled

| Task | Why |
|---|---|
| Muse Spark Open on the Spark | No checkpoint fits one GB10 and its published quality does not beat Qwen or GLM |
