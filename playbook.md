# Bringing up a new model on TensorFold

This playbook takes a new model or checkpoint from "is it worth it" to a production swap target on the DGX Spark
(GB10, 121 GB unified memory). Part 1 is the checklist: work through it in order, one row at a time. A row is done when
its pass criterion holds and the number that proves it is written down. Part 2 explains the rows that need more than a
line. Part 3 is the reference: the bug classes seen so far, the ideas that did not pay, and what each model measured.

Every row comes from a bug found or a number measured between 13 and 18 Oct 2026, on this fork, on upstream TensorFold
or on another engine. The four audits behind it are in `tf-runs/playbook-audit/` (18 Oct): upstream bring-ups (1), this
fork's patches and reviews (2), every model we measured (3), and the serving layer with other engines' practice (4).
Commit hashes refer to this fork unless the row names upstream.

Order matters. A checkpoint that loads wrong makes every later measurement meaningless, and a speed number counts only
after the exactness gates hold.

## Part 1. Checklist

Copy the phase tables into the model's bring-up notes and tick rows as they pass. Rows marked **(gap)** name a check
we do not run yet. Add the test when you reach that row.

### Phase 0. Decide whether to bring it up

| # | Do | Pass when |
|---|---|---|
| 0.1 | Sum the weights from the safetensors headers (an HTTP range read of the first 8 bytes gives the header length) | It fits with a working window: under about 101 GiB alone (GLM sat at 101.1 of 105.4), at or under about 75 GiB beside Clef (18 GB) and d1 (7.3 GiB). Reject above about 175 GB (Muse Spark Open, 175.6 GB) |
| 0.2 | Compute KV bytes a token and fixed recurrent state from `config.json` (`tf-runs/super/size.py`) | One stream at the target window fits in what the weights leave (Part 2.2) |
| 0.3 | Estimate the decode ceiling: 273 GB/s over the bytes a token reads | Expect about 70% of it. Super VL: 9.77 GB a token, 28 tok/s ceiling, 32-33 measured with MTP |
| 0.4 | Compare published quality on the same index as the incumbents | It can beat Qwen (AA 40) or GLM (AA 42) for some role. Treat vendor-only agentic scores as unverified |
| 0.5 | Read the licence of the weights and of every drafter | The use fits. LFM Open caps commercial use below US$10M revenue. DFlash2 is CC-BY-NC-ND |
| 0.6 | List every quantization format and check each against the engine's readers | Each maps to a kernel, or the work to add one is scoped (CSF scales, IQ2_XXS, NVFP4 activations each needed code) |
| 0.7 | Read the chat template | The thinking switch exists and is known to hold or not. LFM2.5 has none |
| 0.8 | Check the vocabulary against our models | It is known whether the model can draft for ours or be drafted by them |
| 0.9 | Check the engine features the family has: streams, prompt reuse, background prefill, graphs | Missing ones are listed as bring-up work (Part 2.2). A one-stream engine flattens aggregate decode and pushed TTFT p50 to 40-79 s under concurrency |

### Phase 1. Set up the workspace

| # | Do | Pass when |
|---|---|---|
| 1.1 | Work on a branch in its own worktree | `fork/dev` is untouched until Phase 14 |
| 1.2 | Run test servers on :31005-31007 | Nothing binds 30000-30010: the gateway scans that range and a :30005 test took real traffic |
| 1.3 | Put every GPU job behind a guard | It starts only with its memory plus a margin free, dies below 8-12 GiB available, and dies when `GET :30000/swap` shows a load |
| 1.4 | Run one GPU job at a time from one queue | Two agents' jobs together drove production to 2 GiB |
| 1.5 | Make queue summaries name failures | A summary that says "4 groups passed" hid 3 failed groups on 18 Oct |
| 1.6 | Start follow-up branches from a base whose GPU tests passed | `perf/bolt-graphs` inherited a broken kernel cache and its whole GPU run was wasted (18 Oct) |

### Phase 2. Read the checkpoint on the CPU

| # | Do | Pass when |
|---|---|---|
| 2.1 | Group `quantization_config.quantized_layers` by recipe and count layers a recipe | Every recipe is named. FP4-Local hid 48 expert layers with `weight_scale_encoding: "csf"` |
| 2.2 | Count tensor suffixes in `model.safetensors.index.json` | Every suffix has a reader (CSF adds `weight_scale.nvfp4_csf_fixed` and `..._exceptions`) |
| 2.3 | Follow the index's shard paths | The reader opens `tensors/` subfolders (FP4-Local) |
| 2.4 | Parse nested configs (`text_config`, `llm_config`) and refuse every unsupported field by name before download | A test feeds each unsupported config and gets a named refusal. Super VL keeps `vocab_size` in `llm_config` (upstream 022344e9) |
| 2.5 | Read per-module widths and refuse what no kernel reads, at load | A mixed-rate layer is refused at load, never at its first forward, which may be a graph capture (c7e0075f). Never cast an unsupported format: an MXFP8 head cast to bf16 decoded "fluent garbage" (upstream 3d7a1478) |
| 2.6 | Normalize norm conventions per tensor class | Layer 0's norms match the reference. EXL3 and NVFP4 store centred RMSNorms as gamma-1 (upstream 5b4b3438). Clef needed +1 (837a86e1). Zero-centred MTP norms gave 0 of 127 drafts accepted (upstream edb8fc03) |
| 2.7 | Merge end tokens from `config.json`, `generation_config.json` and the tokenizer's chat end | The union is in the family's EOS list, with a `test_<family>_eos.py`. A config that held only 248044 dropped `<|im_end|>` 248046 and replies ran past the turn |
| 2.8 | Read the sampling defaults in `generation_config.json` | Each key the CLI drops is restated in `run.sh`. The CLI reads 5 keys and warns at startup on each sampling key it drops (83d56781, `tests/test_hub_and_checks.py`). LFM2.5's `repetition_penalty` 1.05 draws that warning and is not applied |
| 2.9 | Read the model card's serve command | The authors' precision is known (FP4-Local: 4-bit weights, 16-bit decode activations, 4-bit prefill activations) |
| 2.10 | Run the family's `detect`, `require_readable` and `check()` on the real folder | They pass on the CPU before any GPU time |

### Phase 3. Decode the weights

| # | Do | Pass when |
|---|---|---|
| 3.1 | Port each new format's decoder from published source (vLLM, llama.cpp, b12x) as written | It exists with a citation of the source line |
| 3.2 | Compare it with the published decoder on every tensor of the real checkpoint | 100% byte-equal, 0 NaN scales. CSF: 73,728 of 73,728 equal to vLLM's `decode_nvfp4_csf_scale`. LFM2 FP8: 1,416,064 blocks |
| 3.3 | Check decoded scales against the quantizer's rule | ModelOpt's largest e4m3 block scale sits at or near 448 (FP4-Local: median 448, 645 of 913 matrices exactly 448) |
| 3.4 | Decode once at load into a format the kernels already read | The kernels run unchanged (CSF decodes to plain e4m3 scales) |
| 3.5 | Batch the load-time decode by layer, with one host sync | A load decodes in seconds: CSF went from 5.4 min (4.4 ms a matrix) to 2.4 s (16.5 ms a 512-expert batch) |
| 3.6 | Bound load temporaries and keep the stored bytes | Peak temporaries are counted. Bolt's lm_head made 5.5 GiB of fp32 temporaries until it decoded 8,192 rows at a time. Upstream FP4 codes widened to uint16 took 241.6 GB against 60.4 |
| 3.7 | Close files on a failed load | A refused load leaves no open readers (49f4267b) |

### Phase 4. Prove the engine computes the model

| # | Do | Pass when |
|---|---|---|
| 4.1 | Build an fp32 reference that decodes every weight exactly | It runs layer at a time on the GPU or CPU |
| 4.2 | Check it layer by layer against the checkpoint's HF modeling code | Per-layer gap of 1e-5 or less on the same input. Bolt: 6.6e-6 Mamba, 2.7e-6 MoE, 0 attention, once the reference took HF's 1e-3 dt floor (5.6e-4 without). Record every place where HF and the CUDA path differ |
| 4.3 | Run the reference twice before trusting a probability from it | Both runs agree. exllamav3's cache-free forward gave 0.287, 0.078 and 0.094 on three identical runs |
| 4.4 | Build a sublayer trace tool | It prints the first sublayer where engine and reference part (upstream `glm_compare.py --trace`) |
| 4.5 | Write golden files and commit them | Top-20 logits and 32 greedy steps for a chat, a tool and a long prompt, with the smallest top-two gap recorded. Bolt's live outside the repo today **(gap)** |
| 4.6 | Check the engine against the goldens with a stated tolerance | Greedy matches at every step whose gap exceeds 2x the engine's error. LFM2-MoE's top-20 plus gap rule is the template (`tests/cuda/test_lfm2_moe.py:29-34`). Only LFM2-MoE has one **(gap: one shared gate)** |
| 4.7 | Measure teacher-forced fidelity against fp32 truth | KL, NLL and top-1 recorded. Upstream bars: KL ≤ 0.01, top-1 ≥ 90%, perplexity ≤ 1.02x. Compare against fp32: KL against mlx_lm's own samples called Gemma's fused path 4x worse than it was |
| 4.8 | Run at least one check per kernel class at real dimensions | Tiny fakes alone hid skipped layers (upstream a83ed1ef: nothing launched below widths 512/1,024) |

### Phase 5. Exactness of the engine's paths

All four equalities compare token SHA-256 or `torch.equal` on rows and states.

| # | Do | Pass when |
|---|---|---|
| 5.1 | Row invariance: a row alone against the same row in windows of 1 to 64+ | Bytes equal at every width. A 72-row Bolt call took the prefill kernel and differed by one bf16 rounding (f8869622) |
| 5.2 | Drafted against `"draft": false` | 10 of 10 replies, greedy and sampled, at every depth. The drafter reads the engine's keyed samples |
| 5.3 | Prompt chunks against one pass | First token equal, last-row logits within bf16, next 48 tokens equal, at 8k and 32k, cuts at 2,047, 2,049, 4,097, 8,193, from a fresh process. A 300-token fixture hid seam bugs (upstream 2a431e00) |
| 5.4 | Resumed against fresh | Equal after a resend, a regenerate, an edit and a restart. Bolt's resumed tail ran the decode kernel while a fresh chunk ran the prefill kernel (269ce0b1) |
| 5.5 | Concurrent against solo | Each of 2 and 4 streams with unequal lengths equals its solo run |
| 5.6 | A fresh process against a warm one | Equal. Uninitialized fields gave a different first token only in a fresh process (upstream bd8e0de0) |
| 5.7 | Cancel, then the next request | The next request equals plain. A cancelled 64k prompt held the GPU 31 s upstream before the fix |
| 5.8 | Graph replay against eager, greedy and sampled | Equal. A recapture left a placeholder sampling rule and greedy ran sampled graphs (upstream c4cb2bc8) |
| 5.9 | Write down any default that breaks a gate | The recipe names the gate. `TF_GLM_DENSE_PROMPT=1` makes reply rows differ (max 1.65), so a reply-end resume is exact only with it off, backed by one TrueScore pair (-0.7 [-4.0, +2.6]) |

### Phase 6. Kernels

| # | Do | Pass when |
|---|---|---|
| 6.1 | Compare a new or changed kernel byte for byte against the one it replaces, or against an FP64 reference | Equal, or within `|want|·½ulp + γ(depth)·mag` at min, tile±step and max shapes (upstream kernel contract gate 3) |
| 6.2 | Plant a control that must fail | The test goes red on a kernel that drops its tail input or sums in half |
| 6.3 | Sum K slices in a fixed order | No atomicAdd split-K: it is not repeatable (upstream 4315bc2d) |
| 6.4 | Check launch limits for every target GPU | Threads, shared bytes and registers fit. 832 compiled threads against 1,024 launched left outputs at zero (upstream a7408dba). K 2560 needs the >48 KB shared-memory opt-in (412f528c) |
| 6.5 | Treat a per-kernel driver attribute as state with last-write semantics | Cache only a high-water mark. 688bc309 cached `cudaFuncSetAttribute` per size, a smaller K lowered the ceiling, and the next larger K failed with "invalid argument" (7 Bolt failures, 18 Oct) |
| 6.6 | Turn off FMA contraction where the reference sums products in sequence | `enable_fp_fusion=False` on that kernel (d79986c1) |
| 6.7 | Size scratch and partial buffers for the rows actually launched | No NaN past a chunk edge. 128-row buffers under 2,048-row chunks gave "!!!!" past 2,051 tokens upstream |
| 6.8 | Run any `.cu` or GPU-path change on the GPU before merging | The area's GPU test files pass. The CPU suite (4,794 passed) could not see 688bc309 |
| 6.9 | Check a fused reduction at every shape it accepts, including one loop iteration | Byte equality holds at the smallest dispatch shape too. Triton 3.7.1 folded `total + dot(x, w, 0)` into `dot(x, w, total)` when a K slice held one block, so the in-program split-K sum changed bits on the tiny checkpoint (1 block a slice) and not on FP4-Local (5): guard `k // sk // bk >= 2` (edc00823, 18 Oct) |

### Phase 7. Memory

| # | Do | Pass when |
|---|---|---|
| 7.1 | Estimate start-up memory from headers before loading | The estimate is at or above the measured peak and a refusal names the budget |
| 7.2 | Read capacity from MemAvailable less a floor | `cudaMemGetInfo` counts the page cache as used on the GB10: a 10.7 GiB load was refused with 5.5 GiB "free" and 30 available (f7807ae6) |
| 7.3 | Charge every resident buffer | Drafter (0.93 GiB was missing), prompt workspace (256 MiB), graphs (0.35 GiB), sessions, vision tower, lm_head load spike. A charged tensor that does not exist (1.37 GB) is removed |
| 7.4 | Drop the page cache after load | Upstream freed 20.1 GiB in 0.26 s (646119d2) |
| 7.5 | Start the model in `run.sh`'s order beside Clef and d1 | Each documented combination keeps 8 GiB or more available. Clef after FP4-Local was refused, Clef first left 21 GiB. d1 beside GLM fell under 8 GiB, so `run.sh` never starts it there |
| 7.6 | Record the reserve the model needs | GLM needs `TENSORFOLD_MEMORY_RESERVE_GIB=11` (2 of 6 loads refused at 12.2) |
| 7.7 | Measure available memory during and after a request | It stays above the 12 GiB guard |

### Phase 8. Text layer: tokenizer, template, thinking, tools, stops

| # | Do | Pass when |
|---|---|---|
| 8.1 | Round-trip the tokenizer and diff ids against HF on an adversarial text set (whitespace, emoji, CJK, added tokens) | Equal ids and decoded text. `tests/test_tokenizer_parity.py` checks 39 `CHK_TXT`-style texts plus each checkpoint's added tokens on the five served checkpoints (519fc9a0) |
| 8.2 | Check BOS handling | One BOS, never two. `tests/test_tokenizer_parity.py` checks each BOS or template prefix token on the five served checkpoints (519fc9a0) |
| 8.3 | Render the template against transformers | Byte-equal with thinking on and off, tools, tool results, a late system message and templates that rewrite earlier turns. Add the model to `tests/test_prompt_parity.py`. Bolt, LFM2.5 and FP4-Local joined it in 519fc9a0. LFM2.5 with thinking off is a strict xfail: the Mac path renders `<think></think>` and CUDA `<think>` |
| 8.4 | Compare `/tokenize` with `apply_chat_template` ids | Equal (Qwen: 4 of 4) |
| 8.5 | Test the thinking switch on the served route | Count reasoning tokens a reply. Z.ai's switch returned reasoning in 56 of 152 and 159 of 160 replies |
| 8.6 | Set a thinking budget | `--thinking-budget 8000` in `run.sh`. Unbudgeted GLM turns ran 17,410 and 21,193 tokens, one past 998 s |
| 8.7 | Check reasoning markup | No `<think>`, `</think>` or channel text in `content`, streamed or not, thinking on or off |
| 8.8 | Parse the model's own tool envelope | Calls arrive as `tool_calls`, streamed and not, single and parallel, and with `parallel_tool_calls: false` (LFM2 leaked there, 4b395750. Gemma 4 and DeepSeek-V4 leaked there, 9d420f42) |
| 8.9 | Feed the parser hostile text | Another model's opener inside prose keeps the real calls (ba00b9d8) and the streamed text after it (1200f819, `tests/test_tool_call_mentions.py`). Non-object `parameters` gets no 500 (a42d5222). Parse time stays linear (6.1 s at 8,000 keys before 05dd2367) |
| 8.10 | Repair only what then parses strictly | A call cut by the end token, or missing `<arg_key>`, is closed only if the result parses whole (d45e5cb4) |
| 8.11 | Decode typed arguments by schema | Python-spelled `False` becomes a boolean. Upstream `lockDoors(unlock="False")` unlocked the doors |
| 8.12 | Write a streamed leak test | With tools and thinking on, `content` holds no markup and the reply ends on an EOS from 2.7 **(gap)** |
| 8.13 | Stop every decode loop on its feed's stop | A budget cut, a disconnect, an end token mid-round and a count cap each end the round. `requests_running` drops to 0 within seconds of a disconnect. GLM decoded 13 min for a departed client before c40e7e7d |
| 8.14 | Match stop strings token by token | A wide drafted round cannot move the cut |

### Phase 9. Prompt reuse and disk states

| # | Do | Pass when |
|---|---|---|
| 9.1 | Keep states at the system block's end, each message start, the last user message's start and one token before the prompt's end | A history sent five times with different questions resumes on every send after the first (GLM: 0 cached before 14200b09, 6,073 after) |
| 9.2 | Test with a pattern the engine can resume | A regenerated turn and a growing conversation (Part 2.5) |
| 9.3 | Evict superseded turns first and keep branch points | A history resumed after a later turn extended it survives until LRU |
| 9.4 | Key snapshots on their block names as well as their ids | Equal ids do not mean equal rows: decoded rows differ from a cold prefill (c5f31a9f) |
| 9.5 | Write disk states and read them back | Disk equals memory bit for bit. Qwen read 12,071 tokens back in 1.6 s with the cold reply's hash |
| 9.6 | Fingerprint everything that changes bits | Shards, toolchain, GPU and all code (27b8c2a2) |
| 9.7 | Make disk failures harmless | A failed read prefills and frees its slot (bc0a381c). Idle writes yield within one block (7643e8fb). A `length` finish is persisted (b3801bd1) |
| 9.8 | Prune | TTL 72 h on unused states and old builds' folders, one budget over every folder |
| 9.9 | Answer `/internal/persist` | A model without disk states answers `{"disk": false}` at once |

### Phase 10. Speed

| # | Do | Pass when |
|---|---|---|
| 10.1 | Estimate serial decode from bytes a token (Part 2.1) | The estimate is written before the first GPU run |
| 10.2 | Measure serial decode, then launches a window | Launch-bound decode is identified (Bolt: about 1,300 launches a window, 29 tok/s against 9 ms of reads) |
| 10.3 | Add CUDA graphs to launch-bound decode | Graph replay equals eager (5.8) |
| 10.4 | Measure cold prefill at 4k, 32k and 131k against the fastest engine on the same checkpoint | The gap is known. Work the ideas in Part 2.3 when it is 1.4x or more |
| 10.5 | Measure drafting per text kind | tok/s and accept length on code, prose, chat, tools and long, greedy and sampled, 4+ seeds. Drafted beats serial on every kind, or depth adapts. Bolt at depth 7 ran prose at 22.4 against 29.0 serial. A zero acceptance rate is a bug even with exact tokens (upstream: a draft head read at group 64 for 32) |
| 10.6 | Add copy drafts for code and edit traffic | Qwen EXL3: a 2,252-token prompt 36.5 to 89.0 tok/s (e773dbdb) |
| 10.7 | Count a speed number only from a GPU A/B | Arms alternated on the same image and box, `tools/bench_openai.py --tokens 256 --reps 5 --temperatures 1.0,0`, token SHAs checked first. Two GB10s ran identical code 4-6% apart. Run no CPU-heavy job beside it: on 18 Oct two full CPU test suites during a prefill A/B spread one config's 32k prompts over 17.8-24.6 s and flipped 4k between 1.10x and 0.72x |
| 10.8 | Before tuning a decode kernel, take the row counts from the drafter (Part 2.10) | The served verify-window rows are written down per mode: one stream, four streams, serial. Every kernel sweep times those rows and states its gain at them, not at 1 row |

### Phase 11. Serving wiring

| # | Do | Pass when |
|---|---|---|
| 11.1 | Pin the download in `prepare.sh` | `(repo rev folder)` array plus a disk-space `need`, and `fetch` dies without `config.json` |
| 11.2 | Write the `run.sh` start function | It checks `config.json` before stopping anything, then `image`, `stop_all`, `serve`, `wait_ready`, then `d1` when room allows (bolt and lfm2 are the templates) |
| 11.3 | Register the container in `NAMES` and the checkpoint in `uncache` | `stop_all` and `status` see it, and the next load does not evict its page cache page by page |
| 11.4 | Add the dispatch and `flock` cases | The line reads `    <target>) ask_swap "$1" \|\| <fn> ;;` |
| 11.5 | Add the `swap.py` `TARGET` entry and the served name in `tests/test_swap.py` | The served name equals `--name` and `pytest tests/test_swap.py` passes |
| 11.6 | Check `/v1/models` | `owned_by: "tensorfold"`, or the gateway's `reasoning_loop` act falls back to shadow |
| 11.7 | Check the gateway's assumptions about output format | Reasoning in `delta.reasoning` or `reasoning_content`. `reasoning_loop` close prefills `<think>…</think>`, so other markers need a gateway change before act. Clef arms only with `</think>` in the vocabulary |
| 11.8 | Parse every flag at startup | Unknown values raise and unused options are refused before weights load (dae361c3, a06dad1b, 269f7479) |
| 11.9 | Ship as an opt-in swap target first | The default stays unchanged until Phase 13 decides (ceb12076) |

### Phase 12. Served checks on :31005

| # | Do | Pass when |
|---|---|---|
| 12.1 | Smoke | Plain chat, a tool call, a tool result answered, a streamed call whose arguments parse as JSON, and thinking on split into `reasoning_content` and `content` (`tf-runs/super/smoke.py`, `tools/lfm2_moe_smoke.sh`) |
| 12.2 | Served exactness | `tf-runs/exact.py` drafted equals serial, golden replies match |
| 12.3 | Served reuse | A resumable pattern shows cached tokens and the same reply (`tools/glm_message_stops_check.py`) |
| 12.4 | Disk states across a restart | Same reply hash as a cold server |
| 12.5 | Co-residence in production start order | Available memory after load and after a request, with Clef and d1 |
| 12.6 | Empty and looping replies | A 50-seed sweep with `return_token_ids` gives the empty rate with its 95% interval (Qwen: 8 of 200, 1.7% to 7.7%) |
| 12.7 | Behaviour past the window | Refusal or truncation is known and the gateway handles it **(gap: untested on every model)** |
| 12.8 | Persist and swap | A swap persists first, a cut swap skips it, and a swap back resumes |

### Phase 13. Quality and the production decision

| # | Do | Pass when |
|---|---|---|
| 13.1 | Run TrueScore 76x2 and planning x4 under the contract (Part 2.6) | Both finish with transcripts saved in their own directories |
| 13.2 | Run a thinking-on arm for any planner role | Thinking-off planning measures visible reasoning: GLM scored 45.4 thinking off and 99.0 thinking on (13 scenarios) |
| 13.3 | Count reasoning tokens and the budget-hit rate per arm | Arms compared only with thinking matched (LFM2.5's median planning reply was 8,119 tokens against an 8,000 budget) |
| 13.4 | Pair against the incumbent from the same session | The 95% interval of the paired bootstrap clears zero for a claimed lead |
| 13.5 | Read the sub-scores and the failures | A lead is checked for Calibration weight (Super VL: 94.1 weighted, 91.7 mean) and for format zeros (Bolt dropped `}` in 35 of 160 planning replies) |
| 13.6 | Decide speed, memory, swap cost and role | Part 2.7 |
| 13.7 | Measure quantization loss against the maker's model | Local quantized GLM against Z.ai's API: +1.7 [-1.7, +5.3], thinking leaks counted |

### Phase 14. Review and merge

| # | Do | Pass when |
|---|---|---|
| 14.1 | Keep the CPU baseline at 0 failed | Platform-only tests skip by marker. A baseline of 129 failures hid new ones (924aa4d4) |
| 14.2 | Assert invariants in tests | Reply equality and cached ≥ expected, with fakes carrying every attribute the code reads |
| 14.3 | When a fix reverses a behaviour, grep the tests for the old assertion | A GLM test still asserted what 3bde5bbe removed on purpose (18 Oct) |
| 14.4 | Change a golden count or tolerance only with the measured gap in the commit | Bolt's 95 to 94 of 96 cites the 0.25 chat gap under a 0.77 error |
| 14.5 | Give hardening fixes a test on a resource they do not own | They fail open with one warning. a7ff73b3 failed every request on a foreign-owned log |
| 14.6 | Merge into `fork/dev` only after the GPU tests and a served check pass | Bolt, d1 and LFM2 merged with GPU tests "written, not run", and the integration run then found 1 failure in 560 |
| 14.7 | Review after every few merges | The review records its range in TASKS.md, and its fixes change no reply and no speed |
| 14.8 | Write the recipe | `docs/recipes/<model>.md` with the flags, the numbers, the gates any default breaks, and the date. Bolt and FP4-Local have none yet |

## Part 2. Notes

### 2.1 Bytes a token

A single decode stream on the GB10 is bound by the bytes each token reads. Sum them from the headers: the dense
decoder's tensors, the head, and the routed experts times top-k over the expert count. Skip vision, MTP and n-gram
tables, whose rows a token barely reads.

| Read per decoded token | EXL3 3.05 bpw | FP4-Local (NVFP4 + MXFP8) |
|---|---:|---:|
| Routed experts (10 of 512 a layer) | 0.89 GB | 1.26 GB |
| Attention, GDN, shared expert, hyper-connections | 3.32 GB | 4.51 GB |
| lm_head | 0.40 GB | 1.27 GB |
| Total | 4.62 GB | 7.04 GB |

The ratio predicted FP4-Local at 0.65x the EXL3 pack. Served with MTP drafts on 17 Oct it ran at 0.71 to 0.82x (67.8
against 83.0 tok/s on code, greedy), because drafting spreads the dense reads over a verify window's rows while the
experts' reads grow with the rows' distinct picks. Measure drafted speed separately from the estimate.

### 2.2 Size a stream, then check what the engine does with it

One stream holds its KV cache (attention layers x 2 x KV heads x head dim x bytes a value, per token) and its fixed
recurrent state (Mamba or DeltaNet heads x head dim x state x 4 bytes, per layer). Divide the memory the weights leave
by one stream at the window you want.

| Per stream | Nemotron Bolt | Qwen3.8 Flash Next (int8 KV) | Super VL | LFM2.5-24B |
|---|---:|---:|---:|---:|
| KV a token | 8 KiB | about 18 KiB | 8 KiB (9 with MTP) | 20 KiB |
| At 262,144 tokens | 2 GiB | about 4.5 GiB | 2 GiB | 5 GiB |
| Recurrent state | about 160 MiB | DeltaNet, same order | Mamba, not measured | not measured |
| Weights | 23.2 GiB | 52 to 78 GiB by checkpoint | 74.2 GiB with the image tower (load log, 18 Oct) | 24.75 GB |

A small state can hold dozens of long streams only if the engine serves them. On 17 Oct Bolt, GLM and LFM2.5 served one
stream, so their aggregate decode stayed flat at 20, 26-30 and 42 tok/s while Qwen and Super VL scaled to 143-181 at 8
streams. Upstream added Nemotron prompt reuse in 1.0.5 and concurrent rounds in 1.0.9. Those ports turn a small state
into served capacity.

### 2.3 Cold prefill

Compare cold prefill with the fastest engine serving the same checkpoint, at one request. Decode can match while
prefill does not: Mia's Zig recipe and our engine decode FP4-Local at 67.9 and 67.8 tok/s, while Mia prefills 2,583 tok/s
at 8k against our 1,857 (different tools, 1.4x). Investigate these on every new model, each behind a flag, keeping the
golden checks and resumed equals fresh.

| Idea | Where it showed up | Measured |
|---|---|---|
| Decode each used expert's tiles once per prompt chunk, sharing activation rows in shared memory | Our NVFP4 kernel re-decodes every weight fragment for each 16-row pass. Mia's patch 0014 (`fn_experts_tile.cu`) keeps the same bits | +13-15% prefill on Mia's engine (4k 2,325 to 2,665) |
| Read an expert tile once per 32 rows | Qwen EXL3 (d1964f99) | 16k 780 to 828 tok/s, bit-identical |
| Wider prompt expert windows | GLM 128 to 1,024 rows (fca0eca8) | 247 to 314 tok/s, same bits |
| Decode experts by broadcast into one reused buffer | LFM2.5 moved about 500 GB a chunk through two `repeat_interleave`s (de0b3019) | 5 bytes a weight |
| A tensor-core grouped prompt kernel | LFM2.5 (`perf/lfm2-prompts`, d6319a1e) | 1k 1,228 to 4,898 tok/s, 16k 1,792 to 3,920 |
| Keep short chunks on the decode kernel up to the measured crossover | LFM2.5 `PAIRS_ROWS` from `tools/lfm2_moe_prompt_kernels.py` | pairs wins through 64 rows (1.420 against 1.564 ms a layer) |
| Larger prompt chunks when memory allows | Upstream 8,192-row chunks | 2,000-2,500 to 3,300-3,600 tok/s on Flash Next |
| 4-bit activations in prefill | Upstream `--precision checkpoint` | 1.81x prefill at a KL of 0.0077: replies change, opt-in only with a TrueScore pair |
| Fair fill of mixed loads | Mia's `TF_FLASHNEXT_FILL_FAIR=1` | 20% slower on mixed loads. We already decode inside every prompt pass |

A chunk size that follows the memory budget makes replies differ across machines (upstream 0.3.6 known issue). Fix
the chunk size per checkpoint.

### 2.4 Drafting

A drafter (MTP head, DFlash, LiLiCorr) needs its target layers' taps (Bolt's LiLiCorr reads layers 1, 18, 35, 51, 68
and 85) and must hold the prompt's context before its first round. Time depth costs at load on realistic text: a table
timed on the MTP head's own looping chain came out 7-10% too cheap upstream. A change that keeps the tokens can still
move acceptance (the GLM splice gave 83.4 against 76.7 on one seed and 89.5 against 89.1 on another), so check both
the equality gate and the speed.

### 2.5 Prompt reuse

Check where the engine keeps states before testing how it evicts them. GLM kept states at the prompt's end and the
system block's end only, so a history sent five times, each with a different user message, never resumed.
Measure a multi-turn session's prefill time per turn before and after each mechanism.

| Mechanism | Measured |
|---|---|
| Background prefill of the next turn's shared prefix | 4.77 s against 5.51 s over 8 GLM turns |
| Kept states on disk | An 11,735-token turn in 13.9 s against 38.0 s, same reply |
| Gate cuts spliced into the live stream | 6.5-6.9 s of prefill a cut fell to 0.18-0.47 s |
| A kept state at the system block's end | First token 13.7 to 0.33 s (8155c5af) |
| Branch points (upstream #735) | An agent restarting each step from one history stops reading it again |

### 2.6 Benchmark contract and pitfalls

Contract (`tf-runs/truescore-queue.sh`, `tf-runs/bench2.sh`): `spark_bench.py eval --repeats 2 --temperature 0.3
--thinking off --uncapped --parallelism 1`, then planning with `run_ext.py eval --scenario-ids planm --repeats 4
--skip-throughput`. The body carries `chat_template_kwargs` with `enable_thinking` false and `reasoning_effort` xhigh.
Set `SPARK_BENCH_RANDOM_SEED=1` and `--max-tokens` equal to the context. Serve on :31005 behind the guard and run the
smoke first. Pair arms by scenario with `tf-runs/truescore/paired.py` (bootstrap over scenario means) and use McNemar
for pass/fail rates. At 76x2 the interval's half-width runs 3.5 to 5.5 points, and planning x4 gives 2 to 8.

| Pitfall | What happened |
|---|---|
| Prompt-hash seed | Without `SPARK_BENCH_RANDOM_SEED` every repeat copies the first |
| Switch that does not hold | Z.ai: reasoning in 56 of 152 and 159 of 160 replies. LFM2.5 always thinks |
| Thinking-off planning | Qwen writes a median 2,614 tokens of visible reasoning, GLM 188, Super VL 138, Bolt 121 |
| Budget truncation | LFM2.5's planning median sat at the 8,000-token budget and the run took 443 min |
| Weighted score against scenario mean | Calibration carries 25%: Super VL 94.1 against 91.7 |
| Cross-session drift | One Qwen config ranged 84.7 to 93.6. Pair within one session or re-run the reference arm |
| Grader defects | CODE-12 capped at 0.5 by `RLIMIT_NPROC`, CODE-13's mock lacks `getcode()`, TUH-08 cannot pass, 14 of 26 code and agentic scenarios saturated (`spark-bench-ext/AUDIT.md`) |
| Unread failures | Bolt's 35 planning zeros were a missing `}`. The runaway guard may have fired falsely on PO-M07 r2 |
| Aborted runs | Keep them in their own directories (`truescore-aborted-*`) |
| Time | TrueScore 30-131 min an arm, planning 10 min (Bolt) to 443 min (LFM2.5) |

### 2.7 The production decision

A model earns a swap slot when it leads the incumbent on a role, with the paired interval clear of zero, at a speed
and memory cost the role can carry. Check these in order:
1. Quality per role: planning Q with thinking on for a planner, TrueScore code, agentic and tool scores for an executor
   (Bolt: CODE -9 against FP4-Local).
2. Speed: single-stream decode, prefill at 1k, 8k and 32k, aggregate decode at 1 to 8 streams, and TTFT under
   concurrency. A planner and executor split pays only if the executor is clearly faster at equal quality.
3. Memory beside Clef and d1. A model that leaves under about 20 GiB loses both (GLM).
4. Swap cost: load 16-225 s, plus up to 120 s of persist, plus the lost warm cache. With no swap between planner and
   executor, both must co-reside (open in TASKS.md).
5. Role: default, planner and hard reasoning, executor, router or guard.
6. Licence, drafters included.

### 2.8 Working on the shared GPU

Production serves from this box while experiments run. The request path is the gateway on :4000, `swap.py` on :30000
and the model on :30002, with Clef on :30001 and d1 on :30003.
- Pre-build and tag test images with the hash `run.sh` computes. `run.sh` prunes only images labelled with its own
  checkout, and an older build pruned others and caused "pull access denied".
- Use SIGKILL on test scripts that restore state in a trap: a trap that restores state can stop production on its way out.
- A killed build leaves `lock` and `.ninja_lock` in the `tensorfold-ext` volume, and the next build hangs. Remove them
  when no build runs.
- Stop a process by its PID or wait on a marker file. `pkill -f` and `pgrep -f` matched their own shell (exit 144, and
  a 10 h stall).
- Read `GATEWAY_API_KEY` from the environment and never print it or write it to a file. `TENSORFOLD_REQUEST_LOG`
  captures traffic for exact replays and is created 0600 because it holds prompts.
- Trust `GET :30002/health` `requests_running` to show a disconnect freed the GPU. Log lines matched the wrong request once.
- Tag client experiments with `X-Gateway-Tag` and read them from `mini-llm-gateway/gateway.db` (`requests`, `ts` in ms).
- Replay failing requests with `return_token_ids: true` to see the raw ids before any parser. The server refuses
  logprobs on tool or thinking requests, which blocked the empty-reply diagnosis **(gap)**.

### 2.10 Where to look for speed

Profile one decode step and one prompt chunk with nsys (`-t cuda-sw --cuda-graph-trace=node` on this image), set each
kernel's time against its byte bound, then sweep these patterns. Each found at least one win between 17 and 19 Oct.

| Pattern | How to spot it | Seen | Bits |
|---|---|---|---|
| Too few programs at small row counts | grid against 48 SMs at 1 row | FP4 shared expert (51a7daa1), GLM `_plain` | identical: split N, keep each output's K loop |
| Occupancy capped by registers or shared memory | `ptxas -v`, blocks a SM x bytes in flight a warp | EXL3 experts, 3 to 4 blocks a SM | identical if the sum order holds |
| Dependent loads before the first weight read | the kernel prologue's index, id, pointer chain | EXL3 experts, 4 chained reads | identical |
| Weights decoded again for every row tile | bytes read a chunk against weight bytes | expert tile (Mia 0014), +11-13% prefill | identical |
| Too few rows per weight read | achieved bandwidth already near peak | 4,096-row pieces, long prompts only | identical if chunk-invariant |
| Wrong kernel from a dispatch condition | kernel names per phase against intent | FP4 decode on the prompt kernel through a dtype test | usually identical |
| A fusion gated off | grep `isinstance` and flag gates | HC write-back + norm limited to Q4 | identical |
| Split-K partials plus a separate reduce | `_reduce`-style kernels after a matmul | HC down projection, +71 ms a chunk | identical in order; see row 6.9 |
| An intermediate written then read | producer and consumer pairs | HC up + mix | identical |
| Copies and layout conversions | memcpy and copy nodes a step | GLM, 1,067 memcpy nodes a token | identical |
| Work nobody reads | trace outputs to readers | GLM's dead `group_sums` | identical |
| Cost growing with context | per-chunk time against position | indexer +1.6 ms a chunk per 1k tokens | depends |
| Independent work in series | chains with no data dependency | shared expert beside routed experts (Mia) | identical if combined in order |
| Host syncs inside a round | `.item()`, `.tolist()`, blocking H2D, stream syncs a round | d1 head, MTP draft chain (1.8-4.7 ms idle a round) | identical for greedy |
| Launch overhead | launches a step x 0.7 us | PDL, 0.4-0.9% | identical |
| Allocator churn | `cudaMalloc` or `empty_cache` mid-request | 0.4-0.8 s slow windows after cache growth | identical |
| A one-stream engine | concurrent requests queue | GLM, Bolt, LFM2.5 | identical when concurrent equals solo |

Tune decode at the rows the drafter makes. A drafted model never decodes one row: each round verifies the drafter's
tokens as a window. Qwen's MTP verifies 5-7 rows a stream, so 20-28 rows at four streams. Serial decode is the only
1-row case. Read the window sizes from a served trace or the engine's `rounds` and `accepted` counts before writing a
kernel, then sweep at those rows. On 19 Oct an EXL3 expert kernel ran 6-9% faster at 1-4 rows and lost 2-14% at 5-7.
Served, it measured 0.994-1.001x and was dropped. The kernels that shipped the next day targeted the windows MTP
actually makes: grouping at 7 rows ran 2.4x faster, and paired linear tiles above 16 rows gave four streams 1.02-1.06x.
Even a kernel that wins at the right rows can lose served. GLM's expert kernel swept 7-23% faster at every row count,
then measured 0.978-0.990x in a real decode step, serial included.

Measure idle and launch delay with graph-level tracing (`--cuda-graph-trace=graph`) or none, and keep node tracing
for per-kernel times. On 20 Oct node tracing held `cudaGraphLaunch` 1.04-1.37 ms on the host and showed a 0.97 ms
launch-to-first-kernel delay a verify; untraced the call costs 3.2 us and splitting the graph saved 0.007 ms.
Do not time a kernel right behind `torch.cuda._sleep`: the graph after it ran 0.7-1.0 ms slower.

Before trusting a gap, rule out the measurement: a log figure that leaves out work (the done line's prefill missed the last
pass), contention (two CPU suites spread 32k prompts over 17.8-24.6 s), a cold page cache for mapped tables, and clock or
thermal throttling (log SM clocks during long runs). Under several streams, also check L2 thrashing: per-stream decode
falling faster than bandwidth predicts.

### 2.9 Upstream

Upstream TensorFold became a Zig engine at 1.0.0 (7 Oct 2026) and froze Python at 0.6.6. Read its CHANGELOG each
release and port by hand what fits. Its gates are worth copying as they stand: the kernel contract's five gates with
planted controls (`docs/kernel-contract.md`), 320 frozen server goldens replayed byte for byte, template and tokenizer
parity tools, and a release cell that visits base, candidate, candidate, base and calls a candidate slower only when
every cross ratio is below 1 and the mean below 0.98. Its stage ladder for a new family is in
`docs/recipes/adding-a-zig-family.md`, with a "Don't bother" list (Part 3.2).

## Part 3. Reference

### 3.1 Bug classes

| Class | Examples | Caught by |
|---|---|---|
| Format decode | CSF scales unread (d94901b7). Clef RMSNorm +1 (837a86e1). Sushi trellis and 14-bit window (4431d765, 12bb9b1d). An affine6 odd offset raised `cudaErrorMisalignedAddress` and poisoned the context (308a93e9). Three NVFP4 decoders and three IQ2_XXS grids side by side (f19e96fa) | Config inventory, byte compare, review |
| Row-dependent arithmetic | Bolt window over 64 rows took the prefill kernel (f8869622). Resumed tail on another kernel (269ce0b1). Upstream: MLX bf16 1 against 2 rows, DeltaNet below 4 rows | Drafted equals serial, row checks |
| Buffer sizing | Upstream "!!!!" past 2,051 tokens, empty replies past 12k from an unwritten `offsets[e+1]` | Long prompts |
| Kernel state | 688bc309 cached a per-kernel shared-memory ceiling per size | Full GPU test files |
| Templates and thinking | Guard window overflowed Clef's 16,384 (4762edaa). Sessions never closed (cd0653f7). GLM had no budget (2c1c9a49). Upstream: Jinja `capitalize` missing, "No user query found", `medium` rendered Max | Live runs, client sessions |
| Tool calls | `<arg_key>` lost or call cut by the end token (d45e5cb4). Quadratic parse (05dd2367). LFM2, Gemma 4 and DSML markup leaks (4b395750, 9d420f42). Opener in prose stole calls (ba00b9d8) and hid the rest of the stream (1200f819). Upstream: unoffered tool gave empty replies and agent loops | Clients, review |
| Stops | GLM's four decode loops ignored stop (c40e7e7d). A gate failing on the first token ran an extra round (ee825918) | Client sessions |
| Prompt reuse | Copy drafts never fired (e773dbdb). Lone graph slot dropped graphs 25 of 25 (cc097995). Oldest-first eviction lost conversations (ba0db4b2). Branch point evicted (844e2dd6). In-chunk cuts lost the DFlash2 window (3bde5bbe) | Served probes, review |
| Disk states | `length` finish never persisted (b3801bd1). Fingerprint gaps (27b8c2a2). Read failure leaked a slot (bc0a381c). A 64 GiB folder a build never deleted (bd1c8276) | GPU tests, review |
| Drafting | Wrong drafter signatures (0927ea2d). Drafter sampled differently (33378f6a). bf16 head read 3.5x the bytes (92cec6d8). Upstream: draft head at the wrong group size, exact tokens at 0.1-1% acceptance | Merges, golden runs |
| Memory | Page cache counted as used (f7807ae6). Drafter uncharged (ee83a076). DFlash2 cut GLM's window 131,072 to 110,298 (2389628f). A GLM boot within 30 s of a stop fails admission | Served starts |
| Ops | A failed load left `loading` set (e8a36aa3). HF token in `ps` (f152f382). A failed build under the good tag, `/internal/` forwarded, two direct runs loaded two models (7891345f) | Reviews |
| Observability | Missing `cached` reported as 0 (b2d83975). A flag typo read as off (dae361c3). 40 ms delayed ACK (134e0f4c) | Clients |

Upstream records no fix for RoPE or YaRN, norm epsilon, tied embeddings or softcapping in any family. We have no
RoPE or YaRN check against HF at 32k or 128k either **(gap)**.

### 3.2 Ideas that did not pay

| Idea | Result |
|---|---|
| Copy drafts on GLM's DFlash2 path | 0.963-1.000x |
| Expert skipping | 19 settings, no safe threshold, replies failed to stop |
| GLM tree drafts | Oracle sibling kept 16.8% of rounds against 34% needed |
| GLM `auto` drafter chooser | 3-8% slower than `fc5:0.3` |
| EXL3 expert tuning | Forecast 1.35-1.65x, measured +6% |
| EXL3 decode kernel tuned at 1-4 rows (`EXPERT_DECODE`, `F16_NARROW`, `PDL`, 19 Oct) | Served 0.994-1.001x each: MTP verifies 5-7 rows, where the tiles lost |
| EXL3 `EXPERT_DECODE` on GLM-5.3 (20 Oct) | Swept 7-23% faster, served 0.978x drafted and 0.989x serial |
| Qwen expert read-ahead thread | No gain, and a `torch.empty` on a worker thread kept 36 GB resident |
| Deferred n-gram read | First request 1.4-2 s slower |
| Fixed draft depth on Bolt prose | 22.4 against 29.0 serial |
| fp8 KV over int8 | Under 3% capacity (1,056 against 1,088 bytes a layer) |
| Upstream "Don't bother" | Grouped experts read once, one launch for all experts, a persistent kernel with a GPU-wide barrier, 128-row tiles, adaptive depth by landed+1, full-vocabulary drafts, lanes without a drafter |

### 3.3 What each model measured (thinking off unless noted, 17-18 Oct)

| Model | TrueScore 76x2 | Planning x4 (Q) | Decode, 1 stream | Prefill 8k | Weights | Decision |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.8 Flash Next EXL3 3.05 bpw | 87.8 | 89.0 (92) | 55-58 | 877 | 54.8 GiB | Production default, with Clef and vision |
| Qwen3.8 Flash Next FP4-Local | 91.0 | 89.8 (94) | 43-44 | 1,857 | 77.6 GiB | Swap target. EXL3 stays the default after the 18 Oct tie-breaker: FP4-Local - EXL3 +2.3 [-0.9, +6.4], pooled 76x4 |
| GLM-5.3 Flash local hybrid | 93.0 | 60.5 (58), 99.0 on 13 with thinking on | 29-30 | 414 | 101.1 GiB | Swap target. Planning to re-rank with thinking on |
| Nemotron 3.5 Bolt | 92.0 | 38.3 (27) | 22-24 | 2,139 | 23.2 GiB | Swap target. Graphs pending, dropped braces to check |
| Nemotron 3.5 Super VL NVFP4 (native 1.0.12) | 94.1 | 62.6 (59) | 32-33 | 1,844 | 74.2 GiB | Swap target, `run.sh nemotron-3.5-super` (c55dc1ff) |
| LFM2.5-24B-A2B FP8 (always thinks) | 91.7 | 58.6 (62) | 47-53 | 699 | 24.75 GB | Swap target. Prompt kernel 2.2-5.6x faster on `perf/lfm2-prompts` |
| GLM-5.3 Flash, Z.ai API | 91.1 | 72.5 (77) | n/a | n/a | n/a | Reference only. Thinking-off switch did not hold |
| Muse Spark Open | n/a | n/a | n/a | n/a | 175.6 GB smallest | Rejected: does not fit, AA 39 |

Only 1 of 21 TrueScore pairs has an interval clear of zero: EXL3 minus GLM, -5.2 [-10.7, -0.3]. Per-model details,
memory after load and load times are in `tf-runs/playbook-audit/3-models-and-decisions.md`.
