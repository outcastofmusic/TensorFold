# Bringing up a new model on TensorFold

This playbook lists what to check and what to optimize when a new model or checkpoint arrives on the DGX Spark (GB10,
121 GB unified memory). Each item comes from a bug found or a speedup measured between 15 and 17 Oct 2026, and names
the evidence. Work through the sections in order: a checkpoint that loads wrong makes every later measurement
meaningless.

## 1. Read the checkpoint before writing code

Start from `config.json` and the safetensors headers. Neither needs the weights downloaded: an HTTP range request
reads a shard's header (the first 8 bytes give its length).

| Check | How | What it caught |
|---|---|---|
| Every quantization recipe in the config | Group `quantization_config.quantized_layers` by its recipe and count layers per recipe | Qwen3.8-Flash-Next-FP4-Local stores its 48 routed-expert layers with `weight_scale_encoding: "csf"`, the only recipe TensorFold did not read |
| Tensor suffixes | Count the suffixes in `model.safetensors.index.json` | CSF replaces `weight_scale` with `weight_scale.nvfp4_csf_fixed` plus `weight_scale.nvfp4_csf_exceptions` |
| Shard paths | Read `weight_map` values | FP4-Local keeps its shards under `tensors/`, so the reader must follow the index's paths |
| The model card's container and flags | Read the vLLM command the card gives | It names the formats upstream vLLM lacks and the precision the authors serve with (FP4-Local: 4-bit weights with 16-bit activations in decode, 4-bit activations in prefill) |
| The family's own start-up checks | Run `detect`, `require_readable` and the family's `check()` on the real folder, on the CPU | Confirms the config passes before any GPU time is spent |

When a format is new, find its published decoder (vLLM, b12x or llama.cpp source) and port it as written. Then compare
your decoder with theirs on every tensor of the real checkpoint. FP4-Local's check covered all 73,728
expert matrices: all 73,728 decoded to the same bytes as vLLM's `decode_nvfp4_csf_scale`, with no NaN scales.

Check the decoded values against what the quantizer guarantees. ModelOpt sets `weight_scale_2 = amax / (6 x 448)`, so
the largest e4m3 block scale in each matrix sits at or near 448. FP4-Local gave a median of 448 and 645 of 913
matrices at exactly 448 in shard 1.

## 2. Decode at load, keep the kernels

A storage format that decodes losslessly to one the kernels already read costs nothing at decode time. Decode it once
at load and pass the old tensor on. FP4-Local's CSF scales decode to plain e4m3 scales, so the NVFP4 kernels run
unchanged.

Batch the load-time decode by layer. A decode that runs once per matrix pays kernel launches and host syncs on every
call. The first CSF decoder took 4.4 ms per matrix, about 5.4 minutes per load over 73,728 matrices. Decoding each
layer's projection as one 512-expert batch, in uint8 with one host sync, took 16.5 ms a batch, about 2.4 s per load.

## 3. Estimate speed from bytes before measuring

A single decode stream on the GB10 is bound by the bytes each token reads. Sum them from the headers: the dense
decoder's tensors, the head, and the routed experts times top-k over the expert count. Skip vision, MTP and n-gram
tables, whose rows a token barely reads.

| Read per decoded token | EXL3 3.05 bpw | FP4-Local (NVFP4 + MXFP8) |
|---|---:|---:|
| Routed experts (10 of 512 a layer) | 0.89 GB | 1.26 GB |
| Attention, GDN, shared expert, hyper-connections | 3.32 GB | 4.51 GB |
| lm_head | 0.40 GB | 1.27 GB |
| Total | 4.62 GB | 7.04 GB |

The ratio predicts the serial decode ratio before a GPU run: FP4-Local should decode at about 0.65x the EXL3 pack.
The table also shows where a speedup can come from. Here the experts account for 0.37 GB of the 2.42 GB gap and the
MXFP8 dense layers and bf16 head for the rest. Drafted decoding spreads the dense reads over a verify window's rows
while the experts' reads grow with the rows' distinct picks, so measure drafted speed separately. Served with MTP
drafts on 17 Oct, FP4-Local decoded at 0.71 to 0.82x the EXL3 pack (67.8 against 83.0 tok/s on code, greedy), faster
than the 0.65x the bytes predict.

## 4. Size a stream, then check what the engine does with it

Compute what one stream holds from `config.json`: the KV cache per token (attention layers x 2 x KV heads x head dim x
bytes a value) and the fixed recurrent state (Mamba or DeltaNet heads x head dim x state x 4 bytes, per layer). Then
divide the memory the weights leave by one stream at the window you want.

| Per stream | Nemotron Bolt | Qwen3.8 Flash Next (int8 KV) |
|---|---:|---:|
| KV cache per token | 8 KiB (8 attention layers, 2 KV heads of 128, bf16) | about 18 KiB |
| At the native 262,144 tokens | 2 GiB | about 4.5 GiB |
| Recurrent state | about 160 MiB (40 Mamba layers, 128 x 64 x 128, fp32) | DeltaNet, of the same order |
| Weights | 23.2 GiB | 52 to 75 GiB by checkpoint |

A model whose state is this small can hold dozens of long streams. Whether it does depends on the engine, so check
each of these before choosing what to serve:

| Engine feature | Qwen Flash Next | GLM-5.3 | Bolt (17 Oct) |
|---|---|---|---|
| Concurrent streams | 4 (`--parallel 4`) | 1 | 1: requests queue |
| Prompt reuse across requests | kept states at message starts, forks to spare slots | prompt end and system block end only, plus disk | none: every request prefills its whole prompt |
| Background prefill of the next turn | yes | yes | no |
| CUDA decode graphs | yes | yes | no |

Upstream TensorFold added Nemotron prompt reuse in 1.0.5 and concurrent rounds with batched drafting in 1.0.9. Those
are the ports that turn a small state into served capacity.

## 5. Prove the engine computes the model

These gates must hold before any speed number counts.

- **A reference forward.** Build an fp32 reference that decodes every weight exactly, and check it layer by layer
  against the checkpoint's own modeling code. For Nemotron Bolt the reference matched HF's `modeling_nemotron_h.py`
  within 6.6e-6 per layer on layers 0 to 8, once it applied HF's 1e-3 floor on the Mamba time step. Without the floor
  the gap was 5.6e-4. Find and record such differences between the HF path and the CUDA path (vLLM skips that floor).
- **Golden files.** Save the reference's top-20 logits for a chat, a tool and a long prompt, 32 greedy steps each.
  Record the smallest gap between a step's top two logits: an engine whose logits err by less than half that gap
  reproduces the greedy tokens.
- **Exactness of the serving paths.** Drafted replies equal `"draft": false` replies, resumed prompts equal fresh
  ones, and each of 2 and 4 concurrent streams equals its solo run. Check greedy and sampled.
- **A deterministic reference before reading probabilities.** exllamav3's cache-free forward gave "No" 0.287, 0.078
  and 0.094 on three identical runs (findings, 15 Oct). Run any reference twice and compare before trusting a
  probability from it.

## 6. Serving bugs to test for on every new model

Each of these shipped and was found by a client.

| Test | Failure seen | Fix and result |
|---|---|---|
| A reply cut by the thinking budget, and a client that disconnects mid-reply | GLM's decode loops ignored the stop and decoded to the model's own end token | Stop on the first stop: a cut reply went from 1,208.9 s to 111.5 s, and a disconnect frees the GPU in 2 s where it took 13 minutes (0.8.0) |
| A request with no `seed`, sent 10 times | The seed hashes the prompt, so one bad draw repeats on every send | Expected behaviour. Clients that retry should vary `seed` |
| Replies that end with `<|im_end|>` inside an open `<think>` | 8 of 200 draws empty on int8 KV and 8 of 200 on bf16 (McNemar p = 1.0) | The weights put real mass on that token. The KV dtype does not change the rate |
| Tool calls the model writes slightly wrong | A GLM argument key missing its `<arg_key>`, a call cut off inside its value | Repair the call when the repaired call parses strictly |
| Usage on streamed Anthropic and Responses requests | `input_tokens` missing from `message_start` | Count the rendered prompt in `app.prepare` |

Replay real failing requests with `return_token_ids: true` to see the raw ids before any parser. Use
`TENSORFOLD_REQUEST_LOG` to capture traffic for exact replays. The file is created 0600 because it holds prompts and
tool content.

## 7. Prompt reuse

Measure a multi-turn session's prefill time per turn before and after each of these.

| Mechanism | Measured gain | What to check |
|---|---|---|
| Background prefill of the next turn's shared prefix after each reply | 4.77 s against 5.51 s over 8 GLM turns | It steps aside when a foreground request waits |
| Kept states written to disk (`TENSORFOLD_GLM_STATES`) | A turn of 11,735 tokens prefilled in 13.9 s against 38.0 s, same reply | It persists after every request, including a `length` finish |
| Gate cuts spliced into the live stream | 6.5 to 6.9 s of prefill per cut fell to 0.18 to 0.47 s | Replies with no cut keep their hash |
| Branch points in the eviction rule (upstream #735) | Upstream: an agent that restarts each step from one history stops reading it again | A history resumed after a later turn extended it is freed only by LRU |
| A kept state at the system block's end | Another conversation with the same tools resumes from it | Touch it on every use so long sessions don't age it out |

Check where the engine keeps states before testing how it evicts them. GLM used to keep a state at the prompt's end and
the system block's end only, so a history that several requests share, each followed by a different user message,
never resumed: on 17 Oct a 6,059-token history sent five times with different questions showed 0 cached tokens every
time. GLM now also keeps the last user message's start, inside the prefill pass at no extra pass, and the same test
resumes 6,073 tokens on every question after the first (served check, 17 Oct). A served cache test must use a pattern the engine can resume (a regenerated turn, a growing
conversation) before it can test eviction.

## 8. Drafting

Measure acceptance per round and tokens per second on the served path, greedy and sampled, on at least 4 seeds. A
change that keeps the tokens can still move acceptance. The GLM splice gave 83.4 against 76.7 acceptance on one seed
and 89.5 against 89.1 on another. Check both the drafted equals serial gate and the speed.

A new model's drafter (MTP head, DFlash, LiLiCorr) needs its target layers' taps (Bolt's LiLiCorr reads layers 1, 18,
35, 51, 68 and 85) and must hold the prompt's context before its first round.

## 9. Working on the shared GPU

Production serves from this box while experiments run.

- Production's request path is the gateway on `:4000`, then `swap.py` on `:30000`, then the model on `:30002`, with
  the thinking guard on `:30001`. The gateway scans ports 30000 to 30010, so a test instance on `:30005` received real
  traffic. Run test instances on `:31005`.
- Guard every GPU experiment: start only when `free -g` shows the memory the job needs plus a margin, kill it when
  available memory falls below 12 GiB, and kill it the moment `GET :30000/swap` shows a load starting. A GLM swap
  during a bf16 test failed with "memory budget cannot fit" and removed production's containers.
- Coordinate GPU jobs from one place. Two agents' jobs together drove production's available memory to 2 GiB.
- `run.sh` deletes older `tensorfold-run` images when it builds a new one. Pre-build and tag test images with the hash
  `run.sh` computes, or a later start fails with "pull access denied".
- Killing a script that traps exit can stop production on its way out. Use SIGKILL on test scripts that restore
  state in a trap.
- A killed build leaves `lock` and `.ninja_lock` in the `tensorfold-ext` volume, and the next build hangs. Remove them
  when no build runs.
- Read `GATEWAY_API_KEY` from the environment. Never print it or write it to a file.
- Stop a process by its PID. `pkill -f <pattern>` matched the shell that ran it and killed the command (exit 144).

## 10. Measuring

- Speed: `tools/bench_openai.py` with `--tokens 64 --reps 5 --temperatures 1.0,0`, code and chat cells, medians, both
  configs on the same image and box, alternated.
- Comparisons: pair draws by (request, seed) and use McNemar for pass/fail rates. At 8 in 200, a rate's 95% interval
  is 1.7% to 7.7%, so a few points' difference needs more draws.
- Client experiments: tag requests with `X-Gateway-Tag` and read them from `mini-llm-gateway/gateway.db` (`requests`
  table, `ts` in milliseconds).
- Quality: TrueScore 76 scenarios x 2 and the planning measurement set (PA-M, PO-M) x 4, thinking off,
  `reasoning_effort` xhigh in `chat_template_kwargs`, temperature 0.3, uncapped, `SPARK_BENCH_RANDOM_SEED=1` (without it
  TensorFold's prompt-hash seed makes every repeat a copy), and `--max-tokens` equal to the context. The runner is
  `tf-runs/truescore-queue.sh`.
- Compare arms run in the same session, scenario by scenario, with a paired bootstrap over scenario means
  (`tf-runs/truescore/paired.py`). Single TrueScores of one Qwen config on this box ranged from 84.7 to 93.6, mostly
  through Calibration. On 17 Oct FP4-Local beat the EXL3 pack by +2.3 [-1.1, +6.4] on TrueScore and +1.5 [-0.4, +3.5] on
  planning. Both intervals include zero.
- Measure quantization loss against the full-precision model on its maker's API. Z.ai's coding endpoint serves
  `glm-5.3-flash`, but `"thinking": {"type": "disabled"}` did not hold: it returned reasoning in 56 of 152 TrueScore replies
  and 159 of 160 planning replies (17 Oct). The local quantized GLM, thinking off, scored 93.0 against the API's 91.1,
  a paired +1.7 [-1.7, +5.3], so the quantized model lost nothing measurable even against a partly thinking reference.
  Count reasoning tokens per reply before comparing an API arm with a thinking-off arm. The API's speed scores (Resp,
  Eff) measure the service and do not compare with local runs.
- Production health: `GET :30002/health` gives `requests_running`, which is the reliable signal that a disconnect freed
  the GPU. Log lines matched the wrong request once.

## 11. Upstream

Upstream TensorFold became a Zig engine at 1.0.0 (7 Oct 2026). Read its CHANGELOG for each release and port by hand
what fits. Ported or in progress from 1.0.9 to 1.0.12: background prefill, disk states, GLM tool-call repair, branch
points (#735), the owner-only request log, the linear-time GLM key parse (#883) and DeltaNet scan marks (#851). Left
out: `--precision checkpoint` (NVFP4 activations), which prefilled 1.81x faster upstream at a mean KL of 0.0077 nats a
token and so changes replies.
