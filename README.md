# TensorFold, Python 0.6 fork

This repository is a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) that stays on the Python
engine. Upstream rewrote TensorFold in Zig for its 1.0 line and froze the Python code on the `python-0.6` branch, 12
commits past v0.6.6. The fork starts from that branch (`ed78d6fc`) and serves two models on one DGX Spark from one
tree: Qwen3.8 Flash Next and GLM-5.3 Flash.

Branch `fork/dev` is the one to use. Everything below "Upstream README" is upstream's text, unchanged.

## Run it

Two scripts take a fresh clone to a running server on one DGX Spark. The host needs:

- Docker with the NVIDIA container runtime (`docker run --gpus all` works).
- `python3`, for the swap endpoint. Its standard library is enough.
- 281 GB of free disk for both models (99 GB for Qwen alone).
- `HF_TOKEN` in the environment if Hugging Face refuses or rate-limits a download.

```bash
git clone -b fork/dev https://github.com/outcastofmusic/TensorFold.git && cd TensorFold
./prepare.sh      # download both models; ./prepare.sh qwen-flash-next or ./prepare.sh glm-5.3-flash fetches one
./run.sh          # serve both on http://127.0.0.1:30000/v1, Qwen loaded first
curl -s http://127.0.0.1:30000/swap    # {"loaded": "qwen3.8-flash-next", ...} once it is up
```

`./run.sh` with no argument runs `./run.sh swap qwen-flash-next`, the setup the numbers below come from. One endpoint
serves either model, picked by the request's `"model"`. `qwen3.8-flash-next` gets Qwen with Clef-Flash beside it as the
thinking guard, and `glm-5.3-flash` gets GLM-5.3 Flash alone. Qwen loads first, and a request for the other model swaps
it in. `./run.sh qwen-flash-next` or `./run.sh glm-5.3-flash` serves one model with
no swap endpoint. On this branch the think-loop detector is on (`TENSORFOLD_THINK_LOOP`, [TANDEM-EVAL.md](TANDEM-EVAL.md)).

Three more models swap in the same way once `./prepare.sh qwen-flash-next-fp4`, `./prepare.sh nemotron-bolt` or
`./prepare.sh lfm2-24b` has fetched them: `qwen3.8-flash-next-fp4` (NVIDIA's QAD FP4-Local checkpoint, Clef-Flash
started before it, text only), `nemotron-3.5-bolt` (Nemotron 3.5 Bolt 74B-A9B with its LiLiCorr draft model, one
stream, prompt reuse in memory) and `lfm2.5-24b-a2b` (Liquid's LFM2.5-24B-A2B in block FP8, one stream; LFM Open
License v1.0, commercial use only below US$10M annual revenue).
Measured on 17 Oct 2026 against the EXL3 pack on one Spark, FP4-Local decoded at 0.71 to 0.82x its speed and scored
TrueScore 91.0 against 87.8, a paired +2.3 [-1.1, +6.4] over 76 scenarios.

Once `./prepare.sh d1` has fetched it (6.3 GB), `D1=1` makes every model target but GLM also start Liquid AI's
d1-3B decision model on `127.0.0.1:30003` after the model answers. It answers yes/no, choice and score questions about a state with
probabilities (`POST /v1/systemone`, `/v1/decisions`) and writes no text. It answered one question in 25.9 ms on the GB10 (2026-10-17). Nothing calls it yet:
[docs/d1-sidecar.md](docs/d1-sidecar.md) gives its API, memory and three candidate uses.

`prepare.sh` downloads the checkpoints into `~/models` (set `MODELS` for another place) at the revisions the numbers
below were measured on, and converts Qwen's image tower once. Downloads run inside the serving image
(`nvcr.io/nvidia/pytorch:26.07-py3`), so the host needs no Python packages.

| Target | Downloads | Disk |
| --- | --- | --- |
| `qwen-flash-next` | `turboderp/Qwen3.8-Flash-Next-exl3` (3.05 bpw), `Cloudflare/clef-flash`, a converted image tower | 99 GB |
| `glm-5.3-flash` | `0xSero/GLM-5.3-Flash-EXL3-Spark` (M288 Mosaic 12L), the routed experts of `beamster/GLM-5.3-Flash-Sushi-2.4bpw`, `cfontes/GLM-5.3-Flash-DFlash2-TR3-v3` | 182 GB |

`run.sh` serves GLM as a hybrid that `prepare.sh` assembles with `tools/glm5_stacked_experts_pack.py`: Sushi's
2.25/2.5-bit routed experts on Mosaic 12L's dense part. Over TrueScore's 76 scenarios it scored 93.6 against Mosaic's
86.5, a paired difference of +2.7 [-1.1, +7.3], and decodes 5-7% slower (2026-10-13). `GLM_PACK=mosaic ./prepare.sh
glm-5.3-flash` fetches Mosaic alone (93 GB), and `GLM_MODEL=~/models/glm53-flash-mosaic-12l ./run.sh glm-5.3-flash`
serves it.

`run.sh` serves on `http://127.0.0.1:30000/v1`, an OpenAI-compatible API. The two models do not fit in memory
together, so starting one stops the other: a switch to Qwen takes 23 to 24 s and a switch to GLM 21 to 24 s
(7 switches each, 2026-10-13).
The first start after a source change takes 4 to 5 minutes more, because `run.sh` installs the checkout into a local
image and compiles the CUDA extensions and Triton kernels into the `tensorfold-ext` Docker volume. Later starts reuse both.
`./run.sh stop` and `./run.sh status` do what they say. The server listens on loopback and checks no API key.

`./run.sh swap` starts one endpoint for both models on the same port ([`swap.py`](swap.py), Python's standard library).
It loads the model a request names in `"model"`, so a client switches by changing that field. A request for the model
that is not loaded waits for the running requests to finish and for the load. After `SWAP_DRAIN` seconds (600) the swap
cuts the requests still running off. Requests already waiting for the model just loaded run before the next swap. A
streamed request receives an SSE comment every 5 s while it waits, which keeps clients with a read timeout connected.
`SWAP_HOLD=120` keeps each model loaded for at least that many seconds: with two clients on two models, each request
otherwise pays a swap.

Qwen (both checkpoints) and GLM write their kept prompt states to the `tensorfold-states` Docker volume while no request runs. A
conversation's next turn after a swap or a restart reads its state back instead of prefilling the whole conversation.
On GLM, an 11,735-token turn resumed 7,445 tokens from disk in 0.31 s and prefilled in 13.9 s instead of 38.0 s, with
the cold reply's tokens (2026-10-16). A Qwen state at `--kv-dtype int8` takes 115.7 MB plus 18,304 bytes a token,
1.03 GB at 50,000 tokens. Qwen writes each slot's shortest and longest states only, so a regenerated earlier turn after a restart prefills from the latest state on disk it extends (`STATES_ALL=1` writes every kept state). `STATES=0` keeps states in memory only, `STATES_GIB` (64) caps each model's folder with every build's states in it, and a state unused for `STATES_TTL_H` hours (72) is deleted. Before a
swap stops a model, `swap.py` asks it to write the states not on disk yet (`POST /internal/persist`) and waits up to
`SWAP_PERSIST` seconds (120).

| | `./run.sh qwen-flash-next` | `./run.sh glm-5.3-flash` |
| --- | --- | --- |
| Model id | `qwen3.8-flash-next` | `glm-5.3-flash` |
| Window | 262,144 tokens | 98,304 tokens |
| Streams | 4 (`--parallel 4`, `--kv-dtype int8`) | 1 |
| Drafts | MTP head, 1 to 6 a round, and copy drafts in windows of 7 to 32 rows | DFlash2 draft model, up to 5 a round, and copy drafts in windows of 6 to 32 rows |
| Thinking | Clef-Flash guard on port 30001, 8,000-token budget | No guard: 8 GiB stays free beside GLM |
| Images | On | Off |
| Decode, one stream | prose 60, code 98, json 118, rewrite 124 tok/s | prose 34, code 51, json 56, rewrite 60 tok/s |
| Copied text, 20 Oct | A 40-line list 244 tok/s, a 60-line module rewritten 237 (one stream, before the furthest-match choice) | The same list 146 tok/s, the module 100 |
| TrueScore, 76 scenarios x 2 | 88.4 in 78 min, thinking on | 87.1 in 43 min, thinking off |

Each default is the setting that measured best on one Spark. The script's header lists the environment variables
that change them. Three choices need a word:

- **GLM's draft model** raises decode from 29 / 37 / 45 / 52 to 34 / 51 / 56 / 60 tok/s with the same replies, and
  lowers the largest window from 131,072 to 110,298 tokens. `cfontes/GLM-5.3-Flash-DFlash2-TR3-v3` is a finetune of
  `incoai/GLM-5.3-Flash-DFlash2`, whose license is CC-BY-NC-ND-4.0. `DRAFTER=modal ./run.sh glm-5.3-flash` serves
  `modal-labs/GLM-5.3-Flash-DFlash` (MIT) at 33 / 49 / 55 / 57 tok/s. Fetch it with
  `GLM_DRAFTER_REPO=modal-labs/GLM-5.3-Flash-DFlash ./prepare.sh glm-5.3-flash`. `DRAFTER=none` serves without one.
- **GLM's dense prompt matmul** (`TF_GLM_DENSE_PROMPT=1`, set by `run.sh`) reads a 20,500-token prompt at 362 tok/s
  against 310. It changes replies. TrueScore with it and the draft model is 87.1 against 87.9 before, a paired
  difference of -0.7 with a 95% interval of [-4.0, +2.6].
- **Qwen's thinking guard** needs Clef-Flash beside the model (18 GB). `GUARD=0` serves without it.

## What the fork adds

| Addition | Effect, measured on one DGX Spark |
| --- | --- |
| Clef thinking guard (`--thinking-guard URL`) | A decision model closes the think block once the reasoning holds a complete derivation |
| The close is written into the live stream | No second prompt pass on a close: 0 s against 356 s over 58 closes; TrueScore 88.2 and 88.4 against 88.9 |
| Clef decision serving (`/v1/systemone`, decision sessions) | Clef and Clef-Flash served by `tensorfold serve`, checks about 0.3 s each |
| `continue_final_message` | A request continues the last assistant turn |
| GLM-5.3 Flash on one GPU (`--tp 1`, packed EXL3) | The M288 Mosaic 12L checkpoint serves on one Spark |
| GLM prompt experts in 1,024-row windows | Prefill of a 20,500-token prompt: 247 to 314 tok/s, bit-identical output |
| GLM copy drafts | Without a draft model: rewrite 38.5 to 52.1, json 39.0 to 44.8 tok/s, same replies |
| GLM DFlash2 through the packed head | A DFlash2 draft model runs beside the packed checkpoint: 34 / 51 / 56 / 60 tok/s |
| Lone-stream graph slot off by default | One stream under `--parallel 4`: 70.7 to 90.5 tok/s (`TENSORFOLD_LONE_GRAPHS=1` restores it) |
| Copy drafts at 6 drafts a round | A 2,252-token prompt's first 128 tokens: 36.5 to 89.0 tok/s; rewrite 108 to 124 |
| Copy windows widen to 32 rows (`TENSORFOLD_COPY_ROWS`, 20 Oct) | Copy text 119 to 244 tok/s, rewrite 147 to 237; 1.25x one stream and 1.23x four over 12 cells, same replies |
| Copies follow the occurrence that matches furthest back (`TENSORFOLD_COPY_BACK`, 20 Oct) | Qwen, four in flight: copy 110 to 172 tok/s, batch +4.5%; a list's lines no longer copy the previous line |
| GLM copies in DFlash2 rounds (`TF_GLM_DFLASH_COPIES`, `TF_GLM_COPY_BACK`, on in `run.sh`) | Batch 48.6 to 54.5 tok/s, list copy 66.8 to 146.4, rewrite 63.1 to 99.6, json unchanged, same replies |
| EXL3 prompt experts read once for 32 rows | Prefill 780 to 828 tok/s, bit-identical output |
| EXL3 image tower converted from the pack | `tensorfold.vision.exl3_convert` reads a tower stored inside the pack's shards |

Speeds are from 2026-10-11 and 2026-10-12, except the rows marked 20 Oct. Qwen3.8 Flash Next: EXL3 3.05 bpw, `--parallel 4 --kv-dtype int8`,
262,144-token window. GLM-5.3 Flash: Mosaic 12L, 65,536-token window, greedy. Drafted replies equal their
`"draft": false` runs on both models (10/10 each).

Four ideas were measured and left out, so nobody needs to repeat them. Skipping low-weight expert picks on GLM is 19
to 27% faster at a 10% threshold, and replies then fail to stop or open with reasoning text; a sweep of 19 settings
(thresholds, layer bands, prompt or decode rows, fixed pick counts) found no setting that keeps a gain and passes.
Chinese-specific experts are 67 of 12,096 on GLM, too few to prune or offload. A second candidate row beside a
DFlash2 draft would be kept in 16.8% of rounds and needs 34% to pay. The switches behind these measurements
(`TF_GLM_SKIP_*`, `TF_GLM_TREE_PROBE`, `TF_GLM_ROUTE_LOG`) stay in the tree, off by default. A TandemLLM-style draft
tree on Flash Next was not built: TandemLLM's dense 27B verifies 16 rows in the time of 8, while Flash Next's experts
make each row past 7 cost 1.8 to 1.9 ms on real tokens, so a sibling row must add 0.20 tokens a round to pay. Long
copies, whose rows nearly all land, are what that cost favours, which is where the copy windows came from.

## Clef as a thinking control

The idea is [@outcastofmusic](https://github.com/outcastofmusic)'s: a small decision model watches a larger model
think and stops it when the answer is already derived. After each paragraph of reasoning, the server asks
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) one question: "Does the reasoning so far contain a complete
derivation of the final answer?" A probability of 0.8 or more closes the think block at the next paragraph end. A
paragraph is checked only when 128 reasoning tokens have passed since the last check.

Measured on spark-bench TrueScore, 76 scenarios x 2 repeats, thinking on, 2026-10-10, both runs under an 8,000-token
thinking budget:

| Run | TrueScore | Request time | Tokens generated | Reasoning text |
| --- | --- | --- | --- | --- |
| Clef guard on | 88.9 | 4,667 s | 291,536 | 486,948 chars |
| Guard off | 88.4 | 6,176 s | 391,814 | 908,672 chars |

The guard saves 24% of request time (1,509 s of 6,176 s) and 46% of the reasoning text. The score difference, paired
over scenarios, is +0.9 with a 95% interval of [-0.9, +3.1], so quality holds. Pair the guard with
`--thinking-budget`: without a budget, one coding task thought for 26,243 tokens because Clef never rated it above
0.75.

```bash
tensorfold serve ~/models/clef-flash --port 30001 --name clef --context 16384          # the decision model
tensorfold serve ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw --port 30000 --parallel 4 --kv-dtype int8 \
    --thinking-guard http://127.0.0.1:30001/v1 --thinking-budget 8000
```

A request turns the guard off with `"thinking_guard": false` and sets its own budget with `"thinking_budget"`. GLM-5.3
Flash leaves 8 GiB free on a Spark, so Clef-Flash (18 GB) cannot run beside it. `./run.sh qwen-flash-next` starts both
servers with these flags. `./run.sh glm-5.3-flash` serves GLM with `--thinking-budget 8000` and no guard.

## Credits

- **[ashhart/TensorFold](https://github.com/ashhart/TensorFold)** and its contributors wrote the engine. The fork
  changes a small part of it.
- **[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)**
  is the recipe the Qwen3.8 Flash Next setup follows. Its patch 0007 showed that copy drafts pay at 6 drafts a round,
  which led to the copy-draft fix here. The fork ports code from four of its patches under the MIT licence: 0002's
  indexer scores kernel, 0005's tiled-select threshold, 0008's vision rotary and image features, and 0014's prompt
  expert tile. `THIRD_PARTY_NOTICES.md` links each one.
- **[TandemLLM](https://github.com/0xBakeer/TandemLLM)** (AGPL-3.0) measured tree verify, StairCut and its conductor
  checks on one Spark. The fork reimplemented some of its ideas behind flags and copied no code.
- **[sxuff/glm53-flash-single-gb10](https://github.com/sxuff/glm53-flash-single-gb10)** supplied the four patches
  that run GLM-5.3 Flash on a single GB10. The fork ports them from v0.6.5 onto `python-0.6`.
- **[Cloudflare/clef](https://huggingface.co/Cloudflare/clef)** and Clef-Flash are the decision models behind the
  thinking guard.
- **[turboderp](https://huggingface.co/turboderp)** made the EXL3 quantizations of both models, and
  **[0xSero/GLM-5.3-Flash-EXL3-Spark](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark)** is the Mosaic 12L
  mix of them that fits one Spark.
- **[beamster/GLM-5.3-Flash-Sushi-2.4bpw](https://huggingface.co/beamster/GLM-5.3-Flash-Sushi-2.4bpw)**, packed for
  [sushi](https://github.com/beamivalice/sushi), supplies the hybrid's routed experts.
- **[cfontes/GLM-5.3-Flash-DFlash2-TR3-v3](https://huggingface.co/cfontes/GLM-5.3-Flash-DFlash2-TR3-v3)** and
  **[modal-labs/GLM-5.3-Flash-DFlash](https://huggingface.co/modal-labs/GLM-5.3-Flash-DFlash)** are the GLM draft
  models.

## Upstream README

TensorFold serves language models on Apple Silicon and NVIDIA GPUs through an OpenAI-compatible API.
Each model family supplies its own kernels and draft verification.

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

On a Mac, Homebrew installs it too: `brew install ashhart/tensorfold/tensorfold`.

Use `http://127.0.0.1:8080/v1` as the client base URL and the model ID from `/v1/models`. Both backends serve chat
completions, completions, OpenAI Responses (`/v1/responses`) and Anthropic Messages (`/v1/messages`); see the [API reference](docs/api.md).
Python 3.11 or newer is required, and MLX 0.32.2 or newer on a Mac (pip installs it). See the [runbook](RUNBOOK.md)
for installation and a first request. On NVIDIA GPUs the CUDA kernels need compute capability 8.9 or newer: Ada (RTX 40
series), Hopper and Blackwell, including the DGX Spark's GB10 and the RTX 50 series. NVFP4 and FP8 checkpoints run from
8.9: their own math where the GPU has each mma (FP4 on 12.x, FP8 from 8.9), W4A16 elsewhere; the RTX 40, Hopper and
B200 builds are compiled and bit-checked on Blackwell but not yet run on those cards. RTX 30 cards (8.6) aren't
supported, and the server refuses a GPU below 8.9 at startup.

## Image input

Install the vision extra, `python -m pip install 'tensorfold[vision] @ git+https://github.com/ashhart/TensorFold.git'`,
and start a supported GLM-5.3-Flash or Qwen3.5/3.8 dense checkpoint with `--vision` to accept image and text content
parts through the same lane engine; Flash Next CUDA also accepts images with `--vision --parallel 2` or more.
GLM-5.3-Flash images run on MLX; dense Qwen's run on MLX and CUDA. See
[image input](docs/vision.md) for the API, checkpoint requirements, cache behavior and qualification status.

## Models

| Model | Checkpoint | Backend | Drafting |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning | `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | MLX, CUDA | Included MTP head; context copies on MLX |
| Qwen3.8-27B | `TensorFold/Qwen3.8-27B-MLX-4bit` | MLX, CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies; DFlash2 is optional on MLX |
| Qwen3.8 Flash Next | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` | MLX, CUDA | Included MTP head and context copies |
| GLM-5.3-Flash | `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` | MLX on a 256 GB Mac, CUDA with two ranks | MTP; optional DFlash2 on CUDA |
| Gemma 4 26B-A4B | `mlx-community/gemma-4-26b-a4b-it-4bit` | MLX | Context copies; `z-lab/gemma-4-26B-A4B-it-DFlash` is optional |
| DeepSeek-V4-Flash | `mlx-community/DeepSeek-V4-Flash-4bit` | MLX on a 256 GB Mac | `TensorFold/DeepSeek-V4-Flash-DSpark-MLX` or `TensorFold/DeepSeek-V4-Flash-MTP-MLX` |
| Qwen3.8-27B (NVFP4) | `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt: NVFP4 MLP, FP8 attention) | CUDA, one GPU | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8-27B (EXL3, experimental) | `turboderp/Qwen3.8-27B-exl3` (branches `3.00bpw`, `4.00bpw`; any codebook, 1 to 8 bits per weight) | CUDA | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (EXL3, experimental) | `turboderp/Qwen3.8-Flash-Next-exl3` (branch `3.05bpw_h5_ng5`; any codebook, a width per tensor) | CUDA | Included MTP head and context copies |
| Ternary Bonsai 2 27B | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | MLX | `z-lab/Qwen3.8-27B-DFlash2` and context copies |
| Qwen3.8 Flash Next (NVFP4) | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (ModelOpt: NVFP4 experts, MXFP8 attention and DeltaNet); `RadixArk/Qwen3.8-Flash-Next-NVFP4` (bf16 besides the experts) | CUDA, one GPU | Included MTP head and context copies |

`tensorfold models` lists families and checkpoints. `tensorfold info MODEL` checks configuration without
fetching weights. `serve` downloads a missing checkpoint; `pull` downloads it ahead of time.
Those `TensorFold/...` ids moved from the `Vontra` org on Hugging Face on 2 October 2026; the old names redirect.

```bash
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit
```

Qwen3.8-27B reads MLX affine 2-, 3-, 4-, 5-, 6- and 8-bit checkpoints, including mixed layer formats.
It reads packed rows in groups of 32, 64 and 128 on Apple Silicon and CUDA, with hardware qualification still
pending for the newer paths. M5 keeps its native tensor-unit kernels for compatible formats,
and other formats use the row decoder. See [quantized checkpoints](docs/quantization.md) for the exact scope.
On CUDA, pull DFlash2 before serving; without it, explicitly choose `--no-drafts` for the serial reference.

Nemotron uses TensorFold projections and routed-expert kernels. Its load-time row check controls drafting;
keep the installed MLX version within the package requirements. The named checkpoint includes
`mtp-4bit.safetensors`, which `pull` and `serve` check for.

Flash Next requires 4-bit/group-32 weights. Without an MTP head it can run without MTP drafting on MLX;
on CUDA, explicitly pass `--no-drafts`. On one CUDA GPU it also reads the two NVFP4 exports in the table as they
ship, and block-scaled FP8 (ModelOpt `FP8_PB_WO`) linears in such exports; see [the recipe](docs/recipes/qwen3.8-flash-next.md#nvfp4-checkpoints) for their formats, the checks they
passed and what is not supported. Nemotron CUDA requires 4-bit/group-64 weights and an MTP head
unless `--no-drafts` is set. GLM on MLX reads 4-bit/group-64 weights and mlx-lm's mixed-bit conversions,
whose 5-, 6- and 8-bit tensors take their own row kernels; it needs MLX 0.32.2 or later. GLM CUDA reads
MLX 4-bit/group-64 weights and Brandon M. Music's experimental EXL3/TR3 checkpoint
(`brandonmusic/GLM-5.3-Flash-tr3-4bpw`, also re-hosted as `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`). GLM's optional
`incoai/GLM-5.3-Flash-DFlash2` checkpoint has non-commercial license
terms, described in [third-party notices](THIRD_PARTY_NOTICES.md).

Gemma 4 has no draft head. It drafts copies of its context, and chains from z-lab's DFlash model when served with
`--drafter z-lab/gemma-4-26B-A4B-it-DFlash` (pulled once). Its kernels read 4-bit weights in groups of 32 or 64
with an 8-bit router, as the mlx-community conversion stores them; `serve` refuses other Gemma 4 layouts
before downloading.

DeepSeek-V4-Flash reads the mlx-community conversion (affine 4-bit/group-64 weights, mxfp4 routed experts) and
needs MLX 0.32.2 or later. Its draft heads are DeepSeek's DSpark blocks and MTP layer (MIT), converted:
`tensorfold pull TensorFold/DeepSeek-V4-Flash-DSpark-MLX` once and `serve` drafts with it; see
[its recipe](docs/recipes/deepseek-v4-flash.md).

See the [recipes](docs/recipes/README.md) for supported formats and backend limits.

## Exact decoding

A draft is accepted only when it equals the token the same engine would produce serially.
Sampling depends on the prompt or explicit seed, absolute position and token ID. Verify kernels keep each
row's arithmetic independent of the other rows in the call. Compare a request with the same request using
`"draft": false` to check drafted versus serial output.

The MLX engine can share a round across requests. Each stream keeps its own state and sampling key, with
concurrent output required to match its solo output. Load-time checks restrict window width and shared
forwards where a family cannot reproduce its serial arithmetic. On CUDA, `--parallel N` with N greater
than one enables shared rounds for Qwen3.8-27B and Flash Next on one or two ranks and for Qwen3.6-35B-A3B on
one rank. GLM and Nemotron CUDA serve one request at a time; CUDA `--parallel auto` also means one request at a
time.

Exactness is against the same engine, weights, runtime and settings. It does not imply identical output
between MLX and CUDA, different quantizations, or different tensor-parallel rank counts.

## Serve options

| Option | Meaning | Backend |
| --- | --- | --- |
| `--host`, `--port` | Listen address, default `127.0.0.1:8080` | Both |
| `--name` | Model ID advertised to clients | Both |
| `--vision` | Opt-in GLM-5.3-Flash, Qwen3.5/3.8 dense and Flash Next image input | MLX; dense Qwen also CUDA; Flash Next CUDA with `--parallel >=2` |
| `--vision-max-images N` | With `--vision`, images across the full request history (default 4); other image limits still apply | Both |
| `--vision-image-tokens N` | With `--vision`, the visual tokens a request's images share (default 4,096, up to 65,536); each image keeps at most 4,096 | CUDA Qwen |
| `--alias` | Additional model IDs | Both |
| `--context N` | Prompt plus reply capacity | Both |
| `--max-tokens N` | Default reply limit, 4096 | Both |
| `--temperature`, `--top-p`, `--top-k`, `--min-p` | Sampling defaults; temperature zero is greedy | Both |
| `--thinking`, `--no-thinking` | Template thinking toggle | Both |
| `--reasoning-effort` | Template effort when a request sets none; default: the template's own | Both |
| `--thinking-budget N` | Token-count limit inside reasoning | Both |
| `--backend auto`, `mlx`, `cuda` | Select backend; auto uses MLX on macOS | Both |
| `--parallel N` | MLX `auto` admits up to 8 within budget; CUDA `auto` is 1, explicit N enables supported shared rounds | Both |
| `--no-drafts` | Decode serially | Both |
| `--drafter auto`, `none`, or model ID | Select an optional draft model where the family supports it | Both |
| `--mtp-drafts N` | Family-specific cap on MTP drafts | Both |
| `--kv-dtype bf16`, `int8`, `int4` | Flash Next: `int8` or `int4` stores keys and values with one fp16 scale per 32 values. Other families and the MLX path refuse it | CUDA |
| `--mtp-confidence P` | Flash Next: stop a draft chain before a later draft under this probability, 0 to 1 (default 0.70) | CUDA |
| `--prefill-fp8` | Prompt matmuls take FP8 (e4m3) activations, one scale a row, where the checkpoint has an FP8 prompt kernel (Qwen3.8 27B and Qwen3.6 MLX 4-bit, FP8 and MXFP8 layers of NVFP4 checkpoints): faster prompts at lower precision ([measured](docs/recipes/cuda.md#prompt-precision)). Default: bf16 activations, as decode | CUDA |
| `--precision checkpoint`, `full` | NVFP4 checkpoints: `checkpoint` (default) runs their own math, FP4 x FP4 on SM 12.x and FP8 x FP8 from 8.9, W4A16 elsewhere; `full` runs bf16 activations against the stored weights ([measured](docs/recipes/cuda.md#nvfp4-precision)) | CUDA |
| `--tp 2 --rank R --master HOST` | Two-rank CUDA execution; `--master-port P` sets rank 0's rendezvous port (default 29551) | CUDA |
| `--decode-share F` | Mac: while prompts prefill, running replies keep moving for this share of each chunk's time; a new prompt starts at the next chunk, the fewest tokens left first (default 0.25; 0 prefills whole prompts first, in order, as 0.3.6.2). Flash Next on CUDA with `--parallel N`: replies decode inside each prompt pass, and the share sizes the passes so a round's decoding takes it (default 0: whole passes) | Both |
| `--prompt-cache-gib N` | Retained conversation-prefix budget; zero disables retention. Default: the memory the weights, a whole-window request and a shared round leave idle, at least an eighth of RAM up to 16 GiB, given back on demand | MLX |
| `--prefill-pass N` | Plan chunks one forward takes while a prompt fills alone, for families with a prompt pass (default 8; 1 as 0.5.0) | MLX |
| `--pass-cache-gib N` | Freed-buffer cache during such a pass where the memory budget has room, default 16 GiB | MLX |
| `--checkpoint-slots N` | Retained conversation prefixes (default 3 per lane, at least 8); long conversations hit this before the byte budget. On CUDA, the prompt states Qwen3.8-27B keeps under `--parallel` 2 or more (default 3) | Both |
| `--spill-gib N` | Write evicted conversation prefixes to disk (up to N GiB) and read them back instead of prefilling again; zero disables | MLX |
| `--mlx-cache-gib N` | Reusable freed-buffer cache, default 8 GiB | MLX |
| `--snapshot-dir DIR` | Persistent prefix snapshots; `none` disables them | MLX |
| `--max-snapshots N` | System-block snapshots loaded at start, default 3 | MLX |
| `--no-update-check` | Disable the startup release check | Both |

The default sampling settings come from `generation_config.json`. Requests can override sampling and reply
length. CUDA does not implement the MLX-only options above. See [API fields](docs/api.md) for request scope.

In a terminal, `tensorfold serve` keeps one live throughput line under its log; it is off when output is
redirected, and `TENSORFOLD_NO_LIVE=1` turns it off.

<a id="memory"></a>

## Context and memory

On MLX, omitted `--context` targets the model's metadata window and reduces it to the startup memory
estimate when needed, allowing room to retain a prompt for the next turn. An explicit positive value
that cannot fit one request is refused at startup. A larger explicit window may fit a request without
leaving room to retain its prompt, so a later turn can need a full prefill. `--context 0` removes the
metadata cap; finite engine capacity and memory admission still apply. Use the reported context when
configuring client compaction.

On CUDA, Qwen defaults to the affordable native capacity. GLM targets a dense 2,051-token window,
and Nemotron targets 16,384 tokens; the capacity estimate can lower these defaults. Flash Next's `--kv-dtype int8` or `int4` counts
its smaller cache, so the same memory admits a longer window. Explicit `--context 0` targets the affordable native capacity for every CUDA family.
A positive CUDA value must fit both the native window and the capacity estimate on every rank;
otherwise startup refuses it with fitting guidance. Increasing GLM beyond its dense window enables
its sparse-attention path. The startup report distinguishes native and allocated capacity.
The CUDA budget grants a GPU its free memory less a floor of a tenth of that memory, at least 4 GiB: a discrete
card's own memory, or the host's available memory on a unified GPU. `TENSORFOLD_MEMORY_RESERVE_GIB` moves the
floor (at least 2 GiB), and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` caps the grant from above in GiB, an absolute budget
like `TENSORFOLD_MEMORY_LIMIT_GB` on the MLX side. A smaller floor can end requests with CUDA errors mid-reply,
and a floor you choose takes that risk knowingly. A discrete card's host need is its loading buffers, which
startup weighs on its own.

MLX defaults to a process budget of 70% of RAM. A family can state a larger share: GLM-5.3-Flash takes 85%
on a Mac with 256 GB or less, with nothing else loaded. `TENSORFOLD_MEMORY_LIMIT_GB` replaces that default in
GiB, raising or lowering it; physical RAM and the GPU's recommended working set still cap the result.
The server reserves 3 GiB for the rest of the process before setting the MLX allocator limit, so a
110 GiB process budget allows 107 GiB of MLX buffers. Concurrent admission honors the raised budget
while accounting for memory held elsewhere on the machine. Admission accounts for weights, cache
growth, reply tokens and prefill workspace. Retained prefixes and reusable MLX buffers have separate
limits. Admission can evict retained prefixes or queue another stream; fitting weights alone does not
establish a usable context size. A larger budget leaves less RAM for other applications and cached file pages.
The startup line says how far the variable can raise the budget on this Mac, and a startup refusal says how far it
must go. On a 32 GB Mac, Qwen3.8-27B needs more than the default 22.4 GiB, with or without its draft model.

Flash Next's startup weight check excludes n-gram tensors when the loader keeps them in host file
mappings. The startup report shows resident and file-backed bytes separately. Cached file pages still
consume RAM and can be reclaimed by the OS; see [Flash Next memory](docs/recipes/qwen3.8-flash-next.md#mlx-execution).

An explicit reply limit is reserved before prefill. A request that exceeds context or memory is refused
with fitting guidance; an omitted reply limit is capped by the remaining context. MLX reports a
context refusal as HTTP 400 for a non-streamed request or as an error event after opening a stream.
CUDA checks context before opening a stream.

On MLX, streams that share rounds take memory as they grow. A stream beside others holds its next 2,048
tokens of growth, not its whole reply, so `--parallel` streams are admitted while their real contexts fit.
Before each round the server checks that the live streams' next growth fits. If it doesn't, it first frees
MLX's cached buffers and retained prefixes, but only when that makes room. Then the newest streams wait a
round, keeping their state, so their tokens don't change. If even the oldest stream can't grow, the newest one
ends with an error that says so; the tokens it already sent stay valid.

The memory-class table below keeps the model combinations under qualification. Its GiB budget ceilings
emulate the listed RAM classes before the 3 GiB process reserve. The actual default budget uses
OS-reported physical memory and the GPU working set; an explicit memory budget can lower or raise it.
Context and peak-memory results remain TBD until a public
prompt fixture, checkpoint revision, runtime, command and measurement output accompany each result.

| Nominal RAM class | Budget ceiling | Qwen3.8-27B + DFlash2 | Qwen3.8-27B, `--drafter none` | Nemotron 3.5 Lightning | Qwen3.8 Flash Next |
| --- | --- | --- | --- | --- | --- |
| 32 GB | 22.4 GiB | TBD | TBD | TBD | TBD |
| 36 GB | 25.2 GiB | TBD | TBD | TBD | TBD |
| 48 GB | 33.6 GiB | TBD | TBD | TBD | TBD |
| 64 GB | 44.8 GiB | 140,288 tokens, 43.7 GiB (0.3.5.1) | 152,576 tokens, 39.9 GiB (0.3.5.1) | 262,144 tokens, 42.2 GiB (0.3.5.1) | Not run: 4-bit weights exceed the budget |
| 96 GB | 67.2 GiB | TBD | TBD | TBD | TBD |
| 128 GB | 89.6 GiB | TBD | TBD | TBD | TBD |
| 192 GB | 134.4 GiB | TBD | TBD | TBD | TBD |
| 256 GB | 179.2 GiB | TBD | TBD | TBD | TBD |

Each model cell needs the fitted context and peak physical process footprint. An emulated budget on
a larger host is not a measurement on hardware with that RAM size. These are qualification slots,
not minimum-memory promises. Weights that exceed the MLX budget are refused before loading.

The 64 GB row comes from @benwilson's run on a 64 GB M5 Pro with TensorFold 0.3.5.1 and MLX 0.31.2 (#70),
not from this release. Each cell is the fitted context at the default budget and the process's lifetime peak
footprint over cold and resumed prompts at that context. The [Qwen3.8-27B recipe](docs/recipes/qwen3.8-27b.md#a-64-gb-m5-pro-on-0351)
lists the checkpoint revisions, fixture, commands and results. On 0.3.5.1 with DFlash2, prompts above about
100,000 tokens were served but not kept for the next turn (#71).

## Prompt caching

On MLX, chunk starts come from the rendered token sequence. Resume points are assistant-message
starts and the second message start, using markers discovered from the chat template. The planner
skips points less than 256 tokens from the previous chunk start and otherwise cuts at the first
eligible point or after the model family's chunk. Qwen3.8 Flash Next measures one chunk at startup and
takes the largest of 8,192, 4,096 and 2,048 tokens (4,096 and 2,048 on GPUs without tensor units) that
still leaves room for 128K tokens of context; other families use 2,048. Without recognized markers it
uses that grid. There is no configurable `--prefill-grid` option.

Fresh and resumed requests use the same chunk plan. Reuse stops at a matching token prefix and a valid
chunk boundary; the previous reply is prefilled again under the current prompt. A template that rewrites
an earlier turn can reduce reuse. A follow-up therefore need not reprocess a full grid cell, but short
messages or changed earlier text can make it reprocess more than the latest reply and new messages.

Snapshots include the model, runtime, kernel and chunk-plan identity. System prefixes and retained
conversations can survive restarts. CUDA engines keep their own prompt/reply states and do not use the
MLX disk-snapshot or retained-prefix options.

<a id="dgx-spark-and-other-nvidia-gpus"></a>

## NVIDIA GPUs

Use NVIDIA's PyTorch container for CUDA, PyTorch, Triton and the extension compiler; the package has no
`cuda` installation extra. Install TensorFold inside the container without replacing that toolchain.

```bash
docker run -it --gpus all --ipc=host --network host nvcr.io/nvidia/pytorch:26.07-py3
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --host 0.0.0.0
```

Qwen3.8-27B, Flash Next and Nemotron support one or two CUDA ranks; GLM requires two.
For two ranks, see the [CUDA runbook](RUNBOOK.md#nvidia-gpus). Each rank needs its checkpoint and any
optional drafter. Rank 0 serves HTTP. Unified GPU/host memory also holds runtime buffers and file-backed
model data; the startup estimate is not a measured maximum capacity.

## Measurements

Each release's notes give its measured decode, prompt and concurrency numbers against the previous release and the
standard servers, on the machines they name: see [CHANGELOG.md](CHANGELOG.md) and the GitHub releases. The
[recipe book](docs/recipes/README.md#measurements) gives the public prompts and benchmark command, and each family's
recipe keeps its own tables.

## Updating

`tensorfold update --check` checks for a release; `tensorfold update` installs it (`--force` reinstalls the newest
release even when it is current), then the server must restart. A normal installation uses the same interpreter's pip. An editable clone must be clean and able
to fast-forward to the release tag; afterwards run `python -m pip install -e .` in the checkout to refresh
metadata and dependencies. A Homebrew install upgrades with `brew upgrade tensorfold` instead. `--no-update-check`
or `TENSORFOLD_NO_UPDATE_CHECK=1` disables startup checks.

When the update finishes it prints what changed since your version, from [CHANGELOG.md](CHANGELOG.md), which lists
every release. The first time a new version serves, it prints one line linking to its notes.

## Development and license

To send a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md) first.
Family interfaces, kernel layout and verification requirements are in the [recipe book](docs/recipes/README.md),
[family map](src/tensorfold/families/README.md) and [kernel map](src/tensorfold/kernels/README.md).
Apache-2.0 from 0.6.0; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Releases up to 0.5.0 were MIT, and code written before 0.6.0 keeps its [MIT notice](LICENSES/MIT.txt).
Model weights keep their own licenses.

## Contributors

Thank you to everyone who has sent TensorFold a pull request, a measurement or a bug report:

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@akol1](https://github.com/akol1),
[@Alexbob0](https://github.com/Alexbob0),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@barelyworkingcode](https://github.com/barelyworkingcode),
[@Benjamin-Wegener](https://github.com/Benjamin-Wegener),
[@benthecarman](https://github.com/benthecarman),
[@benwilson](https://github.com/benwilson),
[@BHCC2025](https://github.com/BHCC2025),
[@Bizuayeu](https://github.com/Bizuayeu),
[@BlivionIaG](https://github.com/BlivionIaG),
[@BobClawblaw](https://github.com/BobClawblaw),
[@borodach23](https://github.com/borodach23),
[@Boscoeuk](https://github.com/Boscoeuk),
[@boxabirds](https://github.com/boxabirds),
[@brandonmmusic-max](https://github.com/brandonmmusic-max),
[@bunnyfu](https://github.com/bunnyfu),
[@CerebralCoding](https://github.com/CerebralCoding),
[@cesarswong](https://github.com/cesarswong),
[@chadhurley25075-png](https://github.com/chadhurley25075-png),
[@chaog992](https://github.com/chaog992),
[@Charlie-Louis](https://github.com/Charlie-Louis),
[@Chedrian07](https://github.com/Chedrian07),
[@chris247474](https://github.com/chris247474),
[@christrade215](https://github.com/christrade215),
[@crescit](https://github.com/crescit),
[@cshintov](https://github.com/cshintov),
[@cwschroeder](https://github.com/cwschroeder),
[@DakotaTexas](https://github.com/DakotaTexas),
[@danyo1399](https://github.com/danyo1399),
[@Deesha08](https://github.com/Deesha08),
[@Defilan](https://github.com/Defilan),
[@DevRico003](https://github.com/DevRico003),
[@di37](https://github.com/di37),
[@drowzeys](https://github.com/drowzeys),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@feni6](https://github.com/feni6),
[@gbgbgbg](https://github.com/gbgbgbg),
[@GDACONSULT](https://github.com/GDACONSULT),
[@gecobattya](https://github.com/gecobattya),
[@gilby](https://github.com/gilby),
[@Gogo6969](https://github.com/Gogo6969),
[@gprot42](https://github.com/gprot42),
[@GraithSecurity](https://github.com/GraithSecurity),
[@grantoverton](https://github.com/grantoverton),
[@grearjake-star](https://github.com/grearjake-star),
[@greatyingzi](https://github.com/greatyingzi),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@jetnet](https://github.com/jetnet),
[@jkuepker](https://github.com/jkuepker),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jregan-beasley-smc](https://github.com/jregan-beasley-smc),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@lcgutierrez](https://github.com/lcgutierrez),
[@LECYWZA](https://github.com/LECYWZA),
[@lijian1999](https://github.com/lijian1999),
[@liumorrisclaw](https://github.com/liumorrisclaw),
[@LXD-8](https://github.com/LXD-8),
[@m-naoki-m](https://github.com/m-naoki-m),
[@mapamalu](https://github.com/mapamalu),
[@mcclanahanaman](https://github.com/mcclanahanaman),
[@MESevenJourney](https://github.com/MESevenJourney),
[@mgoldwasser](https://github.com/mgoldwasser),
[@MiaAI-Lab](https://github.com/MiaAI-Lab),
[@mikolaj92](https://github.com/mikolaj92),
[@millaguie](https://github.com/millaguie),
[@Mirrdhyn](https://github.com/Mirrdhyn),
[@Moutonc](https://github.com/Moutonc),
[@MovieMaker93](https://github.com/MovieMaker93),
[@mrpmorris](https://github.com/mrpmorris),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@RoscoeTT](https://github.com/RoscoeTT),
[@salmanarshad321](https://github.com/salmanarshad321),
[@samwang0041-star](https://github.com/samwang0041-star),
[@sanjaibalajee](https://github.com/sanjaibalajee),
[@satindergrewal](https://github.com/satindergrewal),
[@scottleimroth](https://github.com/scottleimroth),
[@sethforprivacy](https://github.com/sethforprivacy),
[@sfxnz](https://github.com/sfxnz),
[@shantanugoel](https://github.com/shantanugoel),
[@simon-lin88](https://github.com/simon-lin88),
[@simonmd](https://github.com/simonmd),
[@spenchey](https://github.com/spenchey),
[@squarrier](https://github.com/squarrier),
[@ss-cong](https://github.com/ss-cong),
[@styles01](https://github.com/styles01),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose).
