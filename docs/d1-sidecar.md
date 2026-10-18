# d1-3B decision sidecar

With `D1=1`, `run.sh` starts Liquid AI's d1-3B on `127.0.0.1:30003` beside every model target except GLM, once the model answers. d1-3B reads
a state (text, JSON, pictures) and answers typed questions about it with probabilities in one forward pass, with zero
output tokens. Nothing calls it yet: the gateway and the model servers are unchanged, and the three uses below are
requests a later change could send. `D1=1` turns the sidecar on (off by default until its GPU parity and latency are measured), and `./prepare.sh d1` downloads the checkpoint.

## The model

| | |
| --- | --- |
| Checkpoint | [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) at `051bcc464b01b9f92942b364d9586b0ef5912432` (2026-10-07) |
| Size | 3.12B parameters, 6.25 GB of bf16 safetensors: LFM2 language model (30 layers, 22 gated short convolutions and 8 GQA attention layers, 32 query and 8 KV heads) and a SigLIP2 NaFlex vision tower (400M) |
| Base | LFM2.5-VL-3B, post-trained for single-pass calibrated decisions |
| Context | 32,768 tokens native, 4,096 admitted by the sidecar (`D1_CONTEXT`) |
| License | LFM Open License v1.0: free use, and commercial use only while the user's annual revenue stays below US$10M |

The family is `src/tensorfold/families/lfm2/`, plain torch with no custom kernels. Its prompt and readout follow the
checkpoint's own `prompt.py` and `hybrid.py` and the Python 0.6 recipe in
[plotarmordev/d1-tensorfold](https://github.com/plotarmordev/d1-tensorfold) (Apache 2.0). The state is read once as a
trunk, and each question continues it as a branch packed on the same axis, so three questions on one state cost one
forward.

## API

`POST /v1/systemone` takes llama-server's System One request. `questions` maps an id to a question of type `noul`
(yes or no), `choice` (named options) or `score` (2 to 10 ordered levels, lowest first). `state` is a string, any JSON
value, a chat (`{"messages": [...]}` with `image_url` parts), or `null` when pictures are the whole state. `images` or
`files` carry up to 4 pictures as data URLs.

```bash
curl -s http://127.0.0.1:30003/v1/systemone -d '{
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}'
```

```json
{"model": "d1-3b", "answers": {
   "refund": {"type": "noul", "noul": 0.99},
   "team": {"type": "choice", "choice": "billing", "confidence": 0.985,
            "probabilities": {"billing": 0.985, "technical": 0.01, "fraud": 0.006}},
   "urgency": {"type": "score", "score": 0.72, "confidence": 0.537,
               "probabilities": {"0": 0.369, "1": 0.537, "2": 0.093},
               "legend": {"0": "Can wait", "1": "Today", "2": "Blocking the customer now"}}},
 "usage": {"input_tokens": 116, "output_tokens": 0}}
```

The numbers are the CPU check's below, rounded. An answer is a softmax over the question's options alone: a `noul`
gives P(yes), a `choice` its pick and each option's probability, a `score` the expected level. A malformed request
gets HTTP 400 with the reason.

`POST /v1/decisions` takes the SGLang-style request the GLM and Flash Next servers already serve (`input`, a list of
`questions` with `id` and type `choice`, `score` or `yes_no`). It renders each question through the checkpoint's chat
template and also reports `label_mass`, the share of the vocabulary's probability the labels hold.

Chat, completions, Responses and Messages requests answer 400 with "d1 is a decision model and writes no text".

## Speed, memory and agreement with Transformers

| Measure | Value | Where it comes from |
| --- | --- | --- |
| Top answers equal to Transformers' bf16 Lfm2-VL | 9/9 questions over 6 requests, largest probability difference 0.028 | This family on the CPU with the real weights, 2026-10-17 |
| A 384 x 288 red picture as the whole state | `red` at 0.99 against green and blue, through the checkpoint's processor and SigLIP2 tower | Same run |
| `/v1/decisions`, the same ticket's 3 questions | Same top answers, the labels holding 0.97 to 0.99 of the vocabulary's probability | Same run |
| Prompt tokens for the request above | 116 | Same run. The recipe's server reports 116 for the same request |
| One question, upstream engines on a GB10 | 24.7 ms (Zig, native kernels), 25.3 ms (recipe's Python engine with Triton kernels and CUDA graphs) | Upstream CHANGELOG #844 |
| One question, this family on a GB10 | Not measured yet. Expected 25 to 40 ms: reading the 4.87 GB of language weights once at 273 GB/s takes 18 ms, and eager kernel launches add to it | `tests/cuda/test_lfm2_d1_checkpoint.py` prints the median |
| Memory | Startup estimate 7.3 GiB at `D1_CONTEXT=4096`: 5.8 GiB of weights and 1.5 GiB of loading buffers, then 0.4 GiB of activations at a full window, plus the CUDA context | The admission's estimate. The GPU test prints the measured figure |

Requests run one at a time. A request reads its state once however many questions it asks, and questions that do
not fit one forward's token budget (`TENSORFOLD_LFM2_TOKEN_BUDGET`, at most the context) continue in a second forward
over the same state.

`TF_D1_HEAD` picks how a request reads its option logits after the forward, while the choice is measured
(docs/reviews/2026-10-18.md). `union`, the default, multiplies every question's row by the head rows of all their
options at once. `per_question` multiplies each question's row by its own options' rows and reads all results back
once, which gives `per_question_sync`'s bits. `per_question_sync`, the code before b585b8fb, reads each question back
on its own.

## Memory beside the big models

d1 starts after the model answers, so a model's startup admission never counts it. It admits itself against what the
model left, with a 2 GiB reserve (`TENSORFOLD_MEMORY_RESERVE_GIB=2`). Measured on 2026-10-17: GLM leaves 17 GiB, and
d1's load took available memory under 8 GiB, so `run.sh` does not start d1 beside GLM. Beside Qwen FP4-Local, d1
loaded with 30 GiB left. Qwen's four streams grow their KV caches against the memory available at the time, so d1's
7.3 GiB comes out of that room. On the GB10, one question took 25.9 ms and three 31.6 ms (medians), and
`/v1/systemone` over one kept connection 26.9 ms. A missing checkpoint or a failed d1 load prints a warning and leaves
the model serving.

## Three uses on this box

None of these is wired in. Each shows the request a gateway or a server would send.

**Choosing a thinking budget per request.** Before a turn starts, ask how much reasoning the request needs, and map
the expected level to `thinking_budget` (for example 0, 1,000, 4,000 and 8,000 tokens). On the CPU check, a Lean
proof request scored levels 1 to 3 at 0.34, 0.30 and 0.30, so a mapping should read the expected level or the top two
levels, and the budget stays a cap.

```json
{"state": {"request": "<the user's last message>", "turns_so_far": 3},
 "questions": {"budget": {"type": "score",
   "instructions": "How much thinking does this request need before answering?",
   "criteria": ["none", "a few hundred tokens", "a few thousand tokens", "as much as allowed"]}}}
```

**Routing a request to one of the swap models.** One `choice` over the served names, each described by what it is
good at. On the CPU check, "What is the capital of France?" went to `nemotron-bolt` at 0.86 and the Lean proof to
`glm-5.3-flash` at 0.48 against `qwen-flash-next` at 0.42. A router should act on a pick only above a confidence it
has measured, and a swap costs a model load (25 s to 4 minutes) plus up to SWAP_PERSIST (120 s) of state writing, so it should also weigh the model already loaded.

```json
{"state": {"request": "<the user's last message>", "tools": ["bash", "edit"], "loaded": "qwen3.8-flash-next"},
 "questions": {"model": {"type": "choice", "instructions": "Which model should answer this request?",
   "criteria": {"qwen-flash-next": "fast general model, good at code and tools",
                "qwen-flash-next-fp4": "the same model, more accurate, slower",
                "glm-5.3-flash": "strongest reasoning, long careful answers",
                "nemotron-bolt": "small fast model for simple questions"}}}}
```

**Flagging an empty or looping reply.** After a reply ends, ask whether it is empty, cut off or repeating, and retry
or warn when P(yes) passes a threshold. On the CPU check, an empty reply scored 0.97 and a reply repeating "apple,
banana" eight times 0.76. The gateway's empty-reply retry (`feat/empty-retry`) catches the empty case without a
model, and d1 adds the cut-off and looping cases. A long reply should send its last 2,000 tokens, which keeps the
request inside the 4,096-token window.

```json
{"state": {"request": "<the user's last message>", "reply": "<the reply's last 2,000 tokens>",
           "finish_reason": "stop"},
 "questions": {"bad": {"type": "noul", "instructions": "Is the reply empty, cut off, or repeating itself?"},
               "loop": {"type": "noul", "instructions": "Does the reply repeat the same passage again and again?"}}}
```

d1-3B is over-confident: on the Fast Decisions dev split the recipe measured 61.7% accuracy with an expected
calibration error of 12.3%, and accuracy fell 5.1 to 18.9 points below the stated confidence in every band. Each use
needs its own labelled set before a threshold is trusted.

## Tests

The CPU suite covers the prompt pieces against `prompt.py`'s text, option codes, readout tokens, refusals, the tree
forward against Transformers' Lfm2 on each question alone, picture features against Transformers' projector, and the
HTTP routes (`tests/test_lfm2_d1.py`, a tiny random checkpoint from `tests/lfm2_tiny.py`).

The GPU tests compare the real checkpoint with Transformers' bf16 Lfm2-VL on 6 requests and 9 questions, check a
picture, and print latency and memory. Run them with at least 16 GiB available, since the Transformers
reference loads a second copy of the weights:

```bash
docker run --rm --gpus all --ipc=host -v "$PWD":/tf:ro -v ~/models/d1-3B:/home/agis/models/d1-3B:ro -w /tf \
  -e PYTHONPATH=/tf/src -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 tensorfold-run:86cb86eec28c \
  bash -c "pip install -q 'pytest>=8' pytest-timeout >/dev/null 2>&1; \
           python -m pytest -p no:cacheprovider --basetemp=/tmp/pt --timeout=900 -q -s \
               tests/cuda/test_lfm2_d1_checkpoint.py"
```

The served sidecar, beside any model target:

```bash
D1=1 ./run.sh nemotron-bolt && ./run.sh status   # d1-serve: ready on 127.0.0.1:30003
curl -s http://127.0.0.1:30003/v1/systemone -d '{"state": "The app crashes when I open settings.",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
  "criteria": {"billing": "Charges", "technical": "App faults"}}}}'
```
