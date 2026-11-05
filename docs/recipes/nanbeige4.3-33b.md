# Nanbeige4.3-33B

The CUDA family `src/tensorfold/families/llama_dense/` serves `Nanbeige/Nanbeige4.3-33B` (model_type `llama`,
`LlamaForCausalLM`) on one GPU. As of 2026-10-27 the engine has passed its tests on a tiny random Llama only. No run
on the real weights has happened yet, so this page gives sizes computed from config.json and no measured speeds.

## The checkpoint

| | |
| --- | --- |
| Layers | 52, every one full attention and a SwiGLU MLP |
| Widths | hidden 7,168; 56 query heads and 8 KV heads of 128; intermediate 22,016; vocabulary 166,912 |
| Rotary | rope_theta 1.2e8 over the whole head, no rope_scaling |
| Head | separate `lm_head` (untied); RMSNorm eps 1e-5; no biases |
| Context | 524,288 tokens |
| Weights | 33.1B parameters, 66.2 GB in bf16 over 14 shards |
| End token | `<|im_end|>` (166101), from config.json and generation_config.json |

The card recommends temperature 0.6 and top_p 0.95 for chat, and temperature 1.0 for agentic and tool use.
generation_config.json sets no sampling, so pass them to `tensorfold serve`.

## Formats

| Format | Projections | Embedding, norms | Head |
| --- | --- | --- | --- |
| bf16 as released | `b16.cu`, one warp an output | as stored | bf16 |
| EXL3 (exllamav3's convert) | the shared EXL3 linear, any codebook and width | as stored, embedding cast to bf16 | EXL3 or bf16 |
| NVFP4 / FP8 (ModelOpt or compressed-tensors) | the NVFP4 and FP8 lane matmuls, `--precision` full or checkpoint | as stored | as exported, bf16 by default |

Sizes computed from config.json: the 30.7B layer parameters take 15.4 GB at 4.0 bits, plus 2.4 GB for the bf16
embedding and 0.9 GB for a 6-bit head, about 18.7 GB for an EXL3 pack. NVFP4 stores 4.5 bits a weight with its
e4m3 scales, about 17.3 GB for the layers and 22 GB with a bf16 embedding and head. The KV cache takes 208 KiB a
token (52 layers, 8 KV heads of 128, bf16 keys and values), 21.3 GB for each 100,000 tokens. Startup admission
picks the largest window that fits beside the weights and says so in its log line.

Decode reads every weight once a round, so the GB10's 273 GB/s bounds a one-row round at 4.1 tok/s on bf16 and
near 15 tok/s on a 4-bit pack. A copy window of up to 128 rows verifies several tokens in one weight read.

## Engine

One stream: requests take turns. Decode drafts copies from the context, as qwen3_5 does without its DFlash2 drafter,
and verifies each copy as a chain. With `--drafter ~/models/Nanbeige4.3-33B-DFlash2` (ProCreations' DFlash2 draft
model), a round without a copy verifies a 16-row tree instead: 1.55x copies alone over 12 cells, code 2.1x, prose
1.2x, with every reply the same (2026-11-04). Every drafted token is the token serial decoding gives, since a row gets the same
bits alone and in every verify window. `"draft": false` decodes one row a round and keeps no prompt states. The
prefix cache keeps up to 4 states: the second message's start, the last assistant turn's start, and each prompt's
end less one token. A kept state shares its conversation's KV buffers, so it costs no memory past that
conversation. When a growing cache needs room, other conversations' kept buffers go first, least recently used.
[The package README](../../src/tensorfold/families/llama_dense/cuda/README.md) lists the kernels.

Refused at startup from config.json or the serve options: rope_scaling, attention or MLP biases, a non-SiLU MLP, a
tied head, head_dim other than 128 or 256, more than 16 query heads a KV head, `--tp 2`, `--parallel`, `--vision`,
`--prefill-fp8` and MLX-quantized weights. Structured output (`response_format`) is refused per request.

## Quantize

Both tools load the bf16 model, and on the Spark's unified memory that means stopping the production server first
(`./run.sh stop` in `tensorfold-python/dev`).

EXL3 with exllamav3 (not in the serving image; install it in its own venv). Check `python convert.py -h` of the
clone for its current flags:

```bash
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3 && pip install -r requirements.txt .
python convert.py -i ~/models/Nanbeige4.3-33B -o ~/models/Nanbeige4.3-33B-exl3-4.0bpw \
  -w ~/models/exl3-work -b 4.0 -hb 6
cp ~/models/Nanbeige4.3-33B/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,added_tokens.json,generation_config.json} \
  ~/models/Nanbeige4.3-33B-exl3-4.0bpw/
```

NVFP4 with ModelOpt (0.44.0 is in the serving image `tensorfold-run:c0b5f8af305d`):

```python
import os

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

src, dst = (os.path.expanduser(f"~/models/{n}") for n in ("Nanbeige4.3-33B", "Nanbeige4.3-33B-NVFP4"))
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="cuda")
texts = [...]                                   # 256-512 chat and code samples rendered with the chat template

def calibrate(m):
    for text in texts:
        m(**tok(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda"))

mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=calibrate)
export_hf_checkpoint(model, export_dir=dst)    # then copy the tokenizer files as above
```

The export's config.json must name `quant_method: modelopt`; `tensorfold info <dir>` shows the format it reads.

## First run on the real weights

```bash
tensorfold info ~/models/Nanbeige4.3-33B          # family "Llama dense", runs on CUDA, check passes
tensorfold serve ~/models/Nanbeige4.3-33B --name nanbeige --context 32768 --temperature 0.6 --top-p 0.95
```

Check, in order: the startup estimate fits; a chat reply reads well with thinking on and off; the same request with
`"draft": false` returns the same text at temperature 0; a second turn reports cached prompt tokens. Then serve the
quantized pack the same way and repeat.
