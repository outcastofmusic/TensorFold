# Dense Llama on CUDA

The CUDA engine for plain `LlamaForCausalLM` checkpoints (model type `llama`), written for Nanbeige4.3-33B. It
serves one stream on one GPU and drafts copies from the context, verified as chains. With `--drafter` pointing at a
DFlash2 folder (ProCreations/Nanbeige4.3-33B-DFlash2), a round without a copy verifies a 16-row DFlash2 tree. See
[the recipe](../../../../../docs/recipes/nanbeige4.3-33b.md) for setup and the first real-weights run.

A verify window of up to 128 rows gives each row the bits that row gets alone, so a drafted token is the token
serial decoding produces on this machine. Every kernel on that path is shared with qwen3_5 or the shared CUDA
kernels, except the rotary embedding in `glue.py`:

| Step | Kernel | Why a row's bits do not depend on the window |
| --- | --- | --- |
| bf16 projections | qwen3_5's `b16.cu` (`Plain`) | a warp an output column, each row's sum in fixed k order |
| EXL3 projections | `tensorfold/cuda/exl3` (`Exl3`) | the split-K plan is the weight shape's alone |
| NVFP4 / FP8 projections | `tensorfold/cuda/nvfp4` (`Fp4Linear`, `Fp8Linear`) | the lane matmuls' fixed reduction order |
| Norms, SwiGLU | qwen3_5's `glue.add_rmsnorm`, `glue.swiglu` | one program a row |
| Rotary embedding | `glue.rope` | one program a (row, head); transformers' rotate-half over the whole head |
| Attention | `tensorfold/cuda/kernels/attention.py` | a query's chunks and merge order depend on its own key range |
| Sampling | `tensorfold/cuda/sampling.py` | keyed by seed, position and the row's logits |

## The rest of the package

- `weights.py` reads bf16 checkpoints as stored and checks each projection's shape against config.json.
  `exl3_load.py` and `nvfp4_load.py` read the other two formats through qwen3_5's readers.
- `forward.py` has `tree_forward` and `new_state`. The state is qwen3_5's `State` with every layer attention, so
  qwen3_5's `commit` and `grow` apply unchanged.
- `prefill.py` runs prompt chunks on the prompt kernels (`b16.prompt`, the EXL3 and NVFP4 prompt GEMMs,
  `prefill_attention`). Their bits do not depend on chunk bounds but differ from decode's. The KV cache is the
  whole state, so a kept state is a clone of the prefilled state with `pos` set back.
- `decode.py` has `prefill`, `serial_decode` and `draft_decode`. A copy's window starts at 16 rows, doubles while
  copies land whole and halves when one breaks, up to 128. Without a copy, the drafter proposes a tree and every
  committed row's residuals at the drafter's tap layers (1, 25 and 49 for Nanbeige) feed its context.
- The drafter is qwen3_5's `DFlash2`, reading its tap layers, candidate count and window from the drafter's
  config.json. A config with `sliding_window: null` keeps the last 2,047 taps. Prefill hands it only the rows that
  window keeps at each kept prompt state, and the prefix cache stores its snapshot beside each state.
- `engine.py` is what `tensorfold serve` runs: startup memory admission, prefix reuse at message starts and at each
  prompt's end minus one token.
- `../reference.py` is an fp32 forward in transformers' Llama math for the quality test.

`--parallel N` is refused. qwen3_5's `multi.py` shares rounds across streams through DFlash2 trees and GDN state
tables, and none of it carries over without a rewrite.

The tests are `tests/test_llama_dense_family.py` (CPU) and `tests/cuda/test_llama_dense_*.py` (GPU).
