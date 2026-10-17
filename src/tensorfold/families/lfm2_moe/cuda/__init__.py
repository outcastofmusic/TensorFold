"""LFM2-MoE on CUDA: one stream, prompt chunks then one-row decode steps, keyed sampling as the other CUDA families."""

CONTEXT = 32768            # prompt plus reply tokens when --context is not given (the model's window is 131,072)
PREFILL_ROWS = 2048        # rows of a prompt chunk
DECODE_ROWS = 16           # rows the grouped expert kernel takes; wider calls group the rows by expert
WORKSPACE = 2 << 30        # bytes beside the weights and caches: a chunk's activations and one expert's fp32 decode
