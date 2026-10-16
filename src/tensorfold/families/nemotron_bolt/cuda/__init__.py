"""Nemotron 3.5 Bolt on CUDA: a verify window's rows get the serial step's bits, so drafted replies equal serial ones."""

CONTEXT = 16384       # prompt plus reply tokens when --context is not given
DRAFTS = 7            # draft tokens a round with the LiLiCorr drafter (its card measures draft length 7)
WORKSPACE = 2 << 30   # bytes beside the weights and caches: activations and the fallback's dequantized experts
