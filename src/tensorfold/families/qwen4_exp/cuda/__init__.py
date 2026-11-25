"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

import os

DEPTH = 6            # most MTP drafts a round
# a chain ends before a draft the MTP head gives less than this (one stream or many; TF_FLASHNEXT_DRAFT_FLOOR)
CONFIDENCE = float(os.environ.get("TF_FLASHNEXT_DRAFT_FLOOR", "0.7"))
# TF_FLASHNEXT_DRAFT_PRODUCT=<x>: a chain also ends before a draft that takes the product of its drafts' probabilities
# below x (0: off). Mia's patch 0017 keeps drafts while that product is at least 0.4 over a 0.5 floor: greedy code
# +13.6% at 1 stream and +4.3% at 5 in her harness, against v1.1.4's 0.6 floor (her CHANGELOG, 2026-10-27)
PRODUCT = float(os.environ.get("TF_FLASHNEXT_DRAFT_PRODUCT", "0"))
CONTEXT = 8192       # prompt plus reply tokens the caches hold
