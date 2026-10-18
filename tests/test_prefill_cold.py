"""tools/prefill_cold.py's build: the lengths it accepts against the corpus it cuts prompts from."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cold():
    spec = importlib.util.spec_from_file_location("prefill_cold", Path(__file__).parents[1] / "tools/prefill_cold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_default_lengths_fit_a_corpus_of_twenty_times_the_longest(cold):
    assert cold.checked(cold.LENGTHS, 20 * max(cold.LENGTHS) + 1) == cold.LENGTHS


def test_the_warm_length_is_dropped_from_the_user_s_lengths(cold):
    assert cold.checked((1024, 4096, 1024, 2048), 10**6) == (4096, 2048)


@pytest.mark.parametrize("lengths", [(50_000,), (2048, 50_001), (0,), (-5, 2048)])
def test_a_length_the_corpus_cannot_fill_raises_before_any_tokenizer_work(cold, lengths):
    with pytest.raises(ValueError, match="1 to 49,999 tokens fit this corpus"):
        cold.checked(lengths, 1_000_000)
