"""GLM call arguments read in linear time and equal to the regex they replace (upstream #883)."""

import random
import time

from tensorfold.server.tools import _GLM_ARG_RE, _glm_args, parse_glm_tool_call_block

PIECES = ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", " ", "\n", "k", "v1", "<", ">", "arg"]


def test_matches_and_leftover_equal_the_regex_on_random_calls():
    rng = random.Random(883)
    for _ in range(20_000):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 24)))
        pairs, left = _glm_args(text)
        assert pairs == _GLM_ARG_RE.findall(text), text
        assert left == _GLM_ARG_RE.sub("", text), text


def test_many_keys_without_values_read_quickly():
    block = "f" + "".join(f"<arg_key>k{i}</arg_key>" for i in range(8_000))       # the regex took 6.1 s
    start = time.perf_counter()
    assert parse_glm_tool_call_block(block, None, complete=True) is None
    assert time.perf_counter() - start < 0.5


def test_a_call_reads_as_before():
    block = "get_weather<arg_key>city</arg_key>\n<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>"
    assert parse_glm_tool_call_block(block, None, complete=True) == ("get_weather", {"city": "Paris", "days": "3"})
