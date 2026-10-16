"""``/v1/models`` tells a client which reasoning efforts a model takes (upstream TensorFold 1.0.12, #819)."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import server


class Tok:
    def __init__(self, think_close: bool) -> None:
        self.think_close = think_close

    def token_to_id(self, text):
        return 7 if text == "</think>" and self.think_close else None


def app(efforts=frozenset(), switch=False, think_close=False, thinking=True, effort=None):
    a = server.App.__new__(server.App)
    a.template = NS(efforts=frozenset(efforts), thinking_switch=switch)
    a.tok, a.default_thinking, a.reasoning_effort = Tok(think_close), thinking, effort
    return a


def test_qwen_flash_next_lists_its_levels_highest_first_then_none():
    info = app({"low", "medium", "xhigh"}, switch=True).reasoning_info()
    assert info == {"default_enabled": True, "supported_efforts": ["xhigh", "medium", "low", "none"]}


def test_glm_lists_its_levels_without_none_and_a_server_default_effort_in_the_templates_names():
    info = app({"low", "high", "max"}, think_close=True, effort="medium").reasoning_info()
    assert info == {"default_enabled": True, "default_effort": "high", "supported_efforts": ["max", "high", "low"]}


def test_a_switch_or_a_think_close_without_levels_gives_the_default_alone():
    assert app(switch=True, thinking=False).reasoning_info() == {"default_enabled": False}
    assert app(think_close=True).reasoning_info() == {"default_enabled": True}


def test_a_model_that_does_not_think_has_no_reasoning():
    assert app().reasoning_info() is None


def test_the_models_route_carries_it(tmp_path):
    from tests.test_cuda_admission import http_server
    from tests.test_cuda_tool_choice import Engine, app_for
    from tests.test_responses_api import call

    served = app_for(tmp_path, Engine())                  # its template switches thinking and names no level
    with http_server(served) as port:
        status, raw = call(port, "GET", "/v1/models")
    entry = json.loads(raw)["data"][0]
    assert status == 200 and entry["id"] == "fake-cuda"
    assert entry["reasoning"] == {"default_enabled": False}
