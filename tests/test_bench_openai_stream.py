"""bench_openai times a stream from its first to its last token, reasoning tokens included."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "bench_openai.py"


@pytest.fixture
def bench():
    spec = importlib.util.spec_from_file_location("bench_openai", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _serve(monkeypatch, bench, chunks):
    """urlopen answers with these SSE chunks; the clock moves one second a chunk."""

    lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]
    clock = iter(range(100))

    class Response:
        def __enter__(self):
            return iter(lines)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda req, timeout: Response())
    monkeypatch.setattr(bench.time, "perf_counter", lambda: float(next(clock)))


def _delta(**fields):
    return {"choices": [{"delta": fields}]}


def test_reasoning_tokens_are_timed_and_only_content_is_kept(monkeypatch, bench):
    _serve(monkeypatch, bench, [_delta(reasoning_content="hm"), _delta(reasoning="so"), _delta(content="hi"),
                                {"choices": [], "usage": {"completion_tokens": 3}}])
    got = bench.stream("http://x", "m", bench.PROMPTS[1], 3, 0.0, None)
    assert got == {"ttft_s": 1.0, "decode_s": 2.0, "tokens": 3, "decode_tps": 1.0, "text": "hi"}


def test_a_stream_without_tokens_names_its_prompt(monkeypatch, bench):
    _serve(monkeypatch, bench, [_delta(role="assistant"), {"choices": [], "usage": {"completion_tokens": 0}}])
    with pytest.raises(RuntimeError, match="gpu-chat-no-think: the stream carried no text or reasoning token"):
        bench.stream("http://x", "m", bench.PROMPTS[1], 3, 0.0, None)
