"""With ``parallel_tool_calls: false`` the CUDA server keeps a rejected call envelope (an unoffered tool, a malformed
body) in the reply's content, as parallel replies do, and the envelope uses up no call: a valid call after it is the
reply's one call. Streamed and whole replies agree (upstream 99186d7d, #1207)."""

from __future__ import annotations

import json
import threading

import pytest

from tensorfold.server.tool_policy import ToolCallPolicy

TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}}}}]
UNOFFERED = "<tool_call>\n<function=nope>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
MALFORMED = "<tool_call>\n<function=terminal>\nstray text\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
VALID = "<tool_call>\n<function=terminal>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
SECOND = VALID.replace("ls", "pwd")


def accept(envelope):
    from tensorfold.cuda.reply_text import parse_tool_calls

    return parse_tool_calls(envelope, TOOLS, max_calls=1)[1] is not None


@pytest.mark.parametrize("bad", [UNOFFERED, MALFORMED])
def test_the_policy_shows_a_rejected_envelope_and_hides_the_one_call(bad):
    policy = ToolCallPolicy({"parallel_tool_calls": False})
    text = f"Try {bad} then {VALID} and {SECOND} done"
    assert policy.content(text, accept=accept) == f"Try {bad} then  and  done"
    assert policy.content(text) == "Try  then  and  done"                  # without ``accept``: every envelope hides
    for n in range(len(text)):                                              # streamed text only grows
        assert policy.content(text, accept=accept).startswith(policy.content(text[:n], finished=False, accept=accept))


@pytest.mark.parametrize("bad", [UNOFFERED, MALFORMED])
def test_the_cuda_server_keeps_a_rejected_single_call(tmp_path, bad):
    from tensorfold.cuda import server
    from tests.test_cuda_admission import http_server, post as cuda_post
    from tests.test_think_call import events

    special = {"<tool_call>": 1000, "</tool_call>": 1001}
    reply_text = f"Let me look. {bad}\nThen {VALID}"

    class Tokens:
        def encode(self, text, **kwargs):
            import re

            parts = re.split("(<tool_call>|</tool_call>)", text)
            return type("Encoding", (), {"ids": [i for p in parts for i in ([special[p]] if p in special
                                                                             else [ord(c) for c in p])]})()

        def decode(self, ids, **kwargs):
            names = {v: k for k, v in special.items()}
            return "".join(names.get(i, "" if i == 0 else chr(i)) for i in ids)

        def token_to_id(self, text):
            return special.get(text)

    class Engine:
        eos = (0,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            reply = Tokens().encode(reply_text).ids + [0]
            for at in range(0, len(reply), 5):
                on_tokens(reply[at:at + 5])
            return {"rounds": 1}

    template = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(), "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.sampling, app.max_tokens = False, {"temperature": 0.0}, 512
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    want = f"Let me look. {bad}\nThen"
    for stream in (False, True):
        with http_server(app) as port:
            status, body = cuda_post(port, {"messages": [{"role": "user", "content": "List files"}], "tools": TOOLS,
                                            "parallel_tool_calls": False, "stream": stream}, True)
        assert status == 200
        if stream:
            chunks = events(body)
            deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
            assert "".join(d.get("content") or "" for d in deltas).strip() == want
            calls = [d["tool_calls"][0] for d in deltas if d.get("tool_calls")]
            assert [json.loads(c["function"]["arguments"]) for c in calls] == [{"command": "ls"}]
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        else:
            choice = json.loads(body)["choices"][0]
            assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] == want
            assert [json.loads(c["function"]["arguments"]) for c in choice["message"]["tool_calls"]] == [
                {"command": "ls"}]
