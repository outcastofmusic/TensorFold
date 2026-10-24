"""The CUDA server ends a reply's reasoning at the model's close token, not at the first ``</think>`` the text spells:
a reply that quotes the tag while it reasons keeps the quote in its reasoning, streamed and whole (upstream #1180)."""

from __future__ import annotations

import json
import threading

from tensorfold.server.text import split_thinking

QUOTE = "Qwen closes its block with </think> and then answers.\n"
REASONING = "The user asks how the tag works. " + QUOTE
ANSWER = "It writes the tag once, as one token."
CLOSE = 1002


def test_split_thinking_follows_the_close_tokens_place():
    reply = REASONING + "</think>\n\n" + ANSWER
    at = len(REASONING)
    assert split_thinking(reply, finished=True, close_at=at) == (REASONING, ANSWER)
    assert split_thinking(reply[:at + 3], finished=False, close_at=-1)[1] == ""    # the token has not come yet
    assert split_thinking(REASONING, finished=True, close_at=-1) == (REASONING, "")
    assert split_thinking(reply, finished=True, close_at=at - 1) == (reply, "")     # no tag there: all reasoning
    assert split_thinking(reply, finished=True) == ("The user asks how the tag works. Qwen closes its block with ",
                                                    reply[reply.index("</think>") + 8:].lstrip("\n"))


def test_the_cuda_server_keeps_a_quoted_close_tag_in_the_reasoning(tmp_path):
    from tensorfold.cuda import server
    from tests.test_cuda_admission import http_server, post as cuda_post
    from tests.test_think_call import events

    special = {"</think>": CLOSE}

    class Tokens:
        def encode(self, text, **kwargs):
            return type("Encoding", (), {"ids": [ord(c) for c in text]})()    # a quoted tag is plain characters

        def decode(self, ids, **kwargs):
            names = {v: k for k, v in special.items()}
            return "".join(names.get(i, "" if i == 0 else chr(i)) for i in ids)

        def token_to_id(self, text):
            return special.get(text)

    class Engine:
        eos = (0,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            reply = Tokens().encode(REASONING).ids + [CLOSE] + Tokens().encode("\n\n" + ANSWER).ids + [0]
            for at in range(0, len(reply), 3):
                on_tokens(reply[at:at + 3])
            return {"rounds": 1}

    template = "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:<think>"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = Engine(), "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.sampling, app.max_tokens = True, {"temperature": 0.0}, 256
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    for stream in (False, True):
        with http_server(app) as port:
            status, body = cuda_post(port, {"messages": [{"role": "user", "content": "How?"}],
                                            "chat_template_kwargs": {"enable_thinking": True}, "stream": stream}, True)
        assert status == 200
        if stream:
            deltas = [c["choices"][0]["delta"] for c in events(body) if c.get("choices")]
            assert "".join(d.get("reasoning_content", "") for d in deltas) == REASONING
            assert "".join(d.get("content") or "" for d in deltas) == ANSWER
        else:
            message = json.loads(body)["choices"][0]["message"]
            assert message["reasoning_content"] == REASONING and message["content"] == ANSWER
