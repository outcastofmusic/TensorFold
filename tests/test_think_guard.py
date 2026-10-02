"""The thinking guard: a decision model's yes closes the think block at a paragraph end, decoding never waits for a
check, and a failed check ends the checks while the reply finishes."""

import json
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tensorfold.cuda import server
from tensorfold.engine.call_gate import generate_gated
from tensorfold.engine.think_guard import GuardConfig, ThinkGuard, _ends, guard_state
from tests.test_cuda_admission import http_server, post

END, THINK_END = "<|end|>", "</think>"
TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}")


def tokenizer() -> Tokenizer:
    """One token a byte, so a blank line takes two tokens, plus the end and the think end as special tokens."""

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([END, THINK_END])
    return tok


TOK = tokenizer()
END_ID, THINK_END_ID = TOK.token_to_id(END), TOK.token_to_id(THINK_END)


def ids(text):
    return TOK.encode(text, add_special_tokens=False).ids


def text(tokens):
    return TOK.decode(list(tokens), skip_special_tokens=False)


class ScriptEngine:
    """Thinks ``script`` from wherever its prompt left off, then closes and answers "Uncut."; once its prompt holds a
    think end it answers "Answer." instead. ``pause`` seconds between rounds of ``width`` tokens."""

    eos = (END_ID,)

    def __init__(self, script, *, width=3, pause=0.0):
        self.script, self.width, self.pause, self.prompts = script, width, pause, []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        said = text(prompt).split("<think>\n", 1)[-1]
        if THINK_END in said:
            reply = ids("Answer.") + [END_ID]
        else:
            reply = ids(self.script[len(said):] + THINK_END + "\n\nUncut.") + [END_ID]
        for at in range(0, min(len(reply), max_tokens), self.width):
            if self.pause:
                time.sleep(self.pause)
            if on_tokens(reply[at:at + self.width]):
                break
        return {"rounds": 1}


def run_now(f, *args):
    """A finished check: its answer, or the failure it raised."""

    future = Future()
    try:
        future.set_result(f(*args))
    except OSError as exc:
        future.set_exception(exc)
    return future


def sync_guard(ask, threshold=0.8):
    """A guard whose checks answer before the next token, so cuts land where the rule says without timing."""

    return ThinkGuard(ask, threshold, think_end=THINK_END_ID, decode=text, encode=ids,
                      submit=run_now)


def gated(script, guard, width=3):
    engine, out = ScriptEngine(script, width=width), []
    prompt = ids("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n")
    generate_gated(lambda p, n, feed: engine.generate(p, n, None, feed), prompt, 400, [guard],
                   lambda new: out.extend(new) or False)
    return text(out)


def test_paragraph_ends_span_tokens_and_skip_blank_lines():
    assert _ends("", "One.\n\n") == ([6], "")
    assert _ends("One.\n", "\nTw") == ([1], "Tw")              # the blank line split across two tokens
    assert _ends("", "\n\n\n\nA") == ([], "A")                   # blank lines alone end no paragraph
    assert _ends("A", ".\n\nB.\n\nC") == ([3, 7], "C")


@pytest.mark.parametrize("width", [1, 3, 7])
def test_a_yes_closes_the_block_at_the_paragraph_it_answers(width):
    script = "One.\n\nTwo, settled.\n\nThree.\n\nFour.\n\n"
    asked = []
    guard = sync_guard(lambda r: asked.append(r) or (0.95 if "settled" in r else 0.05))
    assert gated(script, guard, width) == "One.\n\nTwo, settled.\n\n</think>\n\nAnswer.<|end|>"
    assert asked == ["One.\n\n", "One.\n\nTwo, settled.\n\n"]
    assert guard.record["checks"][1]["p"] == 0.95 and guard.record["yes_paragraph"] == 2
    assert guard.record["closed_at_paragraph"] == 2


def test_a_late_yes_closes_at_the_next_paragraph_end():
    """A yes that comes back mid-paragraph closes after that paragraph, inside the token that ends it."""

    check = Future()
    guard = ThinkGuard(lambda r: 0.95, 0.8, think_end=THINK_END_ID, decode=text, encode=ids,
                       submit=lambda f, reasoning: check)
    for token in ids("One.\n\nTwo"):
        assert guard.cut([token]) is None
        guard.observe(token)
    check.set_result(0.95)                       # paragraph 1's yes, back while paragraph 2 is written
    tokens = ids(" and more.\n\nThree")
    at, close = guard.cut(tokens)
    assert guard.yes == 1
    assert text(tokens[:at] + close) == " and more.\n\n</think>\n\n"


def test_no_yes_leaves_the_reply_whole():
    guard = sync_guard(lambda r: 0.1)
    assert gated("One.\n\nTwo.\n\n", guard) == "One.\n\nTwo.\n\n</think>\n\nUncut.<|end|>"
    assert [c["paragraph"] for c in guard.record["checks"]] == [1, 2] and "yes_paragraph" not in guard.record


def test_a_failed_check_ends_the_checks():
    calls = []

    def ask(reasoning):
        calls.append(reasoning)
        raise OSError("HTTP Error 502: Bad Gateway")

    guard = sync_guard(ask)
    assert gated("One.\n\nTwo.\n\nThree.\n\n", guard).endswith("Uncut.<|end|>")
    assert len(calls) == 1 and guard.record["error"] == "OSError: HTTP Error 502: Bad Gateway"


def test_the_state_keeps_the_newest_reasoning():
    state = guard_state("q", "é" * 20000 + "end")
    assert state.startswith("User's request:\nq\n\nReasoning so far:\n[earlier reasoning left out]\n")
    assert state.endswith("end") and "�" not in state


# --- through the server: real check threads, a decision server over HTTP ---------------------------------------

class Decisions:
    """A SystemOne server: yes (0.95) once the state says "settled", else 0.05; ``fail`` answers 502."""

    def __init__(self, fail=False):
        self.states, self.fail = [], fail
        decisions = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                decisions.states.append(body["state"])
                if decisions.fail or self.path != "/v1/systemone" or body["model"] != "clef":
                    self.send_response(502)
                    self.end_headers()
                    return
                p = 0.95 if "settled" in body["state"] else 0.05
                reply = json.dumps({"answers": {"enough": {"type": "noul", "noul": p}}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"


@pytest.fixture
def decisions():
    made = []

    def make(**kwargs):
        made.append(Decisions(**kwargs))
        return made[-1]

    yield make
    for d in made:
        d.httpd.shutdown()
        d.httpd.server_close()


def app_for(tmp_path, engine, guard=None):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = engine, "fake-cuda", TOK
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 2000
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.think_guard = guard
    return app


def ask(app, **fields):
    body = {"messages": [{"role": "user", "content": "Is it done?"}], **fields}
    with http_server(app) as port:
        status, reply = post(port, body, True)
    return status, json.loads(reply)


# paragraphs written after the settled one, so a check has time to come back while decoding goes on
LONG = "One.\n\nTwo, settled.\n\n" + "".join(f"Re-check {n}.\n\n" for n in range(1, 30))


def test_the_server_closes_thinking_after_the_yes_and_reports_the_checks(tmp_path, decisions):
    d = decisions()
    app = app_for(tmp_path, ScriptEngine(LONG, pause=0.002), GuardConfig(d.url, "clef"))
    status, body = ask(app)
    assert status == 200, body
    message, guard = body["choices"][0]["message"], body["tensorfold"]["thinking_guard"]
    assert message["content"] == "Answer."
    assert message["reasoning_content"].startswith("One.\n\nTwo, settled.\n\n")
    assert "Re-check 29" not in message["reasoning_content"]      # closed long before the script's end
    # paragraph 1's check may still run when 2 and 3 end; then 3, the newest, is checked and says yes
    paragraphs = [p for p in message["reasoning_content"].split("\n\n") if p.strip()]
    assert 2 <= guard["yes_paragraph"] <= guard["closed_at_paragraph"] == len(paragraphs)
    assert guard["model"] == "clef" and guard["checks"][-1]["p"] == 0.95
    assert d.states[0] == "User's request:\nIs it done?\n\nReasoning so far:\nOne.\n\n"


def test_decoding_does_not_wait_for_a_check(tmp_path, decisions):
    """A decision server slower than the whole reply: the reply finishes at full speed, unguarded."""

    d = decisions()
    slow = d.httpd.RequestHandlerClass.do_POST
    d.httpd.RequestHandlerClass.do_POST = lambda self: (time.sleep(2), slow(self))
    app = app_for(tmp_path, ScriptEngine(LONG), GuardConfig(d.url, "clef"))
    started = time.perf_counter()
    status, body = ask(app)
    assert status == 200 and time.perf_counter() - started < 1.5
    assert body["choices"][0]["message"]["content"] == "Uncut."
    assert body["tensorfold"]["thinking_guard"]["checks"] == []


@pytest.mark.parametrize("fields, want", [({"thinking_guard": False}, None), ({}, 0.8),
                                          ({"thinking_guard": {"threshold": 0.99}}, 0.99)])
def test_the_request_turns_the_guard_off_or_sets_its_threshold(tmp_path, decisions, fields, want):
    d = decisions()
    app = app_for(tmp_path, ScriptEngine(LONG, pause=0.002), GuardConfig(d.url, "clef"))
    status, body = ask(app, **fields)
    assert status == 200, body
    guard = body["tensorfold"].get("thinking_guard")
    assert (guard and guard["threshold"]) == want
    assert (body["choices"][0]["message"]["content"] == "Answer.") == (want == 0.8)   # 0.95 is under 0.99


def test_a_failing_decision_server_leaves_the_reply_whole(tmp_path, decisions):
    d = decisions(fail=True)
    app = app_for(tmp_path, ScriptEngine(LONG, pause=0.002), GuardConfig(d.url, "clef"))
    status, body = ask(app)
    assert status == 200 and body["choices"][0]["message"]["content"] == "Uncut."
    assert "502" in body["tensorfold"]["thinking_guard"]["error"] and len(d.states) == 1


@pytest.mark.parametrize("fields, words", [
    ({"thinking_guard": {"threshold": 0}}, "above 0 and at most 1"),
    ({"thinking_guard": {"threshold": True}}, "above 0 and at most 1"),
    ({"thinking_guard": "yes"}, 'must be false, true, or {"threshold": t}'),
    ({"thinking_guard": {"url": "http://elsewhere"}}, 'must be false, true, or {"threshold": t}'),
])
def test_a_bad_guard_field_is_refused(tmp_path, decisions, fields, words):
    app = app_for(tmp_path, ScriptEngine(LONG), GuardConfig(decisions().url, "clef"))
    status, body = ask(app, **fields)
    assert status == 400 and words in body["error"]["message"]


def test_a_threshold_without_a_configured_guard_is_refused_and_absent_is_no_guard(tmp_path):
    app = app_for(tmp_path, ScriptEngine("One.\n\n"))
    status, body = ask(app, thinking_guard={"threshold": 0.5})
    assert status == 400 and "--thinking-guard" in body["error"]["message"]
    status, body = ask(app)
    assert status == 200 and "thinking_guard" not in body["tensorfold"]


def test_no_guard_with_thinking_off(tmp_path, decisions):
    d = decisions()
    app = app_for(tmp_path, ScriptEngine(LONG), GuardConfig(d.url, "clef"))
    status, body = ask(app, chat_template_kwargs={"enable_thinking": False})
    assert status == 200 and "thinking_guard" not in body["tensorfold"] and d.states == []
