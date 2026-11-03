"""``"transient": true``: a one-off request (a tool's extraction call inside an agent's turn) keeps no prompt state on
concurrent Flash Next, so its slot frees when it ends and no agent's kept state is pushed out. CPU only."""

import importlib
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tests.test_cuda_admission import http_server
from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the modules import)
from tests.test_cuda_server_errors import HI, app_for


class Recording:
    """A concurrent stand-in engine that records the ``transient`` flag ``App.run`` passed."""

    eos = (0,)
    concurrent = True

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, transient=False):
        self.calls.append(transient)
        on_tokens([self.eos[0]])
        return {}


class NoTransient(Recording):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append(None)
        on_tokens([self.eos[0]])
        return {}


def _post(port, body):
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


@pytest.mark.parametrize("field, expected", [({}, False), ({"transient": False}, False), ({"transient": True}, True)])
def test_the_request_field_reaches_a_concurrent_engine(tmp_path, field, expected):
    app = app_for(tmp_path)
    app.engine = Recording()
    with http_server(app) as port:
        status, _ = _post(port, {"model": "fake-cuda", "messages": HI, "max_tokens": 4, **field})
    assert status == 200 and app.engine.calls == [expected]


TITLE = [{"role": "system", "content": "Write a short title for this conversation."},
         {"role": "user", "content": "How do I free a stream slot?"}]


@pytest.mark.parametrize("field, expected", [({}, True), ({"transient": False}, False), ({"transient": True}, True)])
def test_a_title_request_is_transient_unless_it_says_otherwise(tmp_path, field, expected):
    app = app_for(tmp_path)
    app.engine = Recording()
    with http_server(app) as port:
        status, _ = _post(port, {"model": "fake-cuda", "messages": TITLE, "max_tokens": 4, **field})
    assert status == 200 and app.engine.calls == [expected]


@pytest.mark.parametrize("value", ["yes", 1, None, {}])
def test_a_transient_that_is_not_a_bool_is_refused(tmp_path, value):
    app = app_for(tmp_path)
    app.engine = Recording()
    with http_server(app) as port:
        status, text = _post(port, {"model": "fake-cuda", "messages": HI, "max_tokens": 4, "transient": value})
    assert status == 400 and "transient must be true or false" in text and app.engine.calls == []


def test_an_engine_without_kept_slots_serves_a_transient_request_as_usual(tmp_path):
    app = app_for(tmp_path)
    app.engine = NoTransient()
    with http_server(app) as port:
        status, _ = _post(port, {"model": "fake-cuda", "messages": HI, "max_tokens": 4, "transient": True})
    assert status == 200 and app.engine.calls == [None]


class Lanes:
    """A decoder that records each admitted stream's flag and ends it after one token."""

    def __init__(self):
        self.streams, self.admitted, self.next = {}, [], 0

    def live(self):
        return len(self.streams)

    def admit(self, s):
        s.sid, self.next = self.next, self.next + 1
        self.admitted.append(s.transient)
        self.streams[s.sid] = s
        s.take([1])

    def round(self):
        time.sleep(0.002)
        return [s for s in self.streams.values() if s.done]

    def finish(self, done):
        for s in done:
            self.streams.pop(s.sid, None)

    def drop(self):
        live, self.streams = list(self.streams.values()), {}
        return live


def test_the_scheduler_hands_the_flag_to_the_stream_and_a_replay_keeps_it():
    dec = Lanes()
    sched = Scheduler(dec, max_streams=2)
    try:
        sched.submit([5, 6, 7], 1, None, True, lambda new: False, transient=True)
        sched.submit([5, 6, 7], 1, None, True, lambda new: False)
    finally:
        sched.close()
    assert dec.admitted == [True, False]
    assert Stream([1, 2], 4, None, background=True, transient=True).continued().transient


def _bare_decoder(multi, *, free=(), kept=()):
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.streams, dec.free, dec.kept, dec.keep = {}, list(free), list(kept), 8
    dec.filling, dec.fills, dec.held = [], {}, {}
    dec.solo, dec.link, dec.spec = None, None, None
    return dec


def _stream(prompt, *, transient, st):
    s = Stream(list(prompt), 4, None, transient=transient)
    s.sid, s.st = 0, st
    return s


def test_a_transient_prefill_keeps_no_prompt_state(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    from tensorfold.families.qwen4_exp.cuda.decode import entry_end

    dec = _bare_decoder(multi)
    prompt = list(range(600))
    st = SimpleNamespace(image_positions=None)
    for transient, expected in ((False, [100, entry_end(prompt)]), (True, [])):
        s = _stream(prompt, transient=transient, st=st)
        dec.fills = {0: [SimpleNamespace(stops=[100]), True, 0, None]}
        assert dec._points(s, 0, len(prompt)) == expected


def test_a_transient_reply_end_is_not_kept(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")

    dec = _bare_decoder(multi)
    remembered = []
    dec._remember = lambda *a: remembered.append(a)
    dec._flush = lambda s: pytest.fail("a transient stream's rows are not prepared for keeping")
    s = _stream([1, 2, 3], transient=True, st=SimpleNamespace(image_positions=None))
    dec._keep_end(s, 0, [7], 1)
    assert remembered == []


def test_a_transient_request_reads_no_disk_state(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")

    dec = _bare_decoder(multi)
    dec.disk = SimpleNamespace(find=lambda *a: pytest.fail("the disk is not searched for a transient request"))
    assert dec._on_disk(_stream([1, 2, 3], transient=True, st=None)) is None


def test_a_finished_transient_stream_frees_its_slot_and_leaves_the_agent_kept(allocations):  # noqa: F811
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")

    agent, spare = object(), object()
    agent_entry = (list(range(500)), agent, {}, None)
    dec = _bare_decoder(multi, kept=[agent_entry])
    dec._shrink = lambda st, **kw: None
    s = _stream([9] * 50, transient=True, st=spare)
    dec.streams = {0: s}
    dec.finish([s])                       # no prompt point was kept for it (above), so nothing holds its slot
    assert dec.free == [spare] and dec.kept == [agent_entry]
    chosen, resume, _ = dec._slot_for([8] * 50, True)    # the next one-off takes the freed slot, not the agent's
    assert chosen is spare and resume is None and dec.kept == [agent_entry]
