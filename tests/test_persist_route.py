"""``POST /internal/persist``: the CUDA server writes its kept prompt states to disk and answers once they are there,
under the engine's turns on an engine that serves one request at a time; at once without disk states."""

from __future__ import annotations

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from tensorfold.cuda import server
from tensorfold.server.auth_http import gated


class Engine:
    concurrent = False

    def __init__(self, disk=None, fail=None):
        self.disk, self.fail, self.calls = disk, fail, []

    def persist(self, stop):
        self.calls.append((stop(), self.app.turns.busy if not self.concurrent else None))
        if self.fail is not None:
            raise self.fail
        return 2


def app_for(engine):
    app = object.__new__(server.App)
    app.engine = engine
    engine.app = app
    return app


@contextmanager
def served(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def post(port, path="/internal/persist"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("POST", path, b"", {"Content-Length": "0"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_states_in_memory_only_answer_at_once():
    engine = Engine()
    with served(app_for(engine)) as port:
        assert post(port) == (200, {"disk": False, "written": 0})
    assert engine.calls == []


def test_a_one_request_engine_writes_everything_under_its_turns():
    engine = Engine(disk=SimpleNamespace(entries=[1, 2, 3]))
    app = app_for(engine)
    with served(app) as port:
        status, got = post(port)
    assert status == 200 and got["disk"] and got["written"] == 2 and got["states"] == 3
    assert engine.calls == [(False, True)]                # never stopped, while holding the engine
    assert not app.turns.busy


def test_a_concurrent_engine_writes_through_its_own_worker():
    engine = Engine(disk=SimpleNamespace(entries=[]))
    engine.concurrent = True
    app = app_for(engine)
    with served(app) as port:
        assert post(port)[0] == 200
    assert engine.calls == [(False, None)] and "turns" not in app.__dict__


def test_a_failed_write_is_a_server_error_and_frees_the_engine():
    engine = Engine(disk=SimpleNamespace(entries=[]), fail=OSError(28, "No space left on device"))
    app = app_for(engine)
    with served(app) as port:
        status, got = post(port)
    assert status == 500 and "No space left" in got["error"]["message"]
    assert not app.turns.busy


def test_the_route_is_behind_the_api_key():
    assert gated("/internal/persist")
