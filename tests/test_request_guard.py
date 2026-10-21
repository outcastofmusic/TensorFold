"""Host and Origin checks: a loopback bind answers only loopback names, and a browser Origin must be this server's."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tensorfold.server import auth_http
from tensorfold.server.request_guard import blocked


@pytest.mark.parametrize("host", ["127.0.0.1:30000", "localhost:30000", "[::1]:30000", "127.0.0.1", "LOCALHOST"])
def test_loopback_names_pass_a_loopback_bind(host):
    assert blocked([host], [], loopback_bind=True) is None


@pytest.mark.parametrize("hosts", [["evil.example:30000"], ["192.168.1.5:30000"], [], ["127.0.0.1", "127.0.0.1"],
                                   ["bad host"], ["127.0.0.1:99999"], ["127.0.0.1%25lo"]])
def test_other_hosts_are_refused_on_a_loopback_bind(hosts):
    assert blocked(hosts, [], loopback_bind=True) == "host"


def test_a_network_bind_takes_any_wellformed_host():
    assert blocked(["spark-143b:30000"], [], loopback_bind=False) is None
    assert blocked(["bad host"], [], loopback_bind=False) == "host"


def test_origin_must_be_this_server_or_listed():
    host = ["127.0.0.1:30000"]
    assert blocked(host, ["http://127.0.0.1:30000"], True) is None
    assert blocked(host, ["http://evil.example"], True) == "origin"           # a page elsewhere
    assert blocked(host, ["https://127.0.0.1:30000"], True) == "origin"       # another scheme is another origin
    assert blocked(host, ["http://127.0.0.1:4000"], True) == "origin"         # another port too
    assert blocked(host, ["null"], True) == "origin"
    assert blocked(host, ["http://a", "http://b"], True) == "origin"
    assert blocked(host, ["https://ui.example:8443"], True, ["https://ui.example:8443"]) is None
    assert blocked(["127.0.0.1"], ["http://127.0.0.1"], True) is None          # port 80 on both sides


class _Ok(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), auth_http.handler(_Ok, SimpleNamespace(auth=None)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _get(port, headers):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("GET", "/v1/models", skip_host=True)
    for name, value in headers:
        conn.putheader(name, value)
    conn.endheaders()
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_a_served_request_is_refused_before_dispatch(server):
    host = f"127.0.0.1:{server}"
    assert _get(server, [("Host", host)]) == (200, b"ok")
    assert _get(server, [("Host", host), ("Origin", f"http://{host}")]) == (200, b"ok")
    status, body = _get(server, [("Host", f"rebound.example:{server}")])
    assert status == 403 and "loopback" in json.loads(body)["error"]["message"]
    status, body = _get(server, [("Host", host), ("Origin", "http://evil.example")])
    assert status == 403 and "Origin" in json.loads(body)["error"]["message"]
