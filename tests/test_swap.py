"""swap.py: a failed load frees the endpoint, and the proxy lets go of a backend when its client leaves."""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def swap(monkeypatch):
    spec = importlib.util.spec_from_file_location("swap_under_test", ROOT / "swap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "loaded", lambda: None)
    return module


def test_a_load_that_cannot_start_frees_the_endpoint(swap, tmp_path):
    swap.RUN = str(tmp_path / "missing-run.sh")          # subprocess.run raises FileNotFoundError
    swapper = swap.Swapper()
    with pytest.raises(RuntimeError, match="did not load"):
        swapper.enter("glm-5.3-flash")
    assert swapper.loading is None
    with pytest.raises(RuntimeError):                    # a second request tries again instead of waiting forever
        swapper.enter("glm-5.3-flash")


def test_the_loaded_check_after_a_failed_load_runs_outside_the_lock(swap, tmp_path, monkeypatch):
    swap.RUN = str(tmp_path / "missing-run.sh")
    swapper = swap.Swapper()
    held = []
    monkeypatch.setattr(swap, "loaded", lambda: held.append(swapper.cond._is_owned()))
    with pytest.raises(RuntimeError):
        swapper.enter("glm-5.3-flash")
    assert held == [False]


class _Slow(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    closed = threading.Event()

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        # a non-streamed reply: nothing is sent until generation ends, which a closed connection ends early
        self.connection.settimeout(10)
        try:
            gone = self.connection.recv(1) == b""
        except OSError:
            gone = True
        if gone:
            _Slow.closed.set()


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_client_that_leaves_a_non_streamed_request_closes_the_backend_connection(swap):
    _Slow.closed.clear()
    backend = _serve(_Slow)
    swap.BACKEND = backend.server_address[1]
    swap.SWAP = swap.Swapper()
    swap.SWAP.current = "qwen3.8-flash-next"
    front = _serve(swap.Handler)
    try:
        body = json.dumps({"model": "qwen3.8-flash-next", "messages": []}).encode()
        client = socket.create_connection(front.server_address)
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                       b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        deadline = time.monotonic() + 5
        while swap.SWAP.running == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        client.close()
        assert _Slow.closed.wait(5), "the backend connection stayed open after the client left"
        deadline = time.monotonic() + 5
        while swap.SWAP.running and time.monotonic() < deadline:
            time.sleep(0.05)
        assert swap.SWAP.running == 0
    finally:
        front.shutdown()
        backend.shutdown()
