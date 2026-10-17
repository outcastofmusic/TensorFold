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
    with socket.socket() as free:                        # no backend: a swap's persist must not reach a real server
        free.bind(("127.0.0.1", 0))
        module.BACKEND = free.getsockname()[1]
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


def test_requests_waiting_for_the_model_just_loaded_go_before_a_swap_back(swap, tmp_path):
    log = tmp_path / "loads"
    run = tmp_path / "run.sh"
    run.write_text(f"#!/bin/sh\necho $1 >> {log}\nsleep 0.3\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    swapper = swap.Swapper()
    swapper.current = "qwen3.8-flash-next"
    got = []

    def request(model):
        got.append(swapper.enter(model))
        swapper.leave()

    first = threading.Thread(target=request, args=("glm-5.3-flash",))
    first.start()
    time.sleep(0.1)                                  # the glm load is running: these three wait for it
    waiting = [threading.Thread(target=request, args=(m,))
               for m in ("qwen3.8-flash-next", "glm-5.3-flash", "glm-5.3-flash")]
    for t in waiting:
        t.start()
        time.sleep(0.02)
    for t in [first, *waiting]:
        t.join(5)
    assert log.read_text().split() == ["glm-5.3-flash", "qwen-flash-next"]   # one swap there, one back
    assert got[-1] == "qwen3.8-flash-next"


@pytest.mark.parametrize("headers, status", [
    (b"Content-Length: -1\r\n", b" 400 "),
    (b"Content-Length: lots\r\n", b" 400 "),
    (b"Content-Length: 999999999999\r\n", b" 413 "),
    (b"Transfer-Encoding: chunked\r\n", b" 411 "),
])
def test_a_body_without_a_usable_length_is_refused(swap, headers, status):
    swap.SWAP = swap.Swapper()
    front = _serve(swap.Handler)
    try:
        client = socket.create_connection(front.server_address, timeout=5)
        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + headers + b"\r\n")
        assert status in client.recv(200).split(b"\r\n")[0] + b" "
    finally:
        front.shutdown()


class _Persist(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log: Path

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with open(_Persist.log, "a") as f:
            f.write(f"{self.path}\n")
        body = json.dumps({"disk": True, "written": 2, "states": 5}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_swap_has_the_outgoing_model_write_its_states_before_run_sh_stops_it(swap, tmp_path, capsys):
    log = tmp_path / "order"
    _Persist.log = log
    backend = _serve(_Persist)
    swap.BACKEND = backend.server_address[1]
    run = tmp_path / "run.sh"
    run.write_text(f"#!/bin/sh\necho run $1 >> {log}\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    try:
        swapper = swap.Swapper()
        swapper.current = "qwen3.8-flash-next"
        assert swapper.enter("glm-5.3-flash") == "glm-5.3-flash"
        swapper.leave()
    finally:
        backend.shutdown()
    assert log.read_text().splitlines() == ["/internal/persist", "run glm-5.3-flash"]
    assert "qwen3.8-flash-next wrote 2 kept states to disk" in capsys.readouterr().out


def test_a_swap_goes_on_when_the_outgoing_model_cannot_write(swap, tmp_path, capsys):
    log = tmp_path / "order"
    run = tmp_path / "run.sh"
    run.write_text(f"#!/bin/sh\necho run $1 >> {log}\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    swapper = swap.Swapper()
    swapper.current = "qwen3.8-flash-next"
    assert swapper.enter("glm-5.3-flash") == "glm-5.3-flash"     # nothing listens on the backend port
    assert log.read_text().split() == ["run", "glm-5.3-flash"]
    assert "did not write its kept states" in capsys.readouterr().out


def test_every_served_name_has_a_run_sh_target():
    """swap.py runs `run.sh <target>` for a name: a name whose target run.sh does not dispatch would fail every swap."""
    import re

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("swap_names", root / "swap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dispatch = re.findall(r"^    ([a-z0-9.-]+)\)\s+ask_swap", (root / "run.sh").read_text(), re.M)
    assert set(mod.TARGET.values()) == set(dispatch)
    assert {"qwen3.8-flash-next-fp4", "nemotron-3.5-bolt"} <= set(mod.TARGET)
