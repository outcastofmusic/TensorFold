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


class _NoPersist(BaseHTTPRequestHandler):
    """Upstream's native server: no /internal/persist route, so a 404 like any unknown path."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"error": {"message": f"unknown path {self.path}"}}).encode()
        self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_a_swap_from_a_server_without_a_persist_route_goes_on_quietly(swap, tmp_path, capsys):
    log = tmp_path / "order"
    run = tmp_path / "run.sh"
    run.write_text(f"#!/bin/sh\necho run $1 >> {log}\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    backend = _serve(_NoPersist)
    swap.BACKEND = backend.server_address[1]
    try:
        swapper = swap.Swapper()
        swapper.current = "nemotron-3.5-super"
        assert swapper.enter("qwen3.8-flash-next") == "qwen3.8-flash-next"
    finally:
        backend.shutdown()
    assert log.read_text().split() == ["run", "qwen-flash-next"]
    assert "kept states" not in capsys.readouterr().out


def test_every_served_name_has_a_run_sh_target():
    """swap.py runs `run.sh <target>` for a name: a name whose target run.sh does not dispatch would fail every swap."""
    import re

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("swap_names", root / "swap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dispatch = re.findall(r"^    ([a-z0-9.-]+)\)\s+ask_swap", (root / "run.sh").read_text(), re.M)
    assert set(mod.TARGET.values()) == set(dispatch)
    assert {"qwen3.8-flash-next-fp4", "nemotron-3.5-bolt", "lfm2.5-24b-a2b", "nemotron-3.5-super"} <= set(mod.TARGET)


def _run_logger(swap, tmp_path):
    log = tmp_path / "order"
    run = tmp_path / "run.sh"
    run.write_text(f"#!/bin/sh\necho run $1 >> {log}\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    _Persist.log = log
    backend = _serve(_Persist)
    swap.BACKEND = backend.server_address[1]
    return log, backend


def test_a_swap_that_cuts_requests_off_skips_the_persist(swap, tmp_path, monkeypatch):
    """Requests still running hold the engine, so a persist would wait for them: the swap goes straight to run.sh."""
    log, backend = _run_logger(swap, tmp_path)
    monkeypatch.setattr(swap, "DRAIN", 0)
    try:
        swapper = swap.Swapper()
        swapper.current, swapper.running = "qwen3.8-flash-next", 1
        assert swapper.enter("glm-5.3-flash") == "glm-5.3-flash"
    finally:
        backend.shutdown()
    assert log.read_text().splitlines() == ["run glm-5.3-flash"]


def test_a_first_load_has_nothing_to_persist(swap, tmp_path):
    log, backend = _run_logger(swap, tmp_path)
    try:
        assert swap.Swapper().enter("glm-5.3-flash") == "glm-5.3-flash"
    finally:
        backend.shutdown()
    assert log.read_text().splitlines() == ["run glm-5.3-flash"]


class _SwapEndpoint(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server's name
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def test_a_direct_start_under_the_swap_endpoint_takes_no_lock(tmp_path):
    """A direct `run.sh <target>` hands the load to the swap endpoint, whose run.sh child takes run.lock: the direct run
    must not hold that lock while it waits, or the child waits behind it for the whole swap."""

    import fcntl
    import os
    import subprocess
    import sys

    state = tmp_path / "state" / "tensorfold-run"
    state.mkdir(parents=True)
    fake_swap = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "swap.py"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "docker").chmod(0o755)
    endpoint = _serve(_SwapEndpoint)
    try:
        (state / "swap.pid").write_text(str(fake_swap.pid))
        with open(state / "run.lock", "w") as held:      # the endpoint's child, mid-load
            fcntl.flock(held, fcntl.LOCK_EX)
            env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / "state"), PORT=str(endpoint.server_address[1]),
                       PATH=f"{bin_dir}:{os.environ['PATH']}")
            env.pop("TF_SWAP_CHILD", None)
            done = subprocess.run(["bash", str(ROOT / "run.sh"), "qwen-flash-next"], env=env, capture_output=True,
                                  text=True, timeout=20)
        assert done.returncode == 0, done.stderr
        assert "asking the swap endpoint" in done.stdout
    finally:
        fake_swap.kill()
        endpoint.shutdown()


def test_a_start_whose_swap_endpoint_ends_after_the_lock_check_still_waits_for_the_lock(tmp_path):
    """The lock decision and ask_swap each ask whether the endpoint is alive. When it ends between the two, the
    direct load that follows must take run.lock, or two loads run side by side. A grep shim answers yes for the
    endpoint's /proc cmdline once, then no; the test holds the lock, so the load must wait and run no docker."""

    import fcntl
    import os
    import subprocess
    import sys

    state = tmp_path / "state" / "tensorfold-run"
    state.mkdir(parents=True)
    fake_swap = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "swap.py"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ran = tmp_path / "docker-ran"
    (bin_dir / "docker").write_text(f"#!/bin/sh\ntouch {ran}\nexit 0\n")
    asked = tmp_path / "asked"
    real_grep = subprocess.run(["sh", "-c", "command -v grep"], capture_output=True, text=True).stdout.strip()
    (bin_dir / "grep").write_text(
        "#!/bin/sh\n"
        'case "$*" in *swap.py*/proc/*)\n'
        f'    [ -e {asked} ] && exit 1\n'
        f'    touch {asked}; exit 0 ;;\n'
        "esac\n"
        f'exec {real_grep} "$@"\n')
    for tool in ("docker", "grep"):
        (bin_dir / tool).chmod(0o755)
    try:
        (state / "swap.pid").write_text(str(fake_swap.pid))
        with open(state / "run.lock", "w") as held:      # another start, mid-load
            fcntl.flock(held, fcntl.LOCK_EX)
            env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / "state"), PATH=f"{bin_dir}:{os.environ['PATH']}")
            env.pop("TF_SWAP_CHILD", None)
            with pytest.raises(subprocess.TimeoutExpired):
                subprocess.run(["bash", str(ROOT / "run.sh"), "qwen-flash-next"], env=env, capture_output=True,
                               text=True, timeout=5)
        assert asked.exists()
        assert not ran.exists()
    finally:
        fake_swap.kill()


def _states(swapper):
    return {m["id"]: (m["state"], m.get("error")) for m in swapper.models()}


def test_models_report_a_failed_load_until_the_next_load_succeeds(swap, tmp_path):
    swap.RUN = str(tmp_path / "missing-run.sh")
    swapper = swap.Swapper()
    with pytest.raises(RuntimeError):
        swapper.enter("glm-5.3-flash")
    state, error = _states(swapper)["glm-5.3-flash"]
    assert state == "failed" and "run.sh did not start" in error
    assert _states(swapper)["qwen3.8-flash-next"] == ("unloaded", None)

    log, backend = _run_logger(swap, tmp_path)
    try:
        assert swapper.enter("glm-5.3-flash") == "glm-5.3-flash"
    finally:
        backend.shutdown()
    assert _states(swapper)["glm-5.3-flash"] == ("ready", None)


def test_models_report_the_model_being_loaded_and_the_one_it_replaces(swap, tmp_path):
    run = tmp_path / "run.sh"
    run.write_text("#!/bin/sh\nsleep 0.5\n")
    run.chmod(0o755)
    swap.RUN = str(run)
    swapper = swap.Swapper()
    swapper.current = "qwen3.8-flash-next"
    load = threading.Thread(target=swapper.enter, args=("glm-5.3-flash",))
    load.start()
    deadline = time.monotonic() + 5
    while swapper.loading is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _states(swapper)["glm-5.3-flash"] == ("loading", None)
    assert _states(swapper)["qwen3.8-flash-next"] == ("unloading", None)    # new requests for it wait for the swap
    load.join(5)
    assert _states(swapper)["glm-5.3-flash"] == ("ready", None)
    assert _states(swapper)["qwen3.8-flash-next"] == ("unloaded", None)


def test_the_loaded_model_names_the_backend_it_forwards_to(swap):
    swapper = swap.Swapper()
    swap.SWAP = swapper
    handler = swap.Handler.__new__(swap.Handler)                # state() reads only SWAP
    assert handler.state()["backend"] is None                   # nothing loaded: nothing to forward to
    assert all(m["backend"] is None for m in swapper.models())
    swapper.current = "glm-5.3-flash"
    address = f"127.0.0.1:{swap.BACKEND}"
    assert handler.state() == {"loaded": "glm-5.3-flash", "loading": None, "running": 0, "backend": address}
    backends = {m["id"]: m["backend"] for m in swapper.models()}
    assert backends.pop("glm-5.3-flash") == address
    assert set(backends.values()) == {None}
