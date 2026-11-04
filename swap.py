#!/usr/bin/env python3
"""One endpoint for both models on a DGX Spark: `./run.sh swap` starts this on PORT and it loads what a request names.

A request whose "model" is not the loaded one waits for the running requests to finish, has run.sh load the other
model on BACKEND_PORT, and is then forwarded. A streaming request gets an SSE comment every 5 s while it waits, so
clients that time out on a silent connection stay connected. Python's standard library only.

    GET  /v1/models      every model, "loaded" on the one in memory and a "state": ready, loading, unloading,
                         unloaded or failed (with the "error" of its last load or of its engine, kept until a load of it succeeds),
                         and the "backend" the loaded one is forwarded to ("127.0.0.1:30002", null on the others);
                         "max_model_len" (vLLM's name) is the window the model served on its last load, kept in
                         SWAP_WINDOWS across restarts, and missing for a model whose server has never reported one
    GET  /swap           {"loaded": ..., "loading": ..., "running": n, "backend": "127.0.0.1:30002" or null}
    POST /swap/<model>   load a model now and answer when it is ready

Environment: PORT=30000, BACKEND_PORT=30002, SWAP_DEFAULT=qwen3.8-flash-next (for a request that names neither),
SWAP_DRAIN=600 (seconds a swap waits for running requests before it cuts them off), SWAP_HOLD=0 (seconds a model
stays loaded before a request for the other one may replace it: with two clients on two models, each then gets a turn
of that length where otherwise every request would pay a swap), SWAP_TIMEOUT=3600 (seconds the backend may stay
silent before a request is given up; a non-streamed reply is silent until it is whole), SWAP_PERSIST=120 (seconds the
outgoing model may take to write its kept prompt states to disk before run.sh stops it), SWAP_WINDOWS (a JSON file of
each model's last window; run.sh sets it, and without it the windows last until this process ends).
"""

from __future__ import annotations

import http.client
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))   # the checkout's guard, standard library only
from tensorfold.server import request_guard  # noqa: E402

TARGET = {"qwen3.8-flash-next": "qwen-flash-next", "glm-5.3-flash": "glm-5.3-flash",    # served name: run.sh target
          "qwen3.8-flash-next-fp4": "qwen-flash-next-fp4", "qwen3.8-27b": "qwen-27b", "nemotron-3.5-bolt": "nemotron-bolt",
          "lfm2.5-24b-a2b": "lfm2-24b", "nemotron-3.5-super": "nemotron-3.5-super"}
ALIAS = {target: name for name, target in TARGET.items()}
PORT = int(os.environ.get("PORT", "30000"))
BACKEND = int(os.environ.get("BACKEND_PORT", "30002"))
DEFAULT = os.environ.get("SWAP_DEFAULT", "qwen3.8-flash-next")
DRAIN = float(os.environ.get("SWAP_DRAIN", "600"))
HOLD = float(os.environ.get("SWAP_HOLD", "0"))
TIMEOUT = float(os.environ.get("SWAP_TIMEOUT", "3600"))
PERSIST = float(os.environ.get("SWAP_PERSIST", "120"))
WINDOWS = os.environ.get("SWAP_WINDOWS")
MAX_BODY = 64 << 20                     # bytes: a request with images inlined fits
RUN = str(Path(__file__).resolve().parent / "run.sh")
# Origin too: this endpoint checks it, and the server behind listens on another port, so it would refuse it
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "te", "upgrade", "proxy-connection",
       "origin"}


def say(text: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} swap: {text}", flush=True)


def backend() -> str:
    """Where requests for the loaded model go, so a gateway in front can tell which server answered."""

    return f"127.0.0.1:{BACKEND}"


def served() -> list:
    """The backend's /v1/models entries, or [] when nothing answers."""

    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=3)
        conn.request("GET", "/v1/models")
        data = json.loads(conn.getresponse().read())["data"]
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
        return []
    return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []


def fault() -> str | None:
    """Why the backend's engine needs a restart (its /health answers 503 after a lost CUDA context), or None."""

    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read() or b"{}")
    except (OSError, ValueError, http.client.HTTPException):
        return None
    if resp.status != 503:
        return None
    return str(body.get("fault") or body.get("status") or "unhealthy") if isinstance(body, dict) else "unhealthy"


def loaded() -> str | None:
    """The model the backend port serves now, or None."""

    name = next(iter(served()), {}).get("id")
    return name if name in TARGET else None


def window(model: str) -> int | None:
    """The max_model_len the backend gives ``model``, or None (upstream's native server, Super's, gives none)."""

    for entry in served():
        value = entry.get("max_model_len")
        if entry.get("id") == model and isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def saved_windows() -> dict[str, int]:
    """SWAP_WINDOWS' windows; none when the file is missing or unreadable, as each load writes it afresh."""

    try:
        got = json.loads(Path(WINDOWS).read_text()) if WINDOWS else {}
    except (OSError, ValueError):
        return {}
    if not isinstance(got, dict):
        return {}
    return {k: v for k, v in got.items() if k in TARGET and isinstance(v, int) and not isinstance(v, bool)}


def save_windows(windows: dict[str, int]) -> None:
    if not WINDOWS:
        return
    path = Path(WINDOWS)
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(windows, sort_keys=True))
        tmp.replace(path)                       # a reader never sees half a file
    except OSError as exc:                      # the windows still serve from memory
        say(f"could not write {path}: {exc}")


def persist(model: str) -> None:
    """Have the loaded model write its kept prompt states to disk (``POST /internal/persist``) before run.sh stops it,
    waiting up to PERSIST seconds; a model without disk states answers at once, upstream's native server (Super) has
    no such route and answers 404, and a failure only costs the states."""

    start = time.monotonic()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=PERSIST)
        conn.request("POST", "/internal/persist", b"", {"Content-Length": "0"})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
    except (OSError, http.client.HTTPException) as exc:
        return say(f"{model} did not write its kept states to disk: {exc}")
    if resp.status == 404:                      # no such route, as {"disk": false}: nothing to write
        return
    try:
        got = json.loads(body) if resp.status == 200 else None
    except ValueError:
        got = None
    if not isinstance(got, dict):
        return say(f"{model} did not write its kept states to disk: HTTP {resp.status} {body[:200]!r}")
    if got.get("disk"):
        say(f"{model} wrote {got.get('written', 0)} kept states to disk in {time.monotonic() - start:.1f} s "
            f"({got.get('states', 0)} there)")


class Swapper:
    """Which model is in memory, and the one swap at a time that changes it."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.current = loaded()
        self.loading: str | None = None
        self.running = 0
        self.since = 0.0                 # when the loaded model became ready
        self.waiting: Counter[str | None] = Counter()   # requests in enter(), by the model they name
        self.failed: dict[str, str] = {}                # model: why its last load failed
        self.windows = saved_windows()                  # model: the max_model_len it served on its last load
        if self.current is not None:
            self.note_window(self.current, window(self.current))

    def note_window(self, model: str, value: int | None) -> None:
        """Keep ``model``'s window from the load that just finished. None (a probe that timed out, or a server that
        reports none) keeps the last one."""

        if value is None or self.windows.get(model) == value:
            return
        self.windows[model] = value
        save_windows(dict(self.windows))

    def models(self) -> list[dict]:
        """The /v1/models entries, each with the model's state."""

        with self.cond:
            out = []
            for name in TARGET:
                entry = {"id": name, "object": "model", "owned_by": "tensorfold", "loaded": name == self.current,
                         "backend": backend() if name == self.current else None}
                if name == self.loading:
                    entry["state"] = "loading"
                elif name == self.current:              # a swap away from it holds new requests for it
                    entry["state"] = "ready" if self.loading is None else "unloading"
                elif name in self.failed:
                    entry["state"], entry["error"] = "failed", self.failed[name]
                else:
                    entry["state"] = "unloaded"
                if name in self.windows:
                    entry["max_model_len"] = self.windows[name]
                out.append(entry)
            return out

    def ready(self, model: str | None) -> bool:
        with self.cond:
            return self.loading is None and self.current is not None and model in (None, self.current)

    def enter(self, model: str | None) -> str:
        """Count a request in once ``model`` (None: whatever is loaded, or the default) is in memory; RuntimeError when its load fails."""

        with self.cond:
            self.waiting[model] += 1
            try:
                while True:
                    while self.loading is not None:
                        self.cond.wait()
                    want = model or self.current or DEFAULT
                    if want == self.current:
                        self.running += 1
                        return want
                    # requests that waited out a load for the model now in memory go first: one swap serves them all
                    if self.current is None or not self.waiting[self.current] + self.waiting[None]:
                        break
                    self.cond.wait()
            finally:
                self.waiting[model] -= 1
                self.cond.notify_all()
            self.loading = want
            while self.current is not None and (left := self.since + HOLD - time.monotonic()) > 0:
                self.cond.wait(left)
            deadline = time.monotonic() + DRAIN
            while self.running and time.monotonic() < deadline:
                self.cond.wait(1)
            cut, before = self.running, self.current
        say(f"{before} -> {want}" + (f", cutting off {cut} running requests" if cut else ""))
        # requests still running hold the engine, so a persist would wait for them: their states are left behind
        if before is not None and not cut:
            persist(before)
        start = time.monotonic()
        try:
            code = subprocess.run([RUN, TARGET[want]], env={**os.environ, "PORT": str(BACKEND), "TF_SWAP_CHILD": "1"},
                                  stdin=subprocess.DEVNULL, check=False).returncode
            why = f"run.sh exited {code}"
        except OSError as exc:                       # run.sh missing or not executable, or no memory to fork it
            code, why = None, f"run.sh did not start: {exc}"
        say(f"{want} {'ready' if code == 0 else 'FAILED'} after {time.monotonic() - start:.0f} s")
        now = want if code == 0 else loaded()        # HTTP probes, kept out of the lock
        served_window = window(want) if code == 0 else None
        with self.cond:
            self.current = now
            self.since = time.monotonic()
            self.loading = None
            self.cond.notify_all()
            if code != 0:
                self.failed[want] = why
                raise RuntimeError(f"{want} did not load ({why}); see the swap log")
            self.failed.pop(want, None)
            self.note_window(want, served_window)
            self.running += 1
            return want

    def lost(self, why: str) -> None:
        """The loaded model's engine died: the next request for it loads it afresh."""

        with self.cond:
            if self.current is None or self.loading is not None:
                return
            say(f"{self.current} lost its engine ({why[:200]}); the next request reloads it")
            self.failed[self.current] = f"engine fault: {why}"
            self.current = None
            self.cond.notify_all()

    def leave(self) -> None:
        with self.cond:
            self.running -= 1
            self.cond.notify_all()


SWAP: Swapper


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:        # the servers behind log the requests
        pass

    def reply(self, status: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def parse_request(self) -> bool:
        """Refuse a non-loopback Host (DNS rebinding) or a browser Origin other than this endpoint's, with 403."""
        if not super().parse_request():
            return False
        reason = request_guard.blocked(self.headers.get_all("Host") or [], self.headers.get_all("Origin") or [], True,
                                       request_guard.allowed_origins())
        if reason is None:
            return True
        self.close_connection = True
        self.reply(403, {"error": {"message": request_guard.HOST_MESSAGE if reason == "host" else
                                   request_guard.ORIGIN_MESSAGE, "type": "invalid_request_error"}})
        return False

    def state(self) -> dict:
        return {"loaded": SWAP.current, "loading": SWAP.loading, "running": SWAP.running,
                "backend": backend() if SWAP.current is not None else None}

    def do_GET(self) -> None:
        self.close_connection = True
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            self.reply(200, {"object": "list", "data": SWAP.models()})
        elif path == "/swap":
            self.reply(200, self.state())
        elif SWAP.ready(None):
            self.forward(b"", sent=False)
        else:
            self.reply(503, {"error": {"message": "no model is loaded", "type": "unavailable"}})

    def do_POST(self) -> None:
        self.close_connection = True
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return self.reply(411, {"error": {"message": "send the body with a Content-Length"}})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            return self.reply(400 if length < 0 else 413,
                              {"error": {"message": f"Content-Length must be 0 to {MAX_BODY} bytes"}})
        body = self.rfile.read(length)
        if self.path.startswith("/internal/"):          # the model's own routes (persist): this endpoint calls them itself
            return self.reply(404, {"error": {"message": f"unknown path {self.path}"}})
        if self.path.startswith("/swap/"):
            name = self.path[len("/swap/"):].strip("/")
            name = ALIAS.get(name, name)
            if name not in TARGET:
                return self.reply(404, {"error": {"message": f"unknown model {name}: {', '.join(TARGET)}"}})
            try:
                SWAP.enter(name)
            except RuntimeError as exc:
                return self.reply(503, {"error": {"message": str(exc), "type": "unavailable"}})
            SWAP.leave()
            return self.reply(200, self.state())
        try:
            request = json.loads(body) if body else {}
        except ValueError:
            request = {}
        model = request.get("model") if isinstance(request, dict) else None
        model = ALIAS.get(model, model)
        model = model if model in TARGET else None
        sent = False
        ready = SWAP.ready(model)
        if not ready:
            say(f"{self.path} for {model or 'no named model'} from {self.headers.get('User-Agent', 'an unnamed client')} waits for a load")
        if isinstance(request, dict) and request.get("stream") and not ready:
            sent = self.hold(model)
            if sent is None:
                return
        else:
            try:
                SWAP.enter(model)
            except RuntimeError as exc:
                return self.reply(503, {"error": {"message": str(exc), "type": "unavailable"}})
        try:
            self.forward(body, sent=sent)
        finally:
            SWAP.leave()

    def hold(self, model: str | None) -> bool | None:
        """Open the event stream and send a comment every 5 s until the model is in memory; None when the request ended here."""

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        done, out = threading.Event(), {}

        def work() -> None:
            try:
                SWAP.enter(model)
            except RuntimeError as exc:
                out["error"] = str(exc)
            done.set()

        threading.Thread(target=work, daemon=True).start()
        gone = False
        while not done.wait(5):
            if not gone:
                try:
                    self.wfile.write(b": loading the model\n\n")
                    self.wfile.flush()
                except OSError:
                    gone = True          # the client left: the load it started still finishes
        if "error" in out:
            if not gone:
                self.event({"error": {"message": out["error"], "type": "unavailable"}})
            return None
        if gone:
            SWAP.leave()
            return None
        return True

    def event(self, payload: dict) -> None:
        try:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()
        except OSError:
            pass

    def forward(self, body: bytes, *, sent: bool) -> None:
        """Relay this request to the backend; ``sent``: the event stream's headers already went out."""

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=TIMEOUT)
        started = sent
        answered = threading.Event()
        try:
            conn.request(self.command, self.path, body or None, headers)
            threading.Thread(target=self.watch, args=(conn, answered), daemon=True).start()
            resp = conn.getresponse()
            answered.set()
            if resp.status in (500, 503) and (why := fault()) is not None:
                SWAP.lost(why)          # a dead CUDA context: the client may retry, and its retry reloads the model
                payload = {"error": {"message": f"the model's engine failed and reloads on the next request: {why}",
                                     "type": "unavailable"}}
                return self.event(payload) if sent else self.reply(503, payload, {"Retry-After": "1"})
            if sent and resp.status != 200:
                return self.event({"error": {"message": resp.read().decode(errors="replace"), "code": resp.status}})
            if not sent:
                started = True
                self.send_response(resp.status)
                for key, value in resp.getheaders():
                    if key.lower() not in HOP:
                        self.send_header(key, value)
                self.send_header("Connection", "close")      # the body ends where the connection does
                self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            if not started:
                try:
                    self.reply(502, {"error": {"message": f"the model server did not answer: {exc}", "type": "unavailable"}})
                except OSError:
                    pass
        finally:
            answered.set()
            conn.close()                                     # a client that left ends the generation behind it

    def watch(self, conn: http.client.HTTPConnection, answered: threading.Event) -> None:
        """Until the backend answers, close its connection once the client hangs up, so the generation ends with it."""

        while not answered.is_set():
            readable, _, _ = select.select([self.connection], [], [], 1)
            if not readable or answered.is_set():
                continue
            try:
                left = self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                left = True
            if not left:                                     # bytes after the body: not ours to read, so just wait
                answered.wait(1)
                continue
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return


def main() -> None:
    global SWAP
    SWAP = Swapper()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    if os.environ.get("SWAP_PIDFILE"):
        Path(os.environ["SWAP_PIDFILE"]).write_text(str(os.getpid()))
    say(f"listening on http://127.0.0.1:{PORT}/v1, models on port {BACKEND}, loaded: {SWAP.current}")
    server.serve_forever()


if __name__ == "__main__":
    main()
