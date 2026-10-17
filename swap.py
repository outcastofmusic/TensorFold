#!/usr/bin/env python3
"""One endpoint for both models on a DGX Spark: `./run.sh swap` starts this on PORT and it loads what a request names.

A request whose "model" is not the loaded one waits for the running requests to finish, has run.sh load the other
model on BACKEND_PORT, and is then forwarded. A streaming request gets an SSE comment every 5 s while it waits, so
clients that time out on a silent connection stay connected. Python's standard library only.

    GET  /v1/models      both models, "loaded" on the one in memory
    GET  /swap           {"loaded": ..., "loading": ..., "running": n}
    POST /swap/<model>   load a model now and answer when it is ready

Environment: PORT=30000, BACKEND_PORT=30002, SWAP_DEFAULT=qwen3.8-flash-next (for a request that names neither),
SWAP_DRAIN=600 (seconds a swap waits for running requests before it cuts them off), SWAP_HOLD=0 (seconds a model
stays loaded before a request for the other one may replace it: with two clients on two models, each then gets a turn
of that length where otherwise every request would pay a swap), SWAP_TIMEOUT=3600 (seconds the backend may stay
silent before a request is given up; a non-streamed reply is silent until it is whole), SWAP_PERSIST=120 (seconds the
outgoing model may take to write its kept prompt states to disk before run.sh stops it).
"""

from __future__ import annotations

import http.client
import json
import os
import select
import socket
import subprocess
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TARGET = {"qwen3.8-flash-next": "qwen-flash-next", "glm-5.3-flash": "glm-5.3-flash"}    # served name: run.sh target
ALIAS = {target: name for name, target in TARGET.items()}
PORT = int(os.environ.get("PORT", "30000"))
BACKEND = int(os.environ.get("BACKEND_PORT", "30002"))
DEFAULT = os.environ.get("SWAP_DEFAULT", "qwen3.8-flash-next")
DRAIN = float(os.environ.get("SWAP_DRAIN", "600"))
HOLD = float(os.environ.get("SWAP_HOLD", "0"))
TIMEOUT = float(os.environ.get("SWAP_TIMEOUT", "3600"))
PERSIST = float(os.environ.get("SWAP_PERSIST", "120"))
MAX_BODY = 64 << 20                     # bytes: a request with images inlined fits
RUN = str(Path(__file__).resolve().parent / "run.sh")
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "te", "upgrade", "proxy-connection"}


def say(text: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} swap: {text}", flush=True)


def loaded() -> str | None:
    """The model the backend port serves now, or None."""

    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=3)
        conn.request("GET", "/v1/models")
        name = json.loads(conn.getresponse().read())["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError, http.client.HTTPException):
        return None
    return name if name in TARGET else None


def persist(model: str) -> None:
    """Have the loaded model write its kept prompt states to disk (``POST /internal/persist``) before run.sh stops it,
    waiting up to PERSIST seconds; a model without disk states answers at once, and a failure only costs the states."""

    start = time.monotonic()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=PERSIST)
        conn.request("POST", "/internal/persist", b"", {"Content-Length": "0"})
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
    except (OSError, http.client.HTTPException) as exc:
        return say(f"{model} did not write its kept states to disk: {exc}")
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
        now = want if code == 0 else loaded()        # an HTTP probe, kept out of the lock
        with self.cond:
            self.current = now
            self.since = time.monotonic()
            self.loading = None
            self.cond.notify_all()
            if code != 0:
                raise RuntimeError(f"{want} did not load ({why}); see the swap log")
            self.running += 1
            return want

    def leave(self) -> None:
        with self.cond:
            self.running -= 1
            self.cond.notify_all()


SWAP: Swapper


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:        # the servers behind log the requests
        pass

    def reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def state(self) -> dict:
        return {"loaded": SWAP.current, "loading": SWAP.loading, "running": SWAP.running}

    def do_GET(self) -> None:
        self.close_connection = True
        path = self.path.split("?")[0].rstrip("/")
        if path == "/v1/models":
            self.reply(200, {"object": "list", "data": [
                {"id": name, "object": "model", "owned_by": "tensorfold", "loaded": name == SWAP.current} for name in TARGET]})
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
