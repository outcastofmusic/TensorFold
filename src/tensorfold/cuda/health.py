"""The CUDA server's /health counters: finished requests' own engine stats, plus live replies read off the rounds."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

from tensorfold.server import metrics

STATS = {"prefill_s": "prefill_seconds_total", "decode_s": "decode_seconds_total", "cached": "cached_tokens_total",
         "rounds": "rounds_total", "drafted": "drafted_total", "accepted": "accepted_total"}
_MADE = threading.Lock()

# CUDA's asynchronous execution faults: the context is lost and every later call fails until the process restarts
# (cudaError 214, 700, 702, 709, 710, 714-719). Torch and Triton both raise them with the driver's text.
STICKY = {214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719}
STICKY_TEXT = ("uncorrectable ecc error", "illegal memory access", "launch timed out", "device-side assert",
               "hardware stack error", "illegal instruction", "misaligned address", "invalid address space",
               "invalid program counter", "unspecified launch failure", "exception occurred on the device")


def sticky(exc: BaseException) -> bool:
    """A CUDA fault the context cannot recover from, by torch's error code or the driver's message."""

    if not isinstance(exc, RuntimeError):          # torch.AcceleratorError and Triton's launch errors; never a request's
        return False
    if getattr(exc, "error_code", None) in STICKY:
        return True
    text = str(exc).lower()
    return "cuda" in text and any(phrase in text for phrase in STICKY_TEXT)


class Request:
    """One running request: its prompt length and the server's own list of its reply tokens (only ever read here)."""

    def __init__(self, prompt: int, out: list[int], arrived: float | None = None) -> None:
        self.prompt, self.out, self.stats = prompt, out, None
        self.started = time.perf_counter() if arrived is None else float(arrived)
        self.first: float | None = None

    def saw(self) -> None:
        """The first generated token has landed in ``out``."""

        if self.first is None and self.out:
            self.first = time.perf_counter()


class Health:
    """Totals of finished requests and the requests running now; the engine's rounds never call in here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set[Request] = set()
        self.fault: str | None = None           # the first sticky CUDA fault: /health answers 503 from then on
        self.totals: dict[str, float] = dict.fromkeys(("requests_total", "prompt_tokens_total",
                                                       "completion_tokens_total", *STATS.values()), 0)

    @contextmanager
    def running(self, prompt: int, out: list[int], arrived: float | None = None):
        """Count a request as running while its ``generate`` runs, then fold its reply and ``stats`` into the totals."""

        request = Request(prompt, out, arrived)
        with self.lock:
            self.live.add(request)
        try:
            yield request
        except BaseException as exc:
            self.note(exc)
            raise
        finally:
            with self.lock:
                self.live.discard(request)
                self._fold(request)
            self._metrics(request)

    def note(self, exc: BaseException) -> None:
        """Keep ``exc`` as the engine's fault when CUDA's context is lost with it."""

        if self.fault is None and sticky(exc):
            self.fault = f"{type(exc).__name__}: {exc}"[:500]
            print(f"[tensorfold] CUDA context lost, /health answers 503 until a restart: {self.fault}", flush=True)

    def _fold(self, request: Request) -> None:
        t = self.totals
        t["requests_total"] += 1
        t["prompt_tokens_total"] += request.prompt
        t["completion_tokens_total"] += len(request.out)
        for key, name in STATS.items():
            value = (request.stats or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                t[name] += value

    def _metrics(self, request: Request) -> None:
        stats = request.stats or {}
        ended = time.perf_counter()
        # The engine's own decode time when it reports one (the same figure /health sums); else first token to end.
        decode = stats.get("decode_s")
        if not isinstance(decode, (int, float)) or isinstance(decode, bool):
            decode = (ended - request.first) if request.first is not None else None
        metrics.note(getattr(self, "app", None), prompt=request.prompt, generation=len(request.out),
                     drafted=_stat(stats, "drafted"), accepted=_stat(stats, "accepted"),
                     cached=_stat(stats, "cached") if "cached" in stats else None,   # None: not reported
                     drafters=stats.get("drafters") if isinstance(stats.get("drafters"), dict) else None,
                     latency=max(0.0, ended - request.started),
                     ttft=(request.first - request.started) if request.first is not None else None,
                     decode=None if decode is None else max(0.0, float(decode)))

    def snapshot(self, app) -> dict[str, Any]:
        """The counters now: finished totals, live replies' tokens so far, and a concurrent engine's streams."""

        with self.lock:
            body: dict[str, Any] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.totals.items()}
            body["completion_tokens_total"] += sum(len(r.out) for r in self.live)
            running = len(self.live)
        body = {"ok": self.fault is None, **({"fault": self.fault} if self.fault else {}), "backend": "tensorfold", "busy": running > 0, "requests_running": running, **body}
        scheduler = getattr(getattr(app, "engine", None), "scheduler", None)     # /health answers whatever the app
        decoder = getattr(scheduler, "decoder", None)
        if decoder is not None:                         # read, never locked: sizes of the decoder's own tables
            body["streams"] = {"decoding": len(getattr(decoder, "streams", ())),
                               "prefilling": len(getattr(decoder, "filling", ())), "max": scheduler.max_streams}
        window = getattr(app, "effective_context_window", None)
        if window:
            body["context_length"] = int(window)
        return body


def of(app) -> Health:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("health")
        if found is None:
            found = app.__dict__["health"] = Health()
        found.app = app
        return found


def _stat(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


__all__ = ["Health", "Request", "of", "sticky"]
