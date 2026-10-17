"""Kept prompt states on disk, so a conversation's next turn after a model swap, a restart or an eviction reads its
state back instead of prefilling the whole conversation again (``TENSORFOLD_GLM_STATES``, ``TENSORFOLD_QWEN_STATES``:
a folder each).

A state is its small tensors (GLM: KDA states, conv windows, pending MTP rows, the drafter's window; Qwen: DeltaNet
states, conv windows, n-gram windows, the MTP tail) and its attention rows: each view's first rows, along dimension
0, in blocks of ``BLOCK`` rows. Every piece is a file named by the SHA-256 of its bytes, so blocks two states share
are stored once and every read is checked against its name. A state's JSON is written last, by rename, so only whole
states are found. States past ``budget`` bytes go least recently used first, then the pieces no state names. The
folder holds one subfolder per fingerprint (model, drafter, tensorfold source, row shapes): a state never loads into
another model or build.

Upstream TensorFold 1.0.12 keeps conversations on disk the same way (``--learn-conversations``): a 39,228-token next
turn on a GB10 started in 0.25 s after a restart instead of 6.2 s.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

BLOCK = 256                      # rows of a view a piece holds
MIN_TOKENS = 1024                # a shorter state prefills faster than its files are found and read
KEEP_UNNAMED_S = 3600            # how long a piece no whole state names stays, for the next try of its write
VERSION = 1


class StateError(ValueError):
    """A state on disk that cannot load: a piece missing or not matching its hash, or views of another shape."""


@dataclass
class Entry:
    """A state on disk: its prompt ids and what its JSON names."""

    path: Path
    ids: list[int]
    mtp_len: int
    drafter_end: int
    born: int
    meta: dict[str, Any] = field(repr=False)


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise StateError(f"a state names dtype {name}")
    return dtype


def fingerprint(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def digest(paths: Sequence[Path]) -> str:
    """The SHA-256 of the files' names and bytes, in the order given (a missing file adds nothing)."""

    h = hashlib.sha256()
    for path in paths:
        if path.is_file():
            h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()


def configured(family: str) -> bool:
    return bool(os.environ.get(f"TENSORFOLD_{family}_STATES", "").strip())


def from_env(family: str, parts: dict[str, Any]) -> DiskStates | None:
    """``TENSORFOLD_<family>_STATES``: the folder kept states also go to, under the fingerprint of ``parts``; None
    without it. ``TENSORFOLD_<family>_STATES_GIB`` (64) caps what that fingerprint's folder holds."""

    if not configured(family):
        return None
    root = os.environ[f"TENSORFOLD_{family}_STATES"].strip()
    budget = int(float(os.environ.get(f"TENSORFOLD_{family}_STATES_GIB", "64")) * (1 << 30))
    disk = DiskStates(Path(root), fingerprint(parts), budget)
    print(f"[tensorfold] kept states on disk: {disk.dir} ({len(disk.entries)} states, up to {budget >> 30} GiB)",
          flush=True)
    return disk


class DiskStates:
    def __init__(self, root: Path, key: str, budget: int) -> None:
        self.dir = Path(root) / key
        self.pieces, self.states = self.dir / "pieces", self.dir / "states"
        self.pieces.mkdir(parents=True, exist_ok=True)
        self.states.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget)
        self.entries: list[Entry] = []
        for path in sorted(self.states.glob("*.json")):
            try:
                self.entries.append(self._entry(path, json.loads(path.read_text())))
            except (ValueError, KeyError, OSError):      # half a file from a kill mid-rename never gets here; this does
                path.unlink(missing_ok=True)

    @staticmethod
    def _entry(path: Path, meta: dict[str, Any]) -> Entry:
        if meta.get("version") != VERSION:
            raise ValueError("another version's state")
        return Entry(path, [int(t) for t in meta["ids"]], int(meta["mtp_len"]), int(meta["drafter_end"]),
                     int(meta.get("born", -1)), meta)

    # -- finding ----------------------------------------------------------------------------------------------
    def find(self, prompt: Sequence[int], longer_than: int, fits: Callable[[Entry], bool]) -> Entry | None:
        """The longest state of a strict prefix of ``prompt`` past ``longer_than`` tokens that ``fits``."""

        best = None
        for e in self.entries:
            n = len(e.ids)
            if longer_than < n < len(prompt) and (best is None or n > len(best.ids)) and fits(e) \
                    and list(prompt[:n]) == e.ids:
                best = e
        return best

    def has(self, ids: Sequence[int]) -> bool:
        return any(e.ids == list(ids) for e in self.entries)

    # -- writing ----------------------------------------------------------------------------------------------
    def _put(self, data: bytes) -> str:
        name = hashlib.sha256(data).hexdigest()
        path = self.pieces / name
        if not path.exists():
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            tmp.replace(path)
        return name

    def _tensor(self, t: torch.Tensor) -> dict[str, Any]:
        return {"dtype": str(t.dtype), "shape": list(t.shape), "pieces": [self._put(_bytes(t))]}

    def write(self, ids: Sequence[int], views: Callable[[], Sequence[torch.Tensor] | None],
              extras: dict[str, torch.Tensor | None], ring: Sequence[torch.Tensor] | None, *, mtp_len: int,
              drafter_end: int, born: int, stop: Callable[[], bool]) -> bool:
        """Write a state, a block of every view at a time; ``views()`` gives the rows where they are now, or None
        once they are gone. False when ``stop()`` held between blocks or the rows went: the pieces written stay, so
        the next try writes only the rest."""

        rows = views()
        if rows is None:
            return False
        meta: dict[str, Any] = {"version": VERSION, "ids": list(map(int, ids)), "mtp_len": int(mtp_len),
                                "drafter_end": int(drafter_end), "born": int(born),
                                "views": [{"dtype": str(v.dtype), "shape": list(v.shape), "pieces": []} for v in rows],
                                "extras": {k: self._tensor(t) for k, t in extras.items() if t is not None},
                                "ring": [self._tensor(t) for t in ring or ()]}
        longest = max((v.shape[0] for v in rows), default=0)
        for start in range(0, longest, BLOCK):
            if stop():
                return False
            rows = views()
            if rows is None or len(rows) != len(meta["views"]):
                return False
            for v, out in zip(rows, meta["views"]):
                if start < v.shape[0]:
                    out["pieces"].append(self._put(_bytes(v[start:start + BLOCK])))
        meta["bytes"] = sum(self._size(name) for name in self._names(meta))
        meta["used"] = time.time()
        path = self.states / f"{hashlib.sha256(json.dumps(meta['ids']).encode()).hexdigest()[:24]}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(path)
        self.entries = [e for e in self.entries if e.path != path] + [self._entry(path, meta)]
        self._trim()
        return True

    def _size(self, name: str) -> int:
        try:
            return (self.pieces / name).stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _names(meta: dict[str, Any]) -> list[str]:
        names = [p for v in meta["views"] for p in v["pieces"]]
        names += [p for t in meta["extras"].values() for p in t["pieces"]]
        return names + [p for t in meta["ring"] for p in t["pieces"]]

    def _trim(self) -> None:
        """States past the budget go least recently used first, then every piece no state names."""

        held = {name for e in self.entries for name in self._names(e.meta)}
        total = sum(self._size(name) for name in held)
        for e in sorted(self.entries, key=lambda e: e.meta.get("used", 0)):
            if total <= self.budget or len(self.entries) <= 1:
                break
            self.entries.remove(e)
            e.path.unlink(missing_ok=True)
            keep = {name for x in self.entries for name in self._names(x.meta)}
            total -= sum(self._size(name) for name in held - keep)
            held = keep
        young = time.time() - KEEP_UNNAMED_S
        for path in self.pieces.iterdir():
            if path.name in held or ".tmp" in path.name:
                continue
            try:
                if path.stat().st_mtime < young:          # a stopped write's pieces wait for its next try
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    # -- reading ----------------------------------------------------------------------------------------------
    def _read(self, name: str) -> bytes:
        try:
            data = (self.pieces / name).read_bytes()
        except OSError as exc:
            raise StateError(f"piece {name[:12]} is missing") from exc
        if hashlib.sha256(data).hexdigest() != name:
            raise StateError(f"piece {name[:12]} does not match its hash")
        return data

    def _load(self, spec: dict[str, Any], device) -> torch.Tensor:
        data = b"".join(self._read(name) for name in spec["pieces"])
        t = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(_dtype(spec["dtype"])).reshape(spec["shape"])
        return t.to(device)

    def load(self, entry: Entry, views: Sequence[torch.Tensor], device) -> dict[str, Any]:
        """Copy the state's rows into ``views`` (the live caches, sized for it) block by block, and return its small
        tensors on ``device``; StateError, with the entry forgotten, when a piece fails or the shapes differ."""

        meta = entry.meta
        try:
            if len(views) != len(meta["views"]) or any(list(v.shape) != s["shape"] or str(v.dtype) != s["dtype"]
                                                       for v, s in zip(views, meta["views"])):
                raise StateError("the state's views have other shapes than this engine's")
            for v, spec in zip(views, meta["views"]):
                for k, name in enumerate(spec["pieces"]):
                    part = v[k * BLOCK:(k + 1) * BLOCK]
                    data = self._read(name)
                    src = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(v.dtype).reshape(part.shape)
                    part.copy_(src)
            out = {k: self._load(s, device) for k, s in meta["extras"].items()}
            out["ring"] = [self._load(s, device) for s in meta["ring"]] or None
        except StateError:
            self.forget(entry)
            raise
        meta["used"] = time.time()
        entry.path.write_text(json.dumps(meta))
        return out

    def forget(self, entry: Entry) -> None:
        if entry in self.entries:
            self.entries.remove(entry)
        entry.path.unlink(missing_ok=True)
