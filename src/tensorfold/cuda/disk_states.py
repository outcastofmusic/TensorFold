"""Kept prompt states on disk, so a conversation's next turn after a model swap, a restart or an eviction reads its
state back instead of prefilling the whole conversation again (``TENSORFOLD_GLM_STATES``, ``TENSORFOLD_QWEN_STATES``:
a folder each).

A state is its small tensors (GLM: KDA states, conv windows, pending MTP rows, the drafter's window; Qwen: DeltaNet
states, conv windows, n-gram windows, the MTP tail) and its attention rows: each view's first rows, along dimension
0, in blocks of ``BLOCK`` rows. Every piece is a file named by the SHA-256 of its bytes, so blocks two states share
are stored once and every read is checked against its name. A later state of the same rows names the full blocks an
earlier one wrote (``blocks``, ``write``'s ``known``) instead of copying and hashing them again. A state's JSON is
written last, by rename, so only whole states are found. The folder holds one subfolder per fingerprint (model, drafter, tensorfold source, row shapes): a
state never loads into another model or build.

Pruning runs at start and after every write no request interrupts. A state unused for ``ttl`` seconds goes, and so
does every other fingerprint's folder (an older build or checkpoint) whose newest state is that old. ``budget`` caps
the whole folder: past it, other fingerprints' folders go least recently used first, then this one's states, then
the pieces no state names. A folder used within ``KEEP_UNNAMED_S`` stays whatever its size, as another server may
still be using it. Other folders are scanned at start and then at most every ``SCAN_S``; this folder's piece sizes
are read once, at start.

Upstream TensorFold 1.0.12 keeps conversations on disk the same way (``--learn-conversations``): a 39,228-token next
turn on a GB10 started in 0.25 s after a restart instead of 6.2 s.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

BLOCK = 256                      # rows of a view a piece holds
MIN_TOKENS = 1024                # a shorter state prefills faster than its files are found and read
KEEP_UNNAMED_S = 3600            # how long a piece no whole state names stays, for the next try of its write
TTL_H = 72                       # hours a state stays unused before it goes (TENSORFOLD_<family>_STATES_TTL_H)
PIECE_BYTES = 16 << 20           # a small tensor's piece: a request waits for at most one to hash and write
SCAN_S = 600                     # seconds between scans of the other fingerprints' folders
READERS = 8                      # threads a read's pieces are read and hashed on
_KEY = re.compile(r"[0-9a-f]{16}")  # a fingerprint folder's name: nothing else under the root is pruned
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


def blocks(entry: Entry, rows: Sequence[int] | None = None) -> list[list[str]]:
    """Each view's full blocks' piece names, below ``rows[i]`` rows of view i (default: all of them). A later state
    passes them to ``write`` as ``known`` while its views are these same rows, unchanged since: ``rows`` leaves out a
    row a later token changes (a partial pool)."""

    specs = entry.meta["views"]
    rows = [s["shape"][0] for s in specs] if rows is None else rows
    return [s["pieces"][:min(r, s["shape"][0]) // BLOCK] for s, r in zip(specs, rows)]


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise StateError(f"a state names dtype {name}")
    return dtype


@functools.cache
def _readers() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(READERS, thread_name_prefix="tensorfold-states")


def _key(ids: Sequence[int]) -> str:
    """A state's name: the SHA-256 of its ids as JSON, which ``_keys`` reads at every length of a prompt at once."""

    return hashlib.sha256(json.dumps(list(map(int, ids))).encode()).hexdigest()[:24]


def _keys(prompt: Sequence[int], lengths: Iterable[int]) -> dict[int, str]:
    """``_key(prompt[:n])`` for each n in ``lengths``, in one pass over the prompt (json.dumps joins with ", ")."""

    h, at, out = hashlib.sha256(b"["), 0, {}
    for n in sorted(set(lengths)):
        if n > at:
            h.update(((", " if at else "") + ", ".join(map(str, map(int, prompt[at:n])))).encode())
            at = n
        end = h.copy()
        end.update(b"]")
        out[n] = end.hexdigest()[:24]
    return out


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
    without it. ``TENSORFOLD_<family>_STATES_GIB`` (64) caps the whole folder, every fingerprint's states together,
    and ``TENSORFOLD_<family>_STATES_TTL_H`` (72) is how long a state stays unused."""

    if not configured(family):
        return None
    root = os.environ[f"TENSORFOLD_{family}_STATES"].strip()
    budget = int(float(os.environ.get(f"TENSORFOLD_{family}_STATES_GIB", "64")) * (1 << 30))
    hours = os.environ.get(f"TENSORFOLD_{family}_STATES_TTL_H", str(TTL_H))
    ttl = float(hours) * 3600
    if not ttl > 0:
        raise ValueError(f"TENSORFOLD_{family}_STATES_TTL_H: the hours a state stays unused, above 0, not {hours!r}")
    disk = DiskStates(Path(root), fingerprint(parts), budget, ttl)
    print(f"[tensorfold] kept states on disk: {disk.dir} ({len(disk.entries)} states, up to {budget >> 30} GiB, "
          f"{ttl / 3600:g} h unused)", flush=True)
    return disk


class DiskStates:
    def __init__(self, root: Path, key: str, budget: int, ttl: float = TTL_H * 3600) -> None:
        self.root, self.key, self.ttl = Path(root), key, float(ttl)
        self.dir = Path(root) / key
        self.pieces, self.states = self.dir / "pieces", self.dir / "states"
        self.pieces.mkdir(parents=True, exist_ok=True)
        self.states.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget)
        self.untouched = False           # a state's use could not be recorded (said once)
        self.entries: list[Entry] = []
        self.index: dict[str, Entry] = {}               # _key(ids), its JSON's name -> the entry
        self.sizes: dict[str, int] = {}                 # every piece of this folder -> its bytes
        self.seen: list[tuple[float, int, Path]] | None = None     # _others() as last scanned
        self.seen_at = 0.0                              # when, by time.monotonic()
        young = time.time() - KEEP_UNNAMED_S
        for path in [*self.pieces.iterdir(), *self.states.glob("*.tmp*")]:
            self._found(path, young)
        for path in sorted(self.states.glob("*.json")):
            try:
                self._add(self._entry(path, json.loads(path.read_text())))
            except (ValueError, KeyError, OSError):      # half a file from a kill mid-rename never gets here; this does
                path.unlink(missing_ok=True)
        self._trim()

    def _found(self, path: Path, young: float) -> None:
        """At start: a piece's size, or a killed write's ``.tmp`` file, gone once ``KEEP_UNNAMED_S`` old."""

        try:
            stat = path.stat()
            if ".tmp" not in path.name:
                self.sizes[path.name] = stat.st_size
            elif stat.st_mtime < young:
                path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _entry(path: Path, meta: dict[str, Any]) -> Entry:
        if meta.get("version") != VERSION:
            raise ValueError("another version's state")
        return Entry(path, [int(t) for t in meta["ids"]], int(meta["mtp_len"]), int(meta["drafter_end"]),
                     int(meta.get("born", -1)), meta)

    def _add(self, entry: Entry) -> None:
        old = self.index.pop(_key(entry.ids), None)
        self.entries = [e for e in self.entries if e is not old and e.path != entry.path] + [entry]
        self.index[_key(entry.ids)] = entry

    # -- finding ----------------------------------------------------------------------------------------------
    def find(self, prompt: Sequence[int], longer_than: int, fits: Callable[[Entry], bool]) -> Entry | None:
        """The longest state of a strict prefix of ``prompt`` past ``longer_than`` tokens that ``fits``: the prompt
        hashed once, its prefixes looked up at each length a state has."""

        lengths = {len(e.ids) for e in self.entries if longer_than < len(e.ids) < len(prompt)}
        keys = _keys(prompt, lengths)
        for n in sorted(lengths, reverse=True):
            e = self.index.get(keys[n])
            if e is not None and fits(e) and e.ids == list(prompt[:n]):
                return e
        return None

    def get(self, ids: Sequence[int]) -> Entry | None:
        e = self.index.get(_key(ids))
        return e if e is not None and e.ids == list(map(int, ids)) else None

    def has(self, ids: Sequence[int]) -> bool:
        return self.get(ids) is not None

    # -- writing ----------------------------------------------------------------------------------------------
    def _put(self, data) -> str:
        name = hashlib.sha256(data).hexdigest()
        path = self.pieces / name
        if not path.exists():
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            tmp.replace(path)
        self.sizes[name] = len(data)
        return name

    def _known(self, names: Sequence[str], k: int, size: int) -> str | None:
        """``names[k]``, a full block's piece ``known`` gives, while its file is on disk at ``size`` bytes."""

        if k >= len(names):
            return None
        try:
            if (self.pieces / names[k]).stat().st_size != size:
                return None
        except OSError:                                 # pruned since: the block is written again
            return None
        self.sizes[names[k]] = size
        return names[k]

    def _tensor(self, t: torch.Tensor, stop: Callable[[], bool]) -> dict[str, Any] | None:
        """A small tensor's spec, its bytes in pieces of ``PIECE_BYTES`` with ``stop()`` checked before each (Qwen's
        DeltaNet states are 115.7 MB); None once it holds."""

        if stop():                                      # before the copy off the GPU too
            return None
        data, names = memoryview(_bytes(t)), []
        for at in range(0, max(len(data), 1), PIECE_BYTES):
            if at and stop():
                return None
            names.append(self._put(data[at:at + PIECE_BYTES]))
        return {"dtype": str(t.dtype), "shape": list(t.shape), "pieces": names}

    def write(self, ids: Sequence[int], views: Callable[[], Sequence[torch.Tensor] | None],
              extras: dict[str, torch.Tensor | None], ring: Sequence[torch.Tensor] | None, *, mtp_len: int,
              drafter_end: int, born: int, stop: Callable[[], bool], known: Sequence[Sequence[str]] = ()) -> bool:
        """Write a state, a block of every view at a time, then its small tensors a piece at a time; ``views()``
        gives the rows where they are now, or None once they are gone. False when ``stop()`` held before a block or
        a piece, or the rows went: the pieces written stay, so the next try writes only the rest. Pruning waits
        while ``stop()`` holds.

        ``known[i]`` names view i's first full blocks as an earlier state of these same rows wrote them (``blocks``):
        such a block is neither copied nor hashed. The caller vouches that the rows are unchanged since; equal ids
        do not do it, as a reply decoded and the same text prefilled give other bits."""

        self.pieces.mkdir(parents=True, exist_ok=True)       # another server may have pruned this folder
        self.states.mkdir(parents=True, exist_ok=True)
        rows = views()
        if rows is None:
            return False
        meta: dict[str, Any] = {"version": VERSION, "ids": list(map(int, ids)), "mtp_len": int(mtp_len),
                                "drafter_end": int(drafter_end), "born": int(born),
                                "views": [{"dtype": str(v.dtype), "shape": list(v.shape), "pieces": []} for v in rows],
                                "extras": {}, "ring": []}
        longest = max((v.shape[0] for v in rows), default=0)
        for start in range(0, longest, BLOCK):
            if stop():
                return False
            rows = views()
            if rows is None or len(rows) != len(meta["views"]):
                return False
            for i, (v, out) in enumerate(zip(rows, meta["views"])):
                if start >= v.shape[0]:
                    continue
                full = start + BLOCK <= v.shape[0]
                name = self._known(known[i], start // BLOCK, BLOCK * v[0].nbytes) if full and i < len(known) else None
                out["pieces"].append(name or self._put(_bytes(v[start:start + BLOCK])))
        for name, t in extras.items():
            if t is not None:
                meta["extras"][name] = self._tensor(t, stop)
        meta["ring"] = [self._tensor(t, stop) for t in ring or ()]
        if None in [*meta["extras"].values(), *meta["ring"]]:
            return False
        meta["used"] = time.time()
        path = self.states / f"{_key(meta['ids'])}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(path)
        self._add(self._entry(path, meta))
        if not stop():
            self._trim()
        return True

    @staticmethod
    def _names(meta: dict[str, Any]) -> list[str]:
        names = [p for v in meta["views"] for p in v["pieces"]]
        names += [p for t in meta["extras"].values() for p in t["pieces"]]
        return names + [p for t in meta["ring"] for p in t["pieces"]]

    @staticmethod
    def _folder(path: Path) -> tuple[float, int]:
        """A fingerprint folder's last use (its newest JSON, else the folder's own time) and its bytes."""

        files = [f for f in path.rglob("*") if f.is_file()]
        used = max((f.stat().st_mtime for f in files if f.suffix == ".json"), default=path.stat().st_mtime)
        return used, sum(f.stat().st_size for f in files)

    def _others(self) -> list[tuple[float, int, Path]]:
        """Every other fingerprint's folder under the root: (last use, bytes, path), least recently used first."""

        out = []
        for path in self.root.iterdir() if self.root.is_dir() else ():
            if path.name == self.key or not _KEY.fullmatch(path.name) or not path.is_dir():
                continue
            with contextlib.suppress(OSError):          # another server pruned it meanwhile
                out.append((*self._folder(path), path))
        return sorted(out)

    def _scanned(self) -> list[tuple[float, int, Path]]:
        """``_others()``, scanned at start and then at most every ``SCAN_S``: walking the folders stats every file,
        and other servers change them slowly."""

        if self.seen is None or time.monotonic() - self.seen_at >= SCAN_S:
            self.seen, self.seen_at = self._others(), time.monotonic()
        return self.seen

    def _trim(self) -> None:
        """Unused states past the TTL go, and other fingerprints' folders past it. Over the budget, other folders go
        least recently used first, then this folder's states; then every piece no state names."""

        now = time.time()
        others = []
        for used, size, path in self._scanned():
            if used < now - self.ttl:
                shutil.rmtree(path, ignore_errors=True)
            else:
                others.append((used, size, path))
        for e in [e for e in self.entries if e.meta.get("used", 0) < now - self.ttl]:
            self.forget(e)
        held = {name for e in self.entries for name in self._names(e.meta)}
        total = sum(self.sizes.get(name, 0) for name in held) + sum(size for _, size, _ in others)
        while others and total > self.budget and others[0][0] <= now - KEEP_UNNAMED_S:
            used, size, path = others[0]
            with contextlib.suppress(OSError):          # the scan may be minutes old: look again before removing
                used, size = self._folder(path)
            if used > now - KEEP_UNNAMED_S:
                break
            shutil.rmtree(path, ignore_errors=True)
            others.pop(0)
            total -= size
        self.seen = others
        for e in sorted(self.entries, key=lambda e: e.meta.get("used", 0)):
            if total <= self.budget or len(self.entries) <= 1:
                break
            self.forget(e)
            keep = {name for x in self.entries for name in self._names(x.meta)}
            total -= sum(self.sizes.get(name, 0) for name in held - keep)
            held = keep
        young = time.time() - KEEP_UNNAMED_S
        for name in [n for n in self.sizes if n not in held]:
            path = self.pieces / name
            try:
                if path.stat().st_mtime < young:          # a stopped write's pieces wait for its next try
                    path.unlink(missing_ok=True)
                    del self.sizes[name]
            except FileNotFoundError:
                del self.sizes[name]
            except OSError:
                pass

    # -- reading ----------------------------------------------------------------------------------------------
    def _read(self, name: str, into: memoryview) -> None:
        """Read piece ``name`` into ``into``, which its size must fill exactly, and check it against its hash."""

        try:
            with open(self.pieces / name, "rb", buffering=0) as f:
                got = 0
                while got < len(into) and (n := f.readinto(into[got:])):
                    got += n
                longer = bool(f.read(1))
        except OSError as exc:
            raise StateError(f"piece {name[:12]} is missing") from exc
        if got != len(into) or longer or hashlib.sha256(into).hexdigest() != name:
            raise StateError(f"piece {name[:12]} does not match its hash")

    def _piece_bytes(self, name: str) -> int:
        if name not in self.sizes:                      # another server's piece, written since this one started
            try:
                self.sizes[name] = (self.pieces / name).stat().st_size
            except OSError as exc:
                raise StateError(f"piece {name[:12]} is missing") from exc
        return self.sizes[name]

    def _gather(self, names: Sequence[str], sizes: Sequence[int], total: int, pin: bool) -> torch.Tensor:
        """The pieces end to end in one buffer (uint8, pinned for a copy to the GPU), read and checked on the reader
        threads (file reads and hashlib let go of the GIL)."""

        if sum(sizes) != total:
            raise StateError(f"the pieces hold {sum(sizes)} bytes where the state names {total}")
        out = torch.empty(total, dtype=torch.uint8, pin_memory=pin)
        mem, at = memoryview(out.numpy()), [0]
        for size in sizes:
            at.append(at[-1] + size)
        list(_readers().map(lambda k: self._read(names[k], mem[at[k]:at[k + 1]]), range(len(names))))
        return out

    def _small(self, spec: dict[str, Any], device: torch.device) -> torch.Tensor:
        dtype, shape = _dtype(spec["dtype"]), spec["shape"]
        total = math.prod(shape) * dtype.itemsize
        buf = self._gather(spec["pieces"], [self._piece_bytes(n) for n in spec["pieces"]], total,
                           device.type == "cuda")
        return buf.view(dtype).reshape(shape).to(device, non_blocking=True)

    def load(self, entry: Entry, views: Sequence[torch.Tensor], device) -> dict[str, Any]:
        """Copy the state's rows into ``views`` (the live caches, sized for it), one copy a view from a buffer its
        pieces are read into, and return its small tensors on ``device``; StateError, with the entry forgotten, when
        a piece fails or the shapes differ, and with the entry kept when the disk or the GPU's memory fails the
        read."""

        meta, device = entry.meta, torch.device(device)
        try:
            if len(views) != len(meta["views"]) or any(list(v.shape) != s["shape"] or str(v.dtype) != s["dtype"]
                                                       for v, s in zip(views, meta["views"])):
                raise StateError("the state's views have other shapes than this engine's")
            for v, spec in zip(views, meta["views"]):
                row = math.prod(v.shape[1:]) * v.element_size()
                sizes = [min(BLOCK, v.shape[0] - k * BLOCK) * row for k in range(len(spec["pieces"]))]
                buf = self._gather(spec["pieces"], sizes, v.numel() * v.element_size(), v.is_cuda)
                v.copy_(buf.view(v.dtype).view(v.shape), non_blocking=True)
            out = {k: self._small(s, device) for k, s in meta["extras"].items()}
            out["ring"] = [self._small(s, device) for s in meta["ring"]] or None
            if device.type == "cuda" or any(v.is_cuda for v in views):
                torch.cuda.current_stream(device if device.type == "cuda" else views[0].device).synchronize()
        except StateError:
            self.forget(entry)
            raise
        except (OSError, torch.cuda.OutOfMemoryError) as exc:    # the disk or the GPU failed, not the state: it stays
            raise StateError(f"{type(exc).__name__}: {exc}") from exc
        self._touch(entry)
        return out

    def _touch(self, entry: Entry) -> None:
        """Record the state's use for pruning, best effort: on a full or read-only disk it keeps its old time."""

        entry.meta["used"] = time.time()
        tmp = entry.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(entry.meta))
            tmp.replace(entry.path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            if not self.untouched:
                print(f"[tensorfold] kept states on disk: a state's last use not recorded ({exc}); reads go on",
                      flush=True)
            self.untouched = True

    def forget(self, entry: Entry) -> None:
        self.entries = [e for e in self.entries if e is not entry]
        if self.index.get(_key(entry.ids)) is entry:
            del self.index[_key(entry.ids)]
        entry.path.unlink(missing_ok=True)
