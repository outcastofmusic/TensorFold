"""Flash Next's kept prompt states on disk (``TENSORFOLD_QWEN_STATES``, ``tensorfold.cuda.disk_states``), for the
serial engine and the concurrent decoder alike: a kept entry is (ids, slot, snapshot, tail), as ``prefixes`` keeps it.

A resume (``prefill_begin``) restores the snapshot (DeltaNet states at the point, conv and n-gram windows, the n-gram
history, ``pos`` and ``mtp_len``), absorbs the tail into the MTP head, and reads the slot's attention rows below
``pos``, the MTP head's below ``mtp_len``, and the complete index pools of each (a partial pool is pooled again when
its block completes). Those rows are what a state writes and reads back; nothing else of the slot is read.

A kept snapshot written or read back holds its full blocks' piece names (``BLOCKS``). Every kept entry's rows stay in
its slot unchanged, so a later state of another entry on the slot that one extends, or that extends it, names those
blocks instead of copying and hashing them again.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorfold.cuda import disk_states
from tensorfold.env_flags import flag

# the environment the prompt kernels read: a state written under other values may not be the bits this run computes
KERNEL_FLAGS = ("TENSORFOLD_PREFILL_ROWS", "TENSORFOLD_LONG_PIECE_TOKENS", "TF_QMMF_FUSED_ROWS", "TF_FLASH_STAGE_AHEAD")
BLOCKS = "disk_blocks"           # a kept snapshot's key: its rows' full blocks on disk (disk_states.blocks)


def rows(st, n: int, m: int) -> list[torch.Tensor]:
    """The slot's rows a state of ``n`` positions and ``m`` MTP rows resumes from, as views into its caches."""

    def cache(kc, ikc, pooled, count):
        out = [kc.k[:count], kc.v[:count]]
        if kc.quantized:
            out += [kc.ks[:count], kc.vs[:count]]
        return out + [ikc[:count], pooled[:count // st.ratio]]

    out = [v for i in range(len(st.kc)) for v in cache(st.kc[i], st.ikc[i], st.pooled[i], n)]
    if hasattr(st, "mtp_kc"):
        out += cache(st.mtp_kc, st.mtp_ikc, st.mtp_pooled, m)
    return out


def _extras(snap: dict, tail: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
    history = snap["ple_history"]
    return {"rec": snap["rec"], "conv": snap["conv"], "ple_tail": snap["ple_tail"], "tail": tail,
            "ple_history": None if history is None else torch.from_numpy(np.ascontiguousarray(history))}


def _code() -> list[Path]:
    """The source a state's bits depend on: this family's and the shared CUDA package's Python and kernels."""

    here = Path(__file__).resolve().parent
    return sorted(f for d in (here.parent, here.parents[2] / "cuda") for f in d.rglob("*")
                  if f.suffix in (".py", ".cu", ".cuh", ".cpp", ".h") and f.is_file())


def _checkpoint(folder: Path) -> list:
    """The checkpoint's identity: its folder, the bytes of its configs and index, each shard's name, size and time."""

    index = sorted(folder.glob("*.safetensors.index.json"))
    shards = [(f.name, s.st_size, s.st_mtime_ns) for f in sorted(folder.glob("*.safetensors")) for s in [f.stat()]]
    return [str(folder.resolve()),
            disk_states.digest([folder / "config.json", folder / "quantization_config.json", *index]), shards]


def _toolchain() -> dict[str, Any]:
    """The libraries and GPU the kernels' bits come from."""

    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton_version,
            "gpu": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}


def open_states(model_dir: Path, st, *, engine: str, prefill_rows: int) -> disk_states.DiskStates | None:
    """The folder ``TENSORFOLD_QWEN_STATES`` names, under a fingerprint of everything a state's bits depend on: the
    checkpoint and its shards, the source, torch, CUDA, Triton and the GPU, the row shapes and dtypes, the engine and
    its prompt settings. None when unset."""

    if not disk_states.configured("QWEN"):
        return None
    import tensorfold
    from tensorfold.cuda import prompt_precision

    flags = {k: os.environ.get(k) for k in KERNEL_FLAGS}
    parts = {"tensorfold": tensorfold.__version__, "code": disk_states.digest(_code()),
             "checkpoint": _checkpoint(Path(model_dir)), "toolchain": _toolchain(),
             "views": [(list(v.shape[1:]), str(v.dtype)) for v in rows(st, 4, 4)],
             "states": [list(st.rec.shape[2:]), str(st.rec.dtype), list(st.conv.shape), str(st.conv.dtype),
                        list(st.ple_tail.shape), st.ple_history is not None],
             "engine": engine, "prefill_rows": int(prefill_rows), "fp8": bool(prompt_precision.fp8()), "flags": flags}
    return disk_states.from_env("QWEN", parts)


def _writable(ids, snap) -> bool:
    return len(ids) >= disk_states.MIN_TOKENS and snap["pos"] == len(ids)


def _extended(entry: tuple, kept: Sequence[tuple]) -> bool:
    """Whether a restart has no use for ``entry``: a longer writable entry on its slot extends it, and it is not the
    slot's shortest, the system block's end that a new conversation resumes from (``prefixes._superseded``)."""

    ids, st = entry[0], entry[1]
    chain = [k for k in kept if k[1] is st]
    if len(ids) <= min(len(k[0]) for k in chain):
        return False
    return any(len(k[0]) > len(ids) and k[0][:len(ids)] == ids and _writable(k[0], k[2]) for k in chain)


def _known(entry: tuple, kept: Sequence[tuple]) -> list[list[str]]:
    """The most blocks a kept entry on ``entry``'s slot holds whose ids are a prefix of its ids, or extend them: the
    slot's rows below both, so the same bits."""

    ids, st = entry[0], entry[1]
    same = [k[2][BLOCKS] for k in kept if k[1] is st and BLOCKS in k[2]
            and k[0][:len(ids)] == ids[:len(k[0])]]
    return max(same, key=lambda b: sum(map(len, b)), default=[])


def persist(disk: disk_states.DiskStates, kept: Sequence[tuple], stop: Callable[[], bool]) -> int:
    """Write the kept states of ``MIN_TOKENS`` or more not on disk yet, newest first; the states written. Runs while
    no request does, so every kept entry's rows are in its slot; a full or failing disk ends the pass with a line.

    Each slot's shortest and longest states go to disk, and none between them unless ``TENSORFOLD_QWEN_STATES_ALL=1``:
    each holds a 115.7 MB DeltaNet piece, and only a regenerated or edited earlier turn resumes from one. After a
    restart such a turn resumes from the longest state on disk that it extends and prefills the rest."""

    every = flag("TENSORFOLD_QWEN_STATES_ALL")
    kept = list(kept)
    written = 0
    for entry in reversed(kept):
        ids, st, snap, tail = entry
        n, m = len(ids), int(snap["mtp_len"])
        if not _writable(ids, snap) or disk.has(ids) or not every and _extended(entry, kept):
            continue
        if stop():
            break
        try:
            done = disk.write(ids, lambda st=st, n=n, m=m: rows(st, n, m), _extras(snap, tail), None, mtp_len=m,
                              drafter_end=-1, born=-1, stop=stop, known=_known(entry, kept))
        except OSError as exc:
            print(f"[tensorfold] kept states not written to disk: {exc}", flush=True)
            break
        if done and (got := disk.get(ids)) is not None:      # None: the budget pruned it at once
            snap[BLOCKS] = disk_states.blocks(got)
        written += bool(done)
    return written


def load(disk: disk_states.DiskStates, entry: disk_states.Entry, st) -> dict[str, Any] | None:
    """Copy a state's rows into slot ``st`` (sized for it) and return its resume ({"state", "tail"}, as kept in
    memory); None, with a line, when a piece fails its hash, the state is not this engine's, or the disk or the
    GPU's memory fails the read (``disk_states.load`` raises StateError for each)."""

    n, m = len(entry.ids), int(entry.mtp_len)
    try:
        got = disk.load(entry, rows(st, n, m), st.rec.device)
    except disk_states.StateError as exc:
        print(f"[tensorfold] a kept state on disk did not read ({exc}); the prompt prefills", flush=True)
        return None
    history = got.get("ple_history")
    snap = {"pos": n, "rec": got["rec"], "conv": got["conv"], "ple_tail": got["ple_tail"],
            "ple_history": None if history is None else history.cpu().numpy(), "mtp_len": m,
            BLOCKS: disk_states.blocks(entry)}
    return {"state": snap, "tail": got.get("tail")}
