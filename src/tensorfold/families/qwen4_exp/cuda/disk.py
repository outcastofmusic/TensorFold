"""Flash Next's kept prompt states on disk (``TENSORFOLD_QWEN_STATES``, ``tensorfold.cuda.disk_states``), for the
serial engine and the concurrent decoder alike: a kept entry is (ids, slot, snapshot, tail), as ``prefixes`` keeps it.

A resume (``prefill_begin``) restores the snapshot (DeltaNet states at the point, conv and n-gram windows, the n-gram
history, ``pos`` and ``mtp_len``), absorbs the tail into the MTP head, and reads the slot's attention rows below
``pos``, the MTP head's below ``mtp_len``, and the complete index pools of each (a partial pool is pooled again when
its block completes). Those rows are what a state writes and reads back; nothing else of the slot is read.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorfold.cuda import disk_states

# the environment the prompt kernels read: a state written under other values may not be the bits this run computes
KERNEL_FLAGS = ("TENSORFOLD_PREFILL_ROWS", "TF_QMMF_FUSED_ROWS", "TF_FLASH_STAGE_AHEAD")


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


def open_states(model_dir: Path, st, *, engine: str, prefill_rows: int) -> disk_states.DiskStates | None:
    """The folder ``TENSORFOLD_QWEN_STATES`` names, under a fingerprint of everything a state's bits depend on: the
    checkpoint, the kernels' source, the row shapes and dtypes, the engine and its prompt settings. None when unset."""

    if not disk_states.configured("QWEN"):
        return None
    import tensorfold
    from tensorfold.cuda import prompt_precision

    here = Path(__file__).resolve().parent
    shared = here.parents[2] / "cuda"
    code = sorted([*here.glob("*.py"), *here.glob("*.cu"), *here.glob("*.cpp"), *here.glob("*.h"),
                   *(f for d in ("kernels", "exl3", "nvfp4", "affine6") for f in (shared / d).rglob("*")
                     if f.suffix in (".py", ".cu", ".cuh", ".cpp", ".h")),
                   *(f for f in shared.glob("*") if f.suffix in (".cu", ".cuh", ".cpp", ".h")),
                   shared / "moe.py", shared / "experts.py"])
    folder = Path(model_dir)
    flags = {k: os.environ.get(k) for k in KERNEL_FLAGS}
    parts = {"tensorfold": tensorfold.__version__, "code": disk_states.digest(code),
             "checkpoint": (str(folder.resolve()),
                            disk_states.digest([folder / "config.json", folder / "quantization_config.json"])),
             "views": [(list(v.shape[1:]), str(v.dtype)) for v in rows(st, 4, 4)],
             "states": [list(st.rec.shape[2:]), str(st.rec.dtype), list(st.conv.shape), str(st.conv.dtype),
                        list(st.ple_tail.shape), st.ple_history is not None],
             "engine": engine, "prefill_rows": int(prefill_rows), "fp8": bool(prompt_precision.fp8()), "flags": flags}
    return disk_states.from_env("QWEN", parts)


def persist(disk: disk_states.DiskStates, kept: Sequence[tuple], stop: Callable[[], bool]) -> int:
    """Write the kept states of ``MIN_TOKENS`` or more not on disk yet, newest first; the states written. Runs while
    no request does, so every kept entry's rows are in its slot; a full or failing disk ends the pass with a line."""

    written = 0
    for ids, st, snap, tail in reversed(list(kept)):
        n, m = len(ids), int(snap["mtp_len"])
        if n < disk_states.MIN_TOKENS or snap["pos"] != n or disk.has(ids):
            continue
        if stop():
            break
        try:
            done = disk.write(ids, lambda st=st, n=n, m=m: rows(st, n, m), _extras(snap, tail), None, mtp_len=m,
                              drafter_end=-1, born=-1, stop=stop)
        except OSError as exc:
            print(f"[tensorfold] kept states not written to disk: {exc}", flush=True)
            break
        written += bool(done)
    return written


def load(disk: disk_states.DiskStates, entry: disk_states.Entry, st) -> dict[str, Any] | None:
    """Copy a state's rows into slot ``st`` (sized for it) and return its resume ({"state", "tail"}, as kept in
    memory); None, with a line, when a piece fails its hash or the state is not this engine's."""

    n, m = len(entry.ids), int(entry.mtp_len)
    try:
        got = disk.load(entry, rows(st, n, m), st.rec.device)
    except disk_states.StateError as exc:
        print(f"[tensorfold] a kept state on disk did not read ({exc}); the prompt prefills", flush=True)
        return None
    history = got.get("ple_history")
    snap = {"pos": n, "rec": got["rec"], "conv": got["conv"], "ple_tail": got["ple_tail"],
            "ple_history": None if history is None else history.cpu().numpy(), "mtp_len": m}
    return {"state": snap, "tail": got.get("tail")}
